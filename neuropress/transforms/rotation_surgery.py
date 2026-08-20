"""Whole-model Hadamard incoherence processing.

Walks a pre-norm decoder stack, folds every RMSNorm forward, and rotates the
residual stream so weights become approximately i.i.d. Gaussian.  Optionally
adds the online rotation of the MLP intermediate, which is where activation
outliers concentrate.

Applicable to Qwen3.5-style architectures: each sublayer must end in a plain
``Linear`` writing to the residual, and all norms must be *pre*-norms.  Gemma-3
and Gemma-4 do **not** qualify - their sandwich norms sit after each sublayer,
where an elementwise scale can no longer be folded forward.
"""

from typing import Any

import torch
from torch import nn

from .rotation import (
    UnrotateInput,
    rotate_residual_stream,
    rotation_compute_dtype,
    wrap_linear_with_online_hadamard,
)


def _find_decoder_stack(model: nn.Module) -> nn.Module:
    """Locate the submodule holding ``embed_tokens``, ``layers`` and ``norm``."""
    candidates = [model, *(m for _, m in model.named_modules())]
    for module in candidates:
        if (
            isinstance(getattr(module, "layers", None), nn.ModuleList)
            and isinstance(getattr(module, "embed_tokens", None), nn.Module)
            and isinstance(getattr(module, "norm", None), nn.Module)
        ):
            return module
    raise ValueError(
        "Unrecognized model topology: expected a submodule exposing "
        "`embed_tokens`, `layers` and `norm` (Qwen3.5-style pre-norm stack)."
    )


def _layer_spec(layer: nn.Module) -> dict[str, Any]:
    """Build the rotation spec for one decoder layer."""
    if hasattr(layer, "linear_attn"):
        mixer = layer.linear_attn
        # All four projections read the same norm; missing one corrupts the model.
        consumers = [
            mixer.in_proj_qkv,
            mixer.in_proj_z,
            mixer.in_proj_b,
            mixer.in_proj_a,
        ]
        attn_output = mixer.out_proj
    elif hasattr(layer, "self_attn"):
        mixer = layer.self_attn
        consumers = [mixer.q_proj, mixer.k_proj, mixer.v_proj]
        attn_output = mixer.o_proj
    else:
        raise ValueError(
            f"Layer {type(layer).__name__} has neither `linear_attn` nor "
            f"`self_attn`; cannot determine the residual write-back point."
        )

    return {
        "input_norm": layer.input_layernorm,
        "input_consumers": consumers,
        "attn_output": attn_output,
        "post_norm": layer.post_attention_layernorm,
        "post_consumers": [layer.mlp.gate_proj, layer.mlp.up_proj],
        "mlp_output": layer.mlp.down_proj,
    }


def apply_hadamard_rotation(
    model: nn.Module,
    seed: int = 1234,
    rotate_down_proj: bool = True,
    plus_one_norms: bool = True,
) -> dict[str, Any]:
    """Apply incoherence processing to *model* in place, preserving its output.

    Three transforms are applied, all exactly output-preserving:

    1. Every RMSNorm's elementwise scale is folded into its consumers, leaving
       a pure normalization that commutes with an orthogonal rotation.
    2. The residual stream is rotated. The embedding is deliberately left
       untouched - it is tied to ``lm_head`` and the two roles want different
       corrections - so the stack is bracketed by two hidden-state ops instead.
    3. Optionally the MLP intermediate is rotated online before ``down_proj``,
       which cannot be folded because ``silu(gate) * up`` is elementwise
       nonlinear.

    The bracketing ops are installed as permanent forward hooks. **They do not
    survive a state-dict round-trip**, so a rotated model must be quantized and
    evaluated in the same process until manifest support is added.

    Args:
        model: Causal LM to transform in place.
        seed: Base seed; the online rotation uses ``seed + 1``.
        rotate_down_proj: Whether to add the online MLP-intermediate rotation.
        plus_one_norms: Whether norms use the ``1 + weight`` convention
            (true for Qwen3.5 and Gemma).

    Returns:
        Summary describing what was touched.

    Raises:
        ValueError: If the model topology is not a recognized pre-norm stack.
    """
    stack = _find_decoder_stack(model)
    layers = stack.layers
    hidden_size = stack.embed_tokens.weight.shape[1]

    specs = [_layer_spec(layer) for layer in layers]
    rot = rotate_residual_stream(
        layers=specs,
        hidden_size=hidden_size,
        seed=seed,
        plus_one_norms=plus_one_norms,
    )

    wrapped = 0
    if rotate_down_proj:
        for layer in layers:
            layer.mlp.down_proj = wrap_linear_with_online_hadamard(
                layer.mlp.down_proj, seed=seed + 1
            )
            wrapped += 1

    # Enter the rotated basis after the embedding, and leave it before the
    # final norm, whose scale cannot be folded into the tied lm_head.
    def _enter(_module: nn.Module, _inputs: Any, output: torch.Tensor):
        out_dtype = output.dtype
        compute_dtype = rotation_compute_dtype(out_dtype)
        r = rot.to(device=output.device, dtype=compute_dtype)
        return (output.to(compute_dtype) @ r.T).to(out_dtype)

    # The final norm is wrapped, not hooked: kernel lowering replaces every
    # RMSNorm instance and would discard a hook bound to the old one.
    stack.norm = UnrotateInput(stack.norm, rot)

    handles = [
        stack.embed_tokens.register_forward_hook(_enter),
    ]
    # Keep handles reachable so the hooks are not garbage collected.
    model._neuropress_rotation_handles = handles

    return {
        "layers_rotated": len(specs),
        "down_proj_wrapped": wrapped,
        "hidden_size": hidden_size,
        "seed": seed,
    }
