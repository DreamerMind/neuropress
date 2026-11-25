"""Block-wise calibration of Hugging Face causal language models.

A whole-model :class:`.QuantizationPipeline` run collects the statistics of
every target before estimating any, so all of them are live at once. For MoE
models that is ``E * (D^2 + I^2)`` float32 Hessian entries per layer, about
117 GB on Qwen3-30B-A3B. :func:`calibrate_causal_lm_blockwise` drives
:class:`.BlockQuantizationPipeline` over the decoder layers and then over the
final norm and LM head, so only one block's statistics are live.

Layer inputs are captured in one float forward pass: the first layer's hidden
states, and the keyword arguments of *every* layer (attention masks, rotary
embeddings, ...), so models whose layers take different masks or rope (hybrid
or sliding-window attention) get each layer's own. Tensors shared between
layers are stored once.

With ``input_source="float"`` each block is calibrated on the float model's
activations, which gives the statistics of a whole-model run. With
``"sequential"`` it is calibrated on the outputs of the already quantized
blocks before it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from .block_pipeline import BlockQuantizationPipeline
from .pipeline import QuantizationPipeline
from .recipe_config import QuantizationRecipeConfig
from .t2n.model_io import MixedPrecisionSplitTransform, QuantizationManifest

LOGGER = logging.getLogger(__name__)

__all__ = ["calibrate_causal_lm_blockwise"]


class _Head(nn.Module):
    """The final norm and LM head as one block.

    A tied input embedding is held too, unused by the forward, so the adapter
    sees the tie as it does in a whole-model run.
    """

    def __init__(
        self, norm: nn.Module, lm_head: nn.Module, embed: nn.Module | None
    ) -> None:
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head
        if embed is not None:
            self.embed_tokens = embed

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The statistics come from hooks; the logits feed nothing, and the
        # block pipeline would keep both streams of them on the host (about
        # 40 GB for a 150k vocabulary at 128 x 512 tokens). An empty slice
        # still runs the head.
        return self.lm_head(self.norm(hidden_states))[..., :0]


class _Stop(Exception):
    """Ends the capture forward after the last decoder layer's inputs."""


def _decoder(model: nn.Module) -> nn.Module:
    get_decoder = getattr(model, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not len(layers):
        raise ValueError(
            f"{type(model).__name__}: block calibration needs a Hugging Face "
            "causal LM whose get_decoder() holds the decoder layers as .layers"
        )
    if not isinstance(getattr(decoder, "norm", None), nn.Module):
        raise ValueError(f"{type(decoder).__name__}: no final .norm to calibrate")
    if not isinstance(getattr(model, "lm_head", None), nn.Module):
        raise ValueError(f"{type(model).__name__}: no .lm_head to calibrate")
    return decoder


def _path_of(model: nn.Module, module: nn.Module) -> str:
    for name, candidate in model.named_modules():
        if candidate is module:
            return name
    raise ValueError("module not found in the model")


def _check_one_device(model: nn.Module) -> None:
    """Refuse a model placed across devices: each block moves to one device."""
    # An accelerate dispatch keeps offloaded weights on "meta", so its map
    # is read too; its entries (0, "cuda:0", "cpu", "disk") are compared
    # among themselves only.
    mapped = {str(d) for d in (getattr(model, "hf_device_map", None) or {}).values()}
    devices = {str(t.device) for t in [*model.parameters(), *model.buffers()]}
    if len(mapped) > 1 or len(devices) > 1:
        devices |= mapped
        raise ValueError(
            "block calibration moves each block, and the whole model at the "
            f"end, to one device; this model is spread over {sorted(devices)}. "
            "Load it on one device, or calibrate it without blocks"
        )


def _adopt_splits(
    model: nn.Module, block: nn.Module, full_path: Callable[[str], str]
) -> None:
    """Move the mixed-precision splits *block* recorded into *model*'s manifest.

    The adapter records splits on the module it adapts, here the block, with
    paths relative to it; a load replays the model's manifest only. The
    block's QTensor and MoE scale entries are dropped: saving the model
    rescans both.
    """
    manifest = block.__dict__.pop("__neuropress_manifest__", None)
    if manifest is None:
        return
    splits = {
        full_path(path): spec.model_copy(
            update={
                "shares_branches_with": full_path(spec.shares_branches_with)
                if spec.shares_branches_with
                else None
            }
        )
        for transform in manifest.transforms
        if isinstance(transform, MixedPrecisionSplitTransform)
        for path, spec in transform.modules.items()
    }
    if not splits:
        return
    root = getattr(model, "__neuropress_manifest__", None)
    if root is None:
        root = QuantizationManifest()
        model.__neuropress_manifest__ = root  # type: ignore[attr-defined]
    root.transforms.append(
        MixedPrecisionSplitTransform(type="mixed_precision_split", modules=splits)
    )


def _inside(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}.")


def _targets_outside(
    model: nn.Module, config: QuantizationRecipeConfig, prefixes: list[str]
) -> list[str]:
    """Estimator targets that no block contains."""
    estimator = config.build().param_estimator
    estimator.prepare(model)
    try:
        targets = estimator.declare_targets()
    except NotImplementedError as error:
        raise ValueError(
            f"block calibration must know every target to place it: {error}"
        ) from error
    return sorted(t for t in targets if not any(_inside(t, p) for p in prefixes))


def _host_tree(value: Any, seen: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Host copy of a kwargs tree, one copy per distinct tensor in *seen*."""
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in seen:
            # The original is kept alive with its copy so its id stays unique.
            seen[key] = (value, value.detach().cpu())
        return seen[key][1]
    if isinstance(value, dict):
        return {k: _host_tree(v, seen) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_host_tree(v, seen) for v in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"cannot carry a {type(value).__name__} between decoder blocks")


_DROPPED_KWARGS = frozenset({"past_key_values", "use_cache", "output_attentions"})


@torch.no_grad()
def _capture(
    model: nn.Module, layers: nn.ModuleList, batches: Iterable[dict[str, Any]]
) -> tuple[list[torch.Tensor], list[list[dict[str, Any]]]]:
    """First-layer hidden states and every layer's kwargs, per batch, on the host."""
    hidden: list[torch.Tensor] = []
    kwargs_per_layer: list[list[dict[str, Any]]] = [[] for _ in layers]
    last = len(layers) - 1
    for batch in batches:
        seen: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def hook(
            index: int,
            seen: dict[int, tuple[torch.Tensor, torch.Tensor]],
            args: tuple,
            kwargs: dict,
        ) -> None:
            kwargs = dict(kwargs)
            if args:
                if len(args) > 1:
                    raise ValueError(
                        "decoder layers must take only hidden_states positionally"
                    )
                states = args[0]
            else:
                states = kwargs.pop("hidden_states")
            kwargs.pop("hidden_states", None)
            if index == 0:
                hidden.append(states.detach().cpu())
            kwargs_per_layer[index].append(
                _host_tree(
                    {k: v for k, v in kwargs.items() if k not in _DROPPED_KWARGS},
                    seen,
                )
            )
            if index == last:
                raise _Stop

        handles = [
            layer.register_forward_pre_hook(
                lambda m, a, k, i=i, s=seen: hook(i, s, a, k), with_kwargs=True
            )
            for i, layer in enumerate(layers)
        ]
        try:
            model(**batch, use_cache=False)
        except _Stop:
            pass
        finally:
            for handle in handles:
                handle.remove()
        if len(hidden) != len(kwargs_per_layer[last]):
            raise RuntimeError("the forward did not reach every decoder layer")
    if not hidden:
        raise ValueError("no calibration batches")
    return hidden, kwargs_per_layer


def _hidden_states(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict) and "hidden_states" in output:
        return output["hidden_states"]
    raise TypeError(f"cannot read hidden states from a {type(output).__name__}")


def _quantize_outside(
    model: nn.Module,
    batches: list[dict[str, Any]],
    config: QuantizationRecipeConfig,
    targets: list[str],
) -> dict[str, str]:
    """Quantize *targets* in one whole-model pass restricted to them."""
    pipeline = QuantizationPipeline(config.build())
    with torch.no_grad():
        pipeline.run(model, iter(batches), target_paths=set(targets))
    return dict(pipeline.last_plan.kept_float)


def calibrate_causal_lm_blockwise(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    config: QuantizationRecipeConfig,
    *,
    input_source: Literal["float", "sequential"] = "float",
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, str]:
    """Quantize *model* in place one decoder layer at a time, then its head.

    Estimator targets outside the decoder layers and the head (a vision
    projector, ...) are quantized in one whole-model pass restricted to them:
    before the layers are captured with ``"sequential"``, so the layers see
    them quantized, and after with ``"float"``, so they see a float model as
    in a whole-model run. That pass is not checkpointed.

    A tied input embedding shares its weight with the LM head, so it is
    quantized in the head block, after every layer: with ``"sequential"`` the
    layers are calibrated on its float values. Quantizing it first would
    quantize the head's weight before the head's own calibration.

    Args:
        model: Hugging Face causal LM (``get_decoder().layers``, ``.norm`` and
            ``.lm_head``).
        batches: Model inputs (``input_ids``, ...) on the model's device.
        config: Recipe; its kernel lowering is not applied here.
        input_source: ``"float"`` (whole-model statistics) or
            ``"sequential"`` (statistics on quantized upstream activations).
        device: Forward and estimation device; defaults to the model's.
        checkpoint_dir: Resume finished blocks from here, and save each block
            once done (its adapted weights and both output streams). A block
            resumes only when the recipe, *input_source*, its float weights
            and its inputs all match, so a sequential block is recomputed
            whenever anything upstream of it changed.

    Returns:
        The ``kept_float`` reasons of every block, keyed by full module path.
    """
    decoder = _decoder(model)
    _check_one_device(model)
    layers = decoder.layers
    device = torch.device(device) if device else next(model.parameters()).device
    layer_paths = [_path_of(model, layer) for layer in layers]
    head_path = _path_of(model, model.lm_head)
    norm_path = _path_of(model, decoder.norm)

    embed = getattr(decoder, "embed_tokens", None)
    tied = (
        embed is not None
        and getattr(embed, "weight", None) is not None
        and embed.weight is getattr(model.lm_head, "weight", None)
    )
    head = _Head(decoder.norm, model.lm_head, embed if tied else None)
    head_paths = {
        "norm": norm_path,
        "lm_head": head_path,
        **({"embed_tokens": _path_of(model, embed)} if tied else {}),
    }
    # A tied embedding is part of the head block, not outside the blocks.
    outside = _targets_outside(model, config, [*layer_paths, *head_paths.values()])

    runner = BlockQuantizationPipeline(
        config.model_copy(update={"kernel_lowering": None}),
        input_source=input_source,
        forward_device=device,
        estimation_device=device,
        checkpoint_dir=checkpoint_dir,
        # The block fingerprints cover the recipe, weights and inputs; this
        # only separates model classes in a shared directory.
        run_id=type(model).__name__ if checkpoint_dir is not None else None,
    )
    kept: dict[str, str] = {}
    if outside:
        # Model inputs are token ids: cheap to replay.
        batches = list(batches)
        if input_source == "sequential":
            kept.update(_quantize_outside(model, batches, config, outside))
    student, kwargs_per_layer = _capture(model, layers, batches)
    if outside and input_source == "float":
        kept.update(_quantize_outside(model, batches, config, outside))
    reference = student
    try:
        for index, (path, layer) in enumerate(zip(layer_paths, layers, strict=True)):
            kwargs = kwargs_per_layer[index]
            result = runner.run_block(
                path,
                layer,
                [
                    {"hidden_states": h, **kw}
                    for h, kw in zip(student, kwargs, strict=True)
                ],
                [
                    {"hidden_states": h, **kw}
                    for h, kw in zip(reference, kwargs, strict=True)
                ],
            )
            kwargs_per_layer[index] = []
            _adopt_splits(model, layer, lambda rel, path=path: f"{path}.{rel}")
            student = [_hidden_states(o) for o in result.student_outputs]
            reference = [_hidden_states(o) for o in result.reference_outputs]
            kept.update({f"{path}.{k}": v for k, v in result.kept_float.items()})
            LOGGER.info(
                "block calibration: %s %s",
                path,
                "resumed" if result.resumed else "done",
            )
        result = runner.run_block(
            "head",
            head,
            [{"hidden_states": h} for h in student],
            [{"hidden_states": h} for h in reference],
        )

        def head_path_of(rel: str) -> str:
            child, _, rest = rel.partition(".")
            return head_paths[child] + (f".{rest}" if rest else "")

        # The adapter replaces modules on their parent, here the _Head
        # (mixed-precision splits): install them on the model too.
        for child, full in head_paths.items():
            module = getattr(head, child)
            if module is not model.get_submodule(full):
                parent, _, leaf = full.rpartition(".")
                setattr(model.get_submodule(parent), leaf, module)
        _adopt_splits(model, head, head_path_of)
        kept.update({head_path_of(k): v for k, v in result.kept_float.items()})
    finally:
        # run_block hands every block back on the host.
        model.to(device)
    return kept
