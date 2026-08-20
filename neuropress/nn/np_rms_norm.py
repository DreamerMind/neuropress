"""NPRMSNorm: fused single-pass RMSNorm backed by a CUDA kernel.

Drop-in replacement for any unfused RMSNorm (HuggingFace Gemma, Llama,
Mistral, Qwen, …).  Applied at lowering time via :func:`linearize_rms_norms`.

The HuggingFace default implementation dispatches five separate CUDA kernels
(pow, mean, rsqrt, mul, mul) per forward call.  At ~90 norms per Gemma-3-270M
forward pass this accounts for ~60% of TG latency.  This module replaces all
five with one warp-reduction kernel.

Two RMSNorm calling conventions exist in the wild (see HF PR #29402):

- LLaMA/Mistral-style: ``out = x_normed * weight``, weight initialized/
  trained near **one**.
- Gemma/Qwen3.5-style: ``out = x_normed * (1.0 + weight)``, weight
  initialized/trained near **zero** (``Gemma1RMSNorm``, ``Gemma2RMSNorm``,
  ``Gemma3RMSNorm``, ``Qwen3_5RMSNorm``, and others that copy the pattern).

The kernel itself only ever computes the plain ``weight`` form, so
:func:`linearize_rms_norms` must bake the ``+1`` into the weight it hands
to :class:`NPRMSNorm` whenever the source module uses that convention.
Getting this wrong doesn't crash anything -- it silently attenuates every
normalization layer in the network toward zero, since the "+1" convention's
weight sits near zero on its own. This previously went undetected for
exactly that reason: no error, no NaN, just a uniform collapse in every
quantized (kernel-lowered) eval, indistinguishable at a glance from "the
quantization is just too aggressive."
"""

from __future__ import annotations

import logging

import torch
from torch import nn

__all__ = ["NPRMSNorm", "linearize_rms_norms"]

log = logging.getLogger(__name__)


class NPRMSNorm(nn.Module):
    """Fused RMSNorm: single CUDA kernel, no intermediate allocations.

    Numerically equivalent to::

        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    Args:
        weight: RMSNorm scale parameter ``[H]``.
        eps: Epsilon added to mean-square before rsqrt.
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("weight", weight.clone())
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused RMS normalization over the last dimension of *x*.

        Args:
            x: Input of shape ``[..., H]``.

        Returns:
            Normalized tensor with the same shape as *x*.
        """
        from .. import kernels

        *leading, H = x.shape
        x_2d = x.reshape(-1, H)
        out = kernels.rms_norm(x_2d, self.weight.to(x_2d.dtype), self.eps)
        return out.reshape(*leading, H)

    def extra_repr(self) -> str:
        """Return the hidden size and epsilon for ``repr()``."""
        return f"H={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_unfused_rms_norm(module: nn.Module) -> bool:
    """True when *module* looks like an unfused RMSNorm and can be replaced.

    Criteria:
    - Class name contains "RMSNorm" (catches Gemma3RMSNorm, LlamaRMSNorm, …).
    - Exactly one parameter named ``weight``, shape ``[H]``.
    - Has an ``eps`` attribute.
    - Not already an :class:`NPRMSNorm`.
    """
    if isinstance(module, NPRMSNorm):
        return False
    if "RMSNorm" not in type(module).__name__:
        return False
    params = dict(module.named_parameters(recurse=False))
    if set(params.keys()) != {"weight"}:
        return False
    if params["weight"].dim() != 1:
        return False
    if not hasattr(module, "eps"):
        return False
    return True


def _uses_plus_one_convention(module: nn.Module, weight: torch.Tensor) -> bool:
    """Return ``True`` if *module* uses the "+1" RMSNorm convention.

    That is, it computes ``x_normed * (1 + weight)`` rather than
    ``x_normed * weight`` (see module docstring for the two conventions).

    Detected empirically rather than by class name (robust to future
    architectures copying the pattern under a new name): temporarily zero
    the module's weight and run its original ``forward`` on a random probe.
    A "+1"-convention module leaves the normalized probe scaled by exactly
    one (non-zero output); a plain-weight module zeroes the entire output.
    The original weight is restored before returning either way.
    """
    probe = torch.randn(1, weight.shape[0], device=weight.device, dtype=weight.dtype)
    saved = weight.clone()
    try:
        weight.zero_()
        with torch.no_grad():
            zeroed_out = module(probe)
    finally:
        weight.copy_(saved)
    return bool(zeroed_out.float().abs().max().item() > 1e-3)


# ---------------------------------------------------------------------------
# Model-level transform
# ---------------------------------------------------------------------------


def linearize_rms_norms(model: nn.Module) -> int:
    """Replace all unfused RMSNorm modules in *model* with :class:`NPRMSNorm`.

    Traverses *model* in-place.  Returns the number of replacements made.
    Skips silently when CUDA JIT is unavailable (returns 0).

    Args:
        model: Any ``nn.Module``; typically a HuggingFace causal LM.

    Returns:
        Number of :class:`NPRMSNorm` modules inserted.
    """
    from .backends import is_cuda_jit_available

    if not is_cuda_jit_available():
        log.debug("linearize_rms_norms: CUDA JIT unavailable — skipping")
        return 0

    targets: list[tuple[str, nn.Module]] = [
        (name, mod) for name, mod in model.named_modules() if _is_unfused_rms_norm(mod)
    ]

    n_plus_one = 0
    for name, mod in targets:
        weight = next(mod.parameters()).data
        eps = float(mod.eps)
        if _uses_plus_one_convention(mod, weight):
            effective_weight = 1.0 + weight
            n_plus_one += 1
        else:
            effective_weight = weight.clone()
        np_norm = NPRMSNorm(effective_weight, eps=eps).to(weight.device, weight.dtype)

        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, np_norm)

    if targets:
        log.debug(
            "linearize_rms_norms: replaced %d module(s) (%d used the '+1' convention)",
            len(targets),
            n_plus_one,
        )
    return len(targets)
