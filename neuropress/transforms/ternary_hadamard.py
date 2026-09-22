"""Blockwise signed Hadamard transforms for ternary weight quantization."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .rotation import fast_hadamard_transform, rotation_compute_dtype

PRISM_TERNARY_SOURCE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "lm_head",
)


def is_prism_ternary_source_module(name: str) -> bool:
    """Return whether an HF module maps to a quantized plugin projection."""
    parts = name.split(".")
    return (
        parts[-1] in PRISM_TERNARY_SOURCE_SUFFIXES
        and "mtp" not in parts
        and "visual" not in parts
    )


def block_hadamard_transform(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Apply an orthonormal Hadamard transform to last-dimension blocks."""
    width = x.shape[-1]
    if width % block_size:
        raise ValueError(
            f"input width {width} must be divisible by Hadamard block {block_size}"
        )
    blocked = x.reshape(*x.shape[:-1], -1, block_size)
    return fast_hadamard_transform(blocked).reshape(x.shape)


def shared_signs(
    width: int, seed: int, device: torch.device | None = None
) -> torch.Tensor:
    """Create the deterministic sign vector shared by all layers of *width*."""
    generator = torch.Generator().manual_seed(seed + width)
    values = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
    return (values * 2 - 1).to(device=device, dtype=torch.float32)


class TernaryHadamardInputLinear(nn.Module):
    """Linear wrapper whose stored weight is in a signed-Hadamard basis."""

    hadamard_signs: torch.Tensor

    def __init__(
        self, linear: nn.Linear, signs: torch.Tensor, block_size: int = 1024
    ) -> None:
        super().__init__()
        # Estimator selection must be based on wrapper membership rather than
        # the conventional child name ``linear``.  Real models can contain
        # unrelated modules with that name, and quantizing those as ternary
        # would silently corrupt an otherwise dense checkpoint tensor.
        linear._neuropress_ternary_wrapped = True
        self.linear = linear
        self.block_size = block_size
        self.register_buffer("hadamard_signs", signs.to(torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the input and apply the folded linear."""
        dtype = x.dtype
        compute_dtype = rotation_compute_dtype(dtype)
        signs = self.hadamard_signs.to(device=x.device, dtype=compute_dtype)
        rotated = block_hadamard_transform(x.to(compute_dtype) * signs, self.block_size)
        return self.linear(rotated.to(dtype))


def fold_linear_to_hadamard(
    linear: nn.Linear, signs: torch.Tensor, block_size: int = 1024
) -> TernaryHadamardInputLinear:
    """Fold the inverse runtime transform into *linear* and wrap it."""
    if linear.in_features != signs.numel():
        raise ValueError(
            f"linear width {linear.in_features} differs from sign width {signs.numel()}"
        )
    weight = linear.weight.detach()
    rotated = block_hadamard_transform(
        weight.to(torch.float32) * signs.to(weight.device), block_size
    ).to(weight.dtype)
    linear.weight = nn.Parameter(rotated, requires_grad=linear.weight.requires_grad)
    return TernaryHadamardInputLinear(linear, signs, block_size)


def apply_ternary_hadamard(
    model: nn.Module,
    *,
    block_size: int = 1024,
    seed: int = 1234,
    module_filter: Callable[[str, nn.Linear], bool] | None = None,
    strict_width: bool = False,
) -> dict[int, torch.Tensor]:
    """Fold blockwise rotations into selected linears and return signs by width."""
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and (module_filter is None or module_filter(name, module))
    ]
    incompatible = [
        (name, module.in_features)
        for name, module in candidates
        if module.in_features % block_size
    ]
    if strict_width and incompatible:
        details = ", ".join(f"{name}={width}" for name, width in incompatible)
        raise ValueError(
            f"ternary target widths must be divisible by block_size={block_size}: "
            f"{details}"
        )
    targets = [
        (name, module)
        for name, module in candidates
        if module.in_features % block_size == 0
    ]
    signs_by_width: dict[int, torch.Tensor] = {}
    for name, linear in targets:
        width = linear.in_features
        signs = signs_by_width.setdefault(
            width, shared_signs(width, seed, linear.weight.device)
        )
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, fold_linear_to_hadamard(linear, signs, block_size))
    return signs_by_width
