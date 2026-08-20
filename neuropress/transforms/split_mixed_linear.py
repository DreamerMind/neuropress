"""Split a mixed-precision Linear into per-precision homogeneous sub-linears.

Background
----------
After ``GPTQMixedPrecisionCalibration``, each column-group (panel) of a
``nn.Linear`` weight matrix ``W ∈ R^{out × in}`` carries an independent
quantizer with its own bit-width.  Running mixed-precision inference on a
single monolithic tensor is awkward for most backends (NNEF/tract included).

This transform exploits the column-partition identity::

    y = x W^T = Σ_g  x[…, g_cols] @ W[:, g_cols]^T

to rewrite the Linear as a set of *homogeneous-precision* sub-linears whose
outputs are summed:

    perm                       input-column permutation (same-precision groups
                               become contiguous)
    x_perm = x[…, perm]        permuted input
    y = Σ_k  L_k(x_perm[…, offset_k : offset_k + in_k])

Each ``L_k`` is a plain ``nn.Linear(in_k, out, bias=False)`` whose weight
contains the column slices for all panels assigned to precision ``k``.

The result is a ``MixedPrecisionLinear`` module that is numerically
equivalent to the original ``nn.Linear`` (modulo the float rounding already
introduced by GPTQ).

Terminology
-----------
* **column permutation** – reorders input features so that all W4 columns
  come before all W2 columns (etc.).  This is the "permute" step.
* **weight concat** – for each precision bucket, the relevant column slices
  of ``W`` are concatenated to form the new sub-linear's weight.  This is
  the "concat" step (on the weight side).
* **output sum** – each sub-linear produces ``[*, out_features]``; they are
  summed to recover the full output.

Usage
-----
::

    assignments = calib._mp_assignments     # from GPTQMixedPrecisionCalibration
    mixed_lin   = split_mixed_precision_linear(linear, assignments, group_size=32)
    # mixed_lin is a drop-in replacement for linear
"""

from __future__ import annotations

import torch
from torch import nn

from ..methods.gptq_panel import PanelAssignment

# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class MixedPrecisionLinear(nn.Module):
    """Numerically equivalent replacement for a mixed-precision ``nn.Linear``.

    Attributes:
        column_perm: LongTensor ``[in_features]`` — index tensor that permutes
            the input so same-precision column-groups are contiguous.
            Apply as ``x[…, column_perm]``.
        sub_linears: One ``nn.Linear(in_k, out, bias=False)`` per distinct precision.
        split_sizes: Number of (permuted) input features consumed by each sub-linear.
        precision_nbits: Bit-width of each sub-linear, in the same order.

    Forward pass applies ``x_perm = x[…, column_perm]``, then sums each
    ``sub_linears[k](x_perm[…, offset_k : offset_k + split_sizes[k]])``, and
    finally adds bias if present.
    """

    def __init__(
        self,
        column_perm: torch.Tensor,
        sub_linears: nn.ModuleList,
        split_sizes: list[int],
        precision_nbits: list[int],
        bias: torch.Tensor | None = None,
    ) -> None:
        """Initialize from pre-built sub-linears and a column permutation."""
        super().__init__()
        self.register_buffer("column_perm", column_perm)
        self.sub_linears = sub_linears
        self.split_sizes = list(split_sizes)
        self.precision_nbits = list(precision_nbits)
        self.register_buffer("bias", bias)  # None is valid; checked in forward

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def in_features(self) -> int:
        """Total number of input features (length of the column permutation)."""
        perm = self.column_perm
        assert isinstance(perm, torch.Tensor)
        return int(perm.shape[0])

    @property
    def out_features(self) -> int:
        """Number of output features (from the first sub-linear)."""
        first = self.sub_linears[0]
        assert isinstance(first, nn.Linear)
        return int(first.out_features)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permute input columns, run sub-linears, and sum their outputs."""
        # Step 1 - permute input columns so same-precision groups are contiguous.
        col_perm = self.column_perm
        assert isinstance(col_perm, torch.Tensor)
        x_perm = x[..., col_perm]

        # Step 2 - run each homogeneous sub-linear and accumulate the sum.
        out: torch.Tensor | None = None
        offset = 0
        for linear, size in zip(self.sub_linears, self.split_sizes, strict=False):
            partial = linear(x_perm[..., offset : offset + size])
            out = partial if out is None else out + partial
            offset += size

        assert out is not None
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        """Return extra module info showing precision buckets."""
        buckets = ", ".join(
            f"{nbits}bx{sz}"
            for nbits, sz in zip(self.precision_nbits, self.split_sizes, strict=False)
        )
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"groups=[{buckets}]"
        )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def split_mixed_precision_linear(
    linear: nn.Linear,
    assignments: list[PanelAssignment],
    group_size: int,
    sort_descending: bool = True,
) -> MixedPrecisionLinear:
    """Split *linear* into per-precision sub-linears based on panel *assignments*.

    Args:
        linear: Source ``nn.Linear``. Its weights should already reflect GPTQ
            quantization (quantize→dequantize applied in-place).
        assignments: One ``PanelAssignment`` per column-group, as produced by
            ``GPTQMixedPrecisionCalibration._mp_assignments``.
        group_size: Number of input columns per panel (must match the GPTQ
            ``group_size`` used during calibration).
        sort_descending: When ``True`` (default), the highest-precision sub-linear
            comes first in the module list and in the input slice.

    Returns:
        Drop-in replacement for *linear*.

    Raises:
        ValueError: If any panel has no selected quantizer, or if ``group_size``
            does not evenly divide ``linear.in_features``.
    """
    in_features = linear.in_features
    if in_features % group_size != 0:
        raise ValueError(
            f"linear.in_features={in_features} is not divisible"
            f" by group_size={group_size}"
        )
    expected_panels = in_features // group_size
    if len(assignments) != expected_panels:
        raise ValueError(
            f"Expected {expected_panels} assignments for in_features={in_features} "
            f"/ group_size={group_size}, got {len(assignments)}"
        )

    # 1. Group panel indices by the selected quantizer's nbits.
    nbits_to_panels: dict[int, list[int]] = {}
    for assignment in sorted(assignments, key=lambda a: a.panel_index):
        if assignment.selected is None:
            raise ValueError(
                f"Panel {assignment.panel_index} has no selected quantizer."
            )
        nbits = assignment.selected.nbits
        nbits_to_panels.setdefault(nbits, []).append(assignment.panel_index)

    # 2. Determine the ordering of precision buckets.
    sorted_nbits = sorted(nbits_to_panels.keys(), reverse=sort_descending)

    # 3. Build the flat column permutation and record split sizes.
    #    Within each precision bucket the original column order is preserved.
    col_perm: list[int] = []
    split_sizes: list[int] = []
    for nbits in sorted_nbits:
        panels = sorted(nbits_to_panels[nbits])
        bucket_cols: list[int] = []
        for panel_idx in panels:
            start = panel_idx * group_size
            bucket_cols.extend(range(start, start + group_size))
        col_perm.extend(bucket_cols)
        split_sizes.append(len(bucket_cols))

    col_perm_tensor = torch.tensor(col_perm, dtype=torch.long)

    # 4. Create one sub-linear per precision bucket.
    W = linear.weight.data  # [out_features, in_features]
    out_features = W.shape[0]
    sub_linears = nn.ModuleList()
    offset = 0
    for size in split_sizes:
        cols = col_perm_tensor[offset : offset + size]
        sub_W = W[:, cols].clone()
        sub_lin = nn.Linear(size, out_features, bias=False)
        with torch.no_grad():
            sub_lin.weight.copy_(sub_W)
        sub_linears.append(sub_lin)
        offset += size

    # 5. Detach bias (added once at the end in forward, not per sub-linear).
    bias = linear.bias.data.clone() if linear.bias is not None else None

    return MixedPrecisionLinear(
        column_perm=col_perm_tensor,
        sub_linears=sub_linears,
        split_sizes=split_sizes,
        precision_nbits=sorted_nbits,
        bias=bias,
    )


# ---------------------------------------------------------------------------
# Convenience: apply to a whole model in-place
# ---------------------------------------------------------------------------


def apply_split_to_model(
    model: nn.Module,
    assignments_map: dict[str, list[PanelAssignment]],
    group_size: int,
    sort_descending: bool = True,
) -> nn.Module:
    """Replace every ``nn.Linear`` in *assignments_map* with a ``MixedPrecisionLinear``.

    Args:
        model: The model whose layers should be replaced (modified **in-place**).
        assignments_map: Mapping from fully-qualified module path (e.g.
            ``"layers.0.fc"``) to the corresponding ``List[PanelAssignment]``.
        group_size: GPTQ group size used during calibration.
        sort_descending: Passed through to ``split_mixed_precision_linear``.

    Returns:
        The same *model* object, with targeted layers replaced.
    """
    for path, assignments in assignments_map.items():
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child = getattr(parent, parts[-1])
        if not isinstance(child, nn.Linear):
            raise TypeError(f"Module at '{path}' is {type(child)}, expected nn.Linear")
        mixed = split_mixed_precision_linear(
            child, assignments, group_size, sort_descending=sort_descending
        )
        setattr(parent, parts[-1], mixed)

    # Record the split transform in the model manifest if one is present.
    _manifest = getattr(model, "__neuropress_manifest__", None)
    if _manifest is not None and hasattr(_manifest, "transforms"):
        from neuropress.t2n.model_io import (
            MixedPrecisionModuleSpec,
            MixedPrecisionSplitTransform,
        )

        module_specs: dict[str, MixedPrecisionModuleSpec] = {}
        for path, _assignments in assignments_map.items():
            parts = path.split(".")
            mod = model
            for part in parts:
                mod = getattr(mod, part)
            # mod is now the MixedPrecisionLinear we just installed.
            module_specs[path] = MixedPrecisionModuleSpec(
                split_sizes=mod.split_sizes,
                precision_nbits=mod.precision_nbits,
                out_features=mod.out_features,
                has_bias=mod.bias is not None,
            )
        _manifest.transforms.append(
            MixedPrecisionSplitTransform(
                type="mixed_precision_split", modules=module_specs
            )
        )

    return model
