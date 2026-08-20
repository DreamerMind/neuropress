"""weight-transforms: registers Q4_0, Q2_0, Q4_1, and identity weight transforms.

Populates :data:`.WEIGHT_TRANSFORM_REGISTRY` on import with the following keys:

``"q4_0"`` / ``"q4_0_unpacked"``
    Standard tract Q4_0: one 4-bit value per ``uint8`` byte in
    :attr:`~torch_to_nnef.tensor.quant.base.QTensor.u8_blob`.
    Effective bpw: 8.5 (8 raw + 0.5 scale per 32-element group).

``"q4_0_packed"``
    Bit-packed Q4_0: two nibbles per byte via
    :data:`~neuropress.t2n.packing.BitPacking4`.
    Effective bpw: 4.5 (4 raw + 0.5 scale per 32-element group).

``"q4_1_packed"``
    Bit-packed asymmetric Q4_1: two nibbles per byte with per-group f16 scale
    and zero-point.  Effective bpw: 5.0 (4 raw + 0.5 scale + 0.5 zero-point
    per 32-element group).  **Not exportable to tract NNEF** (no Q4_1 binary
    header type exists); works for in-memory use and safetensors storage.

``"q2_0_packed"``
    Bit-packed Q2_0: four 2-bit values per byte via
    :data:`~neuropress.t2n.packing.BitPacking2`.
    Effective bpw: 2.5 (2 raw + 0.5 scale per 32-element group).

``"identity"``
    Returns the GPTQ-corrected dequantized float weight unchanged.
    No storage reduction (16 bpw).

Import :mod:`neuropress.transforms` (or this module directly) to make all
keys available for :class:`~neuropress.estimator_configs.GptqEstimatorConfig`::

    {"type": "gptq", "weight_transform": "q4_0_packed", "group_size": 32}

Q4_0 / Q2_0 shape constraint: the product of all weight dimensions after dim-0
must be divisible by 32 (the group size).  For ``nn.Linear`` this means
``in_features % 32 == 0``.
"""

from __future__ import annotations

import functools
import logging
import operator
from typing import Any

import torch
from torch import nn
from torch_to_nnef.compress import offloaded_tensor_qtensor
from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.tensor.quant import fp_to_tract_q4_0_with_min_max_calibration
from torch_to_nnef.tensor.quant.base import (
    QScheme,
    QTensor,
    qscale_per_group_f16_min_max_calibration,
)
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

from ..estimator_configs import WEIGHT_TRANSFORM_REGISTRY, WeightTransformEntry
from ..t2n.packing import BitPacking2, BitPacking4

LOGGER = logging.getLogger(__name__)

Q4_X_GROUP_SIZE: int = 32
"""Column-group size required by the tract Q4_0 format."""


def _identity_transform(weight: torch.Tensor, name: str) -> torch.Tensor:
    """Return the GPTQ-corrected *weight* unchanged.

    Args:
        weight: GPTQ-corrected dequantized float weight tensor.
        name: Fully-qualified module name (unused).

    Returns:
        *weight* as-is.
    """
    return weight


"""Identity weight-transform: ``"identity"`` in :data:`.WEIGHT_TRANSFORM_REGISTRY`.

The identity transform returns the GPTQ-corrected float weight unchanged.
Use it when you want GPTQ's numerical weight correction without converting to
a backend-specific quantized format (e.g. for float16 export or debugging)::

    {"type": "gptq", "weight_transform": "identity"}

Unlike ``"q4_0"``, the identity transform has no shape constraints and
accepts any module targeted by the estimator.
"""
WEIGHT_TRANSFORM_REGISTRY["identity"] = WeightTransformEntry(
    transform=_identity_transform,
    # Identity stores the dequantized float weight unchanged (no size reduction).
    storage_bits=16,
)

# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _is_q4_x_compatible(mod: nn.Module) -> bool:
    """Return True if *mod*'s weight shape satisfies tract Q4_0 constraints.

    tract Q4_0 requires the product of all dims after dim-0 to be divisible
    by :data:`Q4_X_GROUP_SIZE`.  For ``nn.Linear`` this is
    ``in_features % 32 == 0``.

    Args:
        mod: Module to inspect.

    Returns:
        True iff the module has a weight whose inner dimensions are
        divisible by :data:`Q4_X_GROUP_SIZE`.
    """
    w = getattr(mod, "weight", None)
    if w is None or w.dim() < 2:
        return False
    return functools.reduce(operator.mul, w.shape[1:]) % Q4_X_GROUP_SIZE == 0


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _make_q4_0_tensor(
    weight: torch.Tensor,
    name: str,
    *,
    suffix_name: str = "q4_0",
) -> torch.Tensor:
    """Wrap *weight* as a standard (unpacked) ``QTensorTractScaleOnly``.

    Args:
        weight: The quantization-corrected dequantized float tensor.
        name: Fully-qualified module name used as ``nnef_name`` prefix.
        suffix_name: Suffix appended to the tensor name for offloaded storage.

    Returns:
        A ``QTensorTractScaleOnly`` wrapping *weight*.
    """
    orig_shape = weight.shape

    def _calib(w: torch.Tensor) -> Any:
        # offloaded_tensor_qtensor may pass w as a flat 1-D storage view.
        w = w.detach().reshape(orig_shape)
        q = fp_to_tract_q4_0_with_min_max_calibration(w)
        q.nnef_name = f"{name}.weight"
        return q

    return offloaded_tensor_qtensor(_calib, weight, suffix_name)


def _make_q4_0_packed_tensor(
    weight: torch.Tensor,
    name: str,
    *,
    suffix_name: str = "q4_0_packed",
) -> torch.Tensor:
    """Wrap *weight* as a bit-packed ``QTensorTractScaleOnly``.

    Uses :data:`~neuropress.t2n.packing.BitPacking4` as a ``u8_compressor``
    so that the stored
    :attr:`~torch_to_nnef.tensor.quant.base.QTensor.u8_blob` is 2× smaller
    than the standard Q4_0 representation (two 4-bit values per byte).

    Args:
        weight: The quantization-corrected dequantized float tensor.
        name: Fully-qualified module name used as ``nnef_name`` prefix.
        suffix_name: Suffix appended to the tensor name for offloaded storage.

    Returns:
        A ``QTensorTractScaleOnly`` with ``BitPacking4`` compression applied.
    """
    orig_shape = weight.shape

    def _calib(w: torch.Tensor) -> Any:
        # offloaded_tensor_qtensor may pass w as a flat 1-D storage view.
        w = w.detach().reshape(orig_shape)
        with torch.no_grad():
            q_scheme = qscale_per_group_f16_min_max_calibration(
                w, n_bits=4, group_size=Q4_X_GROUP_SIZE
            )
        q = QTensorTractScaleOnly(
            w,
            qscheme=q_scheme,
            dequant_to_dtype=w.dtype,
            u8_compressors=[BitPacking4],
        )
        q.nnef_name = f"{name}.weight"
        return q

    return offloaded_tensor_qtensor(_calib, weight, suffix_name)


def _safe_q4_0_transform(weight: torch.Tensor, name: str) -> Any:
    """Wrap *weight* as Q4_0, returning *None* on quantization failure.

    Args:
        weight: Float weight tensor (quantization-corrected).
        name: Fully-qualified module name.

    Returns:
        A ``QTensorTractScaleOnly``, or *None* if quantization is impossible.
    """
    if weight.dim() < 2:
        LOGGER.warning("Skipping Q4_0 for '%s': %dD %s", name, tuple(weight.shape))
        return None
    try:
        return _make_q4_0_tensor(weight, name)
    except T2NErrorImpossibleQuantization as exc:
        LOGGER.error("Q4_0 wrapping failed for '%s': %s — skipping", name, exc)
        return None


def _safe_q4_0_packed_transform(weight: torch.Tensor, name: str) -> Any:
    """Wrap *weight* as bit-packed Q4_0, returning *None* on failure.

    Args:
        weight: Float weight tensor (quantization-corrected).
        name: Fully-qualified module name.

    Returns:
        A bit-packed ``QTensorTractScaleOnly``, or *None* if quantization
        is impossible.
    """
    if weight.dim() < 2:
        LOGGER.warning("Skipping Q4_0p for '%s': %dD %s", name, tuple(weight.shape))
        return None
    try:
        return _make_q4_0_packed_tensor(weight, name)
    except T2NErrorImpossibleQuantization as exc:
        LOGGER.error("Q4_0 (packed) wrapping failed for '%s': %s — skipping", name, exc)
        return None


# ---------------------------------------------------------------------------
# Registration (side-effect on import)
# ---------------------------------------------------------------------------


def _Q4_X_FILTER(_, mod):
    return _is_q4_x_compatible(mod)


def _Q4_X_NO_LM_HEAD_FILTER(name: str, mod: nn.Module) -> bool:
    # Exclude lm_head: when tied to embed_tokens, GPTQ correction corrupts
    # the embedding weights in-place via the shared tensor reference.
    return _is_q4_x_compatible(mod) and name.split(".")[-1] != "lm_head"


WEIGHT_TRANSFORM_REGISTRY["q4_0_unpacked"] = WeightTransformEntry(
    transform=_safe_q4_0_transform,
    filter_fn=_Q4_X_FILTER,
    # Standard tract Q4_0: one nibble per uint8 byte (unpacked) → 8 storage bits.
    # + f16 scale per 32-element group → 0.5 bpw overhead → 8.5 effective bpw.
    storage_bits=8.5,
    group_size=Q4_X_GROUP_SIZE,
)
# "q4_0" is the canonical short alias for the unpacked format.
WEIGHT_TRANSFORM_REGISTRY["q4_0"] = WEIGHT_TRANSFORM_REGISTRY["q4_0_unpacked"]

WEIGHT_TRANSFORM_REGISTRY["q4_0_packed"] = WeightTransformEntry(
    transform=_safe_q4_0_packed_transform,
    filter_fn=_Q4_X_FILTER,
    # Bit-packed Q4_0: two nibbles per uint8 byte → 4 storage bits.
    # + scale in f16
    storage_bits=4.5,
    group_size=Q4_X_GROUP_SIZE,
)

WEIGHT_TRANSFORM_REGISTRY["q4_0_packed_no_lm_head"] = WeightTransformEntry(
    transform=_safe_q4_0_packed_transform,
    filter_fn=_Q4_X_NO_LM_HEAD_FILTER,
    storage_bits=4.5,
    group_size=Q4_X_GROUP_SIZE,
)

# ---------------------------------------------------------------------------
# Generic packed QTensor helper (used by GPTQPanelEstimator for Q4 + Q2 subs)
# ---------------------------------------------------------------------------

_PACKERS = {2: BitPacking2, 4: BitPacking4}


def _make_packed_qtensor(
    weight: torch.Tensor,
    name: str,
    nbits: int,
    group_size: int = Q4_X_GROUP_SIZE,
) -> Any:
    """Wrap *weight* as a bit-packed ``QTensorTractScaleOnly`` with *nbits* bits.

    Generalises :func:`_make_q4_0_packed_tensor` to arbitrary bit-widths.
    Uses :data:`~neuropress.t2n.packing.BitPacking2` for 2-bit and
    :data:`~neuropress.t2n.packing.BitPacking4` for 4-bit storage.

    Args:
        weight: Quantization-corrected dequantized float weight tensor.
        name: Fully-qualified module path used as ``nnef_name`` prefix.
        nbits: Number of bits for quantization (2 or 4).
        group_size: Per-group quantization group size.

    Returns:
        A ``QTensorTractScaleOnly`` with bit-packed ``u8_blob``.

    Raises:
        T2NErrorImpossibleQuantization: If the weight shape is incompatible.
    """
    packer = _PACKERS.get(nbits)
    suffix = f"q{nbits}_0_packed"
    orig_shape = weight.shape

    def _calib(w: torch.Tensor) -> Any:
        # offloaded_tensor_qtensor may pass w as a flat 1-D storage view.
        w = w.detach().reshape(orig_shape)
        with torch.no_grad():
            q_scheme = qscale_per_group_f16_min_max_calibration(
                w, n_bits=nbits, group_size=group_size
            )
        q = QTensorTractScaleOnly(
            w,
            qscheme=q_scheme,
            dequant_to_dtype=w.dtype,
            u8_compressors=[packer] if packer is not None else [],
        )
        q.nnef_name = f"{name}.weight"
        return q

    return offloaded_tensor_qtensor(_calib, weight, suffix)


WEIGHT_TRANSFORM_REGISTRY["q2_0_packed"] = WeightTransformEntry(
    transform=lambda w, n: _make_packed_qtensor(w, n, nbits=2),
    filter_fn=_Q4_X_FILTER,
    # Bit-packed Q2_0: four 2-bit values per uint8 byte → 2 storage bits.
    storage_bits=2,
    group_size=Q4_X_GROUP_SIZE,
)

# ---------------------------------------------------------------------------
# Asymmetric Q4_1 packed (scale + zero-point per group, safetensors only)
# ---------------------------------------------------------------------------


class QScaleZpPerGroupF16(QScheme):
    """Asymmetric f16 scale + zero-point per group (Q4_1 format).

    Unlike the symmetric :class:`~torch_to_nnef.tensor.quant.base.QScheme`
    subclasses used for Q4_0, this scheme stores both a *scale* and a
    *zero_point* for every quantization group, enabling asymmetric 4-bit
    quantization.  The format is **not** supported by the tract NNEF binary
    serialiser; it is intended for in-memory use and safetensors storage only.

    Quantization (per group of ``group_size`` elements):

    .. code-block:: text

        scale     = (max - min) / (2^n_bits - 1)
        zero_point = round(-min / scale)     # integer in [0, 2^n_bits - 1]
        q          = clamp(round(x / scale + zero_point), 0, 2^n_bits - 1)

    Dequantization:

    .. code-block:: text

        x̂ = (q - zero_point) * scale

    Attributes:
        group_size: Number of weight elements per quantization group.
        scale: Per-group scale in f16, shape ``[n_groups, 1]``.
        zero_point: Per-group zero-point in f16, shape ``[n_groups, 1]``.
        n_bits: Quantization bit-width (typically 4).
    """

    def __init__(
        self,
        group_size: int,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        n_bits: int,
    ) -> None:
        """Initialise the qscheme from pre-calibrated parameters.

        Args:
            group_size: Number of elements per group.
            scale: f16 scale tensor, shape ``[n_groups, 1]``.
            zero_point: f16 zero-point tensor, shape ``[n_groups, 1]``.
            n_bits: Quantization bit-width.
        """
        self.group_size = group_size
        self.scale = scale.to(torch.float16)
        self.zero_point = zero_point.to(torch.float16)
        self.n_bits = n_bits

    def quantize_as_torch(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Raise: native torch does not support asymmetric per-group Q."""
        raise T2NErrorNotImplemented(
            "QScaleZpPerGroupF16 does not support quantize_as_torch"
        )

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Quantize *fp_tensor* to uint8 using per-group scale + zero-point.

        Args:
            fp_tensor: Float tensor of any shape whose volume is divisible by
                :attr:`group_size`.

        Returns:
            uint8 tensor of the same shape as *fp_tensor*.
        """
        fp_per_group = fp_tensor.view(-1, self.group_size)
        recip_scale = torch.where(self.scale == 0, self.scale, 1.0 / self.scale)
        q = (
            (fp_per_group * recip_scale + self.zero_point)
            .round()
            .clamp(0, 2**self.n_bits - 1)
        )
        return q.to(torch.uint8).reshape(fp_tensor.shape)

    def dequantize(
        self, u8_tensor: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantize *u8_tensor* back to float.

        Args:
            u8_tensor: uint8 tensor of the same shape that was passed to
                :meth:`quantize_as_u8`.
            target_dtype: Output float dtype.

        Returns:
            Dequantized float tensor in *target_dtype*.
        """
        u8_per_group = u8_tensor.view(-1, self.group_size)
        fp = (
            u8_per_group.to(target_dtype) - self.zero_point.to(target_dtype)
        ) * self.scale.to(target_dtype)
        return fp.reshape(u8_tensor.shape)


def _qscalezp_per_group_f16_min_max_calibration(
    fp_tensor: torch.Tensor,
    n_bits: int,
    group_size: int,
) -> QScaleZpPerGroupF16:
    """Calibrate *fp_tensor* and return a :class:`QScaleZpPerGroupF16`.

    Uses asymmetric min-max calibration: the range ``[min, max]`` of each
    group is mapped linearly to ``[0, 2^n_bits - 1]``.

    Args:
        fp_tensor: Weight tensor to calibrate.  Its volume must be divisible
            by *group_size*.
        n_bits: Quantization bit-width.
        group_size: Number of weight elements per group.

    Returns:
        A calibrated :class:`QScaleZpPerGroupF16` instance.

    Raises:
        T2NErrorImpossibleQuantization: If ``volume % group_size != 0``.
    """
    volume = fp_tensor.numel()
    if volume % group_size != 0:
        raise T2NErrorImpossibleQuantization(
            f"tensor volume {volume} is not divisible by group_size {group_size}"
        )
    fp_per_group = fp_tensor.view(-1, group_size)  # [n_groups, group_size]
    n_groups = fp_per_group.shape[0]
    max_val = fp_per_group.max(dim=1).values  # [n_groups]
    min_val = fp_per_group.min(dim=1).values  # [n_groups]
    max_val_int = 2**n_bits - 1
    scale = (max_val - min_val) / max_val_int  # [n_groups]
    # Avoid division by zero for constant groups.
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zero_point = (-min_val / safe_scale).round().clamp(0, max_val_int)  # [n_groups]
    qshape = [n_groups, 1]
    return QScaleZpPerGroupF16(
        group_size=group_size,
        scale=scale.reshape(qshape).half(),
        zero_point=zero_point.reshape(qshape).half(),
        n_bits=n_bits,
    )


class QTensorScaleZp(QTensor):
    """Asymmetric Q4_1 packed tensor with f16 scale and zero-point per group.

    Extends :class:`~torch_to_nnef.tensor.quant.base.QTensor` with a
    :class:`QScaleZpPerGroupF16` qscheme.  The underlying bit-packing uses
    :data:`~neuropress.t2n.packing.BitPacking4` (two 4-bit values per byte),
    giving 4 raw storage bits plus 0.5 scale bits and 0.5 zero-point bits per
    32-element group — a total of 5 effective bpw.

    .. note::

       The tract NNEF binary format has no Q4_1 header type, so
       :meth:`write_in_file` always raises
       :exc:`~torch_to_nnef.exceptions.T2NErrorNotImplemented`.  Use
       :func:`~neuropress.t2n.qtensors.save_qtensor` and
       :func:`~neuropress.t2n.qtensors.load_qtensor` for persistence.
    """

    def write_in_file(self, dirpath: Any, label: str, **kwargs: Any) -> None:
        """Raise: Q4_1 cannot be exported to tract NNEF.

        Args:
            dirpath: Unused.
            label: Unused.
            **kwargs: Unused.

        Raises:
            T2NErrorNotImplemented: Always.
        """
        raise T2NErrorNotImplemented(
            "QTensorScaleZp (Q4_1) cannot be written to tract NNEF — "
            "use save_qtensor() for safetensors persistence."
        )


def _make_q4_1_packed_tensor(
    weight: torch.Tensor,
    name: str,
    *,
    suffix_name: str = "q4_1_packed",
) -> torch.Tensor:
    """Wrap *weight* as a bit-packed asymmetric Q4_1 :class:`QTensorScaleZp`.

    Uses :data:`~neuropress.t2n.packing.BitPacking4` for two-nibble packing
    and :class:`QScaleZpPerGroupF16` for per-group f16 scale + zero-point.

    Args:
        weight: Quantization-corrected dequantized float weight tensor.
        name: Fully-qualified module name used as ``nnef_name`` prefix.
        suffix_name: Suffix for offloaded tensor naming.

    Returns:
        A :class:`QTensorScaleZp` wrapping *weight*.
    """
    orig_shape = weight.shape

    def _calib(w: torch.Tensor) -> Any:
        # offloaded_tensor_qtensor may pass w as a flat 1-D storage view.
        w = w.detach().reshape(orig_shape)
        with torch.no_grad():
            q_scheme = _qscalezp_per_group_f16_min_max_calibration(
                w, n_bits=4, group_size=Q4_X_GROUP_SIZE
            )
        q = QTensorScaleZp(
            w,
            qscheme=q_scheme,
            dequant_to_dtype=w.dtype,
            u8_compressors=[BitPacking4],
        )
        q.nnef_name = f"{name}.weight"
        return q

    return offloaded_tensor_qtensor(_calib, weight, suffix_name)


def _safe_q4_1_packed_transform(weight: torch.Tensor, name: str) -> Any:
    """Wrap *weight* as bit-packed asymmetric Q4_1, returning *None* on failure.

    Args:
        weight: Float weight tensor (quantization-corrected).
        name: Fully-qualified module name.

    Returns:
        A :class:`QTensorScaleZp`, or *None* if quantization is impossible.
    """
    if weight.dim() < 2:
        LOGGER.warning("Skipping Q4_1p for '%s': %dD %s", name, tuple(weight.shape))
        return None
    try:
        return _make_q4_1_packed_tensor(weight, name)
    except T2NErrorImpossibleQuantization as exc:
        LOGGER.error("Q4_1 (packed) wrapping failed for '%s': %s — skipping", name, exc)
        return None


WEIGHT_TRANSFORM_REGISTRY["q4_1_packed"] = WeightTransformEntry(
    transform=_safe_q4_1_packed_transform,
    filter_fn=_Q4_X_FILTER,
    # Bit-packed asymmetric Q4_1: 4 raw bits + f16 scale + f16 zero-point per
    # 32-element group → 4 + 16/32 + 16/32 = 5.0 effective bpw.
    storage_bits=5.0,
    group_size=Q4_X_GROUP_SIZE,
)
