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
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_to_nnef.compress import offloaded_tensor_qtensor
from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.tensor.opaque import OFFLOAD_STATE_KEY, SupportsOffloadState
from torch_to_nnef.tensor.quant import fp_to_tract_q4_0_with_min_max_calibration
from torch_to_nnef.tensor.quant.base import (
    QScalePerGroupF16,
    QScheme,
    QTensor,
    qscale_per_group_f16_min_max_calibration,
)
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

from ..estimator_configs import WEIGHT_TRANSFORM_REGISTRY, WeightTransformEntry
from ..plan import F16_MIN_POSITIVE, QuantizedWeight
from ..qscheme.zp_scale import ZpScaleScheme
from ..t2n.packing import BitPacking2, BitPacking4, BitPacking8
from ..t2n.qtensors import qtensor_from_blob, qtensor_shape

LOGGER = logging.getLogger(__name__)

Q4_X_GROUP_SIZE: int = 32
"""Column-group size required by the tract Q4_0 format."""

_TRANSPOSED_ERROR = (
    "Conv1D scale groups run along the first weight dimension, which flat "
    "per-block storage cannot express without recalibrating the weights"
)

_F16_MAX: float = 65504.0
"""Largest finite f16 value; tract and f16 kernels decode weights in this range."""


def _identity_transform(weight: QuantizedWeight, name: str) -> torch.Tensor:
    """Materialize the selected values without changing their scheme."""
    return weight.dequantize()


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


def _f16_block_scales(
    scale: torch.Tensor, blocks: int, name: str, max_steps: torch.Tensor | float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round per-group scales to f16 and repeat them for each storage block.

    A scale below the f16 range belongs to a group of near-zero weights. It is
    raised to the smallest positive f16 value, and the returned per-block mask
    tells the caller to store those blocks as exact zeros: their codes were
    chosen for a far smaller scale and would otherwise decode thousands of
    times too large.

    Args:
        scale: Per-group scales.
        blocks: Storage blocks per group.
        name: Layer name for messages.
        max_steps: Largest distance in steps between a code and the offset.

    Returns:
        ``(scales, underflow)``, both shaped ``[n_blocks, 1]``.

    Raises:
        T2NErrorImpossibleQuantization: If a scale is negative, NaN or above
            the f16 range, or a code would decode outside that range.
    """
    scales = scale.reshape(-1, 1).repeat_interleave(blocks, dim=0).to(torch.float16)
    if not torch.all(torch.isfinite(scales) & (scales >= 0)):
        raise T2NErrorImpossibleQuantization(
            f"{name}: quantization scales are not representable in f16"
        )
    if torch.any(scales.float() * max_steps > _F16_MAX):
        raise T2NErrorImpossibleQuantization(
            f"{name}: quantized weights would decode outside the f16 range"
        )
    underflow = scales == 0
    if underflow.any():
        LOGGER.warning(
            "%s: storing %d blocks as zero, their scales are below the f16 range",
            name,
            int(underflow.sum()),
        )
        scales = scales.masked_fill(underflow, F16_MIN_POSITIVE)
    return scales, underflow


def _pack_keeping_shape(packer: Any, codes: torch.Tensor) -> torch.Tensor:
    """Bit-pack *codes*, refusing a layout that decompression could not restore.

    Packing keeps the shape only when the last dimension is divisible by the
    values per byte, and QTensors decode whatever shape comes back.

    Raises:
        ValueError: If the codes cannot be packed or would decode flat.
    """
    packed = packer.compress(codes)
    if packed.dim() != codes.dim():
        raise ValueError(
            f"{packer.nbits}-bit packing cannot restore weight shape "
            f"{tuple(codes.shape)}; use an unpacked format"
        )
    return packed


def _store_blocks_as_zero(
    codes: torch.Tensor,
    blocks: torch.Tensor,
    zero_codes: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Return *codes* with the masked storage blocks set to *zero_codes*."""
    if not blocks.any():
        return codes
    selected = blocks.flatten()
    per_block = codes.reshape(-1, block_size).clone()
    per_block[selected] = zero_codes.reshape(-1, 1)[selected]
    return per_block.reshape(codes.shape)


@torch.no_grad()
def _q4_0_from_assignment(
    weight: QuantizedWeight,
    name: str,
    *,
    packed: bool = False,
) -> torch.Tensor:
    """Pack selected symmetric 4-bit codes without recalibrating the scheme.

    Blocks whose scale is below the f16 range are stored as exact zeros.

    Raises:
        ValueError: If the scheme cannot be expressed in Q4_0.
        T2NErrorImpossibleQuantization: If a scale is outside the f16 range.
    """
    scheme = weight.scheme
    if not isinstance(scheme, ZpScaleScheme):
        raise ValueError("Q4_0 requires a homogeneous affine scheme")
    if weight.transposed:
        raise ValueError(_TRANSPOSED_ERROR)
    group_size = scheme.calibration_data_shape[-1]
    if scheme.nbits != 4 or group_size % 32 or not torch.all(scheme.zero_point == 8):
        raise ValueError(
            "Q4_0 requires symmetric 4-bit quantization with groups divisible by 32"
        )
    scales, underflow = _f16_block_scales(
        scheme.scale, group_size // Q4_X_GROUP_SIZE, name, max_steps=8.0
    )
    codes = _store_blocks_as_zero(
        weight.codes, underflow, torch.full_like(underflow, 8, dtype=torch.uint8), 32
    )
    return qtensor_from_blob(
        QTensorTractScaleOnly,
        _pack_keeping_shape(BitPacking4, codes) if packed else codes,
        codes.shape,
        QScalePerGroupF16(Q4_X_GROUP_SIZE, scales, 4),
        weight.dtype,
        [BitPacking4] if packed else [],
        f"{name}.weight",
    )


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
    transform=_q4_0_from_assignment,
    filter_fn=_Q4_X_FILTER,
    # Standard tract Q4_0: one nibble per uint8 byte (unpacked) → 8 storage bits.
    # + f16 scale per 32-element group → 0.5 bpw overhead → 8.5 effective bpw.
    storage_bits=8.5,
    group_size=Q4_X_GROUP_SIZE,
    nbits=4,
    symmetric=True,
)
# "q4_0" is the canonical short alias for the unpacked format.
WEIGHT_TRANSFORM_REGISTRY["q4_0"] = WEIGHT_TRANSFORM_REGISTRY["q4_0_unpacked"]

WEIGHT_TRANSFORM_REGISTRY["q4_0_packed"] = WeightTransformEntry(
    transform=functools.partial(_q4_0_from_assignment, packed=True),
    filter_fn=_Q4_X_FILTER,
    # Bit-packed Q4_0: two nibbles per uint8 byte → 4 storage bits.
    # + scale in f16
    storage_bits=4.5,
    group_size=Q4_X_GROUP_SIZE,
    nbits=4,
    symmetric=True,
)

WEIGHT_TRANSFORM_REGISTRY["q4_0_packed_no_lm_head"] = WeightTransformEntry(
    transform=functools.partial(_q4_0_from_assignment, packed=True),
    filter_fn=_Q4_X_NO_LM_HEAD_FILTER,
    storage_bits=4.5,
    group_size=Q4_X_GROUP_SIZE,
    nbits=4,
    symmetric=True,
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
    transform=lambda w, n: _affine_from_assignment(w, n, nbits=2),
    filter_fn=_Q4_X_FILTER,
    # Bit-packed Q2_0: four 2-bit values per uint8 byte → 2 storage bits,
    # + f16 scale per 32-element group, + the same again for per-block offsets.
    storage_bits=2.5,
    asymmetric_storage_bits=3.0,
    group_size=Q4_X_GROUP_SIZE,
    nbits=2,
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
        zero_point: Per-group zero-point in f16, shape ``[n_groups, 1]``, or
            ``[1, 1]`` when every group shares one value, as symmetric schemes
            do: a repeated offset would cost 16 bits per group for nothing.
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
        zero_point = zero_point.to(torch.float16)
        if zero_point.numel() > 1 and torch.all(zero_point == zero_point.flatten()[0]):
            zero_point = zero_point.flatten()[:1].reshape(1, 1)
        self.zero_point = zero_point
        self.n_bits = n_bits

    def per_group_zero_point(self) -> torch.Tensor:
        """Return one zero-point for each group, as kernels index them."""
        return self.zero_point.expand_as(self.scale).clone()

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


class QTensorScaleZp(QTensor, SupportsOffloadState):
    """Packed affine weights with f16 scale and offset per group.

    Supports 2-, 4- and 8-bit in-memory storage and safetensors persistence.
    These affine layouts have no tract NNEF binary header, so write_in_file
    raises instead of silently changing the format.

    Mixed-precision branches set export_as_float: tract has no format for them
    and the branch split is what makes the rest of the layer exportable, so they
    are written as float variables holding the selected values.
    """

    OFFLOAD_STATE_TAG = "neuropress.qtensor_scale_zp.v1"

    export_as_float: bool = False

    def to_offload_state(self) -> dict[str, Any]:
        """Return the compact state offloaded instead of the float placeholder."""
        return {
            OFFLOAD_STATE_KEY: self.OFFLOAD_STATE_TAG,
            "shape": qtensor_shape(self),
            "u8_blob": self.u8_blob,
            "u8_compressors": self.u8_compressors,
            "qscheme": self.qscheme,
            "dequant_to_dtype": self.dequant_to_dtype,
            "nnef_name": self.nnef_name,
            "export_as_float": self.export_as_float,
        }

    @classmethod
    def from_offload_state(
        cls, state: dict[str, Any], target_device: Any = None
    ) -> QTensorScaleZp:
        """Rebuild the tensor from :meth:`to_offload_state` output."""
        assert cls.is_offload_state(state), state
        q = qtensor_from_blob(
            cls,
            state["u8_blob"],
            torch.Size(state["shape"]),
            state["qscheme"],
            state["dequant_to_dtype"],
            state["u8_compressors"],
            state["nnef_name"],
        )
        q.export_as_float = state["export_as_float"]
        if target_device is not None and q.device != target_device:
            q.to_device(target_device)
        return q

    @classmethod
    def write_offload_state_in_file(
        cls,
        state: dict[str, Any],
        dirpath: Any,
        label: str,
        inference_target: Any = None,
    ) -> None:
        """Export an offloaded tensor without keeping it loaded afterwards."""
        cls.from_offload_state(state).write_in_file(dirpath, label, inference_target)

    def write_in_file(
        self, dirpath: Any, label: str, inference_target: Any = None
    ) -> None:
        """Write a mixed-precision branch as a float NNEF variable.

        Args:
            dirpath: Output directory.
            label: Variable label, used as the file name.
            inference_target: Unused; float variables have one format.

        Raises:
            T2NErrorNotImplemented: If this tensor is not a mixed branch.
        """
        if not self.export_as_float:
            raise T2NErrorNotImplemented(
                "QTensorScaleZp cannot be written to tract NNEF - "
                "use save_qtensor() for safetensors persistence."
            )
        from torch_to_nnef.nnef_io.writer import write_nnef_tensor

        write_nnef_tensor(
            self.decompress().detach().cpu().numpy(),
            str(Path(dirpath) / f"{label}.dat"),
            quantized=False,
        )


_AFFINE_PACKERS = {2: BitPacking2, 4: BitPacking4, 8: BitPacking8}


@torch.no_grad()
def _affine_from_assignment(
    weight: QuantizedWeight,
    name: str,
    *,
    nbits: int,
    block_size: int = Q4_X_GROUP_SIZE,
) -> torch.Tensor:
    """Pack affine codes with f16 scale/offset for in-memory and safetensors use.

    Raises:
        ValueError: If the scheme cannot be expressed in this layout.
        T2NErrorImpossibleQuantization: If a scale or offset is outside the
            f16 range.
    """
    scheme = weight.scheme
    if not isinstance(scheme, ZpScaleScheme) or scheme.nbits != nbits:
        raise ValueError(f"Expected a homogeneous {nbits}-bit affine scheme")
    if weight.transposed:
        raise ValueError(_TRANSPOSED_ERROR)
    group_size = scheme.calibration_data_shape[-1]
    if group_size % block_size:
        raise ValueError(
            f"Packed affine formats require groups divisible by {block_size}"
        )
    packer = _AFFINE_PACKERS.get(nbits)
    if packer is None:
        raise ValueError(f"Packed affine formats do not support {nbits} bits")
    blocks = group_size // block_size
    zero = (
        scheme.zero_point.reshape(-1, 1)
        .repeat_interleave(blocks, dim=0)
        .to(torch.float16)
    )
    if not torch.isfinite(zero).all():
        raise T2NErrorImpossibleQuantization(
            f"{name}: quantization offsets are not representable in f16"
        )
    max_code = 2**nbits - 1
    max_steps = torch.maximum(zero.float().abs(), (max_code - zero.float()).abs())
    scale, underflow = _f16_block_scales(scheme.scale, blocks, name, max_steps)
    codes = _store_blocks_as_zero(
        weight.codes,
        underflow,
        zero.float().round().clamp(0, max_code).to(torch.uint8),
        block_size,
    )
    return qtensor_from_blob(
        QTensorScaleZp,
        _pack_keeping_shape(packer, codes),
        codes.shape,
        QScaleZpPerGroupF16(block_size, scale, zero, nbits),
        weight.dtype,
        [packer],
        f"{name}.weight",
    )


WEIGHT_TRANSFORM_REGISTRY["q4_1_packed"] = WeightTransformEntry(
    transform=functools.partial(_affine_from_assignment, nbits=4),
    filter_fn=_Q4_X_FILTER,
    # Bit-packed Q4_1: 4 raw bits + f16 scale per 32-element group → 4.5 bpw.
    # Asymmetric schemes add an f16 zero-point per group → 5.0 effective bpw.
    storage_bits=4.5,
    asymmetric_storage_bits=5.0,
    group_size=Q4_X_GROUP_SIZE,
    nbits=4,
)
