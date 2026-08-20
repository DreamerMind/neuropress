"""ZpScaleScheme – concrete QScheme using zero-point / scale quantization.

Inherits from ``torch_to_nnef.tensor.quant.QScheme`` as required by PLAN.md §5.2.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch_to_nnef.tensor.quant import QScheme


class ZpScaleScheme(QScheme):
    """Affine zero-point/scale quantization scheme.

    Args:
        zero_point: Integer offset tensor. Shape matches ``qscheme_data_shape``.
        scale: Positive scale tensor. Shape matches ``qscheme_data_shape``.
        nbits: Bit-width of the quantized representation.
        calibration_data_shape: 4-D shape ``[rows, 1, n_groups, group_size]`` used
            to reshape a weight tensor before applying quantization. The odd dims
            (1, group_size) are the "collapsed" (reduction) dims; the even dims
            (rows, n_groups) are the "kept" dims.
        qscheme_data_shape: Same rank as *calibration_data_shape* with collapsed
            dims set to 1, i.e. the shape of *scale* / *zero_point*.
    """

    def __init__(
        self,
        zero_point: torch.Tensor,
        scale: torch.Tensor,
        nbits: int,
        calibration_data_shape: list[int],
        qscheme_data_shape: list[int],
    ) -> None:
        """Initialize the scheme with the given tensors and shape descriptors."""
        self.zero_point = zero_point
        self.scale = scale
        self.nbits = nbits
        self.calibration_data_shape = calibration_data_shape
        self.qscheme_data_shape = qscheme_data_shape

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def maxq(self) -> int:
        """Maximum quantized value (``2**nbits - 1``)."""
        return 2**self.nbits - 1

    @property
    def rank(self) -> int:
        """Number of dimensions in the qscheme data shape."""
        return len(self.qscheme_data_shape)

    @property
    def dims_indexes_to_collapse(self) -> list[int]:
        """Odd-indexed dims (reduction axes)."""
        return list(range(1, self.rank, 2))

    @property
    def dims_indexes_to_keep(self) -> list[int]:
        """Even-indexed dims."""
        return list(range(0, self.rank, 2))

    # ------------------------------------------------------------------
    # QScheme interface
    # ------------------------------------------------------------------

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Quantize *fp_tensor* to uint8.

        Works for:
        - Full weight tensor whose numel matches ``calibration_data_shape``.
        - Single column / sub-tensor: broadcast via rank-matching unsqueeze.
        """
        total = math.prod(self.calibration_data_shape)
        if fp_tensor.numel() == total:
            data = fp_tensor.reshape(self.calibration_data_shape).to(self.scale.dtype)
        else:
            # Broadcast path: add trailing singleton dims to match scale rank.
            extra = len(self.scale.shape) - fp_tensor.dim()
            view_shape = list(fp_tensor.shape) + [1] * extra
            data = fp_tensor.view(view_shape).to(self.scale.dtype)

        q = (data / self.scale + self.zero_point).clamp(0, self.maxq).round()
        return q.reshape(fp_tensor.shape).to(torch.uint8)

    def quantize_as_torch(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Quantize *fp_tensor* and return as int32 (for NNEF export)."""
        return self.quantize_as_u8(fp_tensor).to(torch.int32)

    def dequantize(
        self, u8_tensor: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantize *u8_tensor* (uint8 / int32) back to *target_dtype* float."""
        total = math.prod(self.calibration_data_shape)
        if u8_tensor.numel() == total:
            data = u8_tensor.reshape(self.calibration_data_shape).to(target_dtype)
        else:
            extra = len(self.scale.shape) - u8_tensor.dim()
            view_shape = list(u8_tensor.shape) + [1] * extra
            data = u8_tensor.view(view_shape).to(target_dtype)

        result = (data - self.zero_point.to(target_dtype)) * self.scale.to(target_dtype)
        return result.reshape(u8_tensor.shape)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def offload_tensors(self, name: str, offload_dir: Path) -> ZpScaleScheme:
        """Return a new ZpScaleScheme with scale/zp offloaded to disk."""
        try:
            from torch_to_nnef.tensor.offload import OffloadedTensor

            scale = OffloadedTensor.from_original_tensor(
                self.scale, f"{name}.scale", offload_dir
            )
            zp = OffloadedTensor.from_original_tensor(
                self.zero_point, f"{name}.zero_point", offload_dir
            )
        except Exception:
            # Fallback: keep in memory if offloading is unavailable.
            scale = self.scale
            zp = self.zero_point

        return ZpScaleScheme(
            zero_point=zp,
            scale=scale,
            nbits=self.nbits,
            calibration_data_shape=list(self.calibration_data_shape),
            qscheme_data_shape=list(self.qscheme_data_shape),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ZpScaleScheme(nbits={self.nbits}, "
            f"calib_shape={self.calibration_data_shape}, "
            f"qscheme_shape={self.qscheme_data_shape})"
        )
