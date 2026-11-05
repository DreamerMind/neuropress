"""Strict ternary zero-point/scale quantization."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from .zp_scale import ZpScaleScheme


class TernaryScaleScheme(ZpScaleScheme):
    """Quantize values to ``{-scale, 0, +scale}`` with codes ``{0, 1, 2}``."""

    def __init__(
        self,
        scale: torch.Tensor,
        calibration_data_shape: list[int],
        qscheme_data_shape: list[int],
        threshold_ratio: float | torch.Tensor = 0.5,
    ) -> None:
        ratio = torch.as_tensor(threshold_ratio)
        if not bool(((ratio > 0.0) & (ratio <= 1.0)).all()):
            raise ValueError("threshold_ratio must be in (0, 1]")
        super().__init__(
            zero_point=torch.ones_like(scale, dtype=torch.int32),
            scale=scale,
            nbits=2,
            calibration_data_shape=calibration_data_shape,
            qscheme_data_shape=qscheme_data_shape,
        )
        # A float applies to every group; a tensor shaped like ``scale``
        # carries one threshold per group.
        self.threshold_ratio = threshold_ratio

    @property
    def maxq(self) -> int:
        """Maximum code. Code 3 is deliberately invalid."""
        return 2

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Return strict ternary codes with the same shape as *fp_tensor*."""
        total = math.prod(self.calibration_data_shape)
        if fp_tensor.numel() == total:
            data = fp_tensor.reshape(self.calibration_data_shape).to(self.scale.dtype)
        else:
            extra = self.scale.dim() - fp_tensor.dim()
            view_shape = list(fp_tensor.shape) + [1] * extra
            data = fp_tensor.view(view_shape).to(self.scale.dtype)

        ratio = torch.as_tensor(
            self.threshold_ratio, dtype=self.scale.dtype, device=self.scale.device
        )
        threshold = self.scale * ratio
        codes = torch.ones_like(data, dtype=torch.uint8)
        codes = torch.where(data >= threshold, torch.full_like(codes, 2), codes)
        codes = torch.where(data <= -threshold, torch.zeros_like(codes), codes)
        return codes.reshape(fp_tensor.shape)

    def offload_tensors(self, name: str, offload_dir: Path) -> TernaryScaleScheme:
        """Return an offloaded scheme while preserving ternary semantics."""
        base = super().offload_tensors(name, offload_dir)
        return TernaryScaleScheme(
            scale=base.scale,
            calibration_data_shape=list(self.calibration_data_shape),
            qscheme_data_shape=list(self.qscheme_data_shape),
            threshold_ratio=self.threshold_ratio,
        )
