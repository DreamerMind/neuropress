"""Ternary calibration and GPTQ integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..qscheme.ternary import TernaryScaleScheme
from ..qscheme.zp_scale import ZpScaleScheme
from .gptq import GPTQCalibration
from .zp_scale_calib import ZpScaleCalibration


@dataclass
class TernaryScaleCalibration(ZpScaleCalibration):
    """Fit one least-squares ternary scale per quantization group."""

    nbits: int = field(default=2, init=False)
    symmetric: bool = field(default=True, init=False)
    threshold_ratio: float = 0.5
    max_iterations: int = 12
    _data: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def maxq(self) -> int:
        """Return the largest valid ternary code."""
        return 2

    def init(self) -> None:
        """Reset stored calibration data."""
        super().init()
        self._data = None

    def update_statistics(self, data: torch.Tensor) -> None:
        """Store the current weight group."""
        self._data = data

    def compute_qscheme(self) -> TernaryScaleScheme:
        """Fit scales and return a strict ternary scheme."""
        if self._data is None:
            raise RuntimeError("TernaryScaleCalibration has no data")
        if not 0.0 < self.threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be in (0, 1]")
        assert self.calibration_data_shape is not None
        assert self.qscheme_data_shape is not None

        data = self._data.reshape(self.calibration_data_shape)
        collapse = tuple(range(1, data.dim(), 2))
        abs_data = data.abs()
        scale = abs_data.mean(dim=collapse, keepdim=True).clamp_min(
            torch.finfo(data.dtype).eps
        )
        for _ in range(self.max_iterations):
            mask = abs_data >= scale * self.threshold_ratio
            count = mask.sum(dim=collapse, keepdim=True)
            fitted = (abs_data * mask).sum(
                dim=collapse, keepdim=True
            ) / count.clamp_min(1)
            fitted = torch.where(count > 0, fitted, scale)
            if torch.allclose(fitted, scale, rtol=1e-5, atol=0.0):
                scale = fitted
                break
            scale = fitted

        return TernaryScaleScheme(
            scale=scale.reshape(self.qscheme_data_shape),
            calibration_data_shape=list(self.calibration_data_shape),
            qscheme_data_shape=list(self.qscheme_data_shape),
            threshold_ratio=self.threshold_ratio,
        )


@dataclass
class TernaryGPTQCalibration(GPTQCalibration):
    """GPTQ calibration that retains strict ternary global schemes."""

    possible_quantizers: list[ZpScaleCalibration] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Install a ternary quantizer when constructed through the registry."""
        if not self.possible_quantizers:
            self.possible_quantizers = [
                TernaryScaleCalibration(granularity=self.granularity)
            ]
        super().__post_init__()

    def build_global_tensor_qscheme(
        self, qschemes: list[ZpScaleScheme], n_cols: int
    ) -> TernaryScaleScheme:
        """Concatenate per-group scales without losing ternary semantics."""
        if not qschemes or not all(isinstance(s, TernaryScaleScheme) for s in qschemes):
            raise TypeError("TernaryGPTQCalibration requires ternary group schemes")
        first = qschemes[0]
        assert isinstance(first, TernaryScaleScheme)
        col_dim = 2
        shape = list(first.qscheme_data_shape)
        shape[col_dim] = len(qschemes)
        calibration_shape = list(shape)
        calibration_shape[-1] = n_cols // len(qschemes)
        return TernaryScaleScheme(
            scale=torch.cat([s.scale for s in qschemes], dim=col_dim),
            calibration_data_shape=calibration_shape,
            qscheme_data_shape=shape,
            threshold_ratio=first.threshold_ratio,
        )
