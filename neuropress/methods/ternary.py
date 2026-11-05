"""Ternary calibration and GPTQ integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..qscheme.ternary import TernaryScaleScheme
from ..qscheme.zp_scale import ZpScaleScheme
from .gptq import GPTQCalibration, QProcessingData
from .zp_scale_calib import ZpScaleCalibration


@dataclass
class TernaryScaleCalibration(ZpScaleCalibration):
    """Fit one least-squares ternary scale per quantization group."""

    nbits: int = field(default=2, init=False)
    symmetric: bool = field(default=True, init=False)
    threshold_ratio: float = 0.5
    max_iterations: int = 12
    threshold_grid: tuple[float, ...] = ()
    column_weights: torch.Tensor | None = field(default=None, repr=False)
    _data: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def maxq(self) -> int:
        """Return the largest valid ternary code."""
        return 2

    def init(self) -> None:
        """Reset stored calibration data."""
        super().init()
        self._data = None
        self.column_weights = None

    def update_statistics(self, data: torch.Tensor) -> None:
        """Store the current weight group."""
        self._data = data

    def compute_qscheme(self) -> TernaryScaleScheme:
        """Fit scales (and per-group thresholds when a grid is set)."""
        if self._data is None:
            raise RuntimeError("TernaryScaleCalibration has no data")
        ratios = self.threshold_grid or (self.threshold_ratio,)
        if not all(0.0 < ratio <= 1.0 for ratio in ratios):
            raise ValueError("threshold_ratio must be in (0, 1]")
        assert self.calibration_data_shape is not None
        assert self.qscheme_data_shape is not None

        data = self._data.reshape(self.calibration_data_shape)
        collapse = tuple(range(1, data.dim(), 2))
        weights = self._broadcast_column_weights(data)

        best_scale = best_ratio = best_error = None
        for ratio in ratios:
            scale = self._fit_scale(data, collapse, ratio, weights)
            error = self._weighted_error(data, collapse, scale, ratio, weights)
            ratio_t = torch.full_like(scale, ratio)
            if best_error is None:
                best_scale, best_ratio, best_error = scale, ratio_t, error
                continue
            better = error < best_error
            best_scale = torch.where(better, scale, best_scale)
            best_ratio = torch.where(better, ratio_t, best_ratio)
            best_error = torch.where(better, error, best_error)

        assert best_scale is not None and best_ratio is not None
        return TernaryScaleScheme(
            scale=best_scale.reshape(self.qscheme_data_shape),
            calibration_data_shape=list(self.calibration_data_shape),
            qscheme_data_shape=list(self.qscheme_data_shape),
            threshold_ratio=(
                best_ratio.reshape(self.qscheme_data_shape)
                if self.threshold_grid
                else self.threshold_ratio
            ),
        )

    def _broadcast_column_weights(self, data: torch.Tensor) -> torch.Tensor:
        """Return per-column weights broadcastable to *data* (ones if unset)."""
        if self.column_weights is None:
            return torch.ones_like(data)
        weights = self.column_weights.to(device=data.device, dtype=data.dtype)
        if weights.numel() != data.shape[-1]:
            raise ValueError(
                f"column_weights has {weights.numel()} entries, expected "
                f"{data.shape[-1]} group columns"
            )
        return weights.reshape([1] * (data.dim() - 1) + [-1]).expand_as(data)

    def _fit_scale(
        self,
        data: torch.Tensor,
        collapse: tuple[int, ...],
        ratio: float,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Alternate the ternary support and its weighted least-squares scale."""
        abs_data = data.abs()
        scale = (
            (abs_data * weights).sum(dim=collapse, keepdim=True)
            / weights.sum(dim=collapse, keepdim=True)
        ).clamp_min(torch.finfo(data.dtype).eps)
        for _ in range(self.max_iterations):
            mask = (abs_data >= scale * ratio).to(data.dtype) * weights
            denominator = mask.sum(dim=collapse, keepdim=True)
            fitted = (abs_data * mask).sum(dim=collapse, keepdim=True) / (
                denominator.clamp_min(torch.finfo(data.dtype).tiny)
            )
            fitted = torch.where(denominator > 0, fitted, scale)
            if torch.allclose(fitted, scale, rtol=1e-5, atol=0.0):
                return fitted
            scale = fitted
        return scale

    @staticmethod
    def _weighted_error(
        data: torch.Tensor,
        collapse: tuple[int, ...],
        scale: torch.Tensor,
        ratio: float,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted squared reconstruction error per group."""
        threshold = scale * ratio
        levels = (data >= threshold).to(data.dtype) - (data <= -threshold).to(
            data.dtype
        )
        return ((data - levels * scale).square() * weights).sum(
            dim=collapse, keepdim=True
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
        ratios = [
            torch.as_tensor(
                s.threshold_ratio, dtype=s.scale.dtype, device=s.scale.device
            ).expand_as(s.scale)
            for s in qschemes
            if isinstance(s, TernaryScaleScheme)
        ]
        uniform = all(
            not isinstance(s.threshold_ratio, torch.Tensor)
            for s in qschemes
            if isinstance(s, TernaryScaleScheme)
        )
        return TernaryScaleScheme(
            scale=torch.cat([s.scale for s in qschemes], dim=col_dim),
            calibration_data_shape=calibration_shape,
            qscheme_data_shape=shape,
            threshold_ratio=(
                first.threshold_ratio if uniform else torch.cat(ratios, dim=col_dim)
            ),
        )

    def _configure_group_quantizer(
        self,
        quantizer: ZpScaleCalibration,
        q_data: QProcessingData,
        columns: torch.Tensor,
    ) -> None:
        """Weight group columns by GPTQ's per-column loss factor 1 / U[c, c]^2."""
        if isinstance(quantizer, TernaryScaleCalibration) and quantizer.threshold_grid:
            diagonal = torch.diagonal(q_data.inv_hessian_chol_upper)[columns]
            quantizer.column_weights = diagonal.reciprocal().square()
