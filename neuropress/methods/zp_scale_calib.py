"""Zero-point / scale calibration methods.

Adapted from examples/methods/zp_scale.py and examples/methods/zp_scale_norm.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..qscheme.descriptor import GranularityType
from ..qscheme.zp_scale import ZpScaleScheme
from .base import CalibrationMethod


@dataclass
class ZpScaleCalibration(CalibrationMethod[ZpScaleScheme]):
    """Base class: zero-point / scale helpers shared by all Min-Max methods."""

    nbits: int = field(default=8)
    symmetric: bool = field(default=False)
    force_include_zero: bool = field(default=True)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def maxq(self) -> int:
        """Maximum quantized integer value (``2**nbits - 1``)."""
        return 2**self.nbits - 1

    def bpw(self, scale_n_bits: int = 16, offset_n_bits: int = 16) -> float:
        """Effective bits-per-weight including scale and zero-point overhead."""
        if self.granularity.granularity_type is not GranularityType.PER_GROUP:
            raise ValueError("bpw() requires PER_GROUP granularity")
        assert self.granularity.group_sizes
        n_elems = math.prod(self.granularity.group_sizes)
        n_total = (
            n_elems * self.nbits
            + scale_n_bits
            + offset_n_bits * (1 - int(self.symmetric))
        )
        return float(n_total) / n_elems

    # ------------------------------------------------------------------
    # Min / max helpers
    # ------------------------------------------------------------------

    def prepare_min_max(
        self, xmin: torch.Tensor, xmax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Clamp, symmetrize, and guard against zero-range."""
        if self.force_include_zero:
            xmin = xmin.clamp_max(0.0)
            xmax = xmax.clamp_min(0.0)
        if self.symmetric:
            xmax = torch.maximum(xmin.abs(), xmax)
            neg = xmin < 0
            if torch.any(neg):
                xmin[neg] = -xmax[neg]
        # guard against zero range
        zero_range = (xmin == 0) & (xmax == 0)
        xmin[zero_range] = -1.0
        xmax[zero_range] = 1.0
        return xmin, xmax

    def compute_zp_scale_from_min_max(
        self, xmin: torch.Tensor, xmax: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute zero_point and scale from prepared min / max."""
        scale = (xmax - xmin) / self.maxq
        if self.symmetric:
            scale = scale * 2
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = (-xmin / scale).round()
        return zero, scale

    # ------------------------------------------------------------------
    # Abstract interface (concrete in subclasses)
    # ------------------------------------------------------------------

    def update_statistics(self, data: torch.Tensor) -> None:  # pragma: no cover
        """Not implemented in base; subclasses must override."""
        raise NotImplementedError

    def compute_qscheme(self) -> ZpScaleScheme:  # pragma: no cover
        """Not implemented in base; subclasses must override."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MinMaxZpScaleCalibration
# ---------------------------------------------------------------------------


@dataclass
class MinMaxZpScaleCalibration(ZpScaleCalibration):
    """Tracks running min / max across calibration batches."""

    current_min: torch.Tensor | None = field(default=None, repr=False)
    current_max: torch.Tensor | None = field(default=None, repr=False)

    def init(self) -> None:
        """Reset running min/max state."""
        super().init()
        self.current_min = None
        self.current_max = None

    def update_statistics(self, data: torch.Tensor) -> None:
        """Accumulate the min and max of reshaped *data*."""
        assert self.calibration_data_shape is not None
        assert self.reduction_axes is not None

        calibration_data = data.reshape(self.calibration_data_shape)
        mn = torch.amin(calibration_data, dim=self.reduction_axes, keepdim=True)
        mx = torch.amax(calibration_data, dim=self.reduction_axes, keepdim=True)
        self.current_min = (
            mn if self.current_min is None else torch.minimum(self.current_min, mn)
        )
        self.current_max = (
            mx if self.current_max is None else torch.maximum(self.current_max, mx)
        )

    def compute_qscheme(self) -> ZpScaleScheme:
        """Compute and return a ZpScaleScheme from the accumulated min/max."""
        assert self.current_min is not None
        assert self.current_max is not None
        assert self.qscheme_data_shape is not None
        assert self.calibration_data_shape is not None

        xmin, xmax = self.prepare_min_max(self.current_min, self.current_max)
        zp, scale = self.compute_zp_scale_from_min_max(xmin, xmax)
        return ZpScaleScheme(
            zero_point=zp.reshape(self.qscheme_data_shape).to(torch.int32),
            scale=scale.reshape(self.qscheme_data_shape),
            nbits=self.nbits,
            calibration_data_shape=list(self.calibration_data_shape),
            qscheme_data_shape=list(self.qscheme_data_shape),
        )


# ---------------------------------------------------------------------------
# NormZpScaleCalibration  (MSE / Lp-norm grid search)
# ---------------------------------------------------------------------------


@dataclass
class NormZpScaleCalibration(ZpScaleCalibration):
    """Min-max init followed by Lp-norm grid search (a.k.a. MSE calibration).

    Adapted from examples/methods/zp_scale_norm.py.
    """

    norm_value: float = field(default=2.0)
    grid: int = field(default=100)
    maxshrink: float = field(default=0.8)

    # Stores data from a single weight sample (weights don't change batch to batch).
    _data: torch.Tensor | None = field(default=None, repr=False)

    def init(self) -> None:
        """Reset stored data."""
        super().init()
        self._data = None

    def update_statistics(self, data: torch.Tensor) -> None:
        """Store the data; supports a single unique sample (weight is fixed)."""
        if self._data is not None and (
            self._data.shape != data.shape or not torch.all(self._data == data)
        ):
            raise NotImplementedError(
                "NormZpScaleCalibration does not support multiple distinct samples"
            )
        self._data = data

    def compute_qscheme(self) -> ZpScaleScheme:
        """Run the Lp-norm grid search and return the best ZpScaleScheme."""
        assert self._data is not None
        assert self.calibration_data_shape is not None
        assert self.qscheme_data_shape is not None

        data = self._data.reshape(self.calibration_data_shape)
        dev = data.device
        rank = len(data.shape)
        assert rank % 2 == 0, f"calibration_data_shape rank must be even, got {rank}"

        dims_to_collapse = list(range(1, rank, 2))
        dims_to_keep = [data.shape[i] for i in range(0, rank, 2)]

        xmin, xmax = self.prepare_min_max(
            data.amin(dim=dims_to_collapse),
            data.amax(dim=dims_to_collapse),
        )
        zero, scale = self.compute_zp_scale_from_min_max(xmin, xmax)

        if self.norm_value > 0.0:
            best = torch.full(dims_to_keep, float("inf"), device=dev, dtype=data.dtype)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1.0 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                zero1, scale1 = self.compute_zp_scale_from_min_max(xmin1, xmax1)
                rscale1 = scale1
                rzero1 = zero1
                for d in dims_to_collapse:
                    rscale1 = rscale1.unsqueeze(d)
                    rzero1 = rzero1.unsqueeze(d)
                candidate = ZpScaleScheme(
                    zero_point=rzero1,
                    scale=rscale1,
                    nbits=self.nbits,
                    calibration_data_shape=list(data.shape),
                    qscheme_data_shape=list(rscale1.shape),
                )
                q = candidate.dequantize(candidate.quantize_as_u8(data), data.dtype)
                err = (q - data).abs_().pow_(self.norm_value).sum(dim=dims_to_collapse)
                improved = err < best
                if torch.any(improved):
                    best[improved] = err[improved]
                    scale[improved] = scale1[improved]
                    zero[improved] = zero1[improved]

        # Unsqueeze collapsed dims back to get broadcastable shape.
        rscale = scale
        rzero = zero
        for d in dims_to_collapse:
            rscale = rscale.unsqueeze(d)
            rzero = rzero.unsqueeze(d)

        return ZpScaleScheme(
            zero_point=rzero,
            scale=rscale,
            nbits=self.nbits,
            calibration_data_shape=list(data.shape),
            qscheme_data_shape=list(rscale.shape),
        )
