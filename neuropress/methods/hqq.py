"""HQQ (Half-Quadratic Quantization) calibration method.

Minimises the quantization error ``||W - dequant(quant(W, s, z), s, z)||``
over scale *s* and zero-point *z* using a half-quadratic proximal splitting
scheme.  No input activations are required — the method is purely
weight-driven.

Reference: Badri & Shaji, "Half-Quadratic Quantization of Large Machine
Learning Models", 2023 (arXiv:2309.15531).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..qscheme.zp_scale import ZpScaleScheme
from .zp_scale_calib import ZpScaleCalibration


def _proximal_shrink(r: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
    """Apply the half-quadratic proximal shrinkage operator to *r*.

    Approximates ``argmin_d (1/2)||d - r||² + (1/beta)|d|^lp_norm``:

    * ``lp_norm == 2.0``: ridge — ``r / (1 + 2/beta)``.
    * ``lp_norm == 1.0``: soft threshold — ``sign(r) * max(|r| - 1/beta, 0)``.
    * otherwise: data-dependent threshold
      ``sign(r) * max(|r| - (lp_norm/beta)|r|^(lp_norm-1), 0)``.

    Args:
        r: Residual tensor (fractional rounding error per element).
        beta: Penalty strength; larger values → smaller shrinkage.
        lp_norm: Lp exponent (e.g. ``0.5``, ``1.0``, ``2.0``).

    Returns:
        Shrunk residual tensor with the same shape as *r*.
    """
    if lp_norm == 2.0:
        return r / (1.0 + 2.0 / beta)
    if lp_norm == 1.0:
        return r.sign() * (r.abs() - 1.0 / beta).clamp(min=0.0)
    # General Lp (p != 1, 2): data-dependent threshold.
    eps = 1e-8
    threshold = (lp_norm / beta) * r.abs().clamp(min=eps).pow(lp_norm - 1.0)
    return r.sign() * (r.abs() - threshold).clamp(min=0.0)


def _hqq_optimize(
    W: torch.Tensor,
    *,
    nbits: int,
    symmetric: bool,
    reduction_axes: list[int],
    num_iter: int,
    beta: float,
    kappa: float,
    lp_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimise per-group quantization error via half-quadratic proximal splitting.

    Initialises scale and zero-point from a min-max estimate (with
    ``force_include_zero`` semantics), then refines them iteratively.

    Each iteration:

    1. Hard-quantize *W* with the current parameters.
    2. Apply :func:`_proximal_shrink` to the fractional rounding residual,
       yielding a soft assignment ``q_float``.
    3. Update scale (and zero-point for asymmetric) via ordinary least squares
       given ``q_float``.
    4. Increase ``beta`` by *kappa*.

    Args:
        W: Weight tensor already reshaped to ``calibration_data_shape``
            (float32 recommended).
        nbits: Quantization bit-width.
        symmetric: When ``True``, zero-point is fixed at ``(maxq + 1) / 2``
            and only scale is updated.
        reduction_axes: Axes to reduce over (e.g. ``[1, 3]`` for 4-D data).
        num_iter: Number of half-quadratic iterations.  ``0`` returns the
            pure min-max initialisation.
        beta: Initial penalty strength for half-quadratic splitting.
        kappa: Multiplicative schedule for *beta* (``beta *= kappa`` per
            iteration).
        lp_norm: Lp exponent passed to :func:`_proximal_shrink`.

    Returns:
        Tuple ``(scale, zero_point)`` — both tensors have the same rank as
        *W* with ``keepdim=True`` applied along *reduction_axes*
        (i.e. shape == ``qscheme_data_shape``).
    """
    maxq = 2**nbits - 1
    eps = 1e-8

    # --- Min-max initialisation (force_include_zero semantics) ---
    w_min = W.amin(dim=reduction_axes, keepdim=True).clamp_max(0.0)
    w_max = W.amax(dim=reduction_axes, keepdim=True).clamp_min(0.0)

    # Guard against zero-range groups.
    zero_range = (w_min == 0) & (w_max == 0)
    w_min = w_min.masked_fill(zero_range, -1.0)
    w_max = w_max.masked_fill(zero_range, 1.0)

    if symmetric:
        w_max = torch.maximum(w_min.abs(), w_max)
        w_min = torch.where(w_min < 0, -w_max, w_min)
        scale = (w_max - w_min) / maxq * 2
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        scale = (w_max - w_min) / maxq
        zero = (-w_min / scale).round().clamp(0.0, float(maxq))

    if num_iter == 0:
        return scale, zero

    # --- Iterative half-quadratic refinement ---
    for _ in range(num_iter):
        q_hard = (W / scale + zero).clamp(0.0, float(maxq)).round()
        r = W / scale + zero - q_hard
        delta = _proximal_shrink(r, beta, lp_norm)
        q_float = q_hard + delta

        if symmetric:
            # Zero-point fixed; update scale only via closed-form regression:
            # W = (q_float - z) * s  =>  s = <W, u> / <u, u>,  u = q_float - z
            u = q_float - zero
            num = (W * u).sum(dim=reduction_axes, keepdim=True)
            den = (u * u).sum(dim=reduction_axes, keepdim=True).clamp(min=eps)
            scale = (num / den).abs().clamp(min=eps)
        else:
            # Joint OLS per group: W ≈ s * (q_float - z).
            # Demeaned covariance gives the slope; then solve for intercept.
            q_mean = q_float.mean(dim=reduction_axes, keepdim=True)
            w_mean = W.mean(dim=reduction_axes, keepdim=True)
            dq = q_float - q_mean
            dw = W - w_mean
            q_var = (dq * dq).sum(dim=reduction_axes, keepdim=True).clamp(min=eps)
            qw_cov = (dq * dw).sum(dim=reduction_axes, keepdim=True)
            scale = (qw_cov / q_var).abs().clamp(min=eps)
            zero = (q_mean - w_mean / scale).clamp(0.0, float(maxq))

        beta *= kappa

    return scale, zero


@dataclass
class HQQCalibration(ZpScaleCalibration):
    """HQQ: half-quadratic quantization error minimization.

    Optimises per-group scale and zero-point by minimising the L2
    reconstruction error ``||W - dequant(quant(W, s, z), s, z)||`` through
    iterative half-quadratic proximal updates.  No input activations or
    calibration batches are needed — the method observes only the weight
    tensor itself.

    Attributes:
        num_iter: Number of half-quadratic optimization iterations.
            Set to ``0`` to fall back to a plain min-max calibration.
        beta: Initial penalty strength for the proximal splitting.
            Larger values let more of the rounding residual pass through.
        kappa: Multiplicative schedule for *beta* (``beta *= kappa`` per
            iteration).  Values slightly above 1.0 anneal the penalty.
        lp_norm: Lp exponent for the proximal operator.  ``0.5`` (default)
            uses the half-quadratic splitting from the original paper; ``1.0``
            gives LASSO-style soft thresholding; ``2.0`` gives ridge.
    """

    num_iter: int = field(default=20)
    beta: float = field(default=1e4)
    kappa: float = field(default=1.01)
    lp_norm: float = field(default=0.5)

    _data: torch.Tensor | None = field(default=None, repr=False)

    def init(self) -> None:
        """Reset stored weight data."""
        super().init()
        self._data = None

    def update_statistics(self, data: torch.Tensor) -> None:
        """Store *data* for later optimization in :meth:`compute_qscheme`.

        Because the weight tensor is constant across batches, only one
        unique sample is supported.  Repeated calls with the identical
        tensor are silently accepted.

        Args:
            data: Weight data tensor (any shape; reshaped internally).

        Raises:
            NotImplementedError: If called with a second, distinct sample.
        """
        if self._data is not None and (
            self._data.shape != data.shape or not torch.all(self._data == data)
        ):
            raise NotImplementedError(
                "HQQCalibration does not support multiple distinct samples; "
                "weight tensors are constant and should be passed once."
            )
        self._data = data

    def compute_qscheme(self) -> ZpScaleScheme:
        """Run HQQ optimisation and return the best ZpScaleScheme.

        Returns:
            :class:`~neuropress.qscheme.zp_scale.ZpScaleScheme` with
            optimised scale and zero-point tensors of shape
            ``qscheme_data_shape``.
        """
        assert self._data is not None, "call update() before compute_qscheme()"
        assert self.calibration_data_shape is not None
        assert self.qscheme_data_shape is not None
        assert self.reduction_axes is not None

        data = self._data.reshape(self.calibration_data_shape).to(torch.float32)
        scale, zero = _hqq_optimize(
            data,
            nbits=self.nbits,
            symmetric=self.symmetric,
            reduction_axes=self.reduction_axes,
            num_iter=self.num_iter,
            beta=self.beta,
            kappa=self.kappa,
            lp_norm=self.lp_norm,
        )
        return ZpScaleScheme(
            zero_point=zero.reshape(self.qscheme_data_shape).to(torch.int32),
            scale=scale.reshape(self.qscheme_data_shape),
            nbits=self.nbits,
            calibration_data_shape=list(self.calibration_data_shape),
            qscheme_data_shape=list(self.qscheme_data_shape),
        )
