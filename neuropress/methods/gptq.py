"""GPTQ calibration method.

Adapted from examples/methods/gptq.py.

Reference:
    GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
    Frantar et al., ICLR 2023 – https://arxiv.org/pdf/2210.17323
"""

from __future__ import annotations

import copy
import logging
import math
import sys
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated

import torch
from torch.types import Device

from ..calibration.probe import HessianAccumulator, StatField
from ..qscheme.descriptor import GranularityType
from ..qscheme.zp_scale import ZpScaleScheme
from .base import CalibrationMethodOnActivation
from .zp_scale_calib import NormZpScaleCalibration, ZpScaleCalibration

LOGGER = logging.getLogger(__name__)

T_SUPPORTED_QSCHEMES = ZpScaleScheme


@dataclass
class GPTQCalibration(CalibrationMethodOnActivation[T_SUPPORTED_QSCHEMES]):
    """GPTQ calibration driven by an inverse-Hessian quadratic loss proxy.

    Default parameters are tuned for 4-bit per-group-32 symmetric quantization.
    """

    # Column ordering: quantize highest-Hessian-diagonal columns first.
    quant_col_in_hessian_diag_magnitude_order: bool = field(default=True)

    # If True, QSchemes are pre-computed on the original float weight;
    # if False, they are computed lazily on the error-corrected weight.
    static_groups: bool = field(default=False)

    damp_percent: float = field(default=0.01)
    damp_auto_increment: float = field(default=0.0025)

    possible_quantizers: list[ZpScaleCalibration] = field(
        default_factory=lambda: [
            NormZpScaleCalibration(
                nbits=4,
                symmetric=True,
                granularity=__import__(
                    "neuropress.qscheme.descriptor", fromlist=["Granularity"]
                ).Granularity.per_group([32]),
            )
        ]
    )

    # Hessian state fields — composed instead of inherited
    hessian: Annotated[torch.Tensor | None, StatField("hessian", "hessian")] = field(
        default=None, repr=False
    )
    nsamples: Annotated[int, StatField("hessian", "nsamples")] = field(default=0)
    device: Device | None = field(default=None)
    offload_intermediate_tensor_on_disk: bool = field(default=False)
    _hessian_accumulator: HessianAccumulator | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Post-init hook preserved for subclass compatibility."""
        pass  # preserved for subclass compatibility (GPTQMixedPrecisionCalibration)

    # ------------------------------------------------------------------
    # Accumulator lifecycle
    # ------------------------------------------------------------------

    def _ensure_accumulator(self) -> HessianAccumulator:
        """Create or sync the accumulator from self's current context + state."""
        if self._hessian_accumulator is None:
            self._hessian_accumulator = HessianAccumulator()
        acc = self._hessian_accumulator
        # Sync context so acc always sees the latest weight / module.
        for attr in (
            "weight_tensor",
            "module_class",
            "module",
            "name",
            "device",
            "offload_intermediate_tensor_on_disk",
        ):
            setattr(acc, attr, getattr(self, attr))
        # Sync accumulation state (handles the case where tests set these directly).
        acc.hessian = self.hessian
        acc.nsamples = self.nsamples
        return acc

    # ------------------------------------------------------------------
    # CalibrationMethod interface
    # ------------------------------------------------------------------

    def update_statistics(self, data: torch.Tensor) -> None:
        """Delegate to the Hessian accumulator and sync state back."""
        acc = self._ensure_accumulator()
        acc.update(data)
        self.hessian = acc.hessian
        self.nsamples = acc.nsamples

    def init(self) -> None:
        """Reset all calibration state."""
        super().init()
        self.hessian = None
        self.nsamples = 0
        self.__dict__.pop("rows", None)
        self.__dict__.pop("columns", None)
        if self._hessian_accumulator is not None:
            self._hessian_accumulator.reset()

    def compute_qscheme(self) -> ZpScaleScheme:
        """Run GPTQ, update weights in-place, and return the final QScheme."""
        quant_weights, qscheme, duration, avg_loss, damp_percent = (
            self.compute_quantization()
        )
        assert self.weight_tensor is not None
        with torch.no_grad():
            self.weight_tensor.set_(quant_weights)  # type: ignore[arg-type]
        LOGGER.info(
            "GPTQ '%s': loss=%.4f damp=%.4f time=%.1fs",
            self.name,
            avg_loss,
            damp_percent,
            duration,
        )
        self.update_qscheme_shapes_with(self.weight_tensor)
        return qscheme

    # ------------------------------------------------------------------
    # Forwarding properties (delegated to accumulator)
    # ------------------------------------------------------------------

    @cached_property
    def rows(self) -> int:
        """Number of output rows of the weight matrix."""
        return self._ensure_accumulator().rows

    @cached_property
    def columns(self) -> int:
        """Number of input columns of the weight matrix."""
        return self._ensure_accumulator().columns

    @property
    def offload_dir(self) -> Path:
        """Shared temporary directory used for tensor offloading."""
        return self._ensure_accumulator().offload_dir

    def _clone_weights(self) -> torch.Tensor:
        return self._ensure_accumulator()._clone_weights()

    # ------------------------------------------------------------------
    # Group qscheme assembly
    # ------------------------------------------------------------------

    def build_global_tensor_qscheme(
        self,
        qschemes: list[ZpScaleScheme],
        n_cols: int,
    ) -> ZpScaleScheme:
        """Concatenate per-group qschemes into one global scheme."""
        assert len(self.possible_quantizers) == 1
        assert all(isinstance(s, ZpScaleScheme) for s in qschemes)

        nbits_set = {s.nbits for s in qschemes}
        assert len(nbits_set) == 1

        col_dim = 2
        zeros = [s.zero_point for s in qschemes]
        scales = [s.scale for s in qschemes]
        qscheme_data_shape = list(qschemes[0].qscheme_data_shape)
        qscheme_data_shape[col_dim] = len(scales)
        calibration_data_shape = list(qscheme_data_shape)
        calibration_data_shape[-1] = n_cols // len(scales)

        return ZpScaleScheme(
            zero_point=torch.cat(zeros, dim=col_dim),
            scale=torch.cat(scales, dim=col_dim),
            nbits=self.possible_quantizers[0].nbits,
            calibration_data_shape=calibration_data_shape,
            qscheme_data_shape=qscheme_data_shape,
        )

    # ------------------------------------------------------------------
    # Granularity helpers
    # ------------------------------------------------------------------

    @property
    def per_channel(self) -> bool:
        """True if granularity is per-channel or per-group."""
        return self.granularity.granularity_type in (
            GranularityType.PER_CHANNEL,
            GranularityType.PER_GROUP,
        )

    @property
    def group_size(self) -> int:
        """Column-group size used by GPTQ (0 for per-tensor, cols for per-channel)."""
        gt = self.granularity.granularity_type
        if gt is GranularityType.PER_GROUP:
            assert self.granularity.group_sizes
            assert len(self.granularity.group_sizes) == 1
            return self.granularity.group_sizes[0]
        if gt is GranularityType.ALL:
            return 0
        if gt is GranularityType.PER_CHANNEL:
            assert self.weight_tensor is not None
            return self.weight_tensor.data.shape[-1]
        raise NotImplementedError(self.granularity)

    @cached_property
    def quantizer(self) -> ZpScaleCalibration:
        """Return the first (and typically only) possible quantizer."""
        if not self.possible_quantizers:
            raise ValueError("possible_quantizers must not be empty")
        assert self.possible_quantizers[0].granularity == self.granularity
        return self.possible_quantizers[0]

    # ------------------------------------------------------------------
    # Quantizer selection (hook for mixed-precision override)
    # ------------------------------------------------------------------

    def select_quantizer_for_group(self, group_idx: int) -> ZpScaleCalibration:
        """Return the quantizer for a given column-group index."""
        return self.quantizer

    # ------------------------------------------------------------------
    # GPTQ core
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _apply_damp_and_compute_inv_hessian_chol_upper(
        self, hessian: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        damp_percent = self.damp_percent
        assert 0 < damp_percent < 1
        hessian_inverse = None
        while 0 < damp_percent < 1:
            try:
                damp = damp_percent * torch.mean(torch.diag(hessian))
                diag_idx = torch.arange(self.columns, device=hessian.device)
                hessian[diag_idx, diag_idx] += damp
                hessian = torch.linalg.cholesky(hessian)
                hessian = torch.cholesky_inverse(hessian)
                hessian = torch.linalg.cholesky(hessian, upper=True)
                hessian_inverse = hessian
                break
            except torch._C._LinAlgError:  # type: ignore[attr-defined]
                if self.damp_auto_increment:
                    LOGGER.warning(
                        "GPTQ: damp_percent=%.5f too low, incrementing by %.5f",
                        damp_percent,
                        self.damp_auto_increment,
                    )
                    damp_percent += self.damp_auto_increment
                else:
                    raise
        if hessian_inverse is None:
            raise ValueError("Failed to compute inverse-Hessian Cholesky factor")
        return hessian_inverse, damp_percent

    def _get_per_group_chunk_qschemes(
        self, full_float_weights: torch.Tensor
    ) -> list[ZpScaleScheme]:
        qschemes: list[ZpScaleScheme] = []
        for group_idx, col_start in enumerate(range(0, self.columns, self.group_size)):
            q = copy.deepcopy(self.select_quantizer_for_group(group_idx))
            q.init()
            q.update(full_float_weights[:, col_start : col_start + self.group_size])
            qschemes.append(q.compute_qscheme())
        return qschemes

    def _pick_group_qscheme(
        self,
        current_qscheme: ZpScaleScheme | None,
        q_data: QProcessingData,
        col_idx: int,
    ) -> ZpScaleScheme:
        """Select or compute the qscheme for the column at *col_idx*."""
        if self.static_groups:
            raw = q_data.perm[col_idx] if q_data.perm is not None else col_idx
            return q_data.qschemes[int(raw) // self.group_size]

        # Dynamic groups
        if q_data.perm is not None:
            if not q_data.qschemes:
                q_data.qschemes = [None] * math.ceil(self.columns / self.group_size)
            g_idx = int(q_data.perm[col_idx]) // self.group_size
            if q_data.qschemes[g_idx] is not None:
                cached = q_data.qschemes[g_idx]
                assert cached is not None
                return cached
            if not hasattr(self, "_inv_perm"):
                self._inv_perm = torch.argsort(q_data.perm)
            inv_indices = self._inv_perm[
                g_idx * self.group_size : (g_idx + 1) * self.group_size
            ]
            q = copy.deepcopy(self.select_quantizer_for_group(g_idx))
            q.init()
            q.update(q_data.full_float_corr_weights[:, inv_indices])
            scheme = q.compute_qscheme()
            q_data.qschemes[g_idx] = scheme
            return scheme

        # Sequential (no perm, dynamic)
        if col_idx % self.group_size == 0:
            g_idx = col_idx // self.group_size
            q = copy.deepcopy(self.select_quantizer_for_group(g_idx))
            q.init()
            q.update(
                q_data.full_float_corr_weights[:, col_idx : col_idx + self.group_size]
            )
            scheme = q.compute_qscheme()
            q_data.qschemes.append(scheme)
            current_qscheme = scheme
        return current_qscheme  # type: ignore[return-value]

    def _quantize_per_block(
        self,
        start: int,
        end: int,
        q_data: QProcessingData,
    ) -> QBlockOutput:
        n = end - start
        fp_block = q_data.full_float_corr_weights[:, start:end].clone()
        hinv_block = q_data.inv_hessian_chol_upper[start:end, start:end]

        qdq = torch.zeros_like(fp_block)
        err = torch.zeros_like(fp_block)
        losses = torch.zeros_like(fp_block)

        apply_per_group = self.group_size != 0
        qscheme = None if apply_per_group else q_data.qschemes[0]

        for offset in range(n):
            w_col = fp_block[:, offset]
            h_diag = hinv_block[offset, offset]
            col_idx = start + offset

            if apply_per_group:
                qscheme = self._pick_group_qscheme(qscheme, q_data, col_idx)

            assert qscheme is not None
            q_col = qscheme.dequantize(
                qscheme.quantize_as_u8(w_col), fp_block.dtype
            ).flatten()
            qdq[:, offset] = q_col
            losses[:, offset] = (w_col - q_col) ** 2 / h_diag**2
            e = (w_col - q_col) / h_diag
            fp_block[:, offset:] -= e.unsqueeze(1) * hinv_block[
                offset, offset:
            ].unsqueeze(0)
            err[:, offset] = e

        corr = err.matmul(q_data.inv_hessian_chol_upper[start:end, end:])
        return QBlockOutput(qdq, losses / 2, corr)

    def _quantize_preprocess(self) -> QProcessingData:
        w = self._clone_weights()
        hessian = self.hessian
        if hessian is not None and not isinstance(hessian, torch.Tensor):
            hessian = hessian.to_base_tensor()
        assert hessian is not None
        # Free the stored hessian to reclaim memory.
        self.hessian = None
        if self._hessian_accumulator is not None:
            self._hessian_accumulator.hessian = None

        w, hessian = _fix_dead_diag(w, hessian)

        qschemes: list = []
        if self.static_groups:
            qschemes = self._get_per_group_chunk_qschemes(w)
        elif self.group_size == 0:
            q = copy.deepcopy(self.quantizer)
            q.init()
            q.update(w)
            qschemes.append(q.compute_qscheme())

        perm = None
        if self.quant_col_in_hessian_diag_magnitude_order:
            perm, w, hessian = _reorder_by_hessian_diag(w, hessian)

        inv_hinv, damp_percent = self._apply_damp_and_compute_inv_hessian_chol_upper(
            hessian
        )

        return QProcessingData(
            full_float_corr_weights=w,
            inv_hessian_chol_upper=inv_hinv,
            qschemes=qschemes,
            perm=perm,
            losses=torch.zeros_like(w),
            quant_weights=torch.zeros_like(w),
            damp_percent=damp_percent,
        )

    def _quantize_postprocess(self, q_data: QProcessingData) -> None:
        if self.quant_col_in_hessian_diag_magnitude_order and q_data.perm is not None:
            q_data.quant_weights = q_data.quant_weights[:, torch.argsort(q_data.perm)]

        assert self.weight_tensor is not None
        if _is_conv1d_transpose(self.module_class):
            q_data.quant_weights = q_data.quant_weights.t()

        if q_data.quant_weights.shape != self.weight_tensor.shape:
            q_data.quant_weights = q_data.quant_weights.reshape(
                self.weight_tensor.shape
            ).type_as(self.weight_tensor.data)
        else:
            q_data.quant_weights = q_data.quant_weights.type_as(self.weight_tensor.data)

        dev = self.device or self.weight_tensor.device
        q_data.quant_weights = q_data.quant_weights.to(device=dev)

    @torch.inference_mode()
    def compute_quantization(
        self,
    ) -> tuple[torch.Tensor, ZpScaleScheme, float, float, float]:
        """Run the full GPTQ pass and return quantized weights and diagnostics."""
        start = time.time()
        _maybe_warn_macos()

        q_data = self._quantize_preprocess()

        for block_start in range(0, self.columns, self.group_size or self.columns):
            block_end = min(
                block_start + (self.group_size or self.columns), self.columns
            )
            blk = self._quantize_per_block(block_start, block_end, q_data)
            q_data.quant_weights[:, block_start:block_end] = blk.quantized
            q_data.losses[:, block_start:block_end] = blk.losses
            q_data.full_float_corr_weights[:, block_end:] -= blk.corr

        dev = self.device or q_data.quant_weights.device
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            if str(dev).startswith("cuda"):
                torch.cuda.synchronize(dev)

        avg_loss = torch.sum(q_data.losses).item() / max(1, self.nsamples)
        if math.isnan(avg_loss):
            raise ValueError("GPTQ: NaN loss encountered")

        self._quantize_postprocess(q_data)

        n_cols = q_data.quant_weights.shape[-1]
        final_qscheme = self.build_global_tensor_qscheme(q_data.qschemes, n_cols)

        if self.offload_intermediate_tensor_on_disk:
            try:
                from torch_to_nnef.tensor.offload import OffloadedTensor

                q_data.quant_weights = OffloadedTensor.from_original_tensor(
                    q_data.quant_weights, f"{self.name}.quant_weights", self.offload_dir
                )
                final_qscheme = final_qscheme.offload_tensors(
                    self.name, self.offload_dir
                )
            except Exception:
                pass

        return (
            q_data.quant_weights,
            final_qscheme,
            time.time() - start,
            avg_loss,
            q_data.damp_percent,
        )


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------


class QProcessingData:
    """Mutable state container for per-layer GPTQ processing."""

    __slots__ = (
        "damp_percent",
        "full_float_corr_weights",
        "inv_hessian_chol_upper",
        "losses",
        "perm",
        "qschemes",
        "quant_weights",
    )

    def __init__(
        self,
        full_float_corr_weights: torch.Tensor,
        inv_hessian_chol_upper: torch.Tensor,
        qschemes: list,
        perm: torch.Tensor | None,
        losses: torch.Tensor,
        quant_weights: torch.Tensor,
        damp_percent: float,
    ) -> None:
        self.full_float_corr_weights = full_float_corr_weights
        self.inv_hessian_chol_upper = inv_hessian_chol_upper
        self.qschemes = qschemes
        self.perm = perm
        self.losses = losses
        self.quant_weights = quant_weights
        self.damp_percent = damp_percent


QBlockOutput = namedtuple("QBlockOutput", ["quantized", "losses", "corr"])


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------


def _fix_dead_diag(
    w: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dead = torch.diag(hessian) == 0
    hessian[dead, dead] = 1.0
    w[:, dead] = 0.0
    return w, hessian


def _reorder_by_hessian_diag(
    w: torch.Tensor, hessian: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = torch.argsort(torch.diag(hessian), descending=True)
    return perm, w[:, perm], hessian[perm][:, perm]


def _is_conv1d_transpose(module_class: type | None) -> bool:
    if module_class is None:
        return False
    return getattr(module_class, "__name__", "") == "Conv1D"


def _maybe_warn_macos() -> None:
    import os

    if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        LOGGER.debug(
            "macOS: set PYTORCH_ENABLE_MPS_FALLBACK=1 if Cholesky fails on MPS"
        )
