"""Panel Quantization for GPTQ – mixed-precision per-column-group allocation.

Adapted from examples/methods/gptq_mixed.py.

Key components:
- ``MixedAllocationStrategy``: scores candidates per group, initialises with
  lowest-bpw, then greedily upgrades until the average bpw meets the target.
- ``GPTQMixedPrecisionCalibration``: extends GPTQCalibration with per-group
  quantizer selection driven by ``MixedAllocationStrategy``.
- ``GroupQuantizer``: adapter that tiles a sub-group quantizer to fill a
  larger target group (e.g. a 32-group quantizer applied twice for a 64-block).
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from ..qscheme.descriptor import Granularity, GranularityType
from ..qscheme.zp_scale import ZpScaleScheme
from .gptq import GPTQCalibration, QProcessingData
from .zp_scale_calib import ZpScaleCalibration

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizerScore:
    """Score assigned to a candidate quantizer for a specific panel."""

    quantizer: ZpScaleCalibration
    score: float  # normalised GPTQ quadratic-proxy loss

    def __str__(self) -> str:
        """Return a compact string representation."""
        return f"{self.quantizer!r}[score={self.score:.2e}]"


@dataclass
class PanelAssignment:
    """Records the selected quantizer for one column-group (panel)."""

    panel_index: int
    selected: ZpScaleCalibration | None
    ranked_scores: list[QuantizerScore]  # sorted best → worst

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        sel = repr(self.selected) if self.selected is not None else "None"
        ranked = ", ".join(str(qs) for qs in self.ranked_scores)
        return (
            f"PanelAssignment(panel={self.panel_index},"
            f" selected={sel}, ranked=[{ranked}])"
        )


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def _score_group_candidates_hinv(
    weight_block: torch.Tensor,
    inv_hessian_cholesky_upper: torch.Tensor,
    quantizers: Sequence[ZpScaleCalibration],
    normalize: bool = True,
) -> torch.Tensor:
    """GPTQ quadratic-proxy loss per candidate quantizer on a single block.

    score(E) = ||E · R||_F²  where R is the lower-triangular Cholesky of
    inv(H) (from the upper-triangular factor).
    """
    chol_lower = inv_hessian_cholesky_upper.t()
    scores = torch.zeros(
        len(quantizers), device=weight_block.device, dtype=weight_block.dtype
    )

    for i, quantizer in enumerate(quantizers):
        candidate = copy.deepcopy(quantizer)
        candidate.init()
        candidate.update(weight_block)
        qscheme = candidate.compute_qscheme()
        dq = qscheme.dequantize(
            qscheme.quantize_as_u8(weight_block), weight_block.dtype
        )
        err = weight_block - dq
        scores[i] = torch.sum((err @ chol_lower) ** 2)

    if normalize:
        trace = torch.sum(torch.diag(inv_hessian_cholesky_upper) ** 2)
        scores = scores / (trace + 1e-8)

    return scores


# ---------------------------------------------------------------------------
# MixedAllocationStrategy
# ---------------------------------------------------------------------------


class MixedAllocationStrategy:
    """Allocate per-group quantizers for mixed precision under a bit budget.

    Algorithm:
    1. Score all candidates on all groups using the GPTQ Hessian proxy.
    2. Initialise each group with the lowest-bpw candidate (tie-break by score).
    3. Compute "benefit-per-added-bit" for every possible upgrade.
    4. Greedily apply upgrades in order of benefit until the average bpw
       meets *target_bpw*.
    """

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        inv_hessian_cholesky_upper_full: torch.Tensor,
        group_size: int,
        quantizers: Sequence[ZpScaleCalibration],
        target_bpw: float,
    ) -> None:
        self.weight_matrix = weight_matrix
        self.inv_hessian_cholesky_upper_full = inv_hessian_cholesky_upper_full
        self.group_size = group_size
        self.quantizers = list(quantizers)
        self.target_bpw = float(target_bpw)

    # ------------------------------------------------------------------
    # Core steps
    # ------------------------------------------------------------------

    def score_all_groups(
        self, candidate_quantizers: Sequence[ZpScaleCalibration]
    ) -> list[torch.Tensor]:
        """Return a list of score vectors (one per group)."""
        num_cols = self.weight_matrix.shape[1]
        group_scores: list[torch.Tensor] = []
        for col_start in range(0, num_cols, self.group_size):
            col_end = min(col_start + self.group_size, num_cols)
            block = self.weight_matrix[:, col_start:col_end]
            inv_chol = self.inv_hessian_cholesky_upper_full[
                col_start:col_end, col_start:col_end
            ]
            scores = _score_group_candidates_hinv(block, inv_chol, candidate_quantizers)
            group_scores.append(scores)
        return group_scores

    @staticmethod
    def initial_selection_lowest_bpw(
        group_scores: list[torch.Tensor],
        candidate_bpw: list[float],
    ) -> list[int]:
        """Pick lowest-bpw candidate per group; tie-break by best (lowest) score."""
        selected: list[int] = []
        n = len(candidate_bpw)
        for scores in group_scores:

            def _key(i: int, s: torch.Tensor = scores) -> tuple[float, float]:
                return (candidate_bpw[i], s[i].item())

            ids = sorted(range(n), key=_key)
            selected.append(ids[0])
        return selected

    @staticmethod
    def average_bpw(selected: list[int], candidate_bpw: list[float]) -> float:
        """Return average bits-per-weight for the current selection."""
        return sum(candidate_bpw[s] for s in selected) / max(1, len(selected))

    @staticmethod
    def compute_upgrades(
        group_scores: list[torch.Tensor],
        candidate_bpw: list[float],
        selected: list[int],
    ) -> list:
        """Return upgrades sorted by benefit-per-bit (descending).

        Each entry: (benefit_per_bit, group_idx, new_candidate_idx, delta_bpw)
        """
        upgrades = []
        n = len(candidate_bpw)
        for g_idx, scores in enumerate(group_scores):
            cur = selected[g_idx]
            for c_idx in range(n):
                if (
                    candidate_bpw[c_idx] > candidate_bpw[cur]
                    and scores[c_idx].item() < scores[cur].item()
                ):
                    delta = candidate_bpw[c_idx] - candidate_bpw[cur]
                    benefit = (scores[cur].item() - scores[c_idx].item()) / delta
                    upgrades.append((benefit, g_idx, c_idx, delta))
        upgrades.sort(key=lambda x: x[0], reverse=True)
        return upgrades

    def apply_upgrades_greedily(
        self,
        upgrades: list,
        selected: list[int],
        candidate_bpw: list[float],
    ) -> list[int]:
        """Apply upgrades from *upgrades* until the target bpw is met."""
        current_avg = self.average_bpw(selected, candidate_bpw)
        n_groups = len(selected)
        for _, g_idx, c_idx, delta in upgrades:
            if current_avg + delta / n_groups <= self.target_bpw + 1e-8:
                selected[g_idx] = c_idx
                current_avg += delta / n_groups
            if current_avg >= self.target_bpw - 1e-8:
                break
        return selected

    @staticmethod
    def build_assignments(
        group_scores: list[torch.Tensor],
        selected: list[int],
        candidate_quantizers: Sequence[ZpScaleCalibration],
    ) -> list[PanelAssignment]:
        """Build a PanelAssignment list from final selections and scores."""
        assignments: list[PanelAssignment] = []
        n = len(candidate_quantizers)
        for g_idx, scores in enumerate(group_scores):
            ranking = sorted(range(n), key=lambda i, s=scores: s[i].item())
            ranked = [
                QuantizerScore(
                    quantizer=candidate_quantizers[i], score=scores[i].item()
                )
                for i in ranking
            ]
            assignments.append(
                PanelAssignment(
                    panel_index=g_idx,
                    selected=candidate_quantizers[selected[g_idx]],
                    ranked_scores=ranked,
                )
            )
        return assignments

    def allocate(self) -> list[PanelAssignment]:
        """Execute the full allocation and return per-group assignments."""
        candidate_bpw = [q.bpw() for q in self.quantizers]
        group_scores = self.score_all_groups(self.quantizers)
        selected = self.initial_selection_lowest_bpw(group_scores, candidate_bpw)
        upgrades = self.compute_upgrades(group_scores, candidate_bpw, selected)
        selected = self.apply_upgrades_greedily(upgrades, selected, candidate_bpw)
        return self.build_assignments(group_scores, selected, self.quantizers)


# ---------------------------------------------------------------------------
# GroupQuantizer
# ---------------------------------------------------------------------------


@dataclass
class GroupQuantizer(ZpScaleCalibration):
    """Tiles a sub-group quantizer to cover a larger target group.

    Example: target_group_size=64 with a 32-group-size quantizer → applies
    the quantizer twice over [0:32] and [32:64].
    """

    # Externally set after construction.
    _base_quantizer: ZpScaleCalibration | None = field(default=None, repr=False)
    _sub_group_size: int = field(default=0, repr=False)
    _sub_quantizers: list[ZpScaleCalibration] = field(default_factory=list, repr=False)

    @classmethod
    def from_base(
        cls,
        base_quantizer: ZpScaleCalibration,
        target_group_size: int,
    ) -> GroupQuantizer:
        """Construct a GroupQuantizer tiling *base_quantizer* to *target_group_size*."""
        assert base_quantizer.granularity.group_sizes is not None
        sub_g = base_quantizer.granularity.group_sizes[0]
        if target_group_size % sub_g != 0:
            raise ValueError(
                f"base group_size={sub_g} must divide"
                f" target_group_size={target_group_size}"
            )
        obj = cls(
            granularity=Granularity.per_group([target_group_size]),
            nbits=base_quantizer.nbits,
            symmetric=base_quantizer.symmetric,
        )
        obj._base_quantizer = copy.deepcopy(base_quantizer)
        obj._sub_group_size = sub_g
        obj._sub_quantizers = []
        return obj

    def init(self) -> None:
        """Reset accumulated sub-quantizers."""
        self._sub_quantizers = []

    def update_statistics(self, data: torch.Tensor) -> None:
        """No-op; statistics are handled in ``update()``."""
        pass  # statistics handled in update()

    def update(self, data: torch.Tensor) -> None:
        """Tile the base quantizer across sub-groups of *data*."""
        assert self._base_quantizer is not None
        num_cols = data.shape[-1]
        assert num_cols % self._sub_group_size == 0
        n_sub = num_cols // self._sub_group_size
        self._sub_quantizers = []
        for i in range(n_sub):
            sub = copy.deepcopy(self._base_quantizer)
            sub.init()
            sub.update(
                data[..., i * self._sub_group_size : (i + 1) * self._sub_group_size]
            )
            self._sub_quantizers.append(sub)

    def compute_qscheme(self) -> ZpScaleScheme:
        """Concatenate sub-quantizer schemes into one combined ZpScaleScheme."""
        col_dim = 2
        sub_schemes = [s.compute_qscheme() for s in self._sub_quantizers]
        first = sub_schemes[0]
        zp = torch.cat([s.zero_point for s in sub_schemes], dim=col_dim)
        scale = torch.cat([s.scale for s in sub_schemes], dim=col_dim)
        qds = list(first.qscheme_data_shape)
        total_blocks = sum(s.qscheme_data_shape[col_dim] for s in sub_schemes)
        qds[col_dim] = total_blocks
        return ZpScaleScheme(
            zero_point=zp,
            scale=scale,
            nbits=first.nbits,
            calibration_data_shape=list(first.calibration_data_shape),
            qscheme_data_shape=qds,
        )


# ---------------------------------------------------------------------------
# GPTQMixedPrecisionCalibration
# ---------------------------------------------------------------------------


@dataclass
class GPTQMixedPrecisionCalibration(GPTQCalibration):
    """GPTQ with mixed-precision per-column-group selection.

    Uses ``MixedAllocationStrategy`` to assign one quantizer per column-group
    from ``possible_quantizers`` under a ``target_bpw`` bit budget.
    """

    target_bpw: float = field(default=4.0)
    # Scoring panel width = panel_group_factor * group_size.
    panel_group_factor: int = field(default=1)

    # Internal state (set during _quantize_preprocess)
    _mp_assignments: list[PanelAssignment] | None = field(default=None, repr=False)
    _mp_quantizers: list[ZpScaleCalibration] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.granularity.granularity_type is not GranularityType.PER_GROUP:
            raise ValueError(
                "GPTQMixedPrecisionCalibration requires PER_GROUP granularity."
            )
        if len(self.possible_quantizers) < 2:
            raise ValueError(
                "GPTQMixedPrecisionCalibration requires at least two"
                " possible_quantizers."
            )
        if not isinstance(self.panel_group_factor, int) or self.panel_group_factor < 1:
            raise ValueError("panel_group_factor must be an integer >= 1.")
        for q in self.possible_quantizers:
            if not hasattr(q, "bpw"):
                raise ValueError("All possible_quantizers must implement bpw().")
        assert self.granularity.group_sizes is not None
        base_g = self.granularity.group_sizes[0]
        for q in self.possible_quantizers:
            if q.granularity.granularity_type is not GranularityType.PER_GROUP:
                raise ValueError(
                    "All possible_quantizers must use PER_GROUP granularity."
                )
            assert q.granularity.group_sizes is not None
            cand_g = q.granularity.group_sizes[0]
            if base_g % cand_g != 0:
                raise ValueError(
                    f"Candidate group_size={cand_g} must divide"
                    f" GPTQ group_size={base_g}."
                )

    # ------------------------------------------------------------------
    # Override preprocessing to inject panel allocation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _quantize_preprocess(self) -> QProcessingData:
        # Run base preprocessing without static qschemes (we'll do them below).
        saved_static = self.static_groups
        self.static_groups = False
        base_data = super()._quantize_preprocess()
        self.static_groups = saved_static

        if self.group_size <= 0:
            raise ValueError("Mixed precision requires group_size > 0.")

        mp_quantizers = [copy.deepcopy(q) for q in self.possible_quantizers]

        assignments = MixedAllocationStrategy(
            weight_matrix=base_data.full_float_corr_weights,
            inv_hessian_cholesky_upper_full=base_data.inv_hessian_chol_upper,
            group_size=self.group_size,
            quantizers=mp_quantizers,
            target_bpw=self.target_bpw,
        ).allocate()

        self._mp_quantizers = mp_quantizers
        self._mp_assignments = assignments

        LOGGER.info(
            "GPTQ-panel '%s': allocated %d groups, target_bpw=%.2f",
            self.name,
            len(assignments),
            self.target_bpw,
        )

        qschemes: list = []
        if self.static_groups:
            qschemes = self._get_per_group_chunk_qschemes(
                base_data.full_float_corr_weights
            )

        return QProcessingData(
            full_float_corr_weights=base_data.full_float_corr_weights,
            inv_hessian_chol_upper=base_data.inv_hessian_chol_upper,
            qschemes=qschemes,
            perm=base_data.perm,
            losses=base_data.losses,
            quant_weights=base_data.quant_weights,
            damp_percent=base_data.damp_percent,
        )

    # ------------------------------------------------------------------
    # Override quantizer selection
    # ------------------------------------------------------------------

    def select_quantizer_for_group(self, group_idx: int) -> ZpScaleCalibration:
        """Return the quantizer assigned to *group_idx*, wrapping if needed."""
        assert self._mp_assignments is not None
        assignment = self._mp_assignments[group_idx]
        assert assignment.selected is not None
        selected = assignment.selected

        # If the selected quantizer's group_size differs from the GPTQ group_size,
        # wrap it in a GroupQuantizer that tiles it to fill the block.
        assert self.granularity.group_sizes is not None
        base_g = self.granularity.group_sizes[0]
        assert selected.granularity.group_sizes is not None
        sel_g = selected.granularity.group_sizes[0]
        if sel_g != base_g:
            return GroupQuantizer.from_base(selected, target_group_size=base_g)
        return selected

    # ------------------------------------------------------------------
    # Override global qscheme builder to allow heterogeneous nbits
    # ------------------------------------------------------------------

    def build_global_tensor_qscheme(
        self,
        qschemes: list,
        n_cols: int,
    ):
        """Assemble global qscheme; may be heterogeneous across groups."""
        if all(isinstance(s, ZpScaleScheme) for s in qschemes):
            col_dim = 2
            # If all share the same nbits, use parent implementation.
            nbits_set = {s.nbits for s in qschemes}
            if len(nbits_set) == 1:
                zeros = [s.zero_point for s in qschemes]
                scales = [s.scale for s in qschemes]
                qds = list(qschemes[0].qscheme_data_shape)
                qds[col_dim] = len(scales)
                cds = list(qds)
                cds[-1] = n_cols // len(scales)
                return ZpScaleScheme(
                    zero_point=torch.cat(zeros, dim=col_dim),
                    scale=torch.cat(scales, dim=col_dim),
                    nbits=next(iter(nbits_set)),
                    calibration_data_shape=cds,
                    qscheme_data_shape=qds,
                )
        # Heterogeneous: return the first scheme as a representative.
        # A full implementation would wrap these in a QSchemeGroup.
        return qschemes[0]
