"""Tests for GPTQMixedPrecisionCalibration, MixedAllocationStrategy,
GroupQuantizer, and related helpers."""

import pytest
import torch
from torch import nn

from neuropress.methods.gptq_panel import (
    GPTQMixedPrecisionCalibration,
    GroupQuantizer,
    MixedAllocationStrategy,
    _score_group_candidates_hinv,
)
from neuropress.methods.zp_scale_calib import NormZpScaleCalibration
from neuropress.qscheme import Granularity
from neuropress.qscheme.zp_scale import ZpScaleScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gran(g):
    return Granularity.per_group([g])


def _norm_q(nbits, g=32):
    return NormZpScaleCalibration(
        nbits=nbits, symmetric=True, granularity=_gran(g), grid=5
    )


def _make_linear(in_f=64, out_f=16):
    return nn.Linear(in_f, out_f, bias=False)


def _calibrate(calib, module, n_batches=4, batch_size=8):
    calib.weight_tensor = module.weight
    calib.module_class = type(module)
    calib.module = module
    calib.name = "test"
    calib.hessian = None
    calib.nsamples = 0
    calib.init()
    for _ in range(n_batches):
        calib.update_statistics(torch.randn(batch_size, module.weight.shape[1]))
    return calib


def _make_strategy(rows=8, cols=64, group_size=32, target_bpw=3.0):
    w = torch.randn(rows, cols)
    inv_chol = torch.eye(cols)
    quantizers = [_norm_q(2, group_size), _norm_q(4, group_size)]
    return MixedAllocationStrategy(
        weight_matrix=w,
        inv_hessian_cholesky_upper_full=inv_chol,
        group_size=group_size,
        quantizers=quantizers,
        target_bpw=target_bpw,
    )


def _make_calib(target_bpw=3.0, group_size=32):
    gran = _gran(group_size)
    return GPTQMixedPrecisionCalibration(
        granularity=gran,
        quant_col_in_hessian_diag_magnitude_order=False,
        static_groups=False,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        possible_quantizers=[_norm_q(2, group_size), _norm_q(4, group_size)],
        target_bpw=target_bpw,
    )


# ---------------------------------------------------------------------------
# _score_group_candidates_hinv
# ---------------------------------------------------------------------------


def test_score_group_candidates_returns_correct_length():
    rows, cols = 8, 32
    w = torch.randn(rows, cols)
    inv_chol = torch.eye(cols)
    candidates = [_norm_q(2), _norm_q(4)]
    scores = _score_group_candidates_hinv(w, inv_chol, candidates)
    assert scores.shape == (2,)


def test_score_group_candidates_higher_bits_lower_score():
    """4-bit should have a lower (better) proxy loss than 2-bit."""
    torch.manual_seed(42)
    rows, cols = 8, 32
    w = torch.randn(rows, cols)
    inv_chol = torch.eye(cols)
    candidates = [_norm_q(2), _norm_q(4)]
    scores = _score_group_candidates_hinv(w, inv_chol, candidates, normalize=False)
    # 4-bit (index 1) should score better (lower) than 2-bit (index 0)
    assert scores[1].item() <= scores[0].item()


def test_score_group_candidates_normalise_reduces_scale():
    torch.manual_seed(0)
    w = torch.randn(4, 32)
    inv_chol = torch.eye(32) * 10.0
    candidates = [_norm_q(4)]
    s_unnorm = _score_group_candidates_hinv(w, inv_chol, candidates, normalize=False)
    s_norm = _score_group_candidates_hinv(w, inv_chol, candidates, normalize=True)
    assert s_norm[0].item() < s_unnorm[0].item()


# ---------------------------------------------------------------------------
# MixedAllocationStrategy
# ---------------------------------------------------------------------------


def test_mixed_allocation_allocate_returns_correct_count():
    strat = _make_strategy(cols=64, group_size=32)
    assignments = strat.allocate()
    assert len(assignments) == 2  # 64 / 32 = 2 groups


def test_mixed_allocation_allocate_all_have_selection():
    strat = _make_strategy()
    for assignment in strat.allocate():
        assert assignment.selected is not None


def test_mixed_allocation_average_bpw_respects_target():
    target = 3.0
    strat = _make_strategy(target_bpw=target)
    assignments = strat.allocate()
    # All assignments should use valid quantizers
    for a in assignments:
        assert a.selected is not None


def test_mixed_allocation_initial_selection_lowest_bpw():
    scores = [torch.tensor([0.5, 0.3]), torch.tensor([0.2, 0.1])]
    bpw = [2.5, 4.5]
    selected = MixedAllocationStrategy.initial_selection_lowest_bpw(scores, bpw)
    # Lowest bpw (index 0) for both groups
    assert all(s == 0 for s in selected)


def test_mixed_allocation_average_bpw_static():
    selected = [0, 1, 0]
    bpw = [2.0, 4.0]
    avg = MixedAllocationStrategy.average_bpw(selected, bpw)
    assert abs(avg - (2 + 4 + 2) / 3) < 1e-6


def test_mixed_allocation_ranked_scores_sorted():
    strat = _make_strategy()
    assignments = strat.allocate()
    for a in assignments:
        scores = [qs.score for qs in a.ranked_scores]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# GroupQuantizer
# ---------------------------------------------------------------------------


def test_group_quantizer_from_base_construction():
    base = _norm_q(4, g=32)
    gq = GroupQuantizer.from_base(base, target_group_size=64)
    assert gq._sub_group_size == 32


def test_group_quantizer_invalid_target_raises():
    base = _norm_q(4, g=32)
    with pytest.raises(ValueError):
        GroupQuantizer.from_base(base, target_group_size=50)  # 50 % 32 != 0


def test_group_quantizer_compute_qscheme_returns_zp_scale():
    base = _norm_q(4, g=32)
    gq = GroupQuantizer.from_base(base, target_group_size=64)
    gq.init()
    w = torch.randn(8, 64)
    gq.update(w)
    scheme = gq.compute_qscheme()
    assert isinstance(scheme, ZpScaleScheme)


def test_group_quantizer_scheme_covers_full_target_group():
    base = _norm_q(4, g=32)
    gq = GroupQuantizer.from_base(base, target_group_size=64)
    gq.init()
    w = torch.randn(8, 64)
    gq.update(w)
    scheme = gq.compute_qscheme()
    # Should have 2 sub-groups along col_dim=2
    assert scheme.qscheme_data_shape[2] == 2


# ---------------------------------------------------------------------------
# GPTQMixedPrecisionCalibration
# ---------------------------------------------------------------------------


def test_gptq_mp_requires_per_group():
    with pytest.raises(ValueError, match="PER_GROUP"):
        GPTQMixedPrecisionCalibration(
            granularity=Granularity.all_tensor(),
            possible_quantizers=[_norm_q(4), _norm_q(2)],
            target_bpw=3.0,
        )


def test_gptq_mp_requires_two_quantizers():
    with pytest.raises(ValueError):
        GPTQMixedPrecisionCalibration(
            granularity=_gran(32),
            possible_quantizers=[_norm_q(4)],
            target_bpw=3.0,
        )


def test_gptq_mp_compute_quantization_returns_scheme():
    module = _make_linear(in_f=64, out_f=16)
    calib = _make_calib()
    _calibrate(calib, module)
    weights, _qscheme, duration, avg_loss, _damp = calib.compute_quantization()
    assert weights.shape == module.weight.shape
    assert duration > 0
    assert avg_loss >= 0


def test_gptq_mp_assignments_populated_after_compute():
    module = _make_linear(in_f=64, out_f=16)
    calib = _make_calib()
    _calibrate(calib, module)
    calib.compute_quantization()
    assert calib._mp_assignments is not None
    # 64 in_features / 32 group_size = 2 groups
    assert len(calib._mp_assignments) == 2


def test_gptq_mp_select_quantizer_for_group():
    module = _make_linear(in_f=64, out_f=16)
    calib = _make_calib()
    _calibrate(calib, module)
    calib.compute_quantization()
    q = calib.select_quantizer_for_group(0)
    assert q is not None


def test_gptq_mp_static_groups_variant():
    module = _make_linear(in_f=64, out_f=16)
    gran = _gran(32)
    calib = GPTQMixedPrecisionCalibration(
        granularity=gran,
        quant_col_in_hessian_diag_magnitude_order=False,
        static_groups=True,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        possible_quantizers=[_norm_q(2, 32), _norm_q(4, 32)],
        target_bpw=3.0,
    )
    _calibrate(calib, module)
    weights, _qscheme, *_ = calib.compute_quantization()
    assert weights.shape == module.weight.shape
