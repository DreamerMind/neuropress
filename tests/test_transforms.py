"""Tests for split_mixed_precision_linear and MixedPrecisionLinear."""

import pytest
import torch
from torch import nn

from neuropress.methods.gptq_panel import (
    GPTQMixedPrecisionCalibration,
    PanelAssignment,
    QuantizerScore,
)
from neuropress.methods.zp_scale_calib import NormZpScaleCalibration
from neuropress.qscheme import Granularity
from neuropress.transforms import (
    MixedPrecisionLinear,
    apply_split_to_model,
    split_mixed_precision_linear,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gran(g):
    return Granularity.per_group([g])


def _norm_q(nbits, g=32):
    return NormZpScaleCalibration(
        nbits=nbits, symmetric=True, granularity=_gran(g), grid=5
    )


def _fake_assignments(
    panel_nbits: list[int], group_size: int = 32
) -> list[PanelAssignment]:
    """Build minimal PanelAssignment list from a list of nbits per panel."""
    result = []
    for i, nbits in enumerate(panel_nbits):
        q = _norm_q(nbits, group_size)
        result.append(
            PanelAssignment(
                panel_index=i,
                selected=q,
                ranked_scores=[QuantizerScore(quantizer=q, score=0.0)],
            )
        )
    return result


def _linear(in_f, out_f, bias=False):
    return nn.Linear(in_f, out_f, bias=bias)


def _calibrate_full(in_features=64, out_features=16, group_size=32):
    """Run GPTQMixedPrecisionCalibration end-to-end and return (module, calib)."""
    module = _linear(in_features, out_features)
    gran = _gran(group_size)
    calib = GPTQMixedPrecisionCalibration(
        granularity=gran,
        quant_col_in_hessian_diag_magnitude_order=False,
        static_groups=False,
        damp_percent=0.01,
        damp_auto_increment=0.01,
        possible_quantizers=[_norm_q(2, group_size), _norm_q(4, group_size)],
        target_bpw=3.0,
    )
    calib.weight_tensor = module.weight
    calib.module_class = type(module)
    calib.module = module
    calib.name = "test"
    calib.hessian = None
    calib.nsamples = 0
    calib.init()
    for _ in range(4):
        calib.update_statistics(torch.randn(8, in_features))
    calib.compute_quantization()
    return module, calib


# ---------------------------------------------------------------------------
# MixedPrecisionLinear - unit tests
# ---------------------------------------------------------------------------


def _make_trivial(in_f=64, out_f=8):
    """Two equal-size precision groups, no bias."""
    W4_lin = nn.Linear(in_f // 2, out_f, bias=False)
    W2_lin = nn.Linear(in_f // 2, out_f, bias=False)
    perm = torch.arange(in_f)  # identity permutation
    return MixedPrecisionLinear(
        column_perm=perm,
        sub_linears=nn.ModuleList([W4_lin, W2_lin]),
        split_sizes=[in_f // 2, in_f // 2],
        precision_nbits=[4, 2],
    )


def test_mixed_precision_linear_forward_shape():
    m = _make_trivial(64, 8)
    x = torch.randn(3, 64)
    y = m(x)
    assert y.shape == (3, 8)


def test_mixed_precision_linear_forward_batched():
    m = _make_trivial(64, 8)
    x = torch.randn(5, 10, 64)
    y = m(x)
    assert y.shape == (5, 10, 8)


def test_mixed_precision_linear_in_features():
    m = _make_trivial(64, 8)
    assert m.in_features == 64


def test_mixed_precision_linear_out_features():
    m = _make_trivial(64, 8)
    assert m.out_features == 8


def test_mixed_precision_linear_bias_added():
    m = _make_trivial(64, 8)
    m.bias = torch.ones(8)  # override the registered None buffer
    x = torch.zeros(2, 64)
    y = m(x)
    assert torch.allclose(y, torch.ones(2, 8))


def test_mixed_precision_linear_extra_repr():
    m = _make_trivial(64, 8)
    r = m.extra_repr()
    assert "4b" in r and "2b" in r


# ---------------------------------------------------------------------------
# split_mixed_precision_linear - correctness tests
# ---------------------------------------------------------------------------


def test_split_numerical_equivalence_two_precisions():
    """split must produce the same output as the original linear."""
    torch.manual_seed(0)
    linear = _linear(64, 16)
    # 4 panels: [W4, W2, W4, W2]
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)

    x = torch.randn(5, 64)
    with torch.no_grad():
        y_orig = linear(x)
        y_mixed = mixed(x)
    assert torch.allclose(y_orig, y_mixed, atol=1e-5), (
        f"max diff: {(y_orig - y_mixed).abs().max()}"
    )


def test_split_numerical_equivalence_all_same_precision():
    """Single precision → still numerically equivalent."""
    torch.manual_seed(1)
    linear = _linear(64, 8)
    assignments = _fake_assignments([4, 4, 4, 4], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)

    x = torch.randn(3, 64)
    with torch.no_grad():
        assert torch.allclose(linear(x), mixed(x), atol=1e-5)


def test_split_numerical_equivalence_three_precisions():
    torch.manual_seed(2)
    linear = _linear(96, 12)
    # 6 groups of 16 → 3 precisions
    assignments = _fake_assignments([4, 2, 8, 4, 2, 8], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)

    x = torch.randn(4, 96)
    with torch.no_grad():
        assert torch.allclose(linear(x), mixed(x), atol=1e-5)


def test_split_bias_preserved():
    torch.manual_seed(3)
    linear = nn.Linear(64, 8, bias=True)
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)

    x = torch.randn(3, 64)
    with torch.no_grad():
        assert torch.allclose(linear(x), mixed(x), atol=1e-5)


def test_split_sub_linear_count_matches_distinct_precisions():
    linear = _linear(64, 8)
    # 3 distinct precisions
    assignments = _fake_assignments([2, 4, 8, 2], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)
    assert len(mixed.sub_linears) == 3


def test_split_precision_nbits_descending_by_default():
    linear = _linear(64, 8)
    assignments = _fake_assignments([2, 4, 2, 4], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)
    # Default sort_descending=True → [4, 2]
    assert mixed.precision_nbits[0] > mixed.precision_nbits[-1]


def test_split_precision_nbits_ascending():
    linear = _linear(64, 8)
    assignments = _fake_assignments([2, 4, 2, 4], group_size=16)
    mixed = split_mixed_precision_linear(
        linear, assignments, group_size=16, sort_descending=False
    )
    assert mixed.precision_nbits[0] < mixed.precision_nbits[-1]


def test_split_column_perm_is_permutation():
    """column_perm must be a valid permutation of [0, in_features)."""
    linear = _linear(64, 8)
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)
    perm = mixed.column_perm
    assert isinstance(perm, torch.Tensor)
    assert perm.shape == (64,)
    assert set(perm.tolist()) == set(range(64))


def test_split_sizes_sum_to_in_features():
    linear = _linear(64, 8)
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    mixed = split_mixed_precision_linear(linear, assignments, group_size=16)
    assert sum(mixed.split_sizes) == 64


def test_split_wrong_group_size_raises():
    linear = _linear(64, 8)
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    with pytest.raises(ValueError, match="divisible"):
        split_mixed_precision_linear(linear, assignments, group_size=7)


def test_split_wrong_assignment_count_raises():
    linear = _linear(64, 8)
    # Only 2 assignments but in_features/group_size = 4
    assignments = _fake_assignments([4, 2], group_size=16)
    with pytest.raises(ValueError, match="Expected 4"):
        split_mixed_precision_linear(linear, assignments, group_size=16)


def test_split_missing_selection_raises():
    linear = _linear(32, 4)
    bad = PanelAssignment(panel_index=0, selected=None, ranked_scores=[])
    good = _fake_assignments([4], group_size=16)
    with pytest.raises(ValueError, match="no selected"):
        split_mixed_precision_linear(linear, [bad, good[0]], group_size=16)


# ---------------------------------------------------------------------------
# Integration: full GPTQ pipeline → split
# ---------------------------------------------------------------------------


def test_split_from_gptq_end_to_end_equivalence():
    """After GPTQ calibration, the split module must match the original."""
    module, calib = _calibrate_full(in_features=64, out_features=16, group_size=32)
    assignments = calib._mp_assignments
    assert assignments is not None

    mixed = split_mixed_precision_linear(module, assignments, group_size=32)

    x = torch.randn(4, 64)
    with torch.no_grad():
        y_orig = module(x)
        y_mixed = mixed(x)
    assert torch.allclose(y_orig, y_mixed, atol=1e-5), (
        f"max diff: {(y_orig - y_mixed).abs().max()}"
    )


def test_split_from_gptq_end_to_end_in_features_preserved():
    module, calib = _calibrate_full(64, 16, 32)
    mixed = split_mixed_precision_linear(module, calib._mp_assignments, 32)
    assert mixed.in_features == 64
    assert mixed.out_features == 16


# ---------------------------------------------------------------------------
# apply_split_to_model
# ---------------------------------------------------------------------------


class _TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 8, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_apply_split_replaces_targeted_layer():
    model = _TwoLayerModel()
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    apply_split_to_model(model, {"fc1": assignments}, group_size=16)
    assert isinstance(model.fc1, MixedPrecisionLinear)
    assert isinstance(model.fc2, nn.Linear)  # untouched


def test_apply_split_model_output_preserved():
    torch.manual_seed(7)
    model = _TwoLayerModel()
    assignments = _fake_assignments([4, 2, 4, 2], group_size=16)
    x = torch.randn(3, 64)
    with torch.no_grad():
        y_before = model(x)
    apply_split_to_model(model, {"fc1": assignments}, group_size=16)
    with torch.no_grad():
        y_after = model(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)


def test_apply_split_non_linear_raises():
    model = nn.Sequential(nn.ReLU(), nn.Linear(32, 8))
    assignments = _fake_assignments([4, 2], group_size=16)
    with pytest.raises(TypeError, match=r"nn\.Linear"):
        apply_split_to_model(model, {"0": assignments}, group_size=16)
