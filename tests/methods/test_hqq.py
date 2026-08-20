"""Tests for HQQCalibration, HQQEstimator, and HqqEstimatorConfig."""

import pytest
import torch
from torch import nn

from neuropress.estimator_configs import HqqEstimatorConfig
from neuropress.estimators import HQQEstimator
from neuropress.methods.hqq import HQQCalibration, _hqq_optimize, _proximal_shrink
from neuropress.methods.zp_scale_calib import MinMaxZpScaleCalibration
from neuropress.plan import QuantizationPlan
from neuropress.qscheme import Granularity
from neuropress.qscheme.zp_scale import ZpScaleScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear(in_features: int = 64, out_features: int = 16) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False)


def _setup(
    calib: HQQCalibration | MinMaxZpScaleCalibration, weight: torch.Tensor
) -> None:
    """Bind *weight* to *calib* and call ``init()`` + ``update()``."""
    calib.weight_tensor = weight
    calib.module_class = nn.Linear
    calib.name = "test"
    calib.init()
    calib.update(weight)


def _quant_error(qscheme: ZpScaleScheme, data: torch.Tensor) -> float:
    """Mean squared dequantization error for *qscheme* applied to *data*."""
    d = data.reshape(qscheme.calibration_data_shape).float()
    zp = qscheme.zero_point.float()
    q = (d / qscheme.scale + zp).clamp(0, qscheme.maxq).round()
    dq = (q - zp) * qscheme.scale
    return (dq - d).pow(2).mean().item()


# ---------------------------------------------------------------------------
# _proximal_shrink
# ---------------------------------------------------------------------------


def test_proximal_shrink_l1_soft_threshold():
    beta = 2.0  # threshold = 1/beta = 0.5
    r = torch.tensor([-2.0, -0.4, 0.0, 0.4, 2.0])
    result = _proximal_shrink(r, beta=beta, lp_norm=1.0)
    expected = torch.tensor([-1.5, 0.0, 0.0, 0.0, 1.5])
    assert torch.allclose(result, expected, atol=1e-6)


def test_proximal_shrink_l1_preserves_sign():
    r = torch.tensor([-3.0, 3.0])
    result = _proximal_shrink(r, beta=1.0, lp_norm=1.0)
    assert result[0] < 0
    assert result[1] > 0


def test_proximal_shrink_l2_ridge():
    beta = 4.0
    r = torch.tensor([1.0, 2.0, -3.0])
    result = _proximal_shrink(r, beta=beta, lp_norm=2.0)
    expected = r / (1.0 + 2.0 / beta)
    assert torch.allclose(result, expected, atol=1e-6)


def test_proximal_shrink_l05_nonnegative_for_nonneg_input():
    r = torch.tensor([0.0, 0.1, 0.5, 1.0, 2.0])
    result = _proximal_shrink(r, beta=10.0, lp_norm=0.5)
    assert (result >= 0).all()


def test_proximal_shrink_l05_zero_stays_zero():
    result = _proximal_shrink(torch.tensor([0.0]), beta=100.0, lp_norm=0.5)
    assert result.item() == pytest.approx(0.0, abs=1e-6)


def test_proximal_shrink_general_shrinks_toward_zero():
    r = torch.tensor([1.0])
    for lp in (0.5, 1.0, 2.0):
        out = _proximal_shrink(r, beta=1.0, lp_norm=lp)
        assert out.abs().item() <= r.abs().item() + 1e-6, f"lp={lp} did not shrink"


# ---------------------------------------------------------------------------
# _hqq_optimize — unit tests on the core algorithm
# ---------------------------------------------------------------------------


def test_hqq_optimize_returns_positive_scale():
    torch.manual_seed(0)
    W = torch.randn(4, 1, 2, 8)  # [rows, 1, n_groups, group_size]
    scale, _zero = _hqq_optimize(
        W,
        nbits=4,
        symmetric=False,
        reduction_axes=[1, 3],
        num_iter=5,
        beta=1e4,
        kappa=1.01,
        lp_norm=0.5,
    )
    assert (scale > 0).all()


def test_hqq_optimize_zero_in_valid_range():
    torch.manual_seed(1)
    W = torch.randn(4, 1, 2, 8)
    maxq = 15  # 4-bit
    _, zero = _hqq_optimize(
        W,
        nbits=4,
        symmetric=False,
        reduction_axes=[1, 3],
        num_iter=10,
        beta=1e4,
        kappa=1.01,
        lp_norm=0.5,
    )
    assert (zero >= 0).all()
    assert (zero <= maxq).all()


def test_hqq_optimize_zero_iter_matches_minmax():
    """num_iter=0 must produce the same result as a plain min-max estimate."""
    torch.manual_seed(2)
    W = torch.randn(4, 1, 2, 32)
    reduction_axes = [1, 3]

    scale_hqq, zero_hqq = _hqq_optimize(
        W,
        nbits=4,
        symmetric=False,
        reduction_axes=reduction_axes,
        num_iter=0,
        beta=1e4,
        kappa=1.01,
        lp_norm=0.5,
    )

    # Manual min-max reference (same logic as ZpScaleCalibration).
    w_min = W.amin(dim=reduction_axes, keepdim=True).clamp_max(0.0)
    w_max = W.amax(dim=reduction_axes, keepdim=True).clamp_min(0.0)
    maxq = 15
    scale_ref = (w_max - w_min) / maxq
    zero_ref = (-w_min / scale_ref).round().clamp(0.0, float(maxq))

    assert torch.allclose(scale_hqq, scale_ref, atol=1e-6)
    assert torch.allclose(zero_hqq, zero_ref, atol=1e-6)


def test_hqq_optimize_symmetric_zero_fixed():
    """For symmetric mode zero-point must stay at (maxq+1)/2 = 8 for 4-bit."""
    torch.manual_seed(3)
    W = torch.randn(4, 1, 2, 8)
    _, zero = _hqq_optimize(
        W,
        nbits=4,
        symmetric=True,
        reduction_axes=[1, 3],
        num_iter=10,
        beta=1e4,
        kappa=1.01,
        lp_norm=0.5,
    )
    expected_zp = (2**4) / 2  # = 8.0; implementation uses (maxq+1)/2 = 16/2 = 8
    assert torch.allclose(zero, torch.full_like(zero, expected_zp), atol=1e-6)


# ---------------------------------------------------------------------------
# HQQCalibration
# ---------------------------------------------------------------------------


def test_hqq_calibration_shapes_per_group():
    module = _make_linear(64, 16)
    gran = Granularity.per_group([32])
    calib = HQQCalibration(granularity=gran, nbits=4)
    _setup(calib, module.weight.data)

    # 16 rows, 64 cols / 32 group = 2 groups.
    assert calib.calibration_data_shape == [16, 1, 2, 32]
    assert calib.qscheme_data_shape == [16, 1, 2, 1]
    assert calib.reduction_axes == [1, 3]


def test_hqq_calibration_shapes_per_channel():
    module = _make_linear(64, 16)
    gran = Granularity.per_channel()
    calib = HQQCalibration(granularity=gran, nbits=4)
    _setup(calib, module.weight.data)

    assert calib.calibration_data_shape == [16, 1, 1, 64]
    assert calib.qscheme_data_shape == [16, 1, 1, 1]


def test_hqq_compute_qscheme_returns_zpscalescheme():
    module = _make_linear(64, 16)
    gran = Granularity.per_group([32])
    calib = HQQCalibration(granularity=gran, nbits=4, num_iter=5)
    _setup(calib, module.weight.data)
    qscheme = calib.compute_qscheme()
    assert isinstance(qscheme, ZpScaleScheme)


def test_hqq_compute_qscheme_nbits_preserved():
    module = _make_linear(64, 16)
    gran = Granularity.per_group([32])
    for nbits in (2, 4, 8):
        calib = HQQCalibration(granularity=gran, nbits=nbits, num_iter=3)
        _setup(calib, module.weight.data)
        assert calib.compute_qscheme().nbits == nbits


def test_hqq_qscheme_scale_positive():
    module = _make_linear(64, 16)
    gran = Granularity.per_group([32])
    calib = HQQCalibration(granularity=gran, nbits=4, num_iter=10)
    _setup(calib, module.weight.data)
    assert (calib.compute_qscheme().scale > 0).all()


def test_hqq_symmetric_zero_point_fixed():
    """Symmetric mode: zero-point must equal (maxq+1)//2 = 8 for 4-bit."""
    module = _make_linear(64, 16)
    gran = Granularity.per_group([32])
    calib = HQQCalibration(granularity=gran, nbits=4, symmetric=True, num_iter=10)
    _setup(calib, module.weight.data)
    qscheme = calib.compute_qscheme()
    expected_zp = (2**4 + 1) // 2  # 8
    assert (qscheme.zero_point == expected_zp).all()


def test_hqq_num_iter_0_matches_minmax():
    """With num_iter=0, HQQ must match MinMaxZpScaleCalibration exactly."""
    torch.manual_seed(42)
    w = torch.randn(16, 64)
    gran = Granularity.per_group([32])

    calib_mm = MinMaxZpScaleCalibration(granularity=gran, nbits=4, symmetric=False)
    _setup(calib_mm, w)
    qs_mm = calib_mm.compute_qscheme()

    calib_hqq = HQQCalibration(granularity=gran, nbits=4, symmetric=False, num_iter=0)
    _setup(calib_hqq, w)
    qs_hqq = calib_hqq.compute_qscheme()

    assert torch.allclose(qs_hqq.scale, qs_mm.scale, atol=1e-5)
    # Zero-points may differ by at most 1 due to float→int rounding.
    assert torch.allclose(qs_hqq.zero_point.float(), qs_mm.zero_point.float(), atol=1.0)


def test_hqq_iterations_produce_valid_scheme():
    """HQQ with many iterations must stay within a sane range of the min-max scheme."""
    torch.manual_seed(7)
    w = torch.randn(16, 64)
    gran = Granularity.per_group([32])

    calib_mm = MinMaxZpScaleCalibration(granularity=gran, nbits=4, symmetric=False)
    _setup(calib_mm, w)
    qs_mm = calib_mm.compute_qscheme()

    calib_hqq = HQQCalibration(granularity=gran, nbits=4, num_iter=20)
    _setup(calib_hqq, w)
    qs_hqq = calib_hqq.compute_qscheme()

    # Scale must stay within 2x of the min-max scale (no catastrophic divergence).
    ratio = qs_hqq.scale / qs_mm.scale.clamp(min=1e-8)
    assert ((ratio >= 0.5) & (ratio <= 2.0)).all()
    # Zero-point must remain in the valid unsigned integer range [0, maxq].
    assert (qs_hqq.zero_point >= 0).all()
    assert (qs_hqq.zero_point <= 2**4 - 1).all()


def test_hqq_lp_norms_all_produce_valid_scheme():
    module = _make_linear(64, 16)
    gran = Granularity.per_group([32])
    for lp in (0.5, 1.0, 2.0):
        calib = HQQCalibration(granularity=gran, nbits=4, num_iter=5, lp_norm=lp)
        _setup(calib, module.weight.data)
        qs = calib.compute_qscheme()
        assert isinstance(qs, ZpScaleScheme)
        assert (qs.scale > 0).all()


def test_hqq_rejects_second_distinct_sample():
    gran = Granularity.per_group([32])
    calib = HQQCalibration(granularity=gran, nbits=4)
    w1 = torch.randn(16, 64)
    w2 = torch.randn(16, 64)
    calib.weight_tensor = w1
    calib.module_class = nn.Linear
    calib.name = "test"
    calib.init()
    calib.update(w1)
    with pytest.raises(NotImplementedError):
        calib.update(w2)


def test_hqq_accepts_same_sample_twice():
    gran = Granularity.per_group([32])
    calib = HQQCalibration(granularity=gran, nbits=4)
    w = torch.randn(16, 64)
    calib.weight_tensor = w
    calib.module_class = nn.Linear
    calib.name = "test"
    calib.init()
    calib.update(w)
    calib.update(w)  # identical tensor — must not raise
    assert isinstance(calib.compute_qscheme(), ZpScaleScheme)


def test_hqq_registered_in_method_registry():
    from neuropress.methods import METHOD_REGISTRY

    assert "hqq" in METHOD_REGISTRY
    assert METHOD_REGISTRY["hqq"] is HQQCalibration


# ---------------------------------------------------------------------------
# HQQEstimator
# ---------------------------------------------------------------------------


class _TwoLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def test_hqq_estimator_declares_no_statistics():
    model = _TwoLinearModel()
    est = HQQEstimator()
    est.prepare(model)
    assert est.declare_statistics() == []


def test_hqq_estimator_updates_weights_in_place():
    torch.manual_seed(0)
    model = _TwoLinearModel()
    w1_before = model.fc1.weight.data.clone()
    w2_before = model.fc2.weight.data.clone()

    est = HQQEstimator(group_size=32, nbits=4, num_iter=2)
    est.prepare(model)
    est.estimate(graph=None, registry=None, recipe=None)

    assert not torch.allclose(model.fc1.weight.data, w1_before)
    assert not torch.allclose(model.fc2.weight.data, w2_before)


def test_hqq_estimator_filter_fn_skips_module():
    model = _TwoLinearModel()
    w2_before = model.fc2.weight.data.clone()

    est = HQQEstimator(filter_fn=lambda name, _: "fc1" in name, group_size=32)
    est.prepare(model)
    est.estimate(graph=None, registry=None, recipe=None)

    # fc2 must be untouched.
    assert torch.allclose(model.fc2.weight.data, w2_before)


def test_hqq_estimator_falls_back_to_per_channel_when_not_divisible():
    # in_features=7 is not divisible by group_size=32.
    model = nn.Sequential(nn.Linear(7, 4, bias=False))
    est = HQQEstimator(module_classes=(nn.Linear,), group_size=32, nbits=4, num_iter=1)
    est.prepare(model)
    plan = est.estimate(graph=None, registry=None, recipe=None)
    assert isinstance(plan, QuantizationPlan)


def test_hqq_estimator_with_weight_transform():
    model = _TwoLinearModel()
    collected: dict[str, torch.Tensor] = {}

    def _transform(weight: torch.Tensor, name: str) -> torch.Tensor:
        collected[name] = weight.clone()
        return weight.clone()

    est = HQQEstimator(weight_transform=_transform, group_size=32, num_iter=1)
    est.prepare(model)
    plan = est.estimate(graph=None, registry=None, recipe=None)

    assert "fc1" in collected
    assert "fc2" in collected
    assert len(plan.weight_assignments) == 2


def test_hqq_estimator_weight_transform_none_skips_plan():
    model = _TwoLinearModel()
    est = HQQEstimator(weight_transform=None, group_size=32, num_iter=1)
    est.prepare(model)
    plan = est.estimate(graph=None, registry=None, recipe=None)
    assert plan.weight_assignments == {}


def test_hqq_estimator_shared_weight_processed_once():
    class _SharedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(64, 32, bias=False)
            self.fc2 = nn.Linear(64, 32, bias=False)
            self.fc2.weight = self.fc1.weight  # intentionally shared

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc1(x) + self.fc2(x)

    model = _SharedModel()
    call_count = [0]

    def _counting_transform(weight: torch.Tensor, name: str) -> torch.Tensor:
        call_count[0] += 1
        return weight.clone()

    est = HQQEstimator(weight_transform=_counting_transform, group_size=32, num_iter=1)
    est.prepare(model)
    est.estimate(graph=None, registry=None, recipe=None)

    assert call_count[0] == 1  # shared weight processed exactly once


def test_hqq_estimator_empty_model():
    model = nn.Sequential(nn.ReLU())  # no Linear layers
    est = HQQEstimator()
    est.prepare(model)
    plan = est.estimate(graph=None, registry=None, recipe=None)
    assert plan.weight_assignments == {}


# ---------------------------------------------------------------------------
# HqqEstimatorConfig
# ---------------------------------------------------------------------------


def test_hqq_estimator_config_registered_in_registry():
    from neuropress.recipe_config import ESTIMATOR_CONFIG_REGISTRY

    assert "hqq" in ESTIMATOR_CONFIG_REGISTRY
    assert ESTIMATOR_CONFIG_REGISTRY["hqq"] is HqqEstimatorConfig


def test_hqq_estimator_config_builds_hqq_estimator():
    config = HqqEstimatorConfig(target="linear", group_size=32, nbits=4, num_iter=5)
    est = config.build()
    assert isinstance(est, HQQEstimator)
    assert est.group_size == 32
    assert est.nbits == 4
    assert est.num_iter == 5


def test_hqq_estimator_config_all_fields_forwarded():
    config = HqqEstimatorConfig(
        group_size=16,
        nbits=8,
        symmetric=True,
        num_iter=10,
        beta=500.0,
        kappa=1.05,
        lp_norm=1.0,
    )
    est = config.build()
    assert est.group_size == 16
    assert est.nbits == 8
    assert est.symmetric is True
    assert est.num_iter == 10
    assert est.beta == pytest.approx(500.0)
    assert est.kappa == pytest.approx(1.05)
    assert est.lp_norm == pytest.approx(1.0)


def test_hqq_recipe_config_roundtrip_from_json():
    from neuropress.recipe_config import QuantizationRecipeConfig

    json_str = (
        '{"estimator": {"type": "hqq", "nbits": 4, "group_size": 32, "num_iter": 10}}'
    )
    config = QuantizationRecipeConfig.model_validate_json(json_str)
    assert isinstance(config.estimator, HqqEstimatorConfig)
    est = config.estimator.build()
    assert isinstance(est, HQQEstimator)
    assert est.nbits == 4
    assert est.num_iter == 10


def test_hqq_estimator_config_invalid_weight_transform_raises():
    config = HqqEstimatorConfig(weight_transform="nonexistent_transform")
    with pytest.raises(ValueError, match="nonexistent_transform"):
        config.build()
