"""Tests for SmoothQuant implementation."""

import pytest
import torch
from torch import nn

from neuropress.calibration import ChannelMaxCollector
from neuropress.methods.smooth_quant import (
    absorb_scale_into_prev,
    apply_scale_to_linear,
    apply_smooth_quant,
    compute_smooth_scales,
)

# ---------------------------------------------------------------------------
# compute_smooth_scales
# ---------------------------------------------------------------------------


def test_compute_smooth_scales_output_shape():
    act_max = torch.ones(8)
    weight = torch.randn(4, 8)
    scale = compute_smooth_scales(act_max, weight)
    assert scale.shape == (8,)


def test_compute_smooth_scales_alpha_zero_pure_weight():
    # alpha=0 -> scale = 1 / weight_max (act_max^0 = 1)
    act_max = torch.full((4,), 5.0)
    weight = torch.zeros(2, 4)
    weight[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weight[1] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    weight_max = weight.abs().amax(dim=0)
    scale = compute_smooth_scales(act_max, weight, alpha=0.0)
    expected = torch.ones(4) / weight_max.clamp(min=1e-8)
    assert torch.allclose(scale, expected)


def test_compute_smooth_scales_alpha_one_pure_activation():
    # alpha=1 -> scale = act_max (weight_max^0 = 1)
    act_max = torch.tensor([2.0, 4.0, 6.0])
    weight = torch.ones(3, 3)
    scale = compute_smooth_scales(act_max, weight, alpha=1.0)
    assert torch.allclose(scale, act_max)


def test_compute_smooth_scales_formula_known_values():
    # act_max=8, weight_max=2, alpha=0.5 -> scale = sqrt(8)/sqrt(2) = 2.0
    act_max = torch.tensor([8.0])
    weight = torch.tensor([[2.0]])
    scale = compute_smooth_scales(act_max, weight, alpha=0.5)
    assert torch.allclose(scale, torch.tensor([2.0]))


def test_compute_smooth_scales_zero_weight_no_nan():
    act_max = torch.tensor([1.0, 1.0])
    weight = torch.zeros(2, 2)
    scale = compute_smooth_scales(act_max, weight)
    assert not torch.isnan(scale).any()
    assert not torch.isinf(scale).any()


def test_compute_smooth_scales_shape_mismatch_raises():
    act_max = torch.ones(5)
    weight = torch.randn(3, 8)
    with pytest.raises(ValueError, match="activation_max shape"):
        compute_smooth_scales(act_max, weight)


# ---------------------------------------------------------------------------
# apply_scale_to_linear
# ---------------------------------------------------------------------------


def test_apply_scale_to_linear_columns_divided():
    linear = nn.Linear(4, 3, bias=False)
    original_weight = linear.weight.data.clone()
    scale = torch.tensor([1.0, 2.0, 4.0, 8.0])
    apply_scale_to_linear(linear, scale)
    expected = original_weight / scale.unsqueeze(0)
    assert torch.allclose(linear.weight.data, expected)


def test_apply_scale_to_linear_bias_unchanged():
    linear = nn.Linear(4, 3, bias=True)
    original_bias = linear.bias.data.clone()
    scale = torch.tensor([1.0, 2.0, 4.0, 8.0])
    apply_scale_to_linear(linear, scale)
    assert torch.allclose(linear.bias.data, original_bias)


def test_apply_scale_to_linear_returns_none():
    linear = nn.Linear(4, 3, bias=False)
    scale = torch.ones(4)
    result = apply_scale_to_linear(linear, scale)
    assert result is None


# ---------------------------------------------------------------------------
# absorb_scale_into_prev
# ---------------------------------------------------------------------------


def test_absorb_scale_linear_weight_rows_multiplied():
    linear = nn.Linear(3, 4, bias=False)
    original_weight = linear.weight.data.clone()
    scale = torch.tensor([2.0, 3.0, 1.0, 0.5])
    absorb_scale_into_prev(linear, scale)
    expected = original_weight * scale.unsqueeze(1)
    assert torch.allclose(linear.weight.data, expected)


def test_absorb_scale_linear_bias_multiplied():
    linear = nn.Linear(3, 4, bias=True)
    original_bias = linear.bias.data.clone()
    scale = torch.tensor([2.0, 3.0, 1.0, 0.5])
    absorb_scale_into_prev(linear, scale)
    assert torch.allclose(linear.bias.data, original_bias * scale)


def test_absorb_scale_layernorm_weight_multiplied():
    ln = nn.LayerNorm(6)
    original_weight = ln.weight.data.clone()
    scale = torch.full((6,), 2.0)
    absorb_scale_into_prev(ln, scale)
    assert torch.allclose(ln.weight.data, original_weight * scale)


def test_absorb_scale_layernorm_bias_multiplied():
    ln = nn.LayerNorm(6)
    original_bias = ln.bias.data.clone()
    scale = torch.full((6,), 3.0)
    absorb_scale_into_prev(ln, scale)
    assert torch.allclose(ln.bias.data, original_bias * scale)


def test_absorb_scale_unsupported_type_raises():
    conv = nn.Conv2d(4, 4, 1)
    scale = torch.ones(4)
    with pytest.raises(TypeError, match="Unsupported preceding layer type"):
        absorb_scale_into_prev(conv, scale)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ChannelMaxCollector  (the Collector ABC implementation used by the registry)
# ---------------------------------------------------------------------------


def test_channel_max_collector_single_update_shape():
    coll = ChannelMaxCollector()
    x = torch.randn(8, 16)
    coll.update(x)
    assert coll.result()["channel_max"].shape == (16,)


def test_channel_max_collector_multiple_updates_running_max():
    coll = ChannelMaxCollector()
    x1 = torch.zeros(4, 8)
    x1[0, 3] = 5.0
    x2 = torch.zeros(4, 8)
    x2[1, 3] = 3.0
    x2[0, 0] = 10.0
    coll.update(x1)
    coll.update(x2)
    result = coll.result()["channel_max"]
    assert result[3].item() == pytest.approx(5.0)
    assert result[0].item() == pytest.approx(10.0)


def test_channel_max_collector_3d_input():
    coll = ChannelMaxCollector()
    x = torch.randn(2, 5, 12)  # [batch, seq, features]
    coll.update(x)
    assert coll.result()["channel_max"].shape == (12,)


def test_channel_max_collector_no_data_returns_none():
    coll = ChannelMaxCollector()
    assert coll.result()["channel_max"] is None


def test_channel_max_collector_result_key():
    coll = ChannelMaxCollector()
    coll.update(torch.randn(4, 8))
    result = coll.result()
    assert "channel_max" in result
    assert isinstance(result["channel_max"], torch.Tensor)


# ---------------------------------------------------------------------------
# apply_smooth_quant
# ---------------------------------------------------------------------------


class _TwoLinearModel(nn.Module):
    """Minimal two-layer model for apply_smooth_quant tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=False)
        self.fc2 = nn.Linear(16, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _make_calib_data(n: int = 4, in_features: int = 8) -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.randn(2, in_features)} for _ in range(n)]


def test_apply_smooth_quant_returns_one_entry_per_linear():
    model = _TwoLinearModel()
    calib = _make_calib_data()
    scales = apply_smooth_quant(model, calib)
    assert set(scales.keys()) == {"fc1", "fc2"}


def test_apply_smooth_quant_scale_shape():
    model = _TwoLinearModel()
    calib = _make_calib_data()
    scales = apply_smooth_quant(model, calib)
    assert scales["fc1"].shape == (8,)
    assert scales["fc2"].shape == (16,)


def test_apply_smooth_quant_numerical_equivalence():
    # W_new @ diag(s) @ x == W_old @ x when no prev_modules absorption.
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=False)
    w_orig = model.weight.data.clone()
    x = torch.randn(2, 8)

    class _Wrapper(nn.Module):
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            self.linear = linear

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    wrapper = _Wrapper(model)
    scales = apply_smooth_quant(wrapper, [{"input": x}])
    s = scales["linear"]

    # After: model.weight.data = W / s[j]  =>  (W / s) @ x
    # Recover W @ x by multiplying input by s: (W / s) @ (s * x)
    y_new = wrapper.linear(x * s.unsqueeze(0))
    y_orig = x @ w_orig.T
    assert torch.allclose(y_new, y_orig, atol=1e-5)


def test_apply_smooth_quant_prev_modules_absorbed():
    model = _TwoLinearModel()
    prev_ln = nn.LayerNorm(8)
    calib = _make_calib_data()
    orig_ln_weight = prev_ln.weight.data.clone()
    apply_smooth_quant(model, calib, prev_modules={"fc1": prev_ln})
    assert not torch.allclose(prev_ln.weight.data, orig_ln_weight)


def test_apply_smooth_quant_module_filter():
    model = _TwoLinearModel()
    calib = _make_calib_data()
    scales = apply_smooth_quant(
        model, calib, module_filter=lambda name, _: name == "fc1"
    )
    assert set(scales.keys()) == {"fc1"}


def test_apply_smooth_quant_no_hooks_remain_after_call():
    # StatisticsRegistry manages hooks; none should survive after collect().
    model = _TwoLinearModel()
    calib = _make_calib_data()
    apply_smooth_quant(model, calib)
    for module in model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_apply_smooth_quant_hooks_removed_on_exception():
    # StatisticsRegistry uses a finally block; hooks are removed even on error.
    model = _TwoLinearModel()

    class _FailingData:
        def __iter__(self):
            yield {"x": torch.randn(2, 8)}
            raise RuntimeError("simulated failure")

    with pytest.raises(RuntimeError, match="simulated failure"):
        apply_smooth_quant(model, _FailingData())  # type: ignore[arg-type]

    for module in model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_apply_smooth_quant_empty_data_raises():
    model = _TwoLinearModel()
    with pytest.raises(RuntimeError, match="channel_max"):
        apply_smooth_quant(model, [])
