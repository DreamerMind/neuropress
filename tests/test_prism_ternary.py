"""Tests for the Prism-compatible ternary PTQ path."""

import base64
import copy
import json

import torch
from safetensors import safe_open
from torch import nn

from neuropress.estimator_configs import TernaryGptqEstimatorConfig
from neuropress.formats.prism_ternary import (
    export_prism_ternary_checkpoint,
    pack_signs,
    pack_ternary_weight,
    unpack_ternary_weight,
)
from neuropress.methods.ternary import TernaryGPTQCalibration, TernaryScaleCalibration
from neuropress.preprocessing import TernaryHadamardStep
from neuropress.qscheme import Granularity, TernaryScaleScheme
from neuropress.transforms.ternary_hadamard import (
    TernaryHadamardInputLinear,
    apply_ternary_hadamard,
    block_hadamard_transform,
)


def test_block_hadamard_is_involution():
    x = torch.randn(3, 16)
    restored = block_hadamard_transform(block_hadamard_transform(x, 8), 8)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)


def test_hadamard_wrapping_preserves_linear_and_shares_signs():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 8))
    x = torch.randn(4, 16)
    expected = model(x)
    signs = apply_ternary_hadamard(model, block_size=8, seed=11)
    actual = model(x)

    assert isinstance(model[0], TernaryHadamardInputLinear)
    assert isinstance(model[1], TernaryHadamardInputLinear)
    assert set(signs) == {8, 16}
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_ternary_calibration_emits_only_three_codes():
    calibration = TernaryScaleCalibration(
        granularity=Granularity.per_group([4]), threshold_ratio=0.5
    )
    weight = torch.tensor([[-2.0, -0.1, 0.2, 1.8, -1.0, 0.0, 0.1, 1.1]])
    calibration.init()
    calibration.update(weight)
    scheme = calibration.compute_qscheme()
    codes = scheme.quantize_as_u8(weight)

    assert isinstance(scheme, TernaryScaleScheme)
    assert set(codes.flatten().tolist()) <= {0, 1, 2}
    assert scheme.qscheme_data_shape == [1, 1, 2, 1]


def test_ternary_scale_fit_converges():
    torch.manual_seed(20260921)
    weight = torch.randn(32, 128) * 0.08
    weight[torch.rand_like(weight) < 0.01] *= 6
    granularity = Granularity.per_group([128])

    results = []
    for iterations in (1, 8, 24):
        calibration = TernaryScaleCalibration(
            granularity=granularity,
            threshold_ratio=0.4,
            max_iterations=iterations,
        )
        calibration.init()
        calibration.update(weight)
        scheme = calibration.compute_qscheme()
        quantized = scheme.dequantize(scheme.quantize_as_u8(weight), weight.dtype)
        results.append((scheme.scale, (weight - quantized).square().mean()))

    torch.testing.assert_close(results[1][0], results[2][0])
    assert results[1][1] < results[0][1]


def _gptq_relative_output_mse(samples: int) -> float:
    torch.manual_seed(1000)
    module = nn.Linear(32, 32, bias=False)
    original = copy.deepcopy(module)
    stream = torch.randn(256, 32)
    validation = torch.randn(256, 32)
    granularity = Granularity.per_group([32])
    calibration = TernaryGPTQCalibration(
        granularity=granularity,
        possible_quantizers=[
            TernaryScaleCalibration(granularity=granularity, threshold_ratio=0.4)
        ],
        damp_percent=0.01,
        damp_auto_increment=0.01,
    )
    calibration.weight_tensor = module.weight
    calibration.module_class = type(module)
    calibration.module = module
    calibration.name = "q_proj.linear"
    calibration.init()
    calibration.update_statistics(stream[:samples])
    calibration.compute_qscheme()
    with torch.no_grad():
        expected = original(validation)
        actual = module(validation)
    return ((expected - actual).square().mean() / expected.square().mean()).item()


def test_ternary_gptq_improves_with_calibration_samples():
    under_calibrated = _gptq_relative_output_mse(16)
    converged = _gptq_relative_output_mse(256)

    assert converged < under_calibrated * 0.7


def test_ternary_gptq_produces_exportable_group_weights():
    granularity = Granularity.per_group([8])
    calibration = TernaryGPTQCalibration(
        granularity=granularity,
        possible_quantizers=[
            TernaryScaleCalibration(granularity=granularity, threshold_ratio=0.4)
        ],
    )
    module = nn.Linear(16, 32, bias=False)
    calibration.weight_tensor = module.weight
    calibration.module_class = type(module)
    calibration.module = module
    calibration.name = "q_proj.linear"
    calibration.init()
    for _ in range(3):
        calibration.update_statistics(torch.randn(8, 16))

    weights, scheme, *_ = calibration.compute_quantization()
    restored = scheme.dequantize(scheme.quantize_as_u8(weights), weights.dtype)

    assert isinstance(scheme, TernaryScaleScheme)
    torch.testing.assert_close(restored, weights)
    pack_ternary_weight(weights, group_size=8)


def test_recipe_targets_only_plugin_supported_wrappers():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 32)
            self.state_proj = nn.Linear(16, 32)

    model = TinyModel()
    TernaryHadamardStep(block_size=8).apply(model, [])
    estimator = TernaryGptqEstimatorConfig(group_size=8).build()
    estimator.prepare(model)

    assert isinstance(model.q_proj, TernaryHadamardInputLinear)
    assert isinstance(model.state_proj, nn.Linear)
    assert list(estimator._target_modules) == ["q_proj.linear"]


def test_pack_roundtrip_and_sign_layout():
    levels = torch.tensor(
        [[-1, 0, 1, 0, -1, 1, 0, 1, -1, 0, 1, 0, -1, 1, 0, 1]],
        dtype=torch.float32,
    )
    weight = levels * 0.25
    words, scales = pack_ternary_weight(weight, group_size=8)
    decoded = unpack_ternary_weight(words, scales, group_size=8)

    assert words.shape == (1, 1)
    torch.testing.assert_close(decoded, weight)
    assert base64.b64decode(
        pack_signs(torch.tensor([-1, 1, -1, 1, 1, 1, 1, 1]))
    ) == bytes([0xA0])


def test_export_checkpoint_uses_original_names(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(1024, 32, bias=False)

    model = TinyModel()
    apply_ternary_hadamard(model, block_size=1024, seed=7)
    wrapper = model.q_proj
    assert isinstance(wrapper, TernaryHadamardInputLinear)
    levels = (
        torch.arange(wrapper.linear.weight.numel()).reshape_as(wrapper.linear.weight)
        % 3
        - 1
    )
    wrapper.linear.weight.data.copy_(levels * 0.125)

    export_prism_ternary_checkpoint(
        model,
        tmp_path,
        {"architectures": ["TinyModel"]},
        max_shard_bytes=1024**2,
    )

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "prism_ternary"
    assert config["quantization_config"]["group_size"] == 128
    with safe_open(tmp_path / "model.safetensors", framework="pt") as checkpoint:
        assert set(checkpoint.keys()) == {"q_proj.scales", "q_proj.weight"}
        words = checkpoint.get_tensor("q_proj.weight")
        scales = checkpoint.get_tensor("q_proj.scales")
    torch.testing.assert_close(
        unpack_ternary_weight(words, scales),
        wrapper.linear.weight.float(),
    )
