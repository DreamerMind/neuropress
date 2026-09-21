"""Tests for the Prism-compatible ternary PTQ path."""

import base64
import copy
import json

import pytest
import torch
from safetensors import safe_open
from torch import nn
from torch_to_nnef.tensor import OffloadedTensor

from neuropress.calibration.probe import (
    HessianAccumulator,
    StatisticRequest,
    StatisticsRegistry,
)
from neuropress.estimator_configs import TernaryGptqEstimatorConfig
from neuropress.estimators import CompositeEstimator
from neuropress.formats.prism_ternary import (
    export_prism_ternary_checkpoint,
    pack_signs,
    pack_ternary_weight,
    unpack_ternary_weight,
)
from neuropress.methods import METHOD_REGISTRY
from neuropress.methods.ternary import TernaryGPTQCalibration, TernaryScaleCalibration
from neuropress.preprocessing import TernaryHadamardStep, TernaryHadamardStepConfig
from neuropress.qscheme import Granularity, TernaryScaleScheme
from neuropress.quantization.cache import DiskCacheManager
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
    estimator = TernaryGptqEstimatorConfig().build()
    estimator.prepare(model)

    assert isinstance(model.q_proj, TernaryHadamardInputLinear)
    assert isinstance(model.state_proj, nn.Linear)
    assert list(estimator._target_modules) == ["q_proj.linear"]


def test_recipe_rejects_plugin_incompatible_constants():
    with pytest.raises(ValueError):
        TernaryGptqEstimatorConfig(target="conv2d")
    with pytest.raises(ValueError):
        TernaryGptqEstimatorConfig(group_size=64)
    with pytest.raises(ValueError):
        TernaryHadamardStepConfig(block_size=512)


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


@pytest.mark.parametrize(
    "signs", [torch.tensor([1.0, 0.0]), torch.tensor([1.0, float("nan")])]
)
def test_pack_signs_rejects_invalid_values(signs):
    with pytest.raises(ValueError, match=r"only -1 and \+1"):
        pack_signs(signs)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_pack_rejects_nonfinite_weights(bad_value):
    weight = torch.zeros(2, 16)
    weight[0, 0] = bad_value
    with pytest.raises(ValueError, match="non-finite"):
        pack_ternary_weight(weight, group_size=8)


def test_pack_is_identical_across_row_chunk_sizes():
    levels = torch.randint(-1, 2, (9, 32), dtype=torch.int64).float()
    scales = torch.linspace(0.125, 1.0, 9)[:, None]
    weight = levels * scales

    expected = pack_ternary_weight(weight, group_size=16, row_chunk_size=64)
    actual = pack_ternary_weight(weight, group_size=16, row_chunk_size=1)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_registry_ternary_gptq_defaults_to_ternary_quantizer():
    calibration = METHOD_REGISTRY["ternary_gptq"](
        granularity=Granularity.per_group([8])
    )
    assert len(calibration.possible_quantizers) == 1
    assert isinstance(calibration.possible_quantizers[0], TernaryScaleCalibration)


def test_ternary_estimator_shares_and_offloads_hessians(tmp_path, monkeypatch):
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 32)
            self.k_proj = nn.Linear(16, 32)
            self.v_proj = nn.Linear(16, 32)
            self.o_proj = nn.Linear(16, 32)

        def forward(self, x):
            return sum(
                projection(x)
                for projection in (
                    self.q_proj,
                    self.k_proj,
                    self.v_proj,
                    self.o_proj,
                )
            )

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(16, 32)
            self.up_proj = nn.Linear(16, 32)
            self.down_proj = nn.Linear(16, 32)

        def forward(self, x):
            return self.gate_proj(x) + self.up_proj(x) + self.down_proj(x)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()

        def forward(self, x):
            return self.self_attn(x) + self.mlp(x)

    model = Layer()
    apply_ternary_hadamard(model, block_size=8)
    estimator = TernaryGptqEstimatorConfig().build()
    scopes = {stage.name: stage.scope for stage in estimator.cache_stages()}
    estimator.set_cache(DiskCacheManager(tmp_path / "cache", "cal", "recipe", scopes))
    estimator.prepare(model)
    requests = estimator.declare_statistics()

    assert [request.target_key for request in requests] == [
        "self_attn.q_proj.linear",
        "self_attn.o_proj.linear",
        "mlp.gate_proj.linear",
        "mlp.down_proj.linear",
    ]
    registry = StatisticsRegistry()
    registry.register_requests(requests)
    estimator.configure_statistics_registry(registry)
    assert len(registry._collectors) == 4
    assert all(
        isinstance(collector, HessianAccumulator)
        and collector.offload_intermediate_tensor_on_disk
        for collector in registry._collectors.values()
    )
    offload_dir = tmp_path / "hessians"
    offload_dir.mkdir()
    monkeypatch.setattr(HessianAccumulator, "_offload_dir", offload_dir)
    registry.collect(model, [torch.randn(2, 16), torch.randn(2, 16)])
    offload_paths = []
    for request in requests:
        hessian = registry.get(request)["hessian"]
        assert hessian.elem.device.type == "meta"
        assert hessian.offload_path.exists()
        offload_paths.append(hessian.offload_path)

    reload_counts: dict[str, int] = {}
    original_reload = OffloadedTensor._reload_unmanaged

    def counted_reload(self, *args, **kwargs):
        reload_counts[self._name] = reload_counts.get(self._name, 0) + 1
        return original_reload(self, *args, **kwargs)

    monkeypatch.setattr(OffloadedTensor, "_reload_unmanaged", counted_reload)
    estimator.estimate(None, registry, None)
    assert len(reload_counts) == len(requests)
    assert set(reload_counts.values()) == {1}
    assert all(path.exists() for path in offload_paths)
    registry.release_all()
    assert not any(path.exists() for path in offload_paths)
    for module in estimator._target_modules.values():
        for row in module.weight.detach().abs():
            assert torch.unique(row).numel() <= 2


def test_partial_disk_cache_collects_only_missing_hessians(tmp_path, monkeypatch):
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 32)
            self.o_proj = nn.Linear(16, 32)

        def forward(self, x):
            return self.q_proj(x) + self.o_proj(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

        def forward(self, x):
            return self.self_attn(x)

    model = Model()
    apply_ternary_hadamard(model, block_size=8)
    estimator = TernaryGptqEstimatorConfig().build()
    scopes = {stage.name: stage.scope for stage in estimator.cache_stages()}
    assert scopes == {"hessian": "recipe"}
    cache = DiskCacheManager(tmp_path / "cache", "cal", "recipe", scopes)
    estimator.set_cache(cache)
    estimator.prepare(model)

    cached_source = "self_attn.q_proj.linear"
    missing_source = "self_attn.o_proj.linear"
    cache.save(
        "hessian",
        cached_source,
        {"hessian": torch.eye(16), "nsamples": 2},
    )
    cached = cache.load("hessian", cached_source)
    assert cached is not None
    assert cached["nsamples"] == 2
    assert isinstance(cached["nsamples"], int)

    requests = estimator.declare_statistics()
    assert [request.target_key for request in requests] == [missing_source]
    registry = StatisticsRegistry()
    registry.register_requests(requests)
    estimator.configure_statistics_registry(registry)
    assert len(registry._collectors) == 1
    assert all(
        collector.offload_intermediate_tensor_on_disk
        for collector in registry._collectors.values()
    )

    offload_dir = tmp_path / "hessians"
    offload_dir.mkdir()
    monkeypatch.setattr(HessianAccumulator, "_offload_dir", offload_dir)
    registry.collect(model, [torch.randn(2, 16)])
    assert registry.get(
        next(request for request in requests if request.target_key == missing_source)
    )["hessian"].offload_path.exists()
    assert not registry.has(
        StatisticRequest(target_key=cached_source, stat_type="hessian")
    )

    estimator.estimate(None, registry, None)
    assert cache.has("hessian", cached_source)
    assert cache.has("hessian", missing_source)
    newly_cached = cache.load("hessian", missing_source)
    assert newly_cached is not None
    assert newly_cached["hessian"].shape == (16, 16)
    assert newly_cached["nsamples"] == 1
    assert isinstance(newly_cached["nsamples"], int)
    cached_estimator = TernaryGptqEstimatorConfig().build()
    cached_estimator.set_cache(cache)
    cached_estimator.prepare(model)
    assert cached_estimator.declare_statistics() == []


def test_nonternary_gptq_hessian_cache_remains_calibration_scoped():
    from neuropress.estimators import GPTQEstimator

    assert GPTQEstimator().cache_stages()[0].scope == "calibration"


def test_composite_delegates_disk_backed_hessian_configuration():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 32)

    model = Model()
    apply_ternary_hadamard(model, block_size=8)
    child = TernaryGptqEstimatorConfig().build()
    estimator = CompositeEstimator([child])
    estimator.prepare(model)
    requests = estimator.declare_statistics()
    registry = StatisticsRegistry()
    registry.register_requests(requests)

    estimator.configure_statistics_registry(registry)

    assert len(registry._collectors) == 1
    collector = next(iter(registry._collectors.values()))
    assert isinstance(collector, HessianAccumulator)
    assert collector.offload_intermediate_tensor_on_disk


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


def test_export_nested_hf_names_and_shard_index(tmp_path):
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(1024, 32, bias=False)
            self.k_proj = nn.Linear(1024, 32, bias=False)
            self.v_proj = nn.Linear(1024, 32, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class InnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = InnerModel()

    model = Model()
    apply_ternary_hadamard(model, block_size=1024, seed=9)
    for module in model.modules():
        if isinstance(module, TernaryHadamardInputLinear):
            levels = torch.randint(-1, 2, module.linear.weight.shape)
            module.linear.weight.data.copy_(levels * 0.25)

    output = tmp_path / "export"
    export_prism_ternary_checkpoint(
        model,
        output,
        {"architectures": ["Qwen3_5ForConditionalGeneration"]},
        max_shard_bytes=9_000,
        packing_row_chunk_size=1,
    )

    index = json.loads((output / "model.safetensors.index.json").read_text())
    expected_prefix = "model.language_model.layers.0.self_attn"
    assert set(index["weight_map"]) == {
        f"{expected_prefix}.{projection}.{suffix}"
        for projection in ("q_proj", "k_proj", "v_proj")
        for suffix in ("weight", "scales")
    }
    assert all(".linear." not in key for key in index["weight_map"])
    assert len(set(index["weight_map"].values())) > 1


def test_export_failure_leaves_no_partial_checkpoint(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(1024, 32, bias=False)

    model = TinyModel()
    apply_ternary_hadamard(model, block_size=1024)
    output = tmp_path / "export"
    output.mkdir()
    with pytest.raises(ValueError, match="not strict ternary"):
        export_prism_ternary_checkpoint(model, output, {"architectures": ["TinyModel"]})
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".export.tmp-*"))
