"""Tests for dense modules, tied embeddings and sequential ternary GPTQ."""

import json

import pytest
import torch
from safetensors import safe_open
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from neuropress.estimator_configs import TernaryGptqEstimatorConfig
from neuropress.estimators import GPTQEstimator
from neuropress.formats.prism_ternary import (
    export_prism_ternary_checkpoint,
    pack_ternary_weight,
)
from neuropress.methods.sequential import quantize_blocks_sequentially
from neuropress.methods.ternary import TernaryGPTQCalibration, TernaryScaleCalibration
from neuropress.preprocessing import TernaryHadamardStep
from neuropress.qscheme import Granularity, TernaryScaleScheme
from neuropress.transforms.ternary_hadamard import TernaryHadamardInputLinear


def _tiny_llama(*, tie: bool = False, layers: int = 2) -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=1024,
        tie_word_embeddings=tie,
    )
    return LlamaForCausalLM(config).eval()


def _batches(n: int = 4, seq: int = 16) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1)
    return [
        {"input_ids": torch.randint(0, 1024, (1, seq), generator=generator)}
        for _ in range(n)
    ]


def _make_wrapped_weights_ternary(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, TernaryHadamardInputLinear):
            weight = module.linear.weight
            levels = torch.arange(weight.numel()).reshape_as(weight) % 3 - 1
            weight.data.copy_(levels * 0.125)


def test_dense_modules_are_not_wrapped_and_are_exported_dense(tmp_path):
    model = _tiny_llama()
    original = model.lm_head.weight.detach().clone()
    TernaryHadamardStep().apply(model, [])

    assert type(model.lm_head) is nn.Linear
    torch.testing.assert_close(model.lm_head.weight, original)
    assert isinstance(
        model.model.layers[0].self_attn.q_proj, TernaryHadamardInputLinear
    )

    _make_wrapped_weights_ternary(model)
    export_prism_ternary_checkpoint(model, tmp_path / "out", model.config.to_dict())

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["quantization_config"]["dense_modules"] == ["lm_head"]
    with safe_open(tmp_path / "out" / "model.safetensors", framework="pt") as ckpt:
        assert ckpt.get_tensor("lm_head.weight").dtype == original.dtype
        assert "lm_head.scales" not in ckpt.keys()


def test_all_ternary_export_omits_dense_modules_key(tmp_path):
    model = _tiny_llama()
    TernaryHadamardStep(dense_modules=()).apply(model, [])
    _make_wrapped_weights_ternary(model)
    export_prism_ternary_checkpoint(model, tmp_path / "out", model.config.to_dict())

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert "dense_modules" not in config["quantization_config"]


def test_ternary_lm_head_unties_embeddings(tmp_path):
    model = _tiny_llama(tie=True)
    embedding = model.model.embed_tokens.weight.detach().clone()
    TernaryHadamardStep(dense_modules=()).apply(model, [])

    assert model.config.tie_word_embeddings is False
    torch.testing.assert_close(model.model.embed_tokens.weight, embedding)

    _make_wrapped_weights_ternary(model)
    base = model.config.to_dict()
    base["tie_word_embeddings"] = True  # as in the original checkpoint
    export_prism_ternary_checkpoint(model, tmp_path / "out", base)

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["tie_word_embeddings"] is False
    with safe_open(tmp_path / "out" / "model.safetensors", framework="pt") as ckpt:
        keys = set(ckpt.keys())
    assert {"model.embed_tokens.weight", "lm_head.weight", "lm_head.scales"} <= keys


def test_dense_lm_head_keeps_tie():
    model = _tiny_llama(tie=True)
    TernaryHadamardStep().apply(model, [])

    assert model.config.tie_word_embeddings is True
    assert model.lm_head.weight is model.model.embed_tokens.weight


GRID = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)


def _weighted_error(data, scheme, weights):
    restored = scheme.dequantize(scheme.quantize_as_u8(data), data.dtype)
    return ((data - restored).square() * weights).sum()


def test_grid_fit_never_scores_worse_than_fixed_threshold():
    torch.manual_seed(3)
    data = torch.randn(64, 128) * torch.linspace(0.2, 3.0, 128)
    weights = torch.rand(128) + 0.1
    granularity = Granularity.per_group([128])

    def fit(grid):
        quantizer = TernaryScaleCalibration(
            granularity=granularity, threshold_ratio=0.4, threshold_grid=grid
        )
        quantizer.init()
        quantizer.column_weights = weights
        quantizer.update(data)
        return quantizer.compute_qscheme()

    fixed, searched = fit(()), fit(GRID)

    assert isinstance(searched, TernaryScaleScheme)
    assert _weighted_error(data, searched, weights) <= _weighted_error(
        data, fixed, weights
    )
    assert searched.threshold_ratio.unique().numel() > 1


def test_grid_ternary_gptq_output_is_strict_and_exportable():
    torch.manual_seed(4)
    granularity = Granularity.per_group([128])
    module = nn.Linear(256, 64, bias=False)
    calibration = TernaryGPTQCalibration(
        granularity=granularity,
        possible_quantizers=[
            TernaryScaleCalibration(
                granularity=granularity, threshold_ratio=0.4, threshold_grid=GRID
            )
        ],
    )
    calibration.weight_tensor = module.weight
    calibration.module_class = nn.Linear
    calibration.module = module
    calibration.name = "q_proj.linear"
    calibration.init()
    calibration.update_statistics(torch.randn(512, 256))

    weights, scheme, *_ = calibration.compute_quantization()
    restored = scheme.dequantize(scheme.quantize_as_u8(weights), weights.dtype)

    torch.testing.assert_close(restored, weights)
    pack_ternary_weight(weights, group_size=128)


def test_config_searches_thresholds_by_default():
    estimator = TernaryGptqEstimatorConfig().build()
    assert 0.4 in estimator.ternary_threshold_grid
    assert len(estimator.ternary_threshold_grid) > 1


def test_config_rejects_invalid_threshold_grid():
    with pytest.raises(ValueError, match="threshold_grid"):
        TernaryGptqEstimatorConfig(threshold_grid=[0.0, 0.5]).build()


def _input_hessians(model, names, batches):
    """2/n * sum(x x^T) of each named module's input over a plain forward."""
    sums, counts, hooks = {}, {}, []
    modules = dict(model.named_modules())

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            sums[name] = sums.get(name, 0) + x.T @ x
            counts[name] = counts.get(name, 0) + x.shape[0]

        return hook

    for name in names:
        hooks.append(modules[name].register_forward_hook(make_hook(name)))
    with torch.inference_mode():
        for batch in batches:
            model(**batch)
    for hook in hooks:
        hook.remove()
    return {name: 2.0 * sums[name] / counts[name] for name in names}


@pytest.mark.parametrize("modify", [False, True])
def test_sequential_hessians_see_the_quantized_prefix(modify):
    model = _tiny_llama()
    TernaryHadamardStep(dense_modules=()).apply(model, [])
    estimator = TernaryGptqEstimatorConfig().build()
    estimator.prepare(model)
    batches = _batches()
    recorded = {}

    def quantize(name, module, stat):
        recorded[name] = stat["hessian"].clone()
        if modify:
            module.weight.mul_(0.5)

    quantize_blocks_sequentially(
        model,
        batches,
        estimator._target_modules,
        estimator._stat_source_by_target,
        quantize,
    )

    sources = {estimator._stat_source_by_target[name] for name in recorded}
    expected = _input_hessians(model, sorted(sources), batches)
    assert set(recorded) == set(estimator._target_modules)
    for name, hessian in recorded.items():
        torch.testing.assert_close(
            hessian,
            expected[estimator._stat_source_by_target[name]],
            rtol=1e-4,
            atol=1e-5,
        )


def test_sequential_estimator_quantizes_every_wrapped_linear():
    model = _tiny_llama()
    lm_head = model.lm_head.weight.detach().clone()
    TernaryHadamardStep().apply(model, [])
    estimator = TernaryGptqEstimatorConfig().build()
    estimator.prepare(model)

    assert estimator.calibrate_sequential(model, iter(_batches())) is True
    plan = estimator.estimate(None, None, None)

    assert plan.weight_assignments == {}
    for module in estimator._target_modules.values():
        pack_ternary_weight(module.weight, group_size=128)
    torch.testing.assert_close(model.lm_head.weight, lm_head)


def test_non_sequential_gptq_declines_sequential_calibration():
    estimator = GPTQEstimator(group_size=32, nbits=4)
    estimator.prepare(nn.Sequential(nn.Linear(64, 32)))

    assert estimator.calibrate_sequential(nn.Sequential(), iter([])) is False
