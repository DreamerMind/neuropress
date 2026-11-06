import copy

import pytest
import torch
from torch import nn

import neuropress.transforms.weight_only  # noqa: F401
from neuropress.adapter import WeightAssignmentAdapter
from neuropress.calibration.probe import StatisticsRegistry
from neuropress.estimator_configs import (
    GptqEstimatorConfig,
    HqqEstimatorConfig,
    MinMaxZpWeightEstimatorConfig,
)
from neuropress.estimators import GPTQPanelEstimator
from neuropress.plan import QuantizedWeight
from neuropress.qscheme.group import QSchemeGroup
from neuropress.recipe_config import QuantizationRecipeConfig
from neuropress.t2n.model_io import (
    attach_manifest,
    load_quantized_model,
    save_quantized_model,
)
from neuropress.transforms.split_mixed_linear import MixedPrecisionLinear


def estimate(estimator, model, batches):
    estimator.prepare(model)
    registry = StatisticsRegistry()
    registry.register_requests(estimator.declare_statistics())
    registry.collect(model, batches)
    return estimator.estimate(None, registry, None)


@pytest.mark.parametrize("kind", ["gptq", "hqq", "min_max_zp_weight"])
@pytest.mark.parametrize(
    "symmetric,format_name", [(True, "q4_0_packed"), (False, "q4_1_packed")]
)
def test_scalar_estimators_defer_conversion_and_preserve_codes(
    kind, symmetric, format_name, tmp_path
):
    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    original = model[0].weight.detach().clone()
    config_type = {
        "gptq": GptqEstimatorConfig,
        "hqq": HqqEstimatorConfig,
        "min_max_zp_weight": MinMaxZpWeightEstimatorConfig,
    }[kind]
    config = config_type(
        target="linear",
        group_size=32,
        nbits=4,
        symmetric=symmetric,
        weight_transform=format_name,
    )
    recipe = QuantizationRecipeConfig(estimator=config, kernel_lowering=None)
    restored_config = QuantizationRecipeConfig.from_json(recipe.model_dump_json())
    assert restored_config.model_dump_json() == recipe.model_dump_json()
    plan = estimate(
        restored_config.build().param_estimator, model, [torch.randn(2, 64, 64)]
    )
    torch.testing.assert_close(model[0].weight, original, rtol=0, atol=0)
    assignment = plan.weight_assignments["0"]
    assert isinstance(assignment, QuantizedWeight)
    assert assignment.destination == format_name
    WeightAssignmentAdapter().apply(model, plan)
    q = model[0].weight
    torch.testing.assert_close(q.decompress_to_u8(), assignment.codes, rtol=0, atol=0)
    torch.testing.assert_close(
        q.qscheme.scale, assignment.scheme.scale.reshape(-1, 1).half(), rtol=0, atol=0
    )
    attach_manifest(model)
    path = tmp_path / "weights.safetensors"
    save_quantized_model(model, path)
    restored = nn.Sequential(nn.Linear(64, 16, bias=False))
    load_quantized_model(restored, path, kernel_lowering=None)
    torch.testing.assert_close(
        restored[0].weight.decompress_to_u8(), assignment.codes, rtol=0, atol=0
    )


def test_panel_estimator_defers_model_surgery_and_preserves_each_scheme(tmp_path):
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(128, 16, bias=True))
    original = copy.deepcopy(model.state_dict())
    estimator = GPTQPanelEstimator(
        module_classes=(nn.Linear,),
        group_size=32,
        possible_nbits=[2, 4],
        target_bpw=3.5,
    )
    plan = estimate(estimator, model, [torch.randn(4, 64, 128)])
    assert isinstance(model[0], nn.Linear)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    assignment = plan.weight_assignments["0"]
    assert isinstance(assignment.scheme, QSchemeGroup)
    bits = {p.qscheme.nbits for p in assignment.scheme.panels}
    assert bits == {2, 4}
    WeightAssignmentAdapter().apply(model, plan)
    mixed = model[0]
    assert isinstance(mixed, MixedPrecisionLinear)
    offset = 0
    for sub, size in zip(mixed.sub_linears, mixed.split_sizes, strict=True):
        columns = mixed.column_perm[offset : offset + size]
        torch.testing.assert_close(
            sub.weight.decompress_to_u8(), assignment.codes[:, columns], rtol=0, atol=0
        )
        offset += size
    x = torch.randn(2, 128)
    before = model(x)
    attach_manifest(model)
    path = tmp_path / "mixed.safetensors"
    save_quantized_model(model, path)
    restored = nn.Sequential(nn.Linear(128, 16, bias=True))
    load_quantized_model(restored, path, kernel_lowering=None)
    torch.testing.assert_close(restored(x), before, rtol=0, atol=0)


def test_unformatted_root_linear_is_applied_by_adapter():
    model = nn.Linear(64, 16, bias=False)
    estimator = HqqEstimatorConfig(group_size=32).build()
    plan = estimate(estimator, model, [])
    expected = plan.weight_assignments[""].dequantize()
    WeightAssignmentAdapter().apply(model, plan)
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


@pytest.mark.parametrize("format_name", ["q4_0", "q4_0_packed"])
def test_lowered_linear_roundtrip_preserves_codes(format_name):
    from neuropress.nn.np_linear import NPLinear

    torch.manual_seed(12)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    estimator = HqqEstimatorConfig(
        group_size=32, symmetric=True, weight_transform=format_name
    ).build()
    plan = estimate(estimator, model, [])
    WeightAssignmentAdapter().apply(model, plan)
    selected = plan.weight_assignments["0"].codes
    lowered = NPLinear.from_linear(model[0], backend="compile")
    restored = lowered.to_qtensor()
    torch.testing.assert_close(restored.decompress_to_u8(), selected, rtol=0, atol=0)
    inputs = torch.randn(3, 64)
    expected = torch.nn.functional.linear(
        inputs, restored.qscheme.dequantize(selected, torch.float32)
    )
    torch.testing.assert_close(lowered(inputs), expected)
