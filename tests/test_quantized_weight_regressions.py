"""Regressions for plan-based quantized weight conversion and adaptation."""

import math
import tarfile
from unittest.mock import patch

import nnef
import pytest
import torch
from torch import nn
from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant.base import QScalePerGroupF16
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

import neuropress.estimators as estimators_module
import neuropress.transforms.weight_only  # noqa: F401
from neuropress.adapter import WeightAssignmentAdapter
from neuropress.calibration.probe import StatisticsRegistry
from neuropress.estimator_configs import (
    GptqEstimatorConfig,
    HqqEstimatorConfig,
    MinMaxZpWeightEstimatorConfig,
)
from neuropress.estimators import (
    GPTQEstimator,
    GPTQPanelEstimator,
    HQQEstimator,
    MinMaxZpWeightEstimator,
)
from neuropress.eval.size_estimate import collect_estimates
from neuropress.nn.np_embedding import NPEmbedding
from neuropress.nn.np_linear import NPLinear
from neuropress.t2n.model_io import (
    attach_manifest,
    load_quantized_model,
    save_quantized_model,
)
from neuropress.transforms.split_mixed_linear import MixedPrecisionLinear
from neuropress.transforms.weight_only import QTensorScaleZp


def estimate(estimator, model, batches=()):
    estimator.prepare(model)
    registry = StatisticsRegistry()
    registry.register_requests(estimator.declare_statistics())
    registry.collect(model, list(batches))
    return estimator.estimate(None, registry, None)


def minmax(format_name="q4_0_packed", symmetric=True, nbits=4, group_size=32):
    return MinMaxZpWeightEstimatorConfig(
        target="linear",
        nbits=nbits,
        symmetric=symmetric,
        group_size=group_size,
        weight_transform=format_name,
    ).build()


# Packing survives lowering


@pytest.mark.parametrize(
    "format_name,compressed", [("q4_0_packed", True), ("q4_0", False)]
)
def test_unlowered_linear_keeps_its_storage_format(format_name, compressed):
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    WeightAssignmentAdapter().apply(model, estimate(minmax(format_name), model))
    q = model[0].weight
    restored = NPLinear.from_linear(model[0], backend="compile").to_qtensor()
    assert bool(restored.u8_compressors) is compressed
    assert restored.u8_blob.numel() == q.u8_blob.numel()
    torch.testing.assert_close(restored.u8_blob, q.u8_blob, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.decompress_to_u8(), q.decompress_to_u8(), rtol=0, atol=0
    )


def test_unlowered_embedding_keeps_packed_storage():
    model = nn.Sequential(nn.Embedding(16, 64))
    estimator = MinMaxZpWeightEstimatorConfig(
        nbits=4, group_size=32, weight_transform="q4_0_packed"
    ).build()
    WeightAssignmentAdapter().apply(model, estimate(estimator, model))
    q = model[0].weight
    restored = NPEmbedding.from_embedding(model[0], backend="compile").to_qtensor()
    assert restored.u8_compressors
    torch.testing.assert_close(restored.u8_blob, q.u8_blob, rtol=0, atol=0)


def test_lowering_accepts_embeddings_stored_unpacked():
    model = nn.Sequential(nn.Embedding(16, 64))
    estimator = MinMaxZpWeightEstimatorConfig(
        nbits=4, group_size=32, weight_transform="q4_0"
    ).build()
    WeightAssignmentAdapter().apply(model, estimate(estimator, model))
    q = model[0].weight
    lowered = NPEmbedding.from_embedding(model[0], backend="compile")
    assert lowered.u8_blob.shape == (16, 32)
    ids = torch.arange(16)
    torch.testing.assert_close(
        lowered(ids).float(), q.decompress(), rtol=1e-3, atol=1e-3
    )


# Size estimation


def test_size_estimate_resolves_named_formats():
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    estimates, _ = collect_estimates(model, minmax())
    assert estimates[0].quant_bytes == math.ceil(64 * 16 * 4.5 / 8)
    assert "untargeted" not in estimates[0].description


# Conv1D


class Conv1D(nn.Module):
    """Same layout as the transformers GPT-2 Conv1D: weight is [in, out]."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def forward(self, x):
        return x @ self.weight


def test_gptq_conv1d_codes_pair_with_their_scales():
    torch.manual_seed(1)
    model = nn.Sequential(Conv1D(16, 64))
    original = model[0].weight.detach().clone()
    estimator = GPTQEstimator(module_classes=(Conv1D,), group_size=16)
    plan = estimate(estimator, model, [torch.randn(8, 64, 64)])
    decoded = plan.weight_assignments["0"].dequantize()
    assert decoded.shape == original.shape
    assert (decoded - original).abs().mean() < 0.012
    WeightAssignmentAdapter().apply(model, plan)
    torch.testing.assert_close(model[0].weight, decoded, rtol=0, atol=0)


def test_gptq_conv1d_rejects_formats_that_cannot_keep_its_groups():
    model = nn.Sequential(Conv1D(32, 64))
    estimator = GPTQEstimator(
        module_classes=(Conv1D,), group_size=32, weight_transform="q4_0_packed"
    )
    with pytest.raises(ValueError, match="Conv1D"):
        estimator.prepare(model)


def test_gptq_conv1d_legacy_callback_receives_weight_layout():
    torch.manual_seed(2)
    model = nn.Sequential(Conv1D(16, 64))
    seen = {}

    def transform(weight, name):
        seen[name] = weight.shape
        return weight

    estimator = GPTQEstimator(
        module_classes=(Conv1D,), group_size=16, weight_transform=transform
    )
    plan = estimate(estimator, model, [torch.randn(8, 64, 64)])
    WeightAssignmentAdapter().apply(model, plan)
    assert seen == {"0": torch.Size([64, 16])}


# Tied weights


class Tied(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.a = nn.Linear(features, 16, bias=False)
        self.b = nn.Linear(features, 16, bias=False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.a(x) + self.b(x)


TIED_ESTIMATORS = {
    "gptq": lambda fmt: GPTQEstimator(group_size=32, weight_transform=fmt),
    "hqq": lambda fmt: HQQEstimator(
        group_size=32, symmetric=True, num_iter=1, weight_transform=fmt
    ),
    "minmax": lambda fmt: MinMaxZpWeightEstimator(
        module_classes=(nn.Linear,),
        nbits=4,
        symmetric=True,
        group_size=32,
        weight_transform=fmt,
    ),
}


@pytest.mark.parametrize("kind", TIED_ESTIMATORS)
@pytest.mark.parametrize("format_name", ["q4_0_packed", None])
def test_tied_targets_stay_tied_and_quantized(kind, format_name):
    torch.manual_seed(4)
    model = Tied()
    original = model.a.weight.detach().clone()
    plan = estimate(TIED_ESTIMATORS[kind](format_name), model, [torch.randn(4, 32, 64)])
    assert plan.weight_assignments["a"] is plan.weight_assignments["b"]
    WeightAssignmentAdapter().apply(model, plan)
    assert model.a.weight is model.b.weight
    if format_name:
        assert isinstance(model.b.weight, QTensorTractScaleOnly)
    else:
        assert not torch.equal(model.b.weight, original)


def test_tied_qtensor_survives_save_and_load(tmp_path):
    torch.manual_seed(5)
    model = Tied()
    WeightAssignmentAdapter().apply(model, estimate(minmax(), model))
    attach_manifest(model)
    path = tmp_path / "tied.safetensors"
    save_quantized_model(model, path)
    restored = Tied()
    load_quantized_model(restored, path, kernel_lowering=None)
    assert restored.a.weight is restored.b.weight
    x = torch.randn(2, 64)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_tied_mixed_linears_share_branches_and_survive_reload(tmp_path):
    torch.manual_seed(6)
    model = Tied(128)
    estimator = GPTQPanelEstimator(
        module_classes=(nn.Linear,),
        group_size=32,
        possible_nbits=[2, 4],
        target_bpw=3.5,
    )
    plan = estimate(estimator, model, [torch.randn(4, 64, 128)])
    WeightAssignmentAdapter().apply(model, plan)
    assert isinstance(model.b, MixedPrecisionLinear)
    assert model.a.sub_linears is model.b.sub_linears
    attach_manifest(model)
    path = tmp_path / "tied-mixed.safetensors"
    save_quantized_model(model, path)
    restored = Tied(128)
    load_quantized_model(restored, path, kernel_lowering=None)
    assert restored.a.sub_linears is restored.b.sub_linears
    x = torch.randn(2, 128)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_untied_float_sibling_keeps_its_saved_values_on_load(tmp_path):
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(16, 64)
            self.head = nn.Linear(64, 16, bias=False)
            self.head.weight = self.embed.weight

    torch.manual_seed(8)
    model = Mixed()
    original = model.head.weight.detach().clone()
    estimator = MinMaxZpWeightEstimatorConfig(
        nbits=4, group_size=32, weight_transform="q4_0_packed"
    ).build()
    WeightAssignmentAdapter().apply(model, estimate(estimator, model))
    assert model.head.weight is not model.embed.weight
    attach_manifest(model)
    path = tmp_path / "head.safetensors"
    save_quantized_model(model, path)
    restored = Mixed()
    load_quantized_model(restored, path, kernel_lowering=None)
    torch.testing.assert_close(restored.head.weight, original, rtol=0, atol=0)
    assert isinstance(restored.embed.weight, QTensorTractScaleOnly)


# f16 range


def test_minmax_keeps_groups_whose_scale_underflows_f16():
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    with torch.no_grad():
        model[0].weight[:, :32] *= 1e-7
    original = model[0].weight.detach().clone()
    WeightAssignmentAdapter().apply(model, estimate(minmax(), model))
    q = model[0].weight
    assert isinstance(q, QTensorTractScaleOnly)
    assert (q.decompress() - original)[:, :32].abs().max() < 1e-6


def test_minmax_skips_layer_whose_scale_overflows_f16():
    model = nn.Sequential(nn.Linear(64, 16, bias=False), nn.Linear(64, 16, bias=False))
    with torch.no_grad():
        model[0].weight *= 1e8
    plan = estimate(minmax(), model)
    assert list(plan.weight_assignments) == ["1"]


def test_adapter_keeps_float_values_when_format_cannot_hold_a_scale():
    torch.manual_seed(10)
    model = nn.Sequential(nn.Linear(64, 16, bias=False), nn.Linear(64, 16, bias=False))
    with torch.no_grad():
        model[0].weight[:, :32] *= 1e7

    class Parallel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = model[0]
            self.second = model[1]

        def forward(self, inputs):
            return self.first(inputs) + self.second(inputs)

    parallel = Parallel()
    plan = estimate(
        GPTQEstimator(group_size=32, weight_transform="q4_0_packed"),
        parallel,
        [torch.randn(4, 32, 64)],
    )
    selected = plan.weight_assignments["first"].dequantize()
    WeightAssignmentAdapter().apply(parallel, plan)
    assert not isinstance(parallel.first.weight, QTensorTractScaleOnly)
    torch.testing.assert_close(parallel.first.weight, selected, rtol=0, atol=0)
    assert isinstance(parallel.second.weight, QTensorTractScaleOnly)


# Config validation


@pytest.mark.parametrize(
    "config",
    [
        lambda: HqqEstimatorConfig(group_size=32, weight_transform="q4_0_packed"),
        lambda: MinMaxZpWeightEstimatorConfig(
            target="linear", group_size=32, weight_transform="q4_0_packed"
        ),
        lambda: GptqEstimatorConfig(group_size=16, weight_transform="q4_0_packed"),
        lambda: GptqEstimatorConfig(
            group_size=32, nbits=4, weight_transform="q2_0_packed"
        ),
    ],
)
def test_incompatible_format_fails_at_build(config):
    with pytest.raises(ValueError, match="requires"):
        config().build()


# Asymmetric f16 grid


@pytest.mark.parametrize("center", [1.0, 100.0, -100.0, 3000.0])
def test_asymmetric_minmax_decodes_identically_after_storage(center):
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(center + 1e-3 * torch.rand(16, 64))
    original = model[0].weight.detach().clone()
    estimator = minmax("q4_1_packed", symmetric=False)
    plan = estimate(estimator, model)
    planned = plan.weight_assignments["0"].dequantize()
    WeightAssignmentAdapter().apply(model, plan)
    stored = model[0].weight.decompress()
    torch.testing.assert_close(stored, planned, rtol=0, atol=0)
    assert (stored - original).abs().max() <= abs(center) * 1.5e-3


# Panel estimator


def test_panel_estimator_skips_only_the_failed_layer(monkeypatch):
    real = estimators_module._gptq_panel_layer

    def flaky(name, *args, **kwargs):
        if name == "first":
            raise RuntimeError("cholesky failed")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(estimators_module, "_gptq_panel_layer", flaky)

    class Parallel(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(128, 16, bias=False)
            self.second = nn.Linear(128, 16, bias=False)

        def forward(self, x):
            return self.first(x) + self.second(x)

    estimator = GPTQPanelEstimator(
        module_classes=(nn.Linear,),
        group_size=32,
        possible_nbits=[2, 4],
        target_bpw=3.5,
    )
    plan = estimate(estimator, Parallel(), [torch.randn(4, 64, 128)])
    assert list(plan.weight_assignments) == ["second"]


def _mixed_model(group_size=32, features=128, tmp_seed=3):
    torch.manual_seed(tmp_seed)
    model = nn.Sequential(nn.Linear(features, 16, bias=False))
    estimator = GPTQPanelEstimator(
        module_classes=(nn.Linear,),
        group_size=group_size,
        possible_nbits=[2, 4],
        target_bpw=3.5,
    )
    plan = estimate(estimator, model, [torch.randn(4, 64, features)])
    expected = plan.weight_assignments["0"].dequantize()
    WeightAssignmentAdapter().apply(model, plan)
    return model, expected


def test_panel_groups_not_divisible_by_32_apply_and_round_trip(tmp_path):
    model, expected = _mixed_model(group_size=16)
    x = torch.randn(2, 128)
    torch.testing.assert_close(model(x), x @ expected.T, rtol=1e-3, atol=1e-3)
    attach_manifest(model)
    path = tmp_path / "group16.safetensors"
    save_quantized_model(model, path)
    restored = nn.Sequential(nn.Linear(128, 16, bias=False))
    load_quantized_model(restored, path, kernel_lowering=None)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


# NNEF export


def test_mixed_branch_without_tract_format_exports_as_float(tmp_path):
    model, _ = _mixed_model()
    branches = [
        s.weight for s in model[0].sub_linears if isinstance(s.weight, QTensorScaleZp)
    ]
    assert branches
    branches[0].write_in_file(tmp_path, "branch", TractNNEF.latest())
    with (tmp_path / "branch.dat").open("rb") as stream:
        array = nnef.read_tensor(stream)
    torch.testing.assert_close(
        torch.from_numpy(array), branches[0].decompress(), rtol=0, atol=0
    )


def test_mixed_branch_export_survives_reload(tmp_path):
    model, _ = _mixed_model()
    attach_manifest(model)
    path = tmp_path / "mixed.safetensors"
    save_quantized_model(model, path)
    restored = nn.Sequential(nn.Linear(128, 16, bias=False))
    load_quantized_model(restored, path, kernel_lowering=None)
    weight = next(
        s.weight
        for s in restored[0].sub_linears
        if isinstance(s.weight, QTensorScaleZp)
    )
    weight.write_in_file(tmp_path, "reloaded", TractNNEF.latest())
    assert (tmp_path / "reloaded.dat").exists()


def test_mixed_model_with_bias_exports_to_nnef(tmp_path):
    torch.manual_seed(16)
    model = nn.Sequential(nn.Linear(128, 16, bias=True), nn.ReLU()).eval()
    estimator = GPTQPanelEstimator(
        module_classes=(nn.Linear,),
        group_size=32,
        possible_nbits=[2, 4],
        target_bpw=3.5,
    )
    plan = estimate(estimator, model, [torch.randn(4, 64, 128)])
    WeightAssignmentAdapter().apply(model, plan)
    assert not model[0].bias.requires_grad
    target = TractNNEF.latest()
    target.check_io = False
    archive = export_model_to_nnef(
        model,
        args=(torch.randn(1, 128),),
        file_path_export=tmp_path / "mixed.nnef.tgz",
        input_names=["x"],
        output_names=["y"],
        inference_target=target,
    )
    with tarfile.open(archive) as bundle:
        assert sum(n.endswith(".dat") for n in bundle.getnames()) >= 3


def test_standalone_affine_format_still_refuses_nnef(tmp_path):
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    estimator = minmax("q4_1_packed", symmetric=False)
    WeightAssignmentAdapter().apply(model, estimate(estimator, model))
    with pytest.raises(T2NErrorNotImplemented):
        model[0].weight.write_in_file(tmp_path, "weights", TractNNEF.latest())


# Memory


@pytest.mark.parametrize(
    "format_name,symmetric",
    [("q4_0_packed", True), ("q4_1_packed", False)],
)
def test_offloaded_weights_stay_offloaded(format_name, symmetric, tmp_path):
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    offloaded = OffloadedTensor.from_original_tensor(
        model[0].weight.detach().clone(), "weight", tmp_path
    )
    plan = estimate(minmax(format_name, symmetric), model)
    selected = plan.weight_assignments["0"]
    model[0]._parameters["weight"] = offloaded
    WeightAssignmentAdapter().apply(model, plan)
    weight = model[0].weight
    assert isinstance(weight, OffloadedTensor)
    base = weight.to_base_tensor()
    torch.testing.assert_close(base.decompress_to_u8(), selected.codes, rtol=0, atol=0)


@pytest.mark.parametrize(
    "format_name,symmetric",
    [("q4_0_packed", True), ("q4_1_packed", False)],
)
def test_converted_model_can_be_pickled(format_name, symmetric, tmp_path):
    torch.manual_seed(14)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    WeightAssignmentAdapter().apply(
        model, estimate(minmax(format_name, symmetric), model)
    )
    path = tmp_path / "model.pt"
    torch.save(model, path)
    restored = torch.load(path, weights_only=False)
    x = torch.randn(2, 64)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_offloaded_affine_state_excludes_the_float_placeholder(tmp_path):
    torch.manual_seed(15)
    model = nn.Sequential(nn.Linear(6400, 64, bias=False))
    offloaded = OffloadedTensor.from_original_tensor(
        model[0].weight.detach().clone(), "weight", tmp_path
    )
    plan = estimate(minmax("q4_1_packed", False), model)
    model[0]._parameters["weight"] = offloaded
    WeightAssignmentAdapter().apply(model, plan)
    float_bytes = 6400 * 64 * 4
    assert model[0].weight.offload_path.stat().st_size < float_bytes / 4


def test_conversion_never_quantizes_a_placeholder():
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    plan = estimate(minmax(), model)
    with patch.object(
        QScalePerGroupF16, "quantize_as_u8", side_effect=AssertionError("placeholder")
    ):
        WeightAssignmentAdapter().apply(model, plan)
    q = model[0].weight
    assert q.shape == (16, 64)
    assert q.dtype == torch.float32
