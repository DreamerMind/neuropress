"""Regressions for plan-based quantized weight conversion and adaptation."""

import math
import tarfile
from dataclasses import replace
from unittest.mock import patch

import nnef
import pytest
import torch
from torch import nn
from torch_to_nnef import export_model_to_nnef
from torch_to_nnef.exceptions import (
    T2NErrorImpossibleQuantization,
    T2NErrorNotImplemented,
)
from torch_to_nnef.inference_target import TractNNEF
from torch_to_nnef.tensor.offload import OffloadedTensor
from torch_to_nnef.tensor.quant.base import QScalePerGroupF16
from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

import neuropress.estimators as estimators_module
import neuropress.transforms.weight_only  # noqa: F401
from neuropress.adapter import WeightAssignmentAdapter
from neuropress.calibration.probe import StatisticsRegistry
from neuropress.estimator_configs import (
    WEIGHT_TRANSFORM_REGISTRY,
    GptqEstimatorConfig,
    GptqPanelEstimatorConfig,
    HqqEstimatorConfig,
    MinMaxZpWeightEstimatorConfig,
)
from neuropress.estimators import (
    GPTQEstimator,
    GPTQPanelEstimator,
    HQQEstimator,
    MinMaxZpWeightEstimator,
    collect_target_modules,
    collect_tied_aliases,
)
from neuropress.eval.size_estimate import _parse_specs, collect_estimates
from neuropress.nn.np_embedding import NPEmbedding
from neuropress.nn.np_linear import NPLinear, _kernel_blob, _pack_u8
from neuropress.plan import QuantizationPlan, QuantizedWeight
from neuropress.qscheme.group import PanelSpec, QSchemeGroup, TensorSlice
from neuropress.qscheme.zp_scale import ZpScaleScheme
from neuropress.t2n.model_io import (
    attach_manifest,
    load_quantized_model,
    save_quantized_model,
)
from neuropress.t2n.packing import BitPacking
from neuropress.t2n.qtensors import _SERDE_REGISTRY, qtensor_shape
from neuropress.transforms.split_mixed_linear import MixedPrecisionLinear
from neuropress.transforms.weight_only import QTensorScaleZp, _affine_from_assignment


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


# ---------------------------------------------------------------------------
# Second review round
# ---------------------------------------------------------------------------


class _ParallelLinears(nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(32, 8, bias=False) for _ in range(count))

    def forward(self, x):
        return sum(layer(x) for layer in self.layers)


def _offload_all(model, directory):
    for index, layer in enumerate(model.layers):
        layer.weight = nn.Parameter(
            OffloadedTensor.from_original_tensor(
                layer.weight.data, f"w{index}", offload_dir=directory
            ),
            requires_grad=False,
        )


def test_offloaded_layers_are_not_tied_by_reused_addresses(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = _ParallelLinears().eval()
    originals = [layer.weight.detach().clone() for layer in model.layers]
    _offload_all(model, tmp_path)
    # Reloads land in transient buffers, so an address says nothing about identity.
    monkeypatch.setattr(OffloadedTensor, "data_ptr", lambda self: 1, raising=False)

    assert len(collect_target_modules(model, (nn.Linear,))) == 3
    assert collect_tied_aliases(model, (nn.Linear,)) == {}

    estimator = minmax()
    plan = estimate(estimator, model)
    WeightAssignmentAdapter().apply(model, plan)
    for layer, original in zip(model.layers, originals, strict=True):
        assert (layer.weight.to_base_tensor().decompress() - original).abs().max() < 0.1


def test_tied_offloaded_layers_stay_tied(tmp_path):
    model = _ParallelLinears(2).eval()
    _offload_all(model, tmp_path)
    model.layers[1].weight = model.layers[0].weight
    assert collect_tied_aliases(model, (nn.Linear,)) == {"layers.0": ["layers.1"]}


def test_meta_weights_are_not_tied():
    with torch.device("meta"):
        model = _ParallelLinears().eval()
    assert len(collect_target_modules(model, (nn.Linear,))) == 3


def test_conv1d_per_channel_fallback_keeps_a_uniform_scheme():
    torch.manual_seed(0)
    model = nn.Sequential(Conv1D(10, 64)).eval()
    original = model[0].weight.detach().clone()
    plan = estimate(
        GPTQEstimator(module_classes=(Conv1D,), group_size=32),
        model,
        [torch.randn(8, 16, 64)],
    )
    decoded = plan.weight_assignments["0"].dequantize()
    assert decoded.shape == original.shape
    assert (decoded - original).abs().max() < 0.2


@pytest.mark.parametrize("bits", [[1, 4], [3, 4]])
def test_panel_config_rejects_widths_the_formats_cannot_store(bits):
    with pytest.raises(ValueError, match="support 2, 4 and 8 bits"):
        GptqPanelEstimatorConfig(possible_nbits=bits).build()


def test_affine_format_reports_unsupported_widths():
    scheme = ZpScaleScheme(
        torch.full((16, 1, 2, 1), 4.0),
        torch.ones(16, 1, 2, 1),
        3,
        [16, 1, 2, 32],
        [16, 1, 2, 1],
    )
    codes = torch.zeros(16, 64, dtype=torch.uint8)
    with pytest.raises(ValueError, match="do not support 3 bits"):
        _affine_from_assignment(
            QuantizedWeight(codes, scheme, torch.float32), "x", nbits=3
        )


def test_unpackable_mixed_branch_stays_float():
    # 2-bit packing needs a multiple of 4 values; 5 rows of 6 columns has 30.
    scheme = ZpScaleScheme(
        torch.full((5, 1, 1, 1), 2.0),
        torch.ones(5, 1, 1, 1),
        2,
        [5, 1, 1, 6],
        [5, 1, 1, 1],
    )
    four_bits = ZpScaleScheme(
        torch.full((5, 1, 1, 1), 8.0),
        torch.ones(5, 1, 1, 1),
        4,
        [5, 1, 1, 32],
        [5, 1, 1, 1],
    )
    group = QSchemeGroup(
        panels=[
            PanelSpec(region=TensorSlice(dim=1, start=0, end=6), qscheme=scheme),
            PanelSpec(region=TensorSlice(dim=1, start=6, end=38), qscheme=four_bits),
        ]
    )
    model = nn.Sequential(nn.Linear(38, 5, bias=False)).eval()
    codes = torch.full((5, 38), 3, dtype=torch.uint8)
    assignment = QuantizedWeight(codes, group, torch.float32)
    plan = QuantizationPlan(weight_assignments={"0": assignment})
    WeightAssignmentAdapter().apply(model, plan)
    assert isinstance(model[0], MixedPrecisionLinear)
    assert "0.sub_linears.0" in plan.kept_float or "0.sub_linears.1" in plan.kept_float
    assert torch.isfinite(model[0](torch.ones(1, 38))).all()


def test_packing_refuses_shapes_it_cannot_restore():
    # 48 values pack evenly, but a last dimension of 6 is not divisible by 4,
    # so the blob would be flat and decode flat.
    scale = torch.ones(8, 1, 1, 1)
    scheme = ZpScaleScheme(
        torch.full_like(scale, 2.0), scale, 2, [8, 1, 1, 6], list(scale.shape)
    )
    codes = torch.ones(8, 6, dtype=torch.uint8)
    with pytest.raises(ValueError, match="cannot restore weight shape"):
        _affine_from_assignment(
            QuantizedWeight(codes, scheme, torch.float32), "x", nbits=2, block_size=6
        )


def test_format_rejection_keeps_the_other_layers(caplog):
    torch.manual_seed(0)

    class Convs(nn.Module):
        def __init__(self):
            super().__init__()
            self.rejected = nn.Conv2d(64, 8, 1, bias=False)
            self.kept = nn.Conv2d(64, 8, 1, bias=False)

        def forward(self, x):
            return self.rejected(x) + self.kept(x)

    model = Convs().eval()
    plan = estimate(
        GptqEstimatorConfig(target="conv2d", weight_transform="q4_0_packed").build(),
        model,
        [torch.randn(8, 64, 4, 4)],
    )
    with caplog.at_level("WARNING", logger="neuropress.adapter"):
        WeightAssignmentAdapter().apply(model, plan)
    assert set(plan.kept_float) == {"rejected", "kept"}
    assert torch.isfinite(model.rejected.weight).all()
    assert "2 selected layers stayed float" in caplog.text


def test_minmax_skips_only_layers_it_cannot_group():
    class Layers(nn.Module):
        def __init__(self):
            super().__init__()
            self.compatible = nn.Embedding(32, 32)
            self.incompatible = nn.Embedding(33, 32)

    model = Layers()
    plan = estimate(
        MinMaxZpWeightEstimatorConfig(
            target="embedding",
            nbits=4,
            symmetric=True,
            group_size=64,
            weight_transform="q4_0_packed",
        ).build(),
        model,
    )
    assert set(plan.weight_assignments) == {"compatible"}
    assert "group size 64" in plan.kept_float["incompatible"]


def test_tied_mixed_linear_and_embedding_keep_valid_weights():
    plan_source = _mixed_plan()
    assignment = plan_source.weight_assignments["0"]
    for names in (("linear", "emb"), ("emb", "linear")):
        rows, cols = assignment.codes.shape
        holder = nn.ModuleDict(
            {
                "linear": nn.Linear(cols, rows, bias=False),
                "emb": nn.Embedding(rows, cols),
            }
        )
        plan = QuantizationPlan(weight_assignments=dict.fromkeys(names, assignment))
        WeightAssignmentAdapter().apply(holder, plan)
        assert isinstance(holder["linear"], MixedPrecisionLinear)
        assert isinstance(holder["emb"].weight, torch.Tensor)
        assert holder["emb"].weight.shape == (rows, cols)


def _mixed_plan(bits=(2, 4)):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(128, 16, bias=False)).eval()
    estimator = GPTQPanelEstimator(
        module_classes=(nn.Linear,),
        group_size=32,
        possible_nbits=list(bits),
        target_bpw=3.0,
    )
    return estimate(estimator, model, [torch.randn(8, 16, 128)])


def _stored_bits_per_weight(q):
    total = q.u8_blob.numel() + 2 * q.qscheme.scale.numel()
    zero_point = getattr(q.qscheme, "zero_point", None)
    if zero_point is not None:
        total += 2 * zero_point.numel()
    return 8 * total / math.prod(qtensor_shape(q))


def test_symmetric_affine_storage_matches_declared_size():
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    plan = estimate(minmax("q2_0_packed", symmetric=True, nbits=2), model)
    WeightAssignmentAdapter().apply(model, plan)
    q = model[0].weight
    assert q.qscheme.zero_point.numel() == 1
    assert _stored_bits_per_weight(q) == pytest.approx(
        WEIGHT_TRANSFORM_REGISTRY["q2_0_packed"].storage_bits, abs=0.02
    )


def test_asymmetric_affine_size_estimate_counts_its_offsets():
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    plan = estimate(minmax("q2_0_packed", symmetric=False, nbits=2), model)
    WeightAssignmentAdapter().apply(model, plan)
    assert _stored_bits_per_weight(model[0].weight) == pytest.approx(3.0, abs=0.01)
    specs = _parse_specs(minmax("q2_0_packed", symmetric=False, nbits=2))
    assert {spec.storage_bits for spec in specs} == {3.0}


@pytest.mark.parametrize(
    "group_size, symmetric, expected",
    [(32, True, 0.5), (64, True, 0.5), (16, True, 1.0), (32, False, 1.0)],
)
def test_panel_size_estimate_counts_stored_metadata(group_size, symmetric, expected):
    estimator = GPTQPanelEstimator(
        group_size=group_size, target_bpw=3.0, symmetric=symmetric
    )
    specs = _parse_specs(estimator)
    assert specs[0].storage_bits == pytest.approx(3.0 + expected)


def test_constant_zero_point_lowers_and_persists(tmp_path):
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(64, 16, bias=False)).eval()
    plan = estimate(minmax("q4_1_packed", symmetric=True), model)
    WeightAssignmentAdapter().apply(model, plan)
    q = model[0].weight
    assert q.qscheme.zero_point.numel() == 1
    x = torch.randn(4, 64)
    expected = model(x)

    lowered = NPLinear.from_linear(model[0], backend="compile")
    assert lowered.zero_point.numel() == q.qscheme.scale.numel()
    torch.testing.assert_close(lowered(x), expected, rtol=1e-3, atol=1e-3)
    restored = lowered.to_qtensor()
    assert restored.qscheme.zero_point.numel() == 1

    serde = _SERDE_REGISTRY["QTensorScaleZp"]
    rebuilt = serde.from_parts(serde.to_tensors(q), serde.to_metadata(q))
    torch.testing.assert_close(rebuilt.decompress(), q.decompress(), rtol=0, atol=0)


def test_underflowed_affine_blocks_decode_to_zero():
    scale = torch.tensor([1e-12, 1.0]).reshape(-1, 1, 1, 1)
    scheme = ZpScaleScheme(
        torch.full_like(scale, 2.0), scale, 4, [2, 1, 1, 32], list(scale.shape)
    )
    codes = torch.full((2, 32), 15, dtype=torch.uint8)
    q = _affine_from_assignment(
        QuantizedWeight(codes, scheme, torch.float32), "x", nbits=4
    )
    decoded = q.decompress()
    assert torch.all(decoded[0] == 0)
    assert torch.all(decoded[1] == 13)


def test_affine_format_rejects_codes_decoding_outside_f16():
    scale = torch.full((1, 1, 1, 1), 10000.0)
    scheme = ZpScaleScheme(
        torch.full_like(scale, 8.0), scale, 4, [1, 1, 1, 32], list(scale.shape)
    )
    codes = torch.zeros(1, 32, dtype=torch.uint8)
    with pytest.raises(T2NErrorImpossibleQuantization, match="decode outside"):
        _affine_from_assignment(
            QuantizedWeight(codes, scheme, torch.float32), "x", nbits=4
        )


def test_codes_follow_the_scheme_device_only_when_needed():
    scheme = ZpScaleScheme(
        torch.full((1, 1, 1, 1), 8.0),
        torch.ones(1, 1, 1, 1),
        4,
        [1, 1, 1, 32],
        [1, 1, 1, 1],
    )
    weight = QuantizedWeight(
        torch.zeros(1, 32, dtype=torch.uint8), scheme, torch.float32
    )
    assert weight.on_scheme_device() is weight
    moved = replace(
        weight,
        scheme=scheme.__class__(
            scheme.zero_point.to("meta"),
            scheme.scale.to("meta"),
            4,
            [1, 1, 1, 32],
            [1, 1, 1, 1],
        ),
    )
    assert moved.on_scheme_device().codes.is_meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_plan_keeps_codes_on_cpu_and_adapter_restores_the_device():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 16, bias=False)).eval().cuda()
    plan = estimate(
        GPTQEstimator(group_size=32, weight_transform="q4_0_packed"),
        model,
        [torch.randn(8, 16, 64, device="cuda")],
    )
    assert plan.weight_assignments["0"].codes.device.type == "cpu"
    WeightAssignmentAdapter().apply(model, plan)
    assert model[0].weight.u8_blob.device.type == "cuda"


@pytest.mark.parametrize("format_name", ["q4_0_packed", "q4_1_packed"])
def test_persistence_does_not_decompress_codes(format_name):
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    plan = estimate(minmax(format_name, symmetric=format_name == "q4_0_packed"), model)
    WeightAssignmentAdapter().apply(model, plan)
    q = model[0].weight
    serde = _SERDE_REGISTRY[type(q).__name__]
    with patch.object(BitPacking, "decompress", side_effect=AssertionError):
        metadata = serde.to_metadata(q)
        rebuilt = serde.from_parts(serde.to_tensors(q), metadata)
    assert qtensor_shape(rebuilt) == (16, 64)
    assert metadata["shape"] == "[16, 64]"


@pytest.mark.parametrize("format_name", ["q4_0_packed", "q4_0"])
def test_lowering_reuses_a_blob_in_the_kernel_layout(format_name):
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(64, 16, bias=False))
    plan = estimate(minmax(format_name), model)
    WeightAssignmentAdapter().apply(model, plan)
    q = model[0].weight
    expected = _pack_u8(q.decompress_to_u8(), 4)
    torch.testing.assert_close(_kernel_blob(q), expected, rtol=0, atol=0)
    if format_name == "q4_0_packed":
        with patch.object(BitPacking, "decompress", side_effect=AssertionError):
            assert _kernel_blob(q) is q.u8_blob


# Tied modules with different formats, lowering round trips and devices


class TiedEmbeddingHead(nn.Module):
    def __init__(self, vocab=64, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, ids):
        return self.head(self.embed(ids))


@pytest.mark.parametrize("head_format", ["q4_0_packed", "q4_1_packed"])
def test_tied_embedding_and_head_with_different_formats_both_convert(head_format):
    torch.manual_seed(16)
    model = TiedEmbeddingHead()
    embed_plan = estimate(
        MinMaxZpWeightEstimatorConfig(
            target="embedding",
            nbits=4,
            symmetric=True,
            group_size=32,
            weight_transform="q4_0_packed",
        ).build(),
        model,
    )
    head_plan = estimate(
        minmax(head_format, symmetric=head_format == "q4_0_packed"), model
    )
    plan = QuantizationPlan(
        weight_assignments={
            **embed_plan.weight_assignments,
            **head_plan.weight_assignments,
        }
    )
    WeightAssignmentAdapter().apply(model, plan)
    assert model.embed.weight is not model.head.weight
    for name, module in (("embed", model.embed), ("head", model.head)):
        torch.testing.assert_close(
            module.weight.decompress(),
            plan.weight_assignments[name].dequantize(),
            rtol=0,
            atol=0,
        )
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 64)


class PlusOneRMSNorm(nn.Module):
    """Gemma-style norm: scales by ``1 + weight``."""

    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden))
        self.eps = eps

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * (1.0 + self.weight)


def _lower_norms(model):
    """Fuse model[1] as linearize_rms_norms does, without needing CUDA."""
    from neuropress.nn.np_rms_norm import NPRMSNorm

    norm = model[1]
    model[1] = NPRMSNorm(1.0 + norm.weight.detach(), eps=norm.eps, source=norm)


def test_norms_of_a_cpu_model_are_not_fused():
    from neuropress.nn.np_rms_norm import linearize_rms_norms

    model = nn.Sequential(nn.Linear(32, 32), PlusOneRMSNorm(32))
    with patch("neuropress.nn.backends.is_cuda_jit_available", return_value=True):
        assert linearize_rms_norms(model) == 0
    assert isinstance(model[1], PlusOneRMSNorm)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_cuda_norms_unlower_to_the_original_module():
    from neuropress.kernel_lowering import unlower_model
    from neuropress.nn.np_rms_norm import NPRMSNorm, linearize_rms_norms

    torch.manual_seed(20)
    model = nn.Sequential(nn.Linear(32, 32), PlusOneRMSNorm(32)).cuda()
    original = model[1].weight.detach().clone()
    x = torch.randn(4, 32, device="cuda")
    expected = model(x)
    with patch("neuropress.nn.backends.is_cuda_jit_available", return_value=True):
        assert linearize_rms_norms(model) == 1
    assert isinstance(model[1], NPRMSNorm)
    torch.testing.assert_close(model(x), expected, rtol=1e-4, atol=1e-4)
    unlower_model(model)
    assert isinstance(model[1], PlusOneRMSNorm)
    assert torch.equal(model[1].weight, original)


def test_unlowering_restores_plus_one_norms_exactly():
    from neuropress.kernel_lowering import is_lowered, unlower_model

    torch.manual_seed(17)
    model = nn.Sequential(nn.Linear(32, 32), PlusOneRMSNorm(32))
    original = model[1].weight.detach().clone()
    _lower_norms(model)
    assert is_lowered(model)
    unlower_model(model)
    assert isinstance(model[1], PlusOneRMSNorm)
    assert torch.equal(model[1].weight, original)


def test_saving_a_lowered_plus_one_norm_keeps_its_weight(tmp_path):
    torch.manual_seed(18)
    model = nn.Sequential(nn.Linear(32, 32), PlusOneRMSNorm(32))
    original = model[1].weight.detach().clone()
    WeightAssignmentAdapter().apply(model, estimate(minmax(), model))
    _lower_norms(model)
    path = tmp_path / "norm.safetensors"
    save_quantized_model(model, path)
    restored = nn.Sequential(nn.Linear(32, 32), PlusOneRMSNorm(32))
    load_quantized_model(restored, path, kernel_lowering=None)
    assert torch.equal(restored[1].weight, original)


def test_mixed_linear_reports_features_with_lowered_branches():
    model = nn.Sequential(nn.Linear(128, 16, bias=False))
    WeightAssignmentAdapter().apply(model, _mixed_plan())
    mixed = model[0]

    class Lowered(nn.Module):
        in_features = 0
        out_features = 16

    mixed.sub_linears[0] = Lowered()
    assert mixed.out_features == 16
    assert "out_features=16" in repr(mixed)


def test_tied_offloaded_modules_with_different_formats_stay_distinct(tmp_path):
    torch.manual_seed(21)
    model = TiedEmbeddingHead()
    embed_plan = estimate(
        MinMaxZpWeightEstimatorConfig(
            target="embedding",
            nbits=4,
            symmetric=True,
            group_size=32,
            weight_transform="q4_0_packed",
        ).build(),
        model,
    )
    head_plan = estimate(minmax("q4_1_packed", symmetric=False), model)
    shared = nn.Parameter(
        OffloadedTensor.from_original_tensor(
            model.embed.weight.detach().clone(), "shared", tmp_path
        ),
        requires_grad=False,
    )
    model.embed._parameters["weight"] = shared
    model.head._parameters["weight"] = shared
    plan = QuantizationPlan(
        weight_assignments={
            **embed_plan.weight_assignments,
            **head_plan.weight_assignments,
        }
    )
    WeightAssignmentAdapter().apply(model, plan)
    assert [name for name, _ in model.named_parameters()] == [
        "embed.weight",
        "head.weight",
    ]
    for name, module in (("embed", model.embed), ("head", model.head)):
        torch.testing.assert_close(
            module.weight.to_base_tensor().decompress(),
            plan.weight_assignments[name].dequantize(),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mixed", [False, True])
def test_checkpoint_loads_onto_the_model_device(mixed, tmp_path):
    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(128, 16, bias=False)).cuda()
    if mixed:
        plan = estimate(
            GPTQPanelEstimator(
                module_classes=(nn.Linear,),
                group_size=32,
                possible_nbits=[2, 4],
                target_bpw=3.0,
            ),
            model,
            [torch.randn(8, 16, 128, device="cuda")],
        )
    else:
        plan = estimate(minmax(), model)
    WeightAssignmentAdapter().apply(model, plan)
    attach_manifest(model)
    path = tmp_path / "device.safetensors"
    save_quantized_model(model, path)
    restored = nn.Sequential(nn.Linear(128, 16, bias=False)).cuda()
    load_quantized_model(restored, path, kernel_lowering=None)
    for name, tensor in list(restored.named_parameters()) + list(
        restored.named_buffers()
    ):
        assert tensor.device.type == "cuda", name
    x = torch.randn(2, 128, device="cuda")
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize("format_name", ["q4_0_packed", "q4_1_packed"])
def test_lowered_embedding_returns_the_replaced_dtype(format_name):
    torch.manual_seed(22)
    model = nn.Sequential(nn.Embedding(16, 64))
    estimator = MinMaxZpWeightEstimatorConfig(
        target="embedding",
        nbits=4,
        symmetric=format_name == "q4_0_packed",
        group_size=32,
        weight_transform=format_name,
    ).build()
    WeightAssignmentAdapter().apply(model, estimate(estimator, model))
    expected = model[0].weight.decompress()
    lowered = NPEmbedding.from_embedding(model[0], backend="compile")
    ids = torch.arange(16)
    out = lowered(ids)
    assert out.dtype == torch.float32
    # The kernels decode through the f16 scale dtype.
    torch.testing.assert_close(out, expected[ids], rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_vq_gptq_correction_leaves_the_callers_hessian_and_weight_untouched():
    from neuropress.methods.vq_gptq import VQGPTQCorrector

    torch.manual_seed(5)
    module = nn.Linear(16, 8, bias=False)
    x = torch.randn(64, 16, device="cuda")
    x[:, :2] = 0  # dead columns: the corrector patches their diagonal
    full_h = x.t() @ x
    saved = full_h.clone()
    # A float32 weight already on the device is reshaped without a copy.
    module.to("cuda")
    weight = module.weight.detach().clone()
    centroids = torch.randn(16, 4, device="cuda")
    VQGPTQCorrector(vector_len=4).correct(
        "layer", module, centroids, full_h, torch.device("cuda")
    )
    torch.testing.assert_close(full_h, saved, rtol=0, atol=0)
    torch.testing.assert_close(module.weight, weight, rtol=0, atol=0)


def test_vq_gptq_correction_warns_once_when_it_cannot_run(caplog, monkeypatch):
    from neuropress.methods import vq_gptq
    from neuropress.methods.vq_gptq import VQGPTQCorrector

    monkeypatch.setattr(vq_gptq, "_WARNED_DEVICE_TYPES", set())
    module = nn.Linear(16, 8, bias=False)
    centroids = torch.randn(16, 4)
    with caplog.at_level("WARNING", logger="neuropress.methods.vq_gptq"):
        # Block calibration builds a fresh corrector for every stage.
        for _ in range(3):
            assert (
                VQGPTQCorrector(vector_len=4).correct(
                    "layer", module, centroids, torch.eye(16), torch.device("cpu")
                )
                is None
            )
    assert sum("needs a CUDA" in r.getMessage() for r in caplog.records) == 1
