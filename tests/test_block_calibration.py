"""Block execution, composable paired statistics, and teacher reconstruction."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from neuropress.block_pipeline import BlockQuantizationPipeline
from neuropress.calibration.probe import (
    COLLECTOR_REGISTRY,
    PairedCollector,
    StatisticRequest,
    StatisticsRegistry,
)
from neuropress.calibration.probe.reconstruction import reconstruct_linear_weight
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe_config import QuantizationRecipeConfig


def recipe(teacher=False):
    return QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "gptq",
                "group_size": 32,
                "nbits": 4,
                "symmetric": True,
                "weight_transform": "q4_0",
                "reconstruction": "teacher" if teacher else "self",
            },
            "kernel_lowering": None,
        }
    )


def test_teacher_statistics_match_ridge_objective_without_mutation():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(32, 16, bias=True)).eval()
    weight = model[0].weight.detach().clone()
    x = torch.randn(3, 24, 32)
    student = x * torch.linspace(0.7, 1.3, 32)
    registry = StatisticsRegistry()
    registry.register_requests(
        [
            StatisticRequest(target_key="0", stat_type=s)
            for s in ["hessian", "linear_reconstruction"]
        ]
    )
    registry.collect(model, [student], reference_data=[x])
    h = registry.get(StatisticRequest(target_key="0", stat_type="hessian"))["hessian"]
    cross = registry.get(
        StatisticRequest(target_key="0", stat_type="linear_reconstruction")
    )["cross"]
    new = reconstruct_linear_weight(weight, h, cross, 0.01)
    flat = student.flatten(0, 1)
    y = x.flatten(0, 1) @ weight.t()
    ridge = 0.01 * h.diag().mean()
    expected = torch.linalg.solve(
        2 * flat.t() @ flat + ridge * torch.eye(32),
        2 * flat.t() @ y + ridge * weight.t(),
    ).t()
    torch.testing.assert_close(new, expected, rtol=2e-5, atol=2e-6)
    assert (flat @ new.t() - y).square().sum() < (flat @ weight.t() - y).square().sum()
    torch.testing.assert_close(model[0].weight, weight, rtol=0, atol=0)


def test_teacher_gptq_improves_known_upstream_scale_error():
    torch.manual_seed(5)
    original = nn.Sequential(nn.Linear(32, 16, bias=False)).eval()
    x = torch.randn(4, 64, 32)
    student = x * 1.5
    ordinary, teacher = deepcopy(original), deepcopy(original)
    for model, use_teacher in [(ordinary, False), (teacher, True)]:
        pipeline = QuantizationPipeline(recipe(use_teacher).build())
        pipeline.run(model, [student], reference_data=[x], estimation_device="cpu")
        assert not pipeline.last_plan.kept_float
    with torch.no_grad():
        target = original(x)
        assert (teacher(student) - target).norm() < (
            ordinary(student) - target
        ).norm() * 0.3


def test_self_and_teacher_match_when_streams_are_identical():
    torch.manual_seed(6)
    original = nn.Sequential(nn.Linear(32, 16, bias=False)).eval()
    x = torch.randn(4, 32)
    models = [deepcopy(original), deepcopy(original)]
    for model, use_teacher in zip(models, [False, True], strict=True):
        QuantizationPipeline(recipe(use_teacher).build()).run(
            model, [x], reference_data=[x]
        )
    torch.testing.assert_close(
        models[0][0].weight.decompress_to_u8(),
        models[1][0].weight.decompress_to_u8(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("source", ["float", "sequential"])
def test_block_execution_matches_explicit_schedule_and_resume(source, tmp_path):
    torch.manual_seed(11)
    blocks = [
        nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU()).eval() for _ in range(2)
    ]
    inputs = [torch.randn(2, 12, 32)]
    originals = deepcopy(blocks)
    runner = BlockQuantizationPipeline(
        recipe(), input_source=source, checkpoint_dir=tmp_path, run_id="toy-v1"
    )
    actual = list(runner.run(enumerate_names(blocks), inputs))
    reference, student = inputs, inputs
    for expected, result in zip(originals, actual, strict=True):
        with torch.no_grad():
            reference_outputs = [expected(x) for x in reference]
        QuantizationPipeline(recipe().build()).run(
            expected, reference if source == "float" else student
        )
        with torch.no_grad():
            student = [expected(x) for x in student]
        reference = reference_outputs
        torch.testing.assert_close(result.student_outputs[0], student[0])
        torch.testing.assert_close(result.reference_outputs[0], reference[0])
    torch.manual_seed(11)
    fresh = [
        nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU()).eval() for _ in range(2)
    ]
    resumed = list(runner.run(enumerate_names(fresh), inputs))
    assert all(result.resumed for result in resumed)
    for expected, result in zip(actual, resumed, strict=True):
        torch.testing.assert_close(
            result.student_outputs[0], expected.student_outputs[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            result.model[0].weight.decompress_to_u8(),
            expected.model[0].weight.decompress_to_u8(),
            rtol=0,
            atol=0,
        )
    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        runner.run_block(
            "0",
            nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU()).eval(),
            inputs,
            inputs,
        )


def enumerate_names(blocks):
    return ((str(i), block) for i, block in enumerate(blocks))


@pytest.mark.parametrize("failure", ["missing", "length", "shape"])
def test_pairing_failures_remove_hooks(failure):
    model = nn.Sequential(nn.Linear(32, 16)).eval()
    registry = StatisticsRegistry()
    registry.register_requests(
        [StatisticRequest(target_key="0", stat_type="linear_reconstruction")]
    )
    x = torch.randn(2, 32)
    refs = (
        None
        if failure == "missing"
        else ([] if failure == "length" else [torch.randn(3, 32)])
    )
    with pytest.raises(ValueError):
        registry.collect(model, [x], reference_data=refs)
    assert not model[0]._forward_hooks


@pytest.mark.parametrize("estimation_device", [None, "cpu"])
def test_composite_recipe_roundtrip_and_mixed_collection(estimation_device):
    config = QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "composite",
                "methods": [
                    {
                        "type": "gptq",
                        "target": "linear",
                        "group_size": 32,
                        "weight_transform": "q4_0",
                        "reconstruction": "teacher",
                        "reconstruction_damp_percent": 0.03,
                    },
                    {"type": "min_max_zp_weight", "target": "embedding", "nbits": 8},
                ],
            },
            "kernel_lowering": None,
        }
    )
    restored = QuantizationRecipeConfig.from_json(config.model_dump_json())
    assert restored.model_dump() == config.model_dump()
    model = nn.Sequential(nn.Embedding(128, 32), nn.Linear(32, 16)).eval()
    pipeline = QuantizationPipeline(restored.build())
    ids = torch.randint(0, 128, (2, 16))
    pipeline.run(
        model, [ids], reference_data=[ids], estimation_device=estimation_device
    )
    assert not pipeline.last_plan.kept_float
    assert set(pipeline.last_plan.weight_assignments) == {"0", "1"}
    assert hasattr(model[1].weight, "qscheme")


def test_future_paired_collector_composes_with_existing_hessian(monkeypatch):
    class DifferenceCollector(PairedCollector):
        def __init__(self):
            self.value = 0.0

        def update_pair(self, inputs, output, reference):
            self.value += float((output - reference).square().sum())

        def result(self):
            return {"squared_error": self.value}

    monkeypatch.setitem(COLLECTOR_REGISTRY, "future_objective", DifferenceCollector)
    model = nn.Sequential(nn.Linear(32, 16)).eval()
    x = torch.randn(2, 32)
    registry = StatisticsRegistry()
    requests = [
        StatisticRequest(target_key="0", stat_type=s)
        for s in ["hessian", "future_objective"]
    ]
    registry.register_requests(requests + requests)
    registry.collect(model, [x * 2], reference_data=[x])
    assert registry.get(requests[0])["hessian"] is not None
    with torch.no_grad():
        expected = (model(x * 2) - model(x)).square().sum().item()
    assert registry.get(requests[1])["squared_error"] == pytest.approx(expected)


def test_float_block_statistics_match_whole_model_collection():
    torch.manual_seed(14)
    original = nn.Sequential(
        nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU()),
        nn.Sequential(nn.Linear(32, 32, bias=False), nn.ReLU()),
    ).eval()
    whole = deepcopy(original)
    batches = [torch.randn(2, 16, 32) for _ in range(3)]
    QuantizationPipeline(recipe().build()).run(whole, batches)
    blocked = list(
        BlockQuantizationPipeline(recipe(), input_source="float").run(
            enumerate_names(original), batches
        )
    )
    for index, result in enumerate(blocked):
        torch.testing.assert_close(
            result.model[0].weight.decompress_to_u8(),
            whole[index][0].weight.decompress_to_u8(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.model[0].weight.qscheme.scale,
            whole[index][0].weight.qscheme.scale,
            rtol=0,
            atol=0,
        )


def test_reference_model_and_repeated_invocations_are_aligned():
    class Repeated(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, x):
            return self.linear(self.linear(x))

    torch.manual_seed(8)
    teacher = Repeated().eval()
    student = deepcopy(teacher)
    with torch.no_grad():
        student.linear.weight.mul_(0.8)
    registry = StatisticsRegistry()
    request = StatisticRequest(target_key="linear", stat_type="linear_reconstruction")
    registry.register_requests([request])
    registry.collect(
        student,
        [torch.randn(2, 32)],
        reference_data=[torch.randn(2, 32)],
        reference_model=teacher,
    )
    assert registry.get(request)["nsamples"] == 2
    assert not student.linear._forward_hooks
    assert not teacher.linear._forward_hooks


@pytest.mark.parametrize("alias", [False, True])
def test_block_runner_rejects_cross_block_shared_weights(alias):
    first = nn.Sequential(nn.Linear(32, 32, bias=False)).eval()
    second = nn.Sequential(nn.Linear(32, 32, bias=False)).eval()
    second[0].weight = (
        nn.Parameter(first[0].weight.detach().view_as(first[0].weight))
        if alias
        else first[0].weight
    )
    runner = BlockQuantizationPipeline(recipe())
    with pytest.raises(ValueError, match="share parameter storage"):
        list(runner.run(enumerate_names([first, second]), [torch.randn(2, 32)]))


@pytest.mark.parametrize("teacher", [False, True])
def test_estimators_do_not_mutate_shared_statistics(teacher):
    from neuropress.graph.graph import QGraph

    model = nn.Sequential(nn.Linear(32, 16, bias=False)).eval()
    x = torch.randn(2, 20, 32)
    x[..., :3] = 0
    config = recipe(teacher).build()
    estimator = config.param_estimator
    estimator.prepare(model)
    registry = StatisticsRegistry()
    requests = estimator.declare_statistics()
    registry.register_requests(requests)
    registry.collect(model, [x], reference_data=[x * 1.2])
    saved = {
        request.stat_type: {
            key: value.clone()
            for key, value in registry.get(request).items()
            if isinstance(value, torch.Tensor)
        }
        for request in requests
    }
    plan = estimator.estimate(QGraph(), registry, config)
    assert not plan.kept_float
    for request in requests:
        for key, value in saved[request.stat_type].items():
            torch.testing.assert_close(
                registry.get(request)[key], value, rtol=0, atol=0
            )


def test_block_runner_allows_future_gradient_based_estimators(monkeypatch):
    from neuropress.estimator import QuantizationParamEstimator
    from neuropress.estimator_configs import EstimatorConfig
    from neuropress.plan import QuantizationPlan

    class Samples(PairedCollector):
        def update_pair(self, inputs, output, reference):
            self.inputs = inputs[0].detach().clone()
            self.targets = reference.detach().clone()

        def result(self):
            return {"inputs": self.inputs, "targets": self.targets}

    class GradientEstimator(QuantizationParamEstimator):
        def prepare(self, model):
            self.model = model

        def declare_statistics(self):
            return [StatisticRequest(target_key="", stat_type="paired_samples")]

        def estimate(self, graph, registry, recipe):
            samples = registry.get(self.declare_statistics()[0])
            candidate = deepcopy(self.model).requires_grad_(True)
            loss = (candidate(samples["inputs"]) - samples["targets"]).square().mean()
            gradients = torch.autograd.grad(loss, tuple(candidate.parameters()))
            return QuantizationPlan(
                weight_assignments={
                    name: (module.weight - 0.1 * gradient).detach()
                    for (name, module), gradient in zip(
                        [
                            (name, mod)
                            for name, mod in candidate.named_modules()
                            if isinstance(mod, nn.Linear)
                        ],
                        gradients,
                        strict=True,
                    )
                }
            )

    class GradientConfig(EstimatorConfig):
        type: str = "test_gradient"

        def build(self):
            return GradientEstimator()

    monkeypatch.setitem(COLLECTOR_REGISTRY, "paired_samples", Samples)
    model = nn.Sequential(
        nn.Linear(32, 16, bias=False), nn.Tanh(), nn.Linear(16, 8, bias=False)
    ).eval()
    original = [model[i].weight.detach().clone() for i in [0, 2]]
    config = QuantizationRecipeConfig(estimator=GradientConfig(), kernel_lowering=None)
    x = torch.randn(4, 32)
    result = BlockQuantizationPipeline(config).run_block("block", model, [x * 1.5], [x])
    for i, weight in zip([0, 2], original, strict=True):
        assert not torch.equal(result.model[i].weight, weight)
    assert torch.isfinite(result.student_outputs[0]).all()


def test_checkpoint_resume_restores_eval_mode_for_all_modules(tmp_path):
    torch.manual_seed(91)
    original = nn.Sequential(
        nn.Linear(32, 32, bias=False),
        nn.Sequential(nn.BatchNorm1d(32), nn.Dropout(0.75)),
    )
    x = torch.randn(8, 32)
    runner = BlockQuantizationPipeline(recipe(), checkpoint_dir=tmp_path, run_id="eval")
    fresh = runner.run_block("block", deepcopy(original), [x], [x])
    resumed = runner.run_block("block", deepcopy(original), [x], [x])
    assert resumed.resumed
    assert all(not module.training for module in resumed.model.modules())
    assert all(
        parameter.device.type == "cpu" for parameter in resumed.model.parameters()
    )
    with torch.no_grad():
        torch.testing.assert_close(resumed.model(x), fresh.model(x), rtol=0, atol=0)
        torch.testing.assert_close(
            resumed.model(x), resumed.student_outputs[0], rtol=0, atol=0
        )


def test_registry_relocates_one_statistic_per_access():
    class Offloaded(torch.Tensor):
        pass

    registry = StatisticsRegistry()
    hessian = torch.eye(4)
    offloaded = torch.Tensor._make_subclass(Offloaded, torch.eye(4))
    registry.set_stat("a", "hessian", {"hessian": hessian, "nsamples": 3})
    registry.set_stat("b", "hessian", {"hessian": offloaded, "nsamples": 3})
    registry.to("meta")
    a = StatisticRequest(target_key="a", stat_type="hessian")
    b = StatisticRequest(target_key="b", stat_type="hessian")
    assert registry.get(a)["hessian"].device.type == "meta"
    assert registry.get(a)["nsamples"] == 3
    # The collected original stays where the collector left it.
    assert registry.get(a, relocate=False)["hessian"] is hessian
    # Tensor subclasses (e.g. disk-offloaded Hessians) are not materialized.
    assert registry.get(b)["hessian"] is offloaded


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize("dead", [False, True])
def test_gptq_never_writes_to_a_shared_hessian(ordered, dead):
    from neuropress.methods.gptq import GPTQCalibration
    from neuropress.methods.zp_scale_calib import NormZpScaleCalibration
    from neuropress.qscheme.descriptor import Granularity

    torch.manual_seed(12)
    layer = nn.Linear(32, 16, bias=False)
    x = torch.randn(64, 32)
    if dead:
        x[:, :3] = 0
    hessian = 2 * x.t() @ x / x.shape[0]

    def codes(h):
        gran = Granularity.per_group([32])
        calib = GPTQCalibration(
            granularity=gran,
            possible_quantizers=[
                NormZpScaleCalibration(nbits=4, symmetric=True, granularity=gran)
            ],
            quant_col_in_hessian_diag_magnitude_order=ordered,
        )
        calib.weight_tensor = layer.weight
        calib.module_class = nn.Linear
        calib.module = layer
        calib.name = "layer"
        calib.init()
        calib.hessian, calib.nsamples = h, x.shape[0]
        calib.compute_quantization()
        return calib.quantized_codes

    shared = hessian.clone()
    actual = codes(shared)
    torch.testing.assert_close(shared, hessian, rtol=0, atol=0)
    torch.testing.assert_close(actual, codes(hessian.clone()), rtol=0, atol=0)


class _Variadic(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(32, 16, bias=False)

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class _Keyword(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(32, 16, bias=False)

    def forward(self, hidden_states, scale=1.0):
        return self.inner(hidden_states) * scale


@pytest.mark.parametrize("wrapper", [_Variadic, _Keyword])
def test_keyword_calls_feed_collectors_like_positional_calls(wrapper, monkeypatch):
    import neuropress.calibration.probe.registry as registry_module

    lookups = []
    signature = registry_module.inspect.signature

    def counting(fn):
        lookups.append(fn)
        return signature(fn)

    monkeypatch.setattr(registry_module.inspect, "signature", counting)
    torch.manual_seed(13)
    module = wrapper().eval()
    x = [torch.randn(6, 32) for _ in range(3)]
    request = StatisticRequest(target_key="", stat_type="hessian")

    def hessian(batches):
        registry = StatisticsRegistry()
        registry.register_requests([request])
        registry.collect(module, batches)
        return registry.get(request)["hessian"]

    keyword = hessian(
        [{"hidden_states": b} if wrapper is _Keyword else {"input": b} for b in x]
    )
    assert keyword is not None
    assert len(lookups) == 1  # once per hook, not once per call
    torch.testing.assert_close(keyword, hessian(x), rtol=0, atol=0)


def test_block_streams_are_plain_trees_whether_fresh_or_resumed(tmp_path):
    import collections

    class Output(collections.OrderedDict):
        """Stands in for transformers' ModelOutput (an OrderedDict subclass)."""

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 32, bias=False)

        def forward(self, hidden_states):
            return Output(last_hidden_state=self.proj(hidden_states))

    def propagate(batch, output):
        return {"hidden_states": output["last_hidden_state"]}

    torch.manual_seed(14)
    blocks = [Block().eval() for _ in range(2)]
    data = [{"hidden_states": torch.randn(4, 32)}]
    runs = []
    for _ in range(2):  # the second run resumes every block from checkpoints
        runner = BlockQuantizationPipeline(
            recipe(), checkpoint_dir=tmp_path, run_id="trees"
        )
        runs.append(
            list(
                runner.run(enumerate_names(deepcopy(blocks)), data, propagate=propagate)
            )
        )
    fresh, resumed = runs
    assert all(result.resumed for result in resumed)
    for a, b in zip(fresh, resumed, strict=True):
        assert type(a.student_outputs[0]) is dict
        assert type(b.student_outputs[0]) is dict
        torch.testing.assert_close(a.student_outputs, b.student_outputs)


@pytest.mark.parametrize(("device", "prepares"), [("cpu", 1), ("meta", 2)])
def test_estimator_is_rebound_only_when_the_model_moves(device, prepares):
    from neuropress.estimator import QuantizationParamEstimator
    from neuropress.estimator_configs import EstimatorConfig
    from neuropress.plan import QuantizationPlan

    bound = []

    class Counting(QuantizationParamEstimator):
        def prepare(self, model):
            bound.append(next(model.parameters()).device.type)

        def declare_statistics(self):
            return []

        def estimate(self, graph, registry, recipe):
            return QuantizationPlan()

    class CountingConfig(EstimatorConfig):
        type: str = "test_counting"

        def build(self):
            return Counting()

    config = QuantizationRecipeConfig(estimator=CountingConfig(), kernel_lowering=None)
    QuantizationPipeline(config.build()).run(
        nn.Linear(4, 4).eval(), [torch.randn(2, 4)], estimation_device=device
    )
    assert bound == ["cpu", device][:prepares]


@pytest.mark.parametrize("alias", [False, True])
def test_shared_weights_are_caught_after_blocks_change_device(alias, monkeypatch):
    blocks = [nn.Sequential(nn.Linear(32, 32, bias=False)).eval() for _ in range(3)]
    shared = blocks[0][0].weight
    blocks[2][0].weight = (
        nn.Parameter(shared.detach().view_as(shared)) if alias else shared
    )
    runner = BlockQuantizationPipeline(recipe())
    run_block = runner.run_block

    def round_trip(name, block, *args):
        # Moving to a forward device and back gives every parameter new
        # storage, exactly like a cuda round trip does on a GPU host.
        block.double().float()
        return run_block(name, block, *args)

    monkeypatch.setattr(runner, "run_block", round_trip)
    with pytest.raises(ValueError, match="share parameter storage"):
        list(runner.run(enumerate_names(blocks), [torch.randn(2, 32)]))


def test_reconstruction_keeps_float32_for_low_precision_weights():
    torch.manual_seed(15)
    weight = torch.randn(16, 32).to(torch.bfloat16)
    x = torch.randn(256, 32)
    hessian = 2 * x.t() @ x / x.shape[0]
    # A correction far below bf16 resolution at these weight magnitudes.
    cross = hessian @ torch.full((32, 16), 1e-4)
    cross = cross.t().contiguous()
    fitted = reconstruct_linear_weight(weight, hessian, cross, 0.01)
    expected = reconstruct_linear_weight(weight.float(), hessian, cross, 0.01)
    assert fitted.dtype == torch.float32
    torch.testing.assert_close(fitted, expected, rtol=0, atol=0)
    assert not torch.equal(fitted, weight.float())


@pytest.mark.parametrize(
    "error",
    [torch.linalg.LinAlgError("not positive-definite"), ValueError("Nonfinite")],
)
def test_failed_teacher_fit_falls_back_to_self_gptq(error, monkeypatch, caplog):
    import neuropress.calibration.probe.reconstruction as reconstruction

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(reconstruction, "reconstruct_linear_weight", fail)
    torch.manual_seed(16)
    original = nn.Sequential(nn.Linear(32, 16, bias=False)).eval()
    x = torch.randn(4, 32)
    models = [deepcopy(original), deepcopy(original)]
    for model, teacher in zip(models, [False, True], strict=True):
        pipeline = QuantizationPipeline(recipe(teacher).build())
        pipeline.run(model, [x * 1.2], reference_data=[x])
        assert not pipeline.last_plan.kept_float
    torch.testing.assert_close(
        models[1][0].weight.decompress_to_u8(),
        models[0][0].weight.decompress_to_u8(),
        rtol=0,
        atol=0,
    )
    assert "Teacher reconstruction failed" in caplog.text


def test_injected_statistics_are_not_copied_by_the_registry():
    from neuropress.methods.gptq import GPTQCalibration
    from neuropress.qscheme.descriptor import Granularity

    registry = StatisticsRegistry()
    hessian = torch.eye(4)
    registry.set_stat("layer", "hessian", {"hessian": hessian, "nsamples": 3})
    registry.to("meta")
    calib = GPTQCalibration(granularity=Granularity.per_channel())
    registry.inject(calib, "layer")
    # GPTQ relocates per layer to its weights' device; a registry copy here
    # would be a second K x K allocation for every layer.
    assert calib.hessian is hessian
    assert calib.nsamples == 3


@pytest.mark.parametrize("teacher", [False, True])
def test_disk_offloaded_hessians_quantize_like_in_memory_ones(
    teacher, monkeypatch, tmp_path
):
    from dataclasses import dataclass

    from torch_to_nnef.tensor.offload import OffloadedTensor

    from neuropress.calibration.probe import HessianAccumulator

    offloaded = []

    @dataclass
    class OffloadingHessian(HessianAccumulator):
        offload_intermediate_tensor_on_disk: bool = True

        @property
        def offload_dir(self):
            return tmp_path

        def result(self):
            offloaded.append(isinstance(self.hessian, OffloadedTensor))
            return super().result()

    torch.manual_seed(17)
    original = nn.Sequential(nn.Linear(32, 16, bias=False)).eval()
    x = torch.randn(4, 32)
    x[:, :2] = 0  # dead columns make GPTQ write into its Hessian
    models = [deepcopy(original), deepcopy(original)]
    for model, offload in zip(models, [False, True], strict=True):
        if offload:
            monkeypatch.setitem(COLLECTOR_REGISTRY, "hessian", OffloadingHessian)
        pipeline = QuantizationPipeline(recipe(teacher).build())
        pipeline.run(model, [x * 1.2], reference_data=[x])
        assert not pipeline.last_plan.kept_float
    assert offloaded == [True]
    torch.testing.assert_close(
        models[1][0].weight.decompress_to_u8(),
        models[0][0].weight.decompress_to_u8(),
        rtol=0,
        atol=0,
    )


def test_checkpoint_resumes_when_the_device_is_spelled_differently(tmp_path):
    torch.manual_seed(92)
    original = nn.Sequential(nn.Linear(32, 32, bias=False)).eval()
    x = torch.randn(4, 32)
    first, second = (
        BlockQuantizationPipeline(
            recipe(),
            forward_device=device,
            estimation_device=device,
            checkpoint_dir=tmp_path,
            run_id="devices",
        )
        for device in ("cpu", "cpu:0")
    )
    first.run_block("block", deepcopy(original), [x], [x])
    assert second.run_block("block", deepcopy(original), [x], [x]).resumed


def test_checkpoint_fingerprint_is_stable_and_rejects_opaque_leaves():
    import hashlib

    from neuropress.block_pipeline import _fingerprint

    def digest(value):
        result = hashlib.sha256()
        _fingerprint(value, result)
        return result.hexdigest()

    mixed = {"a": torch.ones(2), 1: None, (2, "b"): torch.float16}
    assert digest(mixed) == digest(dict(reversed(mixed.items())))
    assert digest({1: 0}) != digest({"1": 0})
    assert digest([True]) != digest([1])

    class Cache:
        pass

    with pytest.raises(TypeError, match="Cannot checkpoint a block stream"):
        digest({"x": torch.ones(1), "cache": Cache()})


@pytest.mark.parametrize("teacher", [False, True])
def test_extra_gptq_cache_stages_are_not_requested_as_statistics(teacher):
    from dataclasses import dataclass, fields

    from neuropress.estimators import GPTQEstimator
    from neuropress.quantization.cache import CacheStageDescriptor

    @dataclass
    class CachesCodes(GPTQEstimator):
        def cache_stages(self):
            return [
                *super().cache_stages(),
                CacheStageDescriptor("codes", "Packed codes", scope="calibration"),
            ]

    base = recipe(teacher).build().param_estimator
    estimator = CachesCodes(
        **{f.name: getattr(base, f.name) for f in fields(base) if f.init}
    )
    estimator.prepare(nn.Sequential(nn.Linear(32, 16, bias=False)))
    types = {request.stat_type for request in estimator.declare_statistics()}
    assert types == ({"hessian", "linear_reconstruction"} if teacher else {"hessian"})
