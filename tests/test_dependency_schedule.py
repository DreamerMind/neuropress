"""Dependency-aware calibration barriers and recollection on adapted inputs."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from neuropress import BlockQuantizationPipeline, QuantizationSchedule
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe_config import QuantizationRecipeConfig


def config(teacher=False):
    return QuantizationRecipeConfig.model_validate(
        {
            "estimator": {
                "type": "gptq",
                "group_size": 32,
                "nbits": 4,
                "weight_transform": "q4_0",
                "reconstruction": "teacher" if teacher else "self",
            },
            "kernel_lowering": None,
        }
    )


class Branches(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(32, 32, bias=False)
        self.k = nn.Linear(32, 32, bias=False)
        self.out = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        q = torch.sin(self.q(x)).reshape_as(x)
        k = self.k(input=x)
        return self.out(q * k + x)


def test_dependency_stages_follow_functional_edges_not_hook_order():
    model = Branches().eval()
    schedule = QuantizationSchedule.trace(
        model, [torch.randn(2, 32)], ["out", "q", "k"]
    )
    assert schedule.stages == (("q", "k"), ("out",))
    assert str(schedule) == "[q | k] → [out]"
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )


@pytest.mark.parametrize("teacher", [False, True])
def test_stages_match_explicit_recollection_and_resume(teacher, tmp_path):
    torch.manual_seed(81)
    original = Branches().eval()
    actual, expected = deepcopy(original), deepcopy(original)
    data = [torch.randn(4, 32)]
    reference = [data[0] * 1.1]
    cfg = config(teacher)
    runner = BlockQuantizationPipeline(
        cfg, target_paths=("q", "k", "out"), checkpoint_dir=tmp_path, run_id="stages-v1"
    )
    result = runner.run_block("block", actual, data, reference)
    for stage in [("q", "k"), ("out",)]:
        QuantizationPipeline(cfg.build()).run(
            expected,
            data,
            reference_data=reference,
            reference_model=original,
            target_paths=set(stage),
        )
    for name in ["q", "k", "out"]:
        torch.testing.assert_close(
            actual.get_submodule(name).weight.decompress_to_u8(),
            expected.get_submodule(name).weight.decompress_to_u8(),
            rtol=0,
            atol=0,
        )
    restored = runner.run_block("block", deepcopy(original), data, reference)
    assert restored.resumed and restored.schedule == result.schedule
    torch.testing.assert_close(
        restored.student_outputs[0], result.student_outputs[0], rtol=0, atol=0
    )

    changed = BlockQuantizationPipeline(
        cfg, target_paths=("q",), checkpoint_dir=tmp_path, run_id="stages-v1"
    )
    assert not changed.run_block("block", deepcopy(original), data, reference).resumed
    assert runner.run_block("block", deepcopy(original), data, reference).resumed


def test_recollection_avoids_independent_teacher_overcorrection():
    torch.manual_seed(82)
    model = nn.Sequential(*(nn.Linear(32, 32, bias=False) for _ in range(3))).eval()
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.eye(32))
    x = torch.randn(256, 32)
    batch = BlockQuantizationPipeline(config(True)).run_block(
        "b", deepcopy(model), [x * 1.5], [x]
    )
    staged = BlockQuantizationPipeline(
        config(True), target_paths=("0", "1", "2")
    ).run_block("b", deepcopy(model), [x * 1.5], [x])
    assert (staged.student_outputs[0] - x).norm() < (
        batch.student_outputs[0] - x
    ).norm() * 0.1
    for layer in model:
        torch.testing.assert_close(layer.weight, torch.eye(32))


def test_composed_embedding_and_linear_estimators_are_staged():
    model = nn.Sequential(nn.Embedding(64, 32), nn.Linear(32, 16, bias=False)).eval()
    cfg = config().model_copy(
        update={
            "estimator": QuantizationRecipeConfig.model_validate(
                {
                    "estimator": {
                        "type": "composite",
                        "methods": [
                            {
                                "type": "gptq",
                                "target": "linear",
                                "group_size": 32,
                                "weight_transform": "q4_0",
                            },
                            {
                                "type": "min_max_zp_weight",
                                "target": "embedding",
                                "nbits": 8,
                            },
                        ],
                    },
                    "kernel_lowering": None,
                }
            ).estimator
        }
    )
    x = torch.randint(0, 64, (2, 8))
    result = BlockQuantizationPipeline(cfg, target_paths=("0", "1")).run_block(
        "b", model, [x], [x]
    )
    assert result.schedule.stages == (("0",), ("1",))
    assert not result.kept_float


@pytest.mark.parametrize("kind", ["repeated", "inplace", "branch", "missing", "tied"])
def test_unsupported_forwards_fail_and_remove_hooks(kind):
    class Unsupported(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)
            self.b = nn.Linear(32, 32)
            if kind == "tied":
                self.b.weight = nn.Parameter(
                    self.a.weight.detach().view_as(self.a.weight)
                )

        def forward(self, x):
            y = self.a(x)
            if kind == "repeated":
                y = self.a(y)
            if kind == "inplace":
                y.add_(x)
            if kind == "branch" and x.sum().item() > 0:
                y = y + x
            return y if kind == "missing" else self.b(y)

    model = Unsupported().eval()
    with pytest.raises(ValueError):
        QuantizationSchedule.trace(model, [torch.ones(2, 32)], ["a", "b"])
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )


def test_functional_parameter_use_adds_a_dependency():
    class FunctionalWeight(Branches):
        def forward(self, x):
            earlier = self.k(torch.nn.functional.linear(x, self.q.weight))
            return earlier + self.q(x)

    model = FunctionalWeight().eval()
    schedule = QuantizationSchedule.trace(model, [torch.randn(2, 32)], ["k", "q"])
    assert schedule.stages == (("q",), ("k",))


def test_later_stage_hessian_uses_updated_weights():
    from neuropress.calibration.probe import StatisticRequest
    from neuropress.estimator import QuantizationParamEstimator
    from neuropress.estimator_configs import EstimatorConfig
    from neuropress.plan import QuantizationPlan

    observed = {}

    class ScalingEstimator(QuantizationParamEstimator):
        filter_fn = None

        def prepare(self, model):
            self.targets = {
                name: mod
                for name, mod in model.named_modules()
                if isinstance(mod, nn.Linear) and self.filter_fn(name, mod)
            }

        def declare_statistics(self):
            return [
                StatisticRequest(target_key=name, stat_type="hessian")
                for name in self.targets
            ]

        def estimate(self, graph, registry, recipe):
            for request in self.declare_statistics():
                observed[request.target_key] = registry.get(request)["hessian"].clone()
            return QuantizationPlan(
                weight_assignments={
                    name: mod.weight.detach() * 0.5
                    for name, mod in self.targets.items()
                }
            )

    class ScalingConfig(EstimatorConfig):
        type: str = "test_scaling"

        def build(self):
            return ScalingEstimator()

    model = nn.Sequential(
        nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
    ).eval()
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.eye(32))
    x = torch.randn(8, 32)
    cfg = QuantizationRecipeConfig(estimator=ScalingConfig(), kernel_lowering=None)
    BlockQuantizationPipeline(cfg, target_paths=("0", "1")).run_block(
        "b", model, [x], [x]
    )
    torch.testing.assert_close(observed["1"], observed["0"] * 0.25)


def test_dependencies_are_unioned_across_representative_batches():
    class DifferentOrders(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)
            self.b = nn.Linear(32, 32)

        def forward(self, x):
            return self.a(self.b(x)) if x.shape[0] == 2 else self.b(self.a(x))

    model = DifferentOrders().eval()
    with pytest.raises(ValueError, match="cycle"):
        QuantizationSchedule.trace(
            model, [torch.randn(2, 32), torch.randn(3, 32)], ["a", "b"]
        )


@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("collection_failure", [False, True])
def test_pipeline_selections_are_scoped_and_preserve_original_filters(
    composed, collection_failure
):
    from neuropress.estimators import CompositeEstimator

    class FailsDuringCollection(nn.Sequential):
        def forward(self, x):
            # Unpaired statistics are collected under inference mode.
            if torch.is_inference_mode_enabled():
                raise RuntimeError("collection failed")
            return super().forward(x)

    def model(cls=nn.Sequential):
        return cls(*(nn.Linear(32, 32, bias=False) for _ in range(3))).eval()

    recipe = config().build()
    estimator = recipe.param_estimator
    format_filter = estimator.filter_fn

    def original_filter(name, module):
        return name != "2" and (format_filter is None or format_filter(name, module))

    estimator.filter_fn = original_filter
    if composed:
        recipe.param_estimator = CompositeEstimator([CompositeEstimator([estimator])])
    pipeline = QuantizationPipeline(recipe)
    x = torch.randn(8, 32)
    if collection_failure:
        with pytest.raises(RuntimeError, match="collection failed"):
            pipeline.run(model(FailsDuringCollection), [x], target_paths={"0"})
    else:
        pipeline.run(model(), [x], target_paths={"0"})
        assert set(pipeline.last_plan.weight_assignments) == {"0"}
    assert estimator.filter_fn is original_filter
    pipeline.run(model(), [x], target_paths={"1"})
    assert set(pipeline.last_plan.weight_assignments) == {"1"}
    assert estimator.filter_fn is original_filter
    pipeline.run(model(), [x])
    assert set(pipeline.last_plan.weight_assignments) == {"0", "1"}
    assert estimator.filter_fn is original_filter


def test_composite_selection_unwinds_when_a_member_rejects_selection():
    from neuropress.estimator import QuantizationParamEstimator
    from neuropress.estimators import CompositeEstimator
    from neuropress.plan import QuantizationPlan

    class Unsupported(QuantizationParamEstimator):
        def declare_statistics(self):
            return []

        def estimate(self, graph, registry, recipe):
            return QuantizationPlan()

    estimator = config().build().param_estimator
    original_filter = estimator.filter_fn
    composite = CompositeEstimator([estimator, Unsupported()])
    with pytest.raises(NotImplementedError, match="does not support target selection"):
        with composite.select_targets({"0"}):
            pytest.fail("Unsupported member must reject selection")
    assert estimator.filter_fn is original_filter


def test_self_dependent_target_inputs_are_rejected_and_hooks_removed():
    class SelfDependent(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            return self.a(torch.nn.functional.linear(x, self.a.weight))

    model = SelfDependent().eval()
    with pytest.raises(ValueError, match="input depends on its own parameters: a"):
        QuantizationSchedule.trace(model, [torch.randn(2, 32)], ["a"])
    assert all(
        not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()
    )


@pytest.mark.parametrize("teacher", [False, True])
def test_teacher_factory_runs_only_for_paired_statistics(teacher):
    calls = []
    original = Branches().eval()

    def factory():
        calls.append(1)
        return original

    x = torch.randn(4, 32)
    QuantizationPipeline(config(teacher).build()).run(
        deepcopy(original), [x * 1.1], reference_data=[x], reference_model=factory
    )
    assert len(calls) == int(teacher)


@pytest.mark.parametrize("teacher", [False, True])
def test_staged_blocks_place_a_teacher_only_when_a_later_stage_pairs(
    teacher, monkeypatch
):
    import neuropress.block_pipeline as block_pipeline

    placed = []
    run = block_pipeline.QuantizationPipeline.run

    def spy(self, *args, reference_model=None, **kwargs):
        if reference_model is not None:
            assert not isinstance(reference_model, nn.Module)
            factory = reference_model

            def reference_model():
                placed.append(kwargs["target_paths"])
                return factory()

        return run(self, *args, reference_model=reference_model, **kwargs)

    monkeypatch.setattr(block_pipeline.QuantizationPipeline, "run", spy)
    x = torch.randn(4, 32)
    BlockQuantizationPipeline(
        config(teacher), target_paths=("q", "k", "out")
    ).run_block("b", Branches().eval(), [x * 1.1], [x])
    # Stage one pairs against the still-float observed block itself.
    assert placed == ([{"out"}] if teacher else [])


@pytest.mark.parametrize("target_paths", [None, ("q", "k", "out")])
def test_inference_view_serves_collection_with_and_without_stages(target_paths):
    views = []

    def inference_model(model):
        views.append(all(not hasattr(m.weight, "decompress_to_u8") for m in model))
        return model

    class Flat(nn.Sequential):
        def __init__(self):
            super().__init__(*(nn.Linear(32, 32, bias=False) for _ in range(3)))

    model = Flat().eval()
    paths = None if target_paths is None else ("0", "1", "2")
    x = torch.randn(4, 32)
    BlockQuantizationPipeline(
        config(), target_paths=paths, inference_model=inference_model
    ).run_block("b", model, [x], [x])
    stages = 1 if paths is None else 3
    # One view per collection stage, then one for the propagated student.
    assert len(views) == stages + 1
    # The first collection observes the untouched float block.
    assert views[0]


@pytest.mark.parametrize("target_paths", [None, ("0", "1")])
def test_tolerated_layer_failures_are_reported_not_fatal(target_paths, monkeypatch):
    import neuropress.estimators as estimators

    layer = estimators._gptq_layer

    def fail_second(name, *args, **kwargs):
        if name == "1":
            raise ValueError("forced failure")
        return layer(name, *args, **kwargs)

    monkeypatch.setattr(estimators, "_gptq_layer", fail_second)
    model = nn.Sequential(
        nn.Linear(32, 32, bias=False), nn.Linear(32, 32, bias=False)
    ).eval()
    x = torch.randn(4, 32)
    result = BlockQuantizationPipeline(config(), target_paths=target_paths).run_block(
        "b", model, [x], [x]
    )
    assert result.kept_float == {"1": "forced failure"}
    assert hasattr(model[0].weight, "decompress_to_u8")
    assert type(model[1].weight) is nn.Parameter


@pytest.mark.parametrize(
    ("kind", "traced"),
    [
        ("same", 1),
        ("shapes", 2),
        ("scalars", 2),
        ("mask", 4),  # value-dependent shapes: every batch is traced
        ("positions", 1),  # integer indexing keeps a fixed shape
    ],
)
def test_trace_replays_each_batch_structure_once(kind, traced):
    class Counting(Branches):
        calls = 0

        def forward(self, x, scale=1.0):
            Counting.calls += 1
            if kind == "mask":
                x = x[x[:, 0] > -10]  # boolean mask: shape depends on values
            if kind == "positions":
                x = x + torch.ones(64, 32)[torch.arange(x.shape[0])]
            return super().forward(x) * scale

    model = Counting().eval()
    shapes = [2, 2, 3, 3] if kind == "shapes" else [2] * 4
    batches = [torch.randn(n, 32) for n in shapes]
    if kind == "scalars":
        batches = [
            {"x": b, "scale": s} for b, s in zip(batches, [1.0, 2.0] * 2, strict=True)
        ]
    schedule = QuantizationSchedule.trace(model, batches, ["q", "k", "out"])
    assert schedule.stages == (("q", "k"), ("out",))
    assert Counting.calls == traced


@pytest.mark.parametrize("teacher", [False, True])
def test_staged_blocks_copy_a_teacher_only_when_a_later_stage_pairs(
    teacher, monkeypatch
):
    import neuropress.block_pipeline as block_pipeline

    copies = []
    deepcopy_ = block_pipeline.copy.deepcopy

    def spy(value, *args, **kwargs):
        if isinstance(value, Branches):
            copies.append(value)
        return deepcopy_(value, *args, **kwargs)

    monkeypatch.setattr(block_pipeline.copy, "deepcopy", spy)
    x = torch.randn(4, 32)
    BlockQuantizationPipeline(
        config(teacher), target_paths=("q", "k", "out")
    ).run_block("b", Branches().eval(), [x * 1.1], [x])
    assert len(copies) == int(teacher)


@pytest.mark.parametrize("teacher", [False, True])
def test_later_stage_pairing_is_probed_once_per_block(teacher, monkeypatch):
    from neuropress.estimators import GPTQEstimator

    prepared = []
    prepare = GPTQEstimator.prepare

    def spy(self, model):
        prepared.append(self)
        return prepare(self, model)

    monkeypatch.setattr(GPTQEstimator, "prepare", spy)
    model = nn.Sequential(*(nn.Linear(32, 32, bias=False) for _ in range(4))).eval()
    x = torch.randn(4, 32)
    result = BlockQuantizationPipeline(
        config(teacher), target_paths=("0", "1", "2", "3")
    ).run_block("b", model, [x * 1.1], [x])
    assert len(result.schedule.stages) == 4
    # One prepare per stage that runs, plus a single probe for all later ones.
    assert len(prepared) == 4 + 1


class Tracked(Branches):
    """Record device moves; 'cpu:0' stands in for an accelerator on CPU hosts."""

    def __init__(self):
        super().__init__()
        self.location = "cpu"
        self.moves = 0

    def _go(self, location):
        if location != self.location:
            self.moves += 1
            self.location = location

    def to(self, *args, **kwargs):
        device = torch._C._nn._parse_to(*args, **kwargs)[0]
        if device is not None:
            self._go(str(device))
        return super().to(*args, **kwargs)

    def cpu(self):
        self._go("cpu")
        return super().cpu()


def test_block_crosses_to_the_forward_device_once_per_run():
    model = Tracked().eval()
    x = torch.randn(4, 32)
    result = BlockQuantizationPipeline(
        config(),
        forward_device="cpu:0",
        estimation_device="cpu:0",
        target_paths=("q", "k", "out"),
    ).run_block("b", model, [x * 1.1], [x])
    assert len(result.schedule.stages) == 2
    # Reference forward, trace, both stages and the student forward share one
    # residency on the forward device; the block is handed back on the host.
    assert model.moves == 2
    assert model.location == "cpu"


def test_failed_stage_returns_every_placed_module_to_the_host(monkeypatch):
    import neuropress.block_pipeline as block_pipeline

    views, teachers = [], []
    run = block_pipeline.QuantizationPipeline.run

    def fail_second_stage(self, model, data, *, reference_model=None, **kwargs):
        if reference_model is not None:
            teachers.append(reference_model())
            raise RuntimeError("stage failed")
        return run(self, model, data, reference_model=reference_model, **kwargs)

    def inference_model(model):
        views.append(deepcopy(model))
        return views[-1]

    monkeypatch.setattr(block_pipeline.QuantizationPipeline, "run", fail_second_stage)
    model = Tracked().eval()
    x = torch.randn(4, 32)
    with pytest.raises(RuntimeError, match="stage failed"):
        BlockQuantizationPipeline(
            config(True),
            forward_device="cpu:0",
            estimation_device="cpu:0",
            target_paths=("q", "k", "out"),
            inference_model=inference_model,
        ).run_block("b", model, [x * 1.1], [x])
    assert len(views) == 2 and len(teachers) == 1
    # Earlier views are released rather than moved; only this test holds them.
    assert [m.location for m in (model, views[-1], *teachers)] == ["cpu"] * 3
