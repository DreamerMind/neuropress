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
    with pytest.raises(ValueError, match="Checkpoint mismatch"):
        changed.run_block("block", deepcopy(original), data, reference)


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
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 2:
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
