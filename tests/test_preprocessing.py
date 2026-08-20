"""Tests for explicit registries and preprocessing pipeline integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neuropress.adapter import ModuleQuantizationAdapter
from neuropress.calibration import (
    COLLECTOR_REGISTRY,
    Collector,
    StatisticRequest,
)
from neuropress.calibration.probe import StatisticsRegistry
from neuropress.estimator import QuantizationParamEstimator
from neuropress.graph.graph import QGraph
from neuropress.methods import METHOD_REGISTRY, CalibrationMethod
from neuropress.pipeline import QuantizationPipeline
from neuropress.plan import QuantizationPlan
from neuropress.preprocessing import (
    PREPROCESSING_REGISTRY,
    PreprocessingStep,
    SmoothQuantStep,
)
from neuropress.qscheme import Granularity, QSchemeDescriptor
from neuropress.recipe import QuantizationRecipe

# ---------------------------------------------------------------------------
# Dummy pipeline helpers
# ---------------------------------------------------------------------------


class _IdentityEstimator(QuantizationParamEstimator):
    def declare_statistics(self) -> list[StatisticRequest]:
        return []

    def estimate(
        self, graph: QGraph, registry: StatisticsRegistry, recipe: QuantizationRecipe
    ) -> QuantizationPlan:
        return QuantizationPlan()


class _IdentityAdapter(ModuleQuantizationAdapter):
    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        return model


class _SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_recipe(**kwargs: Any) -> QuantizationRecipe:
    return QuantizationRecipe(
        allowed_weight_qschemes=[
            QSchemeDescriptor(
                domain="weight",
                nbits=4,
                symmetric=True,
                granularity=Granularity.per_group([4]),
            )
        ],
        param_estimator=_IdentityEstimator(),
        adapter=_IdentityAdapter(),
        **kwargs,
    )


def _make_calib_data() -> list[dict[str, torch.Tensor]]:
    return [{"x": torch.randn(2, 4)} for _ in range(3)]


# ---------------------------------------------------------------------------
# Collector registry
# ---------------------------------------------------------------------------


def test_collector_registry_contains_builtins() -> None:
    assert "min_max" in COLLECTOR_REGISTRY
    assert "hessian" in COLLECTOR_REGISTRY
    assert "channel_max" in COLLECTOR_REGISTRY


def test_collector_registry_values_are_collector_subclasses() -> None:
    for name, cls in COLLECTOR_REGISTRY.items():
        assert issubclass(cls, Collector), (
            f"{name!r} -> {cls} is not a Collector subclass"
        )


def test_collector_registry_accepts_direct_extension() -> None:
    class _CustomCollector(Collector):
        def update(self, data: torch.Tensor) -> None:
            pass

        def result(self) -> dict[str, Any]:
            return {}

    COLLECTOR_REGISTRY["_test_custom"] = _CustomCollector
    assert COLLECTOR_REGISTRY["_test_custom"] is _CustomCollector
    del COLLECTOR_REGISTRY["_test_custom"]


def test_collect_unknown_stat_type_silently_skipped() -> None:
    """collect() must not crash for an unregistered stat type."""
    model = _SimpleModel()
    registry = StatisticsRegistry()
    registry.register_requests(
        [StatisticRequest(target_key="fc", stat_type="nonexistent_xyz")]
    )
    registry.collect(model, [{"x": torch.randn(2, 4)}])  # should not raise


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


def test_method_registry_contains_builtins() -> None:
    assert "gptq" in METHOD_REGISTRY
    assert "gptq_mixed" in METHOD_REGISTRY
    assert "min_max" in METHOD_REGISTRY
    assert "norm" in METHOD_REGISTRY


def test_method_registry_values_are_calibration_method_subclasses() -> None:
    for name, cls in METHOD_REGISTRY.items():
        assert issubclass(cls, CalibrationMethod), (
            f"{name!r} -> {cls} is not a CalibrationMethod subclass"
        )


def test_method_registry_accepts_direct_extension() -> None:
    @dataclass
    class _CustomMethod(CalibrationMethod[None]):
        granularity: Any = None

        def update_statistics(self, data: torch.Tensor) -> None:
            pass

        def compute_qscheme(self) -> None:
            return None

    METHOD_REGISTRY["_test_custom"] = _CustomMethod
    assert METHOD_REGISTRY["_test_custom"] is _CustomMethod
    del METHOD_REGISTRY["_test_custom"]


# ---------------------------------------------------------------------------
# Preprocessing registry + pipeline integration
# ---------------------------------------------------------------------------


def test_preprocessing_registry_contains_smooth_quant() -> None:
    assert "smooth_quant" in PREPROCESSING_REGISTRY
    assert PREPROCESSING_REGISTRY["smooth_quant"] is SmoothQuantStep


def test_smooth_quant_step_is_preprocessing_step_subclass() -> None:
    assert issubclass(SmoothQuantStep, PreprocessingStep)


def test_preprocessing_registry_accepts_direct_extension() -> None:
    class _CustomStep(PreprocessingStep):
        def apply(
            self, model: nn.Module, calibration_data: list[dict[str, Any]]
        ) -> None:
            pass

    PREPROCESSING_REGISTRY["_test_custom"] = _CustomStep
    assert PREPROCESSING_REGISTRY["_test_custom"] is _CustomStep
    del PREPROCESSING_REGISTRY["_test_custom"]


def test_smooth_quant_step_apply_modifies_weights() -> None:
    model = _SimpleModel()
    original_weight = model.fc.weight.data.clone()
    SmoothQuantStep().apply(model, _make_calib_data())
    assert not torch.allclose(model.fc.weight.data, original_weight), (
        "SmoothQuantStep.apply() should have modified fc.weight"
    )


def test_pipeline_runs_preprocessing_step_instance() -> None:
    model = _SimpleModel()
    original_weight = model.fc.weight.data.clone()
    recipe = _make_recipe(preprocessing=[SmoothQuantStep()])
    QuantizationPipeline(recipe).run(model, _make_calib_data())
    assert not torch.allclose(model.fc.weight.data, original_weight), (
        "Pipeline Step 0 should have applied SmoothQuantStep"
    )


def test_pipeline_empty_preprocessing_runs_without_step0() -> None:
    model = _SimpleModel()
    original_weight = model.fc.weight.data.clone()
    QuantizationPipeline(_make_recipe(preprocessing=[])).run(model, _make_calib_data())
    assert torch.allclose(model.fc.weight.data, original_weight), (
        "Empty preprocessing should not modify weights"
    )


# ---------------------------------------------------------------------------
# HadamardRotationStep — incoherence processing as a Step 0 preprocessing pass
# ---------------------------------------------------------------------------


def test_hadamard_rotation_step_is_registered():
    from neuropress.preprocessing import (
        PREPROCESSING_CONFIG_REGISTRY,
        PREPROCESSING_REGISTRY,
        HadamardRotationStep,
    )

    assert PREPROCESSING_REGISTRY["hadamard_rotation"] is HadamardRotationStep
    assert "hadamard_rotation" in PREPROCESSING_CONFIG_REGISTRY


def test_hadamard_rotation_config_builds_step_with_its_settings():
    from neuropress.preprocessing import (
        HadamardRotationStep,
        HadamardRotationStepConfig,
    )

    step = HadamardRotationStepConfig(seed=99, rotate_down_proj=False).build()

    assert isinstance(step, HadamardRotationStep)
    assert step.seed == 99
    assert step.rotate_down_proj is False


def test_hadamard_rotation_step_preserves_model_output():
    from neuropress.preprocessing import HadamardRotationStep
    from tests.transforms.test_rotation_surgery import _toy

    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids)

    HadamardRotationStep(seed=17).apply(model, [])

    assert torch.allclose(before, model(ids), atol=1e-9)


# ---------------------------------------------------------------------------
# DownProjSmoothStep — per-channel scale migration on down_proj only
# ---------------------------------------------------------------------------


def test_down_proj_smooth_step_is_registered():
    from neuropress.preprocessing import (
        PREPROCESSING_CONFIG_REGISTRY,
        PREPROCESSING_REGISTRY,
        DownProjSmoothStep,
    )

    assert PREPROCESSING_REGISTRY["down_proj_smooth"] is DownProjSmoothStep
    assert "down_proj_smooth" in PREPROCESSING_CONFIG_REGISTRY


def test_down_proj_smooth_config_builds_step_with_alpha():
    from neuropress.preprocessing import (
        DownProjSmoothStep,
        DownProjSmoothStepConfig,
    )

    step = DownProjSmoothStepConfig(alpha=0.25).build()
    assert isinstance(step, DownProjSmoothStep)
    assert step.alpha == 0.25


def test_down_proj_smooth_step_preserves_model_output():
    from neuropress.preprocessing import DownProjSmoothStep
    from tests.methods.test_down_proj_smoothing import _toy

    model = _toy()
    ids = torch.tensor([[1, 5, 9, 2]])
    before = model(ids).clone()

    DownProjSmoothStep(alpha=0.5).apply(model, [{"ids": ids}])

    assert torch.allclose(before, model(ids), atol=1e-9)
