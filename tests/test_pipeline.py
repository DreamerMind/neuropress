"""Integration tests for QuantizationPipeline with dummy implementations."""

import torch
from torch import nn

from neuropress.adapter import ModuleQuantizationAdapter
from neuropress.calibration import StatisticRequest, StatisticsRegistry
from neuropress.estimator import QuantizationParamEstimator
from neuropress.graph.graph import QGraph
from neuropress.pipeline import QuantizationPipeline
from neuropress.plan import QuantizationPlan
from neuropress.qscheme import Granularity, QSchemeDescriptor
from neuropress.recipe import QuantizationRecipe

# ---------------------------------------------------------------------------
# Dummy implementations
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
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_recipe():
    return QuantizationRecipe(
        allowed_weight_qschemes=[
            QSchemeDescriptor(
                domain="weight",
                nbits=4,
                symmetric=True,
                granularity=Granularity.per_group([32]),
            )
        ],
        allowed_activation_qschemes=[],
        param_estimator=_IdentityEstimator(),
        adapter=_IdentityAdapter(),
    )


def test_pipeline_runs():
    model = _SimpleModel()
    recipe = _make_recipe()
    pipeline = QuantizationPipeline(recipe)
    calib_data = [torch.randn(2, 8) for _ in range(3)]
    result = pipeline.run(model, calib_data)
    assert isinstance(result, nn.Module)


def test_pipeline_skips_calibration_forward_when_no_statistics_are_requested():
    class CountingModel(_SimpleModel):
        def __init__(self):
            super().__init__()
            self.forward_calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return super().forward(x)

    model = CountingModel()
    pipeline = QuantizationPipeline(_make_recipe())
    pipeline.run(model, [torch.randn(2, 8) for _ in range(3)])

    assert model.forward_calls == 1


def test_pipeline_output_matches_input_api():
    """The adapted model must preserve the forward signature."""
    model = _SimpleModel()
    recipe = _make_recipe()
    pipeline = QuantizationPipeline(recipe)
    calib_data = [torch.randn(4, 8)]
    adapted = pipeline.run(model, calib_data)
    x = torch.randn(2, 8)
    out = adapted(x)
    assert out.shape == (2, 4)


def test_graph_is_built():
    """Verify the graph provider runs without error."""
    from neuropress.graph import RuntimeModuleGraphProvider

    model = _SimpleModel()
    provider = RuntimeModuleGraphProvider()
    graph = provider.build(model, torch.randn(1, 8))
    assert len(graph.nodes) >= 3  # fc1, relu, fc2 at minimum
