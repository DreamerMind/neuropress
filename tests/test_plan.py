"""Tests for QuantizationPlan and QuantizationRecipe."""

import torch

from neuropress.plan import QuantizationPlan
from neuropress.qscheme import Granularity, QSchemeDescriptor
from neuropress.recipe import QuantizationRecipe

# ---------------------------------------------------------------------------
# QuantizationPlan
# ---------------------------------------------------------------------------


def test_empty_plan():
    plan = QuantizationPlan()
    assert plan.is_empty()


def test_non_empty_plan():
    plan = QuantizationPlan(
        weight_assignments={"n1/weight": torch.zeros(4)},
    )
    assert not plan.is_empty()


def test_activation_assignment():
    plan = QuantizationPlan(
        activation_assignments={"v1": torch.zeros(4)},
    )
    assert not plan.is_empty()
    assert "v1" in plan.activation_assignments


# ---------------------------------------------------------------------------
# QuantizationRecipe
# ---------------------------------------------------------------------------


def _dummy_estimator():
    from neuropress.estimator import QuantizationParamEstimator
    from neuropress.plan import QuantizationPlan

    class _E(QuantizationParamEstimator):
        def declare_statistics(self):
            return []

        def estimate(self, graph, registry, recipe):
            return QuantizationPlan()

    return _E()


def _dummy_adapter():
    from torch import nn

    from neuropress.adapter import ModuleQuantizationAdapter
    from neuropress.plan import QuantizationPlan

    class _A(ModuleQuantizationAdapter):
        def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
            return model

    return _A()


def test_construction():
    recipe = QuantizationRecipe(
        allowed_weight_qschemes=[
            QSchemeDescriptor(
                domain="weight",
                nbits=4,
                symmetric=True,
                granularity=Granularity.per_group([32]),
            )
        ],
        allowed_activation_qschemes=[],
        param_estimator=_dummy_estimator(),
        adapter=_dummy_adapter(),
    )
    assert len(recipe.allowed_weight_qschemes) == 1
