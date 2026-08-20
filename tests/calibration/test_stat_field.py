"""Tests for StatField, StatisticsRegistry.register_method and .inject."""

from typing import Annotated

import torch
from torch import nn

from neuropress.calibration.probe import StatField, StatisticsRegistry

# ---------------------------------------------------------------------------
# StatField
# ---------------------------------------------------------------------------


def test_stat_field_basic_construction():
    sf = StatField("hessian")
    assert sf.stat_type == "hessian"
    assert sf.field_key is None


def test_stat_field_with_field_key():
    sf = StatField("hessian", "nsamples")
    assert sf.stat_type == "hessian"
    assert sf.field_key == "nsamples"


# ---------------------------------------------------------------------------
# Helpers: a fake method class with StatField annotations
# ---------------------------------------------------------------------------


class _FakeMethod:
    """Simulates a CalibrationMethod with Annotated StatField hints."""

    hessian: Annotated[torch.Tensor | None, StatField("hessian", "hessian")] = None
    nsamples: Annotated[int, StatField("hessian", "nsamples")] = 0
    # A plain field — should be ignored by register_method / inject.
    name: str = ""


class _SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# register_method
# ---------------------------------------------------------------------------


def test_register_method_registers_unique_stat_types():
    registry = StatisticsRegistry()
    method = _FakeMethod()
    registry.register_method(method, target_key="fc")
    # Both hessian fields share the same stat_type → deduped to one request.
    assert len(registry._requests) == 1
    key = next(iter(registry._requests))
    assert "hessian" in key


def test_register_method_registers_correct_target_key():
    registry = StatisticsRegistry()
    method = _FakeMethod()
    registry.register_method(method, target_key="encoder.layer")
    req = next(iter(registry._requests.values()))
    assert req.target_key == "encoder.layer"


def test_register_method_plain_fields_not_registered():
    registry = StatisticsRegistry()
    method = _FakeMethod()
    registry.register_method(method, target_key="fc")
    # Only hessian, not "name" or other plain fields.
    for req in registry._requests.values():
        assert req.stat_type == "hessian"


def test_register_method_deduplicates_across_calls():
    registry = StatisticsRegistry()
    method = _FakeMethod()
    registry.register_method(method, target_key="fc")
    registry.register_method(method, target_key="fc")
    assert len(registry._requests) == 1


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------


def _registry_with_hessian(target_key: str = "fc") -> StatisticsRegistry:
    """Build a registry with a pre-populated hessian stat."""
    registry = StatisticsRegistry()
    # Manually populate the stat cache to avoid needing a real model forward pass.
    h = torch.eye(4)
    dedupe_key = f"{target_key}::hessian"
    registry._stats[dedupe_key] = {"hessian": h, "nsamples": 32}
    return registry


def test_inject_sets_hessian_field():
    registry = _registry_with_hessian("fc")
    method = _FakeMethod()
    registry.inject(method, target_key="fc")
    assert method.hessian is not None
    assert method.hessian.shape == (4, 4)


def test_inject_sets_nsamples_field():
    registry = _registry_with_hessian("fc")
    method = _FakeMethod()
    registry.inject(method, target_key="fc")
    assert method.nsamples == 32


def test_inject_missing_stat_leaves_field_unchanged():
    registry = StatisticsRegistry()  # no stats collected
    method = _FakeMethod()
    method.nsamples = 99
    registry.inject(method, target_key="fc")
    assert method.nsamples == 99  # unchanged


def test_inject_uses_field_key_override():
    """field_key in StatField("hessian", "hessian") maps to result["hessian"]."""
    registry = _registry_with_hessian("fc")
    method = _FakeMethod()
    registry.inject(method, target_key="fc")
    # field_key is "hessian" → result dict key is "hessian" → method.hessian set.
    assert isinstance(method.hessian, torch.Tensor)


def test_inject_ignores_plain_fields():
    registry = _registry_with_hessian("fc")
    method = _FakeMethod()
    method.name = "original"
    registry.inject(method, target_key="fc")
    # "name" has no StatField annotation → untouched.
    assert method.name == "original"
