"""Tests for StatisticRequest, CalibrationProbe, StatisticsRegistry."""

from typing import Any

import torch
from torch import nn

from neuropress.calibration import (
    CalibrationProbe,
    StatisticRequest,
    StatisticsRegistry,
)

# ---------------------------------------------------------------------------
# StatisticRequest
# ---------------------------------------------------------------------------


def test_statistic_request_hash_and_equality():
    r1 = StatisticRequest(target_key="mod/weight", stat_type="min_max")
    r2 = StatisticRequest(target_key="mod/weight", stat_type="min_max")
    assert r1 == r2
    assert hash(r1) == hash(r2)


def test_statistic_request_deduplicated_in_set():
    r1 = StatisticRequest(target_key="a", stat_type="min_max")
    r2 = StatisticRequest(target_key="a", stat_type="min_max")
    assert len({r1, r2}) == 1


# ---------------------------------------------------------------------------
# CalibrationProbe (concrete subclass)
# ---------------------------------------------------------------------------


class _SimpleProbe(CalibrationProbe):
    def required_statistics(self) -> list[StatisticRequest]:
        return [StatisticRequest(target_key="fc/weight", stat_type="min_max")]

    def extract_statistics(self, registry: StatisticsRegistry) -> Any:
        return registry.get(self.required_statistics()[0])


def test_calibration_probe_required_statistics():
    probe = _SimpleProbe()
    reqs = probe.required_statistics()
    assert len(reqs) == 1
    assert reqs[0].stat_type == "min_max"


# ---------------------------------------------------------------------------
# StatisticsRegistry
# ---------------------------------------------------------------------------


class _SingleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


def test_statistics_registry_register_and_deduplicate():
    registry = StatisticsRegistry()
    r1 = StatisticRequest(target_key="fc/weight", stat_type="min_max")
    r2 = StatisticRequest(target_key="fc/weight", stat_type="min_max")
    registry.register_requests([r1, r2])
    assert len(registry._requests) == 1


def test_statistics_registry_collect_min_max():
    model = _SingleLinear()
    data = [torch.randn(2, 4), torch.randn(2, 4)]
    registry = StatisticsRegistry()
    req = StatisticRequest(target_key="fc/weight", stat_type="min_max")
    registry.register_requests([req])
    registry.collect(model, iter(data))
    result = registry.get(req)
    assert result is not None
    assert "min" in result and "max" in result


def test_statistics_registry_has_returns_false_before_collect():
    registry = StatisticsRegistry()
    req = StatisticRequest(target_key="x", stat_type="min_max")
    assert not registry.has(req)


def test_statistics_registry_has_returns_true_after_collect():
    model = _SingleLinear()
    data = [torch.randn(2, 4)]
    registry = StatisticsRegistry()
    req = StatisticRequest(target_key="fc/weight", stat_type="min_max")
    registry.register_requests([req])
    registry.collect(model, iter(data))
    assert registry.has(req)
