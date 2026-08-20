"""Calibration probes: Collector ABC, StatisticRequest, StatField, CalibrationProbe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel

if TYPE_CHECKING:
    from .registry import StatisticsRegistry


class Collector(ABC):
    """Contract for statistics accumulators used by StatisticsRegistry."""

    @abstractmethod
    def update(self, data: torch.Tensor) -> None:
        """Accumulate statistics from one data batch."""

    @abstractmethod
    def result(self) -> dict[str, Any]:
        """Return accumulated statistics as a dict."""


COLLECTOR_REGISTRY: dict[str, type[Collector]] = {}


class StatisticRequest(BaseModel):
    """Declarative request for a statistic to collect during calibration."""

    target_key: str  # WeightTarget.key or ActivationTarget.key
    stat_type: str  # e.g. "min_max", "hessian"

    def __hash__(self) -> int:
        return hash((self.target_key, self.stat_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StatisticRequest):
            return NotImplemented
        return self.target_key == other.target_key and self.stat_type == other.stat_type


class StatField:
    """Annotation marker for registry-backed CalibrationMethod fields.

    Usage::

        hessian: Annotated[Optional[Tensor], StatField("hessian")] = None
        nsamples: Annotated[int, StatField("hessian", "nsamples")] = 0

    ``stat_type`` matches ``StatisticRequest.stat_type``.
    ``field_key`` is the key in the result dict; ``None`` means use the attribute name.
    """

    def __init__(self, stat_type: str, field_key: str | None = None) -> None:
        self.stat_type = stat_type
        self.field_key = field_key


class CalibrationProbe(ABC):
    """Declares what statistics are needed and extracts them from a registry."""

    @abstractmethod
    def required_statistics(self) -> list[StatisticRequest]:
        """Return the list of statistics this probe needs."""

    @abstractmethod
    def extract_statistics(self, registry: StatisticsRegistry) -> Any:
        """Retrieve and combine statistics from the registry."""
