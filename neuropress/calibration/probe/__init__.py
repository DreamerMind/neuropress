"""Calibration probe sub-package: collectors, Hessian, and statistics registry."""

from .channel_max import ChannelMaxCollector
from .collectors import (
    COLLECTOR_REGISTRY,
    CalibrationProbe,
    Collector,
    StatField,
    StatisticRequest,
)
from .hessian import HessianAccumulator
from .hessian_diag import HessianDiagAccumulator
from .min_max import MinMaxCollector
from .registry import StatisticsRegistry

COLLECTOR_REGISTRY.update(
    {
        "min_max": MinMaxCollector,
        "channel_max": ChannelMaxCollector,
        "hessian": HessianAccumulator,
        "hessian_diag": HessianDiagAccumulator,
    }
)

__all__ = [
    "COLLECTOR_REGISTRY",
    "CalibrationProbe",
    "ChannelMaxCollector",
    "Collector",
    "HessianAccumulator",
    "HessianDiagAccumulator",
    "MinMaxCollector",
    "StatField",
    "StatisticRequest",
    "StatisticsRegistry",
]
