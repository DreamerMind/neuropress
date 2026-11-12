"""Calibration probe sub-package: collectors, Hessian, and statistics registry."""

from .channel_max import ChannelMaxCollector
from .collectors import (
    COLLECTOR_REGISTRY,
    CalibrationProbe,
    Collector,
    PairedCollector,
    StatField,
    StatisticRequest,
)
from .hessian import HessianAccumulator
from .hessian_diag import HessianDiagAccumulator
from .min_max import MinMaxCollector
from .reconstruction import LinearReconstructionCollector
from .registry import StatisticsRegistry

COLLECTOR_REGISTRY.update(
    {
        "min_max": MinMaxCollector,
        "channel_max": ChannelMaxCollector,
        "hessian": HessianAccumulator,
        "linear_reconstruction": LinearReconstructionCollector,
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
    "LinearReconstructionCollector",
    "MinMaxCollector",
    "PairedCollector",
    "StatField",
    "StatisticRequest",
    "StatisticsRegistry",
]
