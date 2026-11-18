"""Calibration probe sub-package: collectors, Hessian, and statistics registry."""

from .channel_max import ChannelMaxCollector
from .collectors import (
    COLLECTOR_REGISTRY,
    CalibrationProbe,
    Collector,
    InvocationCollector,
    PairedCollector,
    StatField,
    StatisticRequest,
)
from .hessian import HessianAccumulator
from .hessian_diag import HessianDiagAccumulator
from .min_max import MinMaxCollector
from .moe_experts import MoEExpertHessianCollector
from .reconstruction import (
    LinearReconstructionCollector,
    SinkProtectedReconstructionCollector,
)
from .registry import StatisticsRegistry

COLLECTOR_REGISTRY.update(
    {
        "min_max": MinMaxCollector,
        "channel_max": ChannelMaxCollector,
        "hessian": HessianAccumulator,
        "linear_reconstruction": LinearReconstructionCollector,
        "linear_reconstruction_sink": SinkProtectedReconstructionCollector,
        "hessian_diag": HessianDiagAccumulator,
        "moe_expert_hessian": MoEExpertHessianCollector,
    }
)

__all__ = [
    "COLLECTOR_REGISTRY",
    "CalibrationProbe",
    "ChannelMaxCollector",
    "Collector",
    "HessianAccumulator",
    "HessianDiagAccumulator",
    "InvocationCollector",
    "LinearReconstructionCollector",
    "MinMaxCollector",
    "MoEExpertHessianCollector",
    "PairedCollector",
    "SinkProtectedReconstructionCollector",
    "StatField",
    "StatisticRequest",
    "StatisticsRegistry",
]
