"""Calibration methods sub-package: GPTQ, HQQ, and min-max weight quantization."""

from ..calibration.probe import HessianAccumulator
from .base import (
    METHOD_REGISTRY,
    CalibrationMethod,
    CalibrationMethodOnActivation,
    CalibrationObserve,
)
from .gptq import GPTQCalibration, QBlockOutput, QProcessingData
from .gptq_panel import (
    GPTQMixedPrecisionCalibration,
    GroupQuantizer,
    MixedAllocationStrategy,
    PanelAssignment,
    QuantizerScore,
)
from .hqq import HQQCalibration
from .smooth_quant import (
    absorb_scale_into_prev,
    apply_scale_to_linear,
    apply_smooth_quant,
    compute_smooth_scales,
)
from .zp_scale_calib import (
    MinMaxZpScaleCalibration,
    NormZpScaleCalibration,
    ZpScaleCalibration,
)

# ---------------------------------------------------------------------------
# Explicit registry population — no import side-effects.
# ---------------------------------------------------------------------------

METHOD_REGISTRY.update(
    {
        "min_max": MinMaxZpScaleCalibration,
        "norm": NormZpScaleCalibration,
        "gptq": GPTQCalibration,
        "gptq_mixed": GPTQMixedPrecisionCalibration,
        "hqq": HQQCalibration,
    }
)

__all__ = [
    "METHOD_REGISTRY",
    "CalibrationMethod",
    "CalibrationMethodOnActivation",
    "CalibrationObserve",
    "GPTQCalibration",
    "GPTQMixedPrecisionCalibration",
    "GroupQuantizer",
    "HQQCalibration",
    "HessianAccumulator",
    "MinMaxZpScaleCalibration",
    "MixedAllocationStrategy",
    "NormZpScaleCalibration",
    "PanelAssignment",
    "QBlockOutput",
    "QProcessingData",
    "QuantizerScore",
    "ZpScaleCalibration",
    "absorb_scale_into_prev",
    "apply_scale_to_linear",
    "apply_smooth_quant",
    "compute_smooth_scales",
]
