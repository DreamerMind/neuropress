"""Calibration dataset sub-package: loaders and registry."""

from .audio import load_librispeech_calibration
from .llm import (
    load_c4_calibration,
    load_eval_targeted_calibration,
    load_fineweb_calibration,
)
from .registry import DATASET_REGISTRY

DATASET_REGISTRY.update(
    {
        "c4": load_c4_calibration,
        "fineweb": load_fineweb_calibration,
        "eval_targeted": load_eval_targeted_calibration,
        "librispeech": load_librispeech_calibration,
    }
)

__all__ = [
    "DATASET_REGISTRY",
    "load_c4_calibration",
    "load_eval_targeted_calibration",
    "load_fineweb_calibration",
    "load_librispeech_calibration",
]
