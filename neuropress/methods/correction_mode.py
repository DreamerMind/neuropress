"""Correction mode enum for VPTQGroupEstimator post-convergence GPTQ pass."""

from enum import StrEnum


class CorrectionMode(StrEnum):
    """Controls the layer/group ordering used during GPTQ correction.

    Only active when ``gptq_correction_iters >= 1``.  Default is
    ``SEQUENTIAL`` which preserves the original per-layer module-order pass.
    """

    SEQUENTIAL = "sequential"
    GHO_LAYER = "gho_layer"
    CENTROID_CENTRIC = "centroid_centric"
