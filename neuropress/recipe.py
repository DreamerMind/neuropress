"""QuantizationRecipe: coherent quantization method definition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .adapter import ModuleQuantizationAdapter
from .estimator import QuantizationParamEstimator
from .kernel_lowering import KernelLowering
from .preprocessing import PreprocessingStep
from .qscheme.descriptor import QSchemeDescriptor


class CalibrationConfig(BaseModel):
    """Calibration dataset and sampling hyperparameters.

    Bundled into :class:`QuantizationRecipe` so the full quantization setup —
    including *which data* to use and *how much* — is self-contained in a
    single serialisable object.

    Attributes:
        dataset: Key into :data:`.calibration.DATASET_REGISTRY`.
        seq_len: Token sequence length per sample.
        n_samples: Number of calibration sequences to collect.
        batch_size: Sequences per calibration batch.
        seed: Random seed for dataset shuffling and sampling.
    """

    dataset: str = "c4"
    seq_len: int = 512
    n_samples: int = 128
    batch_size: int = 1
    seed: int = 42


class QuantizationRecipe(BaseModel):
    """Ties together backend constraints, calibration config, estimator and adapter.

    Attributes:
        calibration: Dataset and sampling configuration used by
            :func:`.t2n.registry.run_quantization_pipeline`.
        allowed_weight_qschemes: Backend-supported weight layouts.
        allowed_activation_qschemes: Backend-supported activation layouts.
        param_estimator: Estimates concrete QSchemes from statistics.
        adapter: Applies the QuantizationPlan to the model.
        kernel_lowering: Optional hardware-specific kernel lowering pass.
        preprocessing: Optional preprocessing steps run before calibration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calibration: CalibrationConfig = CalibrationConfig()
    allowed_weight_qschemes: list[QSchemeDescriptor] = []
    allowed_activation_qschemes: list[QSchemeDescriptor] = []
    param_estimator: QuantizationParamEstimator
    adapter: ModuleQuantizationAdapter
    kernel_lowering: KernelLowering | None = None
    preprocessing: list[PreprocessingStep] = []
