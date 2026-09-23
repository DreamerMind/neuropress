"""neuropress – graph-aware quantization framework."""

from .adapter import ModuleQuantizationAdapter, WeightAssignmentAdapter
from .calibration import (
    COLLECTOR_REGISTRY,
    DATASET_REGISTRY,
    CalibrationProbe,
    StatisticRequest,
    StatisticsRegistry,
)
from .estimator import QuantizationParamEstimator
from .estimator_configs import (
    WEIGHT_TRANSFORM_REGISTRY,
    GptqEstimatorConfig,
    GptqPanelEstimatorConfig,
    HqqEstimatorConfig,
    MinMaxZpWeightEstimatorConfig,
    TernaryGptqEstimatorConfig,
    WeightTransformEntry,
)
from .estimators import (
    CompositeEstimator,
    GPTQEstimator,
    GPTQPanelEstimator,
    HQQEstimator,
    MinMaxZpWeightEstimator,
)
from .graph import QGraph, QNode, QValue, RuntimeModuleGraphProvider
from .kernel_lowering import AutoKernelLowering, KernelLowering
from .methods import METHOD_REGISTRY
from .pipeline import QuantizationPipeline
from .plan import QuantizationPlan
from .preprocessing import (
    PREPROCESSING_CONFIG_REGISTRY,
    PREPROCESSING_REGISTRY,
    PreprocessingStep,
    PreprocessingStepConfig,
    SmoothQuantStep,
    SmoothQuantStepConfig,
    TernaryHadamardStep,
    TernaryHadamardStepConfig,
    preprocessing_step_config,
)
from .qscheme import (
    Granularity,
    GranularityType,
    PanelSpec,
    QSchemeDescriptor,
    QSchemeGroup,
    TensorSlice,
    TernaryScaleScheme,
)
from .recipe import CalibrationConfig, QuantizationRecipe
from .recipe_config import (
    ADAPTER_CONFIG_REGISTRY,
    ESTIMATOR_CONFIG_REGISTRY,
    KERNEL_LOWERING_CONFIG_REGISTRY,
    AdapterConfig,
    AutoKernelLoweringConfig,
    CompositeEstimatorConfig,
    EstimatorConfig,
    KernelLoweringConfig,
    QuantizationRecipeConfig,
    WeightAssignmentAdapterConfig,
    adapter_config,
    estimator_config,
    kernel_lowering_config,
)
from .t2n import _qtensor_patch as _qtensor_patch  # patches QTensor device handling
from .targets import ActivationTarget, WeightTarget

__all__ = [
    "ADAPTER_CONFIG_REGISTRY",
    "COLLECTOR_REGISTRY",
    "DATASET_REGISTRY",
    "ESTIMATOR_CONFIG_REGISTRY",
    "KERNEL_LOWERING_CONFIG_REGISTRY",
    "METHOD_REGISTRY",
    "PREPROCESSING_CONFIG_REGISTRY",
    "PREPROCESSING_REGISTRY",
    "WEIGHT_TRANSFORM_REGISTRY",
    "ActivationTarget",
    "AdapterConfig",
    "AutoKernelLowering",
    "AutoKernelLoweringConfig",
    "CalibrationConfig",
    "CalibrationProbe",
    "CompositeEstimator",
    "CompositeEstimatorConfig",
    "EstimatorConfig",
    "GPTQEstimator",
    "GPTQPanelEstimator",
    "GptqEstimatorConfig",
    "GptqPanelEstimatorConfig",
    "Granularity",
    "GranularityType",
    "HQQEstimator",
    "HqqEstimatorConfig",
    "KernelLowering",
    "KernelLoweringConfig",
    "MinMaxZpWeightEstimator",
    "MinMaxZpWeightEstimatorConfig",
    "ModuleQuantizationAdapter",
    "PanelSpec",
    "PreprocessingStep",
    "PreprocessingStepConfig",
    "QGraph",
    "QNode",
    "QSchemeDescriptor",
    "QSchemeGroup",
    "QValue",
    "QuantizationParamEstimator",
    "QuantizationPipeline",
    "QuantizationPlan",
    "QuantizationRecipe",
    "QuantizationRecipeConfig",
    "RuntimeModuleGraphProvider",
    "SmoothQuantStep",
    "SmoothQuantStepConfig",
    "StatisticRequest",
    "StatisticsRegistry",
    "TensorSlice",
    "TernaryGptqEstimatorConfig",
    "TernaryHadamardStep",
    "TernaryHadamardStepConfig",
    "TernaryScaleScheme",
    "WeightAssignmentAdapter",
    "WeightAssignmentAdapterConfig",
    "WeightTarget",
    "WeightTransformEntry",
    "adapter_config",
    "estimator_config",
    "kernel_lowering_config",
    "preprocessing_step_config",
]
