"""Modular, JSON-serialisable recipe configuration framework.

Defines abstract bases and registries for estimator, adapter, kernel
lowering, and preprocessing step configurations.  Concrete subclasses
self-register via decorators and are
resolved automatically when deserialising from JSON::

    @estimator_config("my_estimator")
    class MyEstimatorConfig(EstimatorConfig):
        type: Literal["my_estimator"] = "my_estimator"
        group_size: int = 32

        def build(self) -> QuantizationParamEstimator:
            return MyEstimator(group_size=self.group_size)

:class:`QuantizationRecipeConfig` composes all three abstractions into a
single JSON-round-trippable object::

    recipe = QuantizationRecipeConfig.from_json(json_str).build()

The ``type`` discriminator field in each component config is the registry key
used to look up the concrete class at deserialisation time.  Any module that
defines a concrete config must be imported before :meth:`from_json` is called
so that its decorator fires.

Example JSON::

    {
        "calibration": {"dataset": "c4", "n_samples": 128, "seq_len": 512},
        "preprocessing": [{"type": "smooth_quant", "alpha": 0.5}],
        "estimator":  {"type": "gptq", "weight_transform": "q4_0",
                       "group_size": 32, "nbits": 4},
        "adapter":    {"type": "weight_assignment"}
    }
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    model_validator,
)

from .adapter import (
    CompositeAdapter,
    ModuleQuantizationAdapter,
    WeightAssignmentAdapter,
)
from .estimator import QuantizationParamEstimator
from .estimators import CompositeEstimator
from .kernel_lowering import AutoKernelLowering, KernelLowering
from .nn.backends import EmbeddingBackend, LinearBackend, NormBackend, VQBackend
from .preprocessing import PREPROCESSING_CONFIG_REGISTRY, PreprocessingStepConfig
from .recipe import CalibrationConfig, QuantizationRecipe

# ---------------------------------------------------------------------------
# Abstract base configs
# ---------------------------------------------------------------------------


class EstimatorConfig(BaseModel):
    """Base class for JSON-serialisable estimator configurations.

    Concrete subclasses must implement :meth:`build` and self-register with
    :func:`estimator_config`.
    """

    type: str

    def build(self) -> QuantizationParamEstimator:
        """Build the runtime estimator from this config.

        Returns:
            A ready-to-use :class:`.QuantizationParamEstimator`.
        """
        raise NotImplementedError(f"{type(self).__name__}.build()")


class AdapterConfig(BaseModel):
    """Base class for JSON-serialisable adapter configurations.

    Concrete subclasses must implement :meth:`build` and self-register with
    :func:`adapter_config`.
    """

    type: str

    def build(self) -> ModuleQuantizationAdapter:
        """Build the runtime adapter from this config.

        Returns:
            A ready-to-use :class:`.ModuleQuantizationAdapter`.
        """
        raise NotImplementedError(f"{type(self).__name__}.build()")


# ---------------------------------------------------------------------------
# Registries + decorators
# ---------------------------------------------------------------------------

ESTIMATOR_CONFIG_REGISTRY: dict[str, type[EstimatorConfig]] = {}
ADAPTER_CONFIG_REGISTRY: dict[str, type[AdapterConfig]] = {}


def estimator_config(
    name: str,
) -> Callable[[type[EstimatorConfig]], type[EstimatorConfig]]:
    """Class decorator: register an :class:`EstimatorConfig` subclass by name.

    Args:
        name: Registry key — must match the ``type`` field value.

    Returns:
        The unmodified class (registration is a side-effect).
    """

    def _decorate(cls: type[EstimatorConfig]) -> type[EstimatorConfig]:
        ESTIMATOR_CONFIG_REGISTRY[name] = cls
        return cls

    return _decorate


def adapter_config(
    name: str,
) -> Callable[[type[AdapterConfig]], type[AdapterConfig]]:
    """Class decorator: register an :class:`AdapterConfig` subclass by name.

    Args:
        name: Registry key — must match the ``type`` field value.

    Returns:
        The unmodified class (registration is a side-effect).
    """

    def _decorate(cls: type[AdapterConfig]) -> type[AdapterConfig]:
        ADAPTER_CONFIG_REGISTRY[name] = cls
        return cls

    return _decorate


# ---------------------------------------------------------------------------
# Built-in adapter config
# ---------------------------------------------------------------------------


@adapter_config("weight_assignment")
class WeightAssignmentAdapterConfig(AdapterConfig):
    """Config for :class:`.WeightAssignmentAdapter`.

    Attributes:
        type: Discriminator field, always ``"weight_assignment"``.
    """

    type: Literal["weight_assignment"] = "weight_assignment"

    def build(self) -> WeightAssignmentAdapter:
        """Build a :class:`.WeightAssignmentAdapter`.

        Returns:
            A new :class:`.WeightAssignmentAdapter` instance.
        """
        return WeightAssignmentAdapter()


@adapter_config("composite")
class CompositeAdapterConfig(AdapterConfig):
    """Config for :class:`.CompositeAdapter` — chains multiple adapters in order.

    Attributes:
        type: Discriminator field, always ``"composite"``.
        adapters: Ordered list of adapter configs to apply in sequence.
    """

    type: Literal["composite"] = "composite"
    adapters: list[AdapterConfig] = []

    @model_validator(mode="before")
    @classmethod
    def _resolve_adapters(cls, data: Any) -> Any:
        """Resolve each entry in *adapters* from :data:`ADAPTER_CONFIG_REGISTRY`."""
        if isinstance(data, dict) and "adapters" in data:
            data["adapters"] = [
                _resolve_one(ADAPTER_CONFIG_REGISTRY, a, "adapter")
                for a in data["adapters"]
            ]
        return data

    def build(self) -> CompositeAdapter:
        """Build a :class:`.CompositeAdapter` from this configuration.

        Returns:
            A :class:`.CompositeAdapter` wrapping all configured sub-adapters.
        """
        return CompositeAdapter(adapters=[a.build() for a in self.adapters])


# ---------------------------------------------------------------------------
# Kernel lowering config
# ---------------------------------------------------------------------------


class KernelLoweringConfig(BaseModel):
    """Base class for JSON-serialisable kernel lowering configurations.

    Concrete subclasses must implement :meth:`build` and self-register with
    :func:`kernel_lowering_config`.
    """

    type: str

    def build(self) -> KernelLowering:
        """Build the runtime kernel lowering pass from this config.

        Returns:
            A ready-to-use :class:`.KernelLowering`.
        """
        raise NotImplementedError(f"{type(self).__name__}.build()")


KERNEL_LOWERING_CONFIG_REGISTRY: dict[str, type[KernelLoweringConfig]] = {}


def kernel_lowering_config(
    name: str,
) -> Callable[[type[KernelLoweringConfig]], type[KernelLoweringConfig]]:
    """Class decorator: register a :class:`KernelLoweringConfig` subclass by name.

    Args:
        name: Registry key — must match the ``type`` field value.

    Returns:
        The unmodified class (registration is a side-effect).
    """

    def _decorate(
        cls: type[KernelLoweringConfig],
    ) -> type[KernelLoweringConfig]:
        KERNEL_LOWERING_CONFIG_REGISTRY[name] = cls
        return cls

    return _decorate


@kernel_lowering_config("auto")
class AutoKernelLoweringConfig(KernelLoweringConfig):
    """Config for :class:`.AutoKernelLowering`.

    Runs all linearize transforms with configurable per-path backends.
    Each backend field is constrained to the valid choices for its path.

    Attributes:
        type: Discriminator field, always ``"auto"``.
        linear_backend: Backend for scalar-quantized ``nn.Linear``.
        vq_backend: Backend for vector-quantized ``nn.Linear``.
        embedding_backend: Backend for quantized ``nn.Embedding``.
        norm_backend: Backend for RMSNorm modules.  ``"auto"`` and ``"cuda"``
            apply the fused single-pass CUDA kernel when JIT is available;
            ``"default"`` skips replacement.
    """

    type: Literal["auto"] = "auto"
    linear_backend: LinearBackend = LinearBackend.AUTO
    vq_backend: VQBackend = VQBackend.AUTO
    embedding_backend: EmbeddingBackend = EmbeddingBackend.AUTO
    norm_backend: NormBackend = NormBackend.AUTO

    def build(self) -> AutoKernelLowering:
        """Build an :class:`.AutoKernelLowering` from this configuration.

        Returns:
            A new :class:`.AutoKernelLowering` with the configured backends.
        """
        return AutoKernelLowering(
            linear_backend=self.linear_backend.value,
            vq_backend=self.vq_backend.value,
            embedding_backend=self.embedding_backend.value,
            norm_backend=self.norm_backend.value,
        )


# ---------------------------------------------------------------------------
# Polymorphic dispatch helper
# ---------------------------------------------------------------------------


def _resolve_one(
    registry: dict[str, type],
    data: Any,
    label: str,
) -> Any:
    """Resolve *data* to its registered config class instance.

    Args:
        registry: Mapping of type name → config class.
        data: A raw dict with a ``type`` key, or an already-instantiated
            config object (returned unchanged).
        label: Human-readable name for error messages (e.g. ``"estimator"``).

    Returns:
        A validated config instance for the type named in ``data["type"]``.

    Raises:
        ValueError: If ``data["type"]`` is not in *registry*.
    """
    if not isinstance(data, dict):
        return data
    type_key = data.get("type")
    cls = registry.get(type_key)
    if cls is None:
        raise ValueError(
            f"Unknown {label} type {type_key!r}. Available: {sorted(registry)}"
        )
    return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Composite recipe config
# ---------------------------------------------------------------------------


class QuantizationRecipeConfig(BaseModel):
    """Fully JSON-serialisable recipe configuration.

    All three components (estimator, adapter, preprocessing) are expressed as
    typed config objects so the entire setup round-trips through JSON without
    loss of information.  Each component carries a ``type`` discriminator field
    that is resolved at deserialisation time against the corresponding registry.

    Registering a new estimator type is a one-liner::

        @estimator_config("my_gptq_variant")
        class MyGptqConfig(EstimatorConfig):
            type: Literal["my_gptq_variant"] = "my_gptq_variant"
            group_size: int = 64

            def build(self) -> QuantizationParamEstimator:
                return MyGptqEstimator(group_size=self.group_size)

    The resulting config can then be expressed in JSON as::

        {"estimator": {"type": "my_gptq_variant", "group_size": 64}, ...}

    Attributes:
        calibration: Dataset and sampling hyperparameters.
        estimator: Estimator config; must carry a ``type`` registered via
            :func:`estimator_config`.
        adapter: Adapter config (default: ``weight_assignment``).
        kernel_lowering: Optional kernel lowering config (default: ``auto``).
            Set to ``None`` to keep raw QTensor weights without lowering.
        preprocessing: Ordered list of preprocessing step configs.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # SerializeAsAny is load-bearing, not cosmetic: these fields are declared as
    # base types, and Pydantic v2 serialises by the DECLARED type, so every
    # subclass field was dropped and the dump degenerated to {"type": ...}.  The
    # recipe cache fingerprint hashes this JSON, so all VPTQ recipes shared one
    # cache scope and reused each other's codebooks (see tests).
    calibration: CalibrationConfig = CalibrationConfig()
    estimator: SerializeAsAny[EstimatorConfig]
    adapter: SerializeAsAny[AdapterConfig] = Field(
        default_factory=WeightAssignmentAdapterConfig
    )
    kernel_lowering: SerializeAsAny[KernelLoweringConfig] | None = Field(
        default_factory=AutoKernelLoweringConfig
    )
    preprocessing: list[SerializeAsAny[PreprocessingStepConfig]] = []

    @model_validator(mode="before")
    @classmethod
    def _resolve_polymorphic(cls, data: Any) -> Any:
        """Dispatch polymorphic config dicts to their registered types."""
        if not isinstance(data, dict):
            return data
        if "estimator" in data:
            data["estimator"] = _resolve_one(
                ESTIMATOR_CONFIG_REGISTRY, data["estimator"], "estimator"
            )
        if "adapter" in data:
            data["adapter"] = _resolve_one(
                ADAPTER_CONFIG_REGISTRY, data["adapter"], "adapter"
            )
        if "kernel_lowering" in data and data["kernel_lowering"] is not None:
            data["kernel_lowering"] = _resolve_one(
                KERNEL_LOWERING_CONFIG_REGISTRY,
                data["kernel_lowering"],
                "kernel lowering",
            )
        if "preprocessing" in data:
            data["preprocessing"] = [
                _resolve_one(PREPROCESSING_CONFIG_REGISTRY, step, "preprocessing step")
                for step in data["preprocessing"]
            ]
        return data

    @classmethod
    def from_json(cls, s: str) -> QuantizationRecipeConfig:
        """Deserialise a :class:`QuantizationRecipeConfig` from a JSON string.

        Args:
            s: JSON produced by :meth:`model_dump_json` or equivalent.

        Returns:
            A validated :class:`QuantizationRecipeConfig` with all component
            types resolved from their respective registries.
        """
        return cls.model_validate_json(s)

    def build(self) -> QuantizationRecipe:
        """Build a :class:`.QuantizationRecipe` from this configuration.

        Returns:
            A fully constructed :class:`.QuantizationRecipe` ready to be
            passed to :func:`.t2n.registry.run_quantization_pipeline`.
        """
        return QuantizationRecipe(
            calibration=self.calibration,
            param_estimator=self.estimator.build(),
            adapter=self.adapter.build(),
            kernel_lowering=(
                self.kernel_lowering.build()
                if self.kernel_lowering is not None
                else None
            ),
            preprocessing=[step.build() for step in self.preprocessing],
        )


# ---------------------------------------------------------------------------
# Composite estimator config
# ---------------------------------------------------------------------------


@estimator_config("composite")
class CompositeEstimatorConfig(EstimatorConfig):
    """JSON-serialisable config for :class:`.CompositeEstimator`.

    Composes multiple sub-estimator configs, each targeting a different set
    of module classes.  Resolves nested ``methods`` entries from
    :data:`ESTIMATOR_CONFIG_REGISTRY` during deserialisation.

    Example JSON::

        {
            "type": "composite",
            "methods": [
                {"type": "gptq",             "target": "linear"},
                {"type": "gptq_panel",       "target": "conv2d"},
                {"type": "min_max_zp_weight", "target": "embedding"}
            ]
        }

    Attributes:
        type: Discriminator field, always ``"composite"``.
        methods: Ordered list of sub-estimator configs.
    """

    type: Literal["composite"] = "composite"
    methods: list[EstimatorConfig] = []

    @model_validator(mode="before")
    @classmethod
    def _resolve_methods(cls, data: Any) -> Any:
        """Resolve each entry in *methods* from :data:`ESTIMATOR_CONFIG_REGISTRY`."""
        if isinstance(data, dict) and "methods" in data:
            data["methods"] = [
                _resolve_one(ESTIMATOR_CONFIG_REGISTRY, m, "estimator")
                for m in data["methods"]
            ]
        return data

    def build(self) -> QuantizationParamEstimator:
        """Build a :class:`.CompositeEstimator` from this configuration.

        Returns:
            A :class:`.CompositeEstimator` wrapping all configured
            sub-estimators.
        """
        return CompositeEstimator(estimators=[m.build() for m in self.methods])
