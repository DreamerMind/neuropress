"""Preprocessing registry: model transformations applied before calibration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel
from torch import nn

from .methods.smooth_quant import apply_smooth_quant, smooth_down_proj

# ---------------------------------------------------------------------------
# Runtime step hierarchy
# ---------------------------------------------------------------------------


class PreprocessingStep(ABC):
    """A model transformation applied before calibration statistics are collected."""

    @abstractmethod
    def apply(self, model: nn.Module, calibration_data: list[dict[str, Any]]) -> None:
        """Apply transformation to *model* in-place.

        Args:
            model: The model to transform.
            calibration_data: Calibration batches used by transformations that
                require activation statistics (e.g. SmoothQuant).
        """


@dataclass
class SmoothQuantStep(PreprocessingStep):
    """SmoothQuant (Xiao et al., ICML 2023) applied before calibration.

    Attributes:
        alpha: Migration balance [0, 1]. 0.5 = equal split (default).
        prev_modules: Optional map from linear layer name to its preceding
            layer for scale absorption.
        module_filter: Optional callable ``(name, module) -> bool`` to filter
            which ``nn.Linear`` layers are processed.
    """

    alpha: float = field(default=0.5)
    prev_modules: dict[str, nn.Linear | nn.LayerNorm] | None = field(default=None)
    module_filter: Callable[[str, nn.Linear], bool] | None = field(default=None)

    def apply(self, model: nn.Module, calibration_data: list[dict[str, Any]]) -> None:
        """Apply SmoothQuant to all qualifying ``nn.Linear`` layers in *model*.

        Args:
            model: The model to transform in-place.
            calibration_data: Calibration batches for activation statistics.
        """
        apply_smooth_quant(
            model,
            calibration_data,
            self.alpha,
            self.prev_modules,
            self.module_filter,
        )


@dataclass
class DownProjSmoothStep(PreprocessingStep):
    """SmoothQuant scale migration restricted to MLP ``down_proj`` layers.

    ``down_proj`` is where activation outliers concentrate, and water-filling
    shows a per-channel scale recovers a large share of the exploitable
    anisotropy there.  Note that the anisotropy worth exploiting lives in the
    *Hessian* spectrum, not the weight spectrum - measuring the latter makes the
    diagonal look far less useful than it is.

    Restricted to ``down_proj`` deliberately: it is the highest-value target and
    the only one whose migration is exact without touching a norm, since ``up``
    enters ``silu(gate) * up`` linearly.  Extending to the norm-fed projections
    would first require ``absorb_scale_into_prev`` to handle ``1 + weight``
    RMSNorm.

    Attributes:
        alpha: Migration balance in ``[0, 1]``.  The right value for weight-only
            VQ with a *shared* codebook is not settled - under a relative-error
            model the transform is distortion-neutral, so this needs sweeping
            rather than assuming the 0.5 default from the paper.
    """

    alpha: float = field(default=0.5)
    normalize_geomean: bool = field(default=True)

    def apply(self, model: nn.Module, calibration_data: list[dict[str, Any]]) -> None:
        """Migrate per-channel scale on every ``down_proj`` in *model*.

        Args:
            model: The model to transform in-place.
            calibration_data: Batches used for channel-max statistics.
        """
        smooth_down_proj(
            model,
            calibration_data,
            alpha=self.alpha,
            normalize_geomean=self.normalize_geomean,
        )


@dataclass
class HadamardRotationStep(PreprocessingStep):
    """Hadamard incoherence processing applied before calibration.

    Rotates the residual stream so weights become approximately i.i.d.
    Gaussian, which is the precondition high vector-length VQ needs.  Running
    as Step 0 matters: the Hessian is then estimated in the rotated basis,
    which is the basis the quantizer will actually work in.

    Requires a pre-norm architecture whose sublayers end in a plain ``Linear``
    (Qwen3.5 qualifies; Gemma-3/4 do not - their sandwich norms block the
    fold).  The transform is exactly output-preserving.

    Attributes:
        seed: Base seed for the rotation; the online rotation uses ``seed + 1``.
        rotate_down_proj: Whether to add the online rotation of the MLP
            intermediate, where activation outliers concentrate.
        plus_one_norms: Whether norms use the ``1 + weight`` convention.
    """

    seed: int = field(default=1234)
    rotate_down_proj: bool = field(default=True)
    plus_one_norms: bool = field(default=True)

    def apply(self, model: nn.Module, calibration_data: list[dict[str, Any]]) -> None:
        """Rotate *model* in place.  Calibration data is unused (data-free).

        Args:
            model: The model to transform in-place.
            calibration_data: Ignored; the rotation needs no statistics.
        """
        # Imported lazily: `neuropress.transforms` pulls in `t2n`, which cycles
        # back through the config registries onto this module.
        from .transforms.rotation_surgery import apply_hadamard_rotation

        apply_hadamard_rotation(
            model,
            seed=self.seed,
            rotate_down_proj=self.rotate_down_proj,
            plus_one_norms=self.plus_one_norms,
        )


# ---------------------------------------------------------------------------
# Serialisable config hierarchy (JSON-round-trippable counterparts)
# ---------------------------------------------------------------------------


class PreprocessingStepConfig(BaseModel):
    """Base class for JSON-serialisable preprocessing step configurations.

    Concrete subclasses must implement :meth:`build` and self-register with
    :func:`preprocessing_step_config`.
    """

    type: str

    def build(self) -> PreprocessingStep:
        """Build the runtime :class:`PreprocessingStep` from this config.

        Returns:
            A ready-to-use :class:`PreprocessingStep` instance.
        """
        raise NotImplementedError(f"{type(self).__name__}.build()")


PREPROCESSING_CONFIG_REGISTRY: dict[str, type[PreprocessingStepConfig]] = {}


def preprocessing_step_config(
    name: str,
) -> Callable[[type[PreprocessingStepConfig]], type[PreprocessingStepConfig]]:
    """Class decorator: register a :class:`PreprocessingStepConfig` subclass by name.

    Args:
        name: Registry key — must match the ``type`` field value.

    Returns:
        The unmodified class (registration is a side-effect).
    """

    def _decorate(
        cls: type[PreprocessingStepConfig],
    ) -> type[PreprocessingStepConfig]:
        PREPROCESSING_CONFIG_REGISTRY[name] = cls
        return cls

    return _decorate


@preprocessing_step_config("smooth_quant")
class SmoothQuantStepConfig(PreprocessingStepConfig):
    """JSON-serialisable config for :class:`SmoothQuantStep`.

    Attributes:
        type: Discriminator field, always ``"smooth_quant"``.
        alpha: SmoothQuant migration balance in [0, 1].
    """

    type: Literal["smooth_quant"] = "smooth_quant"
    alpha: float = 0.5

    def build(self) -> SmoothQuantStep:
        """Build a :class:`SmoothQuantStep` from this config.

        Returns:
            A :class:`SmoothQuantStep` with the configured alpha.
        """
        return SmoothQuantStep(alpha=self.alpha)


@preprocessing_step_config("down_proj_smooth")
class DownProjSmoothStepConfig(PreprocessingStepConfig):
    """JSON-serialisable config for :class:`DownProjSmoothStep`.

    Attributes:
        type: Discriminator field, always ``"down_proj_smooth"``.
        alpha: Migration balance in ``[0, 1]``.
    """

    type: Literal["down_proj_smooth"] = "down_proj_smooth"
    alpha: float = 0.5
    normalize_geomean: bool = True

    def build(self) -> DownProjSmoothStep:
        """Build a :class:`DownProjSmoothStep` from this config.

        Returns:
            A :class:`DownProjSmoothStep` with the configured settings.
        """
        return DownProjSmoothStep(
            alpha=self.alpha, normalize_geomean=self.normalize_geomean
        )


@preprocessing_step_config("hadamard_rotation")
class HadamardRotationStepConfig(PreprocessingStepConfig):
    """JSON-serialisable config for :class:`HadamardRotationStep`.

    Attributes:
        type: Discriminator field, always ``"hadamard_rotation"``.
        seed: Base seed for the rotation.
        rotate_down_proj: Whether to add the online MLP-intermediate rotation.
        plus_one_norms: Whether norms use the ``1 + weight`` convention.
    """

    type: Literal["hadamard_rotation"] = "hadamard_rotation"
    seed: int = 1234
    rotate_down_proj: bool = True
    plus_one_norms: bool = True

    def build(self) -> HadamardRotationStep:
        """Build a :class:`HadamardRotationStep` from this config.

        Returns:
            A :class:`HadamardRotationStep` with the configured settings.
        """
        return HadamardRotationStep(
            seed=self.seed,
            rotate_down_proj=self.rotate_down_proj,
            plus_one_norms=self.plus_one_norms,
        )


# ---------------------------------------------------------------------------
# Runtime registry (step class lookup by name)
# ---------------------------------------------------------------------------

PREPROCESSING_REGISTRY: dict[str, type[PreprocessingStep]] = {
    "smooth_quant": SmoothQuantStep,
    "hadamard_rotation": HadamardRotationStep,
    "down_proj_smooth": DownProjSmoothStep,
}
