"""Base calibration method abstractions.

CalibrationMethod is the root ABC for all weight-quantization calibration
algorithms.  Subclasses implement ``update_statistics`` and
``compute_qscheme``.

The ``CalibrationObserve`` enum controls whether a method observes weights
directly or input activations (e.g. Hessian-based methods).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from torch import nn

from ..qscheme.descriptor import Granularity, GranularityType

METHOD_REGISTRY: dict[str, type] = {}


class CalibrationObserve(StrEnum):
    """Controls which data a CalibrationMethod observes during calibration."""

    WEIGHT = "weight"
    ACTIVATION = "activation"


@dataclass
class CalibrationMethod[T](ABC):
    """Root ABC for weight calibration methods.

    Fields set at construction time (configuration):
        granularity: quantization granularity.

    Fields set externally before calibration (by observer/hook system):
        weight_tensor: the nn.Parameter being quantized.
        module_class:  Python class of the owning module.
        module:        owning nn.Module instance.
        name:          human-readable name for logging.

    Fields computed lazily on first ``update()`` call:
        calibration_data_shape, qscheme_data_shape, reduction_axes.
    """

    granularity: Granularity

    # Set externally before use
    weight_tensor: torch.Tensor | None = field(default=None, repr=False)
    module_class: type | None = field(default=None, repr=False)
    module: nn.Module | None = field(default=None, repr=False)
    name: str = field(default="")

    # Computed lazily
    calibration_data_shape: list[int] | None = field(default=None, repr=False)
    qscheme_data_shape: list[int] | None = field(default=None, repr=False)
    reduction_axes: list[int] | None = field(default=None, repr=False)

    # Calibration observe mode (weight or activation).
    observe: CalibrationObserve = field(
        default_factory=lambda: CalibrationObserve.WEIGHT
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Reset calibration state.  Called before each calibration pass."""
        self.calibration_data_shape = None
        self.qscheme_data_shape = None
        self.reduction_axes = None

    def update(self, data: torch.Tensor) -> None:
        """Update statistics with *data*.

        Lazily computes calibration shapes on the first call, then
        delegates to ``update_statistics``.
        """
        if self.calibration_data_shape is None:
            self._compute_shapes_from_data(data)
        self.update_statistics(data)

    @abstractmethod
    def update_statistics(self, data: torch.Tensor) -> None:
        """Accumulate statistics from *data* (already reshaped if needed)."""

    @abstractmethod
    def compute_qscheme(self) -> T:
        """Return the estimated QScheme from accumulated statistics."""

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _compute_shapes_from_data(self, data: torch.Tensor) -> None:
        """Derive 4-D calibration / qscheme shapes from *data* + granularity.

        The 4-D pattern is [rows, 1, n_groups, group_size] where:
          - dim 0 (rows)    : kept
          - dim 1 (1)       : collapsed (dummy singleton)
          - dim 2 (n_groups): kept
          - dim 3 (group_sz): collapsed
        """
        if data.dim() == 1:
            rows, cols = 1, data.shape[0]
        else:
            rows = data.shape[0]
            cols = int(math.prod(data.shape[1:]))

        gt = self.granularity.granularity_type
        if gt is GranularityType.ALL:
            self.calibration_data_shape = [1, 1, 1, rows * cols]
        elif gt is GranularityType.PER_CHANNEL:
            self.calibration_data_shape = [rows, 1, 1, cols]
        elif gt is GranularityType.PER_GROUP:
            assert self.granularity.group_sizes
            g = self.granularity.group_sizes[0]
            if cols % g != 0:
                raise ValueError(f"cols={cols} is not divisible by group_size={g}")
            self.calibration_data_shape = [rows, 1, cols // g, g]
        else:
            raise NotImplementedError(gt)

        rank = len(self.calibration_data_shape)
        self.reduction_axes = list(range(1, rank, 2))
        self.qscheme_data_shape = [
            self.calibration_data_shape[i] if i not in self.reduction_axes else 1
            for i in range(rank)
        ]

    def update_qscheme_shapes_with(self, weight: torch.Tensor) -> None:
        """Recompute shapes from *weight* (called after weight mutation in GPTQ)."""
        self._compute_shapes_from_data(weight)


@dataclass
class CalibrationMethodOnActivation[T](CalibrationMethod[T]):
    """Base class for calibration methods that observe input activations."""

    observe: CalibrationObserve = field(
        default_factory=lambda: CalibrationObserve.ACTIVATION
    )
