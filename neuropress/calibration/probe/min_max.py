"""MinMaxCollector: running global min/max statistics accumulator."""

from __future__ import annotations

import torch

from .collectors import Collector


class MinMaxCollector(Collector):
    """Tracks running global min and max over all data batches."""

    def __init__(self) -> None:
        """Initialize with no accumulated data."""
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None

    def update(self, data: torch.Tensor) -> None:
        """Accumulate the min and max of *data*."""
        mn = data.min()
        mx = data.max()
        self._min = mn if self._min is None else torch.minimum(self._min, mn)
        self._max = mx if self._max is None else torch.maximum(self._max, mx)

    def result(self) -> dict[str, torch.Tensor | None]:
        """Return ``{"min": ..., "max": ...}`` tensors."""
        return {"min": self._min, "max": self._max}
