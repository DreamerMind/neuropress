"""ChannelMaxCollector: running per-channel maximum absolute value accumulator."""

from __future__ import annotations

import torch

from .collectors import Collector


class ChannelMaxCollector(Collector):
    """Tracks running per-channel maximum absolute value over all data batches.

    Used by the ``"channel_max"`` stat type for SmoothQuant scale computation.
    Input tensors of any rank are flattened to ``[N, in_features]`` before the
    per-channel max is accumulated.
    """

    def __init__(self) -> None:
        """Initialize with no accumulated data."""
        self._max: torch.Tensor | None = None

    def update(self, data: torch.Tensor) -> None:
        """Accumulate per-channel max of ``|data|``.

        Args:
            data: Activation tensor of shape ``[*, in_features]``.
        """
        x_flat = data.reshape(-1, data.shape[-1])
        channel_max = x_flat.abs().amax(dim=0)
        if self._max is None:
            self._max = channel_max
        else:
            self._max = torch.maximum(self._max, channel_max)

    def result(self) -> dict[str, torch.Tensor | None]:
        """Return ``{"channel_max": Tensor[in_features]}`` or ``None`` if empty."""
        return {"channel_max": self._max}
