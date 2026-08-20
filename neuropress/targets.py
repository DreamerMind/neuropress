"""Quantization targets: what gets quantized."""

from __future__ import annotations

from pydantic import BaseModel


class WeightTarget(BaseModel):
    """A module parameter to be quantized (e.g., nn.Linear.weight)."""

    module_id: str  # QNode.id that owns the parameter
    param_name: str  # e.g. "weight"

    @property
    def key(self) -> str:
        """Unique string key for this target."""
        return f"{self.module_id}/{self.param_name}"


class ActivationTarget(BaseModel):
    """A produced activation tensor (QValue) to be quantized."""

    value_id: str  # QValue.id

    @property
    def key(self) -> str:
        """Unique string key for this target."""
        return self.value_id
