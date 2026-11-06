"""QuantizationPlan: result of parameter estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from .qscheme.group import QSchemeGroup
from .qscheme.zp_scale import ZpScaleScheme


@dataclass(frozen=True)
class QuantizedWeight:
    """Selected codes and scheme, with an optional destination format.

    Codes retain the original weight shape. Mixed schemes describe column
    slices of its flattened matrix. Storage conversion belongs to the adapter.
    Callable destinations support legacy Python weight transforms.
    """

    codes: torch.Tensor
    scheme: ZpScaleScheme | QSchemeGroup
    dtype: torch.dtype
    destination: str | Callable[[torch.Tensor, str], Any] | None = None

    def __post_init__(self) -> None:
        if self.codes.dtype != torch.uint8:
            raise ValueError("Quantized weight codes must be uint8")
        if isinstance(self.scheme, QSchemeGroup):
            columns = self.codes.numel() // self.codes.shape[0]
            offset = 0
            for panel in sorted(self.scheme.panels, key=lambda p: p.region.start):
                region = panel.region
                if region.dim != 1 or region.start != offset or region.end <= offset:
                    raise ValueError(
                        "Mixed panels must cover consecutive, non-overlapping columns"
                    )
                offset = region.end
            if offset != columns:
                raise ValueError("Mixed panels must cover the entire weight matrix")

    def dequantize(self) -> torch.Tensor:
        """Materialize the selected values without recalibration."""
        if isinstance(self.scheme, ZpScaleScheme):
            return self.scheme.dequantize(self.codes, self.dtype)
        codes = self.codes.reshape(self.codes.shape[0], -1)
        result = torch.empty_like(codes, dtype=self.dtype)
        for panel in self.scheme.panels:
            region = panel.region
            if region.dim != 1:
                raise ValueError("Mixed weights require column panels")
            result[:, region.start : region.end] = panel.qscheme.dequantize(
                codes[:, region.start : region.end], self.dtype
            )
        return result.reshape(self.codes.shape)


class QuantizationPlan(BaseModel):
    """Selected weight values and activation schemes.

    Weight assignments contain codes and a scheme, or an already-materialized
    tensor. Activation assignments contain affine schemes or panel layouts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    weight_assignments: dict[str, QuantizedWeight | torch.Tensor] = {}
    activation_assignments: dict[str, ZpScaleScheme | QSchemeGroup | torch.Tensor] = {}

    def is_empty(self) -> bool:
        """Return True if no assignments have been made."""
        return not self.weight_assignments and not self.activation_assignments
