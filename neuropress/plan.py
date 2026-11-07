"""QuantizationPlan: result of parameter estimation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict

from .qscheme.group import QSchemeGroup
from .qscheme.zp_scale import ZpScaleScheme

F16_MIN_POSITIVE: float = 2.0**-24
"""Smallest positive f16 value, used for scales that would round to zero."""


@dataclass(frozen=True)
class QuantizedWeight:
    """Selected codes and scheme, with an optional destination format.

    Codes retain the original weight shape. Mixed schemes describe column
    slices of its flattened matrix. Storage conversion belongs to the adapter.
    Callable destinations support legacy Python weight transforms.

    Codes wait for the adapter on the CPU, as they take one byte per weight for
    the whole model; the scheme stays where it was computed.

    transposed marks a weight stored as the transpose of its scheme's layout,
    as in the transformers Conv1D: scheme groups then run along the weight's
    first dimension, which flat per-block storage formats cannot express.
    """

    codes: torch.Tensor
    scheme: ZpScaleScheme | QSchemeGroup
    dtype: torch.dtype
    destination: str | Callable[[torch.Tensor, str], Any] | None = None
    transposed: bool = False

    @property
    def _scheme_codes(self) -> torch.Tensor:
        return self.codes.t() if self.transposed else self.codes

    def on_scheme_device(self) -> QuantizedWeight:
        """Return this assignment with its codes on the scheme's device."""
        scheme = self.scheme
        if isinstance(scheme, QSchemeGroup):
            scheme = scheme.panels[0].qscheme
        device = scheme.scale.device
        if self.codes.device == device:
            return self
        return replace(self, codes=self.codes.to(device))

    def __post_init__(self) -> None:
        if self.codes.dtype != torch.uint8:
            raise ValueError("Quantized weight codes must be uint8")
        if isinstance(self.scheme, QSchemeGroup):
            columns = self.codes.numel() // self._scheme_codes.shape[0]
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
        scheme_codes = self.on_scheme_device()._scheme_codes
        if isinstance(self.scheme, ZpScaleScheme):
            result = self.scheme.dequantize(scheme_codes, self.dtype)
            return result.t() if self.transposed else result
        codes = scheme_codes.reshape(scheme_codes.shape[0], -1)
        result = torch.empty_like(codes, dtype=self.dtype)
        for panel in self.scheme.panels:
            region = panel.region
            if region.dim != 1:
                raise ValueError("Mixed weights require column panels")
            result[:, region.start : region.end] = panel.qscheme.dequantize(
                codes[:, region.start : region.end], self.dtype
            )
        result = result.reshape(scheme_codes.shape)
        return result.t() if self.transposed else result


class QuantizationPlan(BaseModel):
    """Selected weight values and activation schemes.

    Weight assignments contain codes and a scheme, or an already-materialized
    tensor. Activation assignments contain affine schemes or panel layouts.
    kept_float names selected layers that stayed float, with the reason. Size
    estimates are computed from the recipe alone and do not reflect them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    weight_assignments: dict[str, QuantizedWeight | torch.Tensor] = {}
    activation_assignments: dict[str, ZpScaleScheme | QSchemeGroup | torch.Tensor] = {}
    kept_float: dict[str, str] = {}

    def is_empty(self) -> bool:
        """Return True if no assignments have been made."""
        return not self.weight_assignments and not self.activation_assignments
