"""QuantizationPlan: result of parameter estimation."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from .qscheme.group import QSchemeGroup
from .qscheme.zp_scale import ZpScaleScheme

# Weight/activation assignments map target keys to one of:
# - ZpScaleScheme  : standard affine quantization (per-tensor, per-channel, per-group)
# - QSchemeGroup   : mixed-precision panel layout
# - torch.Tensor   : pre-packed quantized tensor (e.g. tract Q4_0)
_Assignment = ZpScaleScheme | QSchemeGroup | torch.Tensor


class QuantizationPlan(BaseModel):
    """Maps quantization targets to concrete QScheme instances.

    ``weight_assignments``    : WeightTarget.key  → QScheme or QSchemeGroup or Tensor
    ``activation_assignments``: ActivationTarget.key → QScheme or QSchemeGroup
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    weight_assignments: dict[str, _Assignment] = {}
    activation_assignments: dict[str, _Assignment] = {}

    def is_empty(self) -> bool:
        """Return True if no assignments have been made."""
        return not self.weight_assignments and not self.activation_assignments
