"""ModuleQuantizationAdapter: ABC and concrete adapters for applying a plan."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_to_nnef.tensor.updater import ModTensorUpdater

from .plan import QuantizationPlan, QuantizedWeight
from .qscheme.group import QSchemeGroup
from .qscheme.zp_scale import ZpScaleScheme

__all__ = [
    "CompositeAdapter",
    "ModuleQuantizationAdapter",
    "WeightAssignmentAdapter",
]

LOGGER = logging.getLogger(__name__)


class ModuleQuantizationAdapter(ABC):
    """Applies a QuantizationPlan to a model in-place or returns a new model.

    Responsibilities:
    - Rewrite weights (replace float params with quantized/dequantized values).
    - Insert activation quantization wrappers.
    - Handle QSchemeGroup (multi-panel) assignments.
    - Preserve the module's public API (forward signature unchanged).
    """

    @abstractmethod
    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        """Apply *plan* to *model* and return the adapted model."""


class WeightAssignmentAdapter(ModuleQuantizationAdapter):
    """Materialize plan assignments and apply them without changing tied siblings.

    Scalar assignments are packed for their destination format or dequantized
    when no format is selected. Mixed affine panels replace named Linear
    modules with homogeneous-precision branches. Pre-packed tensors remain
    valid assignments.
    """

    @torch.no_grad()
    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        """Swap weights listed in *plan* into *model*.

        Args:
            model: Model whose weights will be replaced in-place.
            plan: Maps module names to their replacement weight tensors.

        Returns:
            The same *model* with the replaced weights.
        """
        if plan.is_empty():
            return model
        from .transforms.vq import QTensorTractVQ

        module_map = dict(model.named_modules())
        updater = ModTensorUpdater(model)
        assignments = list(plan.weight_assignments.items())
        total = len(assignments)
        log_every = max(1, total // 10)
        for i, (name, q_weight) in enumerate(assignments):
            mod = module_map.get(name)
            if mod is None:
                continue
            if isinstance(q_weight, QuantizedWeight):
                if isinstance(q_weight.scheme, QSchemeGroup) and isinstance(
                    mod, nn.Linear
                ):
                    self._apply_mixed(model, name, mod, q_weight)
                    continue
                q_weight = self._convert(q_weight, name)
                if q_weight is None:
                    continue
            if isinstance(q_weight, QTensorTractVQ):
                # ModTensorUpdater calls new.dtype in both check_same_dtype and
                # _set_tensor's fast-path check, triggering __torch_function__ →
                # decompress() → OOM for large VQ tensors (e.g. embed_tokens,
                # lm_head).  Direct _parameters assignment is equivalent to the
                # slow-path branch that _set_tensor would take anyway.
                mod._parameters["weight"] = q_weight
            else:
                # update_by_ref's tie_replacements-always-True behaviour
                # propagates this update to every module sharing the same
                # underlying tensor (e.g. a tied lm_head when only
                # embed_tokens is in the plan) — the plan only lists
                # `name`, so any such sibling was never meant to change.
                # update_by_name(..., tie_replacements=False) updates only
                # the named module, matching the QTensorTractVQ branch above.
                updater.update_by_name(
                    f"{name}.weight" if name else "weight",
                    q_weight,
                    tie_replacements=False,
                    enforce_tensor_consistency=False,
                )
            if (i + 1) % log_every == 0 or (i + 1) == total:
                LOGGER.info(
                    "WeightAssignmentAdapter: %d/%d weights replaced (%s)",
                    i + 1,
                    total,
                    name,
                )
        return model

    @staticmethod
    def _convert(weight: QuantizedWeight, name: str) -> torch.Tensor | None:
        if weight.destination is None:
            return weight.dequantize()
        if callable(weight.destination):
            return weight.destination(weight.dequantize(), name)
        from .estimator_configs import WEIGHT_TRANSFORM_REGISTRY
        from .transforms import weight_only  # noqa: F401

        return WEIGHT_TRANSFORM_REGISTRY[weight.destination].transform(weight, name)

    @staticmethod
    def _apply_mixed(
        model: nn.Module, name: str, mod: nn.Linear, weight: QuantizedWeight
    ) -> None:
        from .t2n.model_io import QuantizationManifest
        from .transforms.split_mixed_linear import MixedPrecisionLinear
        from .transforms.weight_only import (
            _affine_from_assignment,
            _q4_0_from_assignment,
        )

        if not name:
            raise ValueError("Mixed-precision Linear must be a named child module")
        if not isinstance(weight.scheme, QSchemeGroup):
            raise ValueError("Mixed linear assignment requires panel schemes")
        buckets = {}
        for panel in weight.scheme.panels:
            if not isinstance(panel.qscheme, ZpScaleScheme) or panel.region.dim != 1:
                raise ValueError("Mixed linear requires affine column panels")
            buckets.setdefault(panel.qscheme.nbits, []).append(panel)
        sub_linears = nn.ModuleList()
        permutation = []
        sizes = []
        precisions = sorted(buckets, reverse=True)
        for bits in precisions:
            panels = sorted(buckets[bits], key=lambda p: p.region.start)
            columns = [c for p in panels for c in range(p.region.start, p.region.end)]
            permutation.extend(columns)
            sizes.append(len(columns))
            codes = weight.codes[:, columns]
            scales = torch.cat([p.qscheme.scale for p in panels], dim=2)
            zeros = torch.cat([p.qscheme.zero_point for p in panels], dim=2)
            width = panels[0].qscheme.calibration_data_shape[-1]
            scheme = ZpScaleScheme(
                zeros,
                scales,
                bits,
                [codes.shape[0], 1, len(columns) // width, width],
                list(scales.shape),
            )
            sub = QuantizedWeight(codes, scheme, weight.dtype)
            path = f"{name}.sub_linears.{len(sub_linears)}"
            q = (
                _q4_0_from_assignment(sub, path, packed=True)
                if bits == 4 and torch.all(zeros == 8)
                else _affine_from_assignment(sub, path, nbits=bits)
            )
            linear = nn.Linear(
                len(columns),
                mod.out_features,
                bias=False,
                device=codes.device,
                dtype=weight.dtype,
            )
            ModTensorUpdater(linear).update_by_ref(
                linear.weight, q, enforce_tensor_consistency=False
            )
            sub_linears.append(linear)
        perm = torch.tensor(permutation, dtype=torch.long, device=weight.codes.device)
        mixed = MixedPrecisionLinear(perm, sub_linears, sizes, precisions, mod.bias)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, mixed)
        if not hasattr(model, "__neuropress_manifest__"):
            model.__neuropress_manifest__ = QuantizationManifest()
        from .t2n.model_io import MixedPrecisionModuleSpec, MixedPrecisionSplitTransform

        model.__neuropress_manifest__.transforms.append(
            MixedPrecisionSplitTransform(
                type="mixed_precision_split",
                modules={
                    name: MixedPrecisionModuleSpec(
                        split_sizes=sizes,
                        precision_nbits=precisions,
                        out_features=mod.out_features,
                        has_bias=mod.bias is not None,
                    )
                },
            )
        )


class CompositeAdapter(ModuleQuantizationAdapter):
    """Chains multiple adapters, passing the model through each in order.

    Attributes:
        adapters: Ordered sequence of adapters to apply.
    """

    def __init__(self, adapters: list[ModuleQuantizationAdapter]) -> None:
        """Initialise with an ordered list of *adapters*."""
        self.adapters = adapters

    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        """Apply each adapter in sequence, threading the model through.

        Args:
            model: Model to adapt.
            plan: Passed unchanged to every adapter.

        Returns:
            The adapted model after all adapters have been applied.
        """
        for adapter in self.adapters:
            model = adapter.apply(model, plan)
        return model
