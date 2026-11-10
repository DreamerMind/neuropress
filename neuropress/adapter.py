"""ModuleQuantizationAdapter: ABC and concrete adapters for applying a plan."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch_to_nnef.exceptions import T2NErrorImpossibleQuantization
from torch_to_nnef.tensor.offload import OffloadedTensor
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
    """Materialize plan assignments and apply them to the modules they name.

    Scalar assignments are packed for their destination format or dequantized
    when no format is selected. Mixed affine panels replace named Linear
    modules with homogeneous-precision branches. Pre-packed tensors remain
    valid assignments. Tied siblings the plan does not name stay unchanged.
    A weight its destination cannot store stays float rather than discarding
    the other estimated layers, and is recorded in the plan's kept_float.
    """

    @torch.no_grad()
    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        """Swap weights listed in *plan* into *model*.

        Tied modules listed in the plan hold the same assignment object and
        receive one shared tensor or one shared set of mixed branches.

        Args:
            model: Model whose weights will be replaced in-place.
            plan: Maps module names to their replacement weight tensors.

        Returns:
            The same *model* with the replaced weights.
        """
        if plan.is_empty():
            self._report_kept_float(plan)
            return model
        from .transforms.split_mixed_linear import record_mixed_split
        from .transforms.vq import QTensorTractVQ

        module_map = dict(model.named_modules())
        assignments = list(plan.weight_assignments.items())
        total = len(assignments)
        log_every = max(1, total // 10)
        converted_tensors: dict[int, torch.Tensor] = {}
        converted_branches: dict[int, tuple[str, Any]] = {}
        for i, (name, q_weight) in enumerate(assignments):
            mod = module_map.get(name)
            if mod is None:
                continue
            key = id(q_weight)
            if isinstance(q_weight, QuantizedWeight):
                q_weight = q_weight.on_scheme_device()
                if isinstance(q_weight.scheme, QSchemeGroup) and isinstance(
                    mod, nn.Linear
                ):
                    source, branches = converted_branches.get(key, (name, None))
                    mixed = self._mixed_linear(
                        name, mod, q_weight, branches, plan.kept_float
                    )
                    converted_branches[key] = (
                        source,
                        (mixed.column_perm, mixed.sub_linears),
                    )
                    self._replace_module(model, name, mixed)
                    record_mixed_split(
                        model,
                        {name: mixed},
                        None if source == name else {name: source},
                    )
                    continue
                if key in converted_tensors:
                    mod._parameters["weight"] = converted_tensors[key]
                    continue
                q_weight = self._convert(q_weight, name, mod.weight, plan.kept_float)
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
                converted_tensors[key] = self._assign_weight(mod, q_weight)
            if (i + 1) % log_every == 0 or (i + 1) == total:
                LOGGER.info(
                    "WeightAssignmentAdapter: %d/%d weights replaced (%s)",
                    i + 1,
                    total,
                    name,
                )
        self._report_kept_float(plan)
        return model

    @staticmethod
    def _assign_weight(mod: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        """Give *mod* its own weight *tensor*, leaving tied siblings unchanged.

        The plan lists each module that changes, so a sibling sharing the old
        tensor (e.g. a tied lm_head when only embed_tokens is listed) keeps
        it. A tied sibling listed with its own assignment receives its own
        tensor as well. This matches ``ModTensorUpdater.update_by_name`` with
        ``tie_replacements=False``, whose bookkeeping cannot update the second
        name of a tied pair.
        """
        old = mod._parameters.get("weight")
        if not isinstance(tensor, nn.Parameter):
            tensor = nn.Parameter(tensor, requires_grad=False)
        if old is not None and old.requires_grad and not tensor.requires_grad:
            tensor.requires_grad = True
        mod._parameters["weight"] = tensor
        return tensor

    @staticmethod
    def _report_kept_float(plan: QuantizationPlan) -> None:
        if plan.kept_float:
            LOGGER.warning(
                "%d selected layers stayed float, which size estimates do not "
                "reflect: %s",
                len(plan.kept_float),
                ", ".join(sorted(plan.kept_float)),
            )

    @staticmethod
    def _convert(
        weight: QuantizedWeight,
        name: str,
        current: torch.Tensor,
        kept_float: dict[str, str],
    ) -> torch.Tensor | None:
        try:
            result = WeightAssignmentAdapter._materialize(weight, name)
        except (T2NErrorImpossibleQuantization, ValueError) as exc:
            LOGGER.error("'%s' keeps its selected values as float: %s", name, exc)
            kept_float[name] = str(exc)
            result = weight.dequantize()
        if isinstance(current, OffloadedTensor) and not isinstance(
            result, OffloadedTensor | type(None)
        ):
            # Name it after its own module: tied modules converted to different
            # formats start from one offloaded tensor, and OffloadedTensor
            # hashes by name, shape and dtype.
            result = OffloadedTensor.from_original_tensor(
                result,
                f"{name}.weight.quantized" if name else "weight.quantized",
                offload_dir=current.offload_dir,
            )
        return result

    @staticmethod
    def _materialize(weight: QuantizedWeight, name: str) -> torch.Tensor | None:
        if weight.destination is None:
            return weight.dequantize()
        if callable(weight.destination):
            return weight.destination(weight.dequantize(), name)
        from .estimator_configs import WEIGHT_TRANSFORM_REGISTRY
        from .transforms import weight_only  # noqa: F401

        return WEIGHT_TRANSFORM_REGISTRY[weight.destination].transform(weight, name)

    @staticmethod
    def _replace_module(model: nn.Module, name: str, module: nn.Module) -> None:
        if not name:
            raise ValueError("Mixed-precision Linear must be a named child module")
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, module)

    @staticmethod
    def _mixed_linear(
        name: str,
        mod: nn.Linear,
        weight: QuantizedWeight,
        branches: tuple[torch.Tensor, nn.ModuleList] | None,
        kept_float: dict[str, str],
    ) -> Any:
        """Build a split module, reusing a tied sibling's *branches* if given."""
        from .transforms.split_mixed_linear import (
            MixedPrecisionLinear,
            precision_column_layout,
        )

        if not isinstance(weight.scheme, QSchemeGroup):
            raise ValueError("Mixed linear assignment requires panel schemes")
        buckets = {}
        for panel in weight.scheme.panels:
            if not isinstance(panel.qscheme, ZpScaleScheme) or panel.region.dim != 1:
                raise ValueError("Mixed linear requires affine column panels")
            buckets.setdefault(panel.qscheme.nbits, []).append(panel)
        precisions, permutation, sizes = precision_column_layout(
            {
                bits: [(p.region.start, p.region.end) for p in panels]
                for bits, panels in buckets.items()
            }
        )
        if branches is None:
            perm = torch.tensor(
                permutation, dtype=torch.long, device=weight.codes.device
            )
            sub_linears = nn.ModuleList(
                WeightAssignmentAdapter._mixed_branch(
                    f"{name}.sub_linears.{index}",
                    sorted(buckets[bits], key=lambda p: p.region.start),
                    bits,
                    mod.out_features,
                    weight,
                    kept_float,
                )
                for index, bits in enumerate(precisions)
            )
        else:
            perm, sub_linears = branches
        bias = mod.bias.data.clone() if mod.bias is not None else None
        return MixedPrecisionLinear(perm, sub_linears, sizes, precisions, bias)

    @staticmethod
    def _mixed_branch(
        path: str,
        panels: list[Any],
        bits: int,
        out_features: int,
        weight: QuantizedWeight,
        kept_float: dict[str, str],
    ) -> nn.Linear:
        from .transforms.weight_only import (
            Q4_X_GROUP_SIZE,
            _affine_from_assignment,
            _q4_0_from_assignment,
        )

        columns = [c for p in panels for c in range(p.region.start, p.region.end)]
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
        aligned = width % Q4_X_GROUP_SIZE == 0
        try:
            if bits == 4 and aligned and torch.all(zeros == 8):
                q = _q4_0_from_assignment(sub, path, packed=True)
            else:
                q = _affine_from_assignment(
                    sub,
                    path,
                    nbits=bits,
                    block_size=Q4_X_GROUP_SIZE if aligned else width,
                )
                q.export_as_float = True
        except (T2NErrorImpossibleQuantization, ValueError) as exc:
            LOGGER.error("'%s' keeps its selected values as float: %s", path, exc)
            kept_float[path] = str(exc)
            q = sub.dequantize()
        # A meta shell: the weight is replaced at once, and allocating it here
        # would add a full float copy of the branch at peak memory.
        linear = nn.Linear(
            len(columns), out_features, bias=False, device="meta", dtype=weight.dtype
        )
        ModTensorUpdater(linear).update_by_ref(
            linear.weight, q, enforce_tensor_consistency=False
        )
        return linear


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
