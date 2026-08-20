"""ModuleQuantizationAdapter: ABC and concrete adapters for applying a plan."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from torch import nn
from torch_to_nnef.tensor.updater import ModTensorUpdater

from .plan import QuantizationPlan

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
    """Replaces module weights with plan assignments via :class:`ModTensorUpdater`.

    For each entry in ``plan.weight_assignments``, the corresponding module's
    ``weight`` parameter is swapped for the plan value using
    ``ModTensorUpdater.update_by_ref``.  This preserves all internal PyTorch
    bookkeeping (parameter registration, grad tracking, etc.).

    Returns the model unchanged when the plan is empty.
    """

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
                    f"{name}.weight",
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
