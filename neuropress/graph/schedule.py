"""Dependency stages for independently calibrated module targets."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from ..calibration.batches import call_batch


def _tensors(tree):
    return [value for value in tree_flatten(tree)[0] if isinstance(value, torch.Tensor)]


class _Dependencies(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.values = {}

    def sources(self, tree):
        result = set()
        for tensor in _tensors(tree):
            entry = self.values.get(id(tensor))
            if entry is not None and entry[0]() is tensor:
                result.update(entry[1])
        return result

    def tag(self, tree, sources):
        for tensor in _tensors(tree):
            self.values[id(tensor)] = (weakref.ref(tensor), frozenset(sources))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func._schema.is_mutable:
            raise ValueError(
                "Dependency tracing requires a forward without in-place operations"
            )
        if func is torch.ops.aten._local_scalar_dense.default:
            raise ValueError(
                "Dependency tracing does not support tensor-dependent Python control flow"
            )
        sources = self.sources((args, kwargs))
        result = func(*args, **(kwargs or {}))
        self.tag(result, sources)
        return result


@dataclass(frozen=True)
class QuantizationSchedule:
    """Antichains of target modules in observed tensor dataflow order.

    Members of a stage have no observed dependency on one another. Stages are
    calibration barriers, not a promise of hardware concurrency. Trace all
    representative batches on an eval model with a functional forward. Repeated
    targets, nested targets, shared target parameters and unseen targets are
    rejected. Python side effects and unobserved execution paths are not modeled.
    """

    stages: tuple[tuple[str, ...], ...]

    def __str__(self):
        return " → ".join("[" + " | ".join(stage) + "]" for stage in self.stages)

    @classmethod
    def trace(cls, model: nn.Module, batches, targets) -> QuantizationSchedule:
        """Rank explicit module paths using dependencies through tensor operations.

        Functional attention, activations, reshapes and residual additions retain
        provenance. Dependencies are unioned across batches before ranking.
        """
        targets = tuple(targets)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("Supply nonempty, unique target module paths")
        if any(module.training for module in model.modules()):
            raise ValueError("Dependency tracing requires an eval model")
        modules = {name: model.get_submodule(name) for name in targets}
        for name in targets:
            if any(
                other != name and (not other or name.startswith(other + "."))
                for other in targets
            ):
                raise ValueError("Nested targets cannot be scheduled independently")
        storages = {}
        for name, module in modules.items():
            for parameter in module.parameters():
                if not parameter.numel():
                    continue
                key = (str(parameter.device), parameter.untyped_storage().data_ptr())
                if key in storages and storages[key] != name:
                    raise ValueError(
                        "Scheduled targets must not share parameter storage"
                    )
                storages[key] = name
        dependencies = {name: set() for name in targets}
        count = 0
        for batch in batches:
            count += 1
            tracker = _Dependencies()
            for name, module in modules.items():
                tracker.tag(tuple(module.parameters()), {name})
            seen = set()
            active = {}
            hooks = []
            for name, module in modules.items():

                def before(
                    mod,
                    args,
                    kwargs,
                    name=name,
                    seen=seen,
                    active=active,
                    tracker=tracker,
                ):
                    if name in seen:
                        raise ValueError(f"Repeated target invocation: {name}")
                    seen.add(name)
                    active[name] = tracker.sources((args, kwargs))
                    if name in active[name]:
                        raise ValueError(
                            f"Target input depends on its own parameters: {name}"
                        )

                def after(
                    mod, args, kwargs, output, name=name, active=active, tracker=tracker
                ):
                    sources = active[name] | tracker.sources(output)
                    dependencies[name].update(sources - {name})
                    tracker.tag(output, sources | {name})

                hooks.append(module.register_forward_pre_hook(before, with_kwargs=True))
                hooks.append(module.register_forward_hook(after, with_kwargs=True))
            try:
                with torch.no_grad(), tracker:
                    call_batch(model, batch)
            finally:
                for hook in hooks:
                    hook.remove()
            if seen != set(targets):
                raise ValueError(
                    f"Targets not executed in a calibration batch: {set(targets) - seen}"
                )
        if not count:
            raise ValueError("Dependency tracing requires calibration batches")
        stages = []
        remaining = set(targets)
        while remaining:
            ready = tuple(
                name
                for name in targets
                if name in remaining and not dependencies[name] & remaining
            )
            if not ready:
                raise ValueError(
                    "Target dependencies contain a cycle across observed forwards"
                )
            stages.append(ready)
            remaining.difference_update(ready)
        return cls(tuple(stages))
