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


_SCALARS = (bool, int, float, str, type(None))


def _structure(batch) -> tuple | None:
    """Everything except tensor values that can steer a traced forward.

    Value-dependent Python control flow is rejected during tracing, so two
    batches with equal structure execute the same operations and yield the
    same dependencies. ``None`` marks a batch whose leaves cannot be compared
    (arbitrary objects); such batches are always traced.
    """
    leaves, spec = tree_flatten(batch)
    key: list = [str(spec)]
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            key.append(
                (leaf.shape, leaf.stride(), leaf.dtype, leaf.device, leaf.layout)
            )
        elif isinstance(leaf, _SCALARS):
            key.append((type(leaf), leaf))
        else:
            return None
    return tuple(key)


def _value_dependent_shape(func, args) -> bool:
    """True for ops whose output shape depends on tensor values (e.g. nonzero).

    Downstream Python can branch on such shapes, so equal batch structure no
    longer implies an equal execution path. Integer-tensor indexing (rotary
    caches indexed by position ids) has a shape fixed by its index tensors;
    only boolean masks make ``index.Tensor`` value dependent.
    """
    if torch.Tag.dynamic_output_shape not in func.tags:
        return False
    if func is torch.ops.aten.index.Tensor:
        return any(
            index is not None and index.dtype in (torch.bool, torch.uint8)
            for index in args[1]
        )
    return True


class _Dependencies(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.values = {}
        self.value_dependent_shapes = False

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
        if not self.value_dependent_shapes and _value_dependent_shape(func, args):
            self.value_dependent_shapes = True
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
        Batches whose structure (tree, shapes, strides, dtypes, devices and
        scalar leaves) matches an already traced batch are skipped, since they
        replay the same operations; this keeps a fixed-length calibration set
        at one traced forward per block. Batches are always traced when a
        prior trace of the same structure produced value-dependent shapes
        (nonzero, boolean masks), as in token routing.
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
        traced = set()
        for batch in batches:
            count += 1
            structure = _structure(batch)
            if structure is not None and structure in traced:
                continue
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
            if structure is not None and not tracker.value_dependent_shapes:
                traced.add(structure)
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
