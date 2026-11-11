"""Centralized statistics collection with deduplication.

A single global calibration pass feeds all registered probes.
"""

from __future__ import annotations

import inspect
import typing
from collections import deque
from collections.abc import Iterable
from itertools import zip_longest
from typing import Any

import torch
from torch import nn

from ..batches import call_batch, move_batch
from .collectors import (
    COLLECTOR_REGISTRY,
    Collector,
    PairedCollector,
    StatField,
    StatisticRequest,
)


def _forward_signature(module: nn.Module) -> inspect.Signature | None:
    try:
        return inspect.signature(module.forward)
    except (TypeError, ValueError):  # builtins / C extensions
        return None


def _positional_inputs(
    signature: inspect.Signature | None, args: tuple, kwargs: dict
) -> tuple:
    """Flatten a call into its arguments in declaration order.

    Keyword arguments land where the parameter is declared, so a module
    called as ``m(hidden_states=x)`` yields ``x`` first just like ``m(x)``.
    Variadic ``*args`` and ``**kwargs`` are expanded in place rather than
    surfacing as one tuple or dict that no collector can consume.
    """
    if signature is not None:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            pass
        else:
            flat: list[Any] = []
            for name, value in bound.arguments.items():
                kind = signature.parameters[name].kind
                if kind is inspect.Parameter.VAR_POSITIONAL:
                    flat.extend(value)
                elif kind is inspect.Parameter.VAR_KEYWORD:
                    flat.extend(value.values())
                else:
                    flat.append(value)
            return tuple(flat)
    return (*args, *kwargs.values())


class StatisticsRegistry:
    """Collect and store calibration statistics.

    Usage::

        registry = StatisticsRegistry()
        registry.register_requests(my_probe.required_statistics())
        registry.collect(model, calibration_data)
        stats = my_probe.extract_statistics(registry)

    Custom collectors (e.g. a context-aware ``HessianAccumulator``) can be
    pre-registered via :meth:`register_collector`; they take precedence over
    the default instances created from ``stat_type``.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._requests: dict[str, StatisticRequest] = {}  # key → request
        self._stats: dict[str, Any] = {}  # key → statistic value
        self._collectors: dict[str, Collector] = {}  # pre-registered collectors
        self._device: torch.device | None = None  # see to()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_requests(self, requests: list[StatisticRequest]) -> None:
        """Register statistic requests; duplicates are deduplicated by key+type."""
        for req in requests:
            dedupe_key = f"{req.target_key}::{req.stat_type}"
            self._requests[dedupe_key] = req

    def register_collector(
        self, request: StatisticRequest, collector: Collector
    ) -> None:
        """Register a pre-built *collector* for *request*, overriding the default.

        Use this when the default collector created from ``stat_type`` needs
        additional context — for example, a ``HessianAccumulator`` with
        ``module_class`` and ``device`` already set.

        Args:
            request: The statistic to collect.
            collector: A fully-configured ``Collector`` instance to use instead
                of the default one created by :meth:`collect`.
        """
        dedupe_key = f"{request.target_key}::{request.stat_type}"
        self._requests[dedupe_key] = request
        self._collectors[dedupe_key] = collector

    def requires_pairing(self) -> bool:
        """Return True if any registered request is served by a paired collector.

        Callers use this to avoid preparing a teacher model and reference
        stream that :meth:`collect` would never run.
        """
        for dedupe_key, req in self._requests.items():
            collector = self._collectors.get(dedupe_key)
            if collector is not None:
                if isinstance(collector, PairedCollector):
                    return True
            elif issubclass(
                COLLECTOR_REGISTRY.get(req.stat_type, Collector), PairedCollector
            ):
                return True
        return False

    def register_method(self, method: Any, target_key: str) -> None:
        """Auto-register stat requests from ``Annotated[..., StatField(...)]`` hints."""
        hints = typing.get_type_hints(type(method), include_extras=True)
        requests = []
        for hint in hints.values():
            if typing.get_origin(hint) is typing.Annotated:
                for meta in typing.get_args(hint)[1:]:
                    if isinstance(meta, StatField):
                        requests.append(
                            StatisticRequest(
                                target_key=target_key,
                                stat_type=meta.stat_type,
                            )
                        )
        self.register_requests(requests)

    def inject(self, method: Any, target_key: str) -> None:
        """Set annotated fields on *method* from collected stats.

        Statistics are injected as collected, ignoring :meth:`to`: the
        calibration methods consuming them (GPTQ) place each Hessian on their
        weights' device themselves, copying only when they must, so relocating
        here would allocate a second K x K copy per layer.
        """
        hints = typing.get_type_hints(type(method), include_extras=True)
        cache: dict[str, Any] = {}
        for attr, hint in hints.items():
            if typing.get_origin(hint) is not typing.Annotated:
                continue
            for meta in typing.get_args(hint)[1:]:
                if not isinstance(meta, StatField):
                    continue
                if meta.stat_type not in cache:
                    cache[meta.stat_type] = self.get(
                        StatisticRequest(
                            target_key=target_key, stat_type=meta.stat_type
                        ),
                        relocate=False,
                    )
                result = cache[meta.stat_type]
                if result is None:
                    continue
                key = meta.field_key if meta.field_key is not None else attr
                if isinstance(result, dict) and key in result:
                    setattr(method, attr, result[key])

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

    def collect(
        self,
        model: nn.Module,
        calibration_data: Iterable,
        *,
        reference_data: Iterable | None = None,
        reference_model: nn.Module | None = None,
        reference_device: torch.device | str | None = "cpu",
    ) -> None:
        """Run a single forward pass over *calibration_data*, populating stats.

        Collectors are selected through COLLECTOR_REGISTRY. Paired collectors
        additionally require reference_data aligned batch-for-batch. Module paths
        and invocation order must match the reference_model (default: model).
        Teacher outputs of one batch are snapshotted on *reference_device* until
        the student pass consumes them. The CPU default bounds device memory
        when every linear of a whole model pairs; ``None`` keeps them where
        the teacher produced them, which saves two transfers per paired output
        when one block's worth fits on the device. Hooks are removed even when
        pairing or a forward pass fails.
        """
        hooks: list[Any] = []
        module_map = dict(model.named_modules())

        # Build per-target collector instances.
        # Pre-registered collectors (e.g. context-aware HessianAccumulators)
        # take precedence over the defaults created from stat_type.
        collectors: dict[str, Collector] = {}

        for dedupe_key, req in self._requests.items():
            if dedupe_key in self._collectors:
                collectors[dedupe_key] = self._collectors[dedupe_key]
            elif req.stat_type in COLLECTOR_REGISTRY:
                collectors[dedupe_key] = COLLECTOR_REGISTRY[req.stat_type]()

        paired = {
            key: col
            for key, col in collectors.items()
            if isinstance(col, PairedCollector)
        }
        if paired and reference_data is None:
            raise ValueError("Paired statistics require reference_data")
        teacher = reference_model if reference_model is not None else model
        teacher_modules = dict(teacher.named_modules())
        for key in paired:
            target = self._requests[key].target_key
            if target not in module_map or target not in teacher_modules:
                raise ValueError(
                    f"Paired target {target!r} missing from student or teacher"
                )
        observations = {key: deque() for key in paired}
        collecting_student = True
        for key in paired:

            def reference_hook(module, inputs, output, _key=key):
                if not collecting_student:
                    observations[_key].append(
                        move_batch(output, reference_device, copy=True)
                    )

            hooks.append(
                teacher_modules[self._requests[key].target_key].register_forward_hook(
                    reference_hook
                )
            )

        # Install hooks.  target_key format: "{module_path}/{param_name}"
        # or just a value_id (activation).
        for dedupe_key, req in self._requests.items():
            collector = collectors.get(dedupe_key)
            if collector is None:
                continue

            if "/" in req.target_key:
                # Weight target: module_path/param_name
                module_path, param_name = req.target_key.rsplit("/", 1)
                mod = module_map.get(module_path)
                if mod is None:
                    continue

                def _weight_hook(
                    module, inputs, outputs, _col=collector, _pn=param_name
                ):
                    if not collecting_student:
                        return
                    w = getattr(module, _pn, None)
                    if w is not None:
                        _col.update(w.data)

                hooks.append(mod.register_forward_hook(_weight_hook))
            else:
                # Activation target: value_id maps to a module path via naming
                # convention. We hook the module whose output matches the value.
                # The value_id is expected to be the module path.
                mod = module_map.get(req.target_key)
                if mod is None:
                    continue

                # Propagate module context to collectors that need it
                # (e.g. HessianAccumulator uses module_class for Conv2d unfold).
                if (
                    hasattr(collector, "module_class")
                    and collector.module_class is None
                ):
                    collector.module_class = type(mod)
                if hasattr(collector, "module") and collector.module is None:
                    collector.module = mod

                signature_memo: list[inspect.Signature | None] = []

                def _act_hook(
                    module,
                    inputs,
                    kwargs,
                    outputs,
                    _col=collector,
                    _key=dedupe_key,
                    _signature=signature_memo,
                ):
                    if not collecting_student:
                        return
                    if kwargs:
                        # Resolved once per hook: introspection is too slow
                        # for every call in the calibration loop.
                        if not _signature:
                            _signature.append(_forward_signature(module))
                        inputs = _positional_inputs(_signature[0], inputs, kwargs)
                    if isinstance(_col, PairedCollector):
                        if not observations[_key]:
                            raise ValueError(
                                f"No matching teacher invocation for {_key}"
                            )
                        _col.update_pair(inputs, outputs, observations[_key].popleft())
                        return
                    data = inputs[0] if inputs else outputs
                    if isinstance(data, torch.Tensor):
                        _col.update(data)

                hooks.append(mod.register_forward_hook(_act_hook, with_kwargs=True))

        # Paired samples may feed gradient-based estimators, which cannot use
        # inference tensors; everything else keeps inference mode's savings.
        grad_mode = torch.no_grad() if paired else torch.inference_mode()
        try:
            with grad_mode:
                if paired:
                    missing = object()
                    for batch, reference in zip_longest(
                        calibration_data, reference_data, fillvalue=missing
                    ):
                        if batch is missing or reference is missing:
                            raise ValueError("Student and teacher batch counts differ")
                        collecting_student = False
                        call_batch(teacher, reference)
                        collecting_student = True
                        call_batch(model, batch)
                        if any(observations.values()):
                            raise ValueError(
                                "Student and teacher module invocation counts differ"
                            )
                else:
                    for batch in calibration_data:
                        call_batch(model, batch)
        finally:
            for h in hooks:
                h.remove()

        # Store results.
        for dedupe_key, collector in collectors.items():
            self._stats[dedupe_key] = collector.result()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> StatisticsRegistry:
        """Deliver statistics on *device* from now on, one statistic per access.

        Nothing is moved here: :meth:`get` returns a copy on *device* and the
        registry keeps the collected original where its collector left it. A
        Hessian that ``HessianAccumulator`` kept on the CPU (or offloaded to
        disk) therefore reaches the estimation device only while its own layer
        is estimated, instead of every Hessian of the model at once.
        """
        self._device = torch.device(device)
        return self

    def set_stat(self, target_key: str, stat_type: str, value: Any) -> None:
        """Pre-populate a statistic (e.g. restored from a cache on disk).

        Allows estimators to inject cached results so that subsequent calls to
        :meth:`get` succeed without running a calibration forward pass.
        """
        self._stats[f"{target_key}::{stat_type}"] = value

    def get(self, request: StatisticRequest, *, relocate: bool = True) -> Any:
        """Return collected statistic for *request*.

        After :meth:`to`, plain tensors come back on that device unless
        *relocate* is false (e.g. to persist the statistic as collected).
        Tensor subclasses such as disk-offloaded Hessians are returned as-is;
        their consumers materialize them on demand.

        Treat the result as read-only: it is the stored statistic itself when
        it already lives on the target device and a copy otherwise, so a write
        would change the registry in one setup and not the other. Copy before
        writing. Results are not memoized, so each relocating call allocates;
        consumers fetch a statistic once per layer.
        """
        dedupe_key = f"{request.target_key}::{request.stat_type}"
        value = self._stats.get(dedupe_key)
        if relocate and self._device is not None:
            value = move_batch(value, self._device, tensor_subclasses=False)
        return value

    def has(self, request: StatisticRequest) -> bool:
        """Return True if the statistic for *request* has been collected."""
        dedupe_key = f"{request.target_key}::{request.stat_type}"
        return dedupe_key in self._stats
