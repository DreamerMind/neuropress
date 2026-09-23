"""Centralized statistics collection with deduplication.

A single global calibration pass feeds all registered probes.
"""

from __future__ import annotations

import typing
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .collectors import (
    COLLECTOR_REGISTRY,
    Collector,
    StatField,
    StatisticRequest,
)


def _cleanup_offloaded(value: Any) -> None:
    """Remove disk payloads reachable from a collected statistic."""
    values = value.values() if isinstance(value, dict) else (value,)
    for item in values:
        offload_path = getattr(item, "offload_path", None)
        if offload_path is not None:
            Path(offload_path).unlink(missing_ok=True)


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
        """Set annotated fields on *method* from collected stats."""
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
                        )
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
    ) -> None:
        """Run a single forward pass over *calibration_data*, populating stats.

        Supported stat types: "min_max", "hessian".
        Hooks are installed, data is collected, then hooks are removed.
        """
        hooks: list[Any] = []
        module_map: dict[str, nn.Module] = {
            name: mod for name, mod in model.named_modules() if name
        }

        # Build per-target collector instances.
        # Pre-registered collectors (e.g. context-aware HessianAccumulators)
        # take precedence over the defaults created from stat_type.
        collectors: dict[str, Collector] = {}

        for dedupe_key, req in self._requests.items():
            if dedupe_key in self._collectors:
                collectors[dedupe_key] = self._collectors[dedupe_key]
            elif req.stat_type in COLLECTOR_REGISTRY:
                collectors[dedupe_key] = COLLECTOR_REGISTRY[req.stat_type]()

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

                def _act_hook(module, inputs, outputs, _col=collector):
                    data = inputs[0] if inputs else outputs
                    if isinstance(data, torch.Tensor):
                        _col.update(data)

                hooks.append(mod.register_forward_hook(_act_hook))

        try:
            with torch.inference_mode():
                for batch in calibration_data:
                    if isinstance(batch, dict):
                        model(**batch)
                    elif isinstance(batch, list | tuple):
                        model(*batch)
                    else:
                        model(batch)
        except Exception:
            for collector in collectors.values():
                _cleanup_offloaded(collector.result())
            raise
        finally:
            for h in hooks:
                h.remove()

        # Store results.
        for dedupe_key, collector in collectors.items():
            self._stats[dedupe_key] = collector.result()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def set_stat(self, target_key: str, stat_type: str, value: Any) -> None:
        """Pre-populate a statistic (e.g. restored from a cache on disk).

        Allows estimators to inject cached results so that subsequent calls to
        :meth:`get` succeed without running a calibration forward pass.
        """
        self._stats[f"{target_key}::{stat_type}"] = value

    def get(self, request: StatisticRequest) -> Any:
        """Return collected statistic for *request*."""
        dedupe_key = f"{request.target_key}::{request.stat_type}"
        return self._stats.get(dedupe_key)

    def has(self, request: StatisticRequest) -> bool:
        """Return True if the statistic for *request* has been collected."""
        dedupe_key = f"{request.target_key}::{request.stat_type}"
        return dedupe_key in self._stats

    def release(self, request: StatisticRequest) -> None:
        """Drop a statistic and remove any disk-backed tensor payloads."""
        dedupe_key = f"{request.target_key}::{request.stat_type}"
        value = self._stats.pop(dedupe_key, None)
        _cleanup_offloaded(value)

    def release_all(self) -> None:
        """Drop all collected statistics and their disk-backed payloads."""
        for value in self._stats.values():
            _cleanup_offloaded(value)
        self._stats.clear()
