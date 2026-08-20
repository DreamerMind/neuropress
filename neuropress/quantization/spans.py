"""Lightweight span timing for quantization stages.

Each estimator wraps key operations with :func:`timed_span`::

    with timed_span(self._spans, "vptq.centroids", group=group_key, n=n_vectors):
        centroids = self._train_centroids(...)

A :class:`LoggingSpanCollector` is the default — every span is logged at INFO
level as it completes, with no aggregation overhead.

Pass an :class:`AccumulatingSpanCollector` when you want to harvest totals for
``QuantizationTiming.stages`` in the eval report.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SpanCollector(Protocol):
    """Protocol for span timing sinks."""

    def record(self, name: str, duration_s: float, **meta: Any) -> None:
        """Receive one completed span."""
        ...


class NullSpanCollector:
    """Drop all spans silently."""

    def record(self, name: str, duration_s: float, **meta: Any) -> None:
        """Discard the span."""


class LoggingSpanCollector:
    """Log each span at INFO level as it completes."""

    def record(self, name: str, duration_s: float, **meta: Any) -> None:
        """Log the span at INFO level, appending any *meta* as ``k=v`` pairs."""
        suffix = ("  " + "  ".join(f"{k}={v}" for k, v in meta.items())) if meta else ""
        LOGGER.info("span  %-42s  %.3fs%s", name, duration_s, suffix)


@dataclass
class AccumulatingSpanCollector:
    """Store span records; convert to per-name totals via :meth:`to_stage_totals`."""

    records: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)

    def record(self, name: str, duration_s: float, **meta: Any) -> None:
        """Append the span to :attr:`records` for later aggregation."""
        self.records.append((name, duration_s, dict(meta)))

    def to_stage_totals(self) -> dict[str, float]:
        """Return {span_name: total_seconds} summed across all records."""
        totals: dict[str, float] = {}
        for name, dur, _ in self.records:
            totals[name] = round(totals.get(name, 0.0) + dur, 3)
        return totals


@contextlib.contextmanager
def timed_span(
    collector: SpanCollector | None,
    name: str,
    **meta: Any,
) -> Generator[None]:
    """Context manager: record a named span to *collector* on exit.

    If *collector* is ``None`` the timing call is still made (cheap) but
    the result is discarded.  This avoids a conditional at every call site.
    """
    t = time.perf_counter()
    try:
        yield
    finally:
        if collector is not None:
            collector.record(name, time.perf_counter() - t, **meta)
