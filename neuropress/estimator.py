"""QuantizationParamEstimator: ABC for estimating QSchemes from statistics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch import nn

from .calibration.probe import StatisticRequest, StatisticsRegistry
from .graph.graph import QGraph
from .plan import QuantizationPlan
from .quantization.cache import CacheManager, CacheStageDescriptor, NullCacheManager
from .quantization.spans import LoggingSpanCollector, SpanCollector

if TYPE_CHECKING:
    from .recipe import QuantizationRecipe

# Shared no-op singleton so estimators that never call set_cache() still have
# a valid _cache attribute without per-instance allocation.
_NULL_CACHE: CacheManager = NullCacheManager()
_DEFAULT_SPANS: SpanCollector = LoggingSpanCollector()


class QuantizationParamEstimator(ABC):
    """Estimates concrete QSchemes given statistics and backend constraints.

    Responsibilities:

    1. Declare subgraphs / targets to quantize.
    2. Declare required statistics.
    3. Select QScheme(s) from allowed descriptors.
    4. Return a QuantizationPlan.

    Subclasses that need to inspect the model before statistics collection
    (e.g. to discover target layers or record the compute device) should
    override :meth:`prepare`.  The pipeline calls ``prepare(model)``
    automatically before :meth:`declare_statistics` when the method is present,
    so overriding it here makes the contract explicit for all implementations.

    The pipeline wires up :attr:`_cache` and :attr:`_spans` via
    :meth:`set_cache` / :meth:`set_span_collector` before calling
    :meth:`prepare`.  Both are usable immediately (no-ops by default) so
    subclasses never need defensive ``getattr`` fallbacks.
    """

    # Pipeline injects these before prepare(); no-op defaults ensure subclasses
    # can always call self._cache / self._spans unconditionally.
    _cache: CacheManager = _NULL_CACHE
    _spans: SpanCollector = _DEFAULT_SPANS

    # ------------------------------------------------------------------
    # Cache stages
    # ------------------------------------------------------------------

    def cache_stages(self) -> list[CacheStageDescriptor]:
        """Declare which intermediate results this estimator can cache.

        The pipeline (or user) calls :meth:`set_cache` with a
        :class:`~.quantization.cache.CacheManager` before running.  Returned
        descriptors are informational only; the actual caching logic lives in
        the estimator's own methods.

        Override to declare stages.  Default: no stages (caching disabled).
        """
        return []

    def set_cache(self, manager: CacheManager) -> None:
        """Attach a cache manager; called by the pipeline before :meth:`prepare`."""
        self._cache = manager

    def set_span_collector(self, collector: SpanCollector) -> None:
        """Attach a span collector; called by the pipeline before :meth:`prepare`."""
        self._spans = collector

    # ------------------------------------------------------------------
    # Model binding (optional)
    # ------------------------------------------------------------------

    def prepare(self, model: nn.Module) -> None:  # noqa: B027
        """Bind this estimator to *model* before statistics collection.

        The default implementation is a no-op.  Subclasses override this to
        discover target layers, record the compute device, or perform any other
        one-time setup that requires access to the full model.

        Args:
            model: The model that will be passed through the pipeline.
        """
        ...

    @abstractmethod
    def declare_statistics(self) -> list[StatisticRequest]:
        """Return the list of statistics needed for estimation."""

    @abstractmethod
    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Produce a QuantizationPlan from the collected statistics."""
