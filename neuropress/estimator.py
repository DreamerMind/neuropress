"""QuantizationParamEstimator: ABC for estimating QSchemes from statistics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

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


WHOLE_MODEL_ATTR = "_neuropress_whole_model"


def whole_model(model: nn.Module) -> nn.Module:
    """The whole model *model* stands for: itself, unless it is a block.

    A block held at its model paths (:class:`.block_pipeline.BlockAtPaths`)
    keeps a weak reference to the model it was taken from. Estimators read
    the whole model for what depends on more than the targets at hand: the
    position of a layer among all of the model's layers, or whether a module
    a recipe names exists anywhere in the model.
    """
    ref = getattr(model, WHOLE_MODEL_ATTR, None)
    whole = ref() if ref is not None else None
    return whole if whole is not None else model


class UnpreparedTargets(dict):
    """Default target dict of an estimator whose ``prepare`` has not run.

    Empty, so code reading targets early sees none, but recognisable by
    :meth:`QuantizationParamEstimator.declare_targets`, which refuses it.
    """


class QuantizationParamEstimator(ABC):
    """Estimates concrete QSchemes given statistics and backend constraints.

    Responsibilities:

    1. Declare the targets to quantize (:meth:`declare_targets`).
    2. Declare required statistics.
    3. Select QScheme(s) from allowed descriptors.
    4. Return a QuantizationPlan.

    Subclasses that need to inspect the model before statistics collection
    (e.g. to discover target layers or record the compute device) should
    override :meth:`prepare`.  The pipeline calls ``prepare(model)`` once,
    before :meth:`declare_statistics`. An explicit estimation device moves
    the model after collection, so :meth:`estimate` reads device placement
    from the target modules instead of from state recorded in ``prepare``.

    The pipeline wires up :attr:`_cache` and :attr:`_spans` via
    :meth:`set_cache` / :meth:`set_span_collector` before calling
    :meth:`prepare`.  Both are usable immediately (no-ops by default) so
    subclasses never need defensive ``getattr`` fallbacks.
    """

    # Pipeline injects these before prepare(); no-op defaults ensure subclasses
    # can always call self._cache / self._spans unconditionally.
    _cache: CacheManager = _NULL_CACHE
    _spans: SpanCollector = _DEFAULT_SPANS

    #: Whether :meth:`estimate` reads its ``graph`` argument. Building the
    #: graph costs one extra forward pass, so the pipeline passes an empty
    #: :class:`QGraph` to estimators that set this to ``False``. A block
    #: pipeline traces one block at a time, so the graph never spans more.
    requires_graph: ClassVar[bool] = True

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
        discover target layers. It runs once, before statistics collection, and
        must not modify model weights. An explicit estimation device moves the
        model after it runs, so :meth:`estimate` must read device placement
        from the target modules rather than from state recorded here.

        Args:
            model: The model that will be passed through the pipeline.
        """
        ...

    @contextmanager
    def select_targets(self, paths: set[str]) -> Iterator[None]:
        """Temporarily restrict estimation to a stage, preserving target filters.

        Estimators with module filters inherit this implementation. Other
        estimators must implement this contract explicitly to support staging.
        Enter before prepare and leave after adaptation. The original filter
        is restored even when collection or estimation fails. Custom estimators
        must provide the same context-manager contract.
        """
        if not hasattr(self, "filter_fn"):
            raise NotImplementedError(
                f"{type(self).__name__} does not support target selection"
            )
        paths = frozenset(paths)
        previous = self.filter_fn
        self.filter_fn = lambda name, module: (
            name in paths and (previous is None or previous(name, module))
        )
        try:
            yield
        finally:
            self.filter_fn = previous

    def declare_targets(self) -> list[str]:
        """Return the module paths this estimator quantizes, after :meth:`prepare`.

        Not derivable from :meth:`declare_statistics`: estimators that work
        from the weights alone (min-max, HQQ) request no statistics. Tied
        aliases of a target are not listed; the adapter keeps them tied.

        The default reads ``_target_modules``, where estimators that find
        their targets with :func:`.collect_target_modules` keep them (with
        :class:`UnpreparedTargets` as the field default). Others must
        override it.

        Raises:
            NotImplementedError: If the estimator does not keep its targets
                there, or :meth:`prepare` has not run.
        """
        targets = getattr(self, "_target_modules", None)
        if targets is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare its targets "
                "(override declare_targets)"
            )
        if isinstance(targets, UnpreparedTargets):
            raise NotImplementedError(
                f"{type(self).__name__}: call prepare before declare_targets"
            )
        return list(targets)

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
