"""QuantizationPipeline: orchestrates the full quantization flow."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Iterable, Iterator
from typing import Any

from torch import nn

try:
    from rich.progress import track as _rich_track

    _RICH = True
except ImportError:
    _RICH = False

from .calibration.probe import StatisticsRegistry
from .graph.builder import RuntimeModuleGraphProvider
from .graph.graph import QGraph
from .plan import QuantizationPlan
from .recipe import QuantizationRecipe

LOGGER = logging.getLogger(__name__)


class QuantizationPipeline:
    """Orchestrates the full quantization flow.

    Steps (from PLAN.md §12):
    1. Build QGraph via RuntimeModuleGraphProvider.
    2. Define subgraphs (estimator declares what it needs).
    3. Aggregate statistic requests.
    4. Run global calibration (single forward pass).
    5. Estimate QSchemes (torch_to_nnef-based).
    6. Adapt model structure.
    """

    def __init__(self, recipe: QuantizationRecipe) -> None:
        """Initialize the pipeline with the given *recipe*."""
        self.recipe = recipe
        self._graph_provider = RuntimeModuleGraphProvider()

    def run(
        self,
        model: nn.Module,
        calibration_data: Iterable,
        example_inputs: Any = None,
    ) -> nn.Module:
        """Run the full pipeline and return the quantized model.

        *calibration_data* is consumed lazily: only one batch is held in memory
        at a time during the collection pass.  If preprocessing steps are
        configured, the data is materialised into a list first (preprocessing
        may need to iterate it multiple times).

        Args:
            model: Float model to quantize.
            calibration_data: Iterable of batches for calibration.  May be a
                generator — it is iterated exactly once (unless preprocessing
                is active, in which case it is materialised upfront).
            example_inputs: Optional single batch for graph building.
                If *None*, the first element of *calibration_data* is used
                and then replayed at the start of the collection pass.
        """
        # -- Step 0: Preprocessing (materialise if needed) ------------------
        if self.recipe.preprocessing:
            LOGGER.info(
                "Step 0: Running %d preprocessing step(s)",
                len(self.recipe.preprocessing),
            )
            # Materialise so preprocessing steps can each iterate the data.
            calibration_list = list(calibration_data)
            for step in self.recipe.preprocessing:
                step.apply(model, calibration_list)
            calibration_iter: Iterator = iter(calibration_list)
        else:
            calibration_iter = iter(calibration_data)

        # -- Step 1: Build QGraph ------------------------------------------
        LOGGER.info("Step 1: Building QGraph")
        if example_inputs is not None:
            inputs_for_graph = example_inputs
        else:
            # Peek the first batch so it is not lost from the collection pass.
            inputs_for_graph = next(calibration_iter)
            calibration_iter = itertools.chain([inputs_for_graph], calibration_iter)

        graph: QGraph = self._graph_provider.build(model, inputs_for_graph)
        LOGGER.info("QGraph: %d nodes, %d values", len(graph.nodes), len(graph.values))

        # -- Step 2 & 3: Declare + aggregate statistics --------------------
        LOGGER.info("Step 2/3: Aggregating statistic requests")
        estimator = self.recipe.param_estimator
        # Allow estimators to bind themselves to the model before declaring
        # statistics (e.g. GPTQEstimator.prepare discovers target modules).
        if hasattr(estimator, "prepare"):
            estimator.prepare(model)
        requests = estimator.declare_statistics()
        registry = StatisticsRegistry()
        registry.register_requests(requests)
        estimator.configure_statistics_registry(registry)

        # -- Step 4: Global calibration (single lazy pass) -----------------
        LOGGER.info("Step 4: Running global calibration")
        cal = self.recipe.calibration
        total_batches = math.ceil(cal.n_samples / cal.batch_size)
        if _RICH:
            calibration_iter = _rich_track(
                calibration_iter,
                description=(
                    f"[cyan]Hessians[/cyan]  "
                    f"{cal.n_samples} samples / batch {cal.batch_size}"
                ),
                total=total_batches,
            )
        if requests:
            registry.collect(model, calibration_iter)
        else:
            LOGGER.info("Step 4: No statistics requested; skipping calibration pass")

        # -- Step 5: Estimate QSchemes -------------------------------------
        LOGGER.info("Step 5: Estimating QSchemes")
        try:
            plan: QuantizationPlan = estimator.estimate(graph, registry, self.recipe)
        finally:
            registry.release_all()
        LOGGER.info(
            "Plan: %d weight assignments, %d activation assignments",
            len(plan.weight_assignments),
            len(plan.activation_assignments),
        )

        # -- Step 6: Adapt model (weight assignment + manifest + lowering) -------
        LOGGER.info("Step 6: Adapting model structure")
        self.recipe.adapter.apply(model, plan)

        from .t2n.model_io import attach_manifest

        attach_manifest(model)

        if self.recipe.kernel_lowering is not None:
            LOGGER.info("Step 6b: Kernel lowering")
            self.recipe.kernel_lowering.apply(model)
            model.__neuropress_kernel_lowering__ = self.recipe.kernel_lowering

        return model
