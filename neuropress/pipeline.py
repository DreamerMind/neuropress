"""QuantizationPipeline: orchestrates the full quantization flow."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from typing import Any

import torch
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
        self.last_plan = QuantizationPlan()
        self._graph_provider = RuntimeModuleGraphProvider()

    def run(
        self,
        model: nn.Module,
        calibration_data: Iterable,
        example_inputs: Any = None,
        *,
        reference_data: Iterable | None = None,
        reference_model: nn.Module | Callable[[], nn.Module] | None = None,
        reference_device: str | torch.device | None = "cpu",
        estimation_device: str | torch.device | None = None,
        collection_model: nn.Module | None = None,
        target_paths: set[str] | None = None,
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
            reference_data: Aligned original-float batches for paired statistics.
            reference_model: Teacher model; defaults to the unadapted student.
                A zero-argument callable is invoked only when the estimator
                requests paired statistics, so a teacher that is expensive to
                place on the forward device is not placed for nothing.
            reference_device: Where teacher outputs wait for the student pass
                (see :meth:`StatisticsRegistry.collect`); ``None`` keeps them
                on the teacher's device.
            estimation_device: Move the model and statistics here after collection.
                The returned model remains on this device. The estimator is
                prepared again only if this actually moves the model.
            collection_model: Optional inference view used for graph/statistics forwards.
                It must preserve target module paths and their unadapted weights.
            target_paths: Optional estimator selection and assignment boundary.
            example_inputs: Optional single batch for graph building.
                If *None*, the first element of *calibration_data* is used
                and then replayed at the start of the collection pass.
        """
        if collection_model is not None and self.recipe.preprocessing:
            raise ValueError("Apply preprocessing before supplying a collection model")

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

        observed_model = collection_model if collection_model is not None else model
        graph: QGraph = self._graph_provider.build(observed_model, inputs_for_graph)
        LOGGER.info("QGraph: %d nodes, %d values", len(graph.nodes), len(graph.values))

        # -- Step 2 & 3: Declare + aggregate statistics --------------------
        LOGGER.info("Step 2/3: Aggregating statistic requests")
        estimator = self.recipe.param_estimator
        with (
            estimator.select_targets(target_paths)
            if target_paths is not None
            else nullcontext()
        ):
            # Allow estimators to bind themselves to the model before declaring
            # statistics (e.g. GPTQEstimator.prepare discovers target modules).
            if hasattr(estimator, "prepare"):
                estimator.prepare(model)
            requests = estimator.declare_statistics()
            registry = StatisticsRegistry()
            registry.register_requests(requests)
            teacher = reference_model
            if teacher is not None and not isinstance(teacher, nn.Module):
                teacher = teacher() if registry.requires_pairing() else None

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
            registry.collect(
                observed_model,
                calibration_iter,
                reference_data=reference_data,
                reference_model=teacher,
                reference_device=reference_device,
            )
            if estimation_device is not None:
                registry.to(estimation_device)
                # Estimators read the device of their targets when estimation
                # starts, so moving the model needs no second prepare().
                model.to(estimation_device)

            # -- Step 5: Estimate QSchemes -------------------------------------
            LOGGER.info("Step 5: Estimating QSchemes")
            plan: QuantizationPlan = estimator.estimate(graph, registry, self.recipe)
            LOGGER.info(
                "Plan: %d weight assignments, %d activation assignments",
                len(plan.weight_assignments),
                len(plan.activation_assignments),
            )

            if target_paths is not None:
                assigned = set(plan.weight_assignments) | set(plan.kept_float)
                if assigned - target_paths or plan.activation_assignments:
                    raise ValueError(
                        "Estimator assignments escape the selected weight stage"
                    )
                if target_paths - assigned:
                    raise ValueError(
                        f"Stage targets not handled by estimator: {target_paths - assigned}"
                    )

            # -- Step 6: Adapt model (weight assignment + manifest + lowering) -------
            LOGGER.info("Step 6: Adapting model structure")
            self.last_plan = plan
            self.recipe.adapter.apply(model, plan)

            from .t2n.model_io import attach_manifest

            attach_manifest(model)

            if self.recipe.kernel_lowering is not None:
                LOGGER.info("Step 6b: Kernel lowering")
                self.recipe.kernel_lowering.apply(model)
                model.__neuropress_kernel_lowering__ = self.recipe.kernel_lowering

            return model
