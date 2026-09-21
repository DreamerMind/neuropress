"""Concrete QuantizationParamEstimator implementations."""

from __future__ import annotations

import contextlib
import logging
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import reduce
from typing import Any

try:
    from rich.markup import escape as _re
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    def _layer_progress(total: int) -> Progress:
        """Return a configured :class:`~rich.progress.Progress` for per-layer loops."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )

    _RICH = True
except ImportError:
    _RICH = False
    _re = str  # type: ignore[assignment]

import torch
from torch import nn

from .calibration.probe import HessianAccumulator, StatisticRequest, StatisticsRegistry
from .estimator import QuantizationParamEstimator
from .graph.graph import QGraph
from .methods.correction_mode import CorrectionMode
from .methods.gptq import GPTQCalibration
from .methods.gptq_panel import GPTQMixedPrecisionCalibration
from .methods.hqq import HQQCalibration
from .methods.ternary import TernaryGPTQCalibration, TernaryScaleCalibration
from .methods.zp_scale_calib import NormZpScaleCalibration
from .plan import QuantizationPlan
from .qscheme.descriptor import Granularity
from .quantization.cache import CacheStageDescriptor
from .quantization.spans import timed_span
from .recipe import QuantizationRecipe

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared target-collection utility
# ---------------------------------------------------------------------------


def collect_target_modules(
    model: nn.Module,
    module_classes: tuple[type, ...],
    filter_fn: Callable[[str, nn.Module], bool] | None = None,
) -> dict[str, nn.Module]:
    """Return an ordered dict of target modules, deduplicating tied weights.

    Iterates ``model.named_modules()`` in traversal order and keeps the
    *first* name encountered for each unique weight storage pointer.
    Subsequent modules whose ``weight.data_ptr()`` was already seen are
    silently dropped - this handles tied-weight pairs such as
    ``embed_tokens`` / ``lm_head`` automatically without model-specific
    ``exclude_lm_head`` flags.

    Args:
        model: Model to inspect.
        module_classes: Tuple of ``nn.Module`` subclasses to include.
        filter_fn: Optional extra predicate ``(name, mod) -> bool``.

    Returns:
        ``{name: module}`` dict in model traversal order, deduplicated.
    """
    seen_ptrs: set[int] = set()
    result: dict[str, nn.Module] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, module_classes):
            continue
        if filter_fn is not None and not filter_fn(name, mod):
            continue
        w = getattr(mod, "weight", None)
        if w is not None:
            ptr = w.data_ptr()
            if ptr in seen_ptrs:
                LOGGER.info(
                    "collect_target_modules: skipping '%s' — tied weight already collected",
                    name,
                )
                continue
            seen_ptrs.add(ptr)
        result[name] = mod
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gptq_layer(
    name: str,
    mod: nn.Module,
    registry: StatisticsRegistry,
    device: torch.device,
    group_size: int,
    nbits: int,
    symmetric: bool,
    damp_percent: float,
    damp_auto_increment: float,
    ternary_threshold_ratio: float | None = None,
    statistic_target: str | None = None,
) -> None:
    """Apply GPTQ to *mod*, updating its weight in-place to dequantized floats.

    Granularity is chosen per layer: per-group when ``in_features`` is
    divisible by *group_size*, otherwise per-channel.

    Args:
        name: Human-readable layer name for logging.
        mod: Module whose weight is quantized in-place.
        registry: Populated registry providing Hessian statistics.
        device: Computation device.
        group_size: Preferred group size for per-group granularity.
        nbits: Quantization bit-width.
        symmetric: Whether to use symmetric quantization.
        damp_percent: Initial Hessian dampening fraction.
        damp_auto_increment: Dampening increment on Cholesky failure.
        ternary_threshold_ratio: Enable strict ternary quantization with this
            zero/non-zero threshold, or use affine quantization when ``None``.
        statistic_target: Registry key to inject when a sibling layer owns the
            shared Hessian; defaults to *name*.
    """
    in_features = mod.weight.shape[1]
    if in_features >= group_size and in_features % group_size == 0:
        gran = Granularity.per_group([group_size])
    else:
        gran = Granularity.per_channel()

    if ternary_threshold_ratio is None:
        calib = GPTQCalibration(
            granularity=gran,
            possible_quantizers=[
                NormZpScaleCalibration(
                    nbits=nbits, symmetric=symmetric, granularity=gran
                )
            ],
            damp_percent=damp_percent,
            damp_auto_increment=damp_auto_increment,
        )
    else:
        calib = TernaryGPTQCalibration(
            granularity=gran,
            possible_quantizers=[
                TernaryScaleCalibration(
                    granularity=gran,
                    threshold_ratio=ternary_threshold_ratio,
                )
            ],
            damp_percent=damp_percent,
            damp_auto_increment=damp_auto_increment,
        )
    _w = mod.weight
    assert isinstance(_w, torch.Tensor)
    calib.weight_tensor = _w
    calib.module_class = type(mod)
    calib.module = mod
    calib.name = name
    calib.device = device
    calib.init()
    registry.inject(calib, statistic_target or name)
    calib.compute_qscheme()


# ---------------------------------------------------------------------------
# GPTQEstimator
# ---------------------------------------------------------------------------


@dataclass
class GPTQEstimator(QuantizationParamEstimator):
    """GPTQ estimator: collects Hessians then runs GPTQ per target layer.

    Call :meth:`prepare` before ``declare_statistics`` is invoked to bind the
    estimator to a specific model.  :class:`.QuantizationPipeline` does this
    automatically when the estimator exposes a ``prepare`` method.

    Attributes:
        module_classes: Only modules that are instances of these classes are
            targeted.  Defaults to ``(nn.Linear,)``.
        filter_fn: Optional extra filter ``(name, module) -> bool``.  If
            *None*, all instances of *module_classes* are targeted.
        weight_transform: Optional callable ``(weight, name) -> value | None``
            applied to the GPTQ-corrected weight after each layer.  The return
            value is stored in :attr:`.QuantizationPlan.weight_assignments`.
            Return *None* to skip a layer from the plan.  If *None*, GPTQ
            updates weights in-place and no plan entries are created.
        group_size: Preferred group size for per-group granularity.  Layers
            whose ``in_features`` is not divisible by *group_size* fall back
            to per-channel granularity.
        nbits: Quantization bit-width passed to the GPTQ quantizer.
        symmetric: Whether to use symmetric quantization.
        damp_percent: Initial Hessian dampening fraction.
        damp_auto_increment: Dampening increment on Cholesky failure.
        share_hessians: Reuse statistics for known sibling projections that
            consume the same input tensor.
        offload_hessians: Keep full Hessians disk-backed between updates.
    """

    module_classes: tuple[type[nn.Module], ...] = field(default=(nn.Linear,))
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    weight_transform: Callable[[torch.Tensor, str], Any] | None = None
    group_size: int = 128
    nbits: int = 4
    symmetric: bool = True
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025
    ternary_threshold_ratio: float | None = None
    share_hessians: bool = False
    offload_hessians: bool = False

    # Set by prepare() - not part of the public construction interface.
    _target_modules: dict[str, nn.Module] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _device: torch.device | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _stat_source_by_target: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @staticmethod
    def _shared_input_key(name: str) -> str:
        """Return a conservative key for projections fed by the same tensor."""
        projection_name = name.removesuffix(".linear")
        parent, _, leaf = projection_name.rpartition(".")
        parent_leaf = parent.rpartition(".")[2]
        if leaf in {"q_proj", "k_proj", "v_proj"} and parent_leaf == "self_attn":
            return f"{parent}::qkv_input"
        if leaf in {"gate_proj", "up_proj"} and parent_leaf == "mlp":
            return f"{parent}::gate_up_input"
        if leaf in {"in_proj_qkv", "in_proj_z"}:
            return f"{parent}::linear_attn_input"
        return name

    def prepare(self, model: nn.Module) -> None:
        """Bind this estimator to *model*: discover target layers and device.

        Args:
            model: The model that will be passed to the pipeline.  All modules
                matching *module_classes* and *filter_fn* are recorded as
                targets for the subsequent Hessian collection and GPTQ pass.
        """
        self._device = next(model.parameters()).device
        self._target_modules = collect_target_modules(
            model, self.module_classes, self.filter_fn
        )
        self._stat_source_by_target = {}
        shared_sources: dict[str, str] = {}
        for name in self._target_modules:
            key = self._shared_input_key(name) if self.share_hessians else name
            self._stat_source_by_target[name] = shared_sources.setdefault(key, name)
        LOGGER.info(
            "GPTQEstimator: %d target modules, %d Hessians (device=%s)",
            len(self._target_modules),
            len(set(self._stat_source_by_target.values())),
            self._device,
        )

    def cache_stages(self) -> list[CacheStageDescriptor]:
        """Declare the per-layer Hessian cache stage (calibration-scoped)."""
        return [
            CacheStageDescriptor(
                "hessian",
                "Full input Hessians (one per unique projection input)",
                scope="calibration",
            )
        ]

    def declare_statistics(self) -> list[StatisticRequest]:
        """Return one Hessian request per unique target input.

        Returns an empty list when all hessians are already in cache, causing
        the pipeline to skip the calibration forward pass entirely.
        """
        cache = self._cache
        sources = list(dict.fromkeys(self._stat_source_by_target.values()))
        if all(cache.has("hessian", name) for name in sources):
            LOGGER.info(
                "GPTQEstimator: all %d hessians in cache - skipping calibration.",
                len(sources),
            )
            return []
        return [
            StatisticRequest(target_key=name, stat_type="hessian") for name in sources
        ]

    def configure_statistics_registry(self, registry: StatisticsRegistry) -> None:
        """Use disk-backed collectors so full-model Hessians do not fill RAM."""
        if not self.offload_hessians:
            return
        for source in dict.fromkeys(self._stat_source_by_target.values()):
            if self._cache.has("hessian", source):
                continue
            request = StatisticRequest(target_key=source, stat_type="hessian")
            registry.register_collector(
                request,
                HessianAccumulator(
                    name=source,
                    device=torch.device("cpu"),
                    offload_intermediate_tensor_on_disk=True,
                ),
            )

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Run GPTQ per layer and return a :class:`.QuantizationPlan`.

        Weights are updated in-place to GPTQ-corrected dequantized floats.
        If *weight_transform* is set, its output is stored in the plan's
        ``weight_assignments``.

        Args:
            graph: QGraph built by the pipeline (unused; targets are fixed by
                :meth:`prepare`).
            registry: Populated Hessian statistics from the calibration pass.
            recipe: Active recipe (unused).

        Returns:
            A :class:`.QuantizationPlan` populated by *weight_transform* for
            each successfully processed layer, or an empty plan when
            *weight_transform* is *None*.
        """
        ids_seen: set[int] = set()
        assignments: dict[str, Any] = {}
        cache = self._cache
        spans = self._spans
        saved_sources: set[str] = set()

        n_targets = len(self._target_modules)
        prog_ctx = _layer_progress(n_targets) if _RICH else contextlib.nullcontext()
        with prog_ctx as progress:
            task_id = (
                progress.add_task("GPTQ  starting…", total=n_targets) if _RICH else None
            )
            for name, mod in self._target_modules.items():
                stat_source = self._stat_source_by_target.get(name, name)
                raw_weight = mod.weight
                assert isinstance(raw_weight, torch.Tensor)
                weight_id = id(raw_weight)

                if _RICH:
                    progress.update(
                        task_id,
                        description=f"GPTQ  [dim]{_re(name)}[/dim]",
                    )

                if weight_id in ids_seen:
                    LOGGER.info("Shared weight '%s' already processed - skipping", name)
                    if _RICH:
                        progress.advance(task_id)
                    continue
                ids_seen.add(weight_id)

                LOGGER.info(
                    "GPTQ quantizing: %s  shape=%s", name, tuple(raw_weight.shape)
                )

                # Restore hessian from cache when calibration was skipped.
                cached_h = cache.load("hessian", stat_source)
                if cached_h is not None:
                    registry.set_stat(stat_source, "hessian", cached_h)

                try:
                    with timed_span(spans, "gptq.layer", layer=name):
                        _gptq_layer(
                            name,
                            mod,
                            registry,
                            self._device,
                            group_size=self.group_size,
                            nbits=self.nbits,
                            symmetric=self.symmetric,
                            damp_percent=self.damp_percent,
                            damp_auto_increment=self.damp_auto_increment,
                            ternary_threshold_ratio=self.ternary_threshold_ratio,
                            statistic_target=stat_source,
                        )
                except Exception as exc:
                    LOGGER.error("GPTQ failed for '%s': %s - skipping", name, exc)
                    if _RICH:
                        progress.advance(task_id)
                    continue

                # Save hessian to cache for future runs (skip if already cached).
                if cached_h is None and stat_source not in saved_sources:
                    stat = registry.get(
                        StatisticRequest(target_key=stat_source, stat_type="hessian")
                    )
                    if stat is not None:
                        cache.save("hessian", stat_source, stat)
                        saved_sources.add(stat_source)

                if _RICH:
                    progress.advance(task_id)

                if self.weight_transform is not None:
                    if mod.weight.dim() < 2:
                        LOGGER.warning(
                            "Skipping weight_transform for '%s': %dD weight %s",
                            name,
                            mod.weight.dim(),
                            tuple(mod.weight.shape),
                        )
                    else:
                        q = self.weight_transform(mod.weight, name)
                        if q is not None:
                            assignments[name] = q

        for source in dict.fromkeys(self._stat_source_by_target.values()):
            registry.release(StatisticRequest(target_key=source, stat_type="hessian"))

        return QuantizationPlan(weight_assignments=assignments)


# ---------------------------------------------------------------------------
# _gptq_panel_layer - GPTQ with mixed-precision panel selection
# ---------------------------------------------------------------------------


def _gptq_panel_layer(
    name: str,
    mod: nn.Module,
    registry: StatisticsRegistry,
    device: torch.device,
    group_size: int,
    possible_nbits: list[int],
    target_bpw: float,
    symmetric: bool,
    damp_percent: float,
    damp_auto_increment: float,
) -> GPTQMixedPrecisionCalibration | None:
    """Apply GPTQ with mixed-precision panel selection to *mod* in-place.

    Uses the flattened weight column count (``in_channels * kH * kW`` for
    Conv2d, ``in_features`` for Linear) to determine granularity.

    Args:
        name: Human-readable layer name for logging.
        mod: Module whose weight is quantized in-place.
        registry: Populated registry providing Hessian statistics.
        device: Computation device.
        group_size: Column-group size for panel scoring and quantization.
        possible_nbits: Candidate bit-widths (at least two required).
        target_bpw: Average bits-per-weight budget for the mixed allocation.
        symmetric: Whether to use symmetric quantization in each quantizer.
        damp_percent: Initial Hessian dampening fraction.
        damp_auto_increment: Dampening increment on Cholesky failure.

    Returns:
        The calibration object (with ``_mp_assignments`` populated) on
        success, or ``None`` when the layer is skipped.
    """
    weight_cols = reduce(operator.mul, mod.weight.shape[1:])
    if weight_cols < group_size or weight_cols % group_size != 0:
        LOGGER.warning(
            "Layer '%s': %d weight cols not divisible by group_size=%d"
            " - skipping panel GPTQ",
            name,
            weight_cols,
            group_size,
        )
        return None

    gran = Granularity.per_group([group_size])
    possible_quantizers = [
        NormZpScaleCalibration(nbits=n, symmetric=symmetric, granularity=gran)
        for n in possible_nbits
    ]

    calib = GPTQMixedPrecisionCalibration(
        granularity=gran,
        possible_quantizers=possible_quantizers,
        target_bpw=target_bpw,
        damp_percent=damp_percent,
        damp_auto_increment=damp_auto_increment,
    )
    _w = mod.weight
    assert isinstance(_w, torch.Tensor)
    calib.weight_tensor = _w
    calib.module_class = type(mod)
    calib.module = mod
    calib.name = name
    calib.device = device
    calib.init()
    registry.inject(calib, name)
    calib.compute_qscheme()
    return calib


# ---------------------------------------------------------------------------
# GPTQPanelEstimator
# ---------------------------------------------------------------------------


@dataclass
class GPTQPanelEstimator(QuantizationParamEstimator):
    """GPTQ with mixed-precision panel selection per target layer.

    Like :class:`GPTQEstimator` but uses
    :class:`~.methods.gptq_panel.GPTQMixedPrecisionCalibration` to assign
    one quantizer (selected from *possible_nbits*) per column-group under a
    *target_bpw* bit budget.

    Attributes:
        module_classes: Only modules that are instances of these classes are
            targeted.  Defaults to ``(nn.Conv2d,)``.
        filter_fn: Optional extra filter ``(name, module) -> bool``.
        group_size: Column-group size for panel scoring and quantization.
        possible_nbits: Candidate bit-widths (must contain at least two).
        target_bpw: Average bits-per-weight budget for mixed allocation.
        symmetric: Whether to use symmetric quantization.
        damp_percent: Initial Hessian dampening fraction.
        damp_auto_increment: Dampening increment on Cholesky failure.
    """

    module_classes: tuple[type[nn.Module], ...] = field(default=(nn.Conv2d,))
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    group_size: int = 32
    possible_nbits: list[int] = field(default_factory=lambda: [4, 8])
    target_bpw: float = 5.0
    symmetric: bool = True
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025

    _target_modules: dict[str, nn.Module] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _device: torch.device | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _model: nn.Module | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def prepare(self, model: nn.Module) -> None:
        """Bind this estimator to *model*: discover target layers and device.

        Args:
            model: The model that will be passed to the pipeline.
        """
        self._model = model
        self._device = next(model.parameters()).device
        self._target_modules = collect_target_modules(
            model, self.module_classes, self.filter_fn
        )
        LOGGER.info(
            "GPTQPanelEstimator: %d target modules (device=%s)",
            len(self._target_modules),
            self._device,
        )

    def declare_statistics(self) -> list[StatisticRequest]:
        """Return one Hessian request per target module."""
        return [
            StatisticRequest(target_key=name, stat_type="hessian")
            for name in self._target_modules
        ]

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Run panel GPTQ, split layers, and return packed QTensor assignments.

        Steps:

        1. Run :func:`_gptq_panel_layer` on each target module - weights are
           updated in-place to GPTQ-corrected dequantized floats and per-panel
           bit-width assignments are collected.
        2. Replace each quantized ``nn.Linear`` with a
           :class:`~neuropress.transforms.split_mixed_linear.MixedPrecisionLinear`
           that has one homogeneous-precision sub-linear per distinct bit-width.
        3. Wrap each sub-linear's weight as a bit-packed
           ``QTensorTractScaleOnly`` (Q4 → :data:`~neuropress.t2n.packing.BitPacking4`,
           Q2 → :data:`~neuropress.t2n.packing.BitPacking2`).

        Args:
            graph: QGraph built by the pipeline (unused; targets are fixed by
                :meth:`prepare`).
            registry: Populated Hessian statistics from the calibration pass.
            recipe: Active recipe (unused).

        Returns:
            A :class:`.QuantizationPlan` mapping each sub-linear path
            (e.g. ``"layer.sub_linears.0"``) to its packed QTensor.
        """
        # Lazy imports: keep transforms / t2n out of module-level imports.
        from .t2n.model_io import QuantizationManifest as _Manifest
        from .transforms.split_mixed_linear import apply_split_to_model
        from .transforms.weight_only import _make_packed_qtensor

        # ── Step 1: run panel GPTQ on all target layers ──────────────────────
        assignments_map: dict[str, list[Any]] = {}
        for name, mod in self._target_modules.items():
            LOGGER.info(
                "Panel GPTQ quantizing: %s  shape=%s", name, tuple(mod.weight.shape)
            )
            try:
                calib = _gptq_panel_layer(
                    name,
                    mod,
                    registry,
                    self._device,
                    group_size=self.group_size,
                    possible_nbits=self.possible_nbits,
                    target_bpw=self.target_bpw,
                    symmetric=self.symmetric,
                    damp_percent=self.damp_percent,
                    damp_auto_increment=self.damp_auto_increment,
                )
                if calib is not None and calib._mp_assignments is not None:
                    assignments_map[name] = calib._mp_assignments
            except Exception as exc:
                LOGGER.error("Panel GPTQ failed for '%s': %s - skipping", name, exc)

        if not assignments_map or self._model is None:
            return QuantizationPlan()

        # ── Step 2: split each nn.Linear → MixedPrecisionLinear ─────────────
        # apply_split_to_model only handles nn.Linear; Conv2d and other module
        # types are left as-is (their weights were already updated in-place by
        # _gptq_panel_layer above).
        linear_assignments = {
            name: pa
            for name, pa in assignments_map.items()
            if isinstance(self._target_modules.get(name), nn.Linear)
        }

        if not linear_assignments:
            return QuantizationPlan()

        # Pre-create the manifest so apply_split_to_model can record the split
        # transform entry; attach_manifest() will later add qtensor_assignment
        # while preserving this entry.
        if not hasattr(self._model, "__neuropress_manifest__"):
            self._model.__neuropress_manifest__ = _Manifest()  # type: ignore[attr-defined]
        apply_split_to_model(self._model, linear_assignments, self.group_size)

        # ── Step 3: wrap each sub-linear's weight as a packed QTensor ────────
        sub_assignments: dict[str, Any] = {}
        for path in linear_assignments:
            parts = path.split(".")
            mixed_lin: Any = self._model
            for part in parts:
                mixed_lin = getattr(mixed_lin, part)
            for i, (sub_lin, nbits) in enumerate(
                zip(mixed_lin.sub_linears, mixed_lin.precision_nbits, strict=True)
            ):
                sub_path = f"{path}.sub_linears.{i}"
                try:
                    q = _make_packed_qtensor(
                        sub_lin.weight, sub_path, nbits, self.group_size
                    )
                    sub_assignments[sub_path] = q
                except Exception as exc:
                    LOGGER.error(
                        "QTensor wrapping failed for '%s': %s - skipping",
                        sub_path,
                        exc,
                    )

        return QuantizationPlan(weight_assignments=sub_assignments)


# ---------------------------------------------------------------------------
# MinMaxZpWeightEstimator
# ---------------------------------------------------------------------------


@dataclass
class MinMaxZpWeightEstimator(QuantizationParamEstimator):
    """Min-max weight quantisation estimator - no calibration data required.

    Applies per-tensor symmetric or asymmetric uniform quantisation directly
    to each target module's weight and stores the result back in-place.  No
    forward passes are needed (``declare_statistics`` returns an empty list).

    When *weight_transform* is set, the dequantised float is passed through
    the transform (e.g. ``q4_0_packed``) and the resulting packed QTensor is
    stored in the plan.  When *None*, the dequantised float itself is stored.

    Attributes:
        module_classes: Only modules that are instances of these classes are
            targeted.  Defaults to ``(nn.Embedding,)``.
        filter_fn: Optional extra filter ``(name, module) -> bool`` applied
            in addition to *module_classes*.  Use this to enforce shape
            constraints (e.g. group-size divisibility for tract Q4_0).
        weight_transform: Optional callable ``(weight, name) -> value | None``
            applied after in-place quantisation.  The return value is stored
            in :attr:`.QuantizationPlan.weight_assignments`.  Return *None* to
            skip the layer from the plan.
        nbits: Quantisation bit-width.
        symmetric: Whether to use symmetric quantisation.
        group_size: Per-group size used by the weight transform for scale
            storage overhead estimation.  Only relevant when *weight_transform*
            is set.
    """

    module_classes: tuple[type[nn.Module], ...] = field(default=(nn.Embedding,))
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    weight_transform: Callable[[torch.Tensor, str], Any] | None = None
    nbits: int = 8
    symmetric: bool = True
    group_size: int = 32

    _target_modules: dict[str, nn.Module] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def prepare(self, model: nn.Module) -> None:
        """Discover target modules.

        Args:
            model: The model that will be passed to the pipeline.
        """
        self._target_modules = collect_target_modules(
            model, self.module_classes, self.filter_fn
        )
        LOGGER.info(
            "MinMaxZpWeightEstimator: %d target modules", len(self._target_modules)
        )

    def declare_statistics(self) -> list[StatisticRequest]:
        """Return an empty list - weights are quantised directly."""
        return []

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Quantise each target weight in-place and store the plan value.

        When *weight_transform* is set the packed QTensor returned by the
        transform is stored; otherwise the dequantised float tensor is stored.

        Args:
            graph: Unused.
            registry: Unused.
            recipe: Unused.

        Returns:
            A :class:`.QuantizationPlan` mapping each target layer name to its
            plan value (packed QTensor or dequantised float tensor).
        """
        assignments: dict[str, Any] = {}
        half = 2 ** (self.nbits - 1)
        for name, mod in self._target_modules.items():
            # Give this module a private weight tensor before mutating it
            # in-place below.  If mod.weight is tied to another module's
            # weight (e.g. embed_tokens/lm_head with tie_word_embeddings),
            # the in-place .copy_() a few lines down would otherwise
            # silently corrupt that other module too, even though it was
            # never a target of this estimator.
            mod._parameters["weight"] = nn.Parameter(
                mod.weight.detach().clone(), requires_grad=mod.weight.requires_grad
            )
            w = mod.weight.data.float()
            if self.symmetric:
                maxabs = w.abs().max().clamp(min=1e-8)
                scale = maxabs / (half - 1)
                q = (w / scale).round().clamp(-half, half - 1)
                dq = q * scale
            else:
                wmin, wmax = w.min(), w.max()
                scale = (wmax - wmin).clamp(min=1e-8) / (2**self.nbits - 1)
                q = ((w - wmin) / scale).round().clamp(0, 2**self.nbits - 1)
                dq = q * scale + wmin
            dq = dq.to(mod.weight.dtype)
            mod.weight.data.copy_(dq)
            LOGGER.info(
                "MinMaxZp: quantised '%s' in-place (%d-bit, symmetric=%s)",
                name,
                self.nbits,
                self.symmetric,
            )
            if self.weight_transform is not None:
                if mod.weight.dim() < 2:
                    LOGGER.warning(
                        "Skipping weight_transform for '%s': weight is %dD (shape %s)",
                        name,
                        mod.weight.dim(),
                        tuple(mod.weight.shape),
                    )
                else:
                    packed = self.weight_transform(mod.weight, name)
                    if packed is not None:
                        assignments[name] = packed
            else:
                assignments[name] = dq
        return QuantizationPlan(weight_assignments=assignments)


# ---------------------------------------------------------------------------
# VPTQGroupEstimator - shared-codebook vector quantization
# ---------------------------------------------------------------------------


@dataclass
class VPTQGroupEstimator(QuantizationParamEstimator):
    """Two-pass estimator that trains a shared VQ codebook across all target layers.

    **Algorithm overview**

    1. *Prepare*: discover all target :class:`torch.nn.Module` instances.
    2. *Statistics* (optional): request ``"hessian_diag"`` for each target so
       the registry collects per-feature activation statistics in one forward
       pass.  Skipped when :attr:`use_hessian_weighting` is ``False``.
    3. *Estimate*:

       a. **Centroid training** - pool weight vectors from all layers that
          share a proxy (controlled by :attr:`grouping`) and run k-means to
          produce shared centroids ``[num_centroids, vector_len]``.
       b. **Index assignment** - for each layer, assign every weight vector to
          its nearest centroid.  When ``hessian_diag`` statistics are
          available, the distance is importance-weighted via
          :func:`~neuropress.methods.vptq.importance_weighted_assign`.
       c. Each layer gets its own :class:`~neuropress.transforms.vq.VQScheme`
          that *references the same* ``centroids`` tensor, realising the
          shared-proxy property.

    Attributes:
        module_classes: Module types to quantize.  Defaults to
            ``(nn.Linear,)``.
        filter_fn: Optional extra filter ``(name, module) -> bool``.
        num_centroids: Codebook size (number of entries).  Must be ``<= 256``
            (8-bit indices).  Use ``<= 16`` for 4-bit indices.
        vector_len: Number of consecutive weight elements per VQ vector.
            Must evenly divide the number of columns (``in_features`` for
            ``nn.Linear``, ``embedding_dim`` for ``nn.Embedding``).
        proxy_selector: Distance metric for centroid assignment.
            Importance weighting is applied only with ``L2``; ``COSINE`` falls
            back to unweighted assignment when ``hessian_diag`` is available.
        grouping: How to group layers for shared centroids.

            - ``"all"``          - one codebook shared across *all* target layers.
            - ``"by_type"``      - one codebook per distinct module class.  Not
              useful for distinguishing weight roles within a single class
              (e.g. all ``nn.Linear`` layers collapse to one group) — see
              ``"by_leaf_name"``.
            - ``"by_leaf_name"`` - one codebook per distinct trailing
              attribute name (``name.split(".")[-1]``), e.g. ``down_proj``
              vs. ``gate_proj``/``up_proj``.  Use this when different
              structural roles have different weight-magnitude
              distributions and pooling them under one codebook starves the
              rarer/larger-magnitude role of centroids (confirmed on
              Qwen3.5-2B: a shared gate/up/down codebook clipped
              ``down_proj``'s outlier weights to roughly half their true
              range because the pooled codebook never allocated centroids
              reaching them; a dedicated per-leaf-name codebook fixed it).
            - ``"per_layer"``    - one codebook per individual target module
              (group key = the full module name) — no sharing at all.  This
              is the original VPTQ paper's design (each weight matrix gets
              its own codebook); ``"all"``/``"by_type"``/``"by_leaf_name"``
              are this codebase's departures from it, trading some
              per-layer fit quality for many fewer, larger k-means fits.
              Every group needs at least *num_centroids* training vectors
              (``raw.shape[0] // vector_len``), so very small individual
              layers (e.g. a ``(16, hidden)`` DeltaNet gate projection) will
              raise unless excluded via *filter_fn* first.

        cluster_backend: Clustering implementation.  ``None`` → auto-select
            via :func:`~neuropress.transforms.cluster_backends.best_available_backend`.
        use_hessian_weighting: When ``True``, declares ``"hessian_diag"``
            statistic requests and uses them for importance-weighted
            centroid assignment.
        use_tiered: When ``True``, passes ``use_tiered=True`` to every
            :class:`~neuropress.transforms.vq.VQScheme` produced, causing
            :class:`~neuropress.t2n.qtensors.QTensorTractVQ` to store
            indices as two-tier prefix-coded streams via
            :class:`~neuropress.transforms.pack.TieredIndexPacking`.
        zero_escape: When ``True``, passes ``zero_escape=True`` to every
            :class:`~neuropress.transforms.vq.VQScheme` produced, using
            flat 2-bit prefix coding via
            :class:`~neuropress.transforms.vq_packing.ZeroEscapePacking`.
            Overrides *use_tiered*.
    """

    module_classes: tuple[type[nn.Module], ...] = field(default=(nn.Linear,))
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    num_centroids: int = 256
    num_centroids_per_group: dict[str, int] | None = None
    vector_len: int = 8
    proxy_selector: Any = None  # ProxySelector.L2 - set via __post_init__
    grouping: str = "all"
    cluster_backend: Any = None  # ClusterBackend | None
    use_hessian_weighting: bool = True
    use_tiered: bool = False
    zero_escape: bool = False
    max_n_iter: int = 50
    tol: float = 1e-4
    max_train_vectors: int = 50_000_000
    vram_budget: float = 0.5
    centroid_weighting: str = "uniform"
    weighting_alpha: float = 0.5
    coverage_beta: float = 0.0
    coverage_r_max: float | None = None
    temperature_annealing: bool = False
    temperature_decay: float = 0.7
    concentration_beta: float = 0.0
    gptq_correction_iters: int = 0
    gptq_correction_alpha: float = 1.0
    gptq_damp_percent: float = 0.01
    gptq_damp_auto_increment: float = 0.0025
    centroid_attraction: float = 0.0
    centroid_attraction_window: int = 4
    correction_mode: CorrectionMode = CorrectionMode.SEQUENTIAL
    ecvq_beta_schedule: tuple[float, ...] = ()
    ecvq_iters_per_step: int = 5
    # Keep ECVQ off by default.  It trades reconstruction distortion for lower
    # index entropy, but indices are stored flat, so that rate saving is never
    # collected -- it is a pure quality cost.  Only worth enabling once
    # per-layer entropy coding exists to spend the rate it buys.
    ecvq_beta_in_loop: float = 0.0
    ecvq_warmup_iters: int = 20

    _target_modules: dict[str, nn.Module] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _device: torch.device | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cached_vector_weights: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _m_step_acc: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _m_step_counts: torch.Tensor | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # When True, the next _refine_centroids call emits a per-op timing table
    # then resets itself to False.  Toggle via estimator._profile_next_mstep = True
    # or env var VPTQ_PROFILE_MSTEP=1 (set before the first M-step).
    _profile_next_mstep: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _corrector: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Apply defaults and build optional sub-objects after dataclass init."""
        if self.proxy_selector is None:
            from .transforms.vq import ProxySelector

            self.proxy_selector = ProxySelector.L2
        if self.grouping not in {"all", "by_type", "by_leaf_name", "per_layer"}:
            raise ValueError(
                f"grouping must be 'all', 'by_type', 'by_leaf_name', or "
                f"'per_layer', got {self.grouping!r}."
            )
        if self.centroid_weighting not in {"uniform", "hessian"}:
            raise ValueError(
                f"centroid_weighting must be 'uniform' or 'hessian', "
                f"got {self.centroid_weighting!r}."
            )
        if self.correction_mode not in set(CorrectionMode):
            raise ValueError(
                f"correction_mode must be one of {[m.value for m in CorrectionMode]}, "
                f"got {self.correction_mode!r}."
            )
        if self.gptq_correction_iters:
            from .methods.vq_gptq import VQGPTQCorrector

            self._corrector = VQGPTQCorrector(
                vector_len=self.vector_len,
                gptq_correction_alpha=self.gptq_correction_alpha,
                gptq_damp_percent=self.gptq_damp_percent,
                gptq_damp_auto_increment=self.gptq_damp_auto_increment,
                centroid_attraction=self.centroid_attraction,
                centroid_attraction_window=self.centroid_attraction_window,
            )

    def cache_stages(self) -> list[CacheStageDescriptor]:
        """Declare the codebook/assignment stages plus whichever Hessian stage applies.

        GPTQ correction needs the full Hessian; plain Hessian weighting needs only
        the diagonal; neither is requested when the estimator uses neither.
        """
        stages = [
            CacheStageDescriptor(
                "centroids",
                "Trained codebook centroids (one per group)",
                scope="recipe",
            ),
            CacheStageDescriptor(
                "assignments",
                "Packed VQ index assignments (one per layer)",
                scope="recipe",
            ),
        ]
        if self.gptq_correction_iters:
            stages.insert(
                0,
                CacheStageDescriptor(
                    "hessian", "Full input Hessians", scope="calibration"
                ),
            )
        elif self.use_hessian_weighting:
            stages.insert(
                0,
                CacheStageDescriptor(
                    "hessian_diag", "Hessian diagonals", scope="calibration"
                ),
            )
        return stages

    def prepare(self, model: nn.Module) -> None:
        """Discover target layers and record the compute device.

        Args:
            model: Model to quantize.
        """
        self._device = next(model.parameters()).device
        self._target_modules = collect_target_modules(
            model, self.module_classes, self.filter_fn
        )
        LOGGER.info(
            "VPTQGroupEstimator: %d target modules (device=%s)",
            len(self._target_modules),
            self._device,
        )

    def declare_statistics(self) -> list[StatisticRequest]:
        """Return statistic requests for importance weighting and/or GPTQ correction.

        When :attr:`use_gptq_correction` is ``True``, requests the full
        ``"hessian"`` per layer (the diagonal is extracted from it for
        importance-weighted assignment).  Otherwise, requests only the cheaper
        ``"hessian_diag"`` when :attr:`use_hessian_weighting` is ``True``.

        Returns an empty list when neither feature is active, or when all
        required hessians are already in the cache (skipping the forward pass).
        """
        if not self.gptq_correction_iters and not self.use_hessian_weighting:
            return []
        stat_type = "hessian" if self.gptq_correction_iters else "hessian_diag"
        eligible = [
            name
            for name, mod in self._target_modules.items()
            if not isinstance(mod, nn.Embedding)
        ]
        cache = self._cache
        if eligible and all(cache.has(stat_type, name) for name in eligible):
            LOGGER.info(
                "VPTQGroupEstimator: all %d %s in cache - skipping calibration.",
                len(eligible),
                stat_type,
            )
            return []
        requests = []
        for name, mod in self._target_modules.items():
            if isinstance(mod, nn.Embedding):
                continue
            if self.gptq_correction_iters:
                requests.append(StatisticRequest(target_key=name, stat_type="hessian"))
            else:
                requests.append(
                    StatisticRequest(target_key=name, stat_type="hessian_diag")
                )
        return requests

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_layer_vectors(self, mod: nn.Module) -> int | None:
        """Return the number of weight vectors for *mod* without materialising.

        Returns ``None`` when the weight is incompatible.
        """
        w = mod.weight
        if w.ndim < 2:
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            return None
        return w.numel() // self.vector_len

    def _layer_vectors_raw(self, mod: nn.Module) -> torch.Tensor | None:
        """Return a view of the weight reshaped to ``[n_vecs, vector_len]``.

        Keeps the original dtype (no float32 conversion) so the caller can
        subsample *before* converting, keeping allocations minimal.  Returns
        ``None`` when the weight is incompatible.
        """
        w = mod.weight
        if w.ndim < 2:
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            return None
        return w.data.reshape(-1, self.vector_len)  # view when contiguous

    def _layer_vectors(self, mod: nn.Module) -> torch.Tensor | None:
        """Return float32 weight vectors ``[num_vectors, vector_len]`` for *mod*.

        Returns ``None`` when the weight is incompatible (wrong ndim or
        not divisible by :attr:`vector_len`).
        """
        raw = self._layer_vectors_raw(mod)
        if raw is None:
            return None
        return raw.float()

    def cached_centroids_usable(
        self, centroids: torch.Tensor, expected_nc: int | None = None
    ) -> bool:
        """Return whether a cached codebook matches the configured geometry.

        Both dimensions matter, and each has drawn blood:

        * ``vector_len`` (columns) - on 2026-08-05 a VL=8 run consumed a VL=4
          codebook and died ~1000 lines later with a 2x size mismatch, having
          trained nothing.
        * ``num_centroids`` (rows) - on 2026-08-07 a run configured with
          ``num_centroids_per_group`` loaded 12 cached 16384-row codebooks from
          an earlier uniform run.  All passed the ``vector_len`` check, so the
          run silently measured the *uniform* configuration; three arms
          reported as distinct turned out to be one model.

        A wrong row count cannot fail loudly later - it is a perfectly valid
        codebook, just not the configured one - so it must be caught here.

        Args:
            centroids: Candidate codebook from the cache.
            expected_nc: Codebook size required for this group, from
                :meth:`centroids_for_group`.  ``None`` skips the size check.

        Returns:
            True when the tensor is 2-D with ``vector_len`` columns and, when
            *expected_nc* is given, exactly that many rows.
        """
        if centroids.ndim != 2 or centroids.shape[1] != self.vector_len:
            return False
        if expected_nc is not None and centroids.shape[0] != expected_nc:
            return False
        return True

    def cached_assignment_usable(
        self, packed: torch.Tensor, expected_n: int, bits: int
    ) -> bool:
        """Return whether a cached packed assignment has the expected size.

        Checks the *packed blob* rather than the unpacked indices:
        :meth:`IndexPacking.unpack` takes ``n_indices`` as an argument and
        always returns exactly that many, so unpacked length carries no
        information.  A stale blob from a different ``vector_len`` holds a
        different number of indices and therefore a different byte count.

        Args:
            packed: Candidate packed blob from the cache.
            expected_n: Index count implied by the layer shape and
                ``vector_len``.
            bits: Bits per index for the current codebook.

        Returns:
            True when the blob size matches ``ceil(expected_n * bits / 8)``.
        """
        return packed.numel() == (expected_n * bits + 7) // 8

    def centroids_for_group(self, group_key: str) -> int:
        """Return the codebook size to use for *group_key*.

        Measured index entropy varies 7.85-13.69 bits across layer types, so a
        uniform ``num_centroids`` spends 14 bits addressing ~230 effective
        centroids on ``down_proj`` (22% of the model) while ``in_proj_qkv``
        genuinely needs all 16384.  Sizing per group cuts ~15% of the bit-rate
        without starving the types that need capacity.

        The bit width follows automatically: the caller derives it from the
        returned centroid tensor via ``IndexPacking.bits_for``, not from this
        setting.

        Args:
            group_key: Group key from :meth:`_make_groups` — the leaf module
                name under ``grouping="by_leaf_name"``.

        Returns:
            Codebook size for this group.

        Raises:
            ValueError: If an override for *group_key* is not positive.
        """
        if not self.num_centroids_per_group:
            return self.num_centroids
        nc = self.num_centroids_per_group.get(group_key)
        if nc is None:
            return self.num_centroids
        if nc <= 0:
            raise ValueError(
                f"num_centroids_per_group[{group_key!r}] must be positive, got {nc}."
            )
        return nc

    def _make_groups(self) -> dict[str, list[tuple[str, nn.Module]]]:
        """Partition target modules into centroid-sharing groups.

        Returns:
            Mapping from group key to list of ``(name, module)`` pairs.
        """
        groups: dict[str, list[tuple[str, nn.Module]]] = {}
        for name, mod in self._target_modules.items():
            if self.grouping == "by_type":
                key = type(mod).__name__
            elif self.grouping == "by_leaf_name":
                key = name.split(".")[-1]
            elif self.grouping == "per_layer":
                key = name
            else:
                key = "all"
            groups.setdefault(key, []).append((name, mod))
        return groups

    def _compute_layer_vector_weights(
        self,
        name: str,
        mod: nn.Module,
        registry: StatisticsRegistry,
        column_perm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-vector importance weight for one layer.

        Uses the Hessian diagonal to compute a scalar importance per vector
        position, then tiles across rows.  Falls back to uniform (all-ones)
        when the Hessian is unavailable.

        Args:
            name: Module name key in the registry.
            mod: Target module.
            registry: Populated statistics registry.
            column_perm: Optional column permutation to reorder the Hessian
                diagonal before grouping into vectors.

        Returns:
            Float32 tensor ``[n_vectors]``.
        """
        w = mod.weight.data
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        n_rows = w.shape[0]
        n_vecs_per_row = K // self.vector_len
        n_vectors = n_rows * n_vecs_per_row

        diag_h = self._get_diag_h(name, K, registry)
        if diag_h is None:
            return torch.ones(n_vectors, dtype=torch.float32)

        if column_perm is not None:
            diag_h = diag_h[column_perm.to(diag_h.device)]

        # [K] -> [K//VL, VL] -> mean across VL -> [K//VL]
        per_pos = diag_h.float().reshape(-1, self.vector_len).mean(dim=1)
        per_pos = per_pos.clamp(min=1e-12) ** self.weighting_alpha
        # Tile across rows: same column-position importance for every row.
        weights = per_pos.unsqueeze(0).expand(n_rows, -1).reshape(-1)
        return weights

    def _layer_has_hessian(self, name: str, mod: nn.Module, registry) -> bool:
        K = (
            int(mod.weight.shape[1])
            if mod.weight.ndim == 2
            else int(mod.weight.numel() // mod.weight.shape[0])
        )
        return self._get_diag_h(name, K, registry) is not None

    @staticmethod
    def _scale_no_hessian_weights(
        weights: dict[str, torch.Tensor],
        no_h: set[str],
    ) -> None:
        """Scale no-Hessian layer weights to match mean of Hessian-equipped layers."""
        has_h = [n for n in weights if n not in no_h]
        if not has_h or not (no_h & weights.keys()):
            return
        total_w, total_n = 0.0, 0
        for n in has_h:
            w = weights[n]
            total_w += w.sum().item()
            total_n += w.numel()
        h_mean = total_w / max(total_n, 1)
        for n in no_h:
            if n in weights:
                weights[n] = weights[n] * h_mean

    def _train_centroids(
        self,
        members: list[tuple[str, nn.Module]],
        registry: StatisticsRegistry | None = None,
        group_key: str | None = None,
    ) -> torch.Tensor | None:
        """Pool weight vectors from *members* and run k-means.

        Vectors are kept on their original device (GPU when available) so
        that :class:`~.transforms.cluster_backends.TorchKMeansBackend` can
        exploit CUDA without a round-trip through CPU.

        Args:
            members: List of ``(name, module)`` pairs in this group.
            registry: Statistics registry for importance weights.  Only used
                when ``centroid_weighting="hessian"``.
            group_key: Grouping key for this set of members, used to look up a
                per-group codebook size via :meth:`centroids_for_group`.  When
                ``None``, the uniform :attr:`num_centroids` is used.

        Returns:
            ``float16`` centroids of shape ``[num_centroids, vector_len]``,
            or ``None`` when no compatible layers were found.
        """
        n_centroids = (
            self.centroids_for_group(group_key)
            if group_key is not None
            else self.num_centroids
        )
        import time

        from .transforms.cluster_backends import best_available_backend

        use_weighting = self.centroid_weighting == "hessian" and registry is not None

        all_vecs: list[torch.Tensor] = []
        all_names: list[str] = []
        all_wts: list[torch.Tensor] = []
        n_layers = sum(
            1 for _, mod in members if self._n_layer_vectors(mod) is not None
        )
        # Per-layer vector cap: spread max_train_vectors evenly across layers so
        # that torch.cat never OOMs on large models (e.g. 2B+ with grouping="all").
        per_layer_cap = max(
            n_centroids * 10,
            self.max_train_vectors // max(n_layers, 1),
        )
        for name, mod in members:
            raw = self._layer_vectors_raw(mod)
            if raw is None:
                continue
            if raw.shape[0] > per_layer_cap:
                # Subsample *before* dtype conversion so the float32 copy is
                # tiny (per_layer_cap x VL x 4 bytes) rather than full-weight
                # sized.  idx lives on CPU; we move it to the weight's device
                # only for the (cheap) fancy-index step, then come back to CPU.
                idx = torch.randperm(raw.shape[0])[:per_layer_cap]
                vecs = raw[idx.to(raw.device)].float().cpu()
            else:
                vecs = raw.float()
                if isinstance(mod, nn.Embedding) and vecs.is_cuda:
                    vecs = vecs.cpu()
            all_vecs.append(vecs)
            all_names.append(name)
            if use_weighting:
                wt = self._compute_layer_vector_weights(name, mod, registry)
                if vecs.shape[0] < wt.shape[0]:  # was subsampled
                    wt = wt[idx.cpu()]
                all_wts.append(wt)

        if not all_vecs:
            return None

        # Layer-aware scaling: boost no-Hessian layers (embeddings) to compete
        # with Hessian-equipped layers, then scale each layer by its sensitivity.
        if all_wts:
            wts_by_name = dict(zip(all_names, all_wts, strict=True))
            no_h = {
                n for n, m in members if not self._layer_has_hessian(n, m, registry)
            }
            self._scale_no_hessian_weights(wts_by_name, no_h)
            all_wts = [wts_by_name[n] for n in all_names]
            layer_means = [wt.mean().item() for wt in all_wts]
            global_mean = sum(layer_means) / len(layer_means)
            for i, lm in enumerate(layer_means):
                scale = lm / (global_mean + 1e-12)
                all_wts[i] = all_wts[i] * scale

        # Unify device: prefer CUDA when any layer is on CUDA, so the k-means
        # backend can exploit GPU.  Subsampled embedding vecs are small (≤few MB)
        # so moving them to CUDA is cheap.
        target_dev = next(
            (v.device for v in all_vecs if v.device.type == "cuda"),
            torch.device("cpu"),
        )
        # per_layer_cap forces every over-cap layer's subsample to .cpu() (see
        # above) purely to keep the fancy-index step cheap.  When a group has
        # few layers (e.g. grouping="by_leaf_name"), per_layer_cap can end up
        # large enough that EVERY layer trips it, so target_dev falls back to
        # CPU even though the pooled result is tiny relative to VRAM -- and
        # the IVF backend has no CPU fast path (torch.compile-only), so this
        # silently turns a few-minute GPU fit into a many-hour CPU one.
        # Promote back to CUDA when the pooled tensor actually fits.
        if target_dev.type == "cpu" and torch.cuda.is_available():
            # A prior group's k-means fit (e.g. the IVF backend's
            # torch.compile'd kernels) can leave GPU memory allocated but
            # unused; empty_cache() releases PyTorch's cached-but-free
            # blocks back to the driver so mem_get_info() below reports an
            # accurate (not stale-pessimistic) free-memory figure.
            torch.cuda.empty_cache()
            needed_bytes = sum(v.numel() for v in all_vecs) * all_vecs[0].element_size()
            free_bytes, _ = torch.cuda.mem_get_info()
            if needed_bytes < free_bytes * self.vram_budget:
                LOGGER.info(
                    "VPTQGroupEstimator: promoting pooled vectors from CPU to "
                    "CUDA before k-means (%.1f MB fits within %.0f%% VRAM budget; "
                    "per-layer caps had forced every layer's subsample to CPU)",
                    needed_bytes / 2**20,
                    self.vram_budget * 100,
                )
                target_dev = torch.device("cuda")
        pooled = torch.cat([v.to(target_dev) for v in all_vecs], dim=0)  # [N, VL]
        total_vectors = pooled.shape[0]

        # Resolve backend early so the VRAM budget is computed correctly.
        # CudaKMeansBackend uses a tiled shmem E-step - it never materialises
        # the full [N, k] distance matrix, so the VRAM budget formula below
        # does not apply and must be skipped.
        backend = self.cluster_backend or best_available_backend(
            n_clusters=n_centroids, vector_len=self.vector_len
        )

        from .transforms.cluster_backends import CudaKMeansBackend

        cuda_tiled = isinstance(backend, CudaKMeansBackend)

        # Compute effective cap: VRAM-limited on GPU, max_train_vectors on CPU.
        # Skip for CudaKMeansBackend - tiled kernel has no [N, k] matrix.
        effective_max = self.max_train_vectors
        if pooled.device.type == "cuda" and not cuda_tiled:
            # Same staleness concern as above: release any memory a prior
            # group's fit left cached-but-unused before reading free VRAM.
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info(pooled.device)
            bytes_per_elem = pooled.element_size()
            vram_max = int(
                free_bytes * self.vram_budget / (n_centroids * bytes_per_elem)
            )
            effective_max = min(self.max_train_vectors, vram_max)
            LOGGER.info(
                "VPTQGroupEstimator: VRAM %.1f/%.1f GB free  "
                "→ distance-matrix budget %d vectors "
                "(k=%d, dtype=%s, budget=%.0f%%)",
                free_bytes / 2**30,
                total_bytes / 2**30,
                vram_max,
                n_centroids,
                pooled.dtype,
                self.vram_budget * 100,
            )

        pooled_weights: torch.Tensor | None = None
        if all_wts:
            pooled_weights = torch.cat([w.to(target_dev) for w in all_wts], dim=0)
            # Normalize to mean=1.0 to preserve effective sample size.
            pooled_weights = pooled_weights / (pooled_weights.mean() + 1e-12)

        if total_vectors > effective_max:
            LOGGER.info(
                "VPTQGroupEstimator: subsampling %d → %d vectors (device=%s, cap=%s)",
                total_vectors,
                effective_max,
                pooled.device,
                "VRAM" if pooled.device.type == "cuda" else "max_train_vectors",
            )
            idx = torch.randperm(total_vectors)[:effective_max].to(pooled.device)
            pooled = pooled[idx]
            if pooled_weights is not None:
                pooled_weights = pooled_weights[idx]
        else:
            LOGGER.info(
                "VPTQGroupEstimator: using all %d vectors from %d layers "
                "(device=%s, no subsampling needed)",
                total_vectors,
                len(all_vecs),
                pooled.device,
            )

        if pooled.shape[0] < n_centroids:
            raise RuntimeError(
                f"VPTQGroupEstimator: only {pooled.shape[0]} training vectors "
                f"available after subsampling but num_centroids={n_centroids}. "
                f"Reducing NC is not allowed (would silently corrupt quality). "
                f"Use a larger model, more layers per group, or a smaller codebook."
            )
        n_clusters = n_centroids
        LOGGER.info(
            "VPTQGroupEstimator: k-means  %d centroids / %d vectors  "
            "backend=%s  device=%s",
            n_clusters,
            pooled.shape[0],
            type(backend).__name__,
            pooled.device,
        )
        if pooled_weights is not None:
            LOGGER.info(
                "VPTQGroupEstimator: using hessian-weighted centroids "
                "(alpha=%.2f, weight range=%.3f-%.3f)",
                self.weighting_alpha,
                pooled_weights.min().item(),
                pooled_weights.max().item(),
            )
        t0 = time.perf_counter()
        centroids, _ = backend.fit(
            pooled,
            n_clusters,
            sample_weights=pooled_weights,
        )
        LOGGER.info(
            "VPTQGroupEstimator: k-means done in %.1fs",
            time.perf_counter() - t0,
        )
        return centroids.to(torch.float16)

    def _get_diag_h(
        self,
        name: str,
        K: int,
        registry: StatisticsRegistry,
    ) -> torch.Tensor | None:
        """Return the Hessian diagonal ``[K]`` for *name* if weighting is enabled.

        Checks the cache before the registry.  Saves to cache on a registry hit
        so future runs can skip the calibration forward pass.

        Args:
            name: Module name key in the registry.
            K: Expected feature dimension; result is discarded if shape mismatches.
            registry: Populated statistics registry.

        Returns:
            Float tensor ``[K]`` or ``None``.
        """
        from .transforms.vq import ProxySelector

        if not (self.use_hessian_weighting and self.proxy_selector == ProxySelector.L2):
            return None
        if self.gptq_correction_iters:
            h = self._get_full_hessian(name, K, registry)
            return h.diagonal().float() if h is not None else None
        cache = self._cache
        # Try cache first (warm on resumed runs).
        cached = cache.load("hessian_diag", name)
        if cached is not None:
            diag_h = cached.get("hessian_diag")
            if diag_h is not None and diag_h.shape == (K,):
                return diag_h.to(self._device or torch.device("cpu"))
        # Fall back to registry.
        result = registry.get(
            StatisticRequest(target_key=name, stat_type="hessian_diag")
        )
        if result is None:
            LOGGER.debug(
                "VPTQGroupEstimator: no hessian_diag for '%s' - "
                "falling back to plain L2.",
                name,
            )
            return None
        diag_h = result.get("hessian_diag")
        if diag_h is None or diag_h.shape != (K,):
            return None
        # Persist for next run.
        cache.save("hessian_diag", name, {"hessian_diag": diag_h})
        return diag_h

    def _get_full_hessian(
        self,
        name: str,
        K: int,
        registry: StatisticsRegistry,
    ) -> torch.Tensor | None:
        """Return the full ``[K, K]`` Hessian for *name*, or ``None``.

        Checks the cache before the registry.  Saves to cache on a registry hit.
        """
        cache = self._cache
        dev = self._device or torch.device("cpu")
        # Try cache first.
        cached = cache.load("hessian", name)
        if cached is not None:
            h = cached.get("hessian")
            if h is not None and h.shape == (K, K):
                return h.to(dev)
        # Fall back to registry.
        result = registry.get(StatisticRequest(target_key=name, stat_type="hessian"))
        if result is None:
            return None
        h = result.get("hessian")
        if h is None:
            return None
        if not isinstance(h, torch.Tensor):
            h = h.to_base_tensor()
        if h.shape != (K, K):
            return None
        # Persist for next run (CPU float32 to avoid GPU-memory pinning).
        cache.save("hessian", name, {"hessian": h.cpu().float()})
        return h

    def _gptq_corrected_assign(
        self,
        name: str,
        mod: nn.Module,
        centroids: torch.Tensor,
        registry: StatisticsRegistry,
        column_perm: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Delegate GPTQ-corrected assignment to :class:`~.methods.vq_gptq.VQGPTQCorrector`.

        Fetches the full Hessian and optional diagonal from cache/registry,
        then hands off all computation to ``self._corrector``.  Returns
        ``None`` when the Hessian is unavailable (caller keeps existing assignment).
        """
        w = mod.weight.data
        if w.ndim < 2:
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            return None
        dev = self._device or torch.device("cpu")
        full_h = self._get_full_hessian(name, K, registry)
        if full_h is None:
            LOGGER.debug(
                "VPTQGroupEstimator: no full Hessian for '%s' - skipping GPTQ correction.",
                name,
            )
            return None
        diag_h = self._get_diag_h(name, K, registry)
        return self._corrector.correct(
            name=name,
            mod=mod,
            centroids=centroids,
            full_h=full_h,
            device=dev,
            column_perm=column_perm,
            diag_h=diag_h,
        )

    def _gho_sort_members(
        self,
        members: list[tuple[str, nn.Module]],
        registry: StatisticsRegistry,
    ) -> list[tuple[str, nn.Module]]:
        """Return members sorted by total Hessian importance, descending.

        Layers with no Hessian (diag_h=None) are sorted last with importance=0.
        This gives globally-important layers priority in the GPTQ correction pass.
        """

        def _importance(name_mod: tuple[str, nn.Module]) -> float:
            name, mod = name_mod
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            diag_h = self._get_diag_h(name, K, registry)
            if diag_h is None:
                return 0.0
            return float(diag_h.sum().item())

        return sorted(members, key=_importance, reverse=True)

    def _compute_centroid_importance(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        NC: int,
        bits: int,
        registry: StatisticsRegistry,
    ) -> torch.Tensor:
        """Return per-centroid global Hessian importance ``[NC]``.

        For each centroid, sums the diag_h values of all column groups
        assigned to it across all layers.  Centroids with no Hessian
        data contributing receive importance 0.
        """
        from .transforms.vq_packing import IndexPacking

        importance = torch.zeros(NC, dtype=torch.float64)
        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            G = K // self.vector_len
            n_rows = w.shape[0]
            n_indices = n_rows * G
            indices = IndexPacking.unpack(packed, bits, n_indices).reshape(n_rows, G)
            diag_h = self._get_diag_h(name, K, registry)
            if diag_h is None:
                continue
            # Per-group importance: mean diag_h across the group's VL columns. [G]
            per_group_h = diag_h.float().reshape(G, self.vector_len).mean(dim=1)
            # Scatter-add: vectorized over all groups simultaneously.
            weights = per_group_h.unsqueeze(0).expand(n_rows, G)  # [n_rows, G]
            importance.scatter_add_(
                0,
                indices.flatten().to(torch.int64).cpu(),
                weights.flatten().double().cpu(),
            )
        return importance.float()

    def _centric_group_perm(
        self,
        name: str,
        mod: nn.Module,
        packed: torch.Tensor,
        bits: int,
        centroid_importance: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return column permutation encoding centroid-importance group order.

        Groups assigned to globally-important centroids are processed first
        in the GPTQ pass.  Within each group, column order is preserved.

        Returns:
            Int64 column permutation ``[K]``, or ``None`` if not applicable.
        """
        from .transforms.vq_packing import IndexPacking

        w = mod.weight.data
        if w.ndim < 2:
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            return None
        VL = self.vector_len
        G = K // VL
        n_rows = w.shape[0]
        n_indices = n_rows * G
        indices = IndexPacking.unpack(packed, bits, n_indices).reshape(n_rows, G)

        # Representative centroid per group: most-common assignment across rows.
        group_centroid = indices.mode(dim=0).values.cpu()  # [G]

        # Group importance = importance of the representative centroid.
        group_imp = centroid_importance[group_centroid]  # [G]

        # Sort groups by descending importance.
        group_perm = group_imp.argsort(descending=True)  # [G]

        # Expand group permutation to column permutation.
        col_perm = torch.cat(
            [torch.arange(int(g) * VL, int(g) * VL + VL) for g in group_perm]
        )
        return col_perm

    def _compute_global_freq(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        NC: int,
        bits: int,
    ) -> torch.Tensor:
        """Return global centroid frequency vector ``[NC]`` normalised to sum=1."""
        from .transforms.vq_packing import IndexPacking

        counts = torch.zeros(NC, dtype=torch.float32)
        total = 0
        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vectors = w.shape[0] * (K // self.vector_len)
            indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).cpu()
            counts.scatter_add_(0, indices, torch.ones(len(indices)))
            total += len(indices)
        if total > 0:
            counts /= total
        return counts

    def _compute_layer_mean_L2(
        self,
        mod: nn.Module,
        packed: torch.Tensor,
        centroids: torch.Tensor,
        bits: int,
    ) -> float:
        """Return mean unweighted L2 distance from weight vectors to assigned centroids."""
        from .transforms.vq_packing import IndexPacking

        _NC, VL = centroids.shape
        dev = self._device or torch.device("cpu")
        w = mod.weight.data
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        n_vectors = w.shape[0] * (K // VL)
        indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).to(dev)
        vecs = w.float().to(dev).reshape(-1, VL)
        dists = (vecs - centroids.float().to(dev)[indices]).norm(dim=1)
        return dists.mean().item()

    def _compute_mean_global_l2(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        centroids: torch.Tensor,
        bits: int,
    ) -> float:
        """Return the vector-count-weighted mean L2 distance across all layers.

        Used to normalise the ECVQ log-frequency bias so that ``ecvq_beta``
        is dimensionless: ``beta=1.0`` means the maximum per-centroid bias
        equals the mean intra-cluster radius.
        """
        VL = centroids.shape[1]
        total_dist = 0.0
        total_n = 0
        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vecs = w.shape[0] * (K // VL)
            layer_mean = self._compute_layer_mean_L2(mod, packed, centroids, bits)
            total_dist += layer_mean * n_vecs
            total_n += n_vecs
        return total_dist / total_n if total_n > 0 else 1.0

    def _assign_layer(
        self,
        name: str,
        mod: nn.Module,
        centroids: torch.Tensor,
        registry: StatisticsRegistry,
        column_perm: torch.Tensor | None = None,
        temperature: float = 0.0,
        centroid_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Assign weight vectors to *centroids*, returning packed ``uint8``.

        Uses importance-weighted L2 when ``hessian_diag`` is available and
        :attr:`proxy_selector` is L2.  When *column_perm* is provided the
        weight columns are reordered before assignment (high-importance
        features first).

        Args:
            name: Module name for logging.
            mod: Target module.
            centroids: Shared codebook ``[num_centroids, vector_len]``.
            registry: Populated statistics registry.
            column_perm: Optional int64 column permutation ``[K]``.  When
                present the weight is permuted before vectorizing; the
                permutation is NOT applied here (it is stored in the
                ``VQScheme`` by the caller).
            temperature: Softmax temperature for the E-step.  ``0.0`` means
                hard nearest-centroid assignment.
            centroid_bias: Optional per-centroid additive bias ``[num_centroids]``
                applied to assignment distances, used by the ECVQ log-frequency
                term.  ``None`` leaves distances unbiased.

        Returns:
            Packed ``uint8`` index buffer, or ``None`` on incompatible weight.
        """
        from .transforms.vq_packing import IndexPacking

        w = mod.weight.data
        if w.ndim < 2:
            LOGGER.warning(
                "VPTQGroupEstimator: skip '%s' - weight is %dD.", name, w.ndim
            )
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            LOGGER.warning(
                "VPTQGroupEstimator: skip '%s' - K=%d not divisible by vector_len=%d.",
                name,
                K,
                self.vector_len,
            )
            return None

        bits = IndexPacking.bits_for(centroids.shape[0])
        dev = self._device or torch.device("cpu")

        # Load weight to device - but keep on CPU if the tensor alone would not
        # fit (e.g. embed_tokens_per_layer [262k x 8960] fp16 = 4.38 GiB).
        # importance_weighted_assign's lazy per-group path handles CPU weights
        # transparently: each [R, VL] slice is moved to GPU just-in-time.
        w_bytes = w.numel() * w.element_size()
        if dev.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(dev)
            w_on_cpu = w_bytes > free_bytes * 0.5
        else:
            w_on_cpu = False
        w2d = w.reshape(w.shape[0], K) if w_on_cpu else w.to(dev).reshape(w.shape[0], K)

        # Apply column permutation when provided.
        if column_perm is not None:
            perm = column_perm.cpu() if w_on_cpu else column_perm.to(dev)
            w2d = w2d[:, perm]

        diag_h = self._get_diag_h(name, K, registry)
        # Permute diag_h to match the reordered weight columns.
        if diag_h is not None and column_perm is not None:
            diag_h = diag_h.to(dev)[column_perm.to(dev)]

        # Annealing noise: add isotropic Gaussian perturbation to weights so
        # that near-boundary vectors can flip to neighbouring centroids.
        # Applied in unscaled weight space; high-curvature dims still dominate
        # the importance-weighted distance.  Noise decays to zero each iter.
        if temperature > 0.0:
            w2d = w2d + torch.randn_like(w2d) * temperature

        if diag_h is not None:
            from .methods.vptq import importance_weighted_assign

            indices = importance_weighted_assign(
                w2d,
                centroids.float().to(dev),
                diag_h.to(dev),
                self.vector_len,
                centroid_bias=centroid_bias,
            )
        else:
            # No hessian: plain L2, but still process group-by-group to bound
            # peak memory (full cdist on large vocab layers → OOM otherwise).
            from .methods.vptq import importance_weighted_assign

            diag_h_uniform = torch.ones(K, device=dev)
            indices = importance_weighted_assign(
                w2d,
                centroids.float().to(dev),
                diag_h_uniform,
                self.vector_len,
                centroid_bias=centroid_bias,
            )

        # For CPU-fallback weights (embed_tokens etc.), pack on CPU to avoid
        # allocating the full [n_vectors] int64 index tensor on GPU - at 587M
        # vectors that's 4.7 GiB, which OOMs when fp16 model is in VRAM.
        if w_on_cpu and indices.is_cuda:
            indices = indices.cpu()
        return IndexPacking.pack(indices, bits)

    def _compute_column_perm(
        self,
        name: str,
        mod: nn.Module,
        registry: StatisticsRegistry,
    ) -> torch.Tensor | None:
        """Compute a Hessian-guided column permutation for *mod*'s weight.

        Args:
            name: Module name for registry lookup.
            mod: Target module.
            registry: Populated statistics registry.

        Returns:
            Int64 permutation ``[K]`` or ``None`` when Hessian is unavailable.
        """
        w = mod.weight.data
        if w.ndim < 2:
            return None
        K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
        if K % self.vector_len != 0:
            return None
        diag_h = self._get_diag_h(name, K, registry)
        if diag_h is None:
            return None
        from .methods.vptq import hessian_sort_permutation

        return hessian_sort_permutation(diag_h, self.vector_len)

    def _weight_compute_device(
        self,
        w: torch.Tensor,
        dev: torch.device,
    ) -> torch.device:
        """Return GPU dev when w (fp32) fits; fall back to CPU otherwise."""
        if dev.type != "cuda":
            return dev
        w_fp32_bytes = w.numel() * 4
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        return torch.device("cpu") if w_fp32_bytes > free_bytes * 0.4 else dev

    def _compute_cluster_radii(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        centroids: torch.Tensor,
        perm_per_layer: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        """Compute mean L2 radius for each cluster.

        Args:
            members: ``(name, module)`` pairs in this group.
            packed_per_layer: Current packed index buffers keyed by layer name.
            centroids: Codebook ``[NC, VL]``.
            perm_per_layer: Per-layer column permutations.

        Returns:
            Float32 tensor ``[NC]`` of mean L2 distances per cluster.
        """
        from .transforms.vq_packing import IndexPacking

        NC, VL = centroids.shape
        bits = IndexPacking.bits_for(NC)
        dev = self._device or torch.device("cpu")
        c_f32 = centroids.float().to(dev)

        radius_sum = torch.zeros(NC, dtype=torch.float32, device=dev)
        radius_count = torch.zeros(NC, dtype=torch.float32, device=dev)

        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vectors = w.shape[0] * (K // VL)
            cdev = self._weight_compute_device(w, dev)
            c_cdev = c_f32.to(cdev)
            indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).to(cdev)
            w2d = w.to(cdev, dtype=torch.float32).reshape(w.shape[0], K)
            perm = perm_per_layer.get(name)
            if perm is not None:
                w2d = w2d[:, perm.to(cdev)]
            vecs = w2d.reshape(-1, VL)
            dists = (vecs - c_cdev[indices]).norm(dim=1)
            radius_sum.index_add_(0, indices.to(dev), dists.to(dev))
            radius_count.index_add_(
                0, indices.to(dev), torch.ones(n_vectors, device=dev)
            )

        mask = radius_count > 0
        radii = torch.zeros(NC, dtype=torch.float32, device=dev)
        radii[mask] = radius_sum[mask] / radius_count[mask]
        return radii

    def _refine_centroids(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        centroids: torch.Tensor,
        perm_per_layer: dict[str, torch.Tensor | None],
        registry: StatisticsRegistry | None = None,
        cluster_radii: torch.Tensor | None = None,
    ) -> float:
        """Update *centroids* in-place via one k-means M-step.

        For each centroid, collect all weight vectors (across layers) that are
        currently assigned to it and replace the centroid with their
        (optionally importance-weighted) mean.  Centroids with no assigned
        vectors are left unchanged.

        When coverage regularization is active (``coverage_beta > 0``),
        importance weights are blended toward uniform for vectors belonging
        to oversized clusters.

        Args:
            members: ``(name, module)`` pairs in this group.
            packed_per_layer: Current packed index buffers keyed by layer name.
            centroids: Float16 codebook to update in-place ``[NC, VL]``.
            perm_per_layer: Per-layer column permutations (may be ``None``).
            registry: Statistics registry for importance weights.  Only used
                when ``centroid_weighting="hessian"``.
            cluster_radii: Pre-computed mean L2 radius per cluster ``[NC]``.
                Used for coverage regularization.

        Returns:
            Tuple ``(frob_shift, mean_shift, max_shift, dead, mean_count, min_count,
            mean_cov)`` where *frob_shift* is the Frobenius norm of centroid
            displacement (used for convergence), *mean_shift* / *max_shift* are
            per-centroid L2 stats for the occupied centroids, *dead* is the number
            of empty clusters, *mean_count* / *min_count* are cluster-size stats,
            and *mean_cov* is the mean coverage-blend factor (1.0 when inactive).
        """
        import os
        import time as _time

        from .transforms.vq_packing import IndexPacking

        NC, VL = centroids.shape
        bits = IndexPacking.bits_for(NC)
        dev = self._device or torch.device("cpu")

        do_profile = (
            self._profile_next_mstep or os.environ.get("VPTQ_PROFILE_MSTEP") == "1"
        )
        if do_profile:
            self._profile_next_mstep = False
            _cuda = dev.type == "cuda"

            def _sync():
                if _cuda:
                    torch.cuda.synchronize(dev)

            _p_mem_info = 0.0  # _weight_compute_device (mem_get_info + sync wait)
            _p_unpack = 0.0  # IndexPacking.unpack
            _p_cast = 0.0  # w.to(dtype=float32)
            _p_perm = 0.0  # column permutation
            _p_iadd = 0.0  # index_add_ pair
            _p_other = 0.0  # Python overhead + dict lookups within layer loop
            _p_t_loop_start = _time.perf_counter()

        # Lazy-initialise or zero pre-allocated accumulators (avoids 2 large
        # allocations per M-step call).
        if self._m_step_acc is None or self._m_step_acc.shape != (NC, VL):
            self._m_step_acc = torch.zeros(NC, VL, dtype=torch.float32, device=dev)
        else:
            self._m_step_acc.zero_()
        if self._m_step_counts is None or self._m_step_counts.shape != (NC,):
            self._m_step_counts = torch.zeros(NC, dtype=torch.float32, device=dev)
        else:
            self._m_step_counts.zero_()
        acc = self._m_step_acc
        counts = self._m_step_counts

        # Populate importance weight cache on first call (Hessian never changes).
        if not self._cached_vector_weights:
            for n, m in members:
                perm = perm_per_layer.get(n)
                self._cached_vector_weights[n] = self._compute_layer_vector_weights(
                    n, m, registry, column_perm=perm
                )
            # Scale no-Hessian layers (embeddings) to the mean weight of
            # Hessian-equipped layers so they compete fairly for centroids.
            if self.centroid_weighting == "hessian" and registry is not None:
                no_h = {
                    n for n, m in members if not self._layer_has_hessian(n, m, registry)
                }
                self._scale_no_hessian_weights(self._cached_vector_weights, no_h)

        use_coverage = self.coverage_beta > 0 and cluster_radii is not None
        cov_sum = 0.0
        cov_n = 0
        _n_profiled = 0

        for name, mod in members:
            if do_profile:
                _sync()
                _t0 = _time.perf_counter()

            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vectors = w.shape[0] * (K // VL)

            if do_profile:
                _sync()
                _t1 = _time.perf_counter()

            cdev = self._weight_compute_device(w, dev)

            if do_profile:
                _sync()
                _t2 = _time.perf_counter()

            indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).to(cdev)

            if do_profile:
                _sync()
                _t3 = _time.perf_counter()

            w2d = w.to(cdev, dtype=torch.float32).reshape(w.shape[0], K)

            if do_profile:
                _sync()
                _t4 = _time.perf_counter()

            perm = perm_per_layer.get(name)
            if perm is not None:
                w2d = w2d[:, perm.to(cdev)]
            vecs = w2d.reshape(-1, VL)  # [n_vectors, VL]

            if do_profile:
                _sync()
                _t5 = _time.perf_counter()

            if self.centroid_weighting == "hessian" and registry is not None:
                wts = self._cached_vector_weights.get(name)
                if wts is None:
                    wts = torch.ones(n_vectors, dtype=torch.float32)
                wts = wts.to(cdev)
                if use_coverage:
                    r = cluster_radii.to(cdev)[indices]  # [n_vectors]
                    r_max = self.coverage_r_max
                    cov = torch.sigmoid(-self.coverage_beta * (r - r_max))
                    cov_sum += cov.sum().item()
                    cov_n += cov.numel()
                    wts = wts * cov + (1.0 - cov)
                acc.index_add_(0, indices.to(dev), (vecs * wts.unsqueeze(1)).to(dev))
                counts.index_add_(0, indices.to(dev), wts.to(dev))
            else:
                acc.index_add_(0, indices.to(dev), vecs.to(dev))
                counts.index_add_(0, indices.to(dev), torch.ones(n_vectors, device=dev))

            if do_profile:
                _sync()
                _t6 = _time.perf_counter()
                _p_other += _t1 - _t0  # Python overhead before mem_get_info
                _p_mem_info += _t2 - _t1
                _p_unpack += _t3 - _t2
                _p_cast += _t4 - _t3
                _p_perm += _t5 - _t4
                _p_iadd += _t6 - _t5
                _n_profiled += 1

        mask = counts > 0

        if do_profile:
            _sync()
            _t_post = _time.perf_counter()
            _total_loop = _t_post - _p_t_loop_start
            _p_finalize_start = _t_post

        new_centroids = centroids.float().to(dev).clone()
        new_centroids[mask] = acc[mask] / counts[mask].unsqueeze(1)

        per_c = (new_centroids - centroids.float().to(dev)).norm(dim=1)  # [NC]
        frob_shift = per_c.norm().item()
        occupied = per_c[mask]
        mean_shift = occupied.mean().item() if occupied.numel() > 0 else 0.0
        max_shift = occupied.max().item() if occupied.numel() > 0 else 0.0
        dead = int((~mask).sum().item())
        n_moving = (
            int((occupied > self.tol).sum().item()) if occupied.numel() > 0 else 0
        )
        occ_counts = counts[mask]
        mean_count = occ_counts.mean().item() if occ_counts.numel() > 0 else 0.0
        min_count = occ_counts.min().item() if occ_counts.numel() > 0 else 0.0
        std_count = occ_counts.std().item() if occ_counts.numel() > 1 else 0.0
        mean_cov = cov_sum / cov_n if cov_n > 0 else 1.0

        centroids.copy_(new_centroids.to(centroids.dtype))

        if do_profile:
            _sync()
            _p_finalize = _time.perf_counter() - _p_finalize_start
            _accounted = (
                _p_other + _p_mem_info + _p_unpack + _p_cast + _p_perm + _p_iadd
            )
            _unaccounted = _total_loop - _accounted - _p_finalize

            def _ms(x):
                return f"{x * 1e3:8.1f} ms"

            def _pct(x):
                return f"({x / _total_loop * 100:5.1f}%)"

            LOGGER.warning(
                "\n"
                "=== M-step profile  layers=%d  NC=%d  VL=%d ===\n"
                "  mem_get_info  %s  %s   (per layer: %.2f ms)\n"
                "  unpack        %s  %s   (per layer: %.2f ms)\n"
                "  weight cast   %s  %s   (per layer: %.2f ms)\n"
                "  perm+reshape  %s  %s   (per layer: %.2f ms)\n"
                "  index_add_    %s  %s   (per layer: %.2f ms)\n"
                "  python/other  %s  %s   (per layer: %.2f ms)\n"
                "  finalize      %s  %s\n"
                "  unaccounted   %s  %s\n"
                "  TOTAL LOOP    %s",
                _n_profiled,
                NC,
                VL,
                _ms(_p_mem_info),
                _pct(_p_mem_info),
                _p_mem_info / max(_n_profiled, 1) * 1e3,
                _ms(_p_unpack),
                _pct(_p_unpack),
                _p_unpack / max(_n_profiled, 1) * 1e3,
                _ms(_p_cast),
                _pct(_p_cast),
                _p_cast / max(_n_profiled, 1) * 1e3,
                _ms(_p_perm),
                _pct(_p_perm),
                _p_perm / max(_n_profiled, 1) * 1e3,
                _ms(_p_iadd),
                _pct(_p_iadd),
                _p_iadd / max(_n_profiled, 1) * 1e3,
                _ms(_p_other),
                _pct(_p_other),
                _p_other / max(_n_profiled, 1) * 1e3,
                _ms(_p_finalize),
                _pct(_p_finalize),
                _ms(_unaccounted),
                _pct(_unaccounted),
                _ms(_total_loop),
            )

        return (
            frob_shift,
            mean_shift,
            max_shift,
            dead,
            n_moving,
            mean_count,
            min_count,
            std_count,
            mean_cov,
        )

    def _log_centroid_frequency(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        centroids: torch.Tensor,
        bits: int,
    ) -> None:
        """Log global and per-layer centroid usage statistics.

        Computes a usage count histogram over all assigned indices then logs:
        - Global: dead count, unique used, top-5/bottom-5 counts, Shannon entropy.
        - Per layer: layer name, n_vectors, unique centroids used, coverage %,
          top-1 count, per-layer entropy.

        Args:
            members: ``(name, module)`` pairs in the group.
            packed_per_layer: Packed index buffers keyed by layer name.
            centroids: Float16 codebook ``[NC, VL]`` (read-only here).
            bits: Bits per index used for unpacking.
        """
        import math

        from .transforms.vq_packing import IndexPacking

        NC, VL = centroids.shape
        global_counts = torch.zeros(NC, dtype=torch.long)

        layer_stats: list[tuple[str, int, int, float, int, float]] = []

        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vectors = w.shape[0] * (K // VL)
            indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).cpu()
            layer_counts = torch.zeros(NC, dtype=torch.long)
            layer_counts.scatter_add_(
                0, indices, torch.ones(len(indices), dtype=torch.long)
            )
            global_counts.add_(layer_counts)

            unique_used = int((layer_counts > 0).sum().item())
            coverage_pct = 100.0 * unique_used / NC
            top1 = int(layer_counts.max().item())
            # Per-layer entropy
            p = layer_counts[layer_counts > 0].float() / n_vectors
            entropy = -float((p * p.log()).sum().item()) / math.log(2)
            layer_stats.append(
                (name, n_vectors, unique_used, coverage_pct, top1, entropy)
            )

        # Global stats
        total_vecs = int(global_counts.sum().item())
        unique_global = int((global_counts > 0).sum().item())
        dead_global = NC - unique_global
        sorted_counts, _ = global_counts.sort(descending=True)
        top5 = sorted_counts[:5].tolist()
        bot5 = (
            sorted_counts[unique_global - 5 : unique_global].tolist()
            if unique_global >= 5
            else []
        )
        p_global = global_counts[global_counts > 0].float() / total_vecs
        entropy_global = -float((p_global * p_global.log()).sum().item()) / math.log(2)
        max_entropy = math.log2(NC)

        LOGGER.info(
            "centroid_freq: global  NC=%d  total_vecs=%d  unique=%d  dead=%d  "
            "entropy=%.2f/%.2f bits (%.1f%%)  "
            "top5=%s  bot5=%s",
            NC,
            total_vecs,
            unique_global,
            dead_global,
            entropy_global,
            max_entropy,
            100.0 * entropy_global / max_entropy,
            top5,
            bot5,
        )

        # Per-layer breakdown (one line per layer)
        for name, n_vecs, unique, cov_pct, top1, ent in layer_stats:
            LOGGER.info(
                "centroid_freq: layer %-60s  n_vecs=%7d  unique=%5d  cov=%.1f%%  "
                "top1=%5d  entropy=%.2f bits",
                name,
                n_vecs,
                unique,
                cov_pct,
                top1,
                ent,
            )

    def _frequency_sort_centroids(
        self,
        members: list[tuple[str, nn.Module]],
        packed_per_layer: dict[str, torch.Tensor],
        centroids: torch.Tensor,
        bits: int,
    ) -> None:
        """Reorder *centroids* by descending assignment frequency; remap indices.

        After this call ``centroids[0]`` is the most-assigned centroid and all
        packed index buffers in *packed_per_layer* are updated to reflect the
        new ordering.  The centroids tensor is modified **in-place** so the
        shared reference held by all layers stays valid.

        This is a prerequisite for tiered/escape-code index schemes, which
        assume low-index centroids are the most frequent ones.

        Args:
            members: ``(name, module)`` pairs in the group.
            packed_per_layer: Packed index buffers keyed by layer name
                (modified in-place).
            centroids: Float16 codebook ``[NC, VL]`` (modified in-place).
            bits: Bits per index used for packing/unpacking.
        """
        from .transforms.vq_packing import IndexPacking

        NC = centroids.shape[0]
        VL = centroids.shape[1]
        counts = torch.zeros(NC, dtype=torch.long)

        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vectors = w.shape[0] * (K // VL)
            indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).cpu()
            counts.scatter_add_(0, indices, torch.ones(len(indices), dtype=torch.long))

        sort_order = torch.argsort(counts, descending=True)  # [NC]
        # Early exit if already sorted.
        if torch.equal(sort_order, torch.arange(NC)):
            return

        inv_sort = torch.argsort(sort_order)  # inverse permutation [NC]
        centroids.copy_(centroids[sort_order].clone())

        for name, mod in members:
            packed = packed_per_layer.get(name)
            if packed is None:
                continue
            w = mod.weight.data
            K = int(w.shape[1]) if w.ndim == 2 else int(w.numel() // w.shape[0])
            n_vectors = w.shape[0] * (K // VL)
            indices = IndexPacking.unpack(packed, bits, n_indices=n_vectors).cpu()
            packed_per_layer[name] = IndexPacking.pack(inv_sort[indices], bits)

        LOGGER.info(
            "VPTQGroupEstimator: frequency-sorted %d centroids "
            "(top-1 count=%d, bottom-1 count=%d)",
            NC,
            counts[sort_order[0]].item(),
            counts[sort_order[-1]].item(),
        )

    # ------------------------------------------------------------------
    # QuantizationParamEstimator contract
    # ------------------------------------------------------------------

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Train shared centroids then assign each layer to produce a plan.

        **Step 1** - group target modules by :attr:`grouping` strategy.

        **Step 2** - pool all weight vectors in each group and run k-means
        (via :attr:`cluster_backend`) to train shared centroids.

        **Step 3** - for each layer, assign its weight vectors to centroids
        (optionally importance-weighted) and wrap the result as a
        :class:`~neuropress.transforms.vq.QTensorTractVQ` backed by the
        shared centroids tensor.

        Args:
            graph: Unused (targets fixed by :meth:`prepare`).
            registry: Populated statistics registry.
            recipe: Unused.

        Returns:
            A :class:`.QuantizationPlan` mapping each target layer name to
            its :class:`~neuropress.transforms.vq.QTensorTractVQ`.
        """
        import time

        from .transforms.vq import QTensorTractVQ, VQScheme
        from .transforms.vq_packing import IndexPacking

        assignments: dict[str, Any] = {}
        cache = self._cache
        spans = self._spans

        groups = self._make_groups()
        n_groups = len(groups)
        for g_idx, (group_key, members) in enumerate(groups.items(), 1):
            LOGGER.info(
                "VPTQGroupEstimator: [group %d/%d] '%s'  %d layers - "
                "training centroids",
                g_idx,
                n_groups,
                group_key,
                len(members),
            )
            t_group = time.perf_counter()

            # Load centroids from cache when available; otherwise train and save.
            cached_c = cache.load("centroids", group_key)
            expected_nc = self.centroids_for_group(group_key)
            if (
                cached_c is not None
                and "centroids" in cached_c
                and not self.cached_centroids_usable(
                    cached_c["centroids"], expected_nc=expected_nc
                )
            ):
                LOGGER.warning(
                    "VPTQGroupEstimator: discarding cached centroids for '%s' "
                    "- shape %s does not match vector_len=%d / num_centroids=%d; "
                    "retraining.",
                    group_key,
                    tuple(cached_c["centroids"].shape),
                    self.vector_len,
                    expected_nc,
                )
                cached_c = None
            if cached_c is not None and "centroids" in cached_c:
                dev = self._device or torch.device("cpu")
                centroids = cached_c["centroids"].to(dev)
                LOGGER.info(
                    "VPTQGroupEstimator: loaded centroids for '%s' from cache.",
                    group_key,
                )
            else:
                with timed_span(spans, "vptq.centroids", group=group_key):
                    centroids = self._train_centroids(
                        members, registry=registry, group_key=group_key
                    )
                if centroids is not None:
                    cache.save("centroids", group_key, {"centroids": centroids})

            if centroids is None:
                LOGGER.warning(
                    "VPTQGroupEstimator: no compatible layers in group "
                    "'%s' - skipping.",
                    group_key,
                )
                continue

            bits = IndexPacking.bits_for(centroids.shape[0])
            dev = self._device or torch.device("cpu")

            # Pre-compute per-layer column permutations (fixed for the whole
            # refinement loop - derived from Hessian, not from centroids).
            # Move to dev immediately so VQScheme always holds same-device tensors.
            _perm_dev = self._device or torch.device("cpu")
            perm_per_layer: dict[str, torch.Tensor | None] = {
                name: (
                    p.to(_perm_dev)
                    if (p := self._compute_column_perm(name, mod, registry)) is not None
                    else None
                )
                for name, mod in members
            }

            # Pass 1 - initial assignment (with per-layer cache).
            packed_per_layer: dict[str, torch.Tensor] = {}
            n_members = len(members)
            prog_ctx = _layer_progress(n_members) if _RICH else contextlib.nullcontext()
            with prog_ctx as progress:
                task_id = (
                    progress.add_task(f"assign  {group_key}", total=n_members)
                    if _RICH
                    else None
                )
                for l_idx, (name, mod) in enumerate(members, 1):
                    LOGGER.info(
                        "VPTQGroupEstimator: [%d/%d] assigning '%s'  shape=%s",
                        l_idx,
                        n_members,
                        name,
                        tuple(mod.weight.shape),
                    )
                    cached_a = cache.load("assignments", name)
                    if cached_a is not None and "packed" in cached_a:
                        _mw = mod.weight
                        _exp = _mw.shape[0] * (int(_mw.shape[1]) // self.vector_len)
                        if not self.cached_assignment_usable(
                            cached_a["packed"], _exp, bits
                        ):
                            LOGGER.warning(
                                "VPTQGroupEstimator: discarding cached assignment "
                                "for '%s' - blob %d bytes, expected %d for %d "
                                "indices at %d bits; reassigning.",
                                name,
                                cached_a["packed"].numel(),
                                (_exp * bits + 7) // 8,
                                _exp,
                                bits,
                            )
                            cached_a = None
                    if cached_a is not None and "packed" in cached_a:
                        packed = cached_a["packed"].to(
                            self._device or torch.device("cpu")
                        )
                        LOGGER.info(
                            "VPTQGroupEstimator: [%d/%d] '%s' loaded from cache",
                            l_idx,
                            n_members,
                            name,
                        )
                    else:
                        t_assign = time.perf_counter()
                        with timed_span(spans, "vptq.assign", layer=name):
                            packed = self._assign_layer(
                                name,
                                mod,
                                centroids,
                                registry,
                                column_perm=perm_per_layer.get(name),
                            )
                        LOGGER.info(
                            "VPTQGroupEstimator: [%d/%d] '%s' assigned in %.2fs",
                            l_idx,
                            n_members,
                            name,
                            time.perf_counter() - t_assign,
                        )
                        if packed is not None:
                            cache.save("assignments", name, {"packed": packed})
                    if packed is not None:
                        packed_per_layer[name] = packed
                    if _RICH:
                        progress.update(
                            task_id,
                            description=f"assign  [dim]{_re(name)}[/dim]",
                        )
                        progress.advance(task_id)

            # Iterative refinement - Lloyd's M+E steps on the shared codebook.
            use_coverage = (
                self.centroid_weighting == "hessian" and self.coverage_beta > 0
            )
            effective_r_max = self.coverage_r_max

            # Temperature annealing: compute T₀ from mean intra-cluster radius
            # after the initial assignment, then decay geometrically each iter.
            anneal_T0: float = 0.0
            if self.temperature_annealing:
                init_radii = self._compute_cluster_radii(
                    members,
                    packed_per_layer,
                    centroids,
                    perm_per_layer,
                )
                non_zero = init_radii[init_radii > 0]
                anneal_T0 = non_zero.mean().item() if non_zero.numel() > 0 else 0.0
                LOGGER.info(
                    "VPTQGroupEstimator: temperature annealing T0=%.4e  "
                    "decay=%.2f  (mean initial cluster radius)",
                    anneal_T0,
                    self.temperature_decay,
                )

            import os as _os

            _profile_env = _os.environ.get("VPTQ_PROFILE_MSTEP") == "1"

            # Cached once at ecvq_warmup_iters for in-loop ECVQ normalization.
            ecvq_mean_l2: float | None = None

            for iteration in range(self.max_n_iter):
                t_iter = time.perf_counter()

                # Auto-enable M-step profile on iteration 1 (iter 0 = cold cache).
                if iteration == 1 and _profile_env:
                    self._profile_next_mstep = True

                # Compute cluster radii for coverage regularization.
                cluster_radii = None
                if use_coverage:
                    cluster_radii = self._compute_cluster_radii(
                        members,
                        packed_per_layer,
                        centroids,
                        perm_per_layer,
                    )
                    # Auto-tune r_max from median radius on first iteration.
                    if effective_r_max is None:
                        non_zero = cluster_radii[cluster_radii > 0]
                        if non_zero.numel() > 0:
                            effective_r_max = non_zero.median().item()
                        else:
                            effective_r_max = 1.0
                        LOGGER.info(
                            "VPTQGroupEstimator: auto-tuned coverage_r_max="
                            "%.4f (median cluster radius)",
                            effective_r_max,
                        )
                    # Temporarily set r_max so _refine_centroids can read it.
                    self.coverage_r_max = effective_r_max
                    occupied = (cluster_radii > 0).sum().item()
                    LOGGER.info(
                        "VPTQGroupEstimator: cluster radii - "
                        "median=%.4f  mean=%.4f  max=%.4f  "
                        "occupied=%d/%d  r_max=%.4f  beta=%.1f",
                        cluster_radii[cluster_radii > 0].median().item()
                        if occupied > 0
                        else 0.0,
                        cluster_radii[cluster_radii > 0].mean().item()
                        if occupied > 0
                        else 0.0,
                        cluster_radii.max().item(),
                        occupied,
                        centroids.shape[0],
                        effective_r_max,
                        self.coverage_beta,
                    )

                with timed_span(
                    spans, "vptq.mstep", iteration=iteration, group=group_key
                ):
                    (
                        shift,
                        mean_shift,
                        max_shift,
                        dead,
                        n_moving,
                        mean_count,
                        min_count,
                        std_count,
                        mean_cov,
                    ) = self._refine_centroids(
                        members,
                        packed_per_layer,
                        centroids,
                        perm_per_layer,
                        registry=registry,
                        cluster_radii=cluster_radii,
                    )
                ratio = shift / self.tol if self.tol > 0 else float("inf")
                NC = centroids.shape[0]
                occupied = NC - dead
                anneal_T = anneal_T0 * (self.temperature_decay**iteration)
                LOGGER.info(
                    "VPTQGroupEstimator: group '%s' iter %d/%d  "
                    "shift(frob)=%.4e  mean=%.4e  max=%.4e  "
                    "tol=%.2e  ratio=%.1fx  moving=%d/%d  "
                    "dead=%d  cluster(mean=%.0f min=%.0f std=%.0f)  "
                    "cov_blend=%.3f  T=%.2e  %.1fs",
                    group_key,
                    iteration + 1,
                    self.max_n_iter,
                    shift,
                    mean_shift,
                    max_shift,
                    self.tol,
                    ratio,
                    n_moving,
                    occupied,
                    dead,
                    mean_count,
                    min_count,
                    std_count,
                    mean_cov,
                    anneal_T,
                    time.perf_counter() - t_iter,
                )
                if shift < self.tol:
                    LOGGER.info(
                        "VPTQGroupEstimator: group '%s' converged at iter %d "
                        "(shift=%.2e < tol=%.2e).",
                        group_key,
                        iteration + 1,
                        shift,
                        self.tol,
                    )
                    break
                # Re-assign with updated centroids.
                # Two centroid-bias modes (mutually exclusive, ECVQ takes priority):
                #  1. ECVQ in-loop: after warmup iterations, bias by
                #     β·(mean_l2/log(NC))·log(freq) to encourage non-uniform usage.
                #  2. concentration_beta: legacy linear-freq bias (when ecvq off).
                NC = centroids.shape[0]
                global_freq: torch.Tensor | None = None
                ecvq_loop_bias: torch.Tensor | None = None
                if self.ecvq_beta_in_loop > 0 and iteration >= self.ecvq_warmup_iters:
                    global_freq = self._compute_global_freq(
                        members,
                        packed_per_layer,
                        NC,
                        bits,
                    )
                    if ecvq_mean_l2 is None:
                        ecvq_mean_l2 = self._compute_mean_global_l2(
                            members, packed_per_layer, centroids, bits
                        )
                        LOGGER.info(
                            "VPTQGroupEstimator: ECVQ in-loop active from iter %d  "
                            "beta=%.4f  mean_l2=%.4e  log(NC)=%.2f",
                            iteration,
                            self.ecvq_beta_in_loop,
                            ecvq_mean_l2,
                            math.log(NC),
                        )
                    ecvq_loop_bias = (
                        (self.ecvq_beta_in_loop * ecvq_mean_l2 / math.log(NC))
                        * (global_freq + 1e-10).log()
                    ).to(self._device or torch.device("cpu"))
                    p_nz = global_freq[global_freq > 0]
                    entropy = -(p_nz * p_nz.log2()).sum().item()
                    LOGGER.info(
                        "VPTQGroupEstimator: ECVQ in-loop iter %d  entropy=%.4f/%.1f bits",
                        iteration,
                        entropy,
                        math.log2(NC),
                    )
                elif self.concentration_beta > 0 and packed_per_layer:
                    global_freq = self._compute_global_freq(
                        members,
                        packed_per_layer,
                        NC,
                        bits,
                    )
                _e_t0 = time.perf_counter()
                _e_times: list[float] = []
                with timed_span(
                    spans, "vptq.estep", iteration=iteration, group=group_key
                ):
                    for name, mod in members:
                        if name not in packed_per_layer:
                            continue
                        centroid_bias: torch.Tensor | None = None
                        if ecvq_loop_bias is not None:
                            centroid_bias = ecvq_loop_bias
                        elif global_freq is not None:
                            mean_l2 = self._compute_layer_mean_L2(
                                mod,
                                packed_per_layer[name],
                                centroids,
                                bits,
                            )
                            centroid_bias = (
                                self.concentration_beta * mean_l2 * global_freq
                            ).to(self._device or torch.device("cpu"))
                        _le0 = time.perf_counter()
                        new_packed = self._assign_layer(
                            name,
                            mod,
                            centroids,
                            registry,
                            column_perm=perm_per_layer.get(name),
                            temperature=anneal_T,
                            centroid_bias=centroid_bias,
                        )
                        if new_packed is not None:
                            packed_per_layer[name] = new_packed
                        if _profile_env and iteration == 1:
                            if self._device and self._device.type == "cuda":
                                torch.cuda.synchronize(self._device)
                            _e_times.append(time.perf_counter() - _le0)
                if _profile_env and iteration == 1 and _e_times:
                    _e_total = time.perf_counter() - _e_t0
                    _e_times_s = sorted(_e_times, reverse=True)
                    LOGGER.warning(
                        "\n=== E-step profile  layers=%d ===\n"
                        "  total: %.1f s   per-layer avg: %.1f ms\n"
                        "  top-5 slowest layers: %s ms",
                        len(_e_times),
                        _e_total,
                        sum(_e_times) / len(_e_times) * 1e3,
                        "  ".join(f"{t * 1e3:.1f}" for t in _e_times_s[:5]),
                    )

            # ECVQ post-convergence annealing.
            # After Lloyd's converges, gradually introduce an entropy penalty
            # β·log(p_c) into the E-step assignment.  This biases popular
            # centroids to attract even more vectors (rich-get-richer), reducing
            # assignment entropy and making the index stream more compressible.
            # Using log(freq) matches the ECVQ rate-distortion objective
            # D + λ·H exactly.  Starting from a converged solution avoids the
            # limit-cycle instability seen when the penalty is applied from iter 0.
            if self.ecvq_beta_schedule:
                NC = centroids.shape[0]
                # Compute mean intra-cluster L2 once to normalize β.
                # With this scale: β=1.0 means the maximum log-freq bonus
                # (between the most- and least-popular centroid) equals the
                # mean intra-cluster radius.  β should then live in [0.05, 3.0].
                mean_global_l2 = self._compute_mean_global_l2(
                    members, packed_per_layer, centroids, bits
                )
                log_nc = math.log(NC)
                LOGGER.info(
                    "VPTQGroupEstimator: starting ECVQ annealing  "
                    "schedule=%s  iters_per_step=%d  mean_l2=%.4e  log(NC)=%.2f",
                    self.ecvq_beta_schedule,
                    self.ecvq_iters_per_step,
                    mean_global_l2,
                    log_nc,
                )
                for ecvq_beta in self.ecvq_beta_schedule:
                    for ecvq_iter in range(self.ecvq_iters_per_step):
                        global_freq = self._compute_global_freq(
                            members, packed_per_layer, NC, bits
                        )
                        # Normalized bias: β·(mean_l2/log(NC))·log(p_c + ε)
                        # At uniform freq all biases are equal (no effect on rank).
                        # The differential bonus between a 2x popular centroid and
                        # average is β·mean_l2·log(2)/log(NC) ≈ 0.067·β·mean_l2,
                        # so β=1.0 causes a nudge of ~7% of the intra-cluster radius
                        # per doubling of frequency.
                        log_freq_bias = (
                            (ecvq_beta * mean_global_l2 / log_nc)
                            * (global_freq + 1e-10).log()
                        ).to(centroids.device)
                        for name, mod in members:
                            if name not in packed_per_layer:
                                continue
                            new_packed = self._assign_layer(
                                name,
                                mod,
                                centroids,
                                registry,
                                column_perm=perm_per_layer.get(name),
                                centroid_bias=log_freq_bias,
                            )
                            if new_packed is not None:
                                packed_per_layer[name] = new_packed
                        (
                            shift,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                            _,
                        ) = self._refine_centroids(
                            members,
                            packed_per_layer,
                            centroids,
                            perm_per_layer,
                            registry=registry,
                        )
                        freq_after = self._compute_global_freq(
                            members, packed_per_layer, NC, bits
                        )
                        p_nz = freq_after[freq_after > 0]
                        entropy = -(p_nz * p_nz.log2()).sum().item()
                        LOGGER.info(
                            "VPTQGroupEstimator: ECVQ beta=%.3f  iter %d/%d  "
                            "shift=%.4e  entropy=%.4f/%.1f bits",
                            ecvq_beta,
                            ecvq_iter + 1,
                            self.ecvq_iters_per_step,
                            shift,
                            entropy,
                            math.log2(NC),
                        )

            # Frequency-sort: reorder centroids so index 0 = most-assigned.
            # Prerequisite for tiered/escape-code schemes and a free win for
            # the current fixed-width packing (concentrates hot indices in the
            # 4-bit tier when num_centroids <= 256).
            self._frequency_sort_centroids(members, packed_per_layer, centroids, bits)

            # Post-convergence GPTQ correction.
            #
            # gptq_correction_iters controls both whether and how many passes:
            #   0  - disabled (plain Lloyd's only)
            #   1  - one-shot: single GPTQ-corrected E-step on frozen centroids
            #   N>1 - VPTQ-like: N x (GPTQ E-step + M-step), re-sort at end
            #          (original VPTQ paper uses ~10 outer iterations)
            if self.gptq_correction_iters:
                # Compute sort order once - Hessian diagonals don't change between iters.
                _members_for_correction = (
                    self._gho_sort_members(members, registry)
                    if self.correction_mode == CorrectionMode.GHO_LAYER
                    else members
                )
                for vptq_iter in range(self.gptq_correction_iters):
                    _centroid_imp: torch.Tensor | None = None
                    if self.correction_mode == CorrectionMode.CENTROID_CENTRIC:
                        _centroid_imp = self._compute_centroid_importance(
                            members,
                            packed_per_layer,
                            centroids.shape[0],
                            bits,
                            registry,
                        )
                    n_corrected = 0
                    for name, mod in _members_for_correction:
                        if name not in packed_per_layer:
                            continue
                        if _centroid_imp is not None:
                            col_perm = self._centric_group_perm(
                                name,
                                mod,
                                packed_per_layer[name],
                                bits,
                                _centroid_imp,
                            )
                            if col_perm is None:
                                LOGGER.debug(
                                    "VPTQGroupEstimator: CCC skipping perm for '%s' "
                                    "(ineligible weight shape).",
                                    name,
                                )
                        else:
                            col_perm = perm_per_layer.get(name)
                        corrected = self._gptq_corrected_assign(
                            name,
                            mod,
                            centroids,
                            registry,
                            column_perm=col_perm,
                        )
                        if corrected is not None:
                            packed_per_layer[name] = corrected
                            n_corrected += 1
                    LOGGER.info(
                        "VPTQGroupEstimator: group '%s' GPTQ iter %d/%d "
                        "- corrected %d/%d layers (mode=%s).",
                        group_key,
                        vptq_iter + 1,
                        self.gptq_correction_iters,
                        n_corrected,
                        len(members),
                        self.correction_mode,
                    )
                    # M-step: re-estimate centroids from corrected assignments.
                    # Skip on the final iteration - the last E-step IS the output.
                    if vptq_iter < self.gptq_correction_iters - 1:
                        self._refine_centroids(
                            members,
                            packed_per_layer,
                            centroids,
                            perm_per_layer,
                            registry=registry,
                        )
                # Re-sort after any M-steps may have shifted the distribution.
                if self.gptq_correction_iters > 1:
                    self._frequency_sort_centroids(
                        members, packed_per_layer, centroids, bits
                    )

            self._log_centroid_frequency(members, packed_per_layer, centroids, bits)
            LOGGER.info(
                "VPTQGroupEstimator: group '%s' done in %.1fs total",
                group_key,
                time.perf_counter() - t_group,
            )

            # Build QTensorTractVQ from final assignments.
            # from_packed bypasses the redundant quantize_as_u8 cdist call:
            # _assign_layer already computed the correct indices; re-running
            # cdist on centroids[indices] would just re-derive identical results
            # at the cost of a huge [N, NC] temporary allocation.
            for name, mod in members:
                packed = packed_per_layer.get(name)
                if packed is None:
                    continue

                n_vectors = int(mod.weight.numel()) // self.vector_len
                original_shape = mod.weight.shape
                column_perm = perm_per_layer.get(name)
                target_dtype = mod.weight.dtype

                indices_flat = IndexPacking.unpack(
                    packed, bits, n_indices=n_vectors
                ).to(dev)

                vq = VQScheme(
                    centroids=centroids,  # shared tensor reference
                    proxy_selector=self.proxy_selector,
                    original_shape=original_shape,
                    column_permutation=column_perm,
                    use_tiered=self.use_tiered,
                    zero_escape=self.zero_escape,
                )
                assignments[name] = QTensorTractVQ.from_packed(
                    indices_flat, vq, dequant_to_dtype=target_dtype
                )

        return QuantizationPlan(weight_assignments=assignments)


# ---------------------------------------------------------------------------
# CompositeEstimator
# ---------------------------------------------------------------------------


@dataclass
class CompositeEstimator(QuantizationParamEstimator):
    """Applies multiple estimators to different parts of the model.

    Each sub-estimator targets its own set of module classes / filter
    functions.  ``prepare``, ``declare_statistics``, and ``estimate`` are
    delegated to every sub-estimator in order.  Statistic requests are
    deduplicated so duplicate Hessian hooks are not installed.

    Attributes:
        estimators: Ordered list of sub-estimators to apply.
    """

    estimators: list[QuantizationParamEstimator] = field(default_factory=list)

    def prepare(self, model: nn.Module) -> None:
        """Prepare all sub-estimators.

        Args:
            model: The model passed to the pipeline.
        """
        for est in self.estimators:
            if hasattr(est, "prepare"):
                est.prepare(model)

    def declare_statistics(self) -> list[StatisticRequest]:
        """Return deduplicated union of all sub-estimator statistic requests."""
        seen: set[tuple[str, str]] = set()
        requests: list[StatisticRequest] = []
        for est in self.estimators:
            for req in est.declare_statistics():
                key = (req.target_key, req.stat_type)
                if key not in seen:
                    seen.add(key)
                    requests.append(req)
        return requests

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Run all sub-estimators and merge their plans.

        Args:
            graph: QGraph built by the pipeline.
            registry: Populated statistics from the calibration pass.
            recipe: Active recipe.

        Returns:
            Merged :class:`.QuantizationPlan` combining all sub-estimator
            assignments (later estimators win on key collision).
        """
        merged = QuantizationPlan()
        for est in self.estimators:
            plan = est.estimate(graph, registry, recipe)
            merged.weight_assignments.update(plan.weight_assignments)
            merged.activation_assignments.update(plan.activation_assignments)
        return merged


# ---------------------------------------------------------------------------
# _hqq_layer - HQQ per-layer helper
# ---------------------------------------------------------------------------


def _hqq_layer(
    name: str,
    mod: nn.Module,
    group_size: int,
    nbits: int,
    symmetric: bool,
    num_iter: int,
    beta: float,
    kappa: float,
    lp_norm: float,
) -> None:
    """Apply HQQ to *mod*, updating its weight in-place to dequantized floats.

    Granularity is per-group when ``in_features`` is divisible by *group_size*,
    otherwise per-channel.

    Args:
        name: Human-readable layer name for logging.
        mod: Module whose weight is quantized in-place.
        group_size: Preferred group size for per-group granularity.
        nbits: Quantization bit-width.
        symmetric: Whether to use symmetric quantization.
        num_iter: Number of HQQ optimization iterations.
        beta: Initial penalty strength for half-quadratic splitting.
        kappa: Penalty schedule multiplier per iteration.
        lp_norm: Lp exponent for the proximal operator.
    """
    w = mod.weight
    assert isinstance(w, torch.Tensor)

    in_features = int(w.shape[1]) if w.dim() > 1 else w.shape[0]
    if in_features >= group_size and in_features % group_size == 0:
        gran = Granularity.per_group([group_size])
    else:
        gran = Granularity.per_channel()

    calib = HQQCalibration(
        granularity=gran,
        nbits=nbits,
        symmetric=symmetric,
        num_iter=num_iter,
        beta=beta,
        kappa=kappa,
        lp_norm=lp_norm,
    )
    calib.weight_tensor = w
    calib.module_class = type(mod)
    calib.module = mod
    calib.name = name
    calib.init()
    calib.update(w.data)
    qscheme = calib.compute_qscheme()

    # Dequantize and copy result back in-place.
    with torch.no_grad():
        w_float = w.data.float().reshape(calib.calibration_data_shape)
        scale = qscheme.scale
        zp = qscheme.zero_point.float()
        q = (w_float / scale + zp).clamp(0, qscheme.maxq).round()
        dq = (q - zp) * scale
        w.data.copy_(dq.reshape(w.shape).to(w.dtype))


# ---------------------------------------------------------------------------
# HQQEstimator
# ---------------------------------------------------------------------------


@dataclass
class HQQEstimator(QuantizationParamEstimator):
    """HQQ estimator: no calibration data required.

    Applies :class:`~.methods.hqq.HQQCalibration` directly to each target
    module's weight tensor and updates it in-place to the dequantized float
    representation.  Because HQQ is purely weight-driven, no forward pass
    is needed and :meth:`declare_statistics` always returns an empty list.

    Useful standalone or as one component of a
    :class:`CompositeEstimator` alongside a Hessian-based estimator.

    Attributes:
        module_classes: Only modules that are instances of these classes are
            targeted.  Defaults to ``(nn.Linear,)``.
        filter_fn: Optional extra filter ``(name, module) -> bool``.
        weight_transform: Optional callable ``(weight, name) -> value | None``
            applied to the HQQ-corrected weight after each layer.  The return
            value is stored in :attr:`.QuantizationPlan.weight_assignments`.
            Return *None* to skip the layer from the plan.
        group_size: Preferred group size for per-group granularity.  Layers
            whose ``in_features`` is not divisible by *group_size* fall back
            to per-channel granularity.
        nbits: Quantization bit-width.
        symmetric: Whether to use symmetric quantization.
        num_iter: Number of HQQ optimization iterations.  ``0`` = min-max.
        beta: Initial half-quadratic penalty strength.
        kappa: Penalty schedule multiplier (``beta *= kappa`` per iteration).
        lp_norm: Lp exponent for the proximal operator.
    """

    module_classes: tuple[type[nn.Module], ...] = field(default=(nn.Linear,))
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    weight_transform: Callable[[torch.Tensor, str], Any] | None = None
    group_size: int = 64
    nbits: int = 4
    symmetric: bool = False
    num_iter: int = 20
    beta: float = 1e4
    kappa: float = 1.01
    lp_norm: float = 0.5

    _target_modules: dict[str, nn.Module] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def prepare(self, model: nn.Module) -> None:
        """Discover target modules in *model*.

        Args:
            model: The model that will be passed to the pipeline.
        """
        self._target_modules = collect_target_modules(
            model, self.module_classes, self.filter_fn
        )
        LOGGER.info("HQQEstimator: %d target modules", len(self._target_modules))

    def declare_statistics(self) -> list[StatisticRequest]:
        """Return an empty list - HQQ operates directly on weights."""
        return []

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Run HQQ per layer and return a :class:`.QuantizationPlan`.

        Weights are updated in-place to their dequantized float
        representations.  If *weight_transform* is set, its output is stored
        in the plan's ``weight_assignments``.

        Args:
            graph: Unused (targets are fixed by :meth:`prepare`).
            registry: Unused (no statistics needed).
            recipe: Unused.

        Returns:
            A :class:`.QuantizationPlan` populated by *weight_transform* for
            each successfully processed layer, or an empty plan when
            *weight_transform* is ``None``.
        """
        ids_seen: set[int] = set()
        assignments: dict[str, Any] = {}

        for name, mod in self._target_modules.items():
            raw_weight = mod.weight
            assert isinstance(raw_weight, torch.Tensor)
            weight_id = id(raw_weight)
            if weight_id in ids_seen:
                LOGGER.info("Shared weight '%s' already processed - skipping", name)
                continue
            ids_seen.add(weight_id)

            LOGGER.info("HQQ quantizing: %s  shape=%s", name, tuple(raw_weight.shape))

            try:
                _hqq_layer(
                    name,
                    mod,
                    group_size=self.group_size,
                    nbits=self.nbits,
                    symmetric=self.symmetric,
                    num_iter=self.num_iter,
                    beta=self.beta,
                    kappa=self.kappa,
                    lp_norm=self.lp_norm,
                )
            except Exception as exc:
                LOGGER.error("HQQ failed for '%s': %s - skipping", name, exc)
                continue

            if self.weight_transform is not None:
                if mod.weight.dim() < 2:
                    LOGGER.warning(
                        "Skipping weight_transform for '%s': weight is %dD (shape %s)",
                        name,
                        mod.weight.dim(),
                        tuple(mod.weight.shape),
                    )
                else:
                    q = self.weight_transform(mod.weight, name)
                    if q is not None:
                        assignments[name] = q

        return QuantizationPlan(weight_assignments=assignments)
