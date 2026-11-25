"""Trellis estimator: research prototype, registered as estimator type ``trellis``.

Quantizes every target linear with :func:`~neuropress.methods.trellis.trellis_quantize`
against its full input Hessian. The plan holds :class:`~neuropress.transforms.trellis.QTensorTrellis`
weights (packed, saved packed) or, with ``store="dense"``, their dequantized weights. Bits per
weight can differ per type (leaf name) or per module (full name), e.g. ``{"q_proj": 3.5}``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal

import torch
from pydantic import Field
from torch import nn

from .calibration.batches import materialize
from .calibration.probe import StatisticRequest, StatisticsRegistry
from .estimator import QuantizationParamEstimator, UnpreparedTargets
from .estimators import (
    _add_tied_aliases,
    collect_target_modules,
    collect_tied_aliases,
    target_order,
)
from .graph.graph import QGraph
from .methods.trellis import TrellisConfig, trellis_quantize
from .plan import QuantizationPlan
from .quantization.cache import CacheStageDescriptor
from .quantization.spans import timed_span
from .recipe import QuantizationRecipe
from .recipe_config import EstimatorConfig, estimator_config
from .transforms.trellis import QTensorTrellis

LOGGER = logging.getLogger(__name__)


@dataclass
class TrellisEstimator(QuantizationParamEstimator):
    """Trellis-code linear weights against their input Hessians."""

    requires_graph: ClassVar[bool] = False

    module_classes: tuple[type[nn.Module], ...] = field(default=(nn.Linear,))
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    config: TrellisConfig = field(default_factory=TrellisConfig)
    bits_per_group: dict[str, float] = field(default_factory=dict)
    store: Literal["packed", "dense"] = "packed"

    _target_modules: dict[str, nn.Module] = field(
        default_factory=UnpreparedTargets, init=False, repr=False
    )
    _tied_aliases: dict[str, list[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _seed_offsets: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for key, bits in self.bits_per_group.items():
            try:
                TrellisConfig(
                    bits=bits, codebook=self.config.codebook, lut=self.config.lut
                )
            except ValueError as exc:
                raise ValueError(f"bits_per_group[{key!r}]: {exc}") from exc

    def bits_for(self, name: str) -> float:
        """Bits for module *name*: its full name, then its leaf name, then the default."""
        if name in self.bits_per_group:
            return self.bits_per_group[name]
        return self.bits_per_group.get(name.rsplit(".", 1)[-1], self.config.bits)

    def prepare(self, model: nn.Module) -> None:
        """Collect the target linears and their tied aliases."""
        self._target_modules = collect_target_modules(
            model, self.module_classes, self.filter_fn
        )
        self._tied_aliases = collect_tied_aliases(
            model, self.module_classes, self.filter_fn
        )
        # Seed offsets by position among all linears of the whole model, not
        # among the selected targets or the block at hand: a staged run
        # (select_targets) or a block run would otherwise restart them. A
        # block reads the model as it was before any block was adapted.
        self._seed_offsets = {
            name: i for i, name in enumerate(target_order(model, self.module_classes))
        }
        LOGGER.info("TrellisEstimator: %d target modules", len(self._target_modules))

    def cache_stages(self) -> list[CacheStageDescriptor]:
        """Declare the Hessian cache stage."""
        return [
            CacheStageDescriptor(
                "hessian", "Full input Hessians (one per layer)", scope="calibration"
            )
        ]

    def declare_statistics(self) -> list[StatisticRequest]:
        """Request the Hessians not already cached."""
        return [
            StatisticRequest(target_key=name, stat_type="hessian")
            for name in self._target_modules
            if not self._cache.has("hessian", name)
        ]

    def _hessian(self, name: str, registry: StatisticsRegistry) -> torch.Tensor | None:
        cached = self._cache.load("hessian", name)
        if cached is not None and cached.get("hessian") is not None:
            return cached["hessian"]
        stat = registry.get(StatisticRequest(target_key=name, stat_type="hessian"))
        if stat is None or stat.get("hessian") is None:
            return None
        h = materialize(stat["hessian"])
        self._cache.save("hessian", name, {"hessian": h.cpu().float()})
        return h

    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Trellis-code every target linear."""
        assignments: dict[str, Any] = {}
        kept_float: dict[str, str] = {}
        n_weights, n_bits = 0, 0.0
        for i, (name, mod) in enumerate(self._target_modules.items()):
            offset = self._seed_offsets.get(name, i)
            weight = mod.weight
            assert isinstance(weight, torch.Tensor)
            h = self._hessian(name, registry)
            if h is None:
                kept_float[name] = "no Hessian collected"
                continue
            cfg = replace(
                self.config, bits=self.bits_for(name), seed=self.config.seed + offset
            )
            dev = weight.device if weight.is_cuda else torch.device("cuda")
            try:
                with timed_span(self._spans, "trellis.layer", layer=name):
                    tw = trellis_quantize(weight.detach().to(dev), h.to(dev), cfg)
            except (ValueError, torch.linalg.LinAlgError) as exc:
                LOGGER.error("trellis failed for '%s': %s - kept float", name, exc)
                kept_float[name] = str(exc)
                continue
            if self.store == "dense":
                assignments[name] = tw.dequantize(weight.dtype).to(weight.device)
            else:
                assignments[name] = QTensorTrellis.from_weight(
                    tw.to(weight.device), weight.dtype
                )
            n_out, k_in = weight.shape
            n_weights += n_out * k_in
            n_bits += tw.bpw * n_out * k_in
            LOGGER.info(
                "trellis %s K=%s (%d/%d)",
                name,
                cfg.bits,
                i + 1,
                len(self._target_modules),
            )
        if n_weights:
            LOGGER.info(
                "trellis: %.4f bpw over %d quantized weights",
                n_bits / n_weights,
                n_weights,
            )
        return QuantizationPlan(
            weight_assignments=_add_tied_aliases(assignments, self._tied_aliases),
            kept_float=_add_tied_aliases(kept_float, self._tied_aliases),
        )


@estimator_config("trellis")
class TrellisEstimatorConfig(EstimatorConfig):
    """JSON configuration for :class:`TrellisEstimator`."""

    type: Literal["trellis"] = "trellis"
    target: str = "linear"
    bits: float = 3
    bits_per_group: dict[str, float] = Field(default_factory=dict)
    codebook: Literal["mul1", "mcg", "3inst"] = "mul1"
    damp: float = Field(default=0.025, ge=0)
    order: Literal["act", "natural", "reverse"] = "act"
    seed: int = 0
    store: Literal["packed", "dense"] = "packed"
    exclude_leaf_names: list[str] = Field(default_factory=lambda: ["lm_head"])

    def build(self) -> TrellisEstimator:
        """Build a :class:`TrellisEstimator`; ``lm_head`` stays float by default."""
        from .estimator_configs import _resolve_module_class

        excluded = set(self.exclude_leaf_names)

        def filter_fn(name: str, _mod: nn.Module) -> bool:
            return name.rsplit(".", 1)[-1] not in excluded

        return TrellisEstimator(
            module_classes=(_resolve_module_class(self.target),),
            filter_fn=filter_fn,
            config=TrellisConfig(
                bits=self.bits,
                codebook=self.codebook,
                damp=self.damp,
                order=self.order,
                seed=self.seed,
            ),
            bits_per_group=dict(self.bits_per_group),
            store=self.store,
        )
