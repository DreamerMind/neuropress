"""GPTQ Q4_0 for fused MoE experts, handed to torch_to_nnef at extraction.

``GPTQEstimator`` quantizes ``nn.Linear`` modules only, and torch_to_nnef
can only quantize the fused expert tensors of an MoE block with min-max
Q4_0 when it extracts the ``tract_moe_ffn`` operator. This module adds
calibrated expert quantization through the compression registry. The
quantization policy stays in neuropress, and torch_to_nnef only serializes
what neuropress hands it:

1. :class:`.MoEExpertHessianCollector` gathers, for every expert, the
   Hessian of the tokens routed to it and the Hessian of its intermediate
   activations.
2. :class:`GPTQMoEExpertsEstimator` runs GPTQ on each expert (``[2I, D]``
   gate/up rows and ``[D, I]`` down rows, Q4_0 blocks of 32 along the
   reduction dimension) and plans one ``QTensorTractScaleOnly`` per fused
   parameter under ``"<experts path>/gate_up_proj"`` and
   ``"<experts path>/down_proj"``.
3. :class:`MoEExpertsQ40Adapter` writes the dequantized values into the
   float parameters, so the PyTorch model computes what tract will, and marks
   the parent MoE block with the torch_to_nnef markers:
   ``_t2n_quantize_moe_experts_q40``, ``_t2n_moe_expert_layout = "linear"``
   and a quantizer (:func:`expert_q40_from_gptq_codes`) that returns the
   GPTQ codes instead of recalibrating with min-max. Only the f16 block scales
   are kept; the codes are read back from the float parameters, which hold
   exactly what they decode to. Checkpoints written by
   :func:`.save_quantized_model` keep the scales and restore the markers.

The layout is not a free choice. GPTQ groups run along each expert's input
dimension, and a Q4_0 block covers 32 consecutive values of the stored
tensor's last dimension. Only the ``linear`` layout (``w1``/``w3`` ``[E, I,
D]``, ``w2`` ``[E, D, I]``) keeps the input dimension last. The ``canonical``
layout transposes it away, and storing the same codes there would need a
different quantization. The quantizer refuses any tensor it cannot match to
GPTQ codes, so changing the layout or the weights after calibration fails
the export rather than exporting other values.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import torch
from torch import nn

from .adapter import ModuleQuantizationAdapter
from .calibration.batches import materialize
from .calibration.probe import StatisticRequest, StatisticsRegistry
from .calibration.probe.moe_experts import is_fused_experts
from .estimator import QuantizationParamEstimator
from .graph.graph import QGraph
from .methods.gptq import GPTQCalibration
from .methods.zp_scale_calib import NormZpScaleCalibration
from .plan import QuantizationPlan
from .qscheme.descriptor import Granularity
from .recipe import QuantizationRecipe
from .recipe_config import (
    AdapterConfig,
    EstimatorConfig,
    adapter_config,
    estimator_config,
)

LOGGER = logging.getLogger(__name__)

Q40_BLOCK = 32
"""Values per tract Q4_0 block."""

STAT_TYPE = "moe_expert_hessian"

_MATCH_STEPS = 0.1
"""Largest distance, in quantization steps, between a weight and its decoded code.

bf16 rounds a decoded value up to 8 steps from zero by about 3% of a step.
"""

EXPERT_LAYOUT = "linear"
"""The torch_to_nnef expert layout whose last dimension is GPTQ's group axis."""


# ---------------------------------------------------------------------------
# Per-expert GPTQ
# ---------------------------------------------------------------------------


def _gptq_expert(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    nsamples: int,
    *,
    group_size: int,
    damp_percent: float,
    damp_auto_increment: float,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GPTQ codes ``[R, C]`` and scales ``[R, C // group_size]``.

    An expert no calibration token reached is quantized against an identity
    Hessian, which reduces GPTQ to round-to-nearest.
    """
    gran = Granularity.per_group([group_size])
    calib = GPTQCalibration(
        granularity=gran,
        possible_quantizers=[
            NormZpScaleCalibration(nbits=4, symmetric=True, granularity=gran)
        ],
        damp_percent=damp_percent,
        damp_auto_increment=damp_auto_increment,
    )
    calib.weight_tensor = nn.Parameter(weight.detach().clone(), requires_grad=False)
    calib.module_class = nn.Linear
    calib.module = None
    calib.name = name
    calib.device = weight.device
    calib.init()
    if nsamples == 0:
        hessian = torch.eye(weight.shape[1], device=weight.device)
        nsamples = 1
    calib.hessian = hessian.to(weight.device, torch.float32)
    calib.nsamples = nsamples
    _, scheme, *_ = calib.compute_quantization()
    if not torch.all(scheme.zero_point == 8):
        raise ValueError(f"{name}: Q4_0 needs symmetric codes centered on 8")
    rows, cols = weight.shape
    return calib.quantized_codes, scheme.scale.reshape(rows, cols // group_size)


def _q40_qtensor(
    codes: torch.Tensor, block_scales: torch.Tensor, dtype: torch.dtype, name: str
) -> torch.Tensor:
    """Pack ``[..., C]`` codes and their f16 block scales as Q4_0 storage.

    The single builder of expert Q4_0 tensors, for the plan and for export.
    """
    from torch_to_nnef.tensor.quant.base import QScalePerGroupF16
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    from .t2n.packing import BitPacking4
    from .t2n.qtensors import qtensor_from_blob
    from .transforms.weight_only import _pack_keeping_shape

    return qtensor_from_blob(
        QTensorTractScaleOnly,
        _pack_keeping_shape(BitPacking4, codes.cpu()),
        codes.shape,
        QScalePerGroupF16(Q40_BLOCK, block_scales.reshape(-1, 1).half().cpu(), 4),
        dtype,
        [BitPacking4],
        name,
    )


def _q40_tensor(
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Wrap ``[E, R, C]`` codes and ``[E, R, C // g]`` scales as Q4_0 storage.

    Scales are rounded to f16 the way the ``nn.Linear`` Q4_0 export rounds
    them, and blocks whose scale falls below the f16 range are stored as zeros.
    """
    from .transforms.weight_only import _f16_block_scales, _store_blocks_as_zero

    blocks_per_group = codes.shape[-1] // group_scales.shape[-1] // Q40_BLOCK
    scales, underflow = _f16_block_scales(
        group_scales, blocks_per_group, name, max_steps=8.0
    )
    codes = _store_blocks_as_zero(
        codes.cpu(),
        underflow.cpu(),
        torch.full_like(underflow.cpu(), 8, dtype=torch.uint8),
        Q40_BLOCK,
    )
    return _q40_qtensor(codes, scales, dtype, name)


@dataclass
class GPTQMoEExpertsEstimator(QuantizationParamEstimator):
    """GPTQ Q4_0 assignments for the fused expert parameters of MoE models.

    Targets every module :func:`.is_fused_experts` accepts whose ``D`` and
    ``I`` are multiples of ``group_size``. Plans a ``QTensorTractScaleOnly``
    for ``"<path>/gate_up_proj"`` and ``"<path>/down_proj"``; apply them with
    :class:`MoEExpertsQ40Adapter`. ``WeightAssignmentAdapter`` skips these
    keys, so the estimator composes with the ``nn.Linear`` GPTQ estimator.
    Both ``down_proj`` Hessians come from the float ``gate_up_proj``, as all
    statistics of one pipeline run are collected on the float model.
    """

    requires_graph: ClassVar[bool] = False

    group_size: int = Q40_BLOCK
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025
    filter_fn: Callable[[str, nn.Module], bool] | None = None

    _targets: dict[str, nn.Module] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def prepare(self, model: nn.Module) -> None:
        """Find the fused expert modules whose shapes Q4_0 can store."""
        if self.group_size % Q40_BLOCK:
            raise ValueError("Q4_0 needs group_size to be a multiple of 32")
        self._targets = {}
        for name, mod in model.named_modules():
            if not is_fused_experts(mod):
                continue
            if self.filter_fn is not None and not self.filter_fn(name, mod):
                continue
            _, two_inter, d_model = mod.gate_up_proj.shape
            if d_model % self.group_size or (two_inter // 2) % self.group_size:
                LOGGER.warning(
                    "%s: hidden %d / intermediate %d not multiples of %d, "
                    "experts stay float",
                    name,
                    d_model,
                    two_inter // 2,
                    self.group_size,
                )
                continue
            self._targets[name] = mod
        LOGGER.info(
            "GPTQMoEExpertsEstimator: %d fused expert modules", len(self._targets)
        )

    def declare_statistics(self) -> list[StatisticRequest]:
        """Request the per-expert Hessians of every target."""
        return [
            StatisticRequest(target_key=name, stat_type=STAT_TYPE)
            for name in self._targets
        ]

    @torch.no_grad()
    def estimate(
        self,
        graph: QGraph,
        registry: StatisticsRegistry,
        recipe: QuantizationRecipe,
    ) -> QuantizationPlan:
        """Run GPTQ per expert and plan one Q4_0 tensor per fused parameter."""
        plan = QuantizationPlan()
        for name, mod in self._targets.items():
            stat = registry.get(StatisticRequest(target_key=name, stat_type=STAT_TYPE))
            if stat is None or stat["nsamples"] is None:
                for param_name in ("gate_up_proj", "down_proj"):
                    plan.kept_float[f"{name}/{param_name}"] = (
                        "no calibration call reached these experts"
                    )
                continue
            counts = stat["nsamples"].tolist()
            dead = sum(1 for c in counts if c == 0)
            if dead:
                LOGGER.warning(
                    "%s: %d/%d experts saw no calibration token, rounding them "
                    "to nearest",
                    name,
                    dead,
                    len(counts),
                )
            for param_name, hessians in (
                ("gate_up_proj", stat["gate_up"]),
                ("down_proj", stat["down"]),
            ):
                weight = getattr(mod, param_name)
                hessians = materialize(hessians)
                codes, scales = [], []
                for expert, count in enumerate(counts):
                    c, s = _gptq_expert(
                        weight[expert],
                        hessians[expert],
                        count,
                        group_size=self.group_size,
                        damp_percent=self.damp_percent,
                        damp_auto_increment=self.damp_auto_increment,
                        name=f"{name}.{param_name}[{expert}]",
                    )
                    codes.append(c.cpu())
                    scales.append(s.float().cpu())
                key = f"{name}/{param_name}"
                plan.weight_assignments[key] = _q40_tensor(
                    torch.stack(codes), torch.stack(scales), weight.dtype, key
                )
        return plan


# ---------------------------------------------------------------------------
# torch_to_nnef handoff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExpertScales:
    """f16 Q4_0 block scales ``[E, R, C // 32]`` of one expert weight.

    The codes are not kept: the float parameters hold exactly what they decode
    to, so :meth:`codes_for` reads them back from the extracted tensor.
    """

    name: str
    scales: torch.Tensor

    def codes_for(self, tensor: torch.Tensor) -> torch.Tensor | None:
        """Codes ``[E, R, C]`` that decode to *tensor*, or None if none do.

        A weight must sit within a tenth of a quantization step of a grid
        point (neighboring codes are a whole step apart), which tolerates a
        cast of the float parameters to f16 or bf16.
        """
        shape = (*self.scales.shape[:-1], self.scales.shape[-1] * Q40_BLOCK)
        if tuple(tensor.shape) != shape:
            return None
        values = tensor.detach().float().cpu()
        step = self.scales.float().repeat_interleave(Q40_BLOCK, dim=-1)
        levels = (values / step).round()
        if levels.min() < -8 or levels.max() > 7:
            return None
        if not torch.all((values - levels * step).abs() <= _MATCH_STEPS * step):
            return None
        return (levels + 8).to(torch.uint8)


def _decode(qtensor: torch.Tensor, name: str) -> tuple[torch.Tensor, _ExpertScales]:
    """Float32 values of a planned expert Q4_0 tensor, and its block scales."""
    codes = qtensor.decompress_to_u8()
    scales = qtensor.qscheme.scale.reshape(*codes.shape[:-1], -1).half()
    values = (codes.float() - 8.0).unflatten(-1, (-1, Q40_BLOCK))
    values = (values * scales.float().unsqueeze(-1)).flatten(-2)
    return values, _ExpertScales(name, scales)


def _layout_candidates(
    gate_up: _ExpertScales, down: _ExpertScales
) -> list[_ExpertScales]:
    """Return the scales of the tensors torch_to_nnef extracts in the linear layout.

    ``w1`` (gate) and ``w3`` (up) are the two row halves of ``gate_up_proj``.
    Rows are quantized independently, so slicing them keeps every block.
    """
    inter = gate_up.scales.shape[1] // 2
    halves = [
        _ExpertScales(f"{gate_up.name}[w{i}]", gate_up.scales[:, rows].contiguous())
        for i, rows in ((1, slice(0, inter)), (3, slice(inter, None)))
    ]
    return [*halves, _ExpertScales(f"{down.name}[w2]", down.scales)]


def expert_q40_from_gptq_codes(
    tensor: torch.Tensor, *, sources: list[_ExpertScales]
) -> torch.Tensor:
    """torch_to_nnef MoE expert quantizer returning precomputed GPTQ codes.

    torch_to_nnef calls it with each expert weight ``w1``/``w2``/``w3`` after
    applying the module's expert layout. The weight is matched to the source
    whose grid it lies on (:meth:`_ExpertScales.codes_for`), and a
    ``QTensorTractScaleOnly`` holding those exact codes and scales is returned.

    Raises:
        ValueError: If no GPTQ source decodes to *tensor*, for instance in
            the canonical layout or after the weights changed.
    """
    for source in sources:
        codes = source.codes_for(tensor)
        if codes is not None:
            return _q40_qtensor(codes, source.scales, tensor.dtype, source.name)
    raise ValueError(
        f"no GPTQ expert codes decode to the {tuple(tensor.shape)} tensor "
        f"torch_to_nnef extracted; GPTQ Q4_0 experts need expert layout "
        f"{EXPERT_LAYOUT!r} and unchanged expert weights"
    )


def _mark_for_t2n(parent: nn.Module, sources: list[_ExpertScales]) -> bool:
    """Set the torch_to_nnef MoE markers on *parent*; False if it has no extractor."""
    from torch_to_nnef.exceptions import T2NErrorNotImplemented
    from torch_to_nnef.op.custom_extractors import mark_moe_experts_for_q40
    from torch_to_nnef.op.custom_extractors.moe import _get_adapter

    try:
        # restricting the marking to the parent's own class always matches it, so ask
        # the extractor itself whether it dispatches on that class
        _get_adapter(parent)
        mark_moe_experts_for_q40(
            parent,
            quantize_q40=True,
            layout=EXPERT_LAYOUT,
            module_class_names={type(parent).__name__},
        )
    except T2NErrorNotImplemented:
        return False
    parent._t2n_quantize_moe_experts_q40_quantizer = expert_q40_from_gptq_codes
    parent._t2n_quantize_moe_experts_q40_kwargs = {"sources": sources}
    return True


SCALES_ATTR = "_neuropress_q40_block_scales"
"""Experts-module attribute: ``{"gate_up_proj", "down_proj"}`` f16 block scales."""

MANIFEST_TRANSFORM = "moe_experts_q40"
"""Manifest transform type recording the experts that carry GPTQ Q4_0 scales."""


def _mark_experts(model: nn.Module, path: str, scales: dict[str, torch.Tensor]) -> bool:
    """Keep *scales* on the experts at *path* and mark their parent for export.

    Returns False, keeping no marker, when torch_to_nnef has no MoE extractor
    for the parent.
    """
    experts = model.get_submodule(path)
    if set(scales) != {"gate_up_proj", "down_proj"}:
        raise ValueError(f"{path}: expected both fused expert assignments")
    setattr(experts, SCALES_ATTR, scales)
    parent_path = path.rsplit(".", 1)[0] if "." in path else None
    if parent_path is None:
        return False
    sources = _layout_candidates(
        _ExpertScales(f"{path}/gate_up_proj", scales["gate_up_proj"]),
        _ExpertScales(f"{path}/down_proj", scales["down_proj"]),
    )
    return _mark_for_t2n(model.get_submodule(parent_path), sources)


def _warn_unmarked(model: nn.Module, path: str) -> None:
    parent = model.get_submodule(path.rsplit(".", 1)[0]) if "." in path else model
    LOGGER.warning(
        "%s: torch_to_nnef has no MoE extractor for %s, experts keep the GPTQ "
        "values as float",
        path,
        type(parent).__name__,
    )


class MoEExpertsQ40Adapter(ModuleQuantizationAdapter):
    """Apply :class:`GPTQMoEExpertsEstimator` assignments and mark for export.

    Writes the dequantized GPTQ values into the fused float parameters, keeps
    their f16 block scales on the experts module (:data:`SCALES_ATTR`, saved
    and restored by :mod:`neuropress.t2n.model_io`) and sets the torch_to_nnef
    markers on the MoE block owning the experts. A block torch_to_nnef has no
    MoE extractor for keeps the dequantized float values, with a warning.
    """

    @torch.no_grad()
    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        """Write expert assignments into *model* and mark their MoE blocks."""
        per_module: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in plan.weight_assignments.items():
            if "/" not in key:
                continue
            path, param_name = key.rsplit("/", 1)
            if param_name not in ("gate_up_proj", "down_proj"):
                continue
            mod = model.get_submodule(path)
            if not is_fused_experts(mod):
                continue
            values, scales = _decode(value, key)
            param = getattr(mod, param_name)
            param.data.copy_(values.to(param.device, param.dtype))
            per_module.setdefault(path, {})[param_name] = scales.scales
        marked = 0
        for path, scales in per_module.items():
            if _mark_experts(model, path, scales):
                marked += 1
            else:
                _warn_unmarked(model, path)
        LOGGER.info(
            "MoEExpertsQ40Adapter: %d expert modules quantized, %d marked for "
            "torch_to_nnef",
            len(per_module),
            marked,
        )
        return model


def moe_q40_checkpoint(
    model: nn.Module,
) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor]]:
    """The manifest transform and tensors that restore the expert markers.

    Returns ``(None, {})`` when no experts carry GPTQ Q4_0 scales. Tensors are
    keyed ``"<experts path>/<param name>"``.
    """
    tensors = {
        f"{path}/{param}": scales
        for path, mod in model.named_modules()
        for param, scales in (getattr(mod, SCALES_ATTR, None) or {}).items()
    }
    if not tensors:
        return None, {}
    paths = sorted({key.rsplit("/", 1)[0] for key in tensors})
    return {"type": MANIFEST_TRANSFORM, "modules": paths}, tensors


def restore_moe_q40(
    model: nn.Module, transform: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> None:
    """Re-mark the experts listed by a :func:`moe_q40_checkpoint` transform."""
    for path in transform["modules"]:
        scales = {
            param: tensors[f"{path}/{param}"] for param in ("gate_up_proj", "down_proj")
        }
        if not _mark_experts(model, path, scales):
            _warn_unmarked(model, path)


# ---------------------------------------------------------------------------
# JSON configs
# ---------------------------------------------------------------------------


@estimator_config("gptq_moe_experts")
class GptqMoeExpertsEstimatorConfig(EstimatorConfig):
    """JSON config for :class:`GPTQMoEExpertsEstimator` (Q4_0 only)."""

    type: Literal["gptq_moe_experts"] = "gptq_moe_experts"
    group_size: int = Q40_BLOCK
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025

    def build(self) -> GPTQMoEExpertsEstimator:
        """Build the estimator."""
        if self.group_size % Q40_BLOCK:
            raise ValueError("Q4_0 needs group_size to be a multiple of 32")
        return GPTQMoEExpertsEstimator(
            group_size=self.group_size,
            damp_percent=self.damp_percent,
            damp_auto_increment=self.damp_auto_increment,
        )


@adapter_config("moe_experts_q4_0")
class MoEExpertsQ40AdapterConfig(AdapterConfig):
    """JSON config for :class:`MoEExpertsQ40Adapter`."""

    type: Literal["moe_experts_q4_0"] = "moe_experts_q4_0"

    def build(self) -> MoEExpertsQ40Adapter:
        """Build the adapter."""
        return MoEExpertsQ40Adapter()
