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
   GPTQ codes instead of recalibrating with min-max.

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
from typing import ClassVar, Literal

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


def _q40_tensor(
    codes: torch.Tensor,
    group_scales: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Wrap ``[E, R, C]`` codes and ``[E, R, C // g]`` scales as Q4_0 storage."""
    from torch_to_nnef.tensor.quant.base import QScalePerGroupF16
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    from .t2n.packing import BitPacking4
    from .t2n.qtensors import qtensor_from_blob
    from .transforms.weight_only import (
        _f16_block_scales,
        _pack_keeping_shape,
        _store_blocks_as_zero,
    )

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
    return qtensor_from_blob(
        QTensorTractScaleOnly,
        _pack_keeping_shape(BitPacking4, codes),
        codes.shape,
        QScalePerGroupF16(Q40_BLOCK, scales.cpu(), 4),
        dtype,
        [BitPacking4],
        name,
    )


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
                plan.kept_float[name] = "no calibration call reached these experts"
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
class _ExpertCodes:
    """Q4_0 codes ``[E, R, C]`` and f16 block scales ``[E, R, C // 32]``."""

    name: str
    codes: torch.Tensor
    scales: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        """Return the float32 values tract decodes from these codes."""
        values = (self.codes.float() - 8.0).unflatten(-1, (-1, Q40_BLOCK))
        return (values * self.scales.float().unsqueeze(-1)).flatten(-2)


def _expert_codes(qtensor: torch.Tensor, name: str) -> _ExpertCodes:
    codes = qtensor.decompress_to_u8()
    scales = qtensor.qscheme.scale.reshape(*codes.shape[:-1], -1)
    return _ExpertCodes(name, codes, scales)


def _layout_candidates(gate_up: _ExpertCodes, down: _ExpertCodes) -> list[_ExpertCodes]:
    """Return the tensors torch_to_nnef extracts in the linear layout.

    ``w1`` (gate) and ``w3`` (up) are the two row halves of ``gate_up_proj``.
    Rows are quantized independently, so slicing them keeps every block.
    """
    inter = gate_up.codes.shape[1] // 2
    halves = [
        _ExpertCodes(
            f"{gate_up.name}[w{i}]", gate_up.codes[:, rows], gate_up.scales[:, rows]
        )
        for i, rows in ((1, slice(0, inter)), (3, slice(inter, None)))
    ]
    return [*halves, _ExpertCodes(f"{down.name}[w2]", down.codes, down.scales)]


def expert_q40_from_gptq_codes(
    tensor: torch.Tensor, *, sources: list[_ExpertCodes]
) -> torch.Tensor:
    """torch_to_nnef MoE expert quantizer returning precomputed GPTQ codes.

    torch_to_nnef calls it with each expert weight ``w1``/``w2``/``w3`` after
    applying the module's expert layout. The weight is matched to the source
    whose decoded values it holds, within a tenth of a quantization step
    (neighboring codes are a whole step apart), which tolerates a later cast
    of the float parameters to f16 or bf16. Returns a
    ``QTensorTractScaleOnly`` holding those exact codes and scales.

    Raises:
        ValueError: If no GPTQ source decodes to *tensor*, for instance in
            the canonical layout or after the weights changed.
    """
    from torch_to_nnef.tensor.quant.base import QScalePerGroupF16
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    from .t2n.packing import BitPacking4
    from .t2n.qtensors import qtensor_from_blob

    values = tensor.detach().float().cpu()
    for source in sources:
        if tuple(source.codes.shape) != tuple(values.shape):
            continue
        step = source.scales.float().repeat_interleave(Q40_BLOCK, dim=-1)
        if torch.all((values - source.dequantize()).abs() <= _MATCH_STEPS * step):
            return qtensor_from_blob(
                QTensorTractScaleOnly,
                BitPacking4.compress(source.codes),
                source.codes.shape,
                QScalePerGroupF16(Q40_BLOCK, source.scales.reshape(-1, 1).half(), 4),
                tensor.dtype,
                [BitPacking4],
                source.name,
            )
    raise ValueError(
        f"no GPTQ expert codes decode to the {tuple(values.shape)} tensor "
        f"torch_to_nnef extracted; GPTQ Q4_0 experts need expert layout "
        f"{EXPERT_LAYOUT!r} and unchanged expert weights"
    )


def _mark_for_t2n(parent: nn.Module, sources: list[_ExpertCodes]) -> bool:
    """Set the torch_to_nnef MoE markers on *parent*; False if it has no extractor."""
    from torch_to_nnef.exceptions import T2NErrorNotImplemented
    from torch_to_nnef.op.custom_extractors import mark_moe_experts_for_q40

    try:
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


class MoEExpertsQ40Adapter(ModuleQuantizationAdapter):
    """Apply :class:`GPTQMoEExpertsEstimator` assignments and mark for export.

    Writes the dequantized GPTQ values into the fused float parameters and
    sets the torch_to_nnef markers on the MoE block owning the experts. A
    block torch_to_nnef has no MoE extractor for keeps the dequantized float
    values, with a warning.
    """

    @torch.no_grad()
    def apply(self, model: nn.Module, plan: QuantizationPlan) -> nn.Module:
        """Write expert assignments into *model* and mark their MoE blocks."""
        per_module: dict[str, dict[str, _ExpertCodes]] = {}
        for key, value in plan.weight_assignments.items():
            if "/" not in key:
                continue
            path, param_name = key.rsplit("/", 1)
            if param_name not in ("gate_up_proj", "down_proj"):
                continue
            mod = model.get_submodule(path)
            if not is_fused_experts(mod):
                continue
            codes = _expert_codes(value, key)
            param = getattr(mod, param_name)
            param.data.copy_(codes.dequantize().to(param.device, param.dtype))
            per_module.setdefault(path, {})[param_name] = codes
        parents = {
            id(child): parent
            for parent in model.modules()
            for child in parent.children()
        }
        for path, codes in per_module.items():
            if set(codes) != {"gate_up_proj", "down_proj"}:
                raise ValueError(f"{path}: expected both fused expert assignments")
            parent = parents.get(id(model.get_submodule(path)))
            sources = _layout_candidates(codes["gate_up_proj"], codes["down_proj"])
            if parent is None or not _mark_for_t2n(parent, sources):
                LOGGER.warning(
                    "%s: torch_to_nnef has no MoE extractor for %s, experts "
                    "keep the GPTQ values as float",
                    path,
                    type(parent).__name__ if parent is not None else "the root",
                )
        LOGGER.info(
            "MoEExpertsQ40Adapter: %d expert modules quantized", len(per_module)
        )
        return model


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
