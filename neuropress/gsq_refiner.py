"""GSQ refinement of a block's scalar plan, for :class:`BlockQuantizationPipeline`.

:class:`GSQRefiner` is a block refiner: the pipeline calls it for each stage
after the estimator (GPTQ, say) has planned the stage's weights and before the
adapter applies them. It replaces the planned codes and scales of some of the
stage's weights by those GSQ (:mod:`neuropress.methods.gsq`) finds when it
optimizes them jointly to reproduce the output of the float block, or of one
of its submodules, on the student stream. The plan keeps its scheme and
destination format, so the result stores, exports and runs like any GPTQ plan.

The paper's schedule for a Llama-style block, with stages
``(q, k, v, o) -> (gate, up, down)``::

    GSQRefiner(
        groups=(
            GSQGroup(("self_attn.v_proj", "self_attn.o_proj"), objective="self_attn"),
            GSQGroup(("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")),
        )
    )

``q_proj`` and ``k_proj`` keep their GPTQ codes. ``v`` and ``o`` are fitted to
the attention output with the quantized ``q`` and ``k`` in place, and the MLP
to the block output with the quantized attention in place. Matching
``self_attn`` rather than the residual sum gives the same differences, as the
residual is common to both sides.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from .calibration.batches import call_batch, move_batch
from .methods.gsq import (
    GSQConfig,
    GumbelCodes,
    candidate_mode,
    evaluate_codes,
    fit_gumbel_codes,
)
from .plan import QuantizationPlan, QuantizedWeight
from .qscheme import ZpScaleScheme

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GSQGroup:
    """Weights optimized together, and the output they must reproduce.

    Attributes:
        targets: Module paths in the block, each planned with a scalar
            :class:`QuantizedWeight` in the same pipeline stage.
        objective: Path of the submodule whose output is matched; ``""`` is
            the block itself. The forward stops at that submodule.
    """

    targets: tuple[str, ...]
    objective: str = ""


class _Stop(Exception):
    def __init__(self, output: Any) -> None:
        self.output = output


def _primary(output: Any) -> torch.Tensor:
    """The tensor a module output stands for: itself, or its first element."""
    while not isinstance(output, torch.Tensor):
        if isinstance(output, Mapping):
            output = next(iter(output.values()))
        elif isinstance(output, list | tuple):
            output = output[0]
        else:
            raise TypeError(f"cannot match a module output of type {type(output)}")
    return output


def _call(model: nn.Module, params: Mapping[str, torch.Tensor], batch: Any) -> Any:
    if not params:
        return call_batch(model, batch)
    if isinstance(batch, dict):
        return functional_call(model, params, (), batch)
    if isinstance(batch, list | tuple):
        return functional_call(model, params, tuple(batch))
    return functional_call(model, params, (batch,))


def _objective_output(
    model: nn.Module, objective: str, params: Mapping[str, torch.Tensor], batch: Any
) -> torch.Tensor:
    if not objective:
        return _primary(_call(model, params, batch))

    def stop(_module, _args, output):
        raise _Stop(output)

    handle = model.get_submodule(objective).register_forward_hook(stop)
    try:
        _call(model, params, batch)
    except _Stop as found:
        return _primary(found.output)
    finally:
        handle.remove()
    raise RuntimeError(f"the forward never reached {objective!r}")


def _scalar_parts(
    qw: QuantizedWeight,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """``(levels, scale [rows, groups], qmin, qmax, zero_point)`` of a plain scalar plan."""
    scheme = qw.scheme
    if not isinstance(scheme, ZpScaleScheme) or qw.transposed:
        raise ValueError("GSQ refines plain ZpScaleScheme weights only")
    shape = list(scheme.calibration_data_shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"unsupported scheme layout {shape}")
    zero = torch.unique(scheme.zero_point)
    if zero.numel() != 1 or not float(zero).is_integer():
        raise ValueError("GSQ needs one integer zero point (a symmetric scheme)")
    zp = int(zero)
    rows = shape[0]
    levels = qw.codes.reshape(rows, -1).to(torch.int32) - zp
    scale = scheme.scale.reshape(rows, shape[2])
    return levels, scale, -zp, scheme.maxq - zp, zp


def _write_back(qw: QuantizedWeight, codes: GumbelCodes, zp: int) -> QuantizedWeight:
    scheme = qw.scheme
    assert isinstance(scheme, ZpScaleScheme)
    new_codes = (codes.hard_levels() + zp).to(torch.uint8).reshape(qw.codes.shape)
    scale = codes.scale.detach().reshape(scheme.scale.shape)
    new_scheme = ZpScaleScheme(
        zero_point=scheme.zero_point,
        scale=scale.to(device=scheme.scale.device, dtype=scheme.scale.dtype),
        nbits=scheme.nbits,
        calibration_data_shape=list(scheme.calibration_data_shape),
        qscheme_data_shape=list(scheme.qscheme_data_shape),
    )
    return replace(qw, codes=new_codes.to(qw.codes.device), scheme=new_scheme)


@dataclass
class GSQRefiner:
    """Block refiner running GSQ on groups of planned scalar weights.

    Args:
        groups: Weight groups, each optimized in the pipeline stage that plans
            all its targets. Planned weights outside every group keep their
            codes, and are used quantized while the stage's groups are fitted.
        config: GSQ optimization settings.
        keep_better: Keep the starting codes of a group whose fitted codes do
            not lower the objective on the calibration batches.
        cache_device: Where the float outputs wait during a fit; ``None`` keeps
            them on the forward device.

    ``history`` collects one record per fitted group: its targets, the loss of
    the starting codes, the loss of the fitted codes and whether they were kept.
    """

    groups: tuple[GSQGroup, ...]
    config: GSQConfig = field(default_factory=GSQConfig)
    keep_better: bool = True
    cache_device: str | None = None
    history: list[dict[str, Any]] = field(
        default_factory=list, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for group in self.groups:
            if not group.targets or seen & set(group.targets):
                raise ValueError("GSQ groups need distinct, nonempty targets")
            seen.update(group.targets)

    def _check_paths(self, block: nn.Module) -> None:
        """Fail on a group path the block does not have, rather than skip it."""
        modules = dict(block.named_modules())
        for group in self.groups:
            for path in (*group.targets, group.objective):
                if path not in modules:
                    raise ValueError(f"GSQ group path {path!r} is not in the block")

    def refine(
        self,
        block: nn.Module,
        plan: QuantizationPlan,
        batches: list[Any],
        teacher: Callable[[], nn.Module],
        *,
        device: torch.device,
        name: str = "",
    ) -> None:
        """Refine *plan* in place for the groups whose targets it plans.

        Args:
            block: The block, with float weights for the planned targets.
            plan: The stage's plan.
            batches: Student-stream input batches (host tensors).
            teacher: Returns the frozen float block, placed on *device*.
            device: Forward device.
            name: Block name, for logs and ``history``.
        """
        self._check_paths(block)
        planned = set(plan.weight_assignments)
        # Targets the estimator kept float stay float; the rest of the group
        # is still fitted around them.
        handled = planned | set(plan.kept_float)
        groups = []
        for group in self.groups:
            targets = tuple(t for t in group.targets if t in planned)
            if not targets:
                continue
            if not set(group.targets) <= handled:
                raise ValueError(f"a GSQ group spans pipeline stages: {group.targets}")
            if len(targets) < len(group.targets):
                LOGGER.warning(
                    "GSQ %s: %s kept float by the estimator, fitting %s only",
                    name,
                    sorted(set(group.targets) - planned),
                    targets,
                )
            groups.append((self.groups.index(group), replace(group, targets=targets)))
        if not groups:
            return
        home = next(block.parameters()).device
        block.to(device)
        frozen = {p: p.requires_grad for p in block.parameters()}
        for p in frozen:
            p.requires_grad_(False)
        try:
            for index, group in groups:
                self._fit(block, plan, batches, teacher, group, device, name, index)
        finally:
            for p, flag in frozen.items():
                p.requires_grad_(flag)
            block.to(home)

    def _fixed_weights(
        self, block: nn.Module, plan: QuantizationPlan, exclude: set[str], device
    ) -> dict[str, torch.Tensor]:
        """Planned weights outside *exclude*, dequantized as the adapter will see them."""
        fixed = {}
        for path, qw in plan.weight_assignments.items():
            if path in exclude or not isinstance(qw, QuantizedWeight):
                continue
            weight = block.get_submodule(path).weight
            dense = qw.dequantize()
            fixed[f"{path}.weight"] = dense.reshape(weight.shape).to(
                device, weight.dtype
            )
        return fixed

    def _fit(self, block, plan, batches, teacher, group, device, name, index) -> None:
        # distinct noise per block and group (index in self.groups, so groups
        # of different stages differ too), reproducible across processes
        offset = zlib.crc32(name.encode()) % (1 << 30) + index
        config = replace(self.config, seed=self.config.seed + offset)
        generator = torch.Generator(device).manual_seed(config.seed)
        codes: dict[str, GumbelCodes] = {}
        zero_points = {}
        for path in group.targets:
            qw = plan.weight_assignments[path]
            if not isinstance(qw, QuantizedWeight):
                raise ValueError(f"{path} is not planned as a scalar QuantizedWeight")
            if block.get_submodule(path).weight.dim() != 2:
                raise ValueError(
                    f"GSQ fits 2-D weights; {path} has "
                    f"{tuple(block.get_submodule(path).weight.shape)}"
                )
            levels, scale, qmin, qmax, zp = _scalar_parts(qw.on_scheme_device())
            mode = candidate_mode(qmax - qmin + 1, config)
            codes[path] = GumbelCodes(
                levels.to(device), scale.to(device), qmin, qmax, mode, config, generator
            )
            zero_points[path] = zp
        dtypes = {
            path: block.get_submodule(path).weight.dtype for path in group.targets
        }
        fixed = self._fixed_weights(block, plan, set(group.targets), device)

        cache = self.cache_device or device
        # Inputs wait where the targets wait, so a step moves nothing when
        # both stay on the forward device.
        inputs = [move_batch(batch, cache) for batch in batches]
        with torch.no_grad():
            reference = teacher()
            targets = [
                _objective_output(
                    reference, group.objective, {}, move_batch(batch, device)
                ).to(cache)
                for batch in inputs
            ]
        # Only the cached targets need the teacher; the pipeline keeps it on
        # the host between uses, so it gives the fit its device memory back.
        reference.cpu()

        def loss(weights: Mapping[str, torch.Tensor], i: int) -> torch.Tensor:
            params = dict(fixed)
            for path, w in weights.items():
                params[f"{path}.weight"] = w.to(dtypes[path])
            out = _objective_output(
                block, group.objective, params, move_batch(inputs[i], device)
            )
            target = targets[i].to(out.device)
            return (out.float() - target.float()).pow(2).mean()

        start = evaluate_codes(
            {p: c.initial() for p, c in codes.items()}, loss, len(batches)
        )
        label = f"{name} {'|'.join(group.targets)}"
        history = fit_gumbel_codes(
            codes,
            loss,
            len(batches),
            config,
            generator=generator,
            log=lambda line: LOGGER.info("GSQ %s: %s", label, line),
        )
        # Judge the scales the plan will store, not their float32 masters.
        for path, c in codes.items():
            stored = plan.weight_assignments[path].scheme.scale.dtype
            c.scale.data.copy_(c.scale.data.to(stored))
        final = evaluate_codes(
            {p: c.hard() for p, c in codes.items()}, loss, len(batches)
        )
        # On a tie the GPTQ codes stay: fitted codes that change nothing
        # measurable are only noise.
        kept = final < start or not self.keep_better
        LOGGER.info(
            "GSQ %s: start %.4e -> fitted %.4e (%s)",
            label,
            start,
            final,
            "kept" if kept else "start codes kept",
        )
        self.history.append(
            {
                "block": name,
                "targets": group.targets,
                "start_loss": start,
                "fitted_loss": final,
                "train_loss": history,
                "kept": kept,
                "seed": config.seed,
            }
        )
        if kept:
            for path, c in codes.items():
                plan.weight_assignments[path] = _write_back(
                    plan.weight_assignments[path], c, zero_points[path]
                )
