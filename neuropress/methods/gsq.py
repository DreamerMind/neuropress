"""GSQ: Gumbel-Softmax refinement of scalar quantization codes.

Dadgarnia et al., "GSQ: Highly-Accurate Low-Precision Scalar Quantization for
LLMs via Gumbel-Softmax Sampling", arXiv:2604.18556. The reference code is
IST-DASLab/GSQ (Apache-2.0); this is a reimplementation of the method.

A scalar-quantized weight is ``W = scale[group] * level`` with ``level`` on an
integer grid (``-4..3`` at 3 bits, symmetric). GSQ keeps that format and
optimizes the per-weight level choice and the group scales jointly by gradient
descent on a reconstruction loss. Each weight gets a few logits over candidate
levels; a relaxed sample

    p = softmax((kappa * logits + gumbel_noise) / tau)

gives a soft level ``sum(p * candidates)``, so the loss is differentiable in
the logits and the scales. ``tau`` anneals from 2 to 0.05 (samples become
one-hot) while the logit scale ``kappa`` grows from 100 to 500; the final level
is the argmax of the logits.

Candidates:

- ``"full"``: every grid level (4 logits per weight at 2 bits), the logits
  initialized by a Gaussian prior centred on the starting level.
- ``"shift"``: shifts ``-2..2`` around the starting level (5 logits), used
  above 4 levels: the paper finds 99.99% of optimized weights move by at most
  one level at 3 bits.

The starting levels and scales come from another quantizer (GPTQ in the
paper; RTN works almost as well, random levels do not). The optimizer is Lion,
whose sign update does not stall when the relaxation saturates and gradients
vanish (the paper's Appendix B).

This module is independent of how the loss is built: :func:`fit_gumbel_codes`
takes a callable that maps weights to a loss.
:mod:`neuropress.gsq_refiner` uses it for block reconstruction in
:class:`~neuropress.block_pipeline.BlockQuantizationPipeline`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GSQConfig:
    """Optimization settings; defaults follow the paper's Llama runs (Table 8).

    Attributes:
        epochs: Passes over the calibration batches.
        batch_size: Calibration batches per optimizer step. One relaxed sample
            is drawn per step and shared by its batches.
        temperature: Softmax temperature, linear from first to second value
            over all steps.
        logit_scale: Logit scale ``kappa``, linear over all steps.
        lr_logits: Lion learning rate of the logits.
        lr_scales: Lion learning rate of the group scales.
        weight_decay: Decoupled weight decay of the logits (not the scales).
        betas: Lion betas.
        min_lr_ratio: Floor of the cosine learning-rate decay, as a fraction
            of the initial rate.
        init_std: Overall scale of the initial logits.
        init_strength: Weight of the prior relative to unit Gaussian noise in
            the initial logits (``alpha``).
        full_levels: Grids with at most this many levels use ``"full"``
            candidates, larger ones ``"shift"``.
        max_shift: Largest shift of the ``"shift"`` candidates.
        logits_dtype: Storage dtype of the logits.
        seed: Seed of the initial logit noise, the Gumbel noise and the batch
            order.
    """

    epochs: int = 20
    batch_size: int = 4
    temperature: tuple[float, float] = (2.0, 0.05)
    logit_scale: tuple[float, float] = (100.0, 500.0)
    lr_logits: float = 1e-4
    lr_scales: float = 5e-5
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    min_lr_ratio: float = 0.1
    init_std: float = 0.01
    init_strength: float = 6.0
    full_levels: int = 4
    max_shift: int = 2
    logits_dtype: torch.dtype = torch.float32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if min(self.temperature) <= 0:
            raise ValueError("temperatures must be positive")
        if self.max_shift < 1:
            raise ValueError("max_shift must be at least 1")


class Lion(torch.optim.Optimizer):
    """Lion (Chen et al. 2023): sign of interpolated momentum, decoupled decay."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        super().__init__(
            params, {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, decay = group["lr"], group["weight_decay"]
            # 0-d tensors: a compiled update sees new rates without recompiling;
            # one copy per device and group, not per parameter
            rates: dict[torch.device, torch.Tensor] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                if p.device not in rates:
                    rates[p.device] = torch.tensor(
                        [lr, beta1, beta2, decay], device=p.device, dtype=torch.float32
                    )
                # p.detach() aliases p, so the in-place update reaches it
                _fused(_lion_update, p)(
                    p.detach(), p.grad, state["exp_avg"], rates[p.device]
                )
        return loss


def _lion_update(
    p: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, rates: torch.Tensor
) -> None:
    lr, beta1, beta2, decay = rates.unbind()
    grad = grad.to(p.dtype)
    update = (exp_avg * beta1 + grad * (1 - beta1)).sign()
    p.mul_(1 - lr * decay).sub_(update * lr)
    exp_avg.mul_(beta2).add_(grad * (1 - beta2))


_COMPILED: dict = {}


def _fused(fn, like: torch.Tensor):
    """*fn* compiled into fused kernels for CUDA tensors, as is elsewhere.

    The relaxation and the optimizer touch every logit a few times per step; on
    the bandwidth-bound devices this runs on, unfused they cost several times
    the matmuls of the block they serve.
    """
    if not like.is_cuda:
        return fn
    if fn not in _COMPILED:
        # Symbolic sizes: one graph per rank and candidate count, not one per
        # weight shape (with fullgraph, the shape past recompile_limit
        # raised). Callers pass detached tensors, as parameters are always
        # compiled with static shapes.
        _COMPILED[fn] = torch.compile(fn, dynamic=True, fullgraph=True)
    return _COMPILED[fn]


_MASKED = -1e30  # finite: an all-masked guard would cost a pass of its own


def _relaxed_probs(
    logits, uniform, init, candidates, scalars, qmin: int, qmax: int, shift: bool
) -> list[torch.Tensor]:
    """Softmax of ``(kappa * logits + gumbel) / tau`` over the valid candidates.

    The few candidates are unrolled, so the softmax is pointwise and compiles
    into the kernel that consumes it.
    """
    inv_tau, kappa = scalars.unbind()
    z = []
    for k in range(logits.shape[0]):
        gumbel = -torch.log(-torch.log(uniform[k] + 1e-8) + 1e-8)
        zk = (logits[k].float() * kappa + gumbel) * inv_tau
        if shift:
            target = init.float() + candidates[k]
            zk = torch.where((target >= qmin) & (target <= qmax), zk, _MASKED)
        z.append(zk)
    top = z[0]
    for zk in z[1:]:
        top = torch.maximum(top, zk)
    e = [torch.exp(zk - top) for zk in z]
    total = e[0]
    for ek in e[1:]:
        total = total + ek
    return [ek / total for ek in e]


def _relaxed_level(
    logits, uniform, init, candidates, scalars, qmin: int, qmax: int, shift: bool
):
    p = _relaxed_probs(logits, uniform, init, candidates, scalars, qmin, qmax, shift)
    level = init.float() if shift else torch.zeros_like(p[0])
    for k, pk in enumerate(p):
        level = level + pk * candidates[k]
    return level


def _relaxed_grad(
    logits,
    uniform,
    init,
    candidates,
    scalars,
    qmin: int,
    qmax: int,
    shift: bool,
    grad,
):
    p = _relaxed_probs(logits, uniform, init, candidates, scalars, qmin, qmax, shift)
    inv_tau, kappa = scalars.unbind()
    # d level / d z_k = p_k (c_k - sum_j p_j c_j) / tau, and z_k = kappa * logit_k + ...
    mean = p[0] * candidates[0]
    for k in range(1, len(p)):
        mean = mean + p[k] * candidates[k]
    factor = grad * (kappa * inv_tau)
    return torch.stack(
        [
            (pk * (candidates[k] - mean) * factor).to(logits.dtype)
            for k, pk in enumerate(p)
        ]
    )


class _Relax(torch.autograd.Function):
    """Relaxed level from logits and uniform noise, in one pass each way.

    Saves the logits and the noise only; the backward recomputes the softmax,
    so no candidate-sized intermediate stays alive between the passes.
    """

    @staticmethod
    def forward(ctx, logits, uniform, init, candidates, scalars, qmin, qmax, shift):
        ctx.save_for_backward(logits, uniform, init, candidates, scalars)
        ctx.grid = (qmin, qmax, shift)
        return _fused(_relaxed_level, logits)(
            logits.detach(), uniform, init, candidates, scalars, qmin, qmax, shift
        )

    @staticmethod
    def backward(ctx, grad):
        logits, uniform, init, candidates, scalars = ctx.saved_tensors
        grad_logits = _fused(_relaxed_grad, logits)(
            logits.detach(),
            uniform,
            init,
            candidates,
            scalars,
            *ctx.grid,
            grad.float().contiguous(),
        )
        return grad_logits, None, None, None, None, None, None, None


class GumbelCodes(nn.Module):
    """Relaxed level choice of one scalar-quantized weight.

    Args:
        levels: ``[rows, cols]`` integer levels of the starting point (codes
            minus zero point).
        scale: ``[rows, n_groups]`` group scales; groups are consecutive
            columns of equal size.
        qmin: Lowest grid level.
        qmax: Highest grid level.
        mode: ``"full"`` or ``"shift"`` candidates (see the module docstring).
        config: Initialization settings (``init_std``, ``init_strength``,
            ``max_shift``, ``logits_dtype``).
        generator: Generator of the initial logit noise, on the levels' device.
    """

    def __init__(
        self,
        levels: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
        mode: Literal["full", "shift"],
        config: GSQConfig,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if levels.dim() != 2 or scale.dim() != 2 or scale.shape[0] != levels.shape[0]:
            raise ValueError("expected [rows, cols] levels and [rows, n_groups] scales")
        rows, cols = levels.shape
        if cols % scale.shape[1]:
            raise ValueError("columns must split evenly into the scale groups")
        if levels.min() < qmin or levels.max() > qmax:
            raise ValueError("starting levels outside the grid")
        self.mode = mode
        self.qmin, self.qmax = int(qmin), int(qmax)
        self.group_size = cols // scale.shape[1]
        device = levels.device
        small = -128 <= qmin and qmax < 128
        self.register_buffer("init", levels.to(torch.int8 if small else torch.int16))
        if mode == "full":
            candidates = torch.arange(
                qmin, qmax + 1, device=device, dtype=torch.float32
            )
            prior = -0.5 * (levels.float()[None] - candidates[:, None, None]) ** 2
            prior = prior - prior.mean(0, keepdim=True)
        elif mode == "shift":
            candidates = torch.arange(
                -config.max_shift,
                config.max_shift + 1,
                device=device,
                dtype=torch.float32,
            )
            valid = self._valid(candidates, levels)
            prior = (-0.5 * candidates**2)[:, None, None].expand(-1, rows, cols)
            mean = (prior * valid).sum(0, keepdim=True) / valid.sum(0, keepdim=True)
            prior = prior - mean
        else:
            raise ValueError(f"unknown candidate mode {mode!r}")
        self.register_buffer("candidates", candidates)
        noise = torch.randn(prior.shape, device=device, generator=generator)
        logits = config.init_std * (noise + config.init_strength * prior)
        self.logits = nn.Parameter(logits.to(config.logits_dtype))
        self.scale = nn.Parameter(scale.detach().float().clone())
        # Lion moves scales by fixed steps, which can cross zero; a scale
        # keeps its starting sign and at least 1e-3 of its starting size.
        self.register_buffer("scale_floor", self.scale.detach() * 1e-3)

    @torch.no_grad()
    def clamp_scale(self) -> None:
        """Keep each scale on its starting side of zero, away from it."""
        floor = self.scale_floor
        self.scale.copy_(
            torch.where(
                floor >= 0,
                torch.maximum(self.scale, floor),
                torch.minimum(self.scale, floor),
            )
        )

    def _valid(self, shifts: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        target = levels.float()[None] + shifts[:, None, None]
        return (target >= self.qmin) & (target <= self.qmax)

    def _masked_logits(self) -> torch.Tensor:
        logits = self.logits.float()
        if self.mode == "shift":
            logits = logits.masked_fill(
                ~self._valid(self.candidates, self.init), float("-inf")
            )
        return logits

    def _expand(self, scale: torch.Tensor) -> torch.Tensor:
        return scale.repeat_interleave(self.group_size, dim=1)

    def soft(
        self,
        temperature: float,
        logit_scale: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """A relaxed sample of the weight, float32, differentiable."""
        device = self.logits.device
        uniform = torch.rand(self.logits.shape, device=device, generator=generator)
        scalars = torch.tensor(
            [1.0 / temperature, logit_scale], device=device, dtype=torch.float32
        )
        level = _Relax.apply(
            self.logits,
            uniform,
            self.init,
            self.candidates,
            scalars,
            self.qmin,
            self.qmax,
            self.mode == "shift",
        )
        return level * self._expand(self.scale)

    @torch.no_grad()
    def hard_levels(self) -> torch.Tensor:
        """``[rows, cols]`` int levels of the argmax candidates."""
        index = self._masked_logits().argmax(0)
        chosen = self.candidates[index]
        if self.mode == "shift":
            chosen = chosen + self.init.float()
        return chosen.round().to(torch.int32)

    @torch.no_grad()
    def hard(self) -> torch.Tensor:
        """The weight at the argmax candidates, float32."""
        return self.hard_levels().float() * self._expand(self.scale)

    @torch.no_grad()
    def initial(self) -> torch.Tensor:
        """The starting weight (starting levels and scales), float32."""
        return self.init.float() * self._expand(self.scale)


def candidate_mode(levels: int, config: GSQConfig) -> Literal["full", "shift"]:
    """``"full"`` for grids of at most ``config.full_levels`` levels, else ``"shift"``."""
    return "full" if levels <= config.full_levels else "shift"


def _cosine(step: int, total: int, floor: float) -> float:
    progress = step / max(total - 1, 1)
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * progress))


def _linear(bounds: tuple[float, float], step: int, total: int) -> float:
    t = step / max(total - 1, 1)
    return bounds[0] + (bounds[1] - bounds[0]) * t


LossFn = Callable[[Mapping[str, torch.Tensor], int], torch.Tensor]


@torch.no_grad()
def evaluate_codes(
    weights: Mapping[str, torch.Tensor], loss: LossFn, n_items: int
) -> float:
    """Mean loss over all items with fixed weights."""
    total = sum(loss(weights, i).detach().double() for i in range(n_items))
    return float(total) / n_items


@torch.enable_grad()
def _check_reached(codes, loss, n_items: int) -> None:
    """Refuse codes the loss never reaches, before any optimizer step.

    Their logits would get no gradient, and the fit would hand back the
    argmax of their noisy initial logits. A well-posed group is reached by
    the first item: one forward and backward at the starting weights.
    """
    missing = set(codes)
    for i in range(n_items):
        leaves = {n: codes[n].initial().detach().requires_grad_() for n in missing}
        value = loss(leaves, i)
        if value.requires_grad:
            value.backward()
        missing = {n for n in missing if leaves[n].grad is None}
        if not missing:
            return
    raise ValueError(
        f"the loss does not depend on {sorted(missing)}: is the objective "
        "downstream of every target?"
    )


@torch.enable_grad()
def fit_gumbel_codes(
    codes: Mapping[str, GumbelCodes],
    loss: LossFn,
    n_items: int,
    config: GSQConfig,
    *,
    generator: torch.Generator | None = None,
    log: Callable[[str], None] | None = None,
) -> list[float]:
    """Optimize the logits and scales of *codes* jointly on ``loss``.

    Args:
        codes: Relaxed weights, by name.
        loss: ``loss(weights, i)`` returns the scalar loss of calibration item
            ``i`` when the weights take the values in ``weights`` (same keys as
            *codes*, tensors of the weight shapes).
        n_items: Number of calibration items.
        config: Optimization settings.
        generator: Generator for the Gumbel noise and the item order; on the
            device of the codes.
        log: Called with one line per epoch.

    Returns:
        Mean training loss of each epoch.

    Each step draws one relaxed sample of every weight, runs ``batch_size``
    items with that sample as a leaf tensor, and backpropagates the
    accumulated weight gradient through the relaxation once, so the relaxation
    cost does not grow with the batch.
    """
    if n_items < 1:
        raise ValueError("no calibration items")
    groups = [
        {
            "params": [c.logits for c in codes.values()],
            "lr": config.lr_logits,
            "weight_decay": config.weight_decay,
        },
        {
            "params": [c.scale for c in codes.values()],
            "lr": config.lr_scales,
            "weight_decay": 0.0,
        },
    ]
    optimizer = Lion(groups, betas=config.betas)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    steps_per_epoch = math.ceil(n_items / config.batch_size)
    total = steps_per_epoch * config.epochs
    order_generator = torch.Generator().manual_seed(config.seed)
    _check_reached(codes, loss, n_items)
    history = []
    step = 0
    for epoch in range(config.epochs):
        order = torch.randperm(n_items, generator=order_generator).tolist()
        # summed on the device, read once per epoch
        epoch_loss: torch.Tensor | float = 0.0
        for start in range(0, n_items, config.batch_size):
            items = order[start : start + config.batch_size]
            tau = _linear(config.temperature, step, total)
            kappa = _linear(config.logit_scale, step, total)
            factor = _cosine(step, total, config.min_lr_ratio)
            for group, lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            soft = {name: c.soft(tau, kappa, generator) for name, c in codes.items()}
            leaves = {name: w.detach().requires_grad_() for name, w in soft.items()}
            for i in items:
                value = loss(leaves, i) / len(items)
                if value.requires_grad:  # an item may reach none of the codes
                    value.backward()
                epoch_loss = epoch_loss + value.detach().double() * len(items)
            names = [n for n in soft if leaves[n].grad is not None]
            torch.autograd.backward(
                [soft[n] for n in names], [leaves[n].grad for n in names]
            )
            optimizer.step()
            for c in codes.values():
                c.clamp_scale()
            step += 1
        history.append(float(epoch_loss) / n_items)
        if log is not None:
            log(
                f"epoch {epoch + 1}/{config.epochs} loss {history[-1]:.4e} "
                f"tau {tau:.3f} kappa {kappa:.0f}"
            )
    return history
