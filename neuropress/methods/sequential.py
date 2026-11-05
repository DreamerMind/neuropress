"""Block-sequential calibration: quantize decoder blocks on propagated inputs.

Global calibration collects every Hessian from the unquantized model, so each
layer is fitted as if all earlier layers were exact. Here decoder blocks are
processed in order. Within a block, each shared-input Hessian is collected
only after every projection executed before it has been quantized, and the
quantized block's outputs feed the next block.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..transforms.rotation_surgery import _find_decoder_stack

LOGGER = logging.getLogger(__name__)

QuantizeFn = Callable[[str, nn.Module, dict[str, Any]], None]

_CACHE_KWARGS = ("past_key_values", "past_key_value")


class _StopForward(Exception):
    """Raised by the capture hook once a batch's block-0 inputs are stored."""


@dataclass
class BlockInput:
    """Positional hidden states plus the keyword arguments of one block call."""

    hidden_states: torch.Tensor
    kwargs: dict[str, Any]


def run_model(model: nn.Module, batch: Any) -> Any:
    """Call *model* with the same batch conventions as the global pass."""
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, list | tuple):
        return model(*batch)
    return model(batch)


def capture_block_inputs(
    model: nn.Module, block: nn.Module, batches: list[Any]
) -> list[BlockInput]:
    """Record what *block* receives for each batch without running the rest."""
    captured: list[BlockInput] = []

    def _hook(_module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        kwargs = dict(kwargs)
        hidden = args[0] if args else kwargs.pop("hidden_states")
        for key in _CACHE_KWARGS:
            if key in kwargs:
                kwargs[key] = None
        if "use_cache" in kwargs:
            kwargs["use_cache"] = False
        captured.append(BlockInput(hidden.detach(), kwargs))
        raise _StopForward

    handle = block.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        for batch in batches:
            try:
                run_model(model, batch)
            except _StopForward:
                pass
    finally:
        handle.remove()
    return captured


def run_block(block: nn.Module, block_input: BlockInput) -> torch.Tensor:
    """Run one decoder block and return its hidden-state output."""
    output = block(block_input.hidden_states, **block_input.kwargs)
    return output[0] if isinstance(output, tuple) else output


class _HessianSum:
    """Accumulate ``2/n * sum(x x^T)`` matching ``HessianAccumulator``."""

    def __init__(self) -> None:
        self.total: torch.Tensor | None = None
        self.count = 0

    def update(self, x: torch.Tensor) -> None:
        rows = x.reshape(-1, x.shape[-1]).float()
        product = rows.T @ rows
        self.total = product if self.total is None else self.total + product
        self.count += rows.shape[0]

    def stat(self) -> dict[str, Any]:
        assert self.total is not None, "module never ran during calibration"
        return {"hessian": 2.0 * self.total / self.count, "nsamples": self.count}


def _collect(
    modules: dict[str, nn.Module], run: Callable[[], None]
) -> dict[str, dict[str, Any]]:
    """Accumulate input Hessians of *modules* while *run* executes."""
    sums = {name: _HessianSum() for name in modules}
    hooks = [
        module.register_forward_hook(
            lambda _m, inputs, _o, _s=sums[name]: _s.update(inputs[0])
        )
        for name, module in modules.items()
    ]
    try:
        run()
    finally:
        for hook in hooks:
            hook.remove()
    return {name: total.stat() for name, total in sums.items()}


def _execution_order(
    block: nn.Module, sources: dict[str, nn.Module], probe: BlockInput
) -> list[str]:
    """Return Hessian source names in the order the block first calls them."""
    order: list[str] = []

    def _record(name: str) -> None:
        if name not in order:
            order.append(name)

    hooks = [
        module.register_forward_hook(lambda _m, _i, _o, _n=name: _record(_n))
        for name, module in sources.items()
    ]
    try:
        run_block(block, probe)
    finally:
        for hook in hooks:
            hook.remove()
    missing = [name for name in sources if name not in order]
    if missing:
        raise RuntimeError(f"projections never executed in their block: {missing}")
    return order


def quantize_blocks_sequentially(
    model: nn.Module,
    batches: list[Any],
    targets: dict[str, nn.Module],
    source_of: dict[str, str],
    quantize: QuantizeFn,
) -> None:
    """Quantize *targets* block by block on inputs from the quantized prefix.

    Args:
        model: Model whose decoder stack exposes ``layers``.
        batches: Calibration batches, iterated several times.
        targets: Module name to linear for every module to quantize.
        source_of: Target name to the target whose input Hessian it shares.
        quantize: Called once per target with its Hessian statistic; must
            quantize the module's weight in place.
    """
    stack = _find_decoder_stack(model)
    blocks = list(stack.layers)
    names = {id(module): name for name, module in model.named_modules()}
    remaining = dict(targets)

    with torch.inference_mode():
        inputs = capture_block_inputs(model, blocks[0], batches)
        for index, block in enumerate(blocks):
            prefix = names[id(block)] + "."
            in_block = {n: m for n, m in remaining.items() if n.startswith(prefix)}
            for name in in_block:
                del remaining[name]
            sources = {source_of[n]: targets[source_of[n]] for n in in_block}
            LOGGER.info(
                "Sequential GPTQ block %d/%d: %d projections, %d Hessians",
                index + 1,
                len(blocks),
                len(in_block),
                len(sources),
            )

            def _run_block(
                block: nn.Module = block, block_inputs: list[BlockInput] = inputs
            ) -> None:
                for block_input in block_inputs:
                    run_block(block, block_input)

            for source in _execution_order(block, sources, inputs[0]):
                stat = _collect({source: sources[source]}, _run_block)[source]
                for name in (n for n in in_block if source_of[n] == source):
                    quantize(name, in_block[name], stat)
                del stat
            inputs = [
                BlockInput(run_block(block, block_input), block_input.kwargs)
                for block_input in inputs
            ]

        if remaining:
            # Projections outside the stack (e.g. a ternary lm_head) see the
            # full quantized model's activations.
            leftover_sources = {source_of[n]: targets[source_of[n]] for n in remaining}

            def _run_model() -> None:
                for batch in batches:
                    run_model(model, batch)

            stats = _collect(leftover_sources, _run_model)
            for name, module in remaining.items():
                quantize(name, module, stats[source_of[name]])
