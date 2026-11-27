"""Per-expert input Hessians of fused MoE experts.

Transformers stores the routed experts of most MoE models (Qwen2/3/3.5-MoE,
OLMoE, Mixtral, Granite MoE, ...) as one module holding 3-D parameters:

- ``gate_up_proj`` ``[E, 2I, D]``: the gate rows, then the up rows
- ``down_proj`` ``[E, D, I]``

called as ``experts(hidden_states [T, D], top_k_index [T, k], top_k_weights
[T, k])``. Each expert is two linear maps, so GPTQ needs two Hessians per
expert: one over the tokens routed to it (input of ``gate_up_proj``) and one
over its intermediate activations ``act(gate) * up`` (input of
``down_proj``). Hessians follow :class:`.HessianAccumulator`'s convention,
``2 / n * sum(x x^T)`` in float32.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .collectors import InvocationCollector


def is_fused_experts(module: nn.Module) -> bool:
    """Return True for a fused experts module in the layout this file reads.

    Requires 3-D ``gate_up_proj [E, 2I, D]`` and ``down_proj [E, D, I]``
    parameters, an ``act_fn`` and no expert biases or transposed storage
    (GPT-OSS stores ``[E, D, 2I]`` with interleaved gate/up and biases).
    """
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not (isinstance(gate_up, torch.Tensor) and isinstance(down, torch.Tensor)):
        return False
    if gate_up.dim() != 3 or down.dim() != 3 or not hasattr(module, "act_fn"):
        return False
    if getattr(module, "is_transposed", False) or getattr(module, "has_bias", False):
        return False
    if getattr(module, "has_gate", True) is False:
        return False
    experts, two_inter, hidden = gate_up.shape
    return tuple(down.shape) == (experts, hidden, two_inter // 2) and (
        two_inter % 2 == 0
    )


class MoEExpertHessianCollector(InvocationCollector):
    """Accumulate one ``gate_up_proj`` and one ``down_proj`` Hessian per expert.

    The ``down_proj`` input is recomputed from the module's current
    ``gate_up_proj`` and ``act_fn``, whatever experts implementation
    (eager loop, batched or grouped mm) the module forward dispatches to.
    Every routed token counts once, whatever its routing weight.
    """

    def __init__(self) -> None:
        """Start with no observations."""
        self._gate_up: torch.Tensor | None = None
        self._down: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._result: dict[str, Any] | None = None
        # Host-side Hessians: products wait on the device and cross in
        # batches of _FLUSH experts, not one transfer per expert and call.
        self._pending: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    _FLUSH = 8

    @torch.no_grad()
    def update_call(self, module: Any, inputs: tuple, output: Any) -> None:
        """Accumulate the tokens of one call into their experts' Hessians."""
        if len(inputs) < 2:
            raise ValueError(
                "fused experts must be called with (hidden_states, top_k_index, ...)"
            )
        hidden, top_k_index = inputs[0], inputs[1]
        gate_up = module.gate_up_proj
        n_experts, two_inter, d_model = gate_up.shape
        inter = two_inter // 2
        hidden = hidden.reshape(-1, d_model)
        top_k_index = top_k_index.reshape(hidden.shape[0], -1)
        if self._result is not None:
            raise RuntimeError("the expert Hessians were already read")
        if self._gate_up is None:
            # Both stacks on one device, decided together.
            device = self._device(n_experts, (d_model, inter), hidden.device)
            # Normal tensors even under inference mode, so result() can scale
            # them in place and GPTQ can damp them.
            with torch.inference_mode(False):
                self._gate_up = torch.zeros(
                    n_experts, d_model, d_model, device=device, dtype=torch.float32
                )
                self._down = torch.zeros(
                    n_experts, inter, inter, device=device, dtype=torch.float32
                )
            self._counts = torch.zeros(n_experts, dtype=torch.long)
        assert self._down is not None and self._counts is not None
        # Group the (token, slot) pairs by expert with one host sync: a token
        # reaches an expert at most once, and masked-out slots (index outside
        # 0..E-1) go to an extra bin that is dropped.
        expert_of = top_k_index.flatten()
        expert_of = torch.where(
            (expert_of >= 0) & (expert_of < n_experts), expert_of, n_experts
        )
        order = expert_of.argsort(stable=True)
        tokens_by_expert = (order // top_k_index.shape[1]).split(
            torch.bincount(expert_of, minlength=n_experts + 1).tolist()
        )
        for expert, tokens in enumerate(tokens_by_expert[:n_experts]):
            if not len(tokens):
                continue
            x = hidden[tokens]
            weight = gate_up[expert]
            gate, up = nn.functional.linear(x, weight.to(x.dtype)).chunk(2, dim=-1)
            z = (module.act_fn(gate) * up).float()
            x = x.float()
            self._counts[expert] += x.shape[0]
            if self._gate_up.device == x.device:
                self._gate_up[expert] += x.t() @ x
                self._down[expert] += z.t() @ z
            else:
                self._pending.append((expert, x.t() @ x, z.t() @ z))
                if len(self._pending) >= self._FLUSH:
                    self._flush()
        self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        assert self._gate_up is not None and self._down is not None
        experts = [e for e, _, _ in self._pending]
        for stack, part in ((self._gate_up, 1), (self._down, 2)):
            items = [item[part] for item in self._pending]
            if len({t.device for t in items}) > 1:
                # Calls from several devices: gather them where the stack is.
                items = [t.to(stack.device) for t in items]
            products = torch.stack(items)
            index = torch.tensor(experts, device=stack.device)
            stack.index_add_(0, index, products.to(stack.device))
        self._pending.clear()

    @staticmethod
    def _device(
        n_experts: int, widths: tuple[int, ...], device: torch.device
    ) -> torch.device:
        """Where one layer's expert Hessians go: *device*, or the host if short.

        Same rule as :class:`.HessianAccumulator`: the Hessians of every expert
        of both stacks leave the GPU together when they would take a large
        share of its free memory. The products are still computed on the GPU.
        """
        if device.type == "cuda":
            nbytes = n_experts * sum(w * w for w in widths) * 4
            free, _ = torch.cuda.mem_get_info(device)
            if nbytes * 2 > free * 0.3:
                return torch.device("cpu")
        return device

    def result(self) -> dict[str, Any]:
        """Return ``{"gate_up", "down"}`` Hessians stacked per expert and counts.

        An expert no token reached has a zero Hessian and a count of zero.
        """
        if self._gate_up is None:
            return {"gate_up": None, "down": None, "nsamples": None}
        if self._result is None:
            assert self._down is not None and self._counts is not None
            # Scaled in place: a second copy of every Hessian would double the
            # peak. Further updates are refused from here on.
            scale = (2.0 / self._counts.clamp(min=1).float()).to(self._gate_up.device)
            self._gate_up.mul_(scale.view(-1, 1, 1))
            self._down.mul_(scale.view(-1, 1, 1))
            self._result = {
                "gate_up": self._gate_up,
                "down": self._down,
                "nsamples": self._counts,
            }
        return self._result
