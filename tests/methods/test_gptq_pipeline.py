"""Pipeline-level tests for the GPTQ path (no network, no model download).

``tests/methods/test_gptq.py`` covers the algorithm in isolation and the NNEF
integration test covers the full stack but is gated behind ``RUN_INTEGRATION``.
Neither checks that driving :class:`QuantizationPipeline` with a GPTQ recipe
actually produces a *better* model than naive rounding, which is the whole point
of GPTQ.  These tests close that gap cheaply.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from neuropress.adapter import WeightAssignmentAdapter
from neuropress.estimators import GPTQEstimator
from neuropress.pipeline import QuantizationPipeline
from neuropress.recipe import QuantizationRecipe

D = 128
SEQ = 16
N_SAMPLES = 8


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(D, D, bias=False)
        self.fc2 = nn.Linear(D, D, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(input_ids)))


def _correlated_batches(n: int = N_SAMPLES) -> list[dict[str, torch.Tensor]]:
    """Calibration batches with off-diagonal covariance for GPTQ to exploit."""
    mix = torch.randn(D, D) * 0.15 + torch.eye(D)
    return [{"input_ids": torch.randn(1, SEQ, D) @ mix} for _ in range(n)]


def _rtn_quantize(w: torch.Tensor, group_size: int, nbits: int) -> torch.Tensor:
    """Per-group symmetric round-to-nearest quantize/dequantize baseline."""
    out, k = w.shape
    g = w.reshape(out, k // group_size, group_size)
    qmax = 2 ** (nbits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax)
    return (q * scale).reshape(out, k)


def _capture_layer_inputs(model: _Net, batches) -> dict[str, torch.Tensor]:
    """Record each linear's real input from the float model, as GPTQ sees it."""
    captured: dict[str, list[torch.Tensor]] = {"fc1": [], "fc2": []}

    def make_hook(name: str):
        def hook(_mod, args, _out):
            captured[name].append(args[0].detach().reshape(-1, D).clone())

        return hook

    handles = [
        model.fc1.register_forward_hook(make_hook("fc1")),
        model.fc2.register_forward_hook(make_hook("fc2")),
    ]
    try:
        with torch.no_grad():
            for batch in batches:
                model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    return {k: torch.cat(v, dim=0) for k, v in captured.items()}


def test_gptq_pipeline_beats_round_to_nearest():
    """GPTQ must reconstruct each layer's output better than RTN at equal bits."""
    torch.manual_seed(0)
    group_size, nbits = 32, 4

    model = _Net().eval()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    batches = _correlated_batches()
    layer_inputs = _capture_layer_inputs(model, batches)

    recipe = QuantizationRecipe(
        param_estimator=GPTQEstimator(group_size=group_size, nbits=nbits),
        adapter=WeightAssignmentAdapter(),
    )
    QuantizationPipeline(recipe).run(model, iter(batches))

    updated = dict(model.named_parameters())
    for name, w_orig in original.items():
        w_gptq = updated[name].detach()
        assert not torch.equal(w_orig, w_gptq), f"{name} was not corrected in place"

        x = layer_inputs[name.split(".")[0]]
        err_gptq = ((x @ (w_gptq - w_orig).T) ** 2).mean().item()
        err_rtn = (x @ (_rtn_quantize(w_orig, group_size, nbits) - w_orig).T) ** 2
        err_rtn = err_rtn.mean().item()
        assert err_gptq < err_rtn, (
            f"{name}: GPTQ error {err_gptq:.3e} should beat RTN {err_rtn:.3e}"
        )


def test_gptq_pipeline_preserves_finite_outputs():
    """A GPTQ-corrected model must stay finite and closely track the float model."""
    torch.manual_seed(0)
    model = _Net().eval()
    reference = copy.deepcopy(model)
    batches = _correlated_batches()

    mix = torch.randn(D, D) * 0.15 + torch.eye(D)
    held_out = torch.randn(4, SEQ, D) @ mix
    with torch.no_grad():
        y_float = reference(held_out)

    recipe = QuantizationRecipe(
        param_estimator=GPTQEstimator(group_size=32, nbits=4),
        adapter=WeightAssignmentAdapter(),
    )
    quantized = QuantizationPipeline(recipe).run(model, iter(batches))

    with torch.no_grad():
        y_quant = quantized(held_out)

    assert torch.isfinite(y_quant).all(), "quantized model produced non-finite outputs"
    cos = torch.nn.functional.cosine_similarity(
        y_quant.flatten(), y_float.flatten(), dim=0
    ).item()
    assert cos > 0.95, f"output cosine similarity too low: {cos:.4f}"


@pytest.mark.parametrize(
    "key",
    [
        "open_text_c4_gptq_q4_0",
        "open_text_c4_gptq_q4_0_32samples",
        "open_text_c4_gptq_q4_0_256samples",
    ],
)
def test_gptq_registry_entries_are_wired(key: str):
    """Each public GPTQ registry entry must carry a usable GPTQ recipe."""
    from neuropress.t2n import NEUROPRESS_COMPRESSION

    assert key in NEUROPRESS_COMPRESSION
    recipe = NEUROPRESS_COMPRESSION[key].keywords["recipe"]
    estimator = recipe.param_estimator

    assert isinstance(estimator, GPTQEstimator)
    assert estimator.nbits == 4
    assert estimator.group_size == 32
    # Q4_0 entries must pack real quantized tensors, not leave corrected floats.
    assert estimator.weight_transform is not None
    assert recipe.calibration.n_samples > 0
