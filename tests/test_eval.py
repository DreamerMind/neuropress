"""Tests for neuropress.eval harness (no lm_eval install required)."""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from neuropress.eval.harness import (
    EvalComparison,
    _infer_device,
    _make_hflm,
    compare,
    evaluate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _BufferOnlyModel(nn.Module):
    """Simulates a fully kernel-lowered model: weights registered as
    buffers (like NPRMSNorm/QTensorTractVQ), zero nn.Parameters."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _EmptyModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _fake_result(
    tasks: list[str], metric: str = "acc", value: float = 0.5
) -> dict[str, Any]:
    return {"results": {t: {metric: value} for t in tasks}}


_PATCH_MAKE = "neuropress.eval.harness._make_hflm"
_PATCH_RUN = "neuropress.eval.harness._run_lm_eval"


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_builds_hflm_and_calls_lm_eval():
    model = _TinyModel()
    tokenizer = MagicMock()
    tasks = ["hellaswag", "arc_easy"]
    fake_hflm = MagicMock()
    fake_result = _fake_result(tasks)

    with (
        patch(_PATCH_MAKE, return_value=fake_hflm) as mock_make,
        patch(_PATCH_RUN, return_value=fake_result) as mock_run,
    ):
        result = evaluate(
            model, tokenizer, tasks, limit=10, num_fewshot=0, batch_size=4
        )

    mock_make.assert_called_once_with(model, tokenizer)
    mock_run.assert_called_once_with(
        fake_hflm, tasks, limit=10, num_fewshot=0, batch_size=4
    )
    assert result is fake_result


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_evals_before_quantize_fn():
    model = _TinyModel()
    tokenizer = MagicMock()
    tasks = ["hellaswag"]
    call_order: list[str] = []

    def fake_run(hflm, tasks, *, limit, num_fewshot, batch_size):
        call_order.append("eval")
        return _fake_result(tasks)

    quantize_fn = MagicMock(side_effect=lambda m: call_order.append("quant"))

    with (
        patch(_PATCH_MAKE, return_value=MagicMock()),
        patch(_PATCH_RUN, side_effect=fake_run),
    ):
        compare(model, tokenizer, tasks, quantize_fn)

    # First eval must precede the quantize call
    eval_indices = [i for i, v in enumerate(call_order) if v == "eval"]
    quant_index = call_order.index("quant")
    assert eval_indices[0] < quant_index


def test_compare_calls_quantize_fn():
    model = _TinyModel()
    tokenizer = MagicMock()
    tasks = ["hellaswag"]
    quantize_fn = MagicMock(return_value=model)

    with (
        patch(_PATCH_MAKE, return_value=MagicMock()),
        patch(_PATCH_RUN, return_value=_fake_result(tasks)),
    ):
        compare(model, tokenizer, tasks, quantize_fn)

    quantize_fn.assert_called_once_with(model)


def test_compare_evals_after_quantize_fn():
    model = _TinyModel()
    tokenizer = MagicMock()
    tasks = ["hellaswag"]
    call_order: list[str] = []

    def fake_run(hflm, tasks, *, limit, num_fewshot, batch_size):
        call_order.append("eval")
        return _fake_result(tasks)

    quantize_fn = MagicMock(side_effect=lambda m: call_order.append("quant"))

    with (
        patch(_PATCH_MAKE, return_value=MagicMock()),
        patch(_PATCH_RUN, side_effect=fake_run),
    ):
        compare(model, tokenizer, tasks, quantize_fn)

    # Second eval must come after the quantize call
    eval_indices = [i for i, v in enumerate(call_order) if v == "eval"]
    quant_index = call_order.index("quant")
    assert eval_indices[1] > quant_index


def test_compare_returns_eval_comparison():
    model = _TinyModel()
    tokenizer = MagicMock()
    tasks = ["hellaswag"]
    baseline_result = _fake_result(tasks, value=0.6)
    quantized_result = _fake_result(tasks, value=0.55)
    results_iter = iter([baseline_result, quantized_result])

    with (
        patch(_PATCH_MAKE, return_value=MagicMock()),
        patch(_PATCH_RUN, side_effect=lambda *a, **kw: next(results_iter)),
    ):
        cmp = compare(model, tokenizer, tasks, lambda m: None)

    assert isinstance(cmp, EvalComparison)
    assert cmp.baseline == baseline_result
    assert cmp.quantized == quantized_result


# ---------------------------------------------------------------------------
# EvalComparison
# ---------------------------------------------------------------------------


def test_eval_comparison_delta():
    baseline = {"results": {"hellaswag": {"acc": 0.6, "acc_norm": 0.65}}}
    quantized = {"results": {"hellaswag": {"acc": 0.55, "acc_norm": 0.63}}}
    cmp = EvalComparison(baseline=baseline, quantized=quantized)

    delta = cmp.delta
    assert "hellaswag" in delta
    assert abs(delta["hellaswag"]["acc"] - (-0.05)) < 1e-6
    assert abs(delta["hellaswag"]["acc_norm"] - (-0.02)) < 1e-6


def test_eval_comparison_delta_skips_stderr():
    baseline = {"results": {"hellaswag": {"acc": 0.6, "acc_stderr": 0.01}}}
    quantized = {"results": {"hellaswag": {"acc": 0.55, "acc_stderr": 0.009}}}
    cmp = EvalComparison(baseline=baseline, quantized=quantized)

    delta = cmp.delta
    task_delta = delta.get("hellaswag", {})
    assert "acc_stderr" not in task_delta
    assert "acc" in task_delta


def test_eval_comparison_str_contains_task_names():
    tasks = ["hellaswag", "arc_easy"]
    baseline = {"results": {t: {"acc": 0.5} for t in tasks}}
    quantized = {"results": {t: {"acc": 0.48} for t in tasks}}
    cmp = EvalComparison(baseline=baseline, quantized=quantized)

    text = str(cmp)
    for task in tasks:
        assert task in text


def test_infer_device_from_parameter():
    model = _TinyModel()
    assert _infer_device(model) == next(model.parameters()).device


def test_infer_device_falls_back_to_buffer():
    """Regression test: a fully kernel-lowered model (weights registered as
    buffers, zero nn.Parameters) must not raise StopIteration -- this
    crashed _make_hflm on historical kernel-lowering paths and blocked a
    git bisect until worked around locally."""
    model = _BufferOnlyModel()
    assert next(model.parameters(), None) is None
    assert _infer_device(model) == next(model.buffers()).device


def test_infer_device_defaults_to_cpu_with_no_state():
    model = _EmptyModel()
    assert _infer_device(model) == torch.device("cpu")


def test_missing_lm_eval_raises_import_error():
    model = _TinyModel()
    tokenizer = MagicMock()

    blocked = {
        "lm_eval": None,
        "lm_eval.models": None,
        "lm_eval.models.huggingface": None,
    }
    with patch.dict(sys.modules, blocked):
        with pytest.raises(ImportError, match=r"neuropress\[eval\]"):
            _make_hflm(model, tokenizer)
