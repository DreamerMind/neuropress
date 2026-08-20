"""lm-evaluation-harness integration for neuropress."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from pydantic import BaseModel
from torch import nn

# ---------------------------------------------------------------------------
# Task name utilities
# ---------------------------------------------------------------------------


def normalize_tasks(tasks: list[str]) -> list[str]:
    """Split and flatten comma-separated task tokens.

    Accepts both space-separated (``["hellaswag", "arc_easy"]``) and
    comma-separated (``["hellaswag,arc_easy"]``) formats — or any mix —
    so users can write ``--tasks hellaswag,arc_easy`` or
    ``--tasks hellaswag arc_easy`` interchangeably.

    Args:
        tasks: Raw token list from argparse ``nargs="+"``.

    Returns:
        Flat list of individual task names with whitespace stripped.
    """
    result = []
    for token in tasks:
        result.extend(t.strip() for t in token.split(",") if t.strip())
    return result


def validate_tasks(tasks: list[str]) -> list[str]:
    """Return task names not found in lm_eval's task registry.

    Loads :class:`lm_eval.tasks.TaskManager` to enumerate all known tasks.
    If ``lm_eval`` is not installed, returns an empty list (no validation).

    Args:
        tasks: Normalised list of lm-eval task names to check.

    Returns:
        Sub-list of *tasks* that are unknown to lm_eval, empty if all valid.
    """
    try:
        from lm_eval.tasks import TaskManager  # type: ignore[import]

        tm = TaskManager()
        return [t for t in tasks if t not in tm.task_index]
    except ImportError:
        return []


def _format_bytes(n_bytes: int) -> str:
    """Format a byte count as a human-readable string using Mo/Go units.

    Args:
        n_bytes: Number of bytes.

    Returns:
        String such as ``"2.16 Go"`` or ``"540 Mo"``.
    """
    if n_bytes >= 1024**3:
        return f"{n_bytes / 1024**3:.2f} Go"
    if n_bytes >= 1024**2:
        return f"{n_bytes / 1024**2:.0f} Mo"
    return f"{n_bytes / 1024:.1f} Ko"


def _format_params(n: int) -> str:
    """Format a parameter count with M/B suffix.

    Args:
        n: Number of parameters.

    Returns:
        String such as ``"270 M"`` or ``"7.00 B"``.
    """
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f} M"
    return f"{n / 1e3:.0f} K"


class EvalComparison(BaseModel):
    """Comparison of float vs quantized model evaluation results."""

    baseline: dict[str, Any]
    quantized: dict[str, Any]
    baseline_time_s: float = 0.0
    quantized_time_s: float = 0.0
    n_params: int = 0
    baseline_bytes: int = 0
    quantized_bytes: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def delta(self) -> dict[str, dict[str, float]]:
        """Compute quantized minus baseline per task/metric.

        Returns:
            Nested dict mapping task name to metric to float delta.
            Fields ending in ``_stderr`` are excluded.
        """
        result: dict[str, dict[str, float]] = {}
        baseline_results = self.baseline.get("results", {})
        quantized_results = self.quantized.get("results", {})
        for task, base_metrics in baseline_results.items():
            quant_metrics = quantized_results.get(task, {})
            task_delta: dict[str, float] = {}
            for metric, base_val in base_metrics.items():
                if metric.endswith("_stderr") or not isinstance(base_val, int | float):
                    continue
                quant_val = quant_metrics.get(metric)
                if isinstance(quant_val, int | float):
                    task_delta[metric] = float(quant_val) - float(base_val)
            if task_delta:
                result[task] = task_delta
        return result

    def __str__(self) -> str:
        """Render a tabular summary: task | metric | baseline | quantized | delta."""
        col_task = 30
        col_metric = 25
        col_num = 10
        width = col_task + col_metric + col_num * 3 + 4
        separator = "-" * width

        lines: list[str] = []

        # Model info header (only when size/timing data is available)
        if self.n_params or self.baseline_time_s or self.quantized_time_s:
            lines.append(separator)
            if self.n_params:
                lines.append(f"  Parameters  {_format_params(self.n_params)}")
            if self.baseline_bytes:
                lines.append(
                    f"  Model size  "
                    f"baseline {_format_bytes(self.baseline_bytes)}"
                    + (
                        f"  →  quantized {_format_bytes(self.quantized_bytes)}"
                        if self.quantized_bytes
                        else ""
                    )
                )
            if self.baseline_time_s and self.quantized_time_s:
                rtfx = (
                    self.baseline_time_s / self.quantized_time_s
                    if self.quantized_time_s > 0
                    else float("nan")
                )
                lines.append(
                    f"  Eval time   "
                    f"baseline {self.baseline_time_s:.1f}s  |  "
                    f"quantized {self.quantized_time_s:.1f}s  |  "
                    f"RTFx {rtfx:.2f}x"
                    + (
                        "  (quantized is faster)"
                        if rtfx > 1
                        else "  (quantized is slower)"
                        if rtfx < 1
                        else ""
                    )
                )
            lines.append(separator)

        header = (
            f"{'Task':<{col_task}} {'Metric':<{col_metric}}"
            f" {'Baseline':>{col_num}} {'Quantized':>{col_num}} {'Delta':>{col_num}}"
        )
        lines += [header, separator]

        baseline_results = self.baseline.get("results", {})
        quantized_results = self.quantized.get("results", {})
        deltas = self.delta

        for task in sorted(baseline_results):
            base_metrics = baseline_results[task]
            quant_metrics = quantized_results.get(task, {})
            task_deltas = deltas.get(task, {})
            for metric in sorted(base_metrics):
                base_val = base_metrics[metric]
                if metric.endswith("_stderr") or not isinstance(base_val, int | float):
                    continue
                quant_val = quant_metrics.get(metric, float("nan"))
                delta = task_deltas.get(metric, float("nan"))
                lines.append(
                    f"{task:<{col_task}} {metric:<{col_metric}}"
                    f" {float(base_val):>{col_num}.4f}"
                    f" {float(quant_val):>{col_num}.4f}"
                    f" {delta:>+{col_num}.4f}"
                )
        return "\n".join(lines)


def _infer_device(model: nn.Module) -> torch.device:
    """Infer a model's device from its first parameter or buffer.

    Kernel-lowered modules (e.g. ``NPRMSNorm``, ``QTensorTractVQ``) register
    their weights as buffers rather than parameters, so a fully lowered
    model can end up with zero ``nn.Parameter``s -- ``next(model.parameters())``
    then raises ``StopIteration``. Buffers are checked as a fallback since
    they hold the real device in that case; a model with neither defaults
    to CPU.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        pass
    try:
        return next(model.buffers()).device
    except StopIteration:
        return torch.device("cpu")


def _make_hflm(model: nn.Module, tokenizer: Any) -> Any:
    """Wrap a model and tokenizer in an HFLM instance.

    Device is inferred from the first model parameter or buffer.

    Args:
        model: PyTorch model to wrap.
        tokenizer: HuggingFace tokenizer.

    Returns:
        An ``lm_eval.models.huggingface.HFLM`` instance.

    Raises:
        ImportError: If ``lm-eval`` is not installed.
    """
    try:
        from lm_eval.models.huggingface import HFLM  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "lm-evaluation-harness is required for neuropress.eval. "
            "Install it with: pip install 'neuropress[eval]'"
        ) from exc
    device = _infer_device(model)
    return HFLM(pretrained=model, tokenizer=tokenizer, device=str(device))


def _run_lm_eval(
    hflm: Any,
    tasks: list[str],
    *,
    limit: int | float | None,
    num_fewshot: int | None,
    batch_size: int | str,
) -> dict[str, Any]:
    """Thin wrapper around lm_eval.simple_evaluate.

    Args:
        hflm: An HFLM model instance.
        tasks: List of lm-eval task names.
        limit: Maximum number of examples per task.
        num_fewshot: Number of few-shot examples.
        batch_size: Batch size for evaluation.

    Returns:
        Raw result dict from ``lm_eval.simple_evaluate``.
    """
    from lm_eval import simple_evaluate  # type: ignore[import]

    return simple_evaluate(  # type: ignore[no-any-return]
        model=hflm,
        tasks=tasks,
        limit=limit,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )


def evaluate(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    *,
    limit: int | float | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str = 1,
) -> dict[str, Any]:
    """Run lm-eval tasks on a model and return the raw result dict.

    Args:
        model: PyTorch model to evaluate.
        tokenizer: HuggingFace tokenizer.
        tasks: List of lm-eval task names.
        limit: Maximum number of examples per task (``None`` = all).
        num_fewshot: Number of few-shot examples (``None`` = task default).
        batch_size: Batch size or ``"auto"``.

    Returns:
        Raw result dict from ``lm_eval.simple_evaluate``.
    """
    hflm = _make_hflm(model, tokenizer)
    return _run_lm_eval(
        hflm,
        tasks,
        limit=limit,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )


def compare(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    quantize_fn: Callable[[nn.Module], nn.Module],
    *,
    limit: int | float | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str = 1,
) -> EvalComparison:
    """Evaluate a float model, apply quantize_fn in-place, then evaluate again.

    The model is in a quantized state after this call returns.

    Args:
        model: PyTorch model to evaluate (mutated by ``quantize_fn``).
        tokenizer: HuggingFace tokenizer.
        tasks: List of lm-eval task names.
        quantize_fn: Callable that quantizes the model (may mutate in-place)
            and returns it.
        limit: Maximum number of examples per task (``None`` = all).
        num_fewshot: Number of few-shot examples (``None`` = task default).
        batch_size: Batch size or ``"auto"``.

    Returns:
        :class:`EvalComparison` holding both evaluation result dicts.
    """
    eval_kwargs: dict[str, Any] = {
        "limit": limit,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
    }
    baseline = evaluate(model, tokenizer, tasks, **eval_kwargs)
    quantize_fn(model)
    quantized = evaluate(model, tokenizer, tasks, **eval_kwargs)
    return EvalComparison(baseline=baseline, quantized=quantized)
