"""Pydantic models for per-step run timing and run_config.json."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field

from neuropress.meta import RunMeta


def _utcnow() -> datetime:
    return datetime.now(UTC)


class StepTiming(BaseModel):
    """Wall-clock span for a single pipeline step."""

    start: datetime
    stop: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_s(self) -> float:
        """Elapsed wall-clock seconds between :attr:`start` and :attr:`stop`."""
        return round((self.stop - self.start).total_seconds(), 3)

    @classmethod
    def begin(cls) -> tuple[datetime, type[StepTiming]]:
        """Return ``(start_dt, cls)``; call :meth:`close` with *start_dt* to finish."""
        return _utcnow(), cls

    @classmethod
    def close(cls, start: datetime, **extra: object) -> StepTiming:
        """Build the finished timing from *start*, stopping at the current time.

        Args:
            start: Timestamp returned by :meth:`begin`.
            **extra: Additional fields for subclasses (e.g. a task name).

        Returns:
            The completed timing record.
        """
        return cls(start=start, stop=_utcnow(), **extra)  # type: ignore[call-arg]

    model_config = {"populate_by_name": True}


class TaskTiming(StepTiming):
    """Timing for a single lm-eval task within a phase."""

    cached: bool = False


class EvalPhaseTiming(StepTiming):
    """Timing for a full eval phase (baseline or quantized) with per-task breakdown."""

    per_task: dict[str, TaskTiming] = Field(default_factory=dict)
    skipped: bool = False
    source: str | None = None


class QuantizationTiming(StepTiming):
    """Timing for the calibration + quantization step."""

    resumed_from: str | None = None
    error: str | None = None
    stages: dict[str, float] = Field(default_factory=dict)
    """Per-stage wall-clock totals in seconds, keyed by span name.

    Populated from an :class:`~neuropress.quantization.spans.AccumulatingSpanCollector`
    when the caller passes one to :func:`~neuropress.t2n.registry.run_quantization_pipeline`.
    Empty when no span collector is provided.
    """


class ModelSaveTiming(StepTiming):
    """Timing for the model-save step; captures any exception that aborted it."""

    error: str | None = None


class RunTiming(BaseModel):
    """Complete per-step timing for one neuropress-eval compare run."""

    model_load: StepTiming | None = None
    baseline_eval: EvalPhaseTiming | None = None
    quantization: QuantizationTiming | None = None
    model_save: ModelSaveTiming | None = None
    quantized_eval: EvalPhaseTiming | None = None


class RunConfig(BaseModel):
    """Snapshot of the CLI invocation written to run_config.json at job start."""

    model: str
    compression: str
    tasks: list[str]
    device: str | None = None
    limit: float | None = None
    num_fewshot: int | None = None
    batch_size: int | str | None = None
    skip_baseline: bool = False
    quantization_config: dict[str, Any] | None = None
    neuropress: RunMeta


class EvalReport(BaseModel):
    """Top-level structure of eval_report.json produced by neuropress-eval compare."""

    meta: RunMeta
    timing: RunTiming
    baseline: dict[str, Any] | None = None
    quantized: dict[str, Any] = Field(default_factory=dict)
    delta: dict[str, dict[str, float]] | None = None
    n_params: int = 0
    baseline_bytes: int = 0
    quantized_bytes: int = 0
    baseline_time_s: float = 0.0
    quantized_time_s: float = 0.0
