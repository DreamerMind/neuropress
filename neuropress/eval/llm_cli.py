r"""Command-line interface for LLM evaluation using lm-evaluation-harness.

Entry point: ``neuropress-eval``

Sub-commands:

``eval``
    Load a HuggingFace CausalLM, run lm-evaluation-harness tasks, and print
    a results table.  Optionally save raw output to JSON.

``compare``
    Evaluate the float model, apply a registered neuropress compression
    method (from :data:`neuropress.t2n.NEUROPRESS_COMPRESSION`), evaluate
    again, and print a baseline / quantized / delta table.

``list``
    Print the keys of :data:`neuropress.t2n.NEUROPRESS_COMPRESSION`; works
    with no optional extras installed.

Example usage::

    # Float eval on HellaSwag (first 100 examples)
    neuropress-eval eval meta-llama/Llama-2-7b-hf \
        --tasks hellaswag --limit 100 --device cuda

    # Compare float vs GPTQ-Q4_0 quantized, save JSON report
    neuropress-eval compare meta-llama/Llama-2-7b-hf \
        --tasks hellaswag arc_easy \
        --compression open_text_c4_gptq_q4_0 \
        --device cuda --output report.json

    # List available compression methods
    neuropress-eval list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from torch import nn

if TYPE_CHECKING:
    from neuropress.eval.timing import TaskTiming


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _die(msg: str) -> NoReturn:
    """Print *msg* to stderr and exit with code 1.

    Args:
        msg: Error message to display.
    """
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _batch_size_type(val: str) -> int | str:
    """Parse *val* as a batch-size argument (positive int or ``"auto"``).

    Args:
        val: Raw string from argparse.

    Returns:
        Parsed batch size as ``int`` or the literal string ``"auto"``.

    Raises:
        argparse.ArgumentTypeError: If *val* is neither ``"auto"`` nor a
            positive integer.
    """
    if val == "auto":
        return val
    try:
        n = int(val)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"batch-size must be 'auto' or a positive integer, got {val!r}"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"batch-size must be a positive integer, got {n}"
        )
    return n


def _safe_json(obj: Any) -> Any:
    """Recursively replace non-JSON-serializable objects with their ``repr``.

    lm_eval result dicts contain config objects (functions, classes, …) in
    their ``"config"`` sub-dict that are not JSON-serializable.  This walk
    converts them to strings so ``json.dumps`` never raises.

    Args:
        obj: Arbitrary Python object.

    Returns:
        A JSON-safe copy of *obj*.
    """
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, int | float | str | bool) or obj is None:
        return obj
    return repr(obj)


def _model_bytes(model: nn.Module) -> int:
    """Return the total number of bytes used by all model parameters and buffers.

    For plain tensors this is ``numel * element_size``.  For registered
    QTensor types (e.g. ``QTensorTractScaleOnly`` after Q4_0 quantization)
    the size is computed from the actual packed storage tensors returned by
    the serde framework, so the packed representation (e.g. 2 values per
    byte for 4-bit) is correctly accounted for.

    Buffers are included so that kernel-lowered models (NP* modules that
    store quantized data as buffers) are measured correctly.

    Args:
        model: PyTorch model to measure.

    Returns:
        Total on-disk/in-memory size in bytes across all parameters and
        buffers.
    """
    try:
        from neuropress.t2n.qtensors import _SERDE_REGISTRY
    except ImportError:
        _SERDE_REGISTRY = {}

    total = 0
    seen: set[int] = set()
    for p in model.parameters():
        tid = id(p)
        if tid in seen:
            continue
        seen.add(tid)
        serde = _SERDE_REGISTRY.get(type(p).__name__)
        if serde is not None:
            for t in serde.to_tensors(p).values():
                total += t.numel() * t.element_size()
        else:
            total += p.numel() * p.element_size()

    for buf in model.buffers():
        if buf is not None:
            tid = id(buf)
            if tid in seen:
                continue
            seen.add(tid)
            total += buf.numel() * buf.element_size()
    return total


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model_and_tokenizer(
    model_id: str,
    *,
    device: str,
    dtype: str,
) -> tuple[nn.Module, Any]:
    """Load a HuggingFace CausalLM and its tokenizer.

    Args:
        model_id: HuggingFace model ID or local path.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``).
        dtype: One of ``"auto"``, ``"float16"``, ``"bfloat16"``,
            ``"float32"``.  ``"auto"`` defers to the model config.

    Returns:
        Tuple ``(model, tokenizer)`` — model placed on *device* in eval mode.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        _die(
            "transformers is required for model loading. Install it with:\n"
            "  pip install 'neuropress[eval]'"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    _DTYPES = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_dtype = _DTYPES.get(dtype, "auto")

    # transformers >= 4.50 renamed torch_dtype → dtype; try new API first.
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=load_dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=load_dtype)
    model.to(device)

    # Qwen3.5 (and similar hybrid models) initialize Conv1d layers in float16
    # for the causal-conv1d fast path, but the torch slow path doesn't cast
    # the input to match.  Register hooks that cast input→weight dtype and
    # output back, so this is immune to any dtype juggling in the pipeline.
    def _pre(mod, args):
        x = args[0]
        if x.dtype != mod.weight.dtype:
            return (x.to(mod.weight.dtype),)

    def _post(mod, args, out):
        orig = args[0].dtype
        return out.to(orig) if out.dtype != orig else out

    for m in model.modules():
        if isinstance(m, torch.nn.Conv1d):
            m.register_forward_pre_hook(_pre)
            m.register_forward_hook(_post)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _resolve_cache_dir(
    explicit: str | None,
    no_cache: bool,
    model_id: str,
) -> Path | None:
    """Return the cache dir to use, or None if caching is disabled.

    Priority: explicit > no_cache flag > auto-derive from $NEUROPRESS_DATA_DIR.
    Auto-derive is suppressed when $NEUROPRESS_DATA_DIR is not set so that
    quick local runs don't scatter cache dirs under ./data.
    """
    if no_cache:
        return None
    if explicit is not None:
        return Path(explicit)
    data_dir = os.environ.get("NEUROPRESS_DATA_DIR")
    if not data_dir:
        return None
    safe_model = model_id.replace("/", "__")
    return Path(data_dir) / "cache" / safe_model


def _build_quantize_fn(
    method: str,
    tokenizer: Any,
    cache_dir: str | None = None,
) -> Any:
    """Return a ``quantize_fn`` closure that quantizes and kernel-lowers.

    The closure calls the registry entry which runs the full pipeline
    (calibration → weight assignment → manifest → kernel lowering).  The
    model is returned in the fast inference state;
    :func:`~neuropress.t2n.model_io.save_quantized_model` auto-unlowers
    before saving and re-lowers afterwards.

    Args:
        method: Key in :data:`neuropress.t2n.NEUROPRESS_COMPRESSION`.
        tokenizer: HuggingFace tokenizer forwarded to the compression function.
        cache_dir: Optional path to the quantization stage cache directory.
            Forwarded to :func:`~neuropress.t2n.registry.run_quantization_pipeline`.

    Returns:
        Callable ``(model) -> model``.
    """
    from neuropress.t2n import NEUROPRESS_COMPRESSION

    if method not in NEUROPRESS_COMPRESSION:
        available = "\n  ".join(sorted(NEUROPRESS_COMPRESSION))
        _die(
            f"Unknown compression method {method!r}.\n"
            f"Available methods:\n  {available}\n"
            f"Run `neuropress-eval list` to see them."
        )

    compress_fn = NEUROPRESS_COMPRESSION[method]
    extra: dict[str, Any] = {}
    if cache_dir is not None:
        extra["cache_dir"] = cache_dir

    def quantize_fn(model: nn.Module) -> nn.Module:
        return compress_fn(model, tokenizer=tokenizer, **extra)

    return quantize_fn


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------


def _print_eval_results(result: dict[str, Any]) -> None:
    """Print a formatted results table from a lm_eval result dict.

    Args:
        result: Raw dict from ``lm_eval.simple_evaluate``.
    """
    results = result.get("results", {})
    if not results:
        print("(no results)")
        return

    col_task, col_metric, col_val = 30, 25, 12
    sep = "-" * (col_task + col_metric + col_val + 2)
    print(f"{'Task':<{col_task}} {'Metric':<{col_metric}} {'Value':>{col_val}}")
    print(sep)
    for task in sorted(results):
        for metric in sorted(results[task]):
            val = results[task][metric]
            if metric.endswith("_stderr") or not isinstance(val, int | float):
                continue
            print(
                f"{task:<{col_task}} {metric:<{col_metric}} {float(val):>{col_val}.4f}"
            )


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_size(args: argparse.Namespace) -> None:
    """Estimate quantized model size without running compression or calibration.

    Args:
        args: Parsed CLI namespace.
    """
    from rich.console import Console

    from neuropress.eval.size_estimate import (
        _DTYPE_BYTES,
        collect_estimates,
        format_size_report,
        load_model_meta,
    )
    from neuropress.t2n import NEUROPRESS_COMPRESSION

    if args.compression not in NEUROPRESS_COMPRESSION:
        available = "\n  ".join(sorted(NEUROPRESS_COMPRESSION))
        _die(
            f"Unknown compression method {args.compression!r}.\n"
            f"Available methods:\n  {available}\n"
            f"Run `neuropress-eval list` to see them."
        )

    compress_fn = NEUROPRESS_COMPRESSION[args.compression]
    recipe = compress_fn.keywords.get("recipe")
    if recipe is None:
        _die(
            f"Compression method {args.compression!r} carries no pre-built "
            "recipe and cannot be analysed statically."
        )

    console = Console()
    console.print(f"Loading model config for [bold]{args.model}[/bold] …")
    try:
        model, inferred_dtype_bytes = load_model_meta(args.model)
    except ImportError as exc:
        _die(str(exc))

    dtype_bytes = _DTYPE_BYTES.get(args.dtype, inferred_dtype_bytes)

    estimates, other_bytes = collect_estimates(
        model, recipe.param_estimator, dtype_bytes=dtype_bytes
    )
    report = format_size_report(
        args.model,
        args.compression,
        estimates,
        other_bytes,
        dtype_bytes=dtype_bytes,
    )
    console.print()
    console.print(report)


def _cmd_list(args: argparse.Namespace) -> None:
    """Print available compression methods from the neuropress registry.

    Args:
        args: Parsed CLI namespace (unused).
    """
    from neuropress.t2n import NEUROPRESS_COMPRESSION

    print("Available compression methods:")
    for key in sorted(NEUROPRESS_COMPRESSION):
        print(f"  {key}")


def _cmd_eval(args: argparse.Namespace) -> None:
    """Load a model and evaluate it on lm-eval tasks.

    Args:
        args: Parsed CLI namespace.
    """
    from neuropress.eval.harness import evaluate, normalize_tasks, validate_tasks

    args.tasks = normalize_tasks(args.tasks)
    unknown = validate_tasks(args.tasks)
    if unknown:
        _die(f"Unknown lm-eval task(s): {', '.join(unknown)}")

    model, tokenizer = _load_model_and_tokenizer(
        args.model, device=args.device, dtype=args.dtype
    )
    result = evaluate(
        model,
        tokenizer,
        args.tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    _print_eval_results(result)
    if args.output:
        from neuropress.meta import RunMeta

        data = {"meta": RunMeta.collect().model_dump(mode="json"), **_safe_json(result)}
        Path(args.output).write_text(json.dumps(data, indent=2))
        print(f"\nResults saved to {args.output}")


def _load_partial(partial_dir: Path, prefix: str, task: str) -> dict[str, Any] | None:
    """Load a per-task partial result from *partial_dir* if it exists.

    Args:
        partial_dir: Directory containing partial JSON files.
        prefix: ``"baseline"`` or ``"quantized"``.
        task: lm-eval task name.

    Returns:
        Parsed JSON dict, or ``None`` if the file does not exist.
    """
    path = partial_dir / f"{prefix}_{task}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _save_partial(
    partial_dir: Path, prefix: str, task: str, result: dict[str, Any]
) -> None:
    """Write a per-task partial result to *partial_dir*.

    Args:
        partial_dir: Directory to write to (created if needed).
        prefix: ``"baseline"`` or ``"quantized"``.
        task: lm-eval task name.
        result: Raw lm_eval result dict for this single task.
    """
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = partial_dir / f"{prefix}_{task}.json"
    path.write_text(json.dumps(_safe_json(result), indent=2))


def _merge_results(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-task lm_eval result dicts into a single combined result.

    Args:
        per_task: List of individual ``lm_eval.simple_evaluate`` result dicts.

    Returns:
        Combined result dict with merged ``"results"`` and ``"configs"``.
    """
    merged: dict[str, Any] = {"results": {}, "configs": {}}
    for r in per_task:
        merged["results"].update(r.get("results", {}))
        merged["configs"].update(r.get("configs", {}))
    return merged


def _evaluate_per_task(
    model: nn.Module,
    tokenizer: Any,
    tasks: list[str],
    eval_kwargs: dict[str, Any],
    *,
    partial_dir: Path | None,
    prefix: str,
    console: Any,
) -> tuple[dict[str, Any], float, dict[str, TaskTiming]]:
    """Run evaluation one task at a time, saving partial results.

    If *partial_dir* is set, completed tasks are skipped and results
    are loaded from disk.  New results are written after each task.

    Args:
        model: PyTorch model to evaluate.
        tokenizer: HuggingFace tokenizer.
        tasks: List of lm-eval task names.
        eval_kwargs: Keyword arguments for :func:`evaluate`.
        partial_dir: Directory for per-task JSON checkpoints, or ``None``.
        prefix: ``"baseline"`` or ``"quantized"``.
        console: Rich console for status output.

    Returns:
        Tuple of ``(merged_result, elapsed_seconds, per_task_timing)`` where
        *per_task_timing* maps task name to a :class:`~neuropress.eval.timing.TaskTiming`.
    """
    from neuropress.eval.harness import evaluate
    from neuropress.eval.timing import TaskTiming

    per_task: list[dict[str, Any]] = []
    total_time = 0.0
    per_task_timing: dict[str, TaskTiming] = {}

    for i, task in enumerate(tasks, 1):
        cached = _load_partial(partial_dir, prefix, task) if partial_dir else None
        if cached is not None:
            console.print(
                f"  [{prefix}] ({i}/{len(tasks)}) [dim]{task} — loaded from cache[/dim]"
            )
            per_task.append(cached)
            t = datetime.now(UTC)
            per_task_timing[task] = TaskTiming(start=t, stop=t, cached=True)
            continue

        console.print(f"  [{prefix}] ({i}/{len(tasks)}) {task}")
        t_start = datetime.now(UTC)
        t0 = time.perf_counter()
        result = evaluate(model, tokenizer, [task], **eval_kwargs)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        per_task_timing[task] = TaskTiming.close(t_start)
        per_task.append(result)
        if partial_dir is not None:
            _save_partial(partial_dir, prefix, task, result)
            console.print(
                f"  [{prefix}] ({i}/{len(tasks)}) "
                f"[green]{task} done[/green] ({elapsed:.1f}s) — saved checkpoint"
            )

    return _merge_results(per_task), total_time, per_task_timing


def _write_run_config(args: argparse.Namespace) -> None:
    """Write run_config.json next to the --output file before work begins.

    Contains the model name, planned tasks, full quantization recipe, and the
    neuropress git commit so any result file can be traced back to exactly what
    ran.  Written immediately so the file exists even if the run crashes.
    """
    from neuropress.eval.timing import RunConfig
    from neuropress.meta import RunMeta
    from neuropress.t2n import NEUROPRESS_COMPRESSION

    compress_fn = NEUROPRESS_COMPRESSION.get(args.compression)
    quant_config: dict[str, Any] | None = None
    if compress_fn is not None and hasattr(compress_fn, "__config__"):
        cfg = compress_fn.__config__
        # Serialize each component via its concrete subclass so discriminated-
        # union fields (estimator, adapter) include all subclass parameters,
        # not just the base-class 'type' discriminator.
        quant_config = {
            "calibration": json.loads(cfg.calibration.model_dump_json()),
            "estimator": json.loads(cfg.estimator.model_dump_json()),
            "adapter": json.loads(cfg.adapter.model_dump_json()),
            "kernel_lowering": (
                json.loads(cfg.kernel_lowering.model_dump_json())
                if cfg.kernel_lowering
                else None
            ),
        }

    run_config = RunConfig(
        model=args.model,
        compression=args.compression,
        tasks=list(args.tasks),
        device=getattr(args, "device", None),
        limit=getattr(args, "limit", None),
        num_fewshot=getattr(args, "num_fewshot", None),
        batch_size=getattr(args, "batch_size", None),
        skip_baseline=getattr(args, "skip_baseline", False),
        quantization_config=quant_config,
        neuropress=RunMeta.collect(),
    )

    out = Path(args.output).parent / "run_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(run_config.model_dump_json(indent=2))


def _cmd_compare(args: argparse.Namespace) -> None:
    """Evaluate before and after quantization; print a delta table.

    By default, when ``--output`` is set, per-task evaluation checkpoints
    are written to ``<output_parent>/eval_partial/`` so that completed
    tasks are skipped on retry.  Disable with ``--skip-checkpoints``.

    When ``--save-model`` is set the quantized checkpoint is also used
    for resume: if the file already exists, it is loaded instead of
    re-running calibration + quantization.  Disable with
    ``--no-resume-quantized``.

    Args:
        args: Parsed CLI namespace.
    """
    from rich.console import Console

    from neuropress.eval.harness import (
        EvalComparison,
        normalize_tasks,
        validate_tasks,
    )
    from neuropress.eval.timing import (
        EvalPhaseTiming,
        EvalReport,
        ModelSaveTiming,
        QuantizationTiming,
        RunTiming,
        StepTiming,
    )
    from neuropress.meta import RunMeta

    args.tasks = normalize_tasks(args.tasks)
    unknown = validate_tasks(args.tasks)
    if unknown:
        _die(f"Unknown lm-eval task(s): {', '.join(unknown)}")

    # Write run_config.json before any heavy computation so it exists even
    # if the run crashes during quantization or eval.
    if args.output:
        _write_run_config(args)

    console = Console()
    eval_kwargs: dict[str, Any] = {
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
    }

    # Derive checkpoint directory from --output (unless disabled).
    partial_dir: Path | None = None
    if args.output and not args.skip_checkpoints:
        partial_dir = Path(args.output).parent / "eval_partial"

    run_timing = RunTiming()

    t_load = datetime.now(UTC)
    model, tokenizer = _load_model_and_tokenizer(
        args.model, device=args.device, dtype=args.dtype
    )
    run_timing.model_load = StepTiming.close(t_load)

    # Capture device before baseline eval: lm-eval/accelerate hooks can move
    # model parameters to CPU when the HFLM wrapper is garbage-collected.
    model_device = next(model.parameters()).device
    cache_dir = _resolve_cache_dir(
        explicit=getattr(args, "cache_dir", None),
        no_cache=getattr(args, "no_cache_dir", False),
        model_id=args.model,
    )
    if cache_dir is not None:
        console.print(f"  [dim]cache → {cache_dir}[/dim]")
    quantize_fn = _build_quantize_fn(
        args.compression, tokenizer, cache_dir=str(cache_dir) if cache_dir else None
    )

    n_params = sum(p.numel() for p in model.parameters())
    baseline_bytes = _model_bytes(model)

    skip_baseline = getattr(args, "skip_baseline", False)
    baseline_per_task_timing: dict[str, dict[str, Any]] = {}
    if args.baseline_from is not None:
        console.rule("[bold cyan]1 / 3  Baseline (loaded from file)[/bold cyan]")
        with open(args.baseline_from) as _bf:
            _bl_data = json.load(_bf)
        baseline = _bl_data["baseline"]["results"]
        baseline_time_s = _bl_data.get("baseline_time_s", 0.0)
        console.print(f"  [green]Loaded baseline from {args.baseline_from}[/green]")
        t = datetime.now(UTC)
        run_timing.baseline_eval = EvalPhaseTiming(
            start=t, stop=t, source=args.baseline_from
        )
    elif skip_baseline:
        console.rule("[bold cyan]1 / 3  Baseline (skipped)[/bold cyan]")
        baseline = None
        baseline_time_s = 0.0
        t = datetime.now(UTC)
        run_timing.baseline_eval = EvalPhaseTiming(start=t, stop=t, skipped=True)
    else:
        console.rule("[bold cyan]1 / 3  Baseline evaluation[/bold cyan]")
        t_bl = datetime.now(UTC)
        baseline, baseline_time_s, baseline_per_task_timing = _evaluate_per_task(
            model,
            tokenizer,
            args.tasks,
            eval_kwargs,
            partial_dir=partial_dir,
            prefix="baseline",
            console=console,
        )
        run_timing.baseline_eval = EvalPhaseTiming.close(
            t_bl, per_task=baseline_per_task_timing
        )

    console.rule("[bold cyan]2 / 3  Calibration + quantization[/bold cyan]")
    # Resume from a saved quantized model if it exists.
    save_path = Path(args.save_model) if args.save_model else None
    resume_quantized = (
        save_path is not None and save_path.exists() and not args.no_resume_quantized
    )
    _save_failed_exc: Exception | None = None
    if resume_quantized:
        from neuropress.t2n import load_quantized_model

        console.print(f"[green]Loading pre-quantized model from {save_path}[/green]")
        t_q = datetime.now(UTC)
        load_quantized_model(model, save_path)
        run_timing.quantization = QuantizationTiming.close(
            t_q, resumed_from=str(save_path)
        )
    else:
        # Restore device in case lm-eval/accelerate moved params off GPU.
        model.to(model_device)
        t_q = datetime.now(UTC)
        quantize_fn(model)
        run_timing.quantization = QuantizationTiming.close(t_q)

        if save_path is not None:
            import torch

            from neuropress.t2n import save_quantized_model

            save_path.parent.mkdir(parents=True, exist_ok=True)
            # Flush the CUDA allocator cache before saving — quantization
            # leaves the allocator fragmented and this maximises headroom.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            t_save = datetime.now(UTC)
            try:
                save_quantized_model(model, save_path)
                run_timing.model_save = ModelSaveTiming.close(t_save)
                console.print(f"[green]Quantized model saved to {save_path}[/green]")
            except Exception as exc:
                run_timing.model_save = ModelSaveTiming.close(t_save, error=str(exc))
                _save_failed_exc = exc
                console.print(
                    f"[bold red]ERROR: save_quantized_model failed — "
                    f"model NOT saved.[/bold red]\n{exc}"
                )

    # Ensure the entire model is on the expected device before quantized eval.
    # Non-persistent buffers (e.g. Gemma3's embed_scale) and QTensors
    # reconstructed from a checkpoint may be on CPU at this point.
    model.to(model_device)
    quantized_bytes = _model_bytes(model)

    console.rule("[bold cyan]3 / 3  Quantized evaluation[/bold cyan]")
    t_qe = datetime.now(UTC)
    quantized, quantized_time_s, quantized_per_task_timing = _evaluate_per_task(
        model,
        tokenizer,
        args.tasks,
        eval_kwargs,
        partial_dir=partial_dir,
        prefix="quantized",
        console=console,
    )
    run_timing.quantized_eval = EvalPhaseTiming.close(
        t_qe, per_task=quantized_per_task_timing
    )

    cmp = EvalComparison(
        baseline=baseline or {},
        quantized=quantized,
        baseline_time_s=baseline_time_s,
        quantized_time_s=quantized_time_s,
        n_params=n_params,
        baseline_bytes=baseline_bytes,
        quantized_bytes=quantized_bytes,
    )
    console.rule("[bold]Results[/bold]")
    print(cmp)
    if args.output:
        report = EvalReport(
            meta=RunMeta.collect(),
            timing=run_timing,
            baseline=_safe_json(cmp.baseline) if baseline else None,
            quantized=_safe_json(cmp.quantized),
            delta=cmp.delta if baseline else None,
            n_params=cmp.n_params,
            baseline_bytes=cmp.baseline_bytes,
            quantized_bytes=cmp.quantized_bytes,
            baseline_time_s=cmp.baseline_time_s,
            quantized_time_s=cmp.quantized_time_s,
        )
        Path(args.output).write_text(report.model_dump_json(indent=2))
        print(f"\nResults saved to {args.output}")

    # Raise after results are written so eval metrics are never silently lost.
    if _save_failed_exc is not None:
        raise RuntimeError(
            f"save_quantized_model failed — model not saved. "
            f"Eval results above are valid. Original error: {_save_failed_exc}"
        ) from _save_failed_exc


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add shared model, tasks, and eval hyperparameter arguments.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument(
        "model",
        metavar="MODEL",
        help=("HuggingFace model ID or local path (e.g. 'meta-llama/Llama-2-7b-hf')."),
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        required=True,
        metavar="TASK",
        help="One or more lm-eval task names (e.g. hellaswag arc_easy).",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Max examples per task. "
            "Float in (0, 1] = fraction; int ≥ 1 = count. "
            "Default: all examples."
        ),
    )
    parser.add_argument(
        "--num-fewshot",
        "-n",
        type=int,
        default=None,
        metavar="K",
        help="Number of few-shot examples. Default: task default.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=_batch_size_type,
        default=1,
        metavar="B",
        help="Batch size or 'auto'. Default: 1.",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cpu",
        metavar="DEVICE",
        help="PyTorch device string (e.g. cpu, cuda, cuda:1). Default: cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help=(
            "Model parameter dtype. 'auto' defers to the model config (default: auto)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Save results as JSON to FILE.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` with ``eval``,
        ``compare``, and ``list`` sub-commands.
    """
    parser = argparse.ArgumentParser(
        prog="neuropress-eval",
        description=(
            "Evaluate and compare HuggingFace language models using "
            "lm-evaluation-harness and neuropress quantization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  neuropress-eval eval meta-llama/Llama-2-7b-hf "
            "--tasks hellaswag --limit 100 --device cuda\n"
            "  neuropress-eval compare meta-llama/Llama-2-7b-hf "
            "--tasks hellaswag --compression open_text_c4_gptq_q4_0 "
            "--device cuda\n"
            "  neuropress-eval list"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- eval -----------------------------------------------------------------
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate a float model on lm-eval tasks.",
        description="Load a HuggingFace CausalLM and evaluate it on lm-eval tasks.",
    )
    _add_common_eval_args(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    # -- compare --------------------------------------------------------------
    p_cmp = sub.add_parser(
        "compare",
        help="Evaluate before and after quantization; print a delta table.",
        description=(
            "Evaluate the float model, apply a neuropress compression method, "
            "evaluate the quantized model, and print a delta summary table."
        ),
    )
    _add_common_eval_args(p_cmp)
    p_cmp.add_argument(
        "--compression",
        "-c",
        required=True,
        metavar="METHOD",
        help=(
            "Compression method key from NEUROPRESS_COMPRESSION "
            "(e.g. open_text_c4_gptq_q4_0). "
            "Run `neuropress-eval list` to see all available methods."
        ),
    )
    p_cmp.add_argument(
        "--save-model",
        default=None,
        metavar="FILE",
        help=(
            "Save the quantized model as a safetensors checkpoint to FILE "
            "after quantization (e.g. quantized.safetensors).  On retry, "
            "if the file already exists the model is loaded from it and "
            "calibration + quantization are skipped (disable with "
            "--no-resume-quantized)."
        ),
    )
    p_cmp.add_argument(
        "--skip-checkpoints",
        action="store_true",
        default=False,
        help=(
            "Disable per-task evaluation checkpoints.  By default, when "
            "--output is set, each task result is saved individually so "
            "that completed tasks are skipped on retry."
        ),
    )
    p_cmp.add_argument(
        "--no-resume-quantized",
        action="store_true",
        default=False,
        help=(
            "Do not load a pre-quantized model from the --save-model "
            "path, even if the file already exists.  Forces re-running "
            "calibration + quantization."
        ),
    )
    p_cmp.add_argument(
        "--baseline-from",
        default=None,
        metavar="FILE",
        help=(
            "Reuse baseline results from a prior eval_report.json instead "
            "of re-evaluating the float model.  Saves ~30 min per run when "
            "the same base model is evaluated across multiple compression "
            "methods."
        ),
    )
    p_cmp.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help=(
            "Skip baseline evaluation entirely.  The report will contain "
            "null for baseline results.  Useful when baseline data already "
            "exists from a prior sweep."
        ),
    )
    p_cmp.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory for quantization stage caching (hessians, centroids, "
            "assignments).  When $NEUROPRESS_DATA_DIR is set, defaults to "
            "$NEUROPRESS_DATA_DIR/cache/{model}.  Calibration stages (hessians) "
            "are shared across runs with the same calibration config; recipe "
            "stages (centroids, assignments) are keyed on the full recipe config."
        ),
    )
    p_cmp.add_argument(
        "--no-cache-dir",
        action="store_true",
        default=False,
        help=(
            "Disable quantization stage caching even when $NEUROPRESS_DATA_DIR "
            "is set.  Overrides --cache-dir and the auto-derived default."
        ),
    )
    p_cmp.set_defaults(func=_cmd_compare)

    # -- list -----------------------------------------------------------------
    p_list = sub.add_parser(
        "list",
        help="List available compression methods.",
        description="Print all keys from NEUROPRESS_COMPRESSION.",
    )
    p_list.set_defaults(func=_cmd_list)

    # -- size -----------------------------------------------------------------
    p_size = sub.add_parser(
        "size",
        help="Estimate quantized model size without running compression.",
        description=(
            "Load a HuggingFace model config (no weight download) on the meta "
            "device and estimate the quantized storage size for a given "
            "compression method, without running calibration or GPTQ."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  neuropress-eval size google/gemma-3-270m "
            "--compression open_text_c4_gptq_q4_0_256samples\n"
            "  neuropress-eval size meta-llama/Llama-3.2-1B "
            "--compression open_text_c4_gptq_panel_q4q2_linear_256samples "
            "--dtype bfloat16"
        ),
    )
    p_size.add_argument(
        "model",
        metavar="MODEL",
        help="HuggingFace model ID or local path (e.g. 'google/gemma-3-270m').",
    )
    p_size.add_argument(
        "--compression",
        "-c",
        required=True,
        metavar="METHOD",
        help=(
            "Compression method key from NEUROPRESS_COMPRESSION. "
            "Run `neuropress-eval list` to see all available methods."
        ),
    )
    p_size.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help=(
            "Baseline float dtype for size calculation. "
            "'auto' infers from the model config (default: auto)."
        ),
    )
    p_size.set_defaults(func=_cmd_size)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``neuropress-eval`` CLI.

    Args:
        argv: Argument list (defaults to :data:`sys.argv[1:]`).
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
