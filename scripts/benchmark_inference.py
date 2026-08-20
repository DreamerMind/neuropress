"""End-to-end inference throughput benchmark: float vs quantized RTFx.

Measures three scenarios following the llama-bench convention:

  pp=512   — prompt processing, short  (partially compute-bound)
  pp=1024  — prompt processing, long   (more compute-bound, VQ at disadvantage)
  tg=128   — token generation          (memory-bandwidth bound, VQ advantage)

For each scenario reports absolute tok/s for float (bf16) and quantized, then
RTFx = quantized / float.  Values >1 mean quantized is faster.

Usage (local)::

    uv run python scripts/benchmark_inference.py \\
        --model google/gemma-3-270m \\
        --checkpoint data/eval/2026_04_vq_1024c_v4_hwt_cov/model \\
        --device cuda

Usage (cloud, with pre-saved model dir)::

    uv run python scripts/benchmark_inference.py \\
        --model google/gemma-3-270m \\
        --checkpoint /path/to/checkpoint \\
        --device cuda --output results/bench_270m_1024c.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench_pp(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    """Return prompt-processing throughput in tok/s.

    Runs a full forward pass over `seq_len` tokens. Time is measured
    wall-clock with CUDA synchronization so it includes all kernel launches.

    Returns:
        Median tok/s across `iters` timed runs.
    """
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
        _sync(device)

        timings: list[float] = []
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            model(input_ids)
            _sync(device)
            timings.append(time.perf_counter() - t0)

    timings.sort()
    median_s = timings[len(timings) // 2]
    return seq_len / median_s


def _bench_tg(
    model: nn.Module,
    n_tokens: int,
    vocab_size: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    """Return token-generation throughput in tok/s.

    Times a **single decode step** (one forward pass with a 1-token input and
    a pre-populated KV cache) rather than the full `model.generate()` loop.
    This isolates GPU compute from Python/HF framework overhead, which
    dominates at 270M scale (~40 ms/token overhead vs <1 ms GPU time).

    Strategy:
      1. Run a short prefill (32 tokens) to populate the KV cache.
      2. Time a single 1-token forward pass reusing that KV cache.
      3. Report ``1 / median_forward_s`` as tok/s.

    The ``n_tokens`` parameter is kept for API consistency but only affects
    the number of timed iterations reported.

    Returns:
        Peak single-token decode throughput in tok/s (median over `iters`).
    """
    # Prefill a short context to get a realistic KV cache.
    prefill_ids = torch.randint(0, vocab_size, (1, 32), device=device)
    with torch.no_grad():
        prefill_out = model(prefill_ids, use_cache=True)
    past_key_values = prefill_out.past_key_values

    # Single decode-step input.
    next_token = torch.randint(0, vocab_size, (1, 1), device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(next_token, past_key_values=past_key_values, use_cache=True)
        _sync(device)

        timings: list[float] = []
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            model(next_token, past_key_values=past_key_values, use_cache=True)
            _sync(device)
            timings.append(time.perf_counter() - t0)

    timings.sort()
    median_s = timings[len(timings) // 2]
    return 1.0 / median_s


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_float_model(
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=str(device)
    )
    model.eval()
    return model, tokenizer


def _load_quantized_model(
    model_id: str,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    lowering_config: Path | None = None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    from neuropress.kernel_lowering import AutoKernelLowering
    from neuropress.t2n.model_io import load_quantized_model

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="cpu"
    )
    load_quantized_model(model, checkpoint, kernel_lowering=None)

    AutoKernelLowering.from_config(lowering_config).apply(model)
    model.to(device)
    model.eval()
    return model


def _quantize_model(
    model_id: str,
    compression: str,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer: Any,
    lowering_config: Path | None = None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    from neuropress.kernel_lowering import AutoKernelLowering
    from neuropress.t2n import NEUROPRESS_COMPRESSION

    if compression not in NEUROPRESS_COMPRESSION:
        available = "\n  ".join(sorted(NEUROPRESS_COMPRESSION))
        print(
            f"ERROR: unknown compression {compression!r}.\nAvailable:\n  {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=str(device)
    )
    compress_fn = NEUROPRESS_COMPRESSION[compression]
    compress_fn(model, tokenizer=tokenizer)

    AutoKernelLowering.from_config(lowering_config).apply(model)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------


def _hw_meta(device: torch.device) -> dict[str, object]:
    meta: dict[str, object] = {"platform": platform.node()}
    try:
        meta["git_commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        meta["gpu_name"] = props.name
        meta["gpu_mem_gb"] = round(props.total_memory / 1024**3, 1)
        meta["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    return meta


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    scenario: str  # "pp512", "pp1024", "tg128"
    float_tok_s: float
    quant_tok_s: float
    rtfx: float  # quant / float


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Float vs quantized inference throughput (pp/tg RTFx)"
    )
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--checkpoint",
        help="Path to neuropress quantized checkpoint (.safetensors or dir)",
    )
    src.add_argument(
        "--compression",
        help="Compression key in NEUROPRESS_COMPRESSION — quantizes on the fly",
    )
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument(
        "--pp",
        nargs="+",
        type=int,
        default=[512, 1024],
        help="Prompt lengths to benchmark (default: 512 1024)",
    )
    p.add_argument(
        "--tg",
        type=int,
        default=128,
        help="Token generation length (default: 128)",
    )
    p.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=10, help="Timed iterations")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model compute dtype (default: bfloat16)",
    )
    p.add_argument("--output", help="Write JSON results to this path")
    p.add_argument(
        "--skip-float",
        action="store_true",
        help="Skip float baseline (use when float model does not fit in VRAM alongside quantized)",
    )
    p.add_argument(
        "--lowering-config",
        metavar="PATH",
        help=(
            "Path to a YAML kernel-lowering config (see configs/lowering/). "
            "Controls vq_backend, linear_backend, embedding_backend. "
            "Omit to use all-auto defaults."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    checkpoint: Path | None = None
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint)
        # Resolve checkpoint: if a directory, look for the model file inside.
        if checkpoint.is_dir():
            candidates = list(checkpoint.glob("*.safetensors"))
            if not candidates:
                print(
                    f"ERROR: no .safetensors file found in {checkpoint}",
                    file=sys.stderr,
                )
                sys.exit(1)
            checkpoint = candidates[0]

    meta = _hw_meta(device)

    # Print header.
    print("=" * 60)
    print("neuropress inference benchmark")
    print("=" * 60)
    print(f"Model      : {args.model}")
    if checkpoint:
        print(f"Checkpoint : {checkpoint}")
    else:
        print(f"Compression: {args.compression}")
    print(f"Device     : {meta.get('gpu_name', device)}")
    print(f"Dtype      : {args.dtype}")
    print(f"Lowering   : {args.lowering_config or '(auto)'}")
    print(f"Warmup/iter: {args.warmup}/{args.iters}")
    print(f"Commit     : {meta.get('git_commit', 'unknown')}")
    print()

    results: list[ScenarioResult] = []

    # --- Float baseline ---
    float_tok_s: dict[str, float] = {}
    tokenizer: Any = None
    if not args.skip_float:
        print("Loading float (bf16) model...")
        float_model, tokenizer = _load_float_model(args.model, device, dtype)
        vocab_size = tokenizer.vocab_size or float_model.config.vocab_size

        for pp in args.pp:
            label = f"pp{pp}"
            print(f"  {label} float... ", end="", flush=True)
            tok_s = _bench_pp(
                float_model, pp, vocab_size, device, args.warmup, args.iters
            )
            float_tok_s[label] = tok_s
            print(f"{tok_s:,.0f} tok/s")

        label = f"tg{args.tg}"
        print(f"  {label} float... ", end="", flush=True)
        tok_s = _bench_tg(
            float_model, args.tg, vocab_size, device, args.warmup, args.iters
        )
        float_tok_s[label] = tok_s
        print(f"{tok_s:,.0f} tok/s")

        del float_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    # --- Quantized model ---
    if args.skip_float and tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)

    lowering_config = Path(args.lowering_config) if args.lowering_config else None

    if checkpoint is not None:
        print("Loading quantized model from checkpoint...")
        quant_model = _load_quantized_model(
            args.model, checkpoint, device, dtype, lowering_config
        )
    else:
        print(f"Quantizing model with {args.compression!r}...")
        quant_model = _quantize_model(
            args.model, args.compression, device, dtype, tokenizer, lowering_config
        )

    if args.skip_float:
        vocab_size = tokenizer.vocab_size or quant_model.config.vocab_size

    quant_tok_s: dict[str, float] = {}

    for pp in args.pp:
        label = f"pp{pp}"
        print(f"  {label} quant... ", end="", flush=True)
        tok_s = _bench_pp(quant_model, pp, vocab_size, device, args.warmup, args.iters)
        quant_tok_s[label] = tok_s
        print(f"{tok_s:,.0f} tok/s")

    label = f"tg{args.tg}"
    print(f"  {label} quant... ", end="", flush=True)
    tok_s = _bench_tg(quant_model, args.tg, vocab_size, device, args.warmup, args.iters)
    quant_tok_s[label] = tok_s
    print(f"{tok_s:,.0f} tok/s")

    del quant_model
    print()

    # --- Results table ---
    all_labels = [f"pp{pp}" for pp in args.pp] + [f"tg{args.tg}"]
    for label in all_labels:
        q = quant_tok_s.get(label)
        f = float_tok_s.get(label)
        if q is not None and f is not None:
            rtfx = q / f
            results.append(ScenarioResult(label, f, q, rtfx))

    print("=" * 60)
    print(f"{'Scenario':<10} {'Float tok/s':>12} {'Quant tok/s':>12} {'RTFx':>8}")
    print("-" * 60)
    for r in results:
        faster = "✓" if r.rtfx >= 1.0 else " "
        print(
            f"{r.scenario:<10} {r.float_tok_s:>12,.0f} {r.quant_tok_s:>12,.0f}"
            f" {r.rtfx:>7.3f}x {faster}"
        )
    print("=" * 60)

    if not results and args.skip_float:
        print("Quantized tok/s (no float baseline):")
        for label, tok_s in quant_tok_s.items():
            print(f"  {label}: {tok_s:,.0f} tok/s")

    # --- JSON output ---
    if args.output:
        out = {
            "meta": meta,
            "model": args.model,
            "checkpoint": str(checkpoint),
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": [asdict(r) for r in results],
            "float_tok_s": float_tok_s,
            "quant_tok_s": quant_tok_s,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
