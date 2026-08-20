"""Per-layer forward timing to identify VQ inference bottlenecks.

Hooks each nn.Module, times its forward pass, and reports a ranked table
so we can see which layers dominate and whether lm_head is the bottleneck.

Usage::

    uv run python scripts/profile_layers.py \\
        --model google/gemma-3-270m \\
        --checkpoint data/eval/2026_04_vq_1024c_v4_hwt_cov/eval/quantized_model.safetensors \\
        --lowering-config configs/lowering/cuda_wmma.yaml \\
        --device cuda
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_model(
    model_id: str,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
    lowering_config: Path | None,
    attn_implementation: str | None = None,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    from neuropress.kernel_lowering import AutoKernelLowering
    from neuropress.t2n.model_io import load_quantized_model

    kwargs: dict = {"torch_dtype": dtype, "device_map": "cpu"}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    load_quantized_model(model, checkpoint, kernel_lowering=None)
    AutoKernelLowering.from_config(lowering_config).apply(model)
    model.to(device)
    model.eval()
    return model


def profile_layers(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[dict[str, float], float]:
    """Return (per-module mean forward time in ms, mean total wall time in ms)."""
    timings: dict[str, list[float]] = defaultdict(list)
    wall_times: list[float] = []
    handles: list[torch.utils.hooks.RemovableHook] = []

    def make_hook(name: str):
        def hook(module, inp, out):
            _sync()
            end = time.perf_counter()
            timings[name].append((end - hook._start) * 1000)

        def pre_hook(module, inp):
            _sync()
            hook._start = time.perf_counter()

        hook._start = 0.0
        return pre_hook, hook

    for name, mod in model.named_modules():
        if not list(mod.children()):  # leaf modules only
            pre, post = make_hook(name)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))

    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
        _sync()
        timings.clear()
        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()
            model(input_ids)
            _sync()
            wall_times.append((time.perf_counter() - t0) * 1000)

    for h in handles:
        h.remove()

    mean_wall = sum(wall_times) / len(wall_times)
    return {name: sum(v) / len(v) for name, v in timings.items() if v}, mean_wall


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-layer VQ inference profiler")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lowering-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seq-len", type=int, default=1, help="1=TG, 512=PP")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--top", type=int, default=20, help="Show top-N slowest layers")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Override attention implementation (default: model's own default)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        candidates = list(checkpoint.glob("*.safetensors"))
        assert candidates, f"No .safetensors in {checkpoint}"
        checkpoint = candidates[0]

    print(f"Loading model {args.model}...")
    model = _load_model(
        args.model,
        checkpoint,
        device,
        dtype,
        args.lowering_config,
        attn_implementation=args.attn_implementation,
    )

    # Report which attention implementation is actually active.
    attn_impl = getattr(
        getattr(model, "config", None), "_attn_implementation", "unknown"
    )
    print(f"Attention implementation: {attn_impl}")

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model)
    vocab_size = cfg.vocab_size

    input_ids = torch.randint(0, vocab_size, (1, args.seq_len), device=device)

    print(
        f"Profiling (seq_len={args.seq_len}, warmup={args.warmup}, iters={args.iters})..."
    )
    timings, mean_wall_ms = profile_layers(model, input_ids, args.warmup, args.iters)

    total = sum(timings.values())
    untracked = mean_wall_ms - total
    ranked = sorted(timings.items(), key=lambda x: x[1], reverse=True)

    print(f"\nWall time (full forward): {mean_wall_ms:.3f} ms")
    print(
        f"Tracked (leaf modules):   {total:.3f} ms  ({total / mean_wall_ms * 100:.1f}%)"
    )
    print(
        f"Untracked (SDPA + other): {untracked:.3f} ms  ({untracked / mean_wall_ms * 100:.1f}%)"
    )

    print(f"\n{'Module':<55} {'ms':>8} {'%':>7}")
    print("-" * 73)
    shown = 0
    for name, ms in ranked:
        pct = ms / total * 100
        print(f"{name:<55} {ms:8.3f} {pct:6.1f}%")
        shown += 1
        if shown >= args.top:
            break

    rest_ms = sum(ms for _, ms in ranked[args.top :])
    print(
        f"{'... (' + str(len(ranked) - args.top) + ' more)':<55} {rest_ms:8.3f} {rest_ms / total * 100:6.1f}%"
    )
    print("-" * 73)
    print(f"{'TOTAL (sum of leaves)':<55} {total:8.3f} {'100.0%':>7}")

    # Aggregate by module type
    print("\n=== By module class ===")
    from collections import defaultdict

    by_class: dict[str, float] = defaultdict(float)
    for name, mod in model.named_modules():
        if not list(mod.children()) and name in timings:
            by_class[type(mod).__name__] += timings[name]
    for cls, ms in sorted(by_class.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:<40} {ms:8.3f} ms  {ms / total * 100:5.1f}%")


if __name__ == "__main__":
    main()
