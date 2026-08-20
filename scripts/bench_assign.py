#!/usr/bin/env python3
"""Benchmark: grouped vs per-group-loop k-means assignment.

Compares the old Python-loop approach (G calls to kmeans_assign) against the
new single-launch kmeans_assign_grouped kernel for production-size layers.

Configs tested:
  lm_head : R=262144, G=384, NC=32768, D=4
  q_proj  : R=1024,   G=256, NC=32768, D=4

Usage:
    uv run python scripts/bench_assign.py            # full benchmark
    uv run python scripts/bench_assign.py --smoke    # quick sanity check
    uv run python scripts/bench_assign.py --output out/bench_assign.json
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass


def _time_fn(fn, warmup: int, iters: int) -> float:
    """Return median wall time in milliseconds over `iters` timed calls."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


@dataclass
class BenchResult:
    config: str  # "lm_head" or "q_proj"
    kernel: str  # "old_loop" or "grouped"
    R: int
    G: int
    NC: int
    time_ms: float
    queries_per_sec: float
    speedup: float  # relative to old_loop for same config; 1.0 for old_loop itself


def bench_config(
    name: str,
    R: int,
    G: int,
    NC: int,
    D: int = 4,
    warmup: int = 3,
    iters: int = 10,
) -> list[BenchResult]:
    import torch

    from neuropress.kernels import kmeans_assign, kmeans_assign_grouped

    dev = torch.device("cuda")
    torch.manual_seed(0)
    x3d = torch.randn(R, G, D, dtype=torch.float32, device=dev).contiguous()
    C = torch.randn(NC, D, dtype=torch.float32, device=dev).contiguous()
    imp = torch.rand(G, D, dtype=torch.float32, device=dev).add(0.1).contiguous()

    # ------------------------------------------------------------------
    # Old path: G calls to kmeans_assign in Python loop
    # ------------------------------------------------------------------
    def _old_loop():
        out = torch.empty(R, G, dtype=torch.int64, device=dev)
        for g in range(G):
            xg = (x3d[:, g, :] * imp[g].unsqueeze(0)).contiguous()
            cg = (C * imp[g].unsqueeze(0)).contiguous()
            out[:, g] = kmeans_assign(xg, cg, D)
        return out

    # ------------------------------------------------------------------
    # New path: single grouped kernel launch
    # ------------------------------------------------------------------
    def _grouped():
        return kmeans_assign_grouped(x3d, C, imp, D)

    print(f"  [{name}] warming up old_loop ({warmup} iters) …", flush=True)
    old_ms = _time_fn(_old_loop, warmup, iters)
    old_qps = R * G / (old_ms * 1e-3)

    print(f"  [{name}] warming up grouped  ({warmup} iters) …", flush=True)
    new_ms = _time_fn(_grouped, warmup, iters)
    new_qps = R * G / (new_ms * 1e-3)

    speedup = old_ms / new_ms

    return [
        BenchResult(
            config=name,
            kernel="old_loop",
            R=R,
            G=G,
            NC=NC,
            time_ms=round(old_ms, 3),
            queries_per_sec=round(old_qps, 1),
            speedup=1.0,
        ),
        BenchResult(
            config=name,
            kernel="grouped",
            R=R,
            G=G,
            NC=NC,
            time_ms=round(new_ms, 3),
            queries_per_sec=round(new_qps, 1),
            speedup=round(speedup, 3),
        ),
    ]


def _print_table(results: list[BenchResult]) -> None:
    header = f"{'config':<10}  {'kernel':<10}  {'R':>8}  {'G':>5}  {'NC':>7}  {'time_ms':>10}  {'Mq/s':>10}  {'speedup':>8}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        mqs = r.queries_per_sec / 1e6
        sp = f"{r.speedup:.2f}x" if r.kernel == "grouped" else "-"
        print(
            f"{r.config:<10}  {r.kernel:<10}  {r.R:>8}  {r.G:>5}  {r.NC:>7}"
            f"  {r.time_ms:>10.2f}  {mqs:>10.3f}  {sp:>8}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Grouped vs loop kmeans assign benchmark"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run tiny sizes for a quick sanity check (<5s)",
    )
    parser.add_argument("--warmup", type=int, default=3, metavar="N")
    parser.add_argument("--iters", type=int, default=10, metavar="N")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save JSON results"
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    gpu = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {gpu}  ({gpu_mem_gb:.1f} GB)")

    if args.smoke:
        configs = [
            {"name": "smoke", "R": 64, "G": 8, "NC": 256, "D": 4},
        ]
        warmup, iters = 1, 3
    else:
        configs = [
            {"name": "lm_head", "R": 262144, "G": 384, "NC": 32768, "D": 4},
            {"name": "q_proj", "R": 1024, "G": 256, "NC": 32768, "D": 4},
        ]
        warmup = args.warmup
        iters = args.iters

    all_results: list[BenchResult] = []
    for cfg in configs:
        name = cfg["name"]
        print(
            f"\nBenchmarking {name}: R={cfg['R']}, G={cfg['G']}, NC={cfg['NC']}, D={cfg['D']}"
        )
        try:
            results = bench_config(warmup=warmup, iters=iters, **cfg)
            all_results.extend(results)
            old_r = next(r for r in results if r.kernel == "old_loop")
            new_r = next(r for r in results if r.kernel == "grouped")
            print(f"  old_loop : {old_r.time_ms:.2f} ms")
            print(
                f"  grouped  : {new_r.time_ms:.2f} ms  ({new_r.speedup:.2f}x speedup)"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            raise

    _print_table(all_results)

    output_data = {
        "meta": {
            "gpu": gpu,
            "gpu_mem_gb": round(gpu_mem_gb, 1),
            "warmup": warmup,
            "iters": iters,
            "smoke": args.smoke,
        },
        "results": [asdict(r) for r in all_results],
    }

    if args.output:
        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
            exist_ok=True,
        )
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved -> {args.output}")
    else:
        print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
