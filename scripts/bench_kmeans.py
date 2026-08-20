#!/usr/bin/env python3
"""K-means backend scaling benchmark.

Measures E-step and full-iteration throughput across (N, k, D) combinations,
comparing TorchKMeansBackend vs CudaKMeansBackend.

N  : number of original vectors (out_features × in_features/VL per layer)
k  : number of proxy centroids (codebook size)
D  : vector length — VL=4 in production

Speed is data-distribution independent per iteration; synthetic Gaussian
data is representative of real weight matrices.

Usage (local, quick smoke test):
    uv run python scripts/bench_kmeans.py --n-values 10000 50000 --k-values 256 1024

Usage (vast.ai, full sweep):
    python scripts/bench_kmeans.py --output out/bench_kmeans_4090.json
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

DEFAULT_N_VALUES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
DEFAULT_K_VALUES = [256, 1024, 4096, 16_384, 65_536]  # 64k shows flat-kernel ceiling
DEFAULT_D_VALUES = [4]
DEFAULT_BACKENDS = ["torch", "cuda"]
WARMUP_ITERS = 3
BENCH_ITERS = 10
LLOYDS_ITERS = 5  # fixed Lloyd steps for full-fit timing (avoids convergence variance)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def make_data(N: int, k: int, D: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(N, D, device="cuda", dtype=torch.float32, generator=g)
    centroids = torch.randn(k, D, device="cuda", dtype=torch.float32, generator=g)
    return x, centroids


def time_fn(fn, warmup: int, iters: int) -> float:
    """Median wall time in milliseconds."""
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


# ---------------------------------------------------------------------------
# E-step benchmarks
# ---------------------------------------------------------------------------


def bench_estep_torch(
    x: torch.Tensor, centroids: torch.Tensor, chunk_rows: int, warmup: int, iters: int
) -> float:
    N = x.shape[0]

    def _run():
        labels = torch.empty(N, dtype=torch.int64, device="cuda")
        for start in range(0, N, chunk_rows):
            slab = x[start : start + chunk_rows]
            labels[start : start + chunk_rows] = torch.cdist(
                slab, centroids, p=2
            ).argmin(1)
        return labels

    return time_fn(_run, warmup, iters)


def bench_estep_cuda(
    x: torch.Tensor, centroids: torch.Tensor, D: int, warmup: int, iters: int
) -> float:
    from neuropress.kernels import kmeans_assign

    def _run():
        return kmeans_assign(x, centroids, D)

    return time_fn(_run, warmup, iters)


# ---------------------------------------------------------------------------
# Full Lloyd-iteration benchmark (E + M step, fixed LLOYDS_ITERS)
# ---------------------------------------------------------------------------


def bench_fit_iter(
    backend: str,
    x: torch.Tensor,
    k: int,
    D: int,
    chunk_rows: int,
    warmup: int,
    iters: int,
) -> float:
    """Median per-iteration time (ms): E-step + M-step."""
    N = x.shape[0]
    centroids = x[torch.randperm(N, device="cuda")[:k]].clone()

    if backend == "torch":

        def _one_iter(c):
            labels = torch.empty(N, dtype=torch.int64, device="cuda")
            for start in range(0, N, chunk_rows):
                slab = x[start : start + chunk_rows]
                labels[start : start + chunk_rows] = torch.cdist(slab, c, p=2).argmin(1)
            new_c = torch.zeros_like(c)
            counts = torch.zeros(k, device="cuda")
            new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
            counts.scatter_add_(0, labels, torch.ones(N, device="cuda"))
            mask = counts > 0
            new_c[mask] = new_c[mask] / counts[mask].unsqueeze(1)
            new_c[~mask] = c[~mask]
            return new_c
    else:
        from neuropress.kernels import kmeans_assign

        def _one_iter(c):
            labels = kmeans_assign(x, c, D)
            new_c = torch.zeros_like(c)
            counts = torch.zeros(k, device="cuda")
            new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
            counts.scatter_add_(0, labels, torch.ones(N, device="cuda"))
            mask = counts > 0
            new_c[mask] = new_c[mask] / counts[mask].unsqueeze(1)
            new_c[~mask] = c[~mask]
            return new_c

    # Warmup.
    c = centroids.clone()
    for _ in range(warmup):
        c = _one_iter(c)
    torch.cuda.synchronize()

    times = []
    c = centroids.clone()
    for _ in range(iters):
        t0 = time.perf_counter()
        c = _one_iter(c)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    N: int
    k: int
    D: int
    backend: str
    estep_ms: float
    fit_iter_ms: float
    gfmas_s: float
    speedup_vs_torch: float | None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="K-means backend scaling benchmark")
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES, metavar="N"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES, metavar="K"
    )
    parser.add_argument(
        "--d-values", type=int, nargs="+", default=DEFAULT_D_VALUES, metavar="D"
    )
    parser.add_argument(
        "--backends", nargs="+", default=DEFAULT_BACKENDS, choices=["torch", "cuda"]
    )
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS, metavar="N")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS, metavar="N")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    warmup = args.warmup
    iters = args.iters
    gpu = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU : {gpu}  ({gpu_mem_gb:.1f} GB)")
    print(f"Backends : {args.backends}")
    print()

    results: list[BenchResult] = []
    col = f"{'N':>10}  {'k':>6}  {'D':>2}  {'backend':>8}  {'estep_ms':>10}  {'fit_iter_ms':>12}  {'GFMAs/s':>9}"
    print(col)
    print("-" * len(col))

    for D in args.d_values:
        for k in args.k_values:
            # Centroid table: k x D x 4 bytes. Skip if table alone > 512 MB.
            if k * D * 4 > 512 * 2**20:
                print(f"{'':>10}  {k:>6}  {D:>2}  skip (centroid table > 512 MB)")
                continue

            for N in args.n_values:
                if N * D * 4 > int(gpu_mem_gb * 0.4 * 2**30):
                    print(f"{N:>10}  {k:>6}  {D:>2}  skip (x tensor > 40% VRAM)")
                    continue

                try:
                    x, centroids = make_data(N, k, D)
                except torch.cuda.OutOfMemoryError:
                    print(f"{N:>10}  {k:>6}  {D:>2}  OOM")
                    continue

                if k >= N:
                    print(f"{N:>10}  {k:>6}  {D:>2}  skip (k >= N)")
                    del x, centroids
                    torch.cuda.empty_cache()
                    continue

                free, _ = torch.cuda.mem_get_info()
                chunk_rows = max(256, int(free * 0.10) // (k * 4))

                row: dict[str, BenchResult] = {}

                for backend in args.backends:
                    try:
                        if backend == "torch":
                            estep_ms = bench_estep_torch(
                                x, centroids, chunk_rows, warmup, iters
                            )
                        else:
                            estep_ms = bench_estep_cuda(x, centroids, D, warmup, iters)

                        fit_ms = bench_fit_iter(
                            backend, x, k, D, chunk_rows, warmup, iters
                        )
                        gfmas_s = (N * k * D) / (estep_ms * 1e-3) / 1e9

                        r = BenchResult(
                            N=N,
                            k=k,
                            D=D,
                            backend=backend,
                            estep_ms=estep_ms,
                            fit_iter_ms=fit_ms,
                            gfmas_s=gfmas_s,
                            speedup_vs_torch=None,
                        )
                        results.append(r)
                        row[backend] = r
                        print(
                            f"{N:>10}  {k:>6}  {D:>2}  {backend:>8}  "
                            f"{estep_ms:>10.2f}  {fit_ms:>12.2f}  {gfmas_s:>9.1f}"
                        )

                    except Exception as e:
                        print(f"{N:>10}  {k:>6}  {D:>2}  {backend:>8}  ERROR: {e}")

                if "torch" in row and "cuda" in row:
                    speedup = row["torch"].estep_ms / row["cuda"].estep_ms
                    row["cuda"].speedup_vs_torch = speedup
                    print(
                        f"{'':>10}  {'':>6}  {'':>2}  {'speedup':>8}  {speedup:>10.1f}x"
                    )

                del x, centroids
                torch.cuda.empty_cache()

    output = {
        "meta": {
            "gpu": gpu,
            "gpu_mem_gb": round(gpu_mem_gb, 1),
            "warmup_iters": WARMUP_ITERS,
            "bench_iters": BENCH_ITERS,
            "lloyds_iters": LLOYDS_ITERS,
        },
        "results": [asdict(r) for r in results],
    }

    if args.output:
        os.makedirs(
            os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
            exist_ok=True,
        )
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved → {args.output}")
    else:
        print("\n" + json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
