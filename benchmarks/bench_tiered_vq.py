r"""Evaluation harness comparing all VQ packing schemes at iso-quality.

Measures bits-per-weight (bpw), packed buffer size, pack/unpack speed,
and inference throughput (compile backend) across:

- Fixed-width (:class:`IndexPacking`)
- Two-tier (:class:`TieredIndexPacking`)
- N-tier CBS (:class:`NTieredIndexPacking`)
- Zero-escape (:class:`ZeroEscapePacking`)

Usage::

    uv run python benchmarks/bench_tiered_vq.py

    # Custom config:
    uv run python benchmarks/bench_tiered_vq.py \
        --num-centroids 1024 --vector-len 4 --zipf-s 1.2

    # With CUDA throughput (requires GPU):
    uv run python benchmarks/bench_tiered_vq.py --cuda
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

import neuropress.t2n.qtensors  # noqa: F401  - register serde as side effect
from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme
from neuropress.transforms.vq_packing import (
    IndexPacking,
    NTieredIndexPacking,
    TieredIndexPacking,
    ZeroEscapePacking,
)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _zipf_indices(
    n: int, num_centroids: int, s: float = 1.2, seed: int = 42
) -> torch.Tensor:
    """Generate Zipf-distributed centroid indices.

    Centroid ``k`` (0-indexed) has probability proportional to
    ``1 / (k + 1)^s``.  This mimics the frequency-sorted assignment
    distribution observed in real LLM VQ weights.

    Args:
        n: Number of indices to generate.
        num_centroids: Codebook size.
        s: Zipf exponent.  Higher values = more skewed.
        seed: Random seed.

    Returns:
        1-D ``int64`` tensor of length *n*.
    """
    rng = torch.Generator().manual_seed(seed)
    ranks = torch.arange(1, num_centroids + 1, dtype=torch.float64)
    probs = 1.0 / ranks.pow(s)
    probs /= probs.sum()
    indices = torch.multinomial(probs.float(), n, replacement=True, generator=rng)
    return indices.to(torch.int64)


# ---------------------------------------------------------------------------
# Packing scheme wrappers
# ---------------------------------------------------------------------------


@dataclass
class PackingResult:
    """Result of a packing scheme evaluation."""

    name: str
    packed_bytes: int
    bpi: float
    bpw: float
    pack_ms: float
    unpack_ms: float
    n_indices: int
    vector_len: int


def _time_fn(fn, *args, warmup: int = 2, repeat: int = 5, **kwargs) -> float:
    """Time a function in milliseconds (median of *repeat* runs)."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def eval_fixed(
    indices: torch.Tensor,
    num_centroids: int,
    vector_len: int,
) -> PackingResult:
    """Evaluate fixed-width IndexPacking."""
    bits = IndexPacking.bits_for(num_centroids)
    n = indices.numel()
    pack_ms = _time_fn(IndexPacking.pack, indices, bits)
    packed = IndexPacking.pack(indices, bits)
    unpack_ms = _time_fn(IndexPacking.unpack, packed, bits, n)
    return PackingResult(
        name=f"fixed-{bits}",
        packed_bytes=packed.numel(),
        bpi=float(bits),
        bpw=float(bits) / vector_len,
        pack_ms=pack_ms,
        unpack_ms=unpack_ms,
        n_indices=n,
        vector_len=vector_len,
    )


def eval_tiered(
    indices: torch.Tensor,
    num_centroids: int,
    vector_len: int,
    hot_bits: int = 4,
) -> PackingResult:
    """Evaluate TieredIndexPacking."""
    cold_bits = IndexPacking.bits_for(num_centroids)
    n = indices.numel()
    pack_ms = _time_fn(TieredIndexPacking.pack, indices, cold_bits, hot_bits=hot_bits)
    packed = TieredIndexPacking.pack(indices, cold_bits, hot_bits=hot_bits)
    unpack_ms = _time_fn(TieredIndexPacking.unpack, packed)

    hot_thr = TieredIndexPacking.hot_threshold(hot_bits)
    n_hot = int((indices < hot_thr).sum().item())
    bpi = TieredIndexPacking.effective_bits(n_hot, n, cold_bits, hot_bits)

    return PackingResult(
        name=f"2-tier(h{hot_bits})",
        packed_bytes=packed.numel(),
        bpi=bpi,
        bpw=bpi / vector_len,
        pack_ms=pack_ms,
        unpack_ms=unpack_ms,
        n_indices=n,
        vector_len=vector_len,
    )


def eval_ntier(
    indices: torch.Tensor,
    num_centroids: int,
    vector_len: int,
    tier_bits: list[int] | None = None,
) -> PackingResult:
    """Evaluate NTieredIndexPacking."""
    if tier_bits is None:
        # Auto-select tier_bits based on codebook size.
        bits = IndexPacking.bits_for(num_centroids)
        if bits <= 8:
            tier_bits = [4, bits]
        elif bits <= 12:
            tier_bits = [4, 8, bits]
        else:
            tier_bits = [4, 8, 12, bits]
    n = indices.numel()
    pack_ms = _time_fn(NTieredIndexPacking.pack, indices, tier_bits)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    unpack_ms = _time_fn(NTieredIndexPacking.unpack, packed)

    thresholds = NTieredIndexPacking.tier_thresholds(tier_bits)
    d = len(tier_bits)
    tier_counts = []
    for k in range(d):
        lo = 0 if k == 0 else thresholds[k - 1]
        hi = thresholds[k] if k < d - 1 else (1 << tier_bits[-1])
        tier_counts.append(int(((indices >= lo) & (indices < hi)).sum().item()))
    bpi = NTieredIndexPacking.effective_bits(tier_counts, tier_bits)

    return PackingResult(
        name=f"ntier{tier_bits}",
        packed_bytes=packed.numel(),
        bpi=bpi,
        bpw=bpi / vector_len,
        pack_ms=pack_ms,
        unpack_ms=unpack_ms,
        n_indices=n,
        vector_len=vector_len,
    )


def eval_zero_escape(
    indices: torch.Tensor,
    num_centroids: int,
    vector_len: int,
) -> PackingResult:
    """Evaluate ZeroEscapePacking."""
    cold_bits = IndexPacking.bits_for(num_centroids)
    n = indices.numel()
    pack_ms = _time_fn(ZeroEscapePacking.pack, indices, cold_bits)
    packed = ZeroEscapePacking.pack(indices, cold_bits)
    unpack_ms = _time_fn(ZeroEscapePacking.unpack, packed)

    n_zero = int((indices == 0).sum().item())
    n_hot = int(((indices >= 1) & (indices < 17)).sum().item())
    n_warm = int(((indices >= 17) & (indices < 257)).sum().item())
    n_cold = int((indices >= 257).sum().item())
    bpi = ZeroEscapePacking.effective_bits(n_zero, n_hot, n_warm, n_cold, cold_bits)

    return PackingResult(
        name="zero-esc",
        packed_bytes=packed.numel(),
        bpi=bpi,
        bpw=bpi / vector_len,
        pack_ms=pack_ms,
        unpack_ms=unpack_ms,
        n_indices=n,
        vector_len=vector_len,
    )


# ---------------------------------------------------------------------------
# Inference throughput measurement
# ---------------------------------------------------------------------------


def _measure_linear_throughput(
    out_features: int,
    in_features: int,
    num_centroids: int,
    vector_len: int,
    scheme_kwargs: dict,
    batch_seq: tuple[int, int],
    device: str,
    backend: str,
) -> float:
    """Measure compile-backend linear throughput (ms per call).

    Creates a VQScheme, quantizes a random weight, wraps in nn.Linear +
    QTensorTractVQ, linearizes to NPVQLinear, and measures forward time.

    Args:
        out_features: N.
        in_features: K.
        num_centroids: Codebook size.
        vector_len: VQ vector length.
        scheme_kwargs: Extra kwargs for VQScheme (use_tiered, zero_escape, etc).
        batch_seq: (batch, seq_len) for input.
        device: "cpu" or "cuda".
        backend: NPVQLinear backend string.

    Returns:
        Median forward time in milliseconds.
    """
    from neuropress.nn.np_vq_linear import NPVQLinear, VQProxy

    centroids = torch.randn(num_centroids, vector_len, dtype=torch.float16)
    shape = torch.Size([out_features, in_features])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        **scheme_kwargs,
    )
    weight = torch.randn(out_features, in_features)
    packed = scheme.quantize_as_u8(weight)
    qtensor = QTensorTractVQ(
        scheme.dequantize(packed, torch.float32),
        scheme,
        dequant_to_dtype=torch.float32,
    )

    # Build nn.Linear with QTensor weight.
    linear = nn.Linear(in_features, out_features, bias=False)
    linear._parameters["weight"] = qtensor

    proxy = VQProxy(centroids)
    np_lin = NPVQLinear.from_qtensor_linear(linear, proxy, backend=backend)
    if device == "cuda":
        np_lin = np_lin.cuda()
        proxy = proxy.cuda()

    batch, seq = batch_seq
    x = torch.randn(batch, seq, in_features, device=device)
    if device == "cuda":
        x = x.half()

    # Warmup.
    for _ in range(3):
        np_lin(x)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(10):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        np_lin(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------


def _print_distribution(indices: torch.Tensor, num_centroids: int) -> None:
    """Print assignment distribution statistics."""
    n = indices.numel()
    n_zero = int((indices == 0).sum().item())
    n_hot = int(((indices >= 0) & (indices < 16)).sum().item())
    n_warm = int(((indices >= 0) & (indices < 256)).sum().item())

    print("\n--- Assignment distribution ---")
    print(f"Total indices:     {n:>10}")
    print(f"Centroid-0 (zero): {n_zero:>10} ({100 * n_zero / n:.1f}%)")
    print(f"Top-16 (hot):      {n_hot:>10} ({100 * n_hot / n:.1f}%)")
    print(f"Top-256 (warm):    {n_warm:>10} ({100 * n_warm / n:.1f}%)")
    print(f"Cold (256+):       {n - n_warm:>10} ({100 * (n - n_warm) / n:.1f}%)")

    # Top-5 centroids.
    counts = torch.bincount(indices.to(torch.int32), minlength=num_centroids)
    top5 = torch.topk(counts, min(5, num_centroids))
    print("\nTop-5 centroids:")
    for i, (idx, cnt) in enumerate(zip(top5.indices, top5.values, strict=True)):
        pct = 100 * cnt.item() / n
        print(f"  #{i}: centroid {idx.item():<6d}  {cnt.item():>8d} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the tiered VQ evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Compare VQ packing schemes at iso-quality."
    )
    parser.add_argument(
        "--num-centroids",
        type=int,
        default=256,
        help="Codebook size (default: 256).",
    )
    parser.add_argument(
        "--vector-len",
        type=int,
        default=4,
        help="VQ vector length (default: 4).",
    )
    parser.add_argument(
        "--zipf-s",
        type=float,
        default=1.2,
        help="Zipf exponent (default: 1.2; higher = more skewed).",
    )
    parser.add_argument(
        "--out-features",
        type=int,
        default=1024,
        help="Weight rows / output features (default: 1024).",
    )
    parser.add_argument(
        "--in-features",
        type=int,
        default=1024,
        help="Weight cols / input features (default: 1024).",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Measure CUDA inference throughput (requires GPU).",
    )
    parser.add_argument(
        "--batch-seq",
        type=str,
        default="1,1",
        help="batch,seq_len for throughput (default: 1,1).",
    )
    args = parser.parse_args()

    nc = args.num_centroids
    vl = args.vector_len
    out_f = args.out_features
    in_f = args.in_features
    n_groups = in_f // vl
    n_indices = out_f * n_groups

    print(
        f"Config: {nc} centroids, VL={vl}, "
        f"weight [{out_f}, {in_f}], "
        f"Zipf s={args.zipf_s}"
    )
    print(f"Indices: {n_indices} ({out_f} rows x {n_groups} groups)")

    indices = _zipf_indices(n_indices, nc, s=args.zipf_s)
    _print_distribution(indices, nc)

    # --- Evaluate all packing schemes ---
    results: list[PackingResult] = []

    results.append(eval_fixed(indices, nc, vl))
    if nc >= 16:
        results.append(eval_tiered(indices, nc, vl))
    bits = IndexPacking.bits_for(nc)
    if bits > 4:
        if bits <= 8:
            tb = [4, bits]
        elif bits <= 12:
            tb = [4, 8, bits]
        else:
            tb = [4, 8, 12, bits]
        results.append(eval_ntier(indices, nc, vl, tier_bits=tb))
    if nc >= 257:
        results.append(eval_zero_escape(indices, nc, vl))

    # --- Print results table ---
    print("\n--- Packing comparison ---")
    print(
        f"{'scheme':<20} {'bytes':>8} {'bpi':>7} {'bpw':>7} "
        f"{'pack_ms':>9} {'unpack_ms':>10} {'saving':>8}"
    )
    print("-" * 80)

    baseline_bytes = results[0].packed_bytes
    for r in results:
        saving = 1.0 - r.packed_bytes / baseline_bytes if baseline_bytes else 0
        print(
            f"{r.name:<20} {r.packed_bytes:>8} {r.bpi:>7.2f} {r.bpw:>7.3f} "
            f"{r.pack_ms:>9.2f} {r.unpack_ms:>10.2f} {saving:>7.1%}"
        )

    # --- Inference throughput ---
    batch_seq = tuple(int(x) for x in args.batch_seq.split(","))
    use_cuda = args.cuda and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Build the backend x scheme matrix.
    # Each entry: (display_name, backend_str, scheme_kwargs)
    configs: list[tuple[str, str, dict]] = []

    # compile backend (always available)
    configs.append(("fixed/compile", "compile", {}))
    if nc >= 16:
        configs.append(("2-tier/compile", "compile", {"use_tiered": True}))
    if nc >= 257:
        configs.append(("zero-esc/compile", "compile", {"zero_escape": True}))

    # CUDA kernel backends (only when --cuda)
    if use_cuda:
        configs.append(("fixed/cuda", "cuda", {}))
        if nc >= 16:
            configs.append(("2-tier/cuda_tiered", "cuda_tiered", {"use_tiered": True}))
        if bits > 4:
            ntier_kw: dict = {"tier_bits": tb} if "tb" in dir() else {}
            if ntier_kw:
                configs.append(("ntier/cuda_ntier", "cuda_ntier", ntier_kw))
        # zero-escape has no CUDA kernel yet — compile-only on GPU
        if nc >= 257:
            configs.append(("zero-esc/compile@gpu", "compile", {"zero_escape": True}))

    for bs_label, bs_val in [
        (f"decode({batch_seq[0]}x{batch_seq[1]})", batch_seq),
    ]:
        print(f"\n--- Inference throughput ({device}, {bs_label}) ---")
        print(f"{'scheme/backend':<28} {'fwd_ms':>8}")
        print("-" * 38)

        for name, backend, kwargs in configs:
            try:
                ms = _measure_linear_throughput(
                    out_f,
                    in_f,
                    nc,
                    vl,
                    kwargs,
                    bs_val,
                    device,
                    backend,
                )
                print(f"{name:<28} {ms:>8.3f}")
            except Exception as e:
                print(f"{name:<28} {'ERROR':>8}  {e!s:.60s}")


if __name__ == "__main__":
    main()
