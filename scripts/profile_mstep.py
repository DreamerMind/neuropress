#!/usr/bin/env python3
"""Synthetic M-step + E-step profiling — no real model needed.

Creates fake nn.Linear modules with Gemma 4 E2B-like layer shapes,
synthetic packed indices (NC=32768, 15-bit), and runs 3 Lloyd iterations
with VPTQ_PROFILE_MSTEP=1 to get per-op timing breakdown from _refine_centroids.

Runs in ~2-5 minutes on RTX 4090.  No HuggingFace download required.

Usage:
    VPTQ_PROFILE_MSTEP=1 uv run python scripts/profile_mstep.py
    VPTQ_PROFILE_MSTEP=1 uv run python scripts/profile_mstep.py --nc 4096
    VPTQ_PROFILE_MSTEP=1 uv run python scripts/profile_mstep.py --n-iters 3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Force profile env var before importing estimators.
os.environ.setdefault("VPTQ_PROFILE_MSTEP", "1")

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
)


# ---------------------------------------------------------------------------
# Gemma 4 E2B-like layer shapes (approximate).
# 18 transformer blocks x 7 linear projections each = 126 + extras.
# We replicate to reach ~526 layers matching the measured count.
# ---------------------------------------------------------------------------
_GEMMA_BLOCK_SHAPES = [
    # (out, in) — matches Gemma 4 E2B approximate architecture
    (2048, 2048),  # q_proj
    (256, 2048),  # k_proj  (grouped-query attention, 2 kv heads)
    (256, 2048),  # v_proj
    (2048, 2048),  # o_proj
    (8192, 2048),  # gate_proj
    (8192, 2048),  # up_proj
    (2048, 8192),  # down_proj
]
_N_TRANSFORMER_LAYERS = 18


def _build_layer_shapes(n_extra: int = 0) -> list[tuple[int, int]]:
    shapes = []
    for _ in range(_N_TRANSFORMER_LAYERS):
        shapes.extend(_GEMMA_BLOCK_SHAPES)
    # Pad to target count with small projections.
    for _ in range(n_extra):
        shapes.append((2048, 2048))
    return shapes


def _make_fake_packed(n_vectors: int, nc: int, dev: torch.device) -> torch.Tensor:
    import math

    from neuropress.transforms.vq_packing import IndexPacking

    bits = math.ceil(math.log2(nc))
    indices = torch.randint(0, nc, (n_vectors,), dtype=torch.int64, device=dev)
    return IndexPacking.pack(indices, bits)


@dataclass
class _FakeRegistry:
    """Minimal registry stub returning a flat Hessian diagonal."""

    _data: dict[str, torch.Tensor] = field(default_factory=dict)

    def get(self, target_key: str, stat_type: str):
        return self._data.get(f"{target_key}::{stat_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nc", type=int, default=32768, help="Codebook size (default 32768)"
    )
    parser.add_argument("--vl", type=int, default=4, help="Vector length (default 4)")
    parser.add_argument(
        "--n-iters",
        type=int,
        default=3,
        help="Lloyd iterations to run (profile fires on iter 1, default 3)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=None,
        help="Override number of layers (default: Gemma-like ~126+)",
    )
    args = parser.parse_args()

    NC = args.nc
    VL = args.vl
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("WARNING: No CUDA — timings will be CPU-only.")

    # Build layer shapes.
    shapes = _build_layer_shapes()
    if args.n_layers is not None:
        shapes = (
            shapes[: args.n_layers]
            if args.n_layers <= len(shapes)
            else (shapes + [(2048, 2048)] * (args.n_layers - len(shapes)))
        )
    n_layers = len(shapes)

    print(
        f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"NC      : {NC}   VL={VL}")
    print(f"Layers  : {n_layers}   (Gemma 4 E2B-like shapes)")
    print(f"Iters   : {args.n_iters}  (profile fires on iter 1)")
    print()

    # Create fake nn.Linear modules on CUDA.
    print("Building fake layers...", flush=True)
    t0 = time.perf_counter()
    modules: list[tuple[str, nn.Module]] = []
    for i, (out_f, in_f) in enumerate(shapes):
        if in_f % VL != 0:
            continue
        layer_name = f"layer_{i:03d}"
        mod = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16, device=dev)
        modules.append((layer_name, mod))
    print(
        f"  {len(modules)} layers built in {time.perf_counter() - t0:.1f}s  "
        f"({sum(m.weight.numel() for _, m in modules) / 1e6:.0f}M params)",
        flush=True,
    )

    # Create random centroids [NC, VL] fp16.
    centroids = torch.randn(NC, VL, dtype=torch.float16, device=dev)

    # Create fake packed indices for each layer.
    print("Building fake packed indices...", flush=True)
    t0 = time.perf_counter()
    packed_per_layer: dict[str, torch.Tensor] = {}
    for name, mod in modules:
        w = mod.weight.data
        out_f, in_f = w.shape
        n_vecs = out_f * (in_f // VL)
        packed_per_layer[name] = _make_fake_packed(n_vecs, NC, dev)
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)

    # Build a fake registry with Hessian diagonals (all-ones for simplicity).
    registry = _FakeRegistry()
    for name, mod in modules:
        in_f = mod.weight.shape[1]
        # Fake hessian diagonal: uniform importance
        registry._data[f"{name}::hessian"] = torch.ones(
            in_f, dtype=torch.float32, device=dev
        )

    # Instantiate VPTQGroupEstimator directly (bypasses prepare/estimate).
    from neuropress.estimators import VPTQGroupEstimator

    estimator = VPTQGroupEstimator(
        num_centroids=NC,
        vector_len=VL,
        grouping="all",
        use_hessian_weighting=True,
        centroid_weighting="hessian",
        weighting_alpha=0.5,
        coverage_beta=0.0,
        tol=1e-4,
        max_n_iter=args.n_iters,
        gptq_correction_iters=0,  # no post-Lloyd GPTQ in this profiling run
    )
    estimator._device = dev
    estimator._target_modules = dict(modules)

    # Stub out _get_diag_h to return from our fake registry.
    def _fake_get_diag_h(name, K, reg):
        key = f"{name}::hessian"
        h = registry._data.get(key)
        if h is None:
            return None
        return h[:K].to(dev)

    estimator._get_diag_h = _fake_get_diag_h  # type: ignore[method-assign]

    # Stub out _weight_compute_device so it behaves as in production
    # (checks mem_get_info per call) without needing prepare() to run.
    # Keep the real implementation intact.

    perm_per_layer: dict[str, torch.Tensor | None] = {n: None for n, _ in modules}

    print(
        f"\nRunning {args.n_iters} Lloyd iterations (profile on iter 1)...\n",
        flush=True,
    )

    for iteration in range(args.n_iters):
        t_iter = time.perf_counter()

        # M-step: _refine_centroids.
        if iteration == 1:
            estimator._profile_next_mstep = True

        t_m = time.perf_counter()
        estimator._refine_centroids(
            members=modules,
            packed_per_layer=packed_per_layer,
            centroids=centroids,
            perm_per_layer=perm_per_layer,
            registry=registry,  # type: ignore[arg-type]
        )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_m = time.perf_counter() - t_m

        # E-step: re-assign each layer.
        t_e = time.perf_counter()
        _e_layer_times: list[float] = []
        for name, mod in modules:
            _le0 = time.perf_counter()
            import math

            from neuropress.methods.vptq import importance_weighted_assign
            from neuropress.transforms.vq_packing import IndexPacking

            w = mod.weight.data
            out_f, in_f = w.shape
            diag_h = torch.ones(in_f, dtype=torch.float32, device=dev)
            w2d = w.to(dev).float()
            indices = importance_weighted_assign(w2d, centroids.float(), diag_h, VL)
            bits = math.ceil(math.log2(NC))
            packed_per_layer[name] = IndexPacking.pack(indices, bits)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            _e_layer_times.append(time.perf_counter() - _le0)
        t_e = time.perf_counter() - t_e

        t_iter_total = time.perf_counter() - t_iter
        _e_sorted = sorted(_e_layer_times, reverse=True)
        print(
            f"iter {iteration + 1}/{args.n_iters}:  "
            f"M-step={t_m:.1f}s  E-step={t_e:.1f}s  total={t_iter_total:.1f}s  "
            f"[E-step: avg={sum(_e_layer_times) / len(_e_layer_times) * 1e3:.1f}ms/layer "
            f"top3={[f'{x * 1e3:.1f}ms' for x in _e_sorted[:3]]}]",
            flush=True,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
