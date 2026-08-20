"""Benchmark suite for neuropress quantized kernels.

Two modes:

1. **Synthetic** (default / ``neuropress-bench synthetic``): creates random
   quantized layers and benchmarks all available backends across common
   LLM shapes and batch sizes.

2. **Model** (``neuropress-bench model``): loads a real HuggingFace model,
   quantizes it (or loads a pre-quantized checkpoint), applies kernel
   lowering with different backends, and measures end-to-end inference
   throughput.

Usage::

    neuropress-bench                          # synthetic full sweep
    neuropress-bench synthetic --module vq    # only VQ layers
    neuropress-bench model --model ID --compression METHOD
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Shape presets
# ---------------------------------------------------------------------------

_LINEAR_SHAPES: dict[str, list[tuple[int, int]]] = {
    "small": [(2048, 2048), (2048, 8192)],
    "medium": [(4096, 4096), (4096, 14336)],
    "large": [(5120, 5120), (5120, 13824)],
    "xlarge": [(8192, 8192), (8192, 28672)],
}

_EMBEDDING_SHAPES: dict[str, list[tuple[int, int]]] = {
    "small": [(32000, 2048)],
    "medium": [(32000, 4096)],
    "large": [(128256, 4096)],
}

_DEFAULT_BATCH_SIZES = [1, 16, 256]

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Single benchmark measurement."""

    module_type: str
    shape: str
    batch: int
    backend: str
    median_ms: float
    min_ms: float
    timings: list[float] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def _bench_forward(
    module: nn.Module,
    make_input: Any,
    warmup: int,
    iters: int,
) -> list[float]:
    """Run *module* forward pass and return per-iteration ms."""
    # Modules may only have buffers (no parameters).
    device = torch.device("cpu")
    for p in module.parameters():
        device = p.device
        break
    else:
        for b in module.buffers():
            device = b.device
            break

    for _ in range(warmup):
        x = make_input()
        module(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings: list[float] = []
    for _ in range(iters):
        x = make_input()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        module(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - t0) * 1000.0)

    return timings


# ---------------------------------------------------------------------------
# Input factories (use functools.partial to bind loop vars)
# ---------------------------------------------------------------------------


def _make_randn(
    batch: int, dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(batch, dim, dtype=dtype, device=device)


def _make_randint(vocab: int, batch: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab, (batch,), device=device)


# ---------------------------------------------------------------------------
# Synthetic layer factories
# ---------------------------------------------------------------------------


def _make_qtensor(
    out_features: int,
    in_features: int,
    nbits: int,
    group_size: int,
    qtype: str,
    device: torch.device,
    dequant_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a QTensor by quantizing random float weights.

    Args:
        out_features: Number of output features.
        in_features: Number of input features.
        nbits: Bits per weight element.
        group_size: Quantization group size.
        qtype: ``"sym"`` or ``"asym"``.
        device: Target device.
        dequant_dtype: Dtype for dequantized output (f32 or f16).

    Returns:
        A QTensor on *device*.
    """
    w = torch.randn(out_features, in_features, dtype=torch.float32)
    if qtype == "sym" and nbits == 4 and group_size == 32:
        from torch_to_nnef.tensor.quant.qtract import (
            fp_to_tract_q4_0_with_min_max_calibration,
        )

        q = fp_to_tract_q4_0_with_min_max_calibration(w)
    else:
        from torch_to_nnef.tensor.quant.qtract import (
            qscale_per_group_f16_min_max_calibration,
        )

        q = qscale_per_group_f16_min_max_calibration(
            w,
            n_bits=nbits,
            group_size=group_size,
        )
    q.dequant_to_dtype = dequant_dtype
    if hasattr(q, "to"):
        return q.to(device)
    return q.to_device(device)


def _make_np_linear(
    out_features: int,
    in_features: int,
    nbits: int,
    group_size: int,
    qtype: str,
    backend: str,
    device: torch.device,
    dequant_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Create an NPLinear from random packed weights."""
    from ..nn.np_linear import NPLinear

    # For the default backend, build a real QTensor so F.linear can dequant.
    weight_q = None
    if backend == "default":
        weight_q = _make_qtensor(
            out_features,
            in_features,
            nbits,
            group_size,
            qtype,
            device,
            dequant_dtype=dequant_dtype,
        )

    vpb = 8 // nbits
    u8_blob = torch.randint(
        0,
        256,
        (out_features, in_features // vpb),
        dtype=torch.uint8,
        device=device,
    )
    n_groups = in_features // group_size
    scale = torch.randn(
        out_features * n_groups,
        dtype=torch.float16,
        device=device,
    )
    zero_point = None
    if qtype == "asym":
        zero_point = torch.randn(
            out_features * n_groups,
            dtype=torch.float16,
            device=device,
        )
    return NPLinear(
        u8_blob=u8_blob,
        scale=scale,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        nbits=nbits,
        qtype=qtype,
        backend=backend,
        zero_point=zero_point,
        weight_q=weight_q,
    )


def _zipf_indices(
    n: int,
    num_centroids: int,
    s: float = 1.2,
) -> torch.Tensor:
    """Generate Zipf-distributed indices for realistic tiered benchmarks.

    Args:
        n: Number of indices to generate.
        num_centroids: Codebook size (max index + 1).
        s: Zipf exponent — higher means more skew toward low indices.
            Default 1.2 models typical VQ codebook usage.

    Returns:
        Int64 tensor of shape ``[n]`` with values in ``[0, num_centroids)``.
    """
    ranks = torch.arange(1, num_centroids + 1, dtype=torch.float64)
    probs = 1.0 / ranks.pow(s)
    probs /= probs.sum()
    return torch.multinomial(
        probs.float(),
        n,
        replacement=True,
    ).to(torch.int64)


def _make_np_vq_linear(
    out_features: int,
    in_features: int,
    num_centroids: int,
    vector_len: int,
    bits: int,
    backend: str,
    device: torch.device,
    *,
    tiered: bool = False,
    tier_bits: list[int] | None = None,
    zero_escape: bool = False,
) -> nn.Module:
    """Create an NPVQLinear from random packed indices."""
    from ..nn.np_vq_linear import NPVQLinear, VQProxy
    from ..transforms.vq_packing import (
        IndexPacking,
        NTieredIndexPacking,
        TieredIndexPacking,
        ZeroEscapePacking,
    )

    centroids = torch.randn(
        num_centroids,
        vector_len,
        dtype=torch.float16,
        device=device,
    )
    proxy = VQProxy(centroids)

    n_groups = in_features // vector_len
    n_vectors = out_features * n_groups
    if tiered or tier_bits is not None or zero_escape:
        # Zipf-skewed indices - realistic for tiered encoding.
        indices = _zipf_indices(n_vectors, num_centroids, s=1.2)
    else:
        indices = torch.randint(0, num_centroids, (n_vectors,), dtype=torch.int64)

    if zero_escape:
        cold_bits = bits
        packed = ZeroEscapePacking.pack(indices, cold_bits)
        zesc_row_offsets = ZeroEscapePacking.compute_row_offsets(
            packed, out_features, n_groups
        )
        mod = NPVQLinear(
            proxy=proxy,
            packed_indices=packed.to(device),
            bits=cold_bits,
            out_features=out_features,
            in_features=in_features,
            vector_len=vector_len,
            backend=backend,
            zero_escape=True,
            zesc_row_offsets=[o.to(device) for o in zesc_row_offsets],
        )
    elif tier_bits is not None:
        packed = NTieredIndexPacking.pack(indices, tier_bits)
        ntier_row_offsets = NTieredIndexPacking.compute_row_offsets(
            packed, out_features, n_groups
        )
        mod = NPVQLinear(
            proxy=proxy,
            packed_indices=packed.to(device),
            bits=bits,
            out_features=out_features,
            in_features=in_features,
            vector_len=vector_len,
            backend=backend,
            tier_bits=tier_bits,
            ntier_row_offsets=[o.to(device) for o in ntier_row_offsets],
        )
    elif tiered:
        cold_bits = bits
        packed = TieredIndexPacking.pack(indices, cold_bits)
        hot_off, cold_off = TieredIndexPacking.compute_row_offsets(
            packed, out_features, n_groups
        )
        mod = NPVQLinear(
            proxy=proxy,
            packed_indices=packed.to(device),
            bits=cold_bits,
            out_features=out_features,
            in_features=in_features,
            vector_len=vector_len,
            backend=backend,
            use_tiered=True,
            cold_bits=cold_bits,
            hot_row_offsets=hot_off.to(device),
            cold_row_offsets=cold_off.to(device),
        )
    else:
        packed = IndexPacking.pack(indices.flatten(), bits)
        mod = NPVQLinear(
            proxy=proxy,
            packed_indices=packed.to(device),
            bits=bits,
            out_features=out_features,
            in_features=in_features,
            vector_len=vector_len,
            backend=backend,
        )

    proxy.to(device)
    return mod


def _make_np_embedding(
    num_embeddings: int,
    embedding_dim: int,
    nbits: int,
    group_size: int,
    qtype: str,
    backend: str,
    device: torch.device,
    dequant_dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Create an NPEmbedding from random packed weights."""
    from ..nn.np_embedding import NPEmbedding

    weight_q = None
    if backend == "default":
        weight_q = _make_qtensor(
            num_embeddings,
            embedding_dim,
            nbits,
            group_size,
            qtype,
            device,
            dequant_dtype=dequant_dtype,
        )

    vpb = 8 // nbits
    u8_blob = torch.randint(
        0,
        256,
        (num_embeddings, embedding_dim // vpb),
        dtype=torch.uint8,
        device=device,
    )
    n_groups = embedding_dim // group_size
    scale = torch.randn(
        num_embeddings * n_groups,
        dtype=torch.float16,
        device=device,
    )
    zero_point = None
    if qtype == "asym":
        zero_point = torch.randn(
            num_embeddings * n_groups,
            dtype=torch.float16,
            device=device,
        )
    return NPEmbedding(
        u8_blob=u8_blob,
        scale=scale,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        group_size=group_size,
        nbits=nbits,
        qtype=qtype,
        backend=backend,
        zero_point=zero_point,
        weight_q=weight_q,
    )


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BackendCfg:
    """A backend + dtype combination to benchmark.

    Attributes:
        name: Backend name passed to NP* constructors.
        x_dtype: Input tensor dtype.
        dequant_dtype: QTensor dequant target (only used by ``default``).
        label: Human-readable label for results table.
    """

    name: str
    x_dtype: torch.dtype
    dequant_dtype: torch.dtype
    label: str


def _available_linear_backends(
    device: torch.device,
) -> list[_BackendCfg]:
    """Return backend configs available for NPLinear on *device*."""
    cfgs = [
        _BackendCfg("default", torch.float32, torch.float32, "default(f32)"),
        _BackendCfg("default", torch.float16, torch.float16, "default(f16)"),
    ]
    if device.type == "cuda":
        f16 = torch.float16
        cfgs += [
            _BackendCfg("compile", f16, f16, "compile"),
            _BackendCfg("cuda", f16, f16, "cuda"),
        ]
        try:
            import gemlite  # noqa: F401

            cfgs.append(_BackendCfg("gemlite", f16, f16, "gemlite"))
        except ImportError:
            pass
    return cfgs


def _available_vq_backends(
    device: torch.device,
    *,
    tiered: bool = False,
    ntier: bool = False,
    zero_escape: bool = False,
) -> list[_BackendCfg]:
    """Return backend configs available for NPVQLinear on *device*.

    For tiered/ntier/zero-escape modes, only CUDA backends are returned -
    the default (CPU-unpack) path is too slow for benchmarking at scale.

    Args:
        device: Target device.
        tiered: Whether to return 2-tier-specific backends.
        ntier: Whether to return n-tier-specific backends.
        zero_escape: Whether to return zero-escape backends.

    Returns:
        List of backend configurations.
    """
    f16 = torch.float16
    if zero_escape:
        cfgs: list[_BackendCfg] = []
        if device.type == "cuda":
            cfgs.append(
                _BackendCfg("cuda_zero_escape", f16, f16, "cuda_zero_escape"),
            )
            cfgs.append(
                _BackendCfg("compile", f16, f16, "compile"),
            )
        return cfgs

    if ntier:
        # Skip compile — graph breaks in header parsing cause hangs.
        # Dedicated cuda_ntier kernel is the only viable backend.
        cfgs: list[_BackendCfg] = []
        if device.type == "cuda":
            cfgs.append(
                _BackendCfg("cuda_ntier", f16, f16, "cuda_ntier"),
            )
        return cfgs

    if tiered:
        # Skip compile and default — dedicated cuda_tiered kernel only.
        cfgs = []
        if device.type == "cuda":
            cfgs.append(
                _BackendCfg("cuda_tiered", f16, f16, "cuda_tiered"),
            )
        return cfgs

    cfgs = [
        _BackendCfg("default", torch.float32, torch.float32, "default(f32)"),
        _BackendCfg("default", torch.float16, torch.float16, "default(f16)"),
    ]
    if device.type == "cuda":
        cfgs += [
            _BackendCfg("compile", f16, f16, "compile"),
            _BackendCfg("cuda", f16, f16, "cuda"),
        ]
    return cfgs


def _available_emb_backends(
    device: torch.device,
) -> list[_BackendCfg]:
    """Return backend configs available for NPEmbedding on *device*."""
    cfgs = [
        _BackendCfg("default", torch.float32, torch.float32, "default(f32)"),
        _BackendCfg("default", torch.float16, torch.float16, "default(f16)"),
    ]
    if device.type == "cuda":
        f16 = torch.float16
        cfgs += [
            _BackendCfg("compile", f16, f16, "compile"),
            _BackendCfg("cuda", f16, f16, "cuda"),
        ]
    return cfgs


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------


def _resolve_shapes(
    preset: str,
    shapes_dict: dict[str, list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    """Return shapes for *preset* from *shapes_dict*."""
    if preset == "all":
        out: list[tuple[int, int]] = []
        for v in shapes_dict.values():
            out.extend(v)
        return out
    return shapes_dict[preset]


def _bench_linear(
    shapes: list[tuple[int, int]],
    batch_sizes: list[int],
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    """Benchmark NPLinear across shapes x backends."""
    backends = _available_linear_backends(device)
    results: list[BenchResult] = []

    configs = [
        ("Q4_0 sym", 4, 32, "sym"),
        ("Q2_0 sym", 2, 32, "sym"),
        ("Q4_1 asym", 4, 32, "asym"),
    ]

    for label, nbits, gs, qtype in configs:
        for n, k in shapes:
            if k % gs != 0:
                continue
            for bs in batch_sizes:
                for bc in backends:
                    try:
                        mod = _make_np_linear(
                            n,
                            k,
                            nbits,
                            gs,
                            qtype,
                            bc.name,
                            device,
                            bc.dequant_dtype,
                        )
                    except Exception as e:
                        print(
                            f"  ERR NPLinear {label} {n}x{k} {bc.label}: {e}",
                            file=sys.stderr,
                        )
                        continue
                    make_x = functools.partial(
                        _make_randn,
                        bs,
                        k,
                        bc.x_dtype,
                        device,
                    )
                    try:
                        t = _bench_forward(mod, make_x, warmup, iters)
                    except Exception as e:
                        print(
                            f"  ERR NPLinear {label} {n}x{k} {bc.label}: {e}",
                            file=sys.stderr,
                        )
                    else:
                        results.append(
                            BenchResult(
                                module_type=f"NPLinear ({label})",
                                shape=f"{n}x{k}",
                                batch=bs,
                                backend=bc.label,
                                median_ms=statistics.median(t),
                                min_ms=min(t),
                                timings=t,
                            )
                        )
                    finally:
                        del mod
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    return results


def _bench_vq(
    shapes: list[tuple[int, int]],
    batch_sizes: list[int],
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    """Benchmark NPVQLinear across shapes x backends.

    Sweeps multiple VQ configurations (codebook size × vector_len)
    and both fixed and tiered packing modes.  Tiered configs use
    Zipf-skewed index distributions for realistic compression ratios.

    Args:
        shapes: List of ``(out_features, in_features)`` pairs.
        batch_sizes: Input batch sizes to test.
        device: Target device.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        List of :class:`BenchResult` entries.
    """
    results: list[BenchResult] = []

    # --- Fixed, 2-tier, n-tier, and zero-escape configs ---
    # (label, num_centroids, vector_len, bits, tiered, tier_bits, zesc)
    configs: list[tuple[str, int, int, int, bool, list[int] | None, bool]] = [
        ("VQ 256c VL=4 fixed", 256, 4, 8, False, None, False),
        ("VQ 256c VL=8 fixed", 256, 8, 8, False, None, False),
        ("VQ 256c VL=4 tiered", 256, 4, 8, True, None, False),
        ("VQ 256c VL=8 tiered", 256, 8, 8, True, None, False),
        # N-tier CBS configs (3-tier and 4-tier).
        ("VQ 4Kc VL=4 3tier", 4096, 4, 12, False, [4, 8, 12], False),
        ("VQ 4Kc VL=8 3tier", 4096, 8, 12, False, [4, 8, 12], False),
        ("VQ 64Kc VL=4 4tier", 65536, 4, 16, False, [4, 8, 12, 16], False),
        # Zero-escape configs (compile backend, Zipf-skewed).
        ("VQ 1Kc VL=4 zesc", 1024, 4, 10, False, None, True),
        ("VQ 4Kc VL=4 zesc", 4096, 4, 12, False, None, True),
        ("VQ 1Kc VL=8 zesc", 1024, 8, 10, False, None, True),
    ]

    for cfg in configs:
        label, num_centroids, vector_len, bits, tiered, tier_bits, zesc = cfg
        is_ntier = tier_bits is not None
        backends = _available_vq_backends(
            device,
            tiered=tiered,
            ntier=is_ntier,
            zero_escape=zesc,
        )
        if not backends:
            continue
        for n, k in shapes:
            if k % vector_len != 0:
                continue
            for bs in batch_sizes:
                for bc in backends:
                    try:
                        mod = _make_np_vq_linear(
                            n,
                            k,
                            num_centroids,
                            vector_len,
                            bits,
                            bc.name,
                            device,
                            tiered=tiered,
                            tier_bits=tier_bits,
                            zero_escape=zesc,
                        )
                    except Exception as e:
                        print(
                            f"  ERR NPVQLinear {label} {n}x{k} {bc.label}: {e}",
                            file=sys.stderr,
                        )
                        continue
                    make_x = functools.partial(
                        _make_randn,
                        bs,
                        k,
                        bc.x_dtype,
                        device,
                    )
                    try:
                        t = _bench_forward(mod, make_x, warmup, iters)
                    except Exception as e:
                        print(
                            f"  ERR NPVQLinear {label} {n}x{k} {bc.label}: {e}",
                            file=sys.stderr,
                        )
                    else:
                        results.append(
                            BenchResult(
                                module_type=f"NPVQLinear ({label})",
                                shape=f"{n}x{k}",
                                batch=bs,
                                backend=bc.label,
                                median_ms=statistics.median(t),
                                min_ms=min(t),
                                timings=t,
                            )
                        )
                    finally:
                        del mod
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
    return results


def _bench_embedding(
    shapes: list[tuple[int, int]],
    batch_sizes: list[int],
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[BenchResult]:
    """Benchmark NPEmbedding across shapes x backends."""
    backends = _available_emb_backends(device)
    results: list[BenchResult] = []
    nbits = 4
    gs = 32
    qtype = "sym"

    for vocab, dim in shapes:
        if dim % gs != 0:
            continue
        for bs in batch_sizes:
            for bc in backends:
                try:
                    mod = _make_np_embedding(
                        vocab,
                        dim,
                        nbits,
                        gs,
                        qtype,
                        bc.name,
                        device,
                        bc.dequant_dtype,
                    )
                except Exception as e:
                    print(
                        f"  ERR NPEmbedding {vocab}x{dim} {bc.label}: {e}",
                        file=sys.stderr,
                    )
                    continue
                make_x = functools.partial(
                    _make_randint,
                    vocab,
                    bs,
                    device,
                )
                try:
                    t = _bench_forward(mod, make_x, warmup, iters)
                except Exception as e:
                    print(
                        f"  ERR NPEmbedding {vocab}x{dim} {bc.label}: {e}",
                        file=sys.stderr,
                    )
                else:
                    results.append(
                        BenchResult(
                            module_type="NPEmbedding (Q4_0 sym)",
                            shape=f"{vocab}x{dim}",
                            batch=bs,
                            backend=bc.label,
                            median_ms=statistics.median(t),
                            min_ms=min(t),
                            timings=t,
                        )
                    )
                finally:
                    del mod
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _print_table(results: list[BenchResult]) -> None:
    """Print results as a grouped rich table."""
    from rich.console import Console
    from rich.table import Table

    if not results:
        return

    console = Console()

    # Group by module_type.
    groups: dict[str, list[BenchResult]] = {}
    for r in results:
        groups.setdefault(r.module_type, []).append(r)

    for mod_type, rows in groups.items():
        backends = sorted({r.backend for r in rows})

        table = Table(title=mod_type, show_lines=True)
        table.add_column("Shape", style="cyan", no_wrap=True)
        table.add_column("Batch", justify="right")
        for bk in backends:
            table.add_column(f"{bk} (ms)", justify="right")
        table.add_column("fastest", style="bold green")

        combos: dict[tuple[str, int], dict[str, float]] = {}
        for r in rows:
            key = (r.shape, r.batch)
            combos.setdefault(key, {})[r.backend] = r.median_ms

        for (shape, batch), bk_times in sorted(combos.items()):
            cells = [shape, str(batch)]
            best_bk = min(
                bk_times,
                key=bk_times.get,  # type: ignore[arg-type]
            )
            for bk in backends:
                ms = bk_times.get(bk)
                cells.append(f"{ms:.3f}" if ms is not None else "-")
            cells.append(best_bk)
            table.add_row(*cells)

        console.print(table)
        console.print()


def _short_type(module_type: str) -> str:
    """Shorten module type for summary column headers.

    Args:
        module_type: Full module type string like ``"NPLinear (Q4_0 sym)"``.

    Returns:
        Short form like ``"Lin Q4_0s"``.
    """
    # Extract what's in parentheses.
    if "(" in module_type:
        inner = module_type.split("(", 1)[1].rstrip(")")
    else:
        inner = module_type

    if "NPLinear" in module_type:
        return f"Lin {inner}"
    if "NPVQLinear" in module_type:
        return inner
    if "NPEmbedding" in module_type:
        return f"Emb {inner}"
    return inner


def _print_summary(results: list[BenchResult]) -> None:
    """Print a summary table: best backend per quant method × shape × batch.

    Groups all results by (shape, batch) and shows, for each quant method,
    the fastest backend and its median time.
    """
    from rich.console import Console
    from rich.table import Table

    if not results:
        return

    # Use a wide console to prevent cell wrapping.
    console = Console(width=max(120, Console().width))

    # Collect all module types (quant methods).
    all_types = list(dict.fromkeys(r.module_type for r in results))

    # For each (shape, batch, module_type), find the best backend.
    best: dict[tuple[str, int, str], tuple[str, float]] = {}
    for r in results:
        key = (r.shape, r.batch, r.module_type)
        prev = best.get(key)
        if prev is None or r.median_ms < prev[1]:
            best[key] = (r.backend, r.median_ms)

    # Collect unique (shape, batch) combos.
    combos = sorted(
        {(r.shape, r.batch) for r in results},
    )

    table = Table(
        title="Best backend per quant method",
        show_lines=True,
        expand=True,
    )
    table.add_column("Shape", style="cyan", no_wrap=True)
    table.add_column("Bs", justify="right", no_wrap=True)
    for mt in all_types:
        table.add_column(
            _short_type(mt),
            justify="right",
            no_wrap=True,
        )

    for shape, batch in combos:
        cells: list[str] = [shape, str(batch)]
        for mt in all_types:
            entry = best.get((shape, batch, mt))
            if entry is None:
                cells.append("-")
            else:
                bk, ms = entry
                cells.append(f"{ms:.1f}ms {bk}")
        table.add_row(*cells)

    console.print(table)
    console.print()


def _print_csv(results: list[BenchResult]) -> None:
    """Print results as CSV to stdout."""
    print("module_type,shape,batch,backend,median_ms,min_ms")
    for r in results:
        print(
            f"{r.module_type},{r.shape},{r.batch},"
            f"{r.backend},{r.median_ms:.4f},{r.min_ms:.4f}"
        )


def _print_json(results: list[BenchResult]) -> None:
    """Print results as JSON to stdout."""
    json.dump(_results_to_dicts(results), sys.stdout, indent=2)
    print()


def _results_to_dicts(results: list[BenchResult]) -> list[dict[str, object]]:
    return [
        {
            "module_type": r.module_type,
            "shape": r.shape,
            "batch": r.batch,
            "backend": r.backend,
            "median_ms": round(r.median_ms, 4),
            "min_ms": round(r.min_ms, 4),
        }
        for r in results
    ]


def _cpu_model_name() -> str:
    """Read CPU model name from ``/proc/cpuinfo`` (Linux) or sysctl (macOS).

    Returns:
        CPU model string, e.g. ``"AMD Ryzen 9 7950X"``, or ``"unknown"``.
    """
    import platform

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif platform.system() == "Darwin":
        import subprocess

        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _collect_hw_meta(device: torch.device) -> dict[str, object]:
    """Collect hardware metadata for the current machine.

    Args:
        device: Torch device used for benchmarking.

    Returns:
        Dict with hostname, CPU model, core count, GPU info, and git version.
    """
    import os

    from neuropress.meta import collect_run_meta

    meta: dict[str, object] = {
        **collect_run_meta(),
        "cpu_model": _cpu_model_name(),
        "cpu_cores": os.cpu_count(),
        "device_type": device.type,
    }

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        meta["gpu_name"] = props.name
        meta["gpu_mem_gb"] = round(props.total_memory / 1024**3, 1)
        meta["gpu_compute_capability"] = f"{props.major}.{props.minor}"

    return meta


def _hw_slug(meta: dict[str, object]) -> str:
    """Derive a short filesystem-safe slug from hardware metadata.

    Args:
        meta: Hardware metadata from :func:`_collect_hw_meta`.

    Returns:
        Slug like ``"forge_GTX-1080-Ti"`` or ``"cloud_RTX-4090"``.
    """
    host = str(meta.get("hostname", "unknown")).split(".")[0]
    gpu = str(meta.get("gpu_name", "cpu"))
    # Sanitise for paths: spaces→dashes, drop special chars.
    gpu = gpu.replace(" ", "-").replace("/", "-")
    return f"{host}_{gpu}"


def _write_json_file(
    results: list[BenchResult],
    path: str,
    device: torch.device,
) -> None:
    """Write results + hardware metadata to a JSON file."""
    meta = _collect_hw_meta(device)
    doc = {"meta": meta, "results": _results_to_dicts(results)}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# Full-model benchmark
# ---------------------------------------------------------------------------


def _parse_backend_list(s: str) -> list[str]:
    """Parse comma-separated backend names."""
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_hf_model(
    model_id: str,
    *,
    device: str,
    dtype: str = "auto",
) -> tuple[nn.Module, Any]:
    """Load a HuggingFace CausalLM and its tokenizer.

    Args:
        model_id: HuggingFace model ID or local path.
        device: PyTorch device string.
        dtype: One of ``"auto"``, ``"float16"``, ``"bfloat16"``,
            ``"float32"``.

    Returns:
        Tuple ``(model, tokenizer)``.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(
            "ERROR: transformers is required for model benchmarks.\n"
            "  Install with: pip install 'neuropress[eval]'",
            file=sys.stderr,
        )
        sys.exit(1)

    dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtypes.get(dtype),
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _bench_model_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup: int,
    iters: int,
) -> list[float]:
    """Run model forward passes and return per-iteration ms.

    Args:
        model: Model to benchmark.
        input_ids: Input token IDs ``[batch, seq_len]``.
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        List of per-iteration times in milliseconds.
    """
    device = input_ids.device

    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings: list[float] = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(input_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - t0) * 1000.0)

    return timings


def _bench_one_backend(
    args: argparse.Namespace,
    backend: str,
    checkpoint_path: Path,
    input_ids: torch.Tensor,
    device: torch.device,
    load_quantized_model: Any,
    auto_kernel_lowering_cls: type,
) -> BenchResult | None:
    """Load model from checkpoint and benchmark a single backend.

    Args:
        args: Parsed CLI namespace.
        backend: Backend name to test.
        checkpoint_path: Path to quantized checkpoint.
        input_ids: Pre-generated input token IDs.
        device: Target device.
        load_quantized_model: The load function (avoids top-level import).
        auto_kernel_lowering_cls: AutoKernelLowering class.

    Returns:
        A BenchResult, or None if the backend failed.
    """
    print(f"\n--- Backend: {backend} ---")
    print("  Loading model from checkpoint...")
    model, _ = _load_hf_model(
        args.model,
        device="cpu",
        dtype=args.dtype,
    )
    load_quantized_model(
        model,
        checkpoint_path,
        kernel_lowering=None,
    )

    lowering = auto_kernel_lowering_cls(
        linear_backend=backend,
        vq_backend=backend,
        embedding_backend=backend,
    )
    try:
        lowering.apply(model)
        model.to(device)
        model.eval()

        print(f"  Benchmarking ({args.warmup} warmup + {args.iters} iters)...")
        t = _bench_model_forward(
            model,
            input_ids,
            args.warmup,
            args.iters,
        )
    except Exception as exc:
        print(f"  SKIP: {exc}")
        return None
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    med = statistics.median(t)
    mn = min(t)
    tok_per_s = args.batch_size * args.seq_len / (med / 1000.0)
    print(f"  median={med:.2f}ms  min={mn:.2f}ms  ({tok_per_s:.0f} tok/s)")

    return BenchResult(
        module_type="model",
        shape=(f"{args.model} bs={args.batch_size} seq={args.seq_len}"),
        batch=args.batch_size,
        backend=backend,
        median_ms=med,
        min_ms=mn,
        timings=t,
    )


def _bench_model(args: argparse.Namespace) -> list[BenchResult]:
    """Benchmark a full model with different kernel backends.

    Args:
        args: Parsed CLI namespace from the ``model`` subcommand.

    Returns:
        List of benchmark results, one per backend.
    """
    from neuropress.kernel_lowering import AutoKernelLowering, unlower_model
    from neuropress.t2n import NEUROPRESS_COMPRESSION
    from neuropress.t2n.model_io import load_quantized_model, save_quantized_model

    device = torch.device(args.device)
    backends = args.backends

    # -- Print header --------------------------------------------------------
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        mem_gb = props.total_memory / 1024**3
        print(f"Device: {props.name} ({mem_gb:.1f} GB)")
    else:
        print("Device: CPU (CUDA backends skipped)")
    print(f"Model: {args.model}")
    print(f"Backends: {', '.join(backends)}")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}")
    print(f"Seq len: {args.seq_len}  Batch: {args.batch_size}")
    print()

    # -- Resolve checkpoint path ---------------------------------------------
    checkpoint_path: Path | None = None
    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint_path = Path(args.checkpoint)
    tmp_dir = None

    # -- Load base model + tokenizer (for quantization or vocab size) --------
    print("Loading model...")
    model, tokenizer = _load_hf_model(
        args.model,
        device=str(device),
        dtype=args.dtype,
    )

    # -- Quantize or load checkpoint -----------------------------------------
    if checkpoint_path is not None:
        print(f"Loading pre-quantized checkpoint: {checkpoint_path}")
        load_quantized_model(model, checkpoint_path, kernel_lowering=None)
    else:
        if args.compression not in NEUROPRESS_COMPRESSION:
            available = "\n  ".join(sorted(NEUROPRESS_COMPRESSION))
            print(
                f"ERROR: Unknown compression {args.compression!r}.\n"
                f"Available:\n  {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Quantizing with {args.compression}...")
        compress_fn = NEUROPRESS_COMPRESSION[args.compression]
        model = compress_fn(model, tokenizer=tokenizer)
        # Strip kernel lowering — we'll apply per-backend below.
        unlower_model(model)

    # Save checkpoint for backend iteration.
    if args.save_checkpoint:
        checkpoint_path = Path(args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to {checkpoint_path}...")
        save_quantized_model(model, checkpoint_path)
        print("Checkpoint saved.")

    if checkpoint_path is None:
        # No user-provided path — save to temp file for reloading.
        tmp_dir = tempfile.mkdtemp(prefix="neuropress_bench_")
        checkpoint_path = Path(tmp_dir) / "model.safetensors"
        print("Saving temporary checkpoint for backend iteration...")
        save_quantized_model(model, checkpoint_path)

    # Free the initial model before per-backend iteration.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    vocab_size = tokenizer.vocab_size or 32000
    input_ids = torch.randint(
        0,
        vocab_size,
        (args.batch_size, args.seq_len),
        device=device,
    )

    # -- Benchmark each backend ----------------------------------------------
    results: list[BenchResult] = []
    for backend in backends:
        result = _bench_one_backend(
            args,
            backend,
            checkpoint_path,
            input_ids,
            device,
            load_quantized_model,
            AutoKernelLowering,
        )
        if result is not None:
            results.append(result)

    # Clean up temp checkpoint.
    if tmp_dir is not None:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both subcommands.

    Args:
        parser: Argument parser to add arguments to.
    """
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda:0 if available).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations (default: 5).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed iterations (default: 20).",
    )
    parser.add_argument(
        "--output",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        metavar="PATH",
        help="Write JSON results to file (in addition to stdout output).",
    )


def _run_synthetic(args: argparse.Namespace) -> None:
    """Run the synthetic layer benchmark.

    Args:
        args: Parsed CLI namespace.
    """
    device = torch.device(args.device)
    all_results: list[BenchResult] = []

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        mem_gb = props.total_memory / 1024**3
        print(f"Device: {props.name} ({mem_gb:.1f} GB)")
    else:
        print("Device: CPU (CUDA backends skipped)")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}")
    print()

    run_linear = args.module in ("linear", "all")
    run_vq = args.module in ("vq", "all")
    run_emb = args.module in ("embedding", "all")
    n_batches = len(args.batch_sizes)

    if run_linear:
        shapes = _resolve_shapes(args.shapes, _LINEAR_SHAPES)
        n = len(shapes)
        print(f"Benchmarking NPLinear ({n} shapes x {n_batches} batches)...")
        all_results.extend(
            _bench_linear(
                shapes,
                args.batch_sizes,
                device,
                args.warmup,
                args.iters,
            )
        )

    if run_vq:
        shapes = _resolve_shapes(args.shapes, _LINEAR_SHAPES)
        n = len(shapes)
        print(f"Benchmarking NPVQLinear ({n} shapes x {n_batches} batches)...")
        all_results.extend(
            _bench_vq(
                shapes,
                args.batch_sizes,
                device,
                args.warmup,
                args.iters,
            )
        )

    if run_emb:
        emb_shapes = _resolve_shapes(args.shapes, _EMBEDDING_SHAPES)
        n = len(emb_shapes)
        print(f"Benchmarking NPEmbedding ({n} shapes x {n_batches} batches)...")
        all_results.extend(
            _bench_embedding(
                emb_shapes,
                args.batch_sizes,
                device,
                args.warmup,
                args.iters,
            )
        )

    _output_results(all_results, args, device)


def _run_model(args: argparse.Namespace) -> None:
    """Run the full-model benchmark.

    Args:
        args: Parsed CLI namespace.
    """
    device = torch.device(args.device)
    results = _bench_model(args)
    _output_results(results, args, device)


def _print_hw_header(meta: dict[str, object]) -> None:
    """Print a compact hardware summary line before benchmark results.

    Args:
        meta: Hardware metadata from :func:`_collect_hw_meta`.
    """
    parts = [f"Host: {meta.get('hostname', '?')}"]
    cpu = meta.get("cpu_model", "unknown")
    cores = meta.get("cpu_cores")
    if cores:
        parts.append(f"CPU: {cpu} ({cores} cores)")
    else:
        parts.append(f"CPU: {cpu}")
    if meta.get("gpu_name"):
        gpu_str = str(meta["gpu_name"])
        mem = meta.get("gpu_mem_gb")
        if mem:
            gpu_str += f" ({mem} GB)"
        cc = meta.get("gpu_compute_capability")
        if cc:
            gpu_str += f" [sm {cc}]"
        parts.append(f"GPU: {gpu_str}")
    commit = meta.get("git_commit")
    if commit:
        parts.append(f"Commit: {str(commit)[:8]}")
    print(" | ".join(parts))


def _output_results(
    results: list[BenchResult],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Format and output benchmark results.

    Args:
        results: Benchmark measurements.
        args: Parsed CLI namespace (for output/output_file).
        device: Torch device (for JSON metadata).
    """
    meta = _collect_hw_meta(device)
    _print_hw_header(meta)
    print()
    if args.output == "table":
        _print_table(results)
        _print_summary(results)
    elif args.output == "csv":
        _print_csv(results)
    else:
        _print_json(results)

    if args.output_file is not None:
        _write_json_file(results, args.output_file, device)


def main() -> None:
    """Entry point for ``neuropress-bench``."""
    parser = argparse.ArgumentParser(
        prog="neuropress-bench",
        description="Benchmark suite for neuropress quantized kernels.",
    )
    sub = parser.add_subparsers(dest="command")

    # -- synthetic (default) -------------------------------------------------
    syn = sub.add_parser(
        "synthetic",
        help="Synthetic layer micro-benchmark (default).",
    )
    _add_common_args(syn)
    syn.add_argument(
        "--module",
        choices=["linear", "vq", "embedding", "all"],
        default="all",
        help="Module type to benchmark (default: all).",
    )
    syn.add_argument(
        "--shapes",
        choices=["small", "medium", "large", "xlarge", "all"],
        default="all",
        help="Shape preset (default: all).",
    )
    syn.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=_DEFAULT_BATCH_SIZES,
        help="Comma-separated batch sizes (default: 1,16,256).",
    )

    # -- model ---------------------------------------------------------------
    mdl = sub.add_parser(
        "model",
        help="Full-model end-to-end benchmark.",
    )
    _add_common_args(mdl)
    mdl.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or local path.",
    )
    mdl.add_argument(
        "--compression",
        required=True,
        help="Compression method (key in NEUROPRESS_COMPRESSION).",
    )
    mdl.add_argument(
        "--checkpoint",
        default=None,
        help="Pre-quantized .safetensors checkpoint (skip quantization).",
    )
    mdl.add_argument(
        "--save-checkpoint",
        default=None,
        help="Save quantized checkpoint to this path for reuse.",
    )
    mdl.add_argument(
        "--backends",
        type=_parse_backend_list,
        default=["default", "compile", "cuda"],
        help="Comma-separated backends (default: default,compile,cuda).",
    )
    mdl.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Input sequence length (default: 128).",
    )
    mdl.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1).",
    )
    mdl.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype (default: auto).",
    )

    # If the first positional arg isn't a known subcommand, default to
    # synthetic so that bare `neuropress-bench --module linear` still works.
    argv = sys.argv[1:]
    if not argv or (argv[0] not in ("synthetic", "model", "-h", "--help")):
        argv = ["synthetic", *argv]

    args = parser.parse_args(argv)

    if args.command == "synthetic":
        _run_synthetic(args)
    elif args.command == "model":
        _run_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
