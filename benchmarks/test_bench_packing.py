"""Benchmarks for BitPacking compress / decompress and QTensor linear inference.

Run all benchmarks::

    uv run pytest benchmarks/ --benchmark-only -v

    # Or by explicit file:
    uv run pytest benchmarks/test_bench_packing.py --benchmark-only -v

Save a baseline and compare later::

    uv run pytest benchmarks/ --benchmark-only --benchmark-save=baseline
    uv run pytest benchmarks/ --benchmark-only --benchmark-compare=baseline

The fixtures ``weight_fp``, ``weight_q``, and ``input_*`` are module-scoped so
quantization is performed once regardless of how many benchmark functions use them.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import neuropress.t2n  # noqa: F401  # register transforms as side-effect
from neuropress.nn import Backend, NPLinear
from neuropress.t2n.packing import BitPacking2, BitPacking4
from neuropress.transforms.weight_only import _make_q4_0_packed_tensor

# ---------------------------------------------------------------------------
# Weight / input fixtures
# ---------------------------------------------------------------------------

ROWS = 1024
COLS = 1024


def _inline_offload(q_fn, tensor, suffix):
    """Stand-in for offloaded_tensor_qtensor: calls _calib without offloading."""
    return q_fn(tensor)


@pytest.fixture(scope="module")
def weight_fp() -> torch.Tensor:
    """Full-precision reference weight."""
    torch.manual_seed(0)
    return torch.randn(ROWS, COLS)


@pytest.fixture(scope="module")
def weight_q(weight_fp: torch.Tensor) -> torch.Tensor:
    """Q4_0-packed QTensor weight built from *weight_fp*."""
    with patch(
        "neuropress.transforms.weight_only.offloaded_tensor_qtensor", _inline_offload
    ):
        return _make_q4_0_packed_tensor(weight_fp, "bench")


@pytest.fixture(scope="module")
def u8_2d() -> torch.Tensor:
    """Random uint8 tensor with shape (ROWS, COLS) — input to compress."""
    return torch.randint(0, 16, (ROWS, COLS), dtype=torch.uint8)


@pytest.fixture(scope="module")
def packed_q4(u8_2d: torch.Tensor) -> torch.Tensor:
    """Q4-packed version of *u8_2d* — input to decompress."""
    return BitPacking4.compress(u8_2d)


@pytest.fixture(scope="module")
def packed_q2(u8_2d: torch.Tensor) -> torch.Tensor:
    """Q2-packed version of *u8_2d* — input to decompress."""
    u8_q2 = torch.randint(0, 4, (ROWS, COLS), dtype=torch.uint8)
    return BitPacking2.compress(u8_q2)


# ---------------------------------------------------------------------------
# BitPacking benchmarks
# ---------------------------------------------------------------------------


def test_bitpacking4_compress(benchmark, u8_2d):
    """BitPacking4 compress: (1024, 1024) uint8 → (1024, 512)."""
    benchmark(BitPacking4.compress, u8_2d)


def test_bitpacking4_decompress(benchmark, packed_q4):
    """BitPacking4 decompress: (1024, 512) → (1024, 1024) uint8."""
    benchmark(BitPacking4.decompress, packed_q4)


def test_bitpacking2_compress(benchmark, u8_2d):
    """BitPacking2 compress: (1024, 1024) uint8 → (1024, 256)."""
    u8_q2 = torch.randint(0, 4, (ROWS, COLS), dtype=torch.uint8)
    benchmark(BitPacking2.compress, u8_q2)


def test_bitpacking2_decompress(benchmark, packed_q2):
    """BitPacking2 decompress: (1024, 256) → (1024, 1024) uint8."""
    benchmark(BitPacking2.decompress, packed_q2)


# ---------------------------------------------------------------------------
# QTensor decompress benchmarks
# ---------------------------------------------------------------------------


def test_qtensor_decompress(benchmark, weight_q):
    """Full QTensor decompress (dequant-to-u8 + scale → float)."""
    benchmark(weight_q.decompress)


# ---------------------------------------------------------------------------
# F.linear: float vs QTensor  (realistic decode & prefill shapes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch,seq",
    [(1, 1), (1, 8), (1, 64), (4, 512)],
    ids=["decode_1x1", "decode_1x8", "prefill_1x64", "prefill_4x512"],
)
def test_linear_float(benchmark, weight_fp, batch, seq):
    """F.linear with a plain float weight — baseline."""
    x = torch.randn(batch, seq, COLS)
    benchmark(F.linear, x, weight_fp)


@pytest.mark.parametrize(
    "batch,seq",
    [(1, 1), (1, 8), (1, 64), (4, 512)],
    ids=["decode_1x1", "decode_1x8", "prefill_1x64", "prefill_4x512"],
)
def test_linear_qtensor(benchmark, weight_q, batch, seq):
    """F.linear with a QTensor weight — decompress cost paid every call."""
    x = torch.randn(batch, seq, COLS)
    benchmark(F.linear, x, weight_q)


# ---------------------------------------------------------------------------
# NPLinear backends  (compile and cuda when available)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def np_linear_compile(weight_q):
    """NPLinear wrapping weight_q, compile backend."""
    linear = torch.nn.Linear(COLS, ROWS, bias=False)
    linear._parameters["weight"] = weight_q
    return NPLinear.from_linear(linear, backend=Backend.COMPILE)


@pytest.fixture(scope="module")
def np_linear_cuda(weight_q):
    """NPLinear wrapping weight_q, cuda backend (skipped when unavailable)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    linear = torch.nn.Linear(COLS, ROWS, bias=False)
    linear._parameters["weight"] = weight_q
    np_lin = NPLinear.from_linear(linear, backend=Backend.CUDA)
    # Move buffers to GPU.
    return np_lin.cuda()


@pytest.mark.parametrize(
    "batch,seq",
    [(1, 1), (1, 8), (1, 64), (4, 512)],
    ids=["decode_1x1", "decode_1x8", "prefill_1x64", "prefill_4x512"],
)
def test_linear_np_compile(benchmark, np_linear_compile, batch, seq):
    """NPLinear compile backend — torch.compile'd dequant + F.linear."""
    x = torch.randn(batch, seq, COLS)
    benchmark(np_linear_compile, x)


@pytest.mark.parametrize(
    "batch,seq",
    [(1, 1), (1, 8), (1, 64), (4, 512)],
    ids=["decode_1x1", "decode_1x8", "prefill_1x64", "prefill_4x512"],
)
def test_linear_np_cuda(benchmark, np_linear_cuda, batch, seq):
    """NPLinear cuda backend — fused CUDA kernel (skipped when CUDA unavailable)."""
    x = torch.randn(batch, seq, COLS, device="cuda")
    benchmark(np_linear_cuda, x)
