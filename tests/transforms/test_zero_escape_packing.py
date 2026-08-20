"""Tests for ZeroEscapePacking (Phase 5) and VQScheme.zero_escape.

Covers:
- ZeroEscapePacking.pack / unpack round-trips
- Header parsing and format detection
- effective_bits savings estimation
- compute_row_offsets correctness
- VQScheme(zero_escape=True) quantize -> dequantize
- Savings vs fixed-width at high centroid-0 fractions
"""

import math

import pytest
import torch

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.vq import ProxySelector, VQScheme
from neuropress.transforms.vq_packing import ZeroEscapePacking

# ---------------------------------------------------------------------------
# ZeroEscapePacking.pack / unpack round-trips
# ---------------------------------------------------------------------------


def test_zero_escape_roundtrip_all_zero():
    """All centroid-0 indices must round-trip exactly."""
    n = 64
    indices = torch.zeros(n, dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=8)
    restored = ZeroEscapePacking.unpack(packed)
    assert (restored == indices).all()


def test_zero_escape_roundtrip_all_hot():
    """All tier-1 indices (1-16) must round-trip exactly."""
    n = 64
    indices = torch.randint(1, 17, (n,), dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=8)
    restored = ZeroEscapePacking.unpack(packed)
    assert (restored == indices).all()


def test_zero_escape_roundtrip_all_warm():
    """All tier-2 indices (17-256) must round-trip exactly."""
    n = 64
    indices = torch.randint(17, 257, (n,), dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=10)
    restored = ZeroEscapePacking.unpack(packed)
    assert (restored == indices).all()


def test_zero_escape_roundtrip_all_cold():
    """All tier-3 indices (>= 257) must round-trip exactly."""
    n = 64
    indices = torch.randint(257, 1024, (n,), dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=10)
    restored = ZeroEscapePacking.unpack(packed)
    assert (restored == indices).all()


@pytest.mark.parametrize("cold_bits", [8, 10, 12])
def test_zero_escape_roundtrip_mixed(cold_bits: int):
    """Mixed indices across all tiers must round-trip exactly."""
    torch.manual_seed(cold_bits)
    max_val = min((1 << cold_bits) - 1 + ZeroEscapePacking.TIER_2_MAX, 4095)
    n = 200
    indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits)
    restored = ZeroEscapePacking.unpack(packed)
    assert (restored == indices).all(), (
        f"cold_bits={cold_bits}: round-trip failed -- "
        f"first mismatch at {(restored != indices).nonzero()[0].item()}"
    )


@pytest.mark.parametrize("n", [1, 3, 7, 17, 33, 99, 128])
def test_zero_escape_roundtrip_odd_counts(n: int):
    """Arbitrary index counts must round-trip."""
    indices = torch.arange(n, dtype=torch.int64) % 300
    packed = ZeroEscapePacking.pack(indices, cold_bits=10)
    restored = ZeroEscapePacking.unpack(packed)
    assert (restored == indices).all()


def test_zero_escape_pack_returns_uint8():
    """Packed buffer must be uint8."""
    indices = torch.zeros(16, dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=8)
    assert packed.dtype == torch.uint8


# ---------------------------------------------------------------------------
# Header parsing and format detection
# ---------------------------------------------------------------------------


def test_zero_escape_header_fields():
    """Header must encode counts correctly."""
    # 10 zeros, 20 hot (1-16), 15 warm (17-256), 5 cold (257+)
    zeros = torch.zeros(10, dtype=torch.int64)
    hot = torch.randint(1, 17, (20,), dtype=torch.int64)
    warm = torch.randint(17, 257, (15,), dtype=torch.int64)
    cold = torch.randint(257, 500, (5,), dtype=torch.int64)
    indices = torch.cat([zeros, hot, warm, cold])
    packed = ZeroEscapePacking.pack(indices, cold_bits=10)

    n, cb, n_zero, n_hot, n_warm, H = ZeroEscapePacking._parse_header(packed)
    assert n == 50
    assert n_zero == 10
    assert n_hot == 20
    assert n_warm == 15
    assert cb == 10
    assert H == ZeroEscapePacking.HEADER_BYTES


def test_zero_escape_is_detected():
    """is_zero_escape_buffer must detect the magic number."""
    indices = torch.zeros(16, dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=8)
    assert ZeroEscapePacking.is_zero_escape_buffer(packed)


def test_zero_escape_rejects_short_buffer():
    """Unpack must reject buffers shorter than the header."""
    with pytest.raises(ValueError, match="too short"):
        ZeroEscapePacking.unpack(torch.zeros(4, dtype=torch.uint8))


def test_zero_escape_rejects_bad_magic():
    """Unpack must reject buffers with wrong magic."""
    buf = torch.zeros(24, dtype=torch.uint8)
    with pytest.raises(ValueError, match="bad magic"):
        ZeroEscapePacking._parse_header(buf)


def test_zero_escape_pack_invalid_cold_bits():
    """Pack must reject cold_bits outside valid range."""
    with pytest.raises(ValueError):
        ZeroEscapePacking.pack(torch.zeros(4, dtype=torch.int64), cold_bits=3)
    with pytest.raises(ValueError):
        ZeroEscapePacking.pack(torch.zeros(4, dtype=torch.int64), cold_bits=21)


# ---------------------------------------------------------------------------
# effective_bits savings
# ---------------------------------------------------------------------------


def test_effective_bits_all_zero():
    """All centroid-0: 2 bits per index (prefix only, no payload)."""
    bpi = ZeroEscapePacking.effective_bits(
        n_zero=100, n_hot=0, n_warm=0, n_cold=0, cold_bits=8
    )
    assert abs(bpi - 2.0) < 1e-9


def test_effective_bits_all_hot():
    """All tier-1: 2 (prefix) + 4 (payload) = 6 bpi."""
    bpi = ZeroEscapePacking.effective_bits(
        n_zero=0, n_hot=100, n_warm=0, n_cold=0, cold_bits=8
    )
    assert abs(bpi - 6.0) < 1e-9


def test_effective_bits_all_warm():
    """All tier-2: 2 (prefix) + 8 (payload) = 10 bpi."""
    bpi = ZeroEscapePacking.effective_bits(
        n_zero=0, n_hot=0, n_warm=100, n_cold=0, cold_bits=8
    )
    assert abs(bpi - 10.0) < 1e-9


def test_effective_bits_all_cold():
    """All tier-3: 2 (prefix) + cold_bits bpi."""
    bpi = ZeroEscapePacking.effective_bits(
        n_zero=0, n_hot=0, n_warm=0, n_cold=100, cold_bits=12
    )
    assert abs(bpi - 14.0) < 1e-9


def test_effective_bits_savings_with_zero_fraction():
    """Higher centroid-0 fraction -> lower bpi."""
    bpis = [
        ZeroEscapePacking.effective_bits(
            n_zero=p, n_hot=100 - p, n_warm=0, n_cold=0, cold_bits=8
        )
        for p in [0, 25, 50, 75, 100]
    ]
    # More zeros = fewer bits (2 bpi for zero vs 6 bpi for hot).
    assert bpis == sorted(bpis, reverse=True)


# ---------------------------------------------------------------------------
# compute_row_offsets
# ---------------------------------------------------------------------------


def test_compute_row_offsets_correctness():
    """Row offsets must match manual counting."""
    # 4 rows, 8 groups each = 32 indices total.
    n_rows = 4
    n_groups = 8
    torch.manual_seed(42)
    indices = torch.randint(0, 300, (n_rows * n_groups,), dtype=torch.int64)
    packed = ZeroEscapePacking.pack(indices, cold_bits=10)
    offsets = ZeroEscapePacking.compute_row_offsets(packed, n_rows, n_groups)

    assert len(offsets) == 4
    for off in offsets:
        assert off.shape == (n_rows + 1,)
        assert off[0] == 0

    # Verify totals.
    n_zero = int((indices == 0).sum().item())
    n_hot = int(((indices >= 1) & (indices < 17)).sum().item())
    n_warm = int(((indices >= 17) & (indices < 257)).sum().item())
    n_cold = int((indices >= 257).sum().item())
    assert offsets[0][-1].item() == n_zero
    assert offsets[1][-1].item() == n_hot
    assert offsets[2][-1].item() == n_warm
    assert offsets[3][-1].item() == n_cold


# ---------------------------------------------------------------------------
# VQScheme(zero_escape=True)
# ---------------------------------------------------------------------------


def _make_zero_escape_scheme(
    nc: int = 300,
    vl: int = 4,
    out_f: int = 8,
    in_f: int = 16,
) -> tuple[VQScheme, torch.Tensor]:
    """Return a zero-escape VQScheme and a random weight tensor."""
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([out_f, in_f])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        zero_escape=True,
    )
    weight = torch.randn(out_f, in_f)
    return scheme, weight


def test_zero_escape_scheme_quantize_returns_uint8():
    """Quantized output must be uint8."""
    scheme, weight = _make_zero_escape_scheme()
    packed = scheme.quantize_as_u8(weight)
    assert packed.dtype == torch.uint8


def test_zero_escape_scheme_dequantize_shape():
    """Dequantized output must match original shape."""
    scheme, weight = _make_zero_escape_scheme(out_f=8, in_f=16)
    packed = scheme.quantize_as_u8(weight)
    dequant = scheme.dequantize(packed, torch.float32)
    assert dequant.shape == torch.Size([8, 16])


def test_zero_escape_scheme_dequantize_finite():
    """Dequantized values must be finite."""
    scheme, weight = _make_zero_escape_scheme()
    packed = scheme.quantize_as_u8(weight)
    dequant = scheme.dequantize(packed, torch.float32)
    assert dequant.isfinite().all()


def test_zero_escape_scheme_vectors_match_centroids():
    """Each dequantized vector must match one of the centroids."""
    nc, vl = 300, 4
    scheme, weight = _make_zero_escape_scheme(nc=nc, vl=vl, out_f=8, in_f=16)
    packed = scheme.quantize_as_u8(weight)
    dequant = scheme.dequantize(packed, torch.float32)
    vecs = dequant.reshape(-1, vl)
    centroids_f32 = scheme.centroids.float()
    for vec in vecs:
        diffs = (centroids_f32 - vec.unsqueeze(0)).norm(dim=1)
        assert diffs.min().item() < 1e-4


def test_zero_escape_scheme_matches_standard():
    """Zero-escape and standard schemes must dequantize to the same weight."""
    nc, vl, out_f, in_f = 300, 4, 8, 16
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([out_f, in_f])
    weight = torch.randn(out_f, in_f)
    scheme_std = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
    )
    scheme_ze = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        zero_escape=True,
    )
    dq_std = scheme_std.dequantize(scheme_std.quantize_as_u8(weight), torch.float32)
    dq_ze = scheme_ze.dequantize(scheme_ze.quantize_as_u8(weight), torch.float32)
    assert torch.allclose(dq_std, dq_ze, atol=1e-5), (
        "Zero-escape and standard dequantize must agree"
    )


def test_zero_escape_repr():
    """Repr must mention zero_escape."""
    scheme, _ = _make_zero_escape_scheme()
    r = repr(scheme)
    assert "zero_escape=True" in r


def test_zero_escape_buffer_smaller_when_skewed():
    """Zero-escape buffer must be smaller than fixed for skewed data."""
    nc, vl = 300, 4
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([32, 64])
    scheme_std = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
    )
    scheme_ze = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        zero_escape=True,
    )
    # Force all vectors to cluster around centroid-0.
    c0 = centroids[0].float().unsqueeze(0)
    weight = c0.repeat(32, 16).reshape(shape[0], shape[1])
    packed_std = scheme_std.quantize_as_u8(weight)
    packed_ze = scheme_ze.quantize_as_u8(weight)
    # All-zero: 2 bpi + 24 B header vs 10 bpi fixed.
    # packed_ze header (24 B) + prefix (2*512/8=128 B) = 152 B
    # packed_std = 512 * 10 / 8 = 640 B (for 10-bit indices on 300 centroids)
    assert packed_ze.numel() < packed_std.numel(), (
        f"Zero-escape ({packed_ze.numel()} B) should be smaller than "
        f"fixed ({packed_std.numel()} B) for all-centroid-0 data"
    )


# ---------------------------------------------------------------------------
# Byte count matches formula
# ---------------------------------------------------------------------------


def test_zero_escape_byte_count_matches_formula():
    """Actual packed size must match the theoretical formula."""
    cold_bits = 10
    n = 200
    torch.manual_seed(42)
    indices = torch.randint(0, 500, (n,), dtype=torch.int64)
    n_hot = int(((indices >= 1) & (indices < 17)).sum().item())
    n_warm = int(((indices >= 17) & (indices < 257)).sum().item())
    n_cold = int((indices >= 257).sum().item())

    packed = ZeroEscapePacking.pack(indices, cold_bits)

    expected_bytes = (
        ZeroEscapePacking.HEADER_BYTES
        + math.ceil(n * 2 / 8)  # prefix stream (2 bits each)
        + math.ceil(n_hot * 4 / 8)  # hot stream
        + n_warm  # warm stream (8-bit = 1 byte each)
        + math.ceil(n_cold * cold_bits / 8)  # cold stream
    )
    assert packed.numel() == expected_bytes
