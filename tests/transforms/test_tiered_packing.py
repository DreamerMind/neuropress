"""Tests for TieredIndexPacking (Phase 4) and VQScheme.use_tiered.

Covers:
- TieredIndexPacking.pack / unpack round-trips
- Header parsing and stream layout
- effective_bits savings estimation
- VQScheme(use_tiered=True) quantize → dequantize
- Savings vs fixed-width at high hot-fractions
"""

import math

import pytest
import torch

# Must be imported first to break the transforms → t2n circular import.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.vq import ProxySelector, VQScheme
from neuropress.transforms.vq_packing import TieredIndexPacking

# ---------------------------------------------------------------------------
# TieredIndexPacking.pack / unpack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cold_bits", [4, 5, 6, 7, 8, 10, 12])
def test_tiered_roundtrip_all_hot(cold_bits: int):
    """All-hot indices (0–15) must round-trip exactly."""
    torch.manual_seed(cold_bits)
    n = 64
    hot = TieredIndexPacking.hot_threshold(TieredIndexPacking.DEFAULT_HOT_BITS)
    indices = torch.randint(0, hot, (n,), dtype=torch.int64)
    packed = TieredIndexPacking.pack(indices, cold_bits)
    restored = TieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


@pytest.mark.parametrize("cold_bits", [8, 10, 12])
def test_tiered_roundtrip_all_cold(cold_bits: int):
    """All-cold indices (≥ 16) must round-trip exactly."""
    torch.manual_seed(cold_bits + 100)
    max_val = (1 << cold_bits) - 1
    n = 64
    lo = TieredIndexPacking.hot_threshold(TieredIndexPacking.DEFAULT_HOT_BITS)
    indices = torch.randint(lo, max_val + 1, (n,), dtype=torch.int64)
    packed = TieredIndexPacking.pack(indices, cold_bits)
    restored = TieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


@pytest.mark.parametrize("cold_bits", [8, 10, 12])
def test_tiered_roundtrip_mixed(cold_bits: int):
    """Mixed hot + cold indices must round-trip exactly."""
    torch.manual_seed(cold_bits + 200)
    max_val = (1 << cold_bits) - 1
    n = 128
    indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int64)
    packed = TieredIndexPacking.pack(indices, cold_bits)
    restored = TieredIndexPacking.unpack(packed)
    assert (restored == indices).all(), (
        f"cold_bits={cold_bits}: round-trip failed — "
        f"first mismatch at {(restored != indices).nonzero()[0].item()}"
    )


@pytest.mark.parametrize("n", [1, 3, 7, 17, 33, 99, 128])
def test_tiered_roundtrip_odd_counts(n: int):
    """Arbitrary index counts must round-trip."""
    cold_bits = 8
    indices = torch.arange(n, dtype=torch.int64) % 256
    packed = TieredIndexPacking.pack(indices, cold_bits)
    restored = TieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


def test_tiered_pack_returns_uint8():
    indices = torch.zeros(16, dtype=torch.int64)
    packed = TieredIndexPacking.pack(indices, cold_bits=8)
    assert packed.dtype == torch.uint8


def test_tiered_pack_header_fields():
    """Header must encode n_indices, n_hot, cold_bits correctly."""
    n = 50
    cold_bits = 10
    # 30 hot (< 16) + 20 cold (>= 16)
    hot = torch.arange(0, 16, dtype=torch.int64).repeat(2)[:30]
    cold = torch.arange(16, 36, dtype=torch.int64)
    indices = torch.cat([hot, cold])
    assert indices.numel() == n
    packed = TieredIndexPacking.pack(indices, cold_bits)
    header = packed[: TieredIndexPacking.HEADER_BYTES].view(torch.int32)
    assert int(header[0].item()) == n
    assert int(header[1].item()) == 30
    assert int(header[2].item()) == cold_bits


def test_tiered_unpack_rejects_short_buffer():
    with pytest.raises(ValueError, match="too short"):
        TieredIndexPacking.unpack(torch.zeros(4, dtype=torch.uint8))


def test_tiered_pack_invalid_cold_bits():
    with pytest.raises(ValueError):
        TieredIndexPacking.pack(torch.zeros(4, dtype=torch.int64), cold_bits=3)
    with pytest.raises(ValueError):
        TieredIndexPacking.pack(torch.zeros(4, dtype=torch.int64), cold_bits=21)


# ---------------------------------------------------------------------------
# effective_bits savings
# ---------------------------------------------------------------------------


def test_effective_bits_all_hot():
    """All hot: 1 (prefix) + 4 (hot) = 5 bits per index."""
    bpi = TieredIndexPacking.effective_bits(n_hot=100, n_total=100, cold_bits=8)
    assert abs(bpi - 5.0) < 1e-9


def test_effective_bits_all_cold():
    """All cold: 1 (prefix) + cold_bits = 1 + 8 = 9 bits per index."""
    bpi = TieredIndexPacking.effective_bits(n_hot=0, n_total=100, cold_bits=8)
    assert abs(bpi - 9.0) < 1e-9


def test_effective_bits_half_half():
    """50% hot: 1 + 4*0.5 + 8*0.5 = 7 bpi < 8 fixed."""
    bpi = TieredIndexPacking.effective_bits(n_hot=50, n_total=100, cold_bits=8)
    assert abs(bpi - 7.0) < 1e-9
    assert bpi < 8.0


def test_effective_bits_savings_increase_with_hot_fraction():
    """Higher hot fraction → lower bpi (more savings)."""
    bpis = [
        TieredIndexPacking.effective_bits(n_hot=p, n_total=100, cold_bits=8)
        for p in [0, 25, 50, 75, 100]
    ]
    assert bpis == sorted(bpis, reverse=True)


# ---------------------------------------------------------------------------
# VQScheme(use_tiered=True)
# ---------------------------------------------------------------------------


def _make_tiered_scheme(
    nc: int = 64,
    vl: int = 4,
    out_f: int = 8,
    in_f: int = 16,
) -> tuple[VQScheme, torch.Tensor]:
    """Return a tiered VQScheme and a random weight tensor."""
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([out_f, in_f])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        use_tiered=True,
    )
    weight = torch.randn(out_f, in_f)
    return scheme, weight


def test_tiered_scheme_quantize_returns_uint8():
    scheme, weight = _make_tiered_scheme()
    packed = scheme.quantize_as_u8(weight)
    assert packed.dtype == torch.uint8


def test_tiered_scheme_dequantize_shape():
    scheme, weight = _make_tiered_scheme(out_f=8, in_f=16)
    packed = scheme.quantize_as_u8(weight)
    dequant = scheme.dequantize(packed, torch.float32)
    assert dequant.shape == torch.Size([8, 16])


def test_tiered_scheme_dequantize_finite():
    scheme, weight = _make_tiered_scheme()
    packed = scheme.quantize_as_u8(weight)
    dequant = scheme.dequantize(packed, torch.float32)
    assert dequant.isfinite().all()


def test_tiered_scheme_vectors_match_centroids():
    """Each dequantized vector must exactly equal one of the centroids."""
    nc, vl = 64, 4
    scheme, weight = _make_tiered_scheme(nc=nc, vl=vl, out_f=8, in_f=16)
    packed = scheme.quantize_as_u8(weight)
    dequant = scheme.dequantize(packed, torch.float32)
    vecs = dequant.reshape(-1, vl)
    centroids_f32 = scheme.centroids.float()
    for vec in vecs:
        diffs = (centroids_f32 - vec.unsqueeze(0)).norm(dim=1)
        assert diffs.min().item() < 1e-4


def test_tiered_scheme_matches_standard_scheme():
    """Tiered and standard schemes must dequantize to the same weight."""
    nc, vl, out_f, in_f = 64, 4, 8, 16
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([out_f, in_f])
    weight = torch.randn(out_f, in_f)
    scheme_std = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        use_tiered=False,
    )
    scheme_tiered = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        use_tiered=True,
    )
    dq_std = scheme_std.dequantize(scheme_std.quantize_as_u8(weight), torch.float32)
    dq_tiered = scheme_tiered.dequantize(
        scheme_tiered.quantize_as_u8(weight), torch.float32
    )
    assert torch.allclose(dq_std, dq_tiered, atol=1e-5), (
        "Tiered and standard dequantize must agree"
    )


def test_tiered_scheme_rejects_too_small_codebook():
    """use_tiered=True must fail when num_centroids < HOT_THRESHOLD."""
    with pytest.raises(ValueError):
        VQScheme(
            centroids=torch.randn(8, 4, dtype=torch.float16),
            proxy_selector=ProxySelector.L2,
            use_tiered=True,
        )


def test_tiered_scheme_buffer_smaller_than_fixed_when_skewed():
    """Tiered buffer must be smaller than fixed for highly skewed data."""
    nc, vl = 256, 4
    # All-hot weight: every vector maps to centroids 0-15.
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([32, 64])
    scheme_std = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        use_tiered=False,
    )
    scheme_tiered = VQScheme(
        centroids=centroids.clone(),
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
        use_tiered=True,
    )
    # Force weight to cluster around centroid-0 (most-hot possible).
    c0 = centroids[0].float().unsqueeze(0)
    weight = c0.repeat(32, 16).reshape(shape[0], shape[1])
    packed_std = scheme_std.quantize_as_u8(weight)
    packed_tiered = scheme_tiered.quantize_as_u8(weight)
    # With all-hot data, tiered must be strictly smaller (5 bpi vs 8 bpi).
    assert packed_tiered.numel() < packed_std.numel(), (
        f"Tiered ({packed_tiered.numel()} B) should be smaller than "
        f"fixed ({packed_std.numel()} B) for all-hot data"
    )


def test_tiered_repr():
    scheme, _ = _make_tiered_scheme()
    r = repr(scheme)
    assert "tiered=True" in r


# ---------------------------------------------------------------------------
# Theoretical savings match effective_bits formula
# ---------------------------------------------------------------------------


def test_tiered_byte_count_matches_formula():
    """Actual packed size must match the theoretical formula."""
    cold_bits = 8
    n = 200
    torch.manual_seed(42)
    indices = torch.randint(0, 256, (n,), dtype=torch.int64)
    hot_thr = TieredIndexPacking.hot_threshold(TieredIndexPacking.DEFAULT_HOT_BITS)
    n_hot = int((indices < hot_thr).sum().item())
    n_cold = n - n_hot

    packed = TieredIndexPacking.pack(indices, cold_bits)

    expected_bytes = (
        TieredIndexPacking.HEADER_BYTES
        + math.ceil(n / 8)  # prefix stream
        + math.ceil(n_hot * TieredIndexPacking.DEFAULT_HOT_BITS / 8)  # hot stream
        + math.ceil(n_cold * cold_bits / 8)  # cold stream
    )
    assert packed.numel() == expected_bytes
