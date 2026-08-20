"""Tests for NTieredIndexPacking (cascaded binary splitting).

Covers:
- Pack/unpack round-trips for D=2, 3, 4
- Header parsing
- effective_bits formula
- compute_row_offsets
- Consistency: D=2 n-tier must match TieredIndexPacking output
- Zipf-distributed data
- Edge cases (all-tier-0, all-cold, single index)
"""

import math

import pytest
import torch

from neuropress.transforms.vq_packing import (
    NTieredIndexPacking,
    TieredIndexPacking,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zipf_indices(
    n: int, num_centroids: int, s: float = 1.0, seed: int = 42
) -> torch.Tensor:
    """Generate Zipf-distributed centroid indices."""
    torch.manual_seed(seed)
    ranks = torch.arange(1, num_centroids + 1, dtype=torch.float64)
    probs = 1.0 / ranks.pow(s)
    probs /= probs.sum()
    return torch.multinomial(probs.float(), n, replacement=True).to(torch.int64)


# ---------------------------------------------------------------------------
# Round-trip: D=2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier_bits", [[4, 8], [4, 10], [4, 12], [8, 17]])
def test_roundtrip_2tier(tier_bits: list[int]):
    """2-tier n-tier pack/unpack must round-trip."""
    nc = 1 << tier_bits[-1]
    indices = _zipf_indices(256, nc, seed=sum(tier_bits))
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    restored = NTieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


# ---------------------------------------------------------------------------
# Round-trip: D=3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier_bits", [[4, 8, 12], [4, 8, 17], [4, 10, 16]])
def test_roundtrip_3tier(tier_bits: list[int]):
    """3-tier n-tier pack/unpack must round-trip."""
    nc = 1 << tier_bits[-1]
    indices = _zipf_indices(512, nc, seed=sum(tier_bits))
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    restored = NTieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


# ---------------------------------------------------------------------------
# Round-trip: D=4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier_bits", [[4, 8, 12, 17], [4, 8, 12, 16], [4, 8, 14, 20]])
def test_roundtrip_4tier(tier_bits: list[int]):
    """4-tier n-tier pack/unpack must round-trip."""
    nc = 1 << tier_bits[-1]
    indices = _zipf_indices(1024, nc, seed=sum(tier_bits))
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    restored = NTieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_all_tier0():
    """All indices in tier-0 must round-trip."""
    tier_bits = [4, 8, 12, 17]
    indices = torch.randint(0, 16, (128,), dtype=torch.int64)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    restored = NTieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


def test_all_cold():
    """All indices in coldest tier must round-trip."""
    tier_bits = [4, 8, 17]
    # Centroids 4096..131071 (above all non-cold thresholds).
    indices = torch.randint(256, 1 << 17, (128,), dtype=torch.int64)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    restored = NTieredIndexPacking.unpack(packed)
    assert (restored == indices).all()


def test_single_index():
    """Single index must round-trip."""
    for idx in [0, 15, 16, 255, 256, 4095, 4096]:
        tier_bits = [4, 8, 12, 17]
        indices = torch.tensor([idx], dtype=torch.int64)
        packed = NTieredIndexPacking.pack(indices, tier_bits)
        restored = NTieredIndexPacking.unpack(packed)
        assert (restored == indices).all(), f"Failed for idx={idx}"


def test_odd_counts():
    """Arbitrary index counts must round-trip."""
    tier_bits = [4, 8, 12]
    for n in [1, 3, 7, 17, 33, 99, 128]:
        indices = _zipf_indices(n, 1 << 12, seed=n)
        packed = NTieredIndexPacking.pack(indices, tier_bits)
        restored = NTieredIndexPacking.unpack(packed)
        assert (restored == indices).all(), f"Failed for n={n}"


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_header_fields():
    """Header must encode magic, n_indices, n_tiers, counts, bits."""
    tier_bits = [4, 8, 17]
    indices = _zipf_indices(200, 1 << 17)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    n, d, counts, bits, h = NTieredIndexPacking._parse_header(packed)
    assert n == 200
    assert d == 3
    assert bits == tier_bits
    assert sum(counts) == 200
    assert h == 12 + 8 * 3


def test_is_ntier_buffer():
    """is_ntier_buffer must detect n-tier format."""
    tier_bits = [4, 8]
    indices = torch.zeros(16, dtype=torch.int64)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    assert NTieredIndexPacking.is_ntier_buffer(packed)
    # Legacy tiered buffer should NOT be detected as n-tier.
    legacy = TieredIndexPacking.pack(indices, cold_bits=8)
    assert not NTieredIndexPacking.is_ntier_buffer(legacy)


def test_rejects_bad_magic():
    """Unpack must reject buffers with wrong magic."""
    buf = torch.zeros(100, dtype=torch.uint8)
    with pytest.raises(ValueError, match="bad magic"):
        NTieredIndexPacking.unpack(buf)


def test_rejects_short_buffer():
    """Unpack must reject buffers too short for header."""
    with pytest.raises(ValueError, match="too short"):
        NTieredIndexPacking._parse_header(torch.zeros(8, dtype=torch.uint8))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_non_increasing_bits():
    """tier_bits must be strictly increasing."""
    with pytest.raises(ValueError, match="strictly increasing"):
        NTieredIndexPacking.pack(torch.zeros(4, dtype=torch.int64), [8, 4])


def test_rejects_too_few_tiers():
    """Must have at least 2 tiers."""
    with pytest.raises(ValueError, match="tier_bits length"):
        NTieredIndexPacking.pack(torch.zeros(4, dtype=torch.int64), [8])


def test_rejects_too_many_tiers():
    """Must have at most 4 tiers."""
    with pytest.raises(ValueError, match="tier_bits length"):
        NTieredIndexPacking.pack(torch.zeros(4, dtype=torch.int64), [4, 5, 6, 7, 8])


# ---------------------------------------------------------------------------
# effective_bits
# ---------------------------------------------------------------------------


def test_effective_bits_2tier_matches_tiered():
    """2-tier effective_bits must match TieredIndexPacking formula."""
    n_hot, n_cold = 70, 30
    cold_bits = 8
    hot_bits = 4
    bpi_legacy = TieredIndexPacking.effective_bits(
        n_hot, n_hot + n_cold, cold_bits, hot_bits
    )
    bpi_ntier = NTieredIndexPacking.effective_bits(
        [n_hot, n_cold], [hot_bits, cold_bits]
    )
    assert abs(bpi_legacy - bpi_ntier) < 1e-9


def test_effective_bits_4tier():
    """4-tier effective_bits must be less than 2-tier."""
    # Zipf-ish distribution.
    counts = [273, 221, 223, 283]
    bits_4 = [4, 8, 12, 17]
    bpi_4 = NTieredIndexPacking.effective_bits(counts, bits_4)
    # Equivalent 2-tier: hot=4, cold=17.
    bpi_2 = NTieredIndexPacking.effective_bits([273, 727], [4, 17])
    assert bpi_4 < bpi_2


def test_effective_bits_all_tier0():
    """All in tier-0: prefix overhead + tier-0 bits."""
    bpi = NTieredIndexPacking.effective_bits([100, 0, 0], [4, 8, 12])
    # 2 prefix levels: level 0 covers 100/100=1.0, level 1 covers 0/100=0.
    # data: 4*1.0 = 4.0. Total = 1.0 + 0.0 + 4.0 = 5.0.
    assert abs(bpi - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_row_offsets
# ---------------------------------------------------------------------------


def test_row_offsets_shape():
    """Row offsets must have correct shape and sum."""
    tier_bits = [4, 8, 17]
    n_rows, n_groups = 8, 32
    indices = _zipf_indices(n_rows * n_groups, 1 << 17)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    offsets = NTieredIndexPacking.compute_row_offsets(packed, n_rows, n_groups)
    assert len(offsets) == 3
    for off in offsets:
        assert off.shape == (n_rows + 1,)
        assert off[0].item() == 0
    # Sum of all tier final offsets must equal total indices.
    total = sum(off[n_rows].item() for off in offsets)
    assert total == n_rows * n_groups


def test_row_offsets_monotonic():
    """Row offsets must be non-decreasing."""
    tier_bits = [4, 8, 12, 17]
    n_rows, n_groups = 16, 64
    indices = _zipf_indices(n_rows * n_groups, 1 << 17)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    offsets = NTieredIndexPacking.compute_row_offsets(packed, n_rows, n_groups)
    for k, off in enumerate(offsets):
        for i in range(n_rows):
            assert off[i + 1] >= off[i], f"tier {k}: offset[{i + 1}] < offset[{i}]"


def test_row_offsets_per_row_sum():
    """Each row's tier counts must sum to n_groups."""
    tier_bits = [4, 8, 17]
    n_rows, n_groups = 8, 32
    indices = _zipf_indices(n_rows * n_groups, 1 << 17)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    offsets = NTieredIndexPacking.compute_row_offsets(packed, n_rows, n_groups)
    for row in range(n_rows):
        row_total = sum(int(off[row + 1].item() - off[row].item()) for off in offsets)
        assert row_total == n_groups, (
            f"Row {row}: tier counts sum to {row_total}, expected {n_groups}"
        )


# ---------------------------------------------------------------------------
# Packed size sanity
# ---------------------------------------------------------------------------


def test_packed_size_formula():
    """Packed buffer size must match theoretical formula."""
    tier_bits = [4, 8, 12, 17]
    indices = _zipf_indices(500, 1 << 17)
    packed = NTieredIndexPacking.pack(indices, tier_bits)
    _, d, counts, bits, h = NTieredIndexPacking._parse_header(packed)

    # Header.
    expected = h
    # Prefix streams.
    n_remaining = 500
    for k in range(d - 1):
        expected += (n_remaining + 7) // 8
        n_remaining -= counts[k]
    # Tier data streams.
    for k in range(d):
        n_k = counts[k]
        if bits[k] == 4 and n_k % 2 != 0:
            n_k += 1  # padding
        expected += math.ceil(n_k * bits[k] / 8)
    assert packed.numel() == expected
