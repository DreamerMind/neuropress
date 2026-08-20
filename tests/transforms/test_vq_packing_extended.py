"""Tests for extended IndexPacking bit widths (Phase 3).

Covers bits_for, pack/unpack round-trips for all widths 4-16, edge cases,
and VQScheme construction with large codebook sizes.
"""

import math

import pytest
import torch

# Must be imported first to break the transforms → t2n circular import.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.vq_packing import IndexPacking

# ---------------------------------------------------------------------------
# bits_for
# ---------------------------------------------------------------------------


def test_bits_for_small_codebooks():
    assert IndexPacking.bits_for(2) == 4
    assert IndexPacking.bits_for(16) == 4
    assert IndexPacking.bits_for(17) == 5
    assert IndexPacking.bits_for(32) == 5
    assert IndexPacking.bits_for(33) == 6
    assert IndexPacking.bits_for(64) == 6
    assert IndexPacking.bits_for(65) == 7
    assert IndexPacking.bits_for(128) == 7
    assert IndexPacking.bits_for(129) == 8
    assert IndexPacking.bits_for(256) == 8


def test_bits_for_large_codebooks():
    assert IndexPacking.bits_for(257) == 9
    assert IndexPacking.bits_for(512) == 9
    assert IndexPacking.bits_for(513) == 10
    assert IndexPacking.bits_for(1024) == 10
    assert IndexPacking.bits_for(1025) == 11
    assert IndexPacking.bits_for(2048) == 11
    assert IndexPacking.bits_for(4096) == 12
    assert IndexPacking.bits_for(8192) == 13
    assert IndexPacking.bits_for(16384) == 14
    assert IndexPacking.bits_for(32768) == 15
    assert IndexPacking.bits_for(65536) == 16


def test_bits_for_matches_ceil_log2():
    for nc in [3, 5, 10, 20, 100, 300, 1000, 5000, 50000]:
        expected = max(4, math.ceil(math.log2(nc)))
        assert IndexPacking.bits_for(nc) == expected


def test_bits_for_invalid():
    with pytest.raises(ValueError):
        IndexPacking.bits_for(1)
    with pytest.raises(ValueError):
        IndexPacking.bits_for(0)
    with pytest.raises(ValueError):
        IndexPacking.bits_for((1 << 20) + 1)


# ---------------------------------------------------------------------------
# pack / unpack round-trip — all supported bit widths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", range(4, 17))
def test_pack_unpack_roundtrip(bits: int):
    """pack → unpack must recover original indices for every bit width."""
    torch.manual_seed(bits)
    max_val = (1 << bits) - 1
    n = 128
    indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits)
    restored = IndexPacking.unpack(packed, bits, n_indices=n)
    assert (restored == indices).all(), f"bits={bits}: round-trip failed"


@pytest.mark.parametrize("bits", range(4, 17))
def test_pack_byte_count(bits: int):
    """Packed buffer must be exactly ceil(n * bits / 8) bytes."""
    n = 100
    indices = torch.zeros(n, dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits)
    expected_bytes = math.ceil(n * bits / 8)
    assert packed.numel() == expected_bytes, (
        f"bits={bits}: expected {expected_bytes} bytes, got {packed.numel()}"
    )


@pytest.mark.parametrize("bits", range(4, 17))
def test_pack_max_index(bits: int):
    """Maximum index value 2**bits - 1 must survive the round-trip."""
    max_val = (1 << bits) - 1
    indices = torch.tensor([max_val, 0, max_val, 0], dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits)
    restored = IndexPacking.unpack(packed, bits, n_indices=4)
    assert (restored == indices).all(), f"bits={bits}: max-index round-trip failed"


@pytest.mark.parametrize("bits", range(4, 17))
def test_unpack_trims_to_n_indices(bits: int):
    """unpack must return exactly n_indices elements, not more."""
    # Use 16 (even) so the 4-bit path (which requires even counts) passes too.
    n = 16
    max_val = (1 << bits) - 1
    indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits)
    restored = IndexPacking.unpack(packed, bits, n_indices=n)
    assert restored.numel() == n


@pytest.mark.parametrize("bits", [5, 6, 7, 9, 10, 12, 14])
def test_pack_unpack_odd_count(bits: int):
    """Non-power-of-2 widths must handle any index count, not just aligned ones."""
    for n in [1, 3, 7, 13, 33, 99]:
        max_val = (1 << bits) - 1
        indices = torch.randint(0, max_val + 1, (n,), dtype=torch.int64)
        packed = IndexPacking.pack(indices, bits)
        restored = IndexPacking.unpack(packed, bits, n_indices=n)
        assert (restored == indices).all(), f"bits={bits}, n={n}: round-trip failed"


def test_pack_uint8_output():
    """Packed buffer must always be uint8."""
    for bits in range(4, 21):
        indices = torch.zeros(8, dtype=torch.int64)
        packed = IndexPacking.pack(indices, bits)
        assert packed.dtype == torch.uint8, f"bits={bits}: dtype not uint8"


def test_pack_invalid_bits():
    indices = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(ValueError):
        IndexPacking.pack(indices, 3)
    with pytest.raises(ValueError):
        IndexPacking.pack(indices, 21)


def test_unpack_invalid_bits():
    packed = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(ValueError):
        IndexPacking.unpack(packed, 3, n_indices=4)
    with pytest.raises(ValueError):
        IndexPacking.unpack(packed, 21, n_indices=4)


# ---------------------------------------------------------------------------
# VQScheme construction with large codebooks
# ---------------------------------------------------------------------------


def test_vqscheme_accepts_large_codebook():
    """VQScheme must accept num_centroids > 256 (up to 65536)."""
    from neuropress.transforms.vq import ProxySelector, VQScheme

    for nc in [257, 512, 1024, 4096]:
        centroids = torch.randn(nc, 4, dtype=torch.float16)
        scheme = VQScheme(
            centroids=centroids,
            proxy_selector=ProxySelector.L2,
        )
        assert scheme.num_centroids == nc
        assert scheme.is_packed_indices == IndexPacking.bits_for(nc)


def test_vqscheme_rejects_too_large():
    """VQScheme must raise for num_centroids > 1M."""
    from neuropress.transforms.vq import ProxySelector, VQScheme

    with pytest.raises(ValueError):
        VQScheme(
            centroids=torch.randn((1 << 20) + 1, 4, dtype=torch.float16),
            proxy_selector=ProxySelector.L2,
        )


# ---------------------------------------------------------------------------
# Quantize / dequantize with large codebooks
# ---------------------------------------------------------------------------


def test_quantize_dequantize_large_codebook():
    """End-to-end quantize→dequantize with NC=1024 (10-bit indices)."""
    from neuropress.transforms.vq import ProxySelector, VQScheme

    nc, vl = 1024, 4
    out_f, in_f = 8, 16  # small weight for speed
    centroids = torch.randn(nc, vl, dtype=torch.float16)
    shape = torch.Size([out_f, in_f])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
    )
    weight = torch.randn(out_f, in_f)
    packed = scheme.quantize_as_u8(weight)
    assert packed.dtype == torch.uint8
    dequant = scheme.dequantize(packed, torch.float32)
    assert dequant.shape == shape
    assert dequant.isfinite().all()
    # Each dequantized vector must be one of the centroids.
    vecs = dequant.reshape(-1, vl)
    for vec in vecs:
        diffs = (centroids.float() - vec.unsqueeze(0)).norm(dim=1)
        assert diffs.min().item() < 1e-4
