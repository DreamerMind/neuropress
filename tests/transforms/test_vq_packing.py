"""Tests for IndexPacking — round-trip pack/unpack for 4-bit and 8-bit indices."""

import pytest
import torch

# neuropress.t2n.qtensors must be imported before neuropress.transforms submodules
# to break a circular import cycle: transforms/__init__ → t2n → transforms.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.vq_packing import IndexPacking

# ---------------------------------------------------------------------------
# bits_for
# ---------------------------------------------------------------------------


def test_bits_for_small_codebook():
    assert IndexPacking.bits_for(2) == 4
    assert IndexPacking.bits_for(16) == 4


def test_bits_for_large_codebook():
    assert IndexPacking.bits_for(17) == 5
    assert IndexPacking.bits_for(256) == 8


def test_bits_for_too_large_raises():
    with pytest.raises(ValueError):
        IndexPacking.bits_for((1 << 20) + 1)


def test_bits_for_too_small_raises():
    with pytest.raises(ValueError, match="num_centroids"):
        IndexPacking.bits_for(1)


# ---------------------------------------------------------------------------
# 8-bit round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_8bit_identity():
    indices = torch.arange(256, dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits=8)
    restored = IndexPacking.unpack(packed, bits=8, n_indices=256)
    assert torch.equal(indices, restored)


def test_roundtrip_8bit_random():
    indices = torch.randint(0, 256, (512,), dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits=8)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 512
    restored = IndexPacking.unpack(packed, bits=8, n_indices=512)
    assert torch.equal(indices, restored)


# ---------------------------------------------------------------------------
# 4-bit round-trip
# ---------------------------------------------------------------------------


def test_roundtrip_4bit_simple():
    indices = torch.tensor([0, 1, 2, 3, 14, 15, 7, 8], dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits=4)
    assert packed.dtype == torch.uint8
    assert packed.numel() == 4  # 8 indices → 4 bytes
    restored = IndexPacking.unpack(packed, bits=4, n_indices=8)
    assert torch.equal(indices, restored)


def test_roundtrip_4bit_all_values():
    # Pack 0..15 twice → covers all nibble values in every byte position
    indices = torch.arange(16, dtype=torch.int64).repeat(2)
    packed = IndexPacking.pack(indices, bits=4)
    assert packed.numel() == 16
    restored = IndexPacking.unpack(packed, bits=4, n_indices=32)
    assert torch.equal(indices, restored)


def test_roundtrip_4bit_random():
    torch.manual_seed(42)
    indices = torch.randint(0, 16, (128,), dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits=4)
    assert packed.numel() == 64
    restored = IndexPacking.unpack(packed, bits=4, n_indices=128)
    assert torch.equal(indices, restored)


def test_4bit_odd_count_raises():
    indices = torch.tensor([0, 1, 2], dtype=torch.int64)
    with pytest.raises(ValueError, match="even"):
        IndexPacking.pack(indices, bits=4)


# ---------------------------------------------------------------------------
# LSB-first ordering for 4-bit
# ---------------------------------------------------------------------------


def test_4bit_lsb_first_encoding():
    # indices [lo, hi] → byte == lo | (hi << 4)
    indices = torch.tensor([3, 11], dtype=torch.int64)
    packed = IndexPacking.pack(indices, bits=4)
    expected_byte = (3 & 0xF) | ((11 & 0xF) << 4)
    assert packed[0].item() == expected_byte


# ---------------------------------------------------------------------------
# Unsupported bits
# ---------------------------------------------------------------------------


def test_pack_invalid_bits_raises():
    with pytest.raises(ValueError, match="bits"):
        IndexPacking.pack(torch.tensor([0], dtype=torch.int64), bits=2)


def test_unpack_invalid_bits_raises():
    with pytest.raises(ValueError, match="bits"):
        IndexPacking.unpack(torch.tensor([0], dtype=torch.uint8), bits=2, n_indices=1)
