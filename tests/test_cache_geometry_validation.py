"""Cached VQ artifacts must be rejected when their geometry does not match.

A VL=8 run on 2026-08-05 loaded a VL=4 artifact from cache and failed ~1000
lines later with "size of tensor a (1048576) must match tensor b (524288)" —
exactly the 2x that a wrong vector_len produces.  Zero groups had trained, so
the run measured nothing.  Geometry must be checked at the load site.
"""

import torch

from neuropress.estimators import VPTQGroupEstimator


def _est(vl=8, nc=16384):
    return VPTQGroupEstimator(vector_len=vl, num_centroids=nc, grouping="by_leaf_name")


def test_accepts_centroids_with_matching_vector_len():
    est = _est(vl=8)
    c = torch.zeros(16384, 8)
    assert est.cached_centroids_usable(c) is True


def test_rejects_centroids_with_wrong_vector_len():
    est = _est(vl=8)
    c = torch.zeros(16384, 4)  # stale VL=4 codebook
    assert est.cached_centroids_usable(c) is False


def test_rejects_non_2d_centroids():
    assert _est().cached_centroids_usable(torch.zeros(16384)) is False


def _packed_bytes(n_indices, bits):
    return (n_indices * bits + 7) // 8


def test_accepts_assignment_blob_of_expected_size():
    est = _est(vl=8)
    # 2048x2048 at VL=8 -> 524288 indices at 14 bits
    blob = torch.zeros(_packed_bytes(524288, 14), dtype=torch.uint8)
    assert est.cached_assignment_usable(blob, 524288, 14) is True


def test_rejects_stale_vl4_assignment_blob():
    """A VL=4 blob for the same layer holds 2x the indices, so 2x the bytes.

    Checking the unpacked length cannot catch this: IndexPacking.unpack takes
    n_indices as an argument and always returns exactly that many.  The blob
    size is the only signal.
    """
    est = _est(vl=8)
    stale = torch.zeros(_packed_bytes(1048576, 14), dtype=torch.uint8)
    assert est.cached_assignment_usable(stale, 524288, 14) is False


# ---------------------------------------------------------------------------
# The vector_len check alone is not enough: codebook SIZE must match too.
#
# A run configured with per-group codebook sizes can load cached codebooks from
# an earlier uniform run: every one passes the vector_len check, so nothing
# retrains and the run silently measures the uniform configuration instead.
# Distinct-looking configurations then all resolve to the same model.
# ---------------------------------------------------------------------------


def test_rejects_centroids_with_wrong_codebook_size():
    est = _est(vl=4, nc=256)
    stale = torch.zeros(16384, 4)  # codebook from a uniform NC=16384 run
    assert est.cached_centroids_usable(stale, expected_nc=256) is False


def test_accepts_centroids_with_matching_codebook_size():
    est = _est(vl=4, nc=256)
    assert est.cached_centroids_usable(torch.zeros(256, 4), expected_nc=256) is True


def test_per_group_size_is_what_is_checked_not_the_global_default():
    """down_proj at 256 must reject the global NC=16384 codebook."""
    est = VPTQGroupEstimator(
        vector_len=4,
        num_centroids=16384,
        grouping="by_leaf_name",
        num_centroids_per_group={"down_proj": 256},
    )
    nc = est.centroids_for_group("down_proj")
    assert nc == 256
    assert est.cached_centroids_usable(torch.zeros(16384, 4), expected_nc=nc) is False


def test_expected_nc_is_optional_for_backwards_compatibility():
    """Callers that do not pass expected_nc still get the vector_len check."""
    est = _est(vl=8)
    assert est.cached_centroids_usable(torch.zeros(16384, 8)) is True
    assert est.cached_centroids_usable(torch.zeros(16384, 4)) is False
