"""Tests for per-group codebook sizing."""

import pytest

from neuropress.estimators import VPTQGroupEstimator

# ---------------------------------------------------------------------------
# Measured index entropy per layer type varies 7.85-13.69 bits, so a uniform NC
# spends 14 bits addressing ~230 centroids on down_proj (22% of the model) while
# in_proj_qkv genuinely needs all 16384.  Sizing NC per group cuts 14.8% of the
# bit-rate without starving anything.
# ---------------------------------------------------------------------------


def _est(**kw):
    return VPTQGroupEstimator(grouping="by_leaf_name", **kw)


def test_falls_back_to_uniform_when_no_override():
    est = _est(num_centroids=16384)
    assert est.centroids_for_group("down_proj") == 16384


def test_override_applies_to_named_group():
    est = _est(num_centroids=16384, num_centroids_per_group={"down_proj": 256})
    assert est.centroids_for_group("down_proj") == 256


def test_unlisted_groups_keep_the_default():
    est = _est(num_centroids=16384, num_centroids_per_group={"down_proj": 256})
    assert est.centroids_for_group("in_proj_qkv") == 16384


def test_rejects_non_positive_override():
    est = _est(num_centroids=16384, num_centroids_per_group={"down_proj": 0})
    with pytest.raises(ValueError, match="down_proj"):
        est.centroids_for_group("down_proj")
