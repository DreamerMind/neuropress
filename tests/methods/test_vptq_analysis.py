"""Tests for vptq_analysis: Zipf fitting and distribution measurement."""

import torch
import torch.nn as nn

import neuropress.t2n.qtensors  # noqa: F401  (breaks circular import)
from neuropress.methods.vptq_analysis import (
    _fit_zipf,
    measure_assignment_distribution,
    pooled_stats,
    theoretical_tiered_bpw,
)
from neuropress.transforms.vq import ProxySelector, QTensorTractVQ, VQScheme

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vq_linear(
    out_features: int,
    in_features: int,
    num_centroids: int = 16,
    vector_len: int = 4,
    skew: bool = False,
) -> nn.Linear:
    """Build nn.Linear with a QTensorTractVQ weight."""
    centroids = torch.randn(num_centroids, vector_len, dtype=torch.float16)
    shape = torch.Size([out_features, in_features])
    scheme = VQScheme(
        centroids=centroids,
        proxy_selector=ProxySelector.L2,
        original_shape=shape,
    )
    weight = torch.randn(out_features, in_features)
    if skew:
        # Force most vectors close to centroid 0 for a skewed distribution.
        weight[:] = centroids[0].float().repeat(out_features, in_features // vector_len)
    packed = scheme.quantize_as_u8(weight)
    fp_approx = scheme.dequantize(packed, torch.float32)
    qt = QTensorTractVQ(fp_approx, scheme)
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.__dict__["weight"] = qt
    return linear


# ---------------------------------------------------------------------------
# _fit_zipf
# ---------------------------------------------------------------------------


def test_fit_zipf_uniform_returns_near_zero_s():
    counts = torch.ones(256)
    s, _r2 = _fit_zipf(counts)
    assert s < 0.1


def test_fit_zipf_perfect_zipf_s1():
    nc = 64
    counts = torch.tensor([1.0 / k for k in range(1, nc + 1)])
    counts = (counts / counts.sum() * 10000).long().float()
    counts, _ = counts.sort(descending=True)
    s, r2 = _fit_zipf(counts)
    assert 0.7 < s < 1.3
    assert r2 > 0.9


def test_fit_zipf_single_entry():
    counts = torch.tensor([100])
    s, r2 = _fit_zipf(counts)
    assert s == 0.0
    assert r2 == 0.0


def test_fit_zipf_returns_nonnegative_s():
    counts = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).flip(0)
    s, _ = _fit_zipf(counts)
    assert s >= 0.0


# ---------------------------------------------------------------------------
# measure_assignment_distribution
# ---------------------------------------------------------------------------


def test_measure_finds_vq_layers():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = _make_vq_linear(16, 32)

        def forward(self, x):
            return self.fc(x)

    stats = measure_assignment_distribution(_M())
    assert "fc" in stats


def test_measure_fields_correct():
    out, inp, nc, vl = 16, 32, 8, 4
    linear = _make_vq_linear(out, inp, num_centroids=nc, vector_len=vl)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = linear

        def forward(self, x):
            return self.fc(x)

    stats = measure_assignment_distribution(_M())
    st = stats["fc"]
    assert st.num_centroids == nc
    assert st.vector_len == vl
    assert st.num_vectors == out * (inp // vl)
    assert st.counts.shape == (nc,)
    assert st.counts[0] >= st.counts[-1]  # sorted descending
    assert 0.0 <= st.zipf_s
    assert 0.0 <= st.zipf_r2 <= 1.0
    assert 0.0 <= st.top1_fraction <= 1.0
    assert 0.0 <= st.top16_fraction <= 1.0


def test_measure_skewed_distribution_high_top1():
    linear = _make_vq_linear(32, 64, num_centroids=16, vector_len=4, skew=True)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = linear

        def forward(self, x):
            return self.fc(x)

    stats = measure_assignment_distribution(_M())
    st = stats["fc"]
    assert st.top1_fraction > 0.5


def test_measure_no_vq_layers_returns_empty():
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    stats = measure_assignment_distribution(model)
    assert stats == {}


# ---------------------------------------------------------------------------
# pooled_stats
# ---------------------------------------------------------------------------


def test_pooled_stats_empty():
    assert pooled_stats({}) == {}


def test_pooled_stats_keys():
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = _make_vq_linear(16, 32)

        def forward(self, x):
            return self.fc(x)

    stats = measure_assignment_distribution(_M())
    pooled = pooled_stats(stats)
    for key in ("zipf_s", "zipf_r2", "top1_fraction", "top16_fraction", "mean_bpw"):
        assert key in pooled


# ---------------------------------------------------------------------------
# theoretical_tiered_bpw
# ---------------------------------------------------------------------------


def test_theoretical_bpw_ordering():
    """2-tier must use fewer bits than fixed; entropy is the lower bound."""
    t = theoretical_tiered_bpw(1.2, 256, 4)
    assert t["entropy"] <= t["2tier"] <= t["fixed"]


def test_theoretical_bpw_flat_distribution_small_saving():
    """For s=0.5 (flat), 2-tier saving should be small."""
    t = theoretical_tiered_bpw(0.5, 256, 4)
    saving = (t["fixed"] - t["2tier"]) / t["fixed"]
    assert saving < 0.2


def test_theoretical_bpw_skewed_distribution_larger_saving():
    """For s=1.5 (skewed), 2-tier saving should be substantial."""
    t = theoretical_tiered_bpw(1.5, 256, 4)
    saving = (t["fixed"] - t["2tier"]) / t["fixed"]
    assert saving > 0.2
