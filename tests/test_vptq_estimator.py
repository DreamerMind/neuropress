"""Tests for VPTQGroupEstimator - end-to-end plan production on tiny models."""

import torch
import torch.nn as nn

# Importing neuropress.t2n.qtensors first breaks a circular import that arises
# when neuropress.transforms submodules are imported standalone.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.calibration.probe.registry import StatisticsRegistry
from neuropress.estimators import VPTQGroupEstimator
from neuropress.transforms.vq import QTensorTractVQ

# ---------------------------------------------------------------------------
# Tiny model fixtures
# ---------------------------------------------------------------------------


class _TwoLayer(nn.Module):
    def __init__(self, in_dim: int = 32, hidden: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


# ---------------------------------------------------------------------------
# Basic plan shape
# ---------------------------------------------------------------------------


def test_plan_has_two_entries():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    registry = StatisticsRegistry()
    plan = est.estimate(graph=None, registry=registry, recipe=None)
    assert len(plan.weight_assignments) == 2
    assert "fc1" in plan.weight_assignments
    assert "fc2" in plan.weight_assignments


def test_plan_entries_are_qtensor_tract_vq():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    for name, qt in plan.weight_assignments.items():
        assert isinstance(qt, QTensorTractVQ), (
            f"Expected QTensorTractVQ for '{name}', got {type(qt).__name__}"
        )


# ---------------------------------------------------------------------------
# Shared-proxy property: both layers reference the same centroids object
# ---------------------------------------------------------------------------


def test_shared_centroids_same_object_grouping_all():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        grouping="all",
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    qt1 = plan.weight_assignments["fc1"]
    qt2 = plan.weight_assignments["fc2"]
    assert qt1.qscheme.centroids is qt2.qscheme.centroids, (
        "Both layers should reference the SAME centroids tensor object."
    )


# ---------------------------------------------------------------------------
# by_type grouping: with only Linear layers, still one shared codebook
# ---------------------------------------------------------------------------


def test_by_type_grouping_same_centroids_for_same_class():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        grouping="by_type",
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    qt1 = plan.weight_assignments["fc1"]
    qt2 = plan.weight_assignments["fc2"]
    # Both are Linear → same group → same centroids tensor.
    assert qt1.qscheme.centroids is qt2.qscheme.centroids


# ---------------------------------------------------------------------------
# by_leaf_name grouping: distinct trailing attribute names get dedicated
# codebooks, e.g. so a "down_proj" with a different weight-magnitude
# distribution than "gate_proj"/"up_proj" isn't starved of centroids by a
# pooled fit (see estimators.py VPTQGroupEstimator.grouping docstring).
# ---------------------------------------------------------------------------


def test_by_leaf_name_grouping_separates_different_names():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        grouping="by_leaf_name",
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    qt1 = plan.weight_assignments["fc1"]
    qt2 = plan.weight_assignments["fc2"]
    # "fc1" != "fc2" → separate groups → distinct centroids tensors.
    assert qt1.qscheme.centroids is not qt2.qscheme.centroids


def test_by_leaf_name_grouping_shares_centroids_for_same_leaf_name():
    class _Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer0 = nn.Module()
            self.layer0.down_proj = nn.Linear(32, 16, bias=False)
            self.layer1 = nn.Module()
            self.layer1.down_proj = nn.Linear(32, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer0.down_proj(x) + self.layer1.down_proj(x)

    model = _Container()
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        grouping="by_leaf_name",
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    qt1 = plan.weight_assignments["layer0.down_proj"]
    qt2 = plan.weight_assignments["layer1.down_proj"]
    # Both leaves are named "down_proj" → same group → same centroids tensor.
    assert qt1.qscheme.centroids is qt2.qscheme.centroids


# ---------------------------------------------------------------------------
# per_layer grouping: every individual module gets its own dedicated
# codebook, even when two leaves share a name (unlike by_leaf_name) — the
# original VPTQ paper's per-weight-matrix design.
# ---------------------------------------------------------------------------


def test_per_layer_grouping_separates_even_same_leaf_name():
    class _Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer0 = nn.Module()
            self.layer0.down_proj = nn.Linear(32, 16, bias=False)
            self.layer1 = nn.Module()
            self.layer1.down_proj = nn.Linear(32, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer0.down_proj(x) + self.layer1.down_proj(x)

    model = _Container()
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        grouping="per_layer",
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    qt1 = plan.weight_assignments["layer0.down_proj"]
    qt2 = plan.weight_assignments["layer1.down_proj"]
    # Same leaf name but per_layer grouping never shares → distinct codebooks.
    assert qt1.qscheme.centroids is not qt2.qscheme.centroids


def test_invalid_grouping_raises():
    import pytest

    with pytest.raises(ValueError, match="grouping must be"):
        VPTQGroupEstimator(
            num_centroids=8,
            vector_len=4,
            grouping="bogus",
        )


# ---------------------------------------------------------------------------
# Codebook size is honoured
# ---------------------------------------------------------------------------


def test_codebook_size_honoured():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=4,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    for qt in plan.weight_assignments.values():
        assert qt.qscheme.num_centroids <= 4


# ---------------------------------------------------------------------------
# vector_len must divide in_features - incompatible layer is skipped
# ---------------------------------------------------------------------------


def test_incompatible_layer_skipped():
    class _OddLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.good = nn.Linear(32, 16, bias=False)
            # in_features=15 not divisible by vector_len=4 → skip
            self.bad = nn.Linear(15, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bad(self.good(x[:, :32]))

    model = _OddLayer()
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    assert "good" in plan.weight_assignments
    assert "bad" not in plan.weight_assignments


# ---------------------------------------------------------------------------
# filter_fn excludes layers by name
# ---------------------------------------------------------------------------


def test_filter_fn_excludes_layer():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        filter_fn=lambda name, _: name != "fc2",
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    assert "fc1" in plan.weight_assignments
    assert "fc2" not in plan.weight_assignments


# ---------------------------------------------------------------------------
# 4-bit index packing triggered when num_centroids <= 16
# ---------------------------------------------------------------------------


def test_4bit_indices_when_small_codebook():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=16,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    for qt in plan.weight_assignments.values():
        assert qt.qscheme.is_packed_indices == 4


def test_5bit_indices_when_32_centroids():
    model = _TwoLayer(in_dim=32, hidden=16, out_dim=8)
    est = VPTQGroupEstimator(
        num_centroids=32,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    for qt in plan.weight_assignments.values():
        assert qt.qscheme.is_packed_indices == 5


def test_8bit_indices_when_256_centroids():
    # Need ≥256 vectors (VL=4): 64x64 + 32x64 = 1024+512 = 1536 vectors total.
    model = _TwoLayer(in_dim=64, hidden=64, out_dim=32)
    est = VPTQGroupEstimator(
        num_centroids=256,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    for qt in plan.weight_assignments.values():
        assert qt.qscheme.is_packed_indices == 8


# ---------------------------------------------------------------------------
# declare_statistics: returns hessian_diag requests only when enabled
# ---------------------------------------------------------------------------


def test_declare_statistics_correction_on():
    model = _TwoLayer()
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=True,
        gptq_correction_iters=1,
    )
    est.prepare(model)
    requests = est.declare_statistics()
    assert len(requests) == 2
    assert all(r.stat_type == "hessian" for r in requests)


def test_declare_statistics_hessian_diag_only():
    model = _TwoLayer()
    # Correction disabled → only the cheaper diagonal is collected.
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=True,
        gptq_correction_iters=0,
    )
    est.prepare(model)
    requests = est.declare_statistics()
    assert len(requests) == 2
    assert all(r.stat_type == "hessian_diag" for r in requests)


def test_declare_statistics_hessian_weighting_off():
    model = _TwoLayer()
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
        gptq_correction_iters=0,
    )
    est.prepare(model)
    requests = est.declare_statistics()
    assert requests == []


# ---------------------------------------------------------------------------
# Embedding quantization
# ---------------------------------------------------------------------------


class _EmbeddingModel(nn.Module):
    """Tiny model with an embedding + a linear layer."""

    def __init__(self, vocab: int = 32, dim: int = 16, out: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim, out, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(ids))


def test_embedding_included_in_plan():
    # VL=4 divides both dim=16 (embedding) and out=8 (linear, K=16 groups of 4).
    model = _EmbeddingModel(vocab=64, dim=16, out=8)
    est = VPTQGroupEstimator(
        module_classes=(nn.Linear, nn.Embedding),
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    assert "embed" in plan.weight_assignments
    assert "fc" in plan.weight_assignments


def test_embedding_no_hessian_stat_requested():
    # With hessian weighting on, embeddings must be excluded from hessian_diag
    # requests (integer token indices are not valid activation inputs).
    model = _EmbeddingModel(vocab=64, dim=16, out=8)
    est = VPTQGroupEstimator(
        module_classes=(nn.Linear, nn.Embedding),
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=True,
    )
    est.prepare(model)
    requests = est.declare_statistics()
    stat_names = {r.target_key for r in requests}
    assert "embed" not in stat_names  # embedding excluded
    assert "fc" in stat_names  # linear still requested


def test_embedding_roundtrip():
    # Embedding weight must be recoverable from the VQ indices + centroids.
    model = _EmbeddingModel(vocab=64, dim=16, out=8)
    orig_weight = model.embed.weight.data.clone()
    est = VPTQGroupEstimator(
        module_classes=(nn.Linear, nn.Embedding),
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=False,
    )
    est.prepare(model)
    plan = est.estimate(graph=None, registry=StatisticsRegistry(), recipe=None)
    qt = plan.weight_assignments["embed"]
    reconstructed = qt.dequantize().reshape(orig_weight.shape)
    # Reconstruction should be close to original (within codebook approximation error).
    assert reconstructed.shape == orig_weight.shape
    assert not torch.allclose(reconstructed, torch.zeros_like(reconstructed))
