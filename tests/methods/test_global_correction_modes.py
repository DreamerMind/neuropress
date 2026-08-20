"""Tests for GHO and CCC correction modes in VPTQGroupEstimator."""

import pytest
import torch
import torch.nn as nn

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.estimators import VPTQGroupEstimator
from neuropress.methods.correction_mode import CorrectionMode
from neuropress.transforms.vq_packing import IndexPacking


class _TwoLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


@pytest.fixture
def two_layer_model():
    return _TwoLayer()


def test_invalid_correction_mode_raises():
    with pytest.raises(ValueError, match="correction_mode"):
        VPTQGroupEstimator(
            num_centroids=8,
            vector_len=4,
            correction_mode="bad_mode",
        )


def test_default_is_sequential():
    est = VPTQGroupEstimator(num_centroids=8, vector_len=4)
    assert est.correction_mode == CorrectionMode.SEQUENTIAL


def test_gho_layer_mode_accepted():
    est = VPTQGroupEstimator(
        num_centroids=8, vector_len=4, correction_mode=CorrectionMode.GHO_LAYER
    )
    assert est.correction_mode == CorrectionMode.GHO_LAYER


def test_centroid_centric_mode_accepted():
    est = VPTQGroupEstimator(
        num_centroids=8, vector_len=4, correction_mode=CorrectionMode.CENTROID_CENTRIC
    )
    assert est.correction_mode == CorrectionMode.CENTROID_CENTRIC


def test_str_value_coerces():
    """Config passes str values; the field must accept them via StrEnum."""
    est = VPTQGroupEstimator(
        num_centroids=8, vector_len=4, correction_mode=CorrectionMode("gho_layer")
    )
    assert est.correction_mode is CorrectionMode.GHO_LAYER


def _make_estimator_with_hessians(diag_values: dict[str, torch.Tensor]):
    """Build a VPTQGroupEstimator for GHO sort testing."""
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=True,
        correction_mode=CorrectionMode.GHO_LAYER,
        gptq_correction_iters=1,
    )
    return est


def test_gho_sort_members_descending_importance():
    diags = {
        "fc1": torch.tensor([1.0, 1.0, 1.0, 1.0]),  # sum=4
        "fc2": torch.tensor([10.0, 10.0, 10.0, 10.0]),  # sum=40
    }
    est = _make_estimator_with_hessians(diags)

    fc1 = nn.Linear(4, 4, bias=False)
    fc2 = nn.Linear(4, 4, bias=False)
    members = [("fc1", fc1), ("fc2", fc2)]

    def _mock_get_diag_h(name, K, reg):
        return diags.get(name)

    est._get_diag_h = _mock_get_diag_h
    est._device = torch.device("cpu")

    sorted_members = est._gho_sort_members(members, None)
    names = [n for n, _ in sorted_members]
    assert names == ["fc2", "fc1"], (
        f"Expected fc2 first (higher importance), got {names}"
    )


def test_gho_sort_members_no_hessian_last():
    diags = {"fc1": torch.tensor([5.0, 5.0, 5.0, 5.0])}  # fc2 has no Hessian
    est = _make_estimator_with_hessians(diags)

    fc1 = nn.Linear(4, 4, bias=False)
    fc2 = nn.Linear(4, 4, bias=False)
    members = [("fc2", fc2), ("fc1", fc1)]  # fc2 first in original order

    def _mock_get_diag_h(name, K, reg):
        return diags.get(name)

    est._get_diag_h = _mock_get_diag_h
    est._device = torch.device("cpu")

    sorted_members = est._gho_sort_members(members, None)
    names = [n for n, _ in sorted_members]
    assert names == ["fc1", "fc2"], f"Expected fc1 first (has Hessian), got {names}"


def _make_packed(indices: torch.Tensor, NC: int) -> torch.Tensor:
    bits = IndexPacking.bits_for(NC)
    return IndexPacking.pack(indices.to(torch.int64), bits)


def test_compute_centroid_importance_accumulates_hessian():
    NC, VL = 4, 4
    # fc1: K=8 (2 groups), n_rows=4
    # group0 all rows -> centroid 0, group1 all rows -> centroid 1
    fc1 = nn.Linear(8, 4, bias=False)
    indices = torch.zeros(4, 2, dtype=torch.int64)
    indices[:, 1] = 1  # group1 -> centroid 1
    packed = _make_packed(indices.flatten(), NC)

    diag_h = torch.tensor([2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])  # [K=8]
    # group0 (cols 0-3) mean = 2.0, group1 (cols 4-7) mean = 3.0

    est = VPTQGroupEstimator(num_centroids=NC, vector_len=VL)
    est._device = torch.device("cpu")

    def _mock_diag(name, K, reg):
        return diag_h if name == "fc1" else None

    est._get_diag_h = _mock_diag

    members = [("fc1", fc1)]
    packed_per_layer = {"fc1": packed}
    bits = IndexPacking.bits_for(NC)

    imp = est._compute_centroid_importance(members, packed_per_layer, NC, bits, None)
    # centroid 0: 4 rows x group0_mean=2.0 = 8.0
    # centroid 1: 4 rows x group1_mean=3.0 = 12.0
    assert abs(imp[0].item() - 8.0) < 0.1, f"centroid0 imp={imp[0]}"
    assert abs(imp[1].item() - 12.0) < 0.1, f"centroid1 imp={imp[1]}"
    assert imp[2].item() == 0.0
    assert imp[3].item() == 0.0


def test_centric_group_perm_high_importance_first():
    NC, VL = 4, 4
    # fc1: K=8 (2 groups), n_rows=4
    # group0 all rows -> centroid 0 (low importance)
    # group1 all rows -> centroid 3 (high importance)
    fc1 = nn.Linear(8, 4, bias=False)
    indices = torch.zeros(4, 2, dtype=torch.int64)
    indices[:, 0] = 0  # group0 -> centroid 0
    indices[:, 1] = 3  # group1 -> centroid 3
    packed = _make_packed(indices.flatten(), NC)

    centroid_importance = torch.tensor(
        [1.0, 0.0, 0.0, 100.0]
    )  # centroid 3 most important

    est = VPTQGroupEstimator(num_centroids=NC, vector_len=VL)
    bits = IndexPacking.bits_for(NC)

    col_perm = est._centric_group_perm("fc1", fc1, packed, bits, centroid_importance)
    assert col_perm is not None
    # group1 (cols 4,5,6,7) must come first
    assert col_perm[0].item() == 4, f"Expected col=4 first, got {col_perm[0]}"
    assert col_perm[1].item() == 5
    assert col_perm[2].item() == 6
    assert col_perm[3].item() == 7
    # group0 (cols 0,1,2,3) comes after
    assert col_perm[4].item() == 0


# ---------------------------------------------------------------------------
# Wiring tests: verify CCC and GHO branches call the right methods with the
# right data in the correction loop.
# ---------------------------------------------------------------------------


def test_ccc_wiring_calls_centric_group_perm_with_importance(monkeypatch):
    """CCC mode: _centric_group_perm is called with the importance returned by
    _compute_centroid_importance, and its result is passed to _gptq_corrected_assign
    as column_perm - verifying the full CCC data flow in the correction loop."""
    NC, VL = 4, 4
    fc1 = nn.Linear(8, 4, bias=False)
    indices = torch.zeros(4, 2, dtype=torch.int64)
    indices[:, 0] = 0
    indices[:, 1] = 3  # group1 -> centroid 3 (high importance)
    packed = _make_packed(indices.flatten(), NC)
    bits = IndexPacking.bits_for(NC)

    fake_imp = torch.tensor([1.0, 0.0, 0.0, 100.0])
    # col_perm that _centric_group_perm will return (group1 first)
    fake_perm = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])

    est = VPTQGroupEstimator(
        num_centroids=NC,
        vector_len=VL,
        correction_mode=CorrectionMode.CENTROID_CENTRIC,
        gptq_correction_iters=1,
    )
    est._device = torch.device("cpu")

    importance_calls = []
    perm_calls = []
    assign_column_perms = []

    def fake_compute_importance(members, packed_pl, nc, b, registry):
        importance_calls.append(1)
        return fake_imp

    def fake_centric_perm(name, mod, packed, b, imp):
        perm_calls.append(imp)
        return fake_perm

    def fake_gptq_assign(name, mod, centroids, registry, column_perm=None):
        assign_column_perms.append(column_perm)
        return packed  # return something so n_corrected increments

    monkeypatch.setattr(est, "_compute_centroid_importance", fake_compute_importance)
    monkeypatch.setattr(est, "_centric_group_perm", fake_centric_perm)
    monkeypatch.setattr(est, "_gptq_corrected_assign", fake_gptq_assign)

    members = [("fc1", fc1)]
    packed_per_layer = {"fc1": packed}
    centroids = torch.randn(NC, VL)
    perm_per_layer = {"fc1": torch.arange(8)}  # should NOT be used in CCC mode

    # Simulate the correction loop body for one vptq_iter.
    _centroid_imp = None
    if est.correction_mode == CorrectionMode.CENTROID_CENTRIC:
        _centroid_imp = est._compute_centroid_importance(
            members, packed_per_layer, NC, bits, None
        )
    for name, mod in members:
        if name not in packed_per_layer:
            continue
        if _centroid_imp is not None:
            col_perm = est._centric_group_perm(
                name,
                mod,
                packed_per_layer[name],
                bits,
                _centroid_imp,
            )
        else:
            col_perm = perm_per_layer.get(name)
        est._gptq_corrected_assign(name, mod, centroids, None, column_perm=col_perm)

    # _compute_centroid_importance must be called exactly once.
    assert len(importance_calls) == 1, (
        "expected _compute_centroid_importance called once"
    )

    # _centric_group_perm must receive the importance tensor from _compute_centroid_importance.
    assert len(perm_calls) == 1, "expected _centric_group_perm called once"
    assert torch.equal(perm_calls[0], fake_imp), (
        "_centric_group_perm got wrong importance"
    )

    # _gptq_corrected_assign must receive fake_perm (CCC perm), not perm_per_layer["fc1"].
    assert len(assign_column_perms) == 1, "expected _gptq_corrected_assign called once"
    assert torch.equal(assign_column_perms[0], fake_perm), (
        "column_perm passed to _gptq_corrected_assign must be the CCC perm, not perm_per_layer"
    )


def test_gho_wiring_uses_sorted_member_order(monkeypatch):
    """GHO mode: _gho_sort_members is called before the correction loop and its
    returned ordering is used for iterating layers - verifying GHO data flow."""
    est = VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        correction_mode=CorrectionMode.GHO_LAYER,
        gptq_correction_iters=1,
    )

    fc1 = nn.Linear(4, 4, bias=False)
    fc2 = nn.Linear(4, 4, bias=False)
    original_members = [("fc1", fc1), ("fc2", fc2)]
    # GHO sort should reorder to fc2 first (simulating higher importance).
    reversed_members = [("fc2", fc2), ("fc1", fc1)]

    sort_calls = []

    def fake_gho_sort(members, registry):
        sort_calls.append(list(members))
        return reversed_members

    assign_order = []

    def fake_gptq_assign(name, mod, centroids, registry, column_perm=None):
        assign_order.append(name)
        return None  # corrected=None means packed_per_layer not updated

    monkeypatch.setattr(est, "_gho_sort_members", fake_gho_sort)
    monkeypatch.setattr(est, "_gptq_corrected_assign", fake_gptq_assign)

    packed_per_layer = {
        "fc1": torch.zeros(1, dtype=torch.int64),
        "fc2": torch.zeros(1, dtype=torch.int64),
    }
    centroids = torch.randn(8, 4)
    perm_per_layer = {}

    # Simulate the pre-loop sort and the inner correction loop.
    _members_for_correction = (
        est._gho_sort_members(original_members, None)
        if est.correction_mode == CorrectionMode.GHO_LAYER
        else original_members
    )
    for name, mod in _members_for_correction:
        if name not in packed_per_layer:
            continue
        col_perm = perm_per_layer.get(name)
        est._gptq_corrected_assign(name, mod, centroids, None, column_perm=col_perm)

    # _gho_sort_members must be called once with the original member list.
    assert len(sort_calls) == 1, "expected _gho_sort_members called once"
    assert [n for n, _ in sort_calls[0]] == ["fc1", "fc2"]

    # Layers must be processed in the GHO-sorted order (fc2 before fc1).
    assert assign_order == ["fc2", "fc1"], (
        f"expected GHO order ['fc2','fc1'], got {assign_order}"
    )
