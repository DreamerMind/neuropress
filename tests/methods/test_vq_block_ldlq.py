"""Exact block LDLQ for VQ: VL-column blocks with full-Hessian feedback and metric."""

import torch
import torch.nn as nn

from neuropress.methods.vq_gptq import LDLQStats, VQGPTQCorrector
from neuropress.transforms.vq_packing import IndexPacking

VL = 4


def _setup(rows=24, cols=16, nc=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    mod = nn.Linear(cols, rows, bias=False)
    mod.weight.data = torch.randn(rows, cols, generator=g)
    x = torch.randn(4 * cols, cols, generator=g) @ torch.randn(cols, cols, generator=g)
    h = x.T @ x / x.shape[0]  # correlated, so within-block off-diagonals matter
    centroids = torch.randn(nc, VL, generator=g)
    return mod, h, centroids


def _damped(h, damp=0.01):
    h = h.clone()
    h.diagonal().add_(damp * h.diagonal().mean())
    return h


def _dequant(packed, centroids, rows, cols):
    bits = IndexPacking.bits_for(centroids.shape[0])
    idx = IndexPacking.unpack(packed, bits, rows * cols // VL)
    return centroids[idx].reshape(rows, cols)


def _diagonal_reference(w, centroids, h, corr):
    """The pre-existing path: per-column metric sqrt(diag H), feedback E / U_jj."""
    u = corr.cholesky_inv_upper(h)
    w = w.clone()
    imp = h.diagonal().sqrt()
    out = torch.empty_like(w)
    for c0 in range(0, w.shape[1], VL):
        c1 = c0 + VL
        s = imp[c0:c1]
        idx = torch.cdist(w[:, c0:c1] * s, centroids * s).argmin(1)
        out[:, c0:c1] = centroids[idx]
        err = (w[:, c0:c1] - centroids[idx]) / u.diagonal()[c0:c1]
        w[:, c1:] -= err @ u[c0:c1, c1:]
    return out


def test_reported_loss_equals_full_hessian_proxy_loss():
    """With block feedback, sum_b ||(w'_b - q_b) U_bb^-1||^2 == tr(E H E^T) exactly."""
    mod, h, centroids = _setup()
    stats = LDLQStats.empty(centroids.shape[0], VL)
    corr = VQGPTQCorrector(vector_len=VL, block_metric=True)
    packed = corr.correct(
        "l", mod, centroids, h, torch.device("cpu"), mstep_stats=stats
    )
    q = _dequant(packed, centroids, *mod.weight.shape)
    e = (mod.weight.data - q).double()
    proxy = torch.trace(e @ _damped(h).double() @ e.T)
    torch.testing.assert_close(stats.loss, proxy.item(), rtol=1e-6, atol=0)


def test_block_metric_beats_diagonal_metric_on_proxy_loss():
    mod, h, centroids = _setup(rows=64, cols=32, nc=32, seed=1)
    hd = _damped(h).double()
    losses = {}
    for block in (False, True):
        corr = VQGPTQCorrector(vector_len=VL, block_metric=block)
        if block:
            packed = corr.correct("l", mod, centroids, h, torch.device("cpu"))
            q = _dequant(packed, centroids, *mod.weight.shape)
        else:  # the diagonal path is CUDA-only
            q = _diagonal_reference(mod.weight.data, centroids, h, corr)
        e = (mod.weight.data - q).double()
        losses[block] = torch.trace(e @ hd @ e.T).item()
    assert losses[True] < losses[False]


def test_mstep_minimizes_the_accumulated_loss():
    """The solved codebook gives the lowest loss for the recorded w' and assignments."""
    mod, h, centroids = _setup(seed=2)
    stats = LDLQStats.empty(centroids.shape[0], VL)
    corr = VQGPTQCorrector(vector_len=VL, block_metric=True)
    corr.correct("l", mod, centroids, h, torch.device("cpu"), mstep_stats=stats)
    solved = stats.solve(centroids)
    loss_at = stats.loss_for  # quadratic in the codebook, given w' and assignments
    base = loss_at(solved)
    assert base <= loss_at(centroids) + 1e-9
    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        assert (
            base
            <= loss_at(solved + 1e-3 * torch.randn(solved.shape, generator=g)) + 1e-12
        )


def test_mstep_with_identity_hessian_is_the_plain_mean():
    mod, _, centroids = _setup(seed=4)
    h = torch.eye(mod.weight.shape[1])
    stats = LDLQStats.empty(centroids.shape[0], VL)
    corr = VQGPTQCorrector(vector_len=VL, block_metric=True)
    packed = corr.correct(
        "l", mod, centroids, h, torch.device("cpu"), mstep_stats=stats
    )
    bits = IndexPacking.bits_for(centroids.shape[0])
    idx = IndexPacking.unpack(packed, bits, mod.weight.numel() // VL)
    vecs = mod.weight.data.reshape(-1, VL)
    solved = stats.solve(centroids)
    for k in range(centroids.shape[0]):
        sel = idx == k
        expected = vecs[sel].mean(0) if sel.any() else centroids[k]
        torch.testing.assert_close(solved[k], expected, rtol=1e-5, atol=1e-6)


def test_estimator_runs_block_ldlq_passes_with_a_decreasing_mstep(caplog):
    import re

    from neuropress.calibration.probe import StatisticsRegistry
    from neuropress.estimators import VPTQGroupEstimator

    class _Registry(StatisticsRegistry):
        def __init__(self, hessians):
            self.hessians = hessians

        def get(self, request):
            h = self.hessians.get(request.target_key)
            return (
                {"hessian": h}
                if h is not None and request.stat_type == "hessian"
                else None
            )

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 16, bias=False), nn.Linear(16, 8, bias=False))
    hessians = {}
    for name, k in (("0", 32), ("1", 16)):
        x = torch.randn(4 * k, k) @ torch.randn(k, k)
        hessians[name] = x.T @ x / x.shape[0]
    est = VPTQGroupEstimator(
        num_centroids=16,
        vector_len=VL,
        use_hessian_weighting=True,
        grouping="all",
        gptq_correction_iters=3,
        gptq_block_ldlq=True,
    )
    est.prepare(model)
    with caplog.at_level("INFO", logger="neuropress.estimators"):
        plan = est.estimate(graph=None, registry=_Registry(hessians), recipe=None)
    assert set(plan.weight_assignments) == {"0", "1"}
    msteps = re.findall(r"LDLQ M-step loss (\S+) -> (\S+)", caplog.text)
    passes = re.findall(r"LDLQ pass \d loss (\S+) \((\d) layers\)", caplog.text)
    assert len(msteps) == 2 and len(passes) == 3
    assert all(n == "2" for _, n in passes)
    for before, after in msteps:
        assert float(after) <= float(before)


def test_lazy_feedback_matches_eager_feedback():
    mod, h, centroids = _setup(rows=16, cols=40, seed=5)
    corr = VQGPTQCorrector(vector_len=VL, block_metric=True)
    u = corr.cholesky_inv_upper(h)
    w = mod.weight.data
    eager = corr._block_ldlq(w.clone(), u, centroids, None, lazy_cols=VL)
    for lazy in (8, 12, 128):
        torch.testing.assert_close(
            corr._block_ldlq(w.clone(), u, centroids, None, lazy), eager
        )
