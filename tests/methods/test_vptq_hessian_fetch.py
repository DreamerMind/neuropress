"""VPTQGroupEstimator Hessian fetches must not copy or re-fetch the full [K, K] matrix."""

import torch

from neuropress.calibration.probe import StatisticsRegistry
from neuropress.estimators import VPTQGroupEstimator


class _Registry(StatisticsRegistry):
    def __init__(self, hessians: dict[str, torch.Tensor]) -> None:
        self.hessians = hessians
        self.calls = 0

    def get(self, request):
        self.calls += 1
        h = self.hessians.get(request.target_key)
        return (
            {"hessian": h} if h is not None and request.stat_type == "hessian" else None
        )


def _estimator() -> VPTQGroupEstimator:
    return VPTQGroupEstimator(
        num_centroids=8,
        vector_len=4,
        use_hessian_weighting=True,
        gptq_correction_iters=1,
    )


def test_full_hessian_not_copied_without_cache(monkeypatch):
    """With the default null cache nothing is persisted, so no CPU copy is made."""
    h = torch.eye(8)
    registry = _Registry({"fc": h})

    def _no_copy(self, *args, **kwargs):
        raise AssertionError("full Hessian copied to CPU for a null cache")

    monkeypatch.setattr(torch.Tensor, "cpu", _no_copy)
    out = _estimator()._get_full_hessian("fc", 8, registry)
    assert out is h


def test_diag_h_fetches_full_hessian_once_per_layer():
    h = torch.diag(torch.arange(1.0, 9.0))
    registry = _Registry({"fc": h})
    est = _estimator()
    first = est._get_diag_h("fc", 8, registry)
    second = est._get_diag_h("fc", 8, registry)
    assert torch.equal(first, torch.arange(1.0, 9.0))
    assert torch.equal(second, first)
    assert registry.calls == 1


def test_diag_h_memo_is_per_registry():
    """A new registry (next estimate call, sequential block) must not see stale diagonals."""
    est = _estimator()
    est._get_diag_h("fc", 8, _Registry({"fc": torch.eye(8)}))
    fresh = est._get_diag_h("fc", 8, _Registry({"fc": 2 * torch.eye(8)}))
    assert torch.equal(fresh, torch.full((8,), 2.0))
