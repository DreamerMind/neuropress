"""k-means training data must not depend on how much GPU memory is free.

TorchKMeansBackend chunks its distance computation, so the [N, k]
distance-matrix budget does not apply to it; capping N from free VRAM made the
training set (and the codebook) change with whatever else ran on the GPU.
"""

import pytest
import torch
from torch import nn

import neuropress.t2n.qtensors  # noqa: F401
from neuropress.estimators import VPTQGroupEstimator
from neuropress.transforms.cluster_backends import TorchKMeansBackend


class _SpyBackend(TorchKMeansBackend):
    def __init__(self):
        super().__init__(n_iter=1, chunk_mb=16)
        self.n_fit: list[int] = []

    def fit(self, vectors, n_clusters, sample_weights=None):
        self.n_fit.append(vectors.shape[0])
        return super().fit(vectors, n_clusters, sample_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cap only applies on CUDA")
def test_torch_backend_training_size_ignores_free_vram(monkeypatch):
    model = nn.Sequential(
        nn.Linear(64, 512, bias=False), nn.Linear(512, 64, bias=False)
    ).cuda()
    sizes = []
    for free in (64 * 2**30, 8 * 2**20):  # plenty vs almost nothing
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *a, free=free: (free, 128 * 2**30)
        )
        spy = _SpyBackend()
        est = VPTQGroupEstimator(
            num_centroids=64,
            vector_len=4,
            cluster_backend=spy,
            use_hessian_weighting=False,
            max_train_vectors=10_000,
            seed=0,
        )
        est.prepare(model)
        est._train_centroids(list(est._target_modules.items()))
        sizes.append(spy.n_fit[0])
    assert sizes[0] == sizes[1] == 10_000


def test_chunk_rows_ignore_free_vram(monkeypatch):
    backend = TorchKMeansBackend()
    rows = []
    for free in (64 * 2**30, 2 * 2**30):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "mem_get_info", lambda *a, free=free: (free, 128 * 2**30)
        )
        rows.append(backend._chunk_rows(4096, 4))
    assert rows[0] == rows[1]
