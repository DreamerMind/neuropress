"""VPTQ seeding: a seed reaches the clustering backend and the global RNG."""

import pytest
import torch

from neuropress.estimator_configs import VptqGroupEstimatorConfig
from neuropress.transforms.cluster_backends import TorchKMeansBackend


def test_config_threads_seed_to_estimator_and_backend():
    est = VptqGroupEstimatorConfig(
        num_centroids=16, vector_len=4, cluster_backend_name="torch", seed=7
    ).build()
    assert est.seed == 7
    assert isinstance(est.cluster_backend, TorchKMeansBackend)
    assert est.cluster_backend.seed == 7


def test_seed_defaults_to_none():
    est = VptqGroupEstimatorConfig(num_centroids=16, vector_len=4).build()
    assert est.seed is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_deterministic_kmeans_same_seed_equal_different_seed_differs():
    torch.use_deterministic_algorithms(True)
    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(50_000, 4, generator=g).cuda()

        def fit(seed):
            return TorchKMeansBackend(n_iter=10, seed=seed).fit(x, 256)[0]

        assert torch.equal(fit(1), fit(1))
        assert not torch.equal(fit(1), fit(2))
    finally:
        torch.use_deterministic_algorithms(False)
