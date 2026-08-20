"""Correctness tests for the CUDA k-means E-step kernel.

Strategy: for (N, k, D) inputs, compare kmeans_assign output against
brute-force cdist argmin (ground truth). Every label must match.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def brute_force_assign(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    dists = torch.cdist(x.float(), centroids.float(), p=2)
    return dists.argmin(dim=1)


def _make_inputs(N, k, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g, dtype=torch.float32, device="cuda")
    centroids = torch.randn(k, D, generator=g, dtype=torch.float32, device="cuda")
    return x, centroids


@pytest.fixture(params=[4])
def vector_len(request):
    return request.param


def test_small_exact(vector_len):
    D = vector_len
    x, centroids = _make_inputs(N=512, k=64, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    ref = brute_force_assign(x, centroids)
    assert got.shape == (512,)
    assert got.dtype == torch.int64
    assert (got == ref).all(), f"mismatch: {(got != ref).sum()} / 512 labels differ"


def test_k256(vector_len):
    D = vector_len
    x, centroids = _make_inputs(N=10_000, k=256, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    ref = brute_force_assign(x, centroids)
    assert (got == ref).all(), f"k=256 mismatch: {(got != ref).sum()} / 10000"


def test_k1024(vector_len):
    D = vector_len
    x, centroids = _make_inputs(N=20_000, k=1024, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    ref = brute_force_assign(x, centroids)
    assert (got == ref).all(), f"k=1024 mismatch: {(got != ref).sum()} / 20000"


def test_k4096(vector_len):
    D = vector_len
    x, centroids = _make_inputs(N=20_000, k=4096, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    ref = brute_force_assign(x, centroids)
    assert (got == ref).all(), f"k=4096 mismatch: {(got != ref).sum()} / 20000"


def test_non_tile_aligned_k(vector_len):
    """k=2500: not a multiple of TILE=2048, exercises partial last tile."""
    D = vector_len
    x, centroids = _make_inputs(N=1000, k=2500, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    ref = brute_force_assign(x, centroids)
    assert (got == ref).all(), f"partial-tile mismatch: {(got != ref).sum()} / 1000"


def test_non_warp_aligned_N(vector_len):
    """N=13: not a multiple of WARPS_PER_BLOCK=8, exercises boundary guard."""
    D = vector_len
    x, centroids = _make_inputs(N=13, k=64, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    ref = brute_force_assign(x, centroids)
    assert (got == ref).all()


def test_output_dtype_and_device(vector_len):
    D = vector_len
    x, centroids = _make_inputs(N=100, k=32, D=D)
    from neuropress.kernels import kmeans_assign

    got = kmeans_assign(x, centroids, vector_len=D)
    assert got.dtype == torch.int64
    assert got.device.type == "cuda"


def test_unsupported_vector_len_raises():
    """D=8 is not compiled — must raise, not silently corrupt."""
    from neuropress.kernels import kmeans_assign

    x, centroids = _make_inputs(N=64, k=16, D=8)
    with pytest.raises(Exception, match="vector_len"):
        kmeans_assign(x, centroids, vector_len=8)
