"""Tests for IVFTorchKMeansBackend correctness and integration."""

from __future__ import annotations

import pytest
import torch

# Break circular import: transforms/__init__ → t2n → transforms.
import neuropress.t2n.qtensors  # noqa: F401
from neuropress.transforms.cluster_backends import (
    IVFTorchKMeansBackend,
    TorchKMeansBackend,
    _ivf_assign_chunk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blobs(
    n: int,
    k: int,
    d: int,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(vectors [n, d], true_centroids [k, d])`` drawn from k Gaussians."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    true_c = torch.randn(k, d, generator=g, device=device)
    # Assign each vector to a random centroid then add noise.
    assign = torch.randint(k, (n,), generator=g, device=device)
    noise = torch.randn(n, d, generator=g, device=device) * 0.3
    return true_c[assign] + noise, true_c


# ---------------------------------------------------------------------------
# _ivf_assign_chunk unit tests
# ---------------------------------------------------------------------------


def test_ivf_assign_chunk_exact_when_nprobe_equals_nlist():
    """With nprobe=nlist the IVF chunk must return the exact nearest centroid."""
    torch.manual_seed(42)
    N, D, k = 200, 4, 16
    nlist = 4
    cell_size = k // nlist  # 4

    x = torch.randn(N, D)
    fine_c = torch.randn(k, D)

    # Build a ground-truth exact assignment.
    exact_labels = torch.cdist(x, fine_c).argmin(1)

    # Build a trivial cell partition: cell i holds fine centroids [i*4 .. i*4+3].
    cell_gidx = torch.arange(k).reshape(nlist, cell_size)
    cell_c = fine_c[cell_gidx]
    coarse_c = cell_c.float().mean(1)  # one coarse centroid per cell: [nlist, D]

    # With nprobe == nlist we search every cell → must match exact.
    ivf_labels = _ivf_assign_chunk(x, coarse_c, cell_c.float(), cell_gidx, nprobe=nlist)

    assert ivf_labels.shape == (N,)
    assert ivf_labels.dtype == torch.int64
    assert (ivf_labels == exact_labels).all(), (
        "IVF with nprobe=nlist must match exact nearest-centroid assignment"
    )


def test_ivf_assign_chunk_labels_in_range():
    """All returned labels must be valid global fine-centroid indices."""
    torch.manual_seed(7)
    N, D, k, nlist = 300, 4, 32, 4
    cell_size = k // nlist
    x = torch.randn(N, D)
    fine_c = torch.randn(k, D)
    cell_gidx = torch.arange(k).reshape(nlist, cell_size)
    cell_c = fine_c[cell_gidx].float()
    coarse_c = cell_c.mean(1)

    labels = _ivf_assign_chunk(x, coarse_c, cell_c, cell_gidx, nprobe=2)

    assert labels.min() >= 0
    assert labels.max() < k


def test_ivf_assign_chunk_output_shape():
    """Output shape must equal the number of input query rows."""
    N, D, k, nlist = 128, 8, 64, 8
    cell_size = k // nlist
    x = torch.randn(N, D)
    fine_c = torch.randn(k, D)
    cell_gidx = torch.arange(k).reshape(nlist, cell_size)
    cell_c = fine_c[cell_gidx].float()
    coarse_c = cell_c.mean(1)

    labels = _ivf_assign_chunk(x, coarse_c, cell_c, cell_gidx, nprobe=2)
    assert labels.shape == (N,)


# ---------------------------------------------------------------------------
# IVFTorchKMeansBackend._build_cells
# ---------------------------------------------------------------------------


def test_build_cells_shapes():
    """Cell tensors must have the correct shapes after partition."""
    backend = IVFTorchKMeansBackend(seed=0)
    k, D, nlist = 64, 4, 8
    fine_c = torch.randn(k, D)
    coarse_c = torch.randn(nlist, D)
    cell_size = (k + nlist - 1) // nlist  # ceil

    cell_c, cell_gidx = backend._build_cells(fine_c, coarse_c, nlist)

    assert cell_c.shape == (nlist, cell_size, D)
    assert cell_gidx.shape == (nlist, cell_size)


def test_build_cells_global_indices_in_range():
    """All global indices stored in cells must be valid fine-centroid indices."""
    backend = IVFTorchKMeansBackend(seed=0)
    k, D, nlist = 48, 4, 6
    fine_c = torch.randn(k, D)
    coarse_c = torch.randn(nlist, D)
    _, cell_gidx = backend._build_cells(fine_c, coarse_c, nlist)

    assert cell_gidx.min() >= 0
    assert cell_gidx.max() < k


def test_build_cells_covers_all_centroids():
    """Every fine centroid must appear in at least one cell slot."""
    backend = IVFTorchKMeansBackend(seed=0)
    k, D, nlist = 32, 4, 4
    fine_c = torch.randn(k, D)
    coarse_c = torch.randn(nlist, D)
    _, cell_gidx = backend._build_cells(fine_c, coarse_c, nlist)

    present = cell_gidx.unique()
    assert len(present) == k


# ---------------------------------------------------------------------------
# IVFTorchKMeansBackend.fit — output contracts
# ---------------------------------------------------------------------------


def test_fit_output_shapes():
    """fit() must return (centroids [k, D], labels [N])."""
    N, D, k = 500, 4, 32
    vectors, _ = _make_blobs(N, k, D, seed=1)
    backend = IVFTorchKMeansBackend(n_iter=5, seed=1)
    centroids, labels = backend.fit(vectors, k)

    assert centroids.shape == (k, D)
    assert labels.shape == (N,)
    assert centroids.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_fit_labels_in_range():
    """All returned labels must be in ``[0, k)``."""
    N, D, k = 400, 4, 24
    vectors, _ = _make_blobs(N, k, D, seed=2)
    backend = IVFTorchKMeansBackend(n_iter=5, seed=2)
    _, labels = backend.fit(vectors, k)

    assert int(labels.min()) >= 0
    assert int(labels.max()) < k


def test_fit_high_centroid_occupancy():
    """With enough data, ≥ 90 % of centroids should have at least one vector.

    IVF uses an approximate E-step during Lloyd iterations, so a small number
    of centroids may end up unoccupied if they drift from the data distribution.
    The final E-step is exact (nprobe=nlist), but stranded centroids can still
    attract zero vectors.  90 % occupancy is a conservative lower bound.
    """
    N, D, k = 2000, 4, 32
    vectors, _ = _make_blobs(N, k, D, seed=3)
    backend = IVFTorchKMeansBackend(n_iter=20, seed=3)
    _, labels = backend.fit(vectors, k)

    occupied = labels.unique().numel()
    assert occupied >= int(k * 0.90), f"Only {occupied}/{k} centroids occupied"


# ---------------------------------------------------------------------------
# IVFTorchKMeansBackend.fit — correctness vs exact k-means
# ---------------------------------------------------------------------------


def test_fit_recall_vs_exact():
    """IVF labels must agree with exact nearest-centroid on ≥ 80 % of vectors.

    This is a soft recall test: IVF is approximate when nprobe < nlist, but
    should still assign the correct centroid for the vast majority of vectors
    on well-separated Gaussian blobs.
    """
    N, D, k = 2000, 4, 32
    vectors, _ = _make_blobs(N, k, D, seed=10)
    n_iter = 30

    # Train with IVF.
    ivf = IVFTorchKMeansBackend(n_iter=n_iter, seed=10)
    ivf_c, ivf_labels = ivf.fit(vectors, k)

    # Exact nearest-centroid assignment using the IVF-trained centroids.
    exact_labels = torch.cdist(vectors, ivf_c).argmin(1)

    recall = (ivf_labels == exact_labels).float().mean().item()
    assert recall >= 0.80, f"IVF recall vs exact assignment: {recall:.2%}"


def test_fit_nprobe_nlist_is_exact():
    """Setting nprobe=nlist forces exact search; labels must match torch.cdist."""
    N, D, k = 500, 4, 16
    vectors, _ = _make_blobs(N, k, D, seed=20)

    nlist = 4
    backend = IVFTorchKMeansBackend(n_iter=20, nlist=nlist, nprobe=nlist, seed=20)
    centroids, labels = backend.fit(vectors, k)

    exact = torch.cdist(vectors, centroids).argmin(1)
    assert (labels == exact).all(), "nprobe=nlist must produce exact final assignment"


# ---------------------------------------------------------------------------
# IVFTorchKMeansBackend.fit — convergence
# ---------------------------------------------------------------------------


def test_fit_distortion_decreases():
    """Running more iterations should not increase average distortion."""
    N, D, k = 1000, 4, 16
    vectors, _ = _make_blobs(N, k, D, seed=30)

    def distortion(vecs: torch.Tensor, cents: torch.Tensor) -> float:
        return float(torch.cdist(vecs, cents).min(1).values.pow(2).mean())

    c5, _ = IVFTorchKMeansBackend(n_iter=5, seed=30).fit(vectors, k)
    c30, _ = IVFTorchKMeansBackend(n_iter=30, seed=30).fit(vectors, k)

    assert distortion(vectors, c30) <= distortion(vectors, c5) + 1e-3


# ---------------------------------------------------------------------------
# Large-k smoke test
# ---------------------------------------------------------------------------


def test_fit_large_k_no_error():
    """k=1024 must complete without error on CPU (smoke test for large k path)."""
    N, D, k = 4096, 4, 1024
    torch.manual_seed(99)
    vectors = torch.randn(N, D)
    backend = IVFTorchKMeansBackend(n_iter=3, seed=99)
    centroids, labels = backend.fit(vectors, k)

    assert centroids.shape == (k, D)
    assert labels.shape == (N,)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < k


# ---------------------------------------------------------------------------
# Agreement with TorchKMeansBackend (same data, same seed)
# ---------------------------------------------------------------------------


def test_ivf_vs_torch_similar_distortion():
    """IVF and exact Lloyd's should reach comparable distortion on blob data."""
    N, D, k = 1500, 4, 32
    vectors, _ = _make_blobs(N, k, D, seed=50)

    def distortion(vecs: torch.Tensor, cents: torch.Tensor) -> float:
        return float(torch.cdist(vecs, cents).min(1).values.pow(2).mean())

    exact_c, _ = TorchKMeansBackend(n_iter=50, seed=50).fit(vectors, k)
    ivf_c, _ = IVFTorchKMeansBackend(n_iter=50, seed=50).fit(vectors, k)

    d_exact = distortion(vectors, exact_c)
    d_ivf = distortion(vectors, ivf_c)

    # IVF distortion should be within 20 % of exact Lloyd's on clean blobs.
    assert d_ivf <= d_exact * 1.20, (
        f"IVF distortion {d_ivf:.4f} more than 20 % above exact {d_exact:.4f}"
    )


# ---------------------------------------------------------------------------
# torch.compile does not raise
# ---------------------------------------------------------------------------


def test_compiled_kernel_runs():
    """The compiled IVF chunk function must execute without error."""
    from neuropress.transforms.cluster_backends import _ivf_assign_chunk_compiled

    N, D, k, nlist = 64, 4, 16, 4
    cell_size = k // nlist
    x = torch.randn(N, D)
    fine_c = torch.randn(k, D)
    cell_gidx = torch.arange(k).reshape(nlist, cell_size)
    cell_c = fine_c[cell_gidx].float()
    coarse_c = cell_c.mean(1)

    labels = _ivf_assign_chunk_compiled(x, coarse_c, cell_c, cell_gidx, nprobe=2)
    assert labels.shape == (N,)
    assert labels.dtype == torch.int64


# ---------------------------------------------------------------------------
# Parametrised: various (k, D) combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k,D", [(16, 4), (64, 4), (128, 8), (256, 4)])
def test_fit_parametrised(k: int, D: int) -> None:
    """fit() must succeed and return valid shapes for several (k, D) pairs."""
    N = max(k * 8, 256)
    torch.manual_seed(k + D)
    vectors = torch.randn(N, D)
    backend = IVFTorchKMeansBackend(n_iter=5, seed=k)
    centroids, labels = backend.fit(vectors, k)

    assert centroids.shape == (k, D)
    assert labels.shape == (N,)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < k
