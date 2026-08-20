"""Clustering back-ends for Vector Quantization centroid training.

Provides a uniform :class:`ClusterBackend` ABC so that VPTQ's k-means step
can be swapped between a pure-PyTorch fallback, scikit-learn, and RAPIDS
cuML without changing the calling code.

Available back-ends
-------------------
+----------------------------------+-----------------------------+------------------+
| Class                            | Requirement                 | GPU?             |
+==================================+=============================+==================+
| :class:`TorchKMeansBackend`      | none (always available)     | optional via device |
+----------------------------------+-----------------------------+------------------+
| :class:`IVFTorchKMeansBackend`   | none; torch.compile optional| optional via device |
+----------------------------------+-----------------------------+------------------+
| :class:`SklearnKMeansBackend`    | ``scikit-learn``            | no               |
+----------------------------------+-----------------------------+------------------+
| :class:`CumlKMeansBackend`       | ``cuml`` (RAPIDS)           | yes (required)   |
+----------------------------------+-----------------------------+------------------+

Usage::

    from neuropress.transforms.cluster_backends import (
        ClusterBackend,
        TorchKMeansBackend,
        IVFTorchKMeansBackend,
        SklearnKMeansBackend,
        CumlKMeansBackend,
        best_available_backend,
    )

    backend = best_available_backend()
    centroids, labels = backend.fit(vectors, n_clusters=256)
    # centroids: [n_clusters, D] float32 tensor
    # labels:    [N] int64 tensor
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import torch

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class ClusterBackend(ABC):
    """Abstract k-means clustering back-end.

    All back-ends share the same :meth:`fit` signature so that
    :class:`~neuropress.methods.vptq.VPTQCalibration` can be written
    independently of the underlying clustering library.
    """

    @abstractmethod
    def fit(
        self,
        vectors: torch.Tensor,
        n_clusters: int,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cluster *vectors* into *n_clusters* centroids.

        Args:
            vectors: 2-D float tensor of shape ``[N, D]``.  The backend may
                cast internally; the returned tensors always use float32 /
                int64 regardless of the input dtype.
            n_clusters: Number of cluster centroids.  Must satisfy
                ``2 <= n_clusters <= N``.
            sample_weights: Optional 1-D float tensor ``[N]``.  When provided,
                the M-step computes weighted means instead of uniform means.

        Returns:
            A tuple ``(centroids, labels)`` where:

            - ``centroids`` — float32 tensor of shape ``[n_clusters, D]``.
            - ``labels``    — int64 tensor of shape ``[N]``, mapping each
              input vector to its nearest centroid index.
        """


# ---------------------------------------------------------------------------
# Pure-PyTorch back-end (always available)
# ---------------------------------------------------------------------------


class TorchKMeansBackend(ClusterBackend):
    """K-means implemented in pure PyTorch via Lloyd's algorithm.

    No external dependencies required.  Runs on any device supported by
    PyTorch (CPU, CUDA, MPS).

    Distance computation is **chunked**: only ``[chunk_rows, k]`` elements are
    materialised at a time, so memory usage is bounded regardless of ``N`` or
    ``k``.  This makes the backend viable for codebooks up to ``k=64K`` without
    requiring FAISS or cuML.

    Initialisation strategy is selected automatically based on ``k``:

    - ``k <= kmeans_pp_max``: k-means++ (``O(k)`` passes, better quality).
    - ``k >  kmeans_pp_max``: random sample (``O(1)``; k-means++ would require
      tens of thousands of sequential full-data scans at large ``k``).

    Args:
        n_iter: Maximum number of Lloyd iterations.
        tol: Convergence tolerance.  Stops early when the Frobenius norm of
            the centroid shift falls below *tol*.
        chunk_mb: Memory budget in MiB for the ``[chunk_rows, k]`` distance
            slab.  Larger values use more VRAM but reduce kernel-launch
            overhead.  ``None`` auto-selects based on free VRAM (GPU) or
            defaults to 256 MiB (CPU).
        kmeans_pp_max: Use k-means++ initialisation only when
            ``n_clusters <= kmeans_pp_max``; above this threshold random
            initialisation is used instead.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-4,
        chunk_mb: float | None = None,
        kmeans_pp_max: int = 512,
        seed: int | None = None,
    ) -> None:
        """Initialise TorchKMeansBackend."""
        self.n_iter = n_iter
        self.tol = tol
        self.chunk_mb = chunk_mb
        self.kmeans_pp_max = kmeans_pp_max
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_rows(self, n_clusters: int, elem_size: int) -> int:
        """Return the number of rows per distance-slab chunk.

        Args:
            n_clusters: Number of centroids (columns of the slab).
            elem_size: Bytes per element (e.g. 4 for float32).

        Returns:
            Row count that keeps the slab within the memory budget.
        """
        if self.chunk_mb is not None:
            budget = int(self.chunk_mb * 2**20)
        elif torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            budget = max(int(free * 0.10), 64 * 2**20)  # 10 % free VRAM, min 64 MiB
        else:
            budget = 256 * 2**20  # 256 MiB on CPU
        return max(256, budget // (n_clusters * elem_size))

    def _assign(
        self,
        x: torch.Tensor,
        centroids: torch.Tensor,
        chunk_rows: int,
    ) -> torch.Tensor:
        """Return nearest-centroid labels ``[N]`` via chunked cdist.

        Args:
            x: ``[N, D]`` float32 tensor of vectors.
            centroids: ``[k, D]`` float32 centroid tensor.
            chunk_rows: Rows per distance-slab chunk.

        Returns:
            ``int64`` label tensor of shape ``[N]``.
        """
        N = x.shape[0]
        labels = torch.empty(N, dtype=torch.int64, device=x.device)
        for start in range(0, N, chunk_rows):
            slab = x[start : start + chunk_rows]
            labels[start : start + chunk_rows] = torch.cdist(
                slab, centroids, p=2
            ).argmin(dim=1)
        return labels

    def _init_kmeans_pp(
        self,
        x: torch.Tensor,
        n_clusters: int,
        chunk_rows: int,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """K-means++ initialisation (chunked distance passes).

        Args:
            x: ``[N, D]`` float32 vectors.
            n_clusters: Number of centroids to seed.
            chunk_rows: Rows per distance-slab chunk (for memory control).
            gen: Seeded ``torch.Generator`` for reproducibility.

        Returns:
            ``[n_clusters, D]`` float32 initial centroids.
        """
        N = x.shape[0]
        idx = torch.randint(N, (1,), generator=gen, device=x.device).item()
        centroids = x[idx : idx + 1].clone()

        min_dists = torch.full((N,), float("inf"), device=x.device)
        for _ in range(n_clusters - 1):
            # Update min distance using only the latest centroid (incremental).
            new_c = centroids[-1:]
            for start in range(0, N, chunk_rows):
                slab = x[start : start + chunk_rows]
                d = torch.cdist(slab, new_c, p=2).squeeze(1)
                min_dists[start : start + chunk_rows] = torch.minimum(
                    min_dists[start : start + chunk_rows], d
                )
            probs = min_dists.pow(2)
            probs = probs / probs.sum()
            chosen = int(torch.multinomial(probs, 1, generator=gen).item())
            centroids = torch.cat([centroids, x[chosen : chosen + 1]], dim=0)

        return centroids

    # ------------------------------------------------------------------
    # ClusterBackend contract
    # ------------------------------------------------------------------

    def fit(
        self,
        vectors: torch.Tensor,
        n_clusters: int,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Lloyd's algorithm on *vectors* and return centroids + labels.

        Selects k-means++ or random initialisation based on *n_clusters* vs
        :attr:`kmeans_pp_max`.  Assignment and init passes use chunked
        ``torch.cdist`` so peak memory is bounded by :attr:`chunk_mb`.

        Args:
            vectors: ``[N, D]`` float tensor (any dtype; cast to float32
                internally).
            n_clusters: Number of clusters.
            sample_weights: Optional ``[N]`` float tensor for weighted M-step.

        Returns:
            ``(centroids [n_clusters, D] float32, labels [N] int64)``.
        """
        x = vectors.float()
        N, D = x.shape
        chunk_rows = self._chunk_rows(n_clusters, x.element_size())

        gen = torch.Generator(device=x.device)
        if self.seed is not None:
            gen.manual_seed(self.seed)

        init = "kmeans++" if n_clusters <= self.kmeans_pp_max else "random"
        LOGGER.info(
            "TorchKMeans: k=%d  N=%d  D=%d  init=%s  chunk=%d",
            n_clusters,
            N,
            D,
            init,
            chunk_rows,
        )
        t_init = time.perf_counter()
        if n_clusters <= self.kmeans_pp_max:
            centroids = self._init_kmeans_pp(x, n_clusters, chunk_rows, gen)
        else:
            # Random init: O(1) passes — k-means++ is too slow for large k.
            idx = torch.randperm(N, generator=gen, device=x.device)[:n_clusters]
            centroids = x[idx].clone()
        LOGGER.info("TorchKMeans: init done in %.1fs", time.perf_counter() - t_init)

        # Lloyd iterations.
        for i in range(self.n_iter):
            t_iter = time.perf_counter()
            labels = self._assign(x, centroids, chunk_rows)

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=x.device, dtype=torch.float32)
            if sample_weights is not None:
                w = sample_weights
                new_centroids.scatter_add_(
                    0,
                    labels.unsqueeze(1).expand(-1, D),
                    x * w.unsqueeze(1),
                )
                counts.scatter_add_(0, labels, w)
            else:
                new_centroids.scatter_add_(
                    0,
                    labels.unsqueeze(1).expand(-1, D),
                    x,
                )
                counts.scatter_add_(0, labels, torch.ones(N, device=x.device))
            mask = counts > 0
            new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
            new_centroids[~mask] = centroids[~mask]  # keep old centroid if empty

            shift = (new_centroids - centroids).norm().item()
            centroids = new_centroids
            LOGGER.info(
                "TorchKMeans: iter %d/%d  shift=%.6f  %.2fs",
                i + 1,
                self.n_iter,
                shift,
                time.perf_counter() - t_iter,
            )
            if shift < self.tol:
                LOGGER.info("TorchKMeans: converged at iter %d", i + 1)
                break

        labels = self._assign(x, centroids, chunk_rows)
        return centroids, labels


# ---------------------------------------------------------------------------
# scikit-learn back-end (CPU, multi-threaded)
# ---------------------------------------------------------------------------


class SklearnKMeansBackend(ClusterBackend):
    """K-means back-end wrapping ``sklearn.cluster.KMeans``.

    Delegates all heavy lifting to scikit-learn's optimised Cython
    implementation.  Runs on CPU only; for GPU acceleration use
    :class:`CumlKMeansBackend`.

    Args:
        n_init: Number of random restarts (passed to ``KMeans(n_init=...)``).
        max_iter: Maximum number of iterations per restart.
        seed: Random seed.
        **kwargs: Additional keyword arguments forwarded to ``KMeans``.

    Raises:
        ImportError: At construction time if ``scikit-learn`` is not
            installed.
    """

    def __init__(
        self,
        n_init: int = 10,
        max_iter: int = 300,
        seed: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialise SklearnKMeansBackend, importing sklearn eagerly."""
        try:
            from sklearn.cluster import KMeans  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SklearnKMeansBackend requires scikit-learn.  "
                "Install it with: pip install scikit-learn"
            ) from exc
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed
        self._extra_kwargs = kwargs

    def fit(
        self,
        vectors: torch.Tensor,
        n_clusters: int,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cluster *vectors* using ``sklearn.cluster.KMeans``.

        Moves tensors to CPU, converts to numpy, fits KMeans, then converts
        centroids and labels back to PyTorch tensors.

        Args:
            vectors: ``[N, D]`` float tensor (device-agnostic).
            n_clusters: Number of clusters.
            sample_weights: Optional ``[N]`` float tensor for weighted k-means.

        Returns:
            ``(centroids [n_clusters, D] float32, labels [N] int64)``.
        """
        from sklearn.cluster import KMeans

        x_np = vectors.float().cpu().numpy()
        sw_np = (
            sample_weights.float().cpu().numpy() if sample_weights is not None else None
        )
        km = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.seed,
            **self._extra_kwargs,
        )
        km.fit(x_np, sample_weight=sw_np)
        centroids = torch.from_numpy(km.cluster_centers_).float()
        labels = torch.from_numpy(km.labels_).to(torch.int64)
        return centroids, labels


# ---------------------------------------------------------------------------
# RAPIDS cuML back-end (GPU, fast)
# ---------------------------------------------------------------------------


class CumlKMeansBackend(ClusterBackend):
    """K-means back-end wrapping ``cuml.cluster.KMeans`` (RAPIDS GPU).

    Requires a CUDA-capable GPU and the ``cuml`` package from RAPIDS.
    Achieves very high throughput for large numbers of vectors or clusters.

    Args:
        n_init: Number of random restarts.
        max_iter: Maximum iterations per restart.
        seed: Random seed.
        **kwargs: Additional keyword arguments forwarded to ``cuml.KMeans``.

    Raises:
        ImportError: At construction time if ``cuml`` is not installed.
        RuntimeError: At construction time if no CUDA device is available.
    """

    def __init__(
        self,
        n_init: int = 1,
        max_iter: int = 300,
        seed: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialise CumlKMeansBackend, checking for CUDA and cuml."""
        if not torch.cuda.is_available():
            raise RuntimeError("CumlKMeansBackend requires a CUDA-capable GPU.")
        try:
            import cuml  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CumlKMeansBackend requires the cuml package (RAPIDS).  "
                "See https://rapids.ai for installation instructions."
            ) from exc
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed
        self._extra_kwargs = kwargs

    def fit(
        self,
        vectors: torch.Tensor,
        n_clusters: int,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cluster *vectors* using ``cuml.cluster.KMeans`` on the GPU.

        Keeps tensors on GPU throughout; avoids unnecessary CPU round-trips.

        Args:
            vectors: ``[N, D]`` float tensor (should be on a CUDA device for
                best performance; will be moved to CUDA if not).
            n_clusters: Number of clusters.
            sample_weights: Ignored — cuML KMeans does not support sample
                weights.  A warning is logged if provided.

        Returns:
            ``(centroids [n_clusters, D] float32, labels [N] int64)`` — both
            on the same CUDA device as *vectors*.
        """
        if sample_weights is not None:
            LOGGER.warning(
                "CumlKMeansBackend does not support sample_weights; ignoring."
            )
        import cuml
        import cupy as cp

        x = vectors.float().cuda()
        x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))

        km = cuml.cluster.KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.seed,
            **self._extra_kwargs,
        )
        km.fit(x_cp)

        centroids_cp = km.cluster_centers_
        labels_cp = km.labels_

        centroids = torch.utils.dlpack.from_dlpack(centroids_cp.toDlpack()).float()
        labels = torch.utils.dlpack.from_dlpack(labels_cp.toDlpack()).to(torch.int64)
        return centroids, labels


# ---------------------------------------------------------------------------
# Native CUDA back-end (kmeans_assign.cu — no cuML required)
# ---------------------------------------------------------------------------


class CudaKMeansBackend(ClusterBackend):
    """K-means back-end using the built-in CUDA E-step kernel.

    No cuML or RAPIDS required — uses ``neuropress/kernels/kmeans_assign.cu``
    compiled via ``torch.utils.cpp_extension.load``.  Requires CUDA.

    The M-step uses ``torch.scatter_add_`` (not a throughput bottleneck).

    Args:
        n_iter: Maximum Lloyd iterations.
        tol: Early-stop threshold on centroid-shift Frobenius norm.
        vector_len: Vector dimensionality D — must be 4 (kernel limitation).
        seed: Random seed for reproducibility.

    Raises:
        RuntimeError: At construction time if CUDA is unavailable.
        RuntimeError: At first ``fit()`` call if the kernel fails to build.
    """

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-4,
        vector_len: int = 4,
        seed: int | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaKMeansBackend requires a CUDA device.")
        if vector_len != 4:
            raise ValueError(
                f"CudaKMeansBackend: only vector_len=4 is supported by the CUDA "
                f"kernel, got {vector_len}"
            )
        # Eagerly trigger the JIT build so best_available_backend() can catch
        # compilation failures (e.g. nvcc unavailable) in its try/except and
        # fall through to the next backend instead of crashing at fit() time.
        from neuropress.kernels import _load_kmeans_extension

        _load_kmeans_extension()
        self.n_iter = n_iter
        self.tol = tol
        self.vector_len = vector_len
        self.seed = seed

    def fit(
        self,
        vectors: torch.Tensor,
        n_clusters: int,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Lloyd's algorithm using the CUDA E-step kernel.

        Args:
            vectors: ``[N, D]`` float tensor; D must equal ``self.vector_len``.
            n_clusters: Number of centroids.
            sample_weights: Optional ``[N]`` float weights for the M-step.

        Returns:
            ``(centroids [n_clusters, D] float32, labels [N] int64)`` on CUDA.
        """
        from neuropress.kernels import kmeans_assign as _cuda_assign

        x = vectors.float().cuda()
        N, D = x.shape
        if D != self.vector_len:
            raise ValueError(
                f"CudaKMeansBackend: vectors have D={D} but vector_len={self.vector_len}"
            )

        gen = torch.Generator(device="cuda")
        if self.seed is not None:
            gen.manual_seed(self.seed)

        idx = torch.randperm(N, generator=gen, device="cuda")[:n_clusters]
        centroids = x[idx].clone()

        LOGGER.info("CudaKMeans: k=%d  N=%d  D=%d", n_clusters, N, D)

        for i in range(self.n_iter):
            t = time.perf_counter()

            labels = _cuda_assign(x, centroids, self.vector_len)

            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device="cuda", dtype=torch.float32)
            if sample_weights is not None:
                w = sample_weights.float().cuda()
                new_c.scatter_add_(
                    0, labels.unsqueeze(1).expand(-1, D), x * w.unsqueeze(1)
                )
                counts.scatter_add_(0, labels, w)
            else:
                new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
                counts.scatter_add_(0, labels, torch.ones(N, device="cuda"))

            mask = counts > 0
            new_c[mask] = new_c[mask] / counts[mask].unsqueeze(1)
            new_c[~mask] = centroids[~mask]

            shift = (new_c - centroids).norm().item()
            centroids = new_c
            LOGGER.info(
                "CudaKMeans: iter %d/%d  shift=%.6f  %.2fs",
                i + 1,
                self.n_iter,
                shift,
                time.perf_counter() - t,
            )
            if shift < self.tol:
                LOGGER.info("CudaKMeans: converged at iter %d", i + 1)
                break

        labels = _cuda_assign(x, centroids, self.vector_len)
        return centroids, labels


# ---------------------------------------------------------------------------
# IVF compiled kernel
# ---------------------------------------------------------------------------


def _ivf_assign_chunk(
    x_chunk: torch.Tensor,
    coarse_c: torch.Tensor,
    cell_c: torch.Tensor,
    cell_gidx: torch.Tensor,
    nprobe: int,
) -> torch.Tensor:
    """Assign one chunk of vectors to fine centroids via IVF lookup.

    Uses an L2-decomposition (``||x-c||² = ||x||² - 2x·c + ||c||²``) for
    the local candidate search so ``torch.compile`` can fuse the three terms
    into a single kernel pass instead of materialising the full difference
    tensor.

    Args:
        x_chunk: ``[N, D]`` float32 query vectors.
        coarse_c: ``[nlist, D]`` float32 coarse centroids.
        cell_c: ``[nlist, cell_size, D]`` float32 fine centroids grouped by
            coarse cell (sorted-split partition).
        cell_gidx: ``[nlist, cell_size]`` int64 global fine-centroid indices
            corresponding to *cell_c*.
        nprobe: Number of coarse cells to search per query.

    Returns:
        ``[N]`` int64 global fine-centroid labels.
    """
    # --- coarse search: find nprobe nearest coarse cells ---
    # Use L2 decomposition to avoid a full [N, nlist, D] intermediate.
    x_sq = (x_chunk * x_chunk).sum(1, keepdim=True)  # [N, 1]
    c_sq = (coarse_c * coarse_c).sum(1)  # [nlist]
    cross = x_chunk @ coarse_c.T  # [N, nlist]
    coarse_dists = x_sq + c_sq.unsqueeze(0) - 2.0 * cross  # [N, nlist]
    top_cells = coarse_dists.topk(nprobe, largest=False, dim=1).indices  # [N, nprobe]

    # --- gather candidate fine centroids from selected cells ---
    cands = cell_c[top_cells].flatten(1, 2)  # [N, nprobe*cell_size, D]
    gidx = cell_gidx[top_cells].flatten(1, 2)  # [N, nprobe*cell_size]

    # --- local fine search among candidates ---
    cands_sq = (cands * cands).sum(-1)  # [N, M]
    cross_loc = (x_chunk.unsqueeze(1) * cands).sum(-1)  # [N, M]
    local_dists = x_sq + cands_sq - 2.0 * cross_loc  # [N, M]
    local_idx = local_dists.argmin(1)  # [N]

    return gidx.gather(1, local_idx.unsqueeze(1)).squeeze(1)  # [N]


# Compiled version — falls back silently to eager if compile unavailable.
try:
    _ivf_assign_chunk_compiled: object = torch.compile(
        _ivf_assign_chunk,
        fullgraph=False,
        dynamic=True,
    )
except Exception:
    _ivf_assign_chunk_compiled = _ivf_assign_chunk


# ---------------------------------------------------------------------------
# IVF pure-PyTorch back-end
# ---------------------------------------------------------------------------


class IVFTorchKMeansBackend(ClusterBackend):
    """IVF-accelerated k-means in pure PyTorch with ``torch.compile``.

    Partitions the *k* fine centroids into ``nlist`` coarse cells so that
    the assignment E-step only searches ``nprobe`` cells instead of all *k*
    centroids.  The inner distance kernel is compiled once via
    ``torch.compile``, fusing the L2-decomposition into a single GPU pass.

    **Complexity per Lloyd iteration** (N vectors):

    - Exact k-means: ``O(N × k)``
    - IVF (this class): ``O(N × nlist) + O(N × nprobe × cell_size)``
      where ``cell_size = ceil(k / nlist)``

    At ``k=64K, nlist=256, nprobe=8``: roughly 32× fewer distance ops than
    exact k-means.

    **Initialisation strategy**:

    - ``k <= kmeans_pp_max``: k-means++ seeding (better quality, O(k) passes).
    - ``k >  kmeans_pp_max``: random sampling (O(1); k-means++ requires tens
      of thousands of sequential full-data scans for large *k*).

    **Cell partition**: fine centroids are sorted by their nearest coarse
    centroid and split into equal-size cells (sorted-split).  Cells are
    rebuilt from updated centroids every ``coarse_retrain_interval`` Lloyd
    iterations so the coarse quantizer stays representative.

    Args:
        n_iter: Maximum Lloyd iterations.
        tol: Early-stop threshold on centroid shift Frobenius norm.
        nlist: Coarse cells.  ``None`` → ``max(1, int(sqrt(k)))``.
        nprobe: Cells searched per query.
            ``None`` → ``max(1, nlist // 32)`` (~3 % of cells).
        coarse_retrain_interval: Retrain the coarse quantizer every this many
            Lloyd iterations (keeps the cell partition fresh as centroids
            move).  1 means retrain every iteration; larger values trade
            partition quality for speed.
        chunk_rows: Rows per IVF chunk.  ``None`` → auto from free VRAM /
            256 MiB CPU budget.
        kmeans_pp_max: Use k-means++ only when ``k <= kmeans_pp_max``.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-4,
        nlist: int | None = None,
        nprobe: int | None = None,
        coarse_retrain_interval: int = 10,
        chunk_rows: int | None = None,
        kmeans_pp_max: int = 512,
        seed: int | None = None,
    ) -> None:
        """Initialise IVFTorchKMeansBackend."""
        self.n_iter = n_iter
        self.tol = tol
        self.nlist = nlist
        self.nprobe = nprobe
        self.coarse_retrain_interval = coarse_retrain_interval
        self.chunk_rows = chunk_rows
        self.kmeans_pp_max = kmeans_pp_max
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_nlist(self, k: int) -> int:
        """Return nlist: user override or ``max(1, int(sqrt(k)))``.

        Args:
            k: Number of fine centroids.

        Returns:
            Number of coarse cells.
        """
        return self.nlist if self.nlist is not None else max(1, int(k**0.5))

    def _resolve_nprobe(self, nlist: int) -> int:
        """Return nprobe: user override or ``max(1, nlist // 32)``.

        Args:
            nlist: Number of coarse cells.

        Returns:
            Number of coarse cells to probe per query.
        """
        return self.nprobe if self.nprobe is not None else max(1, nlist // 32)

    def _resolve_chunk_rows(
        self,
        nprobe: int,
        cell_size: int,
        elem_size: int,
        dim: int,
    ) -> int:
        """Return rows per IVF chunk based on candidate-slab memory budget.

        The hot tensor is ``[chunk, nprobe * cell_size, D]``.  The compiled
        kernel materialises ~4 intermediates of comparable shape (cands,
        cands_sq, cross_loc, local_dists), so the budget accounts for 5×
        the slab cost.  Budget is 50 % of free VRAM (GPU) or 256 MiB (CPU).

        Args:
            nprobe: Cells probed per query.
            cell_size: Fine centroids per cell.
            elem_size: Bytes per element (float32 = 4).
            dim: Vector dimensionality.

        Returns:
            Row count per chunk.
        """
        if self.chunk_rows is not None:
            return self.chunk_rows
        # 5x accounts for: cands [N,M,D], cands_sq [N,M], cross_loc [N,M,D]
        # intermediate, local_dists [N,M], plus torch.compile workspace.
        slab_bytes_per_row = nprobe * cell_size * dim * elem_size
        total_per_row = slab_bytes_per_row * 5
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            budget = max(int(free * 0.50), 64 * 2**20)
        else:
            budget = 256 * 2**20
        return max(64, budget // max(total_per_row, 1))

    def _build_cells(
        self,
        fine_c: torch.Tensor,
        coarse_c: torch.Tensor,
        nlist: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Partition fine centroids into *nlist* balanced cells.

        Sorts fine centroids by their nearest coarse centroid and splits
        evenly (sorted-split).  Each cell has exactly ``ceil(k / nlist)``
        slots; any remainder is padded by repeating the last valid index.

        Args:
            fine_c: ``[k, D]`` float32 fine centroids.
            coarse_c: ``[nlist, D]`` float32 coarse centroids.
            nlist: Number of cells.

        Returns:
            Tuple ``(cell_c, cell_gidx)`` where:

            - ``cell_c`` — ``[nlist, cell_size, D]`` float32 fine centroids
              grouped by cell.
            - ``cell_gidx`` — ``[nlist, cell_size]`` int64 global indices of
              the fine centroids within each cell.
        """
        k = fine_c.shape[0]
        cell_size = (k + nlist - 1) // nlist

        dists = torch.cdist(fine_c.float(), coarse_c.float())  # [k, nlist]
        sort_key = dists.argmin(1)  # [k]
        sort_idx = torch.argsort(sort_key, stable=True)  # [k]

        total = nlist * cell_size
        if total > k:
            pad = sort_idx[-1:].expand(total - k)
            sort_idx = torch.cat([sort_idx, pad])

        global_idx = sort_idx.reshape(nlist, cell_size)  # [nlist, cell_size]
        cell_c = fine_c[global_idx]  # [nlist, cell_size, D]
        return cell_c.float(), global_idx

    def _train_coarse(
        self,
        fine_c: torch.Tensor,
        nlist: int,
        seed: int | None,
    ) -> torch.Tensor:
        """Train a coarse quantizer on the current fine centroids.

        Delegates to :class:`TorchKMeansBackend` with a capped iteration
        count since nlist is small.

        Args:
            fine_c: ``[k, D]`` float32 fine centroids.
            nlist: Number of coarse centroids to train.
            seed: Random seed.

        Returns:
            ``[nlist, D]`` float32 coarse centroids.
        """
        backend = TorchKMeansBackend(n_iter=50, seed=seed)
        coarse_c, _ = backend.fit(fine_c, nlist)
        return coarse_c

    def _assign_all(
        self,
        x: torch.Tensor,
        coarse_c: torch.Tensor,
        cell_c: torch.Tensor,
        cell_gidx: torch.Tensor,
        nprobe: int,
        chunk_rows: int,
    ) -> torch.Tensor:
        """Assign all N vectors via chunked IVF lookup.

        Args:
            x: ``[N, D]`` float32 vectors.
            coarse_c: ``[nlist, D]`` coarse centroids.
            cell_c: ``[nlist, cell_size, D]`` fine centroids by cell.
            cell_gidx: ``[nlist, cell_size]`` global indices.
            nprobe: Cells to probe per query.
            chunk_rows: Vectors per chunk.

        Returns:
            ``[N]`` int64 global fine-centroid labels.
        """
        N = x.shape[0]
        labels = torch.empty(N, dtype=torch.int64, device=x.device)
        fn = _ivf_assign_chunk_compiled
        for start in range(0, N, chunk_rows):
            chunk = x[start : start + chunk_rows]
            labels[start : start + chunk_rows] = fn(
                chunk, coarse_c, cell_c, cell_gidx, nprobe
            )
        return labels

    # ------------------------------------------------------------------
    # ClusterBackend contract
    # ------------------------------------------------------------------

    def fit(
        self,
        vectors: torch.Tensor,
        n_clusters: int,
        sample_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run IVF-accelerated Lloyd's algorithm on *vectors*.

        Args:
            vectors: ``[N, D]`` float tensor.
            n_clusters: Number of fine centroids (codebook size).
            sample_weights: Optional ``[N]`` float tensor for weighted M-step.

        Returns:
            ``(centroids [n_clusters, D] float32, labels [N] int64)``.
        """
        x = vectors.float()
        N, D = x.shape

        nlist = self._resolve_nlist(n_clusters)
        nprobe = self._resolve_nprobe(nlist)
        cell_size = (n_clusters + nlist - 1) // nlist
        chunk_rows = self._resolve_chunk_rows(nprobe, cell_size, x.element_size(), D)

        gen = torch.Generator(device=x.device)
        if self.seed is not None:
            gen.manual_seed(self.seed)

        init = "kmeans++" if n_clusters <= self.kmeans_pp_max else "random"
        LOGGER.info(
            "IVFKMeans: k=%d  N=%d  D=%d  nlist=%d  nprobe=%d  "
            "cell_size=%d  chunk=%d  init=%s",
            n_clusters,
            N,
            D,
            nlist,
            nprobe,
            cell_size,
            chunk_rows,
            init,
        )

        # Initialise fine centroids.
        t0 = time.perf_counter()
        if n_clusters <= self.kmeans_pp_max:
            coarse_init = TorchKMeansBackend(n_iter=20, seed=self.seed)
            centroids, _ = coarse_init.fit(x, n_clusters)
        else:
            idx = torch.randperm(N, generator=gen, device=x.device)[:n_clusters]
            centroids = x[idx].clone()
        LOGGER.info("IVFKMeans: init done in %.1fs", time.perf_counter() - t0)

        # Train initial coarse quantizer.
        t0 = time.perf_counter()
        coarse_c = self._train_coarse(centroids, nlist, self.seed)
        cell_c, cell_gidx = self._build_cells(centroids, coarse_c, nlist)
        LOGGER.info(
            "IVFKMeans: initial coarse quantizer + cells in %.1fs",
            time.perf_counter() - t0,
        )

        for iteration in range(self.n_iter):
            t_iter = time.perf_counter()

            # E-step: IVF assignment.
            labels = self._assign_all(
                x, coarse_c, cell_c, cell_gidx, nprobe, chunk_rows
            )

            # M-step: update centroids.
            new_c = torch.zeros_like(centroids)
            counts = torch.zeros(n_clusters, device=x.device, dtype=torch.float32)
            if sample_weights is not None:
                w = sample_weights
                new_c.scatter_add_(
                    0,
                    labels.unsqueeze(1).expand(-1, D),
                    x * w.unsqueeze(1),
                )
                counts.scatter_add_(0, labels, w)
            else:
                new_c.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), x)
                counts.scatter_add_(0, labels, torch.ones(N, device=x.device))
            mask = counts > 0
            new_c[mask] = new_c[mask] / counts[mask].unsqueeze(1)
            new_c[~mask] = centroids[~mask]

            shift = (new_c - centroids).norm().item()
            centroids = new_c

            # Rebuild cell partition; retrain coarse every N iterations.
            retrain = iteration % self.coarse_retrain_interval == 0
            if retrain:
                coarse_c = self._train_coarse(centroids, nlist, self.seed)
            cell_c, cell_gidx = self._build_cells(centroids, coarse_c, nlist)

            LOGGER.info(
                "IVFKMeans: iter %d/%d  shift=%.6f  %.2fs%s",
                iteration + 1,
                self.n_iter,
                shift,
                time.perf_counter() - t_iter,
                "  [coarse retrained]" if retrain else "",
            )
            if shift < self.tol:
                LOGGER.info("IVFKMeans: converged at iter %d", iteration + 1)
                break

        # Final assignment with full nprobe=nlist for exact labels.
        labels = self._assign_all(x, coarse_c, cell_c, cell_gidx, nlist, chunk_rows)
        return centroids, labels


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def best_available_backend(
    seed: int | None = None,
    n_clusters: int = 256,
    vector_len: int = 4,
) -> ClusterBackend:
    """Return the highest-performance back-end available in the current env.

    The selected back-end depends on *n_clusters*:

    - ``n_clusters <= 4096``: :class:`TorchKMeansBackend` (chunked exact
      Lloyd's).  IVF overhead (coarse quantizer + cell rebuilds) exceeds the
      assignment speedup at small codebook sizes.
    - ``n_clusters >  4096``: :class:`IVFTorchKMeansBackend` (two-level
      approximate Lloyd's with ``torch.compile`` inner kernel).

    Priority order within each tier:

    1. :class:`CumlKMeansBackend` — GPU via RAPIDS cuML; fastest when
       available (requires ``cuml``).
    2. :class:`IVFTorchKMeansBackend` or :class:`TorchKMeansBackend` — pure
       PyTorch; selected by *n_clusters* threshold above.
    3. :class:`SklearnKMeansBackend` — CPU only; used as fallback on CPU-only
       machines when *n_clusters* is small.

    Args:
        seed: Optional random seed forwarded to the selected backend.
        n_clusters: Codebook size; used to choose between exact and IVF
            backends.  Defaults to 256.
        vector_len: Vector dimensionality of the weight vectors.  When 4,
            :class:`CudaKMeansBackend` is preferred; any other value falls
            through to :class:`IVFTorchKMeansBackend`.  Defaults to 4.

    Returns:
        A :class:`ClusterBackend` instance ready to use.
    """
    _IVF_THRESHOLD = 4096

    if torch.cuda.is_available():
        # For large codebooks IVFTorchKMeansBackend (100 approximate Lloyd
        # iterations, ~2 s/layer for NC=16384) is orders of magnitude faster
        # than CudaKMeansBackend (100 exact iterations, ~133 s/layer).
        # Reserve CudaKMeansBackend for small codebooks where its exact kernel
        # is worth the per-iteration cost.
        if n_clusters > _IVF_THRESHOLD:
            return IVFTorchKMeansBackend(seed=seed)
        # Small codebooks: built-in CUDA kernel — exact, no external deps.
        try:
            return CudaKMeansBackend(seed=seed, vector_len=vector_len)
        except (ImportError, RuntimeError, ValueError):
            pass
        # cuML fallback (if installed).
        try:
            return CumlKMeansBackend(seed=seed)
        except (ImportError, RuntimeError):
            pass
        return TorchKMeansBackend(seed=seed)

    try:
        return SklearnKMeansBackend(seed=seed)
    except ImportError:
        pass

    if n_clusters > _IVF_THRESHOLD:
        return IVFTorchKMeansBackend(seed=seed)
    return TorchKMeansBackend(seed=seed)


def backend_by_name(
    name: str | None,
    seed: int | None = None,
    n_iter: int | None = None,
) -> ClusterBackend | None:
    """Return a clustering backend selected by name, or ``None`` for auto.

    Exists so a recipe can pin a backend without changing ``num_centroids``
    (which would change the bit-rate).  The auto-selector routes
    ``n_clusters > 4096`` to :class:`IVFTorchKMeansBackend`, whose
    ``torch.compile``d kernel materialises an ``[N, nlist * cell_size]`` slab -
    roughly 45x wider than :meth:`IVFTorchKMeansBackend._resolve_chunk_rows`
    budgets for, since that budget assumes the pruned ``nprobe * cell_size``
    width.  Chunking therefore stops bounding memory and the fit OOM'd both a
    24 GiB and a 40 GiB card at NC=8192.  :class:`TorchKMeansBackend` chunks
    correctly and is the safe choice for large codebooks on constrained VRAM.

    Args:
        name: One of ``"auto"``, ``"torch"``, ``"ivf"``, ``"cuda"``,
            ``"sklearn"``.  ``"auto"`` or ``None`` defers to
            :func:`best_available_backend`.
        seed: Optional seed forwarded to the backend.
        n_iter: Optional Lloyd iteration count, where the backend accepts one.

    Returns:
        A backend instance, or ``None`` when the caller should auto-select.

    Raises:
        ValueError: If *name* is not a recognized backend.
    """
    if name is None or name == "auto":
        return None

    kwargs: dict[str, object] = {"seed": seed}
    if name == "torch":
        if n_iter is not None:
            kwargs["n_iter"] = n_iter
        return TorchKMeansBackend(**kwargs)  # type: ignore[arg-type]
    if name == "ivf":
        return IVFTorchKMeansBackend(**kwargs)  # type: ignore[arg-type]
    if name == "cuda":
        return CudaKMeansBackend(**kwargs)  # type: ignore[arg-type]
    if name == "sklearn":
        return SklearnKMeansBackend(seed=seed)

    raise ValueError(
        f"Unknown cluster backend {name!r}; expected one of "
        f"'auto', 'torch', 'ivf', 'cuda', 'sklearn'."
    )
