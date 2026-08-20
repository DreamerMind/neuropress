"""VPTQ assignment helpers — importance-weighted VQ index computation.

This module provides the core per-layer assignment logic used by
:class:`~neuropress.estimators.VPTQGroupEstimator`.

Distinction from standard VQ assignment
-----------------------------------------
:meth:`~neuropress.transforms.vq.VQScheme.quantize_as_u8` uses a uniform
distance metric (L2 or cosine) where every element of a weight vector
contributes equally to the assignment cost.

:func:`importance_weighted_assign` scales each element by the square root
of its Hessian-diagonal importance before computing L2 distances, so that
input features with high second-order curvature dominate the assignment.
This is consistent with the VPTQ paper's formulation where the assignment
cost for vector ``w`` and centroid ``c`` under importance ``s`` is:

    ``d(w, c) = sum_j s_j * (w_j - c_j)^2``

Equivalently, ``w' = sqrt(s) * w`` and ``c' = sqrt(s) * c`` then standard
L2 distance in the scaled space.

Because the shared-proxy codebook is trained on *pooled* raw weight vectors
(importance weights vary per layer), centroid training is always done in the
unscaled space.  Importance weighting is applied only during assignment.
"""

from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)


def hessian_sort_permutation(
    diag_h: torch.Tensor,
    vector_len: int,
) -> torch.Tensor:
    """Return a column permutation that groups high-importance features together.

    Sorts the ``K`` input features by descending Hessian-diagonal importance,
    so that the most critical features land in the first positions of each VQ
    vector.  The sort is *block-aligned*: features are sorted at the granularity
    of complete ``vector_len``-sized blocks, ensuring the permutation maps full
    VQ vectors to full VQ vectors.

    Algorithm
    ---------
    1. Compute the mean importance per block of size *vector_len*:
       ``block_imp[g] = mean(diag_h[g*VL : (g+1)*VL])``
    2. Sort blocks by descending ``block_imp``.
    3. Expand block permutation back to per-feature indices.

    Args:
        diag_h: Per-feature importance vector ``[K]``.  Typically proportional
            to ``diag(H)`` from
            :class:`~neuropress.calibration.probe.HessianDiagAccumulator`.
            Negative values are clamped to zero before averaging.
        vector_len: Number of consecutive features per VQ vector.
            Must divide ``K``.

    Returns:
        Int64 permutation tensor of shape ``[K]``.  Apply via
        ``weight[:, perm]`` before vectorizing; store in
        :attr:`~neuropress.transforms.vq.VQScheme.column_permutation` so
        that :meth:`~neuropress.transforms.vq.VQScheme.dequantize` inverts
        it automatically.

    Raises:
        ValueError: If ``K % vector_len != 0``.
    """
    K = diag_h.numel()
    if K % vector_len != 0:
        raise ValueError(
            f"diag_h has {K} elements, not divisible by vector_len={vector_len}."
        )
    n_groups = K // vector_len
    # Block-level importance: mean over each VL-sized block.
    block_imp = diag_h.float().clamp(min=0.0).reshape(n_groups, vector_len).mean(dim=1)
    # Sort blocks in descending order of importance.
    sorted_blocks = torch.argsort(block_imp, descending=True)  # [G]
    # Expand block indices to per-feature indices.
    offsets = torch.arange(vector_len, device=diag_h.device)  # [VL]
    perm = (sorted_blocks.unsqueeze(1) * vector_len + offsets).reshape(-1)
    return perm.to(torch.int64)


def importance_weighted_assign(
    weight: torch.Tensor,
    centroids: torch.Tensor,
    diag_h: torch.Tensor,
    vector_len: int,
    centroid_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assign weight vectors to centroids using importance-weighted L2 distance.

    For each weight vector ``w`` of length *vector_len* and its corresponding
    per-element importance ``s = diag_h[k:k+vector_len]``, the assignment is:

        ``idx = argmin_c sum_j s_j * (w_j - c_j)^2``

    Implemented by scaling both weights and centroids by ``sqrt(s)`` and then
    applying standard L2 distance.

    On CUDA with ``vector_len=4``, uses the fused ``kmeans_assign`` CUDA kernel
    which computes argmin directly without materialising the ``[R, NC]`` distance
    matrix — eliminating the dominant VRAM allocation and memory-bandwidth cost.

    On CPU or ``vector_len != 4``, falls back to chunked ``torch.cdist``.

    Args:
        weight: Float weight matrix of shape ``[rows, K]``.  ``K`` must be
            divisible by *vector_len*.
        centroids: Codebook of shape ``[num_centroids, vector_len]``.
        diag_h: Per-feature importance vector of shape ``[K]``.  Typically
            proportional to ``diag(H)`` from :class:`HessianDiagAccumulator`.
            Negative or zero values are clamped to a small positive epsilon
            before taking the square root.
        vector_len: Number of consecutive features per VQ vector.
        centroid_bias: Optional additive bias of shape ``[num_centroids]`` added
            to each candidate's squared distance before the argmin.  Used by the
            ECVQ log-frequency term to trade distortion for lower index entropy.
            ``None`` leaves distances unbiased.

    Returns:
        1-D ``int64`` tensor of length ``rows * (K // vector_len)`` containing
        the centroid index assigned to each weight vector, in row-major order
        ``(row=0,group=0), (row=0,group=1), …, (row=R-1,group=G-1)``.

    Raises:
        ValueError: If ``weight.ndim != 2`` or ``K % vector_len != 0`` or
            ``diag_h.shape != (K,)``.
    """
    rows, K = weight.shape
    if weight.ndim != 2:
        raise ValueError(
            f"weight must be 2-D [rows, K], got shape {list(weight.shape)}."
        )
    if K % vector_len != 0:
        raise ValueError(f"K={K} is not divisible by vector_len={vector_len}.")
    if diag_h.shape != (K,):
        raise ValueError(f"diag_h shape {list(diag_h.shape)} does not match K={K}.")

    n_groups = K // vector_len
    dev = centroids.device

    # Importance scale factors: [G, VL] float32.
    imp_f32 = (
        diag_h.float().to(dev).clamp(min=1e-8).reshape(n_groups, vector_len).sqrt()
    )

    # ------------------------------------------------------------------
    # Fast path: fused CUDA kernel (VL=4 only).
    # ------------------------------------------------------------------
    if dev.type == "cuda" and vector_len == 4:
        bias = (
            centroid_bias.float().to(dev).contiguous()
            if centroid_bias is not None
            else None
        )
        try:
            return _assign_kernel(
                weight, centroids, imp_f32, rows, n_groups, vector_len, dev, bias=bias
            )
        except RuntimeError:
            LOGGER.warning("CUDA assign kernel unavailable, falling back to cdist")

    # ------------------------------------------------------------------
    # Fallback: chunked torch.cdist (CPU or VL != 4).
    # Convert importance to half here so we can free the fp32 tensor before
    # the heavy group loop allocations.
    # ------------------------------------------------------------------
    use_half = dev.type == "cuda"
    imp = imp_f32.half() if use_half else imp_f32
    del imp_f32
    bias = centroid_bias.to(dev) if centroid_bias is not None else None
    return _assign_cdist(
        weight,
        centroids,
        imp,
        use_half,
        rows,
        n_groups,
        vector_len,
        dev,
        centroid_bias=bias,
    )


def _assign_kernel(
    weight: torch.Tensor,
    centroids: torch.Tensor,
    imp_f32: torch.Tensor,
    rows: int,
    n_groups: int,
    vector_len: int,
    dev: torch.device,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from neuropress.kernels import kmeans_assign as _cuda_single
    from neuropress.kernels import kmeans_assign_grouped as _cuda_grouped

    C_f32 = centroids.float().to(dev).contiguous()
    NC = C_f32.shape[0]
    bias_t = bias.contiguous() if bias is not None else None

    # Dispatch: grouped kernel wins when R is small (GPU under-saturated per
    # group).  For large R (e.g. lm_head R=262144) a single group already
    # saturates the GPU; running all G groups concurrently trashes L2 centroid
    # cache and regresses ~7%.  Threshold: 4x the SM count x warps/block.
    n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
    _WARPS_PER_BLOCK = 8
    use_grouped = rows <= n_sm * _WARPS_PER_BLOCK * 4

    free_bytes, _ = torch.cuda.mem_get_info(dev)
    w_float_bytes = rows * n_groups * vector_len * 4

    LOGGER.info(
        "importance_weighted_assign (kernel): R=%d NC=%d G=%d  grouped=%s  fast_path=%s  bias=%s",
        rows,
        NC,
        n_groups,
        use_grouped,
        w_float_bytes <= free_bytes * 0.3,
        bias is not None,
    )

    if use_grouped:
        indices_out = torch.empty(rows, n_groups, dtype=torch.int64, device=dev)
        if w_float_bytes <= int(free_bytes * 0.3):
            W = weight.float().to(dev).reshape(rows, n_groups, vector_len).contiguous()
            return _cuda_grouped(
                W, C_f32, imp_f32.contiguous(), vector_len, bias_t
            ).reshape(-1)
        else:
            chunk_rows = max(1, int(free_bytes * 0.3) // (n_groups * vector_len * 4))
            chunk_rows = min(chunk_rows, rows)
            for row_start in range(0, rows, chunk_rows):
                row_end = min(row_start + chunk_rows, rows)
                W_chunk = (
                    weight[row_start:row_end]
                    .float()
                    .to(dev)
                    .reshape(row_end - row_start, n_groups, vector_len)
                    .contiguous()
                )
                indices_out[row_start:row_end] = _cuda_grouped(
                    W_chunk, C_f32, imp_f32.contiguous(), vector_len, bias_t
                )
            return indices_out.reshape(-1)
    else:
        # Large-R path: sequential per-group loop.  Each group fully saturates
        # the GPU so L2 centroid cache stays warm across the tile passes.
        W_f32 = weight.float().to(dev).reshape(rows, n_groups, vector_len)
        imp_c = imp_f32.contiguous()
        out_parts = []
        for g in range(n_groups):
            xg = (W_f32[:, g, :] * imp_c[g].unsqueeze(0)).contiguous()
            cg = (C_f32 * imp_c[g].unsqueeze(0)).contiguous()
            out_parts.append(_cuda_single(xg, cg, vector_len, bias_t))
        return torch.stack(out_parts, dim=1).reshape(-1)


def _assign_cdist(
    weight: torch.Tensor,
    centroids: torch.Tensor,
    imp: torch.Tensor,
    use_half: bool,
    rows: int,
    n_groups: int,
    vector_len: int,
    dev: torch.device,
    centroid_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # imp is already half (CUDA) or float32 (CPU) — converted by caller.
    C = centroids.to(dev)
    C = C.half() if use_half else C.float()

    NC = C.shape[0]
    elem_bytes = 2 if use_half else 4

    if dev.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        budget_bytes = int(free_bytes * 0.5)
    else:
        budget_bytes = 1 * 2**30  # 1 GB for CPU

    # Pre-scale all groups at once only when the full [R, G, VL] tensor fits.
    # For large-vocab embeddings (R=262144, G=384) this would be ~1.5 GB fp32;
    # in that case we fall back to lazy per-group slicing (4 MB per group).
    w_scaled_bytes = rows * n_groups * vector_len * elem_bytes
    use_fast_path = dev.type != "cuda" or w_scaled_bytes <= free_bytes * 0.3

    if use_fast_path:
        W = weight.float().to(dev).reshape(rows, n_groups, vector_len)
        W_scaled = W * imp.unsqueeze(0)  # [R, G, VL]
        del W
        if dev.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(dev)
            budget_bytes = int(free_bytes * 0.4)
    else:
        W_scaled = None

    # cdist output is float32 regardless of input dtype — size budget in 4 bytes.
    row_chunk = max(1, min(rows, budget_bytes // (NC * 4)))
    LOGGER.info(
        "importance_weighted_assign (cdist): R=%d NC=%d  chunk=%d rows "
        "(budget %.0f MB, %.0f MB/chunk, fast_path=%s)",
        rows,
        NC,
        row_chunk,
        budget_bytes / 2**20,
        row_chunk * NC * 4 / 2**20,
        use_fast_path,
    )

    per_group_indices: list[torch.Tensor] = []
    for g in range(n_groups):
        c_g = C * imp[g].unsqueeze(0)  # [NC, VL]
        if W_scaled is not None:
            w_src = W_scaled[:, g, :]
        else:
            # Lazy: extract one column-group at a time — peak allocation is
            # [R, VL] fp32 = R x VL x 4 B rather than [R, G, VL] fp32.
            w_src = weight[:, g * vector_len : (g + 1) * vector_len].float().to(
                dev
            ) * imp[g].unsqueeze(0)

        if rows <= row_chunk:
            dists = torch.cdist(w_src.float(), c_g.float(), p=2)
            if centroid_bias is not None:
                dists = dists - centroid_bias.unsqueeze(0)
            idx_g = dists.argmin(dim=1)
        else:
            parts: list[torch.Tensor] = []
            for start in range(0, rows, row_chunk):
                w_part = w_src[start : start + row_chunk]
                d = torch.cdist(w_part.float(), c_g.float(), p=2)
                if centroid_bias is not None:
                    d = d - centroid_bias.unsqueeze(0)
                parts.append(d.argmin(dim=1))
            idx_g = torch.cat(parts)
        per_group_indices.append(idx_g)

    indices = torch.stack(per_group_indices, dim=1).reshape(-1)
    return indices.to(torch.int64)
