"""Selective-depth residual VQ: a second codebook on the most important columns.

A layer's VQ vectors are taken in the Hessian-sorted column order the estimator
already uses (:func:`~neuropress.methods.vptq.hessian_sort_permutation`), so the
column groups that matter most come first. A residual stage covers the first
``residual_groups`` of them: each of those vectors is coded as
``base[i] + residual[j]``, every other vector as ``base[i]`` alone. Because the
deep groups are a prefix of the sorted order, a layer needs no mask, only the
count, and the GPTQ pass quantizes them first so their error feeds forward into
columns not yet quantized.

At an equal bit budget the base codebook shrinks to pay for the residual
indices; spending those bits only where the Hessian is large is a variable-rate
code, which a single fixed-rate codebook cannot match.
"""

from __future__ import annotations

import torch


def residual_groups_for(n_groups: int, fraction: float) -> int:
    """Number of leading column groups that get the residual stage."""
    return min(n_groups, round(fraction * n_groups))


def _nearest(x: torch.Tensor, c: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    """Index of the nearest row of ``c`` (L2) for every row of ``x``."""
    c2 = (c * c).sum(1)
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for s in range(0, x.shape[0], chunk):
        out[s : s + chunk] = (c2 - 2 * x[s : s + chunk] @ c.T).argmin(1)
    return out


def weighted_kmeans(
    x: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    n_iter: int = 40,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Weighted Lloyd k-means; empty clusters are reseeded on random points.

    Args:
        x: ``[N, VL]`` float vectors.
        weights: ``[N]`` non-negative vector weights.
        k: Number of centroids.
        n_iter: Lloyd iterations.
        generator: CPU generator for initialisation and reseeding.

    Returns:
        ``[k, VL]`` float32 centroids.
    """
    if x.shape[0] < k:
        raise ValueError(
            f"need at least {k} vectors for {k} centroids, got {x.shape[0]}"
        )
    x = x.float()
    w = weights.float()

    def _rand(n: int) -> torch.Tensor:
        return torch.randint(0, x.shape[0], (n,), generator=generator).to(x.device)

    c = x[torch.randperm(x.shape[0], generator=generator)[:k].to(x.device)].clone()
    for _ in range(n_iter):
        a = _nearest(x, c)
        num = torch.zeros_like(c).index_add_(0, a, x * w[:, None])
        den = torch.zeros(k, device=x.device).index_add_(0, a, w)
        live = den > 0
        c[live] = num[live] / den[live, None]
        dead = (~live).nonzero().squeeze(1)
        if dead.numel():
            c[dead] = x[_rand(dead.numel())]
    return c


def two_stage_assign(
    x: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    beam: int = 8,
    refine_rounds: int = 2,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Code each row of ``x`` as ``base[i] + residual[j]``, beam search on ``i``.

    The ``beam`` nearest base codewords are each completed with their best
    residual codeword and the best pair is kept, then ``refine_rounds`` rounds
    of coordinate descent re-pick ``i`` given ``j`` and ``j`` given ``i``. All
    inputs are in the same (already importance-scaled) metric.

    Returns:
        ``(i, j)``, two ``[N]`` int64 index tensors.
    """
    beam = min(beam, base.shape[0])
    b2 = (base * base).sum(1)
    i_out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    j_out = torch.empty_like(i_out)
    for s in range(0, x.shape[0], chunk):
        xs = x[s : s + chunk]
        cand = (b2 - 2 * xs @ base.T).topk(beam, dim=1, largest=False).indices
        r = xs[:, None, :] - base[cand]  # [n, beam, VL]
        flat = r.reshape(-1, r.shape[-1])
        j = _nearest(flat, residual).reshape(cand.shape)
        err = ((r - residual[j]) ** 2).sum(-1)
        best = err.argmin(1, keepdim=True)
        i_out[s : s + chunk] = cand.gather(1, best).squeeze(1)
        j_out[s : s + chunk] = j.gather(1, best).squeeze(1)
    for _ in range(refine_rounds):
        i_out = _nearest(x - residual[j_out], base)
        j_out = _nearest(x - base[i_out], residual)
    return i_out, j_out
