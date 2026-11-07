"""Triton VQ GEMM: decode weight tiles from packed indices inside the matmul.

    y[m, n] = sum_k x[m, k] * W[n, k],   W[n, k] = C[idx(n * G + k // VL), k % VL]

with ``G = K / VL`` and ``idx`` read from the LSB-first bit-stream written by
:class:`~neuropress.transforms.vq_packing.IndexPacking` (4 to 16 bits).

Each program owns a ``BLOCK_M x BLOCK_N`` output tile.  At every K step it
decodes its ``BLOCK_N x BLOCK_K`` weight tile from the packed bits and the
codebook (which stays in L2), then feeds it to ``tl.dot`` on tensor cores.
A weight element is therefore decoded ``M / BLOCK_M`` times and the dense
weight is never written to memory, unlike decode-then-GEMM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s
    )
    for bm, bn, bk, w, s in [
        (64, 64, 64, 4, 3),
        (64, 128, 64, 4, 3),
        (128, 64, 64, 4, 3),
        (128, 128, 64, 8, 3),
        (128, 128, 32, 8, 4),
        (64, 128, 128, 8, 3),
        (128, 256, 64, 8, 3),
    ]
]


@triton.autotune(configs=_CONFIGS, key=["M_BUCKET", "N", "K", "BITS"])
@triton.jit
def _vq_gemm_kernel(
    x_ptr,
    packed_ptr,
    cent_ptr,
    bias_ptr,
    y_ptr,
    M,
    N,
    K,
    n_bytes,
    stride_xm,
    stride_ym,
    M_BUCKET,
    BITS: tl.constexpr,
    VL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    G = K // VL
    BG: tl.constexpr = BLOCK_K // VL

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rg = tl.arange(0, BG)
    rj = tl.arange(0, VL)
    m_mask = rm < M
    n_mask = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + rk) < K
        x = tl.load(
            x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # Decode the [BLOCK_N, BG] vector indices of this weight tile.
        g = k0 // VL + rg
        flat = rn[:, None].to(tl.int64) * G + g[None, :]
        valid = n_mask[:, None] & (g < G)[None, :]
        bit = flat * BITS
        b0 = bit >> 3
        off = (bit & 7).to(tl.uint32)
        w = tl.load(packed_ptr + b0, mask=valid & (b0 < n_bytes), other=0).to(tl.uint32)
        w |= (
            tl.load(packed_ptr + b0 + 1, mask=valid & (b0 + 1 < n_bytes), other=0).to(
                tl.uint32
            )
            << 8
        )
        w |= (
            tl.load(packed_ptr + b0 + 2, mask=valid & (b0 + 2 < n_bytes), other=0).to(
                tl.uint32
            )
            << 16
        )
        idx = (w >> off) & ((1 << BITS) - 1)
        # Gather centroid rows: [BLOCK_N, BG, VL] -> [BLOCK_N, BLOCK_K].
        c = tl.load(
            cent_ptr + idx[:, :, None].to(tl.int64) * VL + rj[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        )
        wt = tl.reshape(c, (BLOCK_N, BLOCK_K)).to(x.dtype)
        acc = tl.dot(x, tl.trans(wt), acc)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(
        y_ptr + rm[:, None] * stride_ym + rn[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def vq_gemm_triton(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    out_features: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ W.T (+ bias)`` with ``W`` decoded tile by tile from packed indices.

    Args:
        x: ``[M, K]`` CUDA tensor, bfloat16 or float16.
        packed_indices: 1-D ``uint8`` LSB-first bit-stream of ``N * K / VL``
            indices.
        centroids: ``[NC, VL]`` float16 codebook, VL a power of two.
        bits: Bits per index, in ``[4, 16]``.
        out_features: N.
        bias: Optional ``[N]`` bias.

    Returns:
        ``[M, N]`` tensor in ``x.dtype``.
    """
    if x.dim() != 2 or not x.is_cuda:
        raise ValueError("vq_gemm_triton: x must be a 2-D CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("vq_gemm_triton: x must be bfloat16 or float16")
    if not 4 <= bits <= 16:
        raise ValueError(f"vq_gemm_triton: bits must be in [4, 16], got {bits}")
    x = x.contiguous()
    centroids = centroids.contiguous()
    M, K = x.shape
    VL = centroids.shape[1]
    if K % VL:
        raise ValueError(f"vq_gemm_triton: K={K} not divisible by VL={VL}")
    y = torch.empty((M, out_features), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(out_features, meta["BLOCK_N"]),
        )

    _vq_gemm_kernel[grid](
        x,
        packed_indices,
        centroids,
        bias if bias is not None else x,
        y,
        M,
        out_features,
        K,
        packed_indices.numel(),
        x.stride(0),
        y.stride(0),
        # Autotune per power-of-two M bucket, not per exact M.
        1 << max(M - 1, 0).bit_length(),
        BITS=bits,
        VL=VL,
        HAS_BIAS=bias is not None,
    )
    return y
