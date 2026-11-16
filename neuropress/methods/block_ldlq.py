"""Block LDLQ: Hessian-aware error feedback around any block quantizer.

LDLQ (QuIP) is GPTQ's column-by-column error feedback written with the upper Cholesky factor
of the inverse Hessian. Quantizing ``b`` columns at a time lets a vector quantizer code a whole
block jointly (a trellis tile, a VQ codeword spanning columns) while the error of every block is
still pushed onto the columns not yet quantized. The block quantizer is a callable, so the same
loop serves any method; a block order (GPTQ's act-order on blocks) is applied by the caller via
:func:`block_order`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

BlockOrder = Literal["act", "natural", "reverse"]


def upper_inv_chol(h: torch.Tensor, damp: float) -> torch.Tensor:
    """Upper Cholesky factor of ``(H + damp * mean(diag H) * I)^-1``, float32.

    Raises:
        torch.linalg.LinAlgError: If the damped Hessian is not positive definite.
    """
    h = h.double().clone()
    h.diagonal().add_(damp * h.diagonal().mean())
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    return torch.linalg.cholesky(hinv, upper=True).float()


def block_order(h: torch.Tensor, order: BlockOrder, block: int) -> torch.Tensor:
    """Order of the ``n / block`` column blocks for the LDLQ pass.

    ``"act"`` puts the blocks with the largest diagonal Hessian mass first, ``"reverse"`` runs
    from the last block to the first, ``"natural"`` keeps them in place.
    """
    n_blocks = h.shape[0] // block
    if order == "natural":
        return torch.arange(n_blocks, device=h.device)
    if order == "reverse":
        return torch.arange(n_blocks - 1, -1, -1, device=h.device)
    mass = h.diagonal().reshape(n_blocks, block).sum(1)
    return mass.argsort(descending=True)


def block_columns(order: torch.Tensor, block: int) -> torch.Tensor:
    """Column permutation that lays the blocks out in *order*."""
    return (order[:, None] * block + torch.arange(block, device=order.device)).reshape(
        -1
    )


def block_ldlq(
    w: torch.Tensor,
    u: torch.Tensor,
    quantize_block: Callable[[torch.Tensor, int], torch.Tensor],
    block: int = 16,
    lazy: int = 128,
) -> torch.Tensor:
    """Quantize the columns of ``w`` block by block with LDLQ error feedback.

    Args:
        w: ``[out, in]`` weight, columns already in quantization order.
        u: Upper Cholesky factor of the inverse Hessian, in the same order
            (:func:`upper_inv_chol`).
        quantize_block: ``(z, j) -> q``: quantizes block ``j`` (``z`` is ``[out, block]``,
            already corrected) and returns its reconstruction.
        block: Columns per block; must divide ``in``.
        lazy: Columns between full updates of the remaining weight (a multiple of *block*).

    Returns:
        ``[out, in]`` reconstruction, in the same column order as ``w``.
    """
    if w.shape[1] % block or lazy % block:
        raise ValueError("block must divide both the column count and lazy")
    w = w.clone()
    q = torch.empty_like(w)
    n = w.shape[1]
    for b0 in range(0, n, lazy):
        b1 = min(b0 + lazy, n)
        wb = w[:, b0:b1]
        err = torch.empty_like(wb)
        for j in range(0, b1 - b0, block):
            c0, c1 = b0 + j, b0 + j + block
            z = wb[:, j : j + block]
            qz = quantize_block(z, c0 // block)
            q[:, c0:c1] = qz
            e = (z - qz) @ torch.linalg.inv(u[c0:c1, c0:c1])
            err[:, j : j + block] = e
            wb[:, j + block :] -= e @ u[c0:c1, c1:b1]
        w[:, b1:] -= err @ u[b0:b1, b1:]
    return q
