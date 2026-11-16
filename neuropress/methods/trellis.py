"""Trellis-coded weight quantization (QTIP / EXL3 style).

A weight matrix is made incoherent with random sign flips and 128-wide block Hadamard
transforms on both sides, normalised by per-row and per-column rms scales (the global scale is
searched and folded into the column scales), then coded in 16x16 tiles with a
:class:`~neuropress.methods.trellis_code.TrellisCode`. Hessian importance enters through block
LDLQ over the 16 input columns of each tile row (:mod:`neuropress.methods.block_ldlq`), blocks
heaviest first by diagonal Hessian mass, which beat both the natural and EXL3's reversed order
on every linear of the stage 0 proxy.

:func:`trellis_quantize` returns a :class:`TrellisWeight`: the packed code plus its scales and
signs, which :mod:`neuropress.transforms.trellis` stores as a QTensor.

Layout. Tile ``(a, b)`` covers output rows ``16a..16a+15`` and input columns ``16b..16b+15``.
Its 256 values are read as a ``[16 in, 16 out]`` row-major block (EXL3's orientation) and walked
in :data:`TILE_PERM` order, the tensor-core lane order of EXL3's kernels, so the packed tiles
stay compatible with an EXL3-style inference kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from ..transforms.rotation import hadamard_matrix
from .block_ldlq import (
    BlockOrder,
    block_columns,
    block_ldlq,
    block_order,
    upper_inv_chol,
)
from .trellis_code import Codebook, TrellisCode

HADAMARD_BLOCK = 128
TILE = 16


def _tile_perm() -> torch.Tensor:
    perm = [0] * 256
    for t in range(32):
        r0, c0 = (t % 4) * 2, t // 4
        for j in range(8):
            r = r0 + (j & 1) + 8 * ((j >> 1) & 1)
            c = c0 + 8 * (j >> 2)
            perm[t * 8 + j] = r * 16 + c
    return torch.tensor(perm)


# Walk position -> element of the [16 in, 16 out] row-major tile
TILE_PERM = _tile_perm()
TILE_PERM_INV = torch.argsort(TILE_PERM)
# Walk position of each element of the [16 out, 16 in] row-major tile
_ROW_MAJOR_FROM_WALK = TILE_PERM_INV.view(TILE, TILE).T.reshape(-1).contiguous()


@dataclass(frozen=True)
class TrellisConfig:
    """Settings for one linear.

    Attributes:
        bits: Trellis bits per weight, 1..8 in steps of 1/16 (``3.5`` is EXL3's half step).
        codebook: Codebook decoding a 16-bit state.
        damp: Hessian damping, relative to its mean diagonal.
        order: LDLQ order of the 16-column blocks.
        seed: Seed of the random sign flips.
        scale_grid: Candidate global scales, relative to the codebook's scale.
        scale_tiles: Tiles sampled for the global scale search.
        lut: Table of the ``"lut"`` codebook.
    """

    bits: float = 3
    codebook: Codebook = "mul1"
    damp: float = 0.025
    order: BlockOrder = "act"
    seed: int = 0
    scale_grid: tuple[float, ...] = tuple(0.3 + 0.05 * i for i in range(23))
    scale_tiles: int = 512
    lut: torch.Tensor | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _ = self.code  # validates the rate, the codebook and the table
        if self.damp < 0:
            raise ValueError("damp must be non-negative")

    @property
    def code(self) -> TrellisCode:
        """The tile code."""
        return TrellisCode(self.bits, self.codebook, TILE * TILE, self.lut)


@dataclass
class TrellisWeight:
    """A trellis-coded linear weight.

    ``W = B diag(row_scale) T diag(col_scale) A^T`` with ``T`` the decoded tiles,
    ``A = diag(sign_in) H`` and ``B = diag(sign_out) H`` (``H`` block Hadamard).

    Attributes:
        code: Tile code.
        packed: ``[out / 16, in / 16, tile_words]`` int16 packed tiles.
        row_scale: ``[out]`` fp16 row scales.
        col_scale: ``[in]`` fp16 column scales (global scale folded in).
        sign_out: ``[out]`` int8 signs.
        sign_in: ``[in]`` int8 signs.
    """

    code: TrellisCode
    packed: torch.Tensor
    row_scale: torch.Tensor
    col_scale: torch.Tensor
    sign_out: torch.Tensor
    sign_in: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        """``(out, in)``."""
        return self.row_scale.numel(), self.col_scale.numel()

    @property
    def bpw(self) -> float:
        """Stored bits per weight: tiles, fp16 scales and one bit per sign."""
        n, k = self.shape
        return self.code.bits + (16 + 1) * (n + k) / (n * k)

    def to(self, device: torch.device | str) -> TrellisWeight:
        """Copy to *device*."""
        return TrellisWeight(
            self.code,
            *(
                t.to(device)
                for t in (
                    self.packed,
                    self.row_scale,
                    self.col_scale,
                    self.sign_out,
                    self.sign_in,
                )
            ),
        )

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Dense ``[out, in]`` weight, on the device of :attr:`packed`."""
        n, k = self.shape
        dev = self.packed.device
        states = self.code.unpack(self.packed)
        vals = self.code.decode(states.reshape(-1, TILE * TILE))
        # walk position -> tile element [c, r] (16 in x 16 out) -> weight [16a + r, 16b + c]
        tiles = vals.view(n // TILE, k // TILE, TILE * TILE)[
            :, :, _ROW_MAJOR_FROM_WALK.to(dev)
        ]
        wr = tiles.view(n // TILE, k // TILE, TILE, TILE).transpose(1, 2).reshape(n, k)
        wr.mul_(self.row_scale.float()[:, None]).mul_(self.col_scale.float()[None, :])
        # undo the input-side rotation (columns), then the output-side one (rows)
        x = _unrotate(wr, self.sign_in.float())
        return _unrotate(x.T, self.sign_out.float()).T.to(dtype)


@lru_cache(maxsize=4)
def _hadamard(device: torch.device) -> torch.Tensor:
    return (
        (hadamard_matrix(HADAMARD_BLOCK) / math.sqrt(HADAMARD_BLOCK)).float().to(device)
    )


def _rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """``x @ diag(signs) @ blockdiag(H)`` over the columns of ``x``."""
    r, c = x.shape
    h = _hadamard(x.device)
    return ((x * signs).reshape(r, c // HADAMARD_BLOCK, HADAMARD_BLOCK) @ h).reshape(
        r, c
    )


def _unrotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_rotate`."""
    r, c = x.shape
    h = _hadamard(x.device)
    return (x.reshape(r, c // HADAMARD_BLOCK, HADAMARD_BLOCK) @ h.T).reshape(
        r, c
    ) * signs


def _to_walks(z: torch.Tensor) -> torch.Tensor:
    """``[n, 16]`` column block -> ``[n / 16, 256]`` tiles in walk order."""
    n = z.shape[0]
    t = z.reshape(n // TILE, TILE, TILE).transpose(1, 2).reshape(n // TILE, TILE * TILE)
    return t[:, TILE_PERM.to(z.device)]


def _from_walks(v: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_to_walks`."""
    m = v.shape[0]
    t = v[:, TILE_PERM_INV.to(v.device)].reshape(m, TILE, TILE).transpose(1, 2)
    return t.reshape(m * TILE, TILE)


def _global_scale(
    w: torch.Tensor, code: TrellisCode, cfg: TrellisConfig, gen: torch.Generator
) -> float:
    """Multiplier of the codebook scale minimising the tile MSE of ``w`` (rms 1) on a sample."""
    rows, cols = w.shape[0] // TILE, w.shape[1] // TILE
    r0 = torch.randint(0, rows, (cfg.scale_tiles,), generator=gen).tolist()
    c0 = torch.randint(0, cols, (cfg.scale_tiles,), generator=gen).tolist()
    t = (
        torch.stack(
            [
                w[TILE * a : TILE * a + TILE, TILE * b : TILE * b + TILE].T.reshape(-1)
                for a, b in zip(r0, c0, strict=True)
            ]
        )[:, TILE_PERM.to(w.device)]
        * code.scale
    )
    best, best_mse = 1.0, math.inf
    for s in cfg.scale_grid:
        q, _ = code.encode((t * s).contiguous())
        mse = ((q / s - t) ** 2).mean().item()
        if mse < best_mse:
            best, best_mse = s, mse
    return best


def trellis_quantize(
    w: torch.Tensor, h: torch.Tensor, cfg: TrellisConfig
) -> TrellisWeight:
    """Quantize a linear weight against its input Hessian.

    Args:
        w: ``[out, in]`` weight on a CUDA device, both dimensions multiples of 128.
        h: ``[in, in]`` input Hessian (any positive scale), same device.
        cfg: Trellis settings.

    Returns:
        The coded weight; :meth:`TrellisWeight.dequantize` gives the dense reconstruction.
    """
    n_out, k_in = w.shape
    if n_out % HADAMARD_BLOCK or k_in % HADAMARD_BLOCK:
        raise ValueError(
            f"trellis needs both weight dimensions divisible by {HADAMARD_BLOCK}, "
            f"got {tuple(w.shape)}"
        )
    if not w.is_cuda:
        raise ValueError("trellis quantization runs on CUDA")
    code = cfg.code
    dev = w.device
    w = w.float()
    h = h.to(dev, torch.float32)
    gen = torch.Generator().manual_seed(cfg.seed)
    s_in = (torch.randint(0, 2, (k_in,), generator=gen) * 2 - 1).to(dev, torch.float32)
    s_out = (torch.randint(0, 2, (n_out,), generator=gen) * 2 - 1).to(
        dev, torch.float32
    )
    # W' = B^T W A and H' = A^T H A, with A = diag(s_in) H_k and B = diag(s_out) H_n
    wr = _rotate(_rotate(w, s_in).T.contiguous(), s_out).T.contiguous()
    hr = _rotate(_rotate(h, s_in).T.contiguous(), s_in).T.contiguous()
    d_out = wr.square().mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    d_in = (wr / d_out).square().mean(0, keepdim=True).sqrt().clamp_min(1e-12)
    rms = (wr / d_out / d_in).square().mean().sqrt()
    g = _global_scale(wr / d_out / d_in / rms, code, cfg, gen) * code.scale
    # Stored fp16 scales are the ones the codes are fitted against
    d_out = d_out.half().float()
    d_in = (d_in * rms / g).half().float()
    wn = wr / d_out / d_in
    hs = hr * d_in.T * d_in  # metric of the normalised weight
    order = block_order(hs, cfg.order, TILE)
    cols = block_columns(order, TILE)
    states = torch.empty(
        (n_out // TILE, k_in // TILE, TILE * TILE), dtype=torch.int16, device=dev
    )

    order_host = order.tolist()  # one sync, not one per block

    def quantize_block(z: torch.Tensor, j: int) -> torch.Tensor:
        q, s = code.encode(_to_walks(z).contiguous())
        states[:, order_host[j]] = s
        return _from_walks(q)

    block_ldlq(
        wn[:, cols].contiguous(),
        upper_inv_chol(hs[cols][:, cols], cfg.damp),
        quantize_block,
        TILE,
    )
    return TrellisWeight(
        code=code,
        packed=code.pack(states),
        row_scale=d_out.reshape(-1).half(),
        col_scale=d_in.reshape(-1).half(),
        sign_out=s_out.to(torch.int8),
        sign_in=s_in.to(torch.int8),
    )
