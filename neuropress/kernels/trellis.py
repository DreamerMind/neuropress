"""CUDA bitshift-trellis encoder and decoder (``trellis_viterbi.cu``).

Thin wrappers over the JIT-built ``neuropress_trellis`` extension: scratch allocation, batching
and argument checks. The code itself (rates, codebooks, packing) is described by
:class:`neuropress.methods.trellis_code.TrellisCode`, which is the interface callers should use.

Importing this module succeeds without CUDA; the build happens on the first call.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# Codebook ids of the kernel
CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2, "lut": 3}

# Scratch budget for the traceback history of one launch
_HISTORY_BUDGET_BYTES = 2 << 30


@lru_cache(maxsize=1)
def _ext() -> object:
    if not torch.cuda.is_available():
        raise RuntimeError("trellis encoding needs a CUDA device")
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "trellis_viterbi.cu"
    log.info("JIT-building neuropress_trellis from %s ...", src)
    try:
        return load(
            name="neuropress_trellis",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "-std=c++20"],
            extra_cflags=["-O3", "-std=c++20"],
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to JIT-build neuropress_trellis (needs nvcc and ninja on PATH): {exc}"
        ) from exc


@lru_cache(maxsize=64)
def _max_batch(index: int, ka: int, tile_len: int) -> int:
    return int(_ext().viterbi_max_batch(index, ka, tile_len))


def _scratch_tiles(device: torch.device, ka: int, tile_len: int, n: int) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    resident = _max_batch(index, ka, tile_len)
    per_tile = tile_len * (65536 >> ka)
    return max(1, min(n, resident, _HISTORY_BUDGET_BYTES // per_tile))


def viterbi(
    tiles: torch.Tensor,
    ka: int,
    mask: int,
    codebook: str,
    lut: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``[n, L]`` values as closed trellis walks.

    Args:
        tiles: ``[n, L]`` float32 CUDA tensor, ``L`` a multiple of 32.
        ka: Base bits per step, 1-8.
        mask: 16-position pattern of extra bits; step ``i`` shifts in
            ``ka + (mask >> (i % 16) & 1)`` bits. ``0`` for integer rates.
        codebook: ``"mul1"``, ``"mcg"``, ``"3inst"`` or ``"lut"``.
        lut: ``[65536]`` float16 table, for the ``"lut"`` codebook.

    Returns:
        ``(decoded, states)``: float32 values and int16 16-bit window states, both ``[n, L]``.
    """
    tiles = tiles.float().contiguous()
    n, tile_len = tiles.shape
    out = torch.empty_like(tiles)
    states = torch.empty(tiles.shape, dtype=torch.int16, device=tiles.device)
    if n == 0:
        return out, states
    edges = 65536 >> ka
    batch = _scratch_tiles(tiles.device, ka, tile_len, n)
    hist = torch.empty((batch, tile_len, edges), dtype=torch.uint8, device=tiles.device)
    cost_tiles = batch if ka == 1 else 1
    costs = torch.empty(
        (cost_tiles, 2, edges), dtype=torch.float16, device=tiles.device
    )
    if lut is not None:
        lut = lut.to(tiles.device, torch.float16).contiguous()
    with torch.cuda.device(tiles.device):
        _ext().viterbi(
            tiles, out, states, costs, hist, ka, mask, CODEBOOK_IDS[codebook], lut
        )
    return out, states


def decode(
    states: torch.Tensor, codebook: str, lut: torch.Tensor | None = None
) -> torch.Tensor:
    """Decode int16 trellis states on CUDA; returns float32 of the same shape."""
    states = states.contiguous()
    out = torch.empty(states.shape, dtype=torch.float32, device=states.device)
    if lut is not None:
        lut = lut.to(states.device, torch.float16).contiguous()
    with torch.cuda.device(states.device):
        _ext().decode(states, out, CODEBOOK_IDS[codebook], lut)
    return out


def unpack(packed: torch.Tensor, tile_len: int, ka: int, mask: int) -> torch.Tensor:
    """``[n, words]`` int16 packed tiles to ``[n, tile_len]`` int16 states, on CUDA."""
    packed = packed.contiguous()
    states = torch.empty(
        (packed.shape[0], tile_len), dtype=torch.int16, device=packed.device
    )
    with torch.cuda.device(packed.device):
        _ext().unpack(packed, states, ka, mask)
    return states
