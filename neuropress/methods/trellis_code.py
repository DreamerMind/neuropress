"""Bitshift trellis codes: a reusable vector quantizer for sequences of values.

A :class:`TrellisCode` codes a sequence of ``tile_len`` values as one closed walk through a
16-bit shift register (the bitshift trellis of QTIP and EXL3). Step ``i`` shifts ``D(i)`` new
bits in and the 16-bit window decodes to a value through a codebook, so a tile costs
``tile_len * bits`` bits however the values are arranged. Encoding is Viterbi on CUDA
(:mod:`neuropress.kernels.trellis`); decoding, packing and unpacking are plain torch and run
anywhere.

The rate is any multiple of 1/16 bit between 1 and 8: ``D(i) = ka + bit (i mod 16) of mask``,
with the extra bits spread evenly over the 16-position period (0.5 extra bit -> ``0xAAAA``,
EXL3's half-integer pattern).

Codebooks: ``mul1``, ``mcg`` and ``3inst`` are the hash codebooks of QTIP / EXL3 (no stored
table); ``lut`` reads a 65536-entry fp16 table, for learned codebooks.

The code is independent of what the values are: the weight quantizer in
:mod:`neuropress.methods.trellis` feeds it Hadamard-rotated 16x16 weight tiles, but any method
that needs a fixed-rate quantizer of value sequences (residuals, embedding rows, activations
offline) can use it directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import torch

Codebook = Literal["mul1", "mcg", "3inst", "lut"]

# EXL3's normalisation: a weight with this rms fits the hash codebooks' range
HASH_CODEBOOK_SCALE = 1.24371088
_RATE_STEP = 16


def rate_pattern(bits: float) -> tuple[int, int]:
    """``(ka, mask)`` for a rate in 1/16-bit steps, extra bits spread evenly over 16 positions."""
    m16 = bits * _RATE_STEP
    if not float(m16).is_integer() or not (16 <= m16 <= 128):
        raise ValueError(f"trellis bits must be in 1..8 in steps of 1/16, got {bits}")
    ka, extra = divmod(int(m16), _RATE_STEP)
    mask = 0
    for i in range(_RATE_STEP):
        if (i + 1) * extra // _RATE_STEP > i * extra // _RATE_STEP:
            mask |= 1 << i
    return ka, mask


def _sum_halves(x: torch.Tensor) -> torch.Tensor:
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
    hi = (x >> 16).to(torch.int16).view(torch.float16)
    return (lo.float() + hi.float()).half()


def decode_reference(
    states: torch.Tensor, codebook: Codebook, lut: torch.Tensor | None = None
) -> torch.Tensor:
    """Decode 16-bit states in plain torch (any device), bit-exact with the CUDA kernel.

    Args:
        states: Integer tensor of 16-bit window states (int16 values are read as unsigned).
        codebook: Codebook name.
        lut: ``[65536]`` table for ``"lut"``.

    Returns:
        float32 values, same shape as *states*.
    """
    s = states.to(torch.int64) & 0xFFFF
    if codebook == "lut":
        if lut is None:
            raise ValueError("the lut codebook needs a table")
        return lut.to(s.device, torch.float16)[s].float()
    m32 = 0xFFFFFFFF
    if codebook == "3inst":
        return _sum_halves((s * 89226354 + 64248484) & m32).float()
    if codebook == "mcg":
        return _sum_halves((s * 0xCBAC1FED) & m32).float()
    if codebook == "mul1":
        x = (s * 0x83DCD12D) & m32
        byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
        # fp16 fma (1024 + sum) * k_inv + k_bias: one rounding of the exact value
        k_inv = torch.tensor(0x1EEE, dtype=torch.int16).view(torch.float16).double()
        k_bias = (
            torch.tensor(0xC931 - 0x10000, dtype=torch.int16)
            .view(torch.float16)
            .double()
        )
        return ((1024 + byte_sum).double() * k_inv + k_bias).half().float()
    raise ValueError(f"unknown trellis codebook {codebook!r}")


@dataclass(frozen=True)
class TrellisCode:
    """A bitshift trellis code: rate, codebook and tile length.

    Attributes:
        bits: Bits per value, 1..8 in steps of 1/16.
        codebook: Codebook name.
        tile_len: Values per closed walk, a multiple of 32.
        lut: ``[65536]`` fp16 table for the ``"lut"`` codebook.
    """

    bits: float = 3
    codebook: Codebook = "mul1"
    tile_len: int = 256
    lut: torch.Tensor | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        rate_pattern(self.bits)  # validates the rate
        if self.tile_len < 32 or self.tile_len % 32:
            raise ValueError(f"tile_len must be a multiple of 32, got {self.tile_len}")
        if (self.codebook == "lut") != (self.lut is not None):
            raise ValueError("a lut table goes with, and only with, the lut codebook")
        if self.lut is not None and self.lut.numel() != 65536:
            raise ValueError("the lut table must hold 65536 values")

    @property
    def ka(self) -> int:
        """Base bits per step."""
        return rate_pattern(self.bits)[0]

    @property
    def mask(self) -> int:
        """Extra-bit pattern over 16 positions."""
        return rate_pattern(self.bits)[1]

    @cached_property
    def widths(self) -> torch.Tensor:
        """``[tile_len]`` bits shifted in at each position."""
        i = torch.arange(self.tile_len)
        return self.ka + ((self.mask >> (i % _RATE_STEP)) & 1)

    @property
    def tile_bits(self) -> int:
        """Bits of one packed tile."""
        return int(self.widths.sum())

    @property
    def tile_words(self) -> int:
        """int16 words of one packed tile (the last one zero-padded)."""
        return math.ceil(self.tile_bits / 16)

    @property
    def scale(self) -> float:
        """Rms a value should have to fit the codebook's range."""
        if self.lut is None:
            return HASH_CODEBOOK_SCALE
        return float(self.lut.float().square().mean().sqrt())

    # -- encode / decode -----------------------------------------------------

    def encode(self, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Viterbi-code ``[n, tile_len]`` values on CUDA.

        Returns:
            ``(decoded, states)``: float32 reconstruction and int16 states.
        """
        from ..kernels.trellis import viterbi

        if tiles.dim() != 2 or tiles.shape[1] != self.tile_len:
            raise ValueError(
                f"expected [n, {self.tile_len}] tiles, got {tuple(tiles.shape)}"
            )
        if not tiles.is_cuda:
            raise ValueError("trellis encoding runs on CUDA")
        return viterbi(tiles, self.ka, self.mask, self.codebook, self.lut)

    def decode(self, states: torch.Tensor) -> torch.Tensor:
        """Decode states (CUDA kernel on CUDA, torch elsewhere); float32 values."""
        if states.is_cuda:
            from ..kernels.trellis import decode

            return decode(states.to(torch.int16), self.codebook, self.lut)
        return decode_reference(states, self.codebook, self.lut)

    # -- packing -------------------------------------------------------------

    @cached_property
    def _pack_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        """For each stream bit: its source position and the bit to read from that state."""
        w = self.widths
        src = torch.repeat_interleave(torch.arange(self.tile_len), w)
        offs = torch.cat([torch.arange(int(d) - 1, -1, -1) for d in w])
        return src, offs

    @cached_property
    def _unpack_index(self) -> torch.Tensor:
        """``[tile_len, 16]`` stream bit positions of each window, most significant first."""
        ends = torch.cumsum(self.widths, 0)
        pos = ends[:, None] - 16 + torch.arange(16)[None, :]
        return pos % self.tile_bits

    def pack(self, states: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Pack ``[..., tile_len]`` states into ``[..., tile_words]`` int16 words.

        The stream holds each position's ``D(i)`` new bits, most significant first; a window is
        the 16 stream bits ending at its position (cyclically), so a closed walk is stored in
        exactly ``tile_bits`` bits.
        """
        lead = states.shape[:-1]
        s = states.reshape(-1, self.tile_len).to(torch.int32) & 0xFFFF
        src, offs = (t.to(s.device) for t in self._pack_index)
        pad = self.tile_words * 16 - self.tile_bits
        weights = (1 << torch.arange(15, -1, -1, device=s.device)).to(torch.int32)
        out = []
        for c in range(0, s.shape[0], chunk):
            bits = (s[c : c + chunk, src] >> offs) & 1
            bits = torch.nn.functional.pad(bits, (0, pad))
            words = (bits.view(-1, self.tile_words, 16) * weights).sum(-1)
            out.append(words.to(torch.int32).to(torch.int16))
        packed = (
            torch.cat(out)
            if out
            else s.new_zeros((0, self.tile_words), dtype=torch.int16)
        )
        return packed.reshape(*lead, self.tile_words)

    def unpack(self, packed: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Inverse of :meth:`pack`: ``[..., tile_words]`` int16 words to int16 states.

        Runs a CUDA kernel for CUDA tensors, torch elsewhere; both give the same states.
        """
        lead = packed.shape[:-1]
        if packed.is_cuda:
            from ..kernels.trellis import unpack

            flat = packed.reshape(-1, self.tile_words).to(torch.int16)
            return unpack(flat, self.tile_len, self.ka, self.mask).reshape(
                *lead, self.tile_len
            )
        p = packed.reshape(-1, self.tile_words).to(torch.int32) & 0xFFFF
        pos = self._unpack_index.to(p.device)
        shifts = torch.arange(15, -1, -1, device=p.device, dtype=torch.int32)
        weights = (1 << shifts).to(torch.int32)
        out = []
        for c in range(0, p.shape[0], chunk):
            bits = ((p[c : c + chunk, :, None] >> shifts) & 1).reshape(
                -1, self.tile_words * 16
            )
            win = bits[:, pos]  # [n, tile_len, 16]
            out.append((win * weights).sum(-1).to(torch.int32).to(torch.int16))
        states = (
            torch.cat(out)
            if out
            else p.new_zeros((0, self.tile_len), dtype=torch.int16)
        )
        return states.reshape(*lead, self.tile_len)
