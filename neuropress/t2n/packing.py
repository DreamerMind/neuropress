"""N-bit packing utilities for quantized tensors.

Provides :class:`BitPacking`, a concrete subclass of
:class:`~torch_to_nnef.tensor.quant.base.U8Compressor` that packs *n*-bit
unsigned integer values (stored one per ``uint8`` byte) into bit-packed bytes,
and unpacks them back.

Supported bit-widths: ``1``, ``2``, ``4``, ``8``.

Values are packed LSB-first within each byte.  For ``nbits=4`` the layout is::

    packed_byte = (x[2i] & 0xF) | ((x[2i+1] & 0xF) << 4)

so the first value occupies the low nibble and the second the high nibble.

Usage::

    from neuropress.t2n.packing import BitPacking4

    # u8_blob: uint8 tensor with 4-bit values in [0, 15], one per byte
    compressed = BitPacking4.compress(u8_blob)     # numel halved, shape preserved
    restored   = BitPacking4.decompress(compressed)  # original shape restored
    assert (restored == u8_blob).all()

:class:`BitPacking` instances can be passed as ``u8_compressors`` to any
:class:`~torch_to_nnef.tensor.quant.base.QTensor` constructor so that
:attr:`~torch_to_nnef.tensor.quant.base.QTensor.u8_blob` is stored in packed
form and transparently decompressed during inference::

    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly
    from neuropress.t2n.packing import BitPacking4

    q = QTensorTractScaleOnly(
        fp_tensor, qscheme, u8_compressors=[BitPacking4]
    )

Pre-instantiated singletons (``BitPacking1``, ``BitPacking2``,
``BitPacking4``, ``BitPacking8``) are available for the four supported
bit-widths.
"""

from __future__ import annotations

import torch
from torch_to_nnef.tensor.quant.base import U8Compressor


class BitPacking(U8Compressor):
    """Concrete U8Compressor that packs *n*-bit values into ``uint8`` bytes.

    For each supported bit-width the packing ratio (compressed vs. original
    byte count) is ``nbits / 8``:

    - ``nbits=1``: 8 values per byte  → 8× reduction
    - ``nbits=2``: 4 values per byte  → 4× reduction
    - ``nbits=4``: 2 values per byte  → 2× reduction
    - ``nbits=8``: 1 value per byte   → identity (no reduction)

    The input to :meth:`compress` must have its numel divisible by
    ``8 // nbits`` so that :meth:`decompress` can infer the original numel
    without extra metadata.

    Args:
        nbits: Number of bits per value.  Must be one of ``{1, 2, 4, 8}``.

    Raises:
        ValueError: If *nbits* is not in ``{1, 2, 4, 8}``.
    """

    _SUPPORTED: frozenset[int] = frozenset({1, 2, 4, 8})

    def __init__(self, nbits: int) -> None:
        if nbits not in self._SUPPORTED:
            raise ValueError(
                f"nbits must be one of {sorted(self._SUPPORTED)}, got {nbits}"
            )
        self._nbits: int = nbits
        self._values_per_byte: int = 8 // nbits
        self._mask: int = (1 << nbits) - 1  # e.g. 0xF for nbits=4

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nbits(self) -> int:
        """Number of bits per packed value."""
        return self._nbits

    @property
    def values_per_byte(self) -> int:
        """Number of values stored in each output byte."""
        return self._values_per_byte

    # ------------------------------------------------------------------
    # U8Compressor interface
    # ------------------------------------------------------------------

    def compress(self, u8_tensor: torch.Tensor) -> torch.Tensor:
        """Pack *u8_tensor* (one *n*-bit value per byte) into bit-packed bytes.

        The output numel is ``u8_tensor.numel() // values_per_byte``.
        The input numel must be divisible by ``values_per_byte`` so that
        :meth:`decompress` can reconstruct the original tensor exactly.

        Shape is preserved when possible: for a multi-dimensional input whose
        last dimension is divisible by ``values_per_byte``, the output has the
        same leading dimensions with the last dimension shrunk by the packing
        factor (e.g. ``(64, 32)`` → ``(64, 16)`` for 4-bit packing).
        Otherwise the output is flat.

        Args:
            u8_tensor: ``uint8`` tensor with values in ``[0, 2**nbits - 1]``,
                one value per element.

        Returns:
            ``uint8`` tensor with ``numel // values_per_byte`` elements,
            preserving leading dimensions when the last dim is divisible by
            ``values_per_byte``.

        Raises:
            ValueError: If ``u8_tensor.numel()`` is not divisible by
                ``values_per_byte``.
        """
        if self._nbits == 8:
            return u8_tensor

        vpb = self._values_per_byte
        orig_shape = u8_tensor.shape
        flat = u8_tensor.to(torch.uint8).flatten()

        if flat.numel() % vpb != 0:
            raise ValueError(
                f"BitPacking({self._nbits}): input numel ({flat.numel()}) "
                f"must be divisible by values_per_byte ({vpb}) "
                "to allow unambiguous decompression."
            )

        groups = flat.view(-1, vpb)  # [n_bytes_out, vpb]

        # Vectorized: shift each column left by its bit position then sum.
        # Bits from different columns never overlap so sum == bitwise-OR.
        shifts = torch.arange(vpb, dtype=torch.int32, device=flat.device) * self._nbits
        result = (
            ((groups.to(torch.int32) & self._mask) << shifts)
            .sum(dim=-1)
            .to(torch.uint8)
        )

        # Preserve multi-dim shape: shrink last dimension by the packing factor.
        if len(orig_shape) > 1 and orig_shape[-1] % vpb == 0:
            return result.reshape(*orig_shape[:-1], orig_shape[-1] // vpb)
        return result

    def decompress(self, u8_tensor: torch.Tensor) -> torch.Tensor:
        """Unpack bit-packed bytes back to one *n*-bit value per byte.

        The output numel is ``u8_tensor.numel() * values_per_byte``.

        If the input is multi-dimensional the last dimension is expanded by
        ``values_per_byte`` (i.e. the inverse of :meth:`compress`'s shape
        preservation), so ``decompress(compress(x))`` always has the same
        shape as ``x``.

        Args:
            u8_tensor: ``uint8`` tensor produced by :meth:`compress`.

        Returns:
            ``uint8`` tensor with ``numel * values_per_byte`` elements and
            values in ``[0, 2**nbits - 1]``.  Shape matches the original
            input to :meth:`compress` when shape was preserved.
        """
        if self._nbits == 8:
            return u8_tensor

        vpb = self._values_per_byte
        orig_shape = u8_tensor.shape
        packed = u8_tensor.to(torch.int32).flatten()

        # Vectorized: broadcast packed over vpb shifts, mask each nibble/pair.
        # packed:  [n]      shifts: [vpb]  →  result: [n, vpb]
        # result[i, j] = (packed[i] >> (j*nbits)) & mask  =  value j of byte i
        # reshape(-1) interleaves correctly: v0, v1, v2, v3, …
        shifts = (
            torch.arange(vpb, dtype=torch.int32, device=packed.device) * self._nbits
        )
        out = ((packed.unsqueeze(-1) >> shifts) & self._mask).to(torch.uint8)

        # Restore multi-dim shape: expand last dimension by the packing factor.
        if len(orig_shape) > 1:
            return out.reshape(*orig_shape[:-1], orig_shape[-1] * vpb)
        return out.reshape(-1)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def compressed_numel(self, original_numel: int) -> int:
        """Return the byte count of the packed representation.

        Args:
            original_numel: Number of elements before packing.

        Returns:
            Number of ``uint8`` elements after packing.
        """
        return original_numel // self._values_per_byte


# ---------------------------------------------------------------------------
# Pre-instantiated singletons for common bit-widths
# ---------------------------------------------------------------------------

BitPacking1 = BitPacking(1)
"""Pre-instantiated :class:`BitPacking` for 1-bit values (8 per byte)."""

BitPacking2 = BitPacking(2)
"""Pre-instantiated :class:`BitPacking` for 2-bit values (4 per byte)."""

BitPacking4 = BitPacking(4)
"""Pre-instantiated :class:`BitPacking` for 4-bit values (2 per byte)."""

BitPacking8 = BitPacking(8)
"""Pre-instantiated :class:`BitPacking` for 8-bit values (1 per byte, identity)."""
