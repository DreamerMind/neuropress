"""Index packing for Vector Quantization codebook indices.

Packs VQ centroid indices into compact ``uint8`` storage using a
contiguous bit-stream layout that supports any bit width from 4 to 16.

Distinction from :class:`~neuropress.t2n.packing.BitPacking`
--------------------------------------------------------------
:class:`~neuropress.t2n.packing.BitPacking` operates on quantized weight
*values* (e.g. 4-bit nibbles representing dequantized scalars).
:class:`IndexPacking` operates on VQ codebook *indices* — integers that
point into a shared centroid table.  While the bit-level packing mechanics
are identical, the semantic meaning differs: an index references a full
``vector_len``-dimensional centroid, not a single scalar.

Bits-per-weight (bpw) — correct formula
----------------------------------------
A common mistake is to equate bpw with ``log2(NC) / VL``.  That is only
correct for flat (uniform) encoding and ignores both the centroid table
overhead and the actual index frequency distribution.  The true formula is::

    bpw = H(p) / VL  +  (NC × VL × 16) / total_params

where:

* ``H(p) = -∑ᵢ pᵢ log₂(pᵢ)`` is the Shannon entropy of the centroid
  assignment distribution — the theoretical lower bound on average
  bits per index under optimal entropy coding.
* ``NC × VL × 16 / total_params`` amortises the fp16 centroid table
  over all quantized parameters (negligible for large models with a
  shared codebook, significant for small models or per-layer codebooks
  with large NC).

``H(p)`` is **not** a function of NC alone.  It depends on the shape of
the frequency curve.  For a Zipf distribution with exponent ``s``:

* ``s → 0`` (uniform): ``H = log₂(NC)``  — flat encoding, worst case
* ``s = 1`` (classic Zipf): ``H ≈ log₂(NC) / 2``  — rough rule of thumb
* ``s → ∞`` (one dominant centroid): ``H → 0``

The same NC can therefore yield very different bpw depending on where
the model's centroid assignments sit on the power-law curve.  The only
correct way to report bpw for a specific model is to measure the actual
frequency distribution and call one of the ``effective_bits`` methods
(which give the expected code length for the chosen prefix code, an upper
bound on ``H(p)``)::

    # After quantising a model:
    eff = TieredIndexPacking.effective_bits(n_hot, n_total, cold_bits)
    bpw = eff / vector_len + (NC * vector_len * 16) / total_params

Supported bit widths
--------------------
The number of bits required per index is determined by the codebook size
via ``ceil(log2(num_centroids))``, clamped to the range ``[4, 16]``:

    - ``num_centroids <=    16``  →  4 bits/index
    - ``num_centroids <=    32``  →  5 bits/index
    - ``num_centroids <=    64``  →  6 bits/index
    - ``num_centroids <=   128``  →  7 bits/index
    - ``num_centroids <=   256``  →  8 bits/index
    - ``num_centroids <=   512``  →  9 bits/index
    - ``num_centroids <=  1024``  → 10 bits/index
    - ``num_centroids <=  2048``  → 11 bits/index
    - ``num_centroids <=  4096``  → 12 bits/index
    - ``num_centroids <=  8192``  → 13 bits/index
    - ``num_centroids <= 16384``  → 14 bits/index
    - ``num_centroids <= 32768``  → 15 bits/index
    - ``num_centroids <= 65536``  → 16 bits/index

Bit-stream layout
-----------------
Indices are packed into a contiguous bit stream, LSB first.  Index ``i``
occupies bits ``[i*bits, (i+1)*bits)`` of the stream.  The stream is
stored as a sequence of ``uint8`` bytes; the final byte may carry unused
high bits (always zero).

Power-of-2 widths (4, 8) use optimised paths identical to the original
implementation.  All other widths use the generic bit-stream path.

Usage::

    from neuropress.transforms.vq_packing import IndexPacking

    bits = IndexPacking.bits_for(num_centroids=256)  # → 8
    packed = IndexPacking.pack(indices, bits)         # uint8
    restored = IndexPacking.unpack(packed, bits, n_indices=indices.numel())
    assert (restored == indices).all()
"""

from __future__ import annotations

import math

import torch


class IndexPacking:
    """Static utilities for packing and unpacking VQ codebook indices.

    All methods are ``@staticmethod``; no instance is needed.  The bit
    width is always derived from the codebook size via :meth:`bits_for`
    and stored alongside the packed buffer (e.g. in
    ``VQScheme.is_packed_indices``).
    """

    _MIN_BITS: int = 4
    _MAX_BITS: int = 20
    _MAX_CENTROIDS: int = 1 << 20  # 1M centroids (20-bit indices)

    @staticmethod
    def bits_for(num_centroids: int) -> int:
        """Return the minimum bit width required for a given codebook size.

        Computes ``max(4, ceil(log2(num_centroids)))`` clamped to ``[4, 20]``.

        Args:
            num_centroids: Number of entries in the VQ codebook.

        Returns:
            Integer in ``[4, 20]`` — the number of bits stored per index.

        Raises:
            ValueError: If ``num_centroids < 2`` or exceeds 1M.
        """
        if num_centroids < 2:
            raise ValueError(f"num_centroids must be >= 2, got {num_centroids}.")
        if num_centroids > IndexPacking._MAX_CENTROIDS:
            raise ValueError(
                f"num_centroids={num_centroids} exceeds the maximum supported "
                f"value of {IndexPacking._MAX_CENTROIDS} (20-bit indices)."
            )
        raw = math.ceil(math.log2(num_centroids))
        return max(IndexPacking._MIN_BITS, raw)

    @staticmethod
    def pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack a flat integer index tensor into ``uint8`` storage.

        Uses an optimised path for power-of-2 widths (4, 8) and a generic
        bit-stream path for all other widths in ``[4, 16]``.

        Args:
            indices: 1-D integer tensor with values in ``[0, 2**bits - 1]``.
                Must be flat (call ``.flatten()`` first if needed).
            bits: Bits per index — integer in ``[4, 16]``.

        Returns:
            1-D ``uint8`` tensor.  Byte count is
            ``ceil(indices.numel() * bits / 8)``.

        Raises:
            ValueError: If *bits* is outside ``[4, 16]``.
            ValueError: If *bits=4* and ``indices.numel()`` is not even.
        """
        if not (IndexPacking._MIN_BITS <= bits <= IndexPacking._MAX_BITS):
            raise ValueError(
                f"bits must be in [{IndexPacking._MIN_BITS}, "
                f"{IndexPacking._MAX_BITS}], got {bits}."
            )
        flat = indices.flatten().to(torch.int64)
        if bits == 8:
            return flat.to(torch.uint8)
        if bits == 4:
            if flat.numel() % 2 != 0:
                raise ValueError(
                    f"IndexPacking(bits=4): index count ({flat.numel()}) "
                    "must be even for 4-bit packing."
                )
            groups = flat.view(-1, 2)
            packed = (groups[:, 0] & 0xF) | ((groups[:, 1] & 0xF) << 4)
            return packed.to(torch.uint8)
        return IndexPacking._pack_bitstream(flat, bits)

    @staticmethod
    def unpack(packed: torch.Tensor, bits: int, n_indices: int) -> torch.Tensor:
        """Unpack a ``uint8`` buffer back to a 1-D integer index tensor.

        Args:
            packed: 1-D ``uint8`` tensor produced by :meth:`pack`.
            bits: Bits per index — must match the value used during packing.
            n_indices: Expected number of indices in the output.  Used to
                trim any padding introduced during packing.

        Returns:
            1-D ``torch.int64`` tensor of length *n_indices*.

        Raises:
            ValueError: If *bits* is outside ``[4, 16]``.
        """
        if not (IndexPacking._MIN_BITS <= bits <= IndexPacking._MAX_BITS):
            raise ValueError(
                f"bits must be in [{IndexPacking._MIN_BITS}, "
                f"{IndexPacking._MAX_BITS}], got {bits}."
            )
        if bits == 8:
            return packed.to(torch.int64)[:n_indices]
        if bits == 4:
            p = packed.to(torch.int32)
            lo = p & 0xF
            hi = (p >> 4) & 0xF
            interleaved = torch.stack([lo, hi], dim=1).flatten()
            return interleaved.to(torch.int64)[:n_indices]
        return IndexPacking._unpack_bitstream(packed, bits, n_indices)

    # ------------------------------------------------------------------
    # Generic bit-stream helpers (non-power-of-2 widths)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_bitstream(flat: torch.Tensor, bits: int) -> torch.Tensor:
        """Pack *flat* int64 indices using a contiguous ``bits``-wide bit stream.

        Index ``i`` occupies bits ``[i*bits, (i+1)*bits)`` of the output
        stream, stored LSB-first within each byte.

        Args:
            flat: 1-D ``int64`` tensor.
            bits: Bits per index in ``[4, 16]``.

        Returns:
            1-D ``uint8`` tensor of length ``ceil(flat.numel() * bits / 8)``.
        """
        n = flat.numel()
        dev = flat.device

        # CUDA fast path: single kernel launch via atomicOr (no intermediate
        # int32 buffer, no scatter_add_ loop).
        if dev.type == "cuda":
            from neuropress.kernels import pack_indices as _pack_kernel

            free, _ = torch.cuda.mem_get_info(dev)
            n_bytes = (n * bits + 7) // 8
            # Fall back to CPU when even the output buffer would stress VRAM.
            if n_bytes <= free * 0.4:
                try:
                    return _pack_kernel(flat, n, bits)
                except RuntimeError:
                    pass  # kernel unavailable, fall through to CPU path
            # For very large layers (embed_tokens etc.) pack on CPU.
            flat = flat.cpu()
            dev = torch.device("cpu")

        n = flat.numel()
        n_bytes = (n * bits + 7) // 8
        result = torch.zeros(n_bytes, dtype=torch.int32, device=dev)
        idx = torch.arange(n, dtype=torch.int64, device=dev)
        for b in range(bits):
            bit_b = ((flat >> b) & 1).to(torch.int32)
            pos = idx * bits + b
            byte_idx = (pos >> 3).to(torch.int64)
            bit_off = (pos & 7).to(torch.int32)
            result.scatter_add_(0, byte_idx, bit_b << bit_off)
        return result.to(torch.uint8)

    @staticmethod
    def _unpack_bitstream(
        packed: torch.Tensor, bits: int, n_indices: int
    ) -> torch.Tensor:
        """Unpack *n_indices* ``bits``-wide indices from a bit-stream buffer.

        Args:
            packed: 1-D ``uint8`` tensor.
            bits: Bits per index in ``[4, 16]``.
            n_indices: Number of indices to extract.

        Returns:
            1-D ``int64`` tensor of length *n_indices*.
        """
        if packed.is_cuda:
            from neuropress.kernels import unpack_indices

            out_bytes = n_indices * 8  # int64
            free, _ = torch.cuda.mem_get_info(packed.device)
            if out_bytes > free * 0.5:
                # Reserved-but-fragmented memory may block the alloc; release it.
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(packed.device)
            if out_bytes <= free * 0.5:
                try:
                    return unpack_indices(packed, n_indices, bits)
                except RuntimeError:
                    pass  # kernel unavailable, fall through to CPU path
            packed = packed.cpu()
        # CPU fallback: Python bit-loop (only used outside GPU context).
        dev = packed.device
        p = packed.to(torch.int64)
        result = torch.zeros(n_indices, dtype=torch.int64, device=dev)
        idx = torch.arange(n_indices, dtype=torch.int64, device=dev)
        for b in range(bits):
            pos = idx * bits + b
            byte_idx = pos >> 3
            bit_off = pos & 7
            bit_b = (p[byte_idx] >> bit_off) & 1
            result = result | (bit_b << b)
        return result


class TieredIndexPacking:
    """Two-tier prefix-coded packing for frequency-sorted VQ indices.

    Exploits the Zipf-skewed distribution of centroid assignments: a 1-bit
    prefix selects between a *hot* tier (``hot_bits``-bit) and a *cold*
    tier (``cold_bits``-bit), saving bits when the top centroids account
    for a large fraction of all assignments.

    Expected savings at various hot-tier fractions (``p_hot``):

        bits per index = 1 (prefix) + hot_bits x p_hot + cold_bits x (1 − p_hot)

    Wire format (single ``uint8`` buffer, little-endian header)::

        [0:4]   n_indices  (uint32 LE) — total index count
        [4:8]   n_hot      (uint32 LE) — number of hot indices
        [8:12]  cold_bits  (uint32 LE) — bits used for cold indices
        [12:16] hot_bits   (uint32 LE) — bits used for hot indices
        [16:16+prefix_bytes]  prefix stream  (1 bit per index, LSB first)
        [16+prefix_bytes : ...]  hot stream  (hot_bits-bit packed)
        [...  : end]  cold stream  (cold_bits-bit packed)

    Backward compatibility: buffers with the old 12-byte header (``hot_bits``
    absent) are read as ``hot_bits=4``.

    Prerequisites
    -------------
    Centroids **must** be frequency-sorted (index 0 = most-assigned centroid)
    before packing.

    Attributes:
        DEFAULT_HOT_BITS: Default hot-tier bit width (4 → 16 hot centroids).
        HEADER_BYTES: Length of the fixed-size binary header (16 bytes).
    """

    DEFAULT_HOT_BITS: int = 4
    HEADER_BYTES: int = 16
    _LEGACY_HEADER_BYTES: int = 12  # old format without hot_bits field

    @staticmethod
    def hot_threshold(hot_bits: int) -> int:
        """Return the hot-tier centroid count for a given *hot_bits*.

        Args:
            hot_bits: Bit width of hot-tier indices.

        Returns:
            ``2 ** hot_bits`` — centroids ``0 … hot_threshold-1`` are hot.
        """
        return 1 << hot_bits

    @staticmethod
    def pack(
        indices: torch.Tensor,
        cold_bits: int,
        hot_bits: int = 4,
    ) -> torch.Tensor:
        """Pack frequency-sorted indices using a two-tier prefix code.

        Args:
            indices: 1-D ``int64`` tensor of centroid indices (must already
                be frequency-sorted so that low indices are most common).
            cold_bits: Bit width for cold-tier indices; use
                :meth:`~IndexPacking.bits_for` to derive from codebook size.
            hot_bits: Bit width for hot-tier indices (default 4 → top 16
                centroids).  Use 8 for large codebooks (128K+) to cache
                256 hot centroids.

        Returns:
            Self-contained ``uint8`` buffer (header + prefix + hot + cold).

        Raises:
            ValueError: If *cold_bits* or *hot_bits* is outside valid range.
        """
        if not (IndexPacking._MIN_BITS <= cold_bits <= IndexPacking._MAX_BITS):
            raise ValueError(
                f"cold_bits must be in [{IndexPacking._MIN_BITS}, "
                f"{IndexPacking._MAX_BITS}], got {cold_bits}."
            )
        if not (4 <= hot_bits <= 16):
            raise ValueError(f"hot_bits must be in [4, 16], got {hot_bits}.")
        flat = indices.flatten().to(torch.int64)
        dev = flat.device
        n = flat.numel()
        threshold = TieredIndexPacking.hot_threshold(hot_bits)
        hot_mask = flat < threshold  # bool [n]
        n_hot = int(hot_mask.sum().item())
        n_cold = n - n_hot

        # --- prefix stream: 1 bit per index (0=hot, 1=cold) ---------------
        prefix_bits = (~hot_mask).to(torch.int64)  # 0=hot, 1=cold
        prefix_stream = IndexPacking._pack_bitstream(prefix_bits, 1)

        # --- hot stream: hot_bits-bit packed hot indices --------------------
        hot_indices = flat[hot_mask]
        if n_hot > 0:
            if hot_bits == 4 and n_hot % 2 != 0:
                # Pad to even for 4-bit packing.
                hot_indices = torch.cat(
                    [hot_indices, torch.zeros(1, dtype=torch.int64, device=dev)]
                )
            hot_stream = IndexPacking.pack(hot_indices, hot_bits)
        else:
            hot_stream = torch.zeros(0, dtype=torch.uint8, device=dev)

        # --- cold stream: cold_bits-bit packed cold indices ----------------
        cold_indices = flat[~hot_mask]
        if n_cold > 0:
            cold_stream = IndexPacking.pack(cold_indices, cold_bits)
        else:
            cold_stream = torch.zeros(0, dtype=torch.uint8, device=dev)

        # --- header (16 bytes, 4 x uint32 LE) ------------------------------
        header = torch.zeros(4, dtype=torch.int32, device=dev)
        header[0] = n
        header[1] = n_hot
        header[2] = cold_bits
        header[3] = hot_bits
        header_bytes = header.view(torch.uint8)

        return torch.cat([header_bytes, prefix_stream, hot_stream, cold_stream])

    @staticmethod
    def _parse_header(packed: torch.Tensor) -> tuple[int, int, int, int, int]:
        """Parse the tiered buffer header.

        Handles both legacy 12-byte headers (``hot_bits`` absent, defaults
        to 4) and current 16-byte headers.

        Args:
            packed: 1-D ``uint8`` buffer.

        Returns:
            ``(n_indices, n_hot, cold_bits, hot_bits, header_bytes)``

        Raises:
            ValueError: If the buffer is too short for even a legacy header.
        """
        H_legacy = TieredIndexPacking._LEGACY_HEADER_BYTES
        H_new = TieredIndexPacking.HEADER_BYTES
        if packed.numel() < H_legacy:
            raise ValueError(
                f"TieredIndexPacking: buffer too short "
                f"({packed.numel()} bytes, need at least {H_legacy})."
            )
        # Read 3 mandatory fields.
        hdr3 = packed[:H_legacy].view(torch.int32)
        n = int(hdr3[0].item())
        n_hot = int(hdr3[1].item())
        cold_bits = int(hdr3[2].item())

        # Detect format version: if buffer is long enough for 4th field
        # and 4th field is a plausible hot_bits (4-16), treat as new format.
        if packed.numel() >= H_new:
            hdr4 = packed[H_legacy:H_new].view(torch.int32)
            candidate = int(hdr4[0].item())
            if 4 <= candidate <= 16:
                return n, n_hot, cold_bits, candidate, H_new

        # Legacy: hot_bits=4, header=12 bytes.
        return n, n_hot, cold_bits, 4, H_legacy

    @staticmethod
    def unpack(packed: torch.Tensor) -> torch.Tensor:
        """Unpack a tiered buffer produced by :meth:`pack`.

        Handles both legacy (12-byte) and current (16-byte) headers.

        Args:
            packed: 1-D ``uint8`` buffer (header + streams).

        Returns:
            1-D ``int64`` tensor of centroid indices.

        Raises:
            ValueError: If the buffer is too short to contain a valid header.
        """
        n, n_hot, cold_bits, hot_bits, H = TieredIndexPacking._parse_header(packed)
        n_cold = n - n_hot

        offset = H

        # --- decode prefix stream ------------------------------------------
        prefix_bytes = (n + 7) // 8
        prefix_flat = IndexPacking._unpack_bitstream(
            packed[offset : offset + prefix_bytes], 1, n
        )
        offset += prefix_bytes

        # 0=hot, 1=cold
        hot_positions = (prefix_flat == 0).nonzero(as_tuple=True)[0]
        cold_positions = (prefix_flat == 1).nonzero(as_tuple=True)[0]

        # --- decode hot stream ---------------------------------------------
        hot_bytes = math.ceil(n_hot * hot_bits / 8)
        if n_hot > 0:
            hot_indices_padded = IndexPacking.unpack(
                packed[offset : offset + hot_bytes],
                hot_bits,
                n_indices=n_hot,
            )
        else:
            hot_indices_padded = torch.zeros(0, dtype=torch.int64, device=packed.device)
        offset += hot_bytes

        # --- decode cold stream --------------------------------------------
        cold_bytes = math.ceil(n_cold * cold_bits / 8)
        if n_cold > 0:
            cold_indices = IndexPacking.unpack(
                packed[offset : offset + cold_bytes],
                cold_bits,
                n_indices=n_cold,
            )
        else:
            cold_indices = torch.zeros(0, dtype=torch.int64, device=packed.device)

        # --- scatter back into original positions --------------------------
        result = torch.zeros(n, dtype=torch.int64, device=packed.device)
        if n_hot > 0:
            result[hot_positions] = hot_indices_padded
        if n_cold > 0:
            result[cold_positions] = cold_indices
        return result

    @staticmethod
    def compute_row_offsets(
        packed: torch.Tensor, n_rows: int, n_groups: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-row cumulative hot/cold counts from a tiered buffer.

        Required by the CUDA kernel, which needs to jump directly to row n's
        slice of the hot and cold streams without scanning all prior rows.

        Args:
            packed: 1-D ``uint8`` tiered buffer (from :meth:`pack`).
            n_rows: Number of weight rows (``out_features``).
            n_groups: Number of VQ groups per row (``in_features // vector_len``).

        Returns:
            ``(hot_row_offsets, cold_row_offsets)``: two ``int32`` CPU tensors
            of length ``n_rows + 1`` each.  Entry ``i`` is the cumulative count
            of hot (resp. cold) indices in rows ``0 … i-1``.

        Raises:
            ValueError: If ``n_rows * n_groups`` doesn't match the n_indices in
                the header.
        """
        n_indices, _, _, _, H = TieredIndexPacking._parse_header(packed)
        if n_indices != n_rows * n_groups:
            raise ValueError(
                f"compute_row_offsets: n_rows*n_groups={n_rows * n_groups} "
                f"!= n_indices={n_indices} in header."
            )
        prefix_bytes = (n_indices + 7) // 8
        prefix_stream = packed[H : H + prefix_bytes].cpu()

        hot_off = torch.zeros(n_rows + 1, dtype=torch.int32)
        cold_off = torch.zeros(n_rows + 1, dtype=torch.int32)

        for row in range(n_rows):
            h = 0
            c = 0
            for g in range(n_groups):
                flat = row * n_groups + g
                byte_i = flat >> 3
                bit_i = flat & 7
                is_cold = int((prefix_stream[byte_i].item() >> bit_i) & 1)
                if is_cold:
                    c += 1
                else:
                    h += 1
            hot_off[row + 1] = hot_off[row] + h
            cold_off[row + 1] = cold_off[row] + c

        return hot_off, cold_off

    @staticmethod
    def effective_bits(
        n_hot: int,
        n_total: int,
        cold_bits: int,
        hot_bits: int = 4,
    ) -> float:
        """Average bits per index under this 2-tier prefix code.

        This is an empirical upper bound on the Shannon entropy ``H(p)``
        of the assignment distribution, computed from the measured tier
        occupancies.  Use it — not ``log2(NC)`` — when estimating bpw::

            bpw = effective_bits(n_hot, n_total, cold_bits) / VL
                  + (NC * VL * 16) / total_params

        The gap between this value and the true entropy ``H(p)`` reflects
        the inefficiency of a two-level prefix code vs. optimal arithmetic
        coding; it is typically 0.5–2 bits/index for realistic distributions.

        Args:
            n_hot: Number of indices assigned to the hot tier
                (centroids ``0 … 2**hot_bits - 1``).
            n_total: Total number of indices.
            cold_bits: Flat bit width used for cold-tier indices
                (= ``ceil(log2(NC))``).
            hot_bits: Flat bit width used for hot-tier indices (default 4).

        Returns:
            Expected bits per index, including the 1-bit hot/cold prefix.
            Equal to ``log2(NC)`` in the worst case (all cold, uniform).
        """
        if n_total == 0:
            return float(cold_bits)
        p_hot = n_hot / n_total
        return 1.0 + hot_bits * p_hot + cold_bits * (1.0 - p_hot)


class NTieredIndexPacking:
    """N-tier cascaded binary splitting packing for VQ codebook indices.

    Generalises :class:`TieredIndexPacking` from 2 tiers to an arbitrary
    depth ``D`` (2, 3, or 4).  Each level ``k`` uses a 1-bit prefix to
    split the current population into "tier-k" (encoded in ``tier_bits[k]``
    bits) and "rest" (fed to level ``k+1``).  The last tier is the leaf
    (no prefix needed).

    Wire format (``uint8`` buffer, little-endian header)::

        Header (12 + 8*D bytes):
          [0:4]     magic       = 0x4E565154
          [4:8]     n_indices   (uint32)
          [8:12]    n_tiers     (uint32)   -- D
          For k in 0..D-1:
            [12+8k   : 12+8k+4]  n_tier_k  (uint32)
            [12+8k+4 : 12+8k+8]  bits_k    (uint32)

        Streams (concatenated after header):
          prefix_0   : ceil(n_indices / 8) bytes
          prefix_1   : ceil(n_rest_0 / 8) bytes
          ...
          prefix_{D-2}: ceil(n_rest_{D-3} / 8) bytes
          tier_0_stream .. tier_{D-1}_stream

    Tier boundaries are cumulative powers of 2:
    ``tier_bits = [4, 8, 12, 17]`` means tier-0 covers centroids
    ``[0, 16)``, tier-1 covers ``[16, 256)``, tier-2 covers
    ``[256, 4096)``, tier-3 (cold) covers ``[4096, 131072)``.

    Attributes:
        MAGIC: Header magic number for format detection.
        MIN_TIERS: Minimum number of tiers (2).
        MAX_TIERS: Maximum number of tiers (4).
    """

    MAGIC: int = 0x4E565154
    MIN_TIERS: int = 2
    MAX_TIERS: int = 4

    @staticmethod
    def tier_thresholds(tier_bits: list[int]) -> list[int]:
        """Return the cumulative centroid thresholds for each tier.

        Tier ``k`` covers centroids ``[threshold[k-1], threshold[k])``.
        Tier ``D-1`` (cold) covers everything from ``threshold[D-2]``
        to ``2 ** tier_bits[D-1]``.

        Args:
            tier_bits: Bit widths per tier, e.g. ``[4, 8, 12, 17]``.

        Returns:
            List of ``D`` thresholds.  ``threshold[k] = 2 ** tier_bits[k]``
            for ``k < D-1``.  The last entry is ``2 ** tier_bits[D-1]``
            (the full codebook capacity).
        """
        return [1 << b for b in tier_bits]

    @staticmethod
    def _validate_tier_bits(tier_bits: list[int]) -> None:
        """Validate tier_bits list.

        Args:
            tier_bits: Bit widths per tier.

        Raises:
            ValueError: If tier count or bit widths are out of range,
                or if bit widths are not strictly increasing.
        """
        d = len(tier_bits)
        if not (NTieredIndexPacking.MIN_TIERS <= d <= NTieredIndexPacking.MAX_TIERS):
            raise ValueError(
                f"tier_bits length must be in "
                f"[{NTieredIndexPacking.MIN_TIERS}, "
                f"{NTieredIndexPacking.MAX_TIERS}], got {d}."
            )
        for i, b in enumerate(tier_bits):
            if not (IndexPacking._MIN_BITS <= b <= IndexPacking._MAX_BITS):
                raise ValueError(
                    f"tier_bits[{i}]={b} out of range "
                    f"[{IndexPacking._MIN_BITS}, {IndexPacking._MAX_BITS}]."
                )
        for i in range(1, d):
            if tier_bits[i] <= tier_bits[i - 1]:
                raise ValueError(
                    f"tier_bits must be strictly increasing, got "
                    f"tier_bits[{i - 1}]={tier_bits[i - 1]}, "
                    f"tier_bits[{i}]={tier_bits[i]}."
                )

    @staticmethod
    def pack(
        indices: torch.Tensor,
        tier_bits: list[int],
    ) -> torch.Tensor:
        """Pack indices using cascaded binary splitting.

        Args:
            indices: 1-D ``int64`` tensor of centroid indices.
            tier_bits: Bit widths per tier, strictly increasing.
                E.g. ``[4, 8, 12, 17]`` for a 4-tier scheme.

        Returns:
            Self-contained ``uint8`` buffer (header + prefix streams +
            tier data streams).

        Raises:
            ValueError: If *tier_bits* is invalid.
        """
        NTieredIndexPacking._validate_tier_bits(tier_bits)
        flat = indices.flatten().to(torch.int64)
        dev = flat.device
        n = flat.numel()
        d = len(tier_bits)
        thresholds = NTieredIndexPacking.tier_thresholds(tier_bits)

        # Classify each index into its tier.
        # tier_id[i] = k where threshold[k-1] <= index < threshold[k].
        tier_id = torch.full((n,), d - 1, dtype=torch.int64, device=dev)
        for k in range(d - 1):
            lo = 0 if k == 0 else thresholds[k - 1]
            hi = thresholds[k]
            mask = (flat >= lo) & (flat < hi)
            tier_id[mask] = k

        # Count per tier.
        tier_counts = [int((tier_id == k).sum().item()) for k in range(d)]

        # Build cascaded prefix streams and extract tier indices.
        prefix_streams: list[torch.Tensor] = []
        tier_data: list[torch.Tensor] = []
        remaining_mask = torch.ones(n, dtype=torch.bool, device=dev)

        for k in range(d - 1):
            # Among the current remaining population, which are tier-k?
            is_tier_k = remaining_mask & (tier_id == k)
            # Prefix: 0 = tier-k, 1 = rest (among remaining only).
            remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]
            n_remaining = remaining_indices.numel()
            prefix_bits = torch.ones(n_remaining, dtype=torch.int64, device=dev)
            # Map is_tier_k back to positions within remaining.
            is_tier_k_in_remaining = is_tier_k[remaining_indices]
            prefix_bits[is_tier_k_in_remaining] = 0
            prefix_streams.append(IndexPacking._pack_bitstream(prefix_bits, 1))

            # Extract tier-k indices.
            tier_k_vals = flat[is_tier_k]
            # Remap to local range: subtract lower threshold.
            lo = 0 if k == 0 else thresholds[k - 1]
            tier_k_local = tier_k_vals - lo
            tier_data.append(tier_k_local)

            # Remove tier-k from remaining.
            remaining_mask = remaining_mask & ~is_tier_k

        # Last tier: everything remaining (no prefix needed).
        is_last = remaining_mask
        last_vals = flat[is_last]
        lo_last = thresholds[d - 2]
        tier_data.append(last_vals - lo_last)

        # Pack each tier's data stream.
        tier_streams: list[torch.Tensor] = []
        for k in range(d):
            td = tier_data[k]
            if td.numel() > 0:
                bits_k = tier_bits[k]
                if bits_k == 4 and td.numel() % 2 != 0:
                    td = torch.cat([td, torch.zeros(1, dtype=torch.int64, device=dev)])
                tier_streams.append(IndexPacking.pack(td, bits_k))
            else:
                tier_streams.append(torch.zeros(0, dtype=torch.uint8, device=dev))

        # Build header: magic + n_indices + n_tiers + per-tier (count, bits).
        header_ints = [NTieredIndexPacking.MAGIC, n, d]
        for k in range(d):
            header_ints.extend([tier_counts[k], tier_bits[k]])
        header = torch.tensor(header_ints, dtype=torch.int32, device=dev).view(
            torch.uint8
        )

        parts = [header, *prefix_streams, *tier_streams]
        return torch.cat(parts)

    @staticmethod
    def _parse_header(
        packed: torch.Tensor,
    ) -> tuple[int, int, list[int], list[int], int]:
        """Parse an n-tier buffer header.

        Args:
            packed: 1-D ``uint8`` buffer.

        Returns:
            ``(n_indices, n_tiers, tier_counts, tier_bits_list, header_bytes)``

        Raises:
            ValueError: If the magic number doesn't match or the buffer
                is too short.
        """
        if packed.numel() < 12:
            raise ValueError(
                f"NTieredIndexPacking: buffer too short "
                f"({packed.numel()} bytes, need at least 12)."
            )
        hdr = packed[:12].view(torch.int32)
        magic = int(hdr[0].item())
        if magic != NTieredIndexPacking.MAGIC:
            raise ValueError(
                f"NTieredIndexPacking: bad magic 0x{magic:08X}, "
                f"expected 0x{NTieredIndexPacking.MAGIC:08X}."
            )
        n_indices = int(hdr[1].item())
        n_tiers = int(hdr[2].item())
        header_bytes = 12 + 8 * n_tiers
        if packed.numel() < header_bytes:
            raise ValueError(
                f"NTieredIndexPacking: buffer too short for {n_tiers} "
                f"tiers ({packed.numel()} < {header_bytes})."
            )
        tier_hdr = packed[12:header_bytes].view(torch.int32)
        tier_counts = []
        tier_bits_list = []
        for k in range(n_tiers):
            tier_counts.append(int(tier_hdr[2 * k].item()))
            tier_bits_list.append(int(tier_hdr[2 * k + 1].item()))
        return n_indices, n_tiers, tier_counts, tier_bits_list, header_bytes

    @staticmethod
    def unpack(packed: torch.Tensor) -> torch.Tensor:
        """Unpack an n-tier CBS buffer.

        Args:
            packed: 1-D ``uint8`` buffer produced by :meth:`pack`.

        Returns:
            1-D ``int64`` tensor of centroid indices.

        Raises:
            ValueError: If the buffer has an invalid header.
        """
        (n_indices, n_tiers, tier_counts, tier_bits_list, H) = (
            NTieredIndexPacking._parse_header(packed)
        )
        dev = packed.device
        thresholds = NTieredIndexPacking.tier_thresholds(tier_bits_list)

        # Compute stream sizes and offsets.
        offset = H

        # Read prefix streams.
        prefix_streams_data: list[tuple[int, torch.Tensor]] = []
        n_remaining = n_indices
        for k in range(n_tiers - 1):
            pb = (n_remaining + 7) // 8
            prefix_streams_data.append((n_remaining, packed[offset : offset + pb]))
            offset += pb
            n_remaining -= tier_counts[k]

        # Read tier data streams.
        tier_stream_slices: list[tuple[int, torch.Tensor]] = []
        for k in range(n_tiers):
            bits_k = tier_bits_list[k]
            nb = math.ceil(tier_counts[k] * bits_k / 8)
            tier_stream_slices.append((tier_counts[k], packed[offset : offset + nb]))
            offset += nb

        # Decode cascaded prefix streams to get global positions per tier.
        # remaining_positions tracks global indices still in play.
        remaining_positions = torch.arange(n_indices, dtype=torch.int64, device=dev)
        tier_positions: list[torch.Tensor] = []

        for k in range(n_tiers - 1):
            n_rem, prefix_buf = prefix_streams_data[k]
            prefix_flat = IndexPacking._unpack_bitstream(prefix_buf, 1, n_rem)
            # 0 = tier-k, 1 = rest.
            is_tier_k = prefix_flat == 0
            is_rest = prefix_flat == 1
            tier_positions.append(remaining_positions[is_tier_k])
            remaining_positions = remaining_positions[is_rest]

        # Last tier: whatever remains.
        tier_positions.append(remaining_positions)

        # Decode tier data streams and scatter into result.
        result = torch.zeros(n_indices, dtype=torch.int64, device=dev)
        for k in range(n_tiers):
            n_k, stream_buf = tier_stream_slices[k]
            if n_k == 0:
                continue
            local_indices = IndexPacking.unpack(stream_buf, tier_bits_list[k], n_k)
            # Remap from local to global centroid index.
            lo = 0 if k == 0 else thresholds[k - 1]
            result[tier_positions[k]] = local_indices + lo

        return result

    @staticmethod
    def compute_row_offsets(
        packed: torch.Tensor,
        n_rows: int,
        n_groups: int,
    ) -> list[torch.Tensor]:
        """Compute per-row cumulative tier counts from an n-tier buffer.

        Required by the CUDA kernel to jump directly to row ``n``'s
        slice of each tier's data stream.

        Args:
            packed: 1-D ``uint8`` n-tier buffer.
            n_rows: Number of weight rows (``out_features``).
            n_groups: Number of VQ groups per row.

        Returns:
            List of ``D`` int32 CPU tensors, each of shape
            ``[n_rows + 1]``.  Entry ``i`` of ``offsets[k]`` is the
            cumulative count of tier-k indices in rows ``0 .. i-1``.

        Raises:
            ValueError: If ``n_rows * n_groups`` doesn't match the header.
        """
        (n_indices, n_tiers, tier_counts, _tier_bits_list, H) = (
            NTieredIndexPacking._parse_header(packed)
        )
        if n_indices != n_rows * n_groups:
            raise ValueError(
                f"compute_row_offsets: n_rows*n_groups="
                f"{n_rows * n_groups} != n_indices={n_indices}."
            )
        # Read prefix streams to determine tier membership per index.
        offset = H
        prefix_data: list[tuple[int, torch.Tensor]] = []
        n_remaining = n_indices
        for k in range(n_tiers - 1):
            pb = (n_remaining + 7) // 8
            prefix_data.append((n_remaining, packed[offset : offset + pb].cpu()))
            offset += pb
            n_remaining -= tier_counts[k]

        # Decode tier membership for every index.
        tier_id = torch.full((n_indices,), n_tiers - 1, dtype=torch.int32)
        remaining_mask = torch.ones(n_indices, dtype=torch.bool)
        for k in range(n_tiers - 1):
            n_rem, prefix_buf = prefix_data[k]
            prefix_flat = IndexPacking._unpack_bitstream(prefix_buf, 1, n_rem)
            remaining_indices = remaining_mask.nonzero(as_tuple=True)[0]
            is_tier_k = prefix_flat == 0
            tier_id[remaining_indices[is_tier_k]] = k
            remaining_mask[remaining_indices[is_tier_k]] = False

        # Build cumulative offsets per tier per row.
        offsets = [torch.zeros(n_rows + 1, dtype=torch.int32) for _ in range(n_tiers)]
        for row in range(n_rows):
            start = row * n_groups
            end = start + n_groups
            row_tiers = tier_id[start:end]
            for k in range(n_tiers):
                count_k = int((row_tiers == k).sum().item())
                offsets[k][row + 1] = offsets[k][row] + count_k

        return offsets

    @staticmethod
    def effective_bits(
        tier_counts: list[int],
        tier_bits: list[int],
    ) -> float:
        """Average bits per index under this N-tier cascaded prefix code.

        This is an empirical upper bound on the Shannon entropy ``H(p)``
        of the assignment distribution, computed from the measured tier
        occupancies.  Use it — not ``log2(NC)`` — when estimating bpw::

            bpw = effective_bits(tier_counts, tier_bits) / VL
                  + (NC * VL * 16) / total_params

        The result depends on the actual centroid frequency distribution
        (power-law exponent), not on NC alone.  The same NC with a more
        concentrated distribution (higher Zipf exponent) yields a lower
        value and therefore lower bpw.

        Args:
            tier_counts: Number of indices falling into each tier
                (measured from the actual assignment distribution).
            tier_bits: Flat bit widths per tier, strictly increasing
                (e.g. ``[4, 8, 12]``).

        Returns:
            Expected bits per index, including all cascaded prefix bits.
            Approaches ``log2(NC)`` when the distribution is uniform.
        """
        n_total = sum(tier_counts)
        if n_total == 0:
            return float(tier_bits[-1])
        d = len(tier_bits)

        # Prefix overhead: one 1-bit prefix per remaining index at each
        # level, amortised over all indices.
        prefix_bpi = 0.0
        n_remaining = n_total
        for k in range(d - 1):
            prefix_bpi += n_remaining / n_total  # 1 bit per remaining
            n_remaining -= tier_counts[k]

        # Data: bits_k * p_k for each tier.
        data_bpi = sum(tier_bits[k] * tier_counts[k] / n_total for k in range(d))
        return prefix_bpi + data_bpi

    @staticmethod
    def is_ntier_buffer(packed: torch.Tensor) -> bool:
        """Check if a packed buffer uses the n-tier format.

        Args:
            packed: 1-D ``uint8`` buffer.

        Returns:
            ``True`` if the first 4 bytes match :attr:`MAGIC`.
        """
        if packed.numel() < 4:
            return False
        magic = int(packed[:4].view(torch.int32)[0].item())
        return magic == NTieredIndexPacking.MAGIC


class ZeroEscapePacking:
    """Four-tier flat-prefix packing with a dedicated zero-centroid escape.

    Exploits the observation that centroid-0 (the most frequent after
    frequency-sorting) often accounts for 5-20% of all assignments.
    A flat 2-bit prefix per index encodes:

        ``00``         → centroid-0  (2 bits total, no payload)
        ``01`` + 4-bit → centroids 1-16  (6 bits total)
        ``10`` + 8-bit → centroids 1-256  (10 bits total)
        ``11`` + N-bit → full range  (2 + N bits total)

    Unlike :class:`NTieredIndexPacking` which uses cascaded 1-bit prefixes
    (sequential dependency between levels), the flat 2-bit prefix has no
    inter-level dependency — each index can be decoded independently.

    Wire format (``uint8`` buffer, little-endian header)::

        Header (24 bytes):
          [0:4]    magic       = 0x5A455343  ('ZESC')
          [4:8]    n_indices   (uint32)
          [8:12]   cold_bits   (uint32)  — bit width of tier-3 (full range)
          [12:16]  n_zero      (uint32)  — count of centroid-0 indices
          [16:20]  n_hot       (uint32)  — count of tier-1 (centroids 1-16)
          [20:24]  n_warm      (uint32)  — count of tier-2 (centroids 17-256)

        Streams (concatenated after header):
          prefix   : 2 bits x n_indices, packed LSB-first
          hot      : 4-bit packed (centroids 1-16, remapped to 0-15)
          warm     : 8-bit packed (centroids 17-256, remapped to 0-239)
          cold     : cold_bits-bit packed (centroids 257+, remapped)

    Prerequisites
    -------------
    Centroids **must** be frequency-sorted (index 0 = most-assigned centroid)
    before packing.

    Attributes:
        MAGIC: Header magic number for format detection.
        HEADER_BYTES: Length of the fixed-size binary header (24 bytes).
        TIER_0_MAX: Upper bound for tier-0 (exclusive) — centroid-0 only.
        TIER_1_MAX: Upper bound for tier-1 (exclusive) — centroids 1-16.
        TIER_2_MAX: Upper bound for tier-2 (exclusive) — centroids 17-256.
    """

    MAGIC: int = 0x5A455343
    HEADER_BYTES: int = 24
    TIER_0_MAX: int = 1  # centroid-0 only
    TIER_1_MAX: int = 17  # centroids 0-16 (tier-1 payload is 1-16)
    TIER_2_MAX: int = 257  # centroids 0-256 (tier-2 payload is 17-256)

    @staticmethod
    def pack(
        indices: torch.Tensor,
        cold_bits: int,
    ) -> torch.Tensor:
        """Pack frequency-sorted indices using the zero-escape 2-bit prefix.

        Args:
            indices: 1-D ``int64`` tensor of centroid indices.
            cold_bits: Bit width for the cold tier (tier-3).  Must be large
                enough to encode ``max(indices)`` in the cold range.

        Returns:
            Self-contained ``uint8`` buffer (header + prefix + streams).

        Raises:
            ValueError: If *cold_bits* is outside valid range.
        """
        if not (IndexPacking._MIN_BITS <= cold_bits <= IndexPacking._MAX_BITS):
            raise ValueError(
                f"cold_bits must be in [{IndexPacking._MIN_BITS}, "
                f"{IndexPacking._MAX_BITS}], got {cold_bits}."
            )
        flat = indices.flatten().to(torch.int64)
        dev = flat.device
        n = flat.numel()

        # Classify into tiers.
        is_zero = flat == 0
        is_hot = (flat >= 1) & (flat < ZeroEscapePacking.TIER_1_MAX)
        is_warm = (flat >= ZeroEscapePacking.TIER_1_MAX) & (
            flat < ZeroEscapePacking.TIER_2_MAX
        )
        is_cold = flat >= ZeroEscapePacking.TIER_2_MAX

        n_zero = int(is_zero.sum().item())
        n_hot = int(is_hot.sum().item())
        n_warm = int(is_warm.sum().item())

        # --- prefix stream: 2 bits per index ----------------------------------
        # 00=zero, 01=hot, 10=warm, 11=cold
        prefix = torch.zeros(n, dtype=torch.int64, device=dev)
        prefix[is_hot] = 1
        prefix[is_warm] = 2
        prefix[is_cold] = 3
        prefix_stream = IndexPacking._pack_bitstream(prefix, 2)

        # --- hot stream: 4-bit (centroids 1-16 → remapped to 0-15) -----------
        hot_vals = flat[is_hot] - 1  # remap 1-16 → 0-15
        if n_hot > 0:
            if n_hot % 2 != 0:
                hot_vals = torch.cat(
                    [hot_vals, torch.zeros(1, dtype=torch.int64, device=dev)]
                )
            hot_stream = IndexPacking.pack(hot_vals, 4)
        else:
            hot_stream = torch.zeros(0, dtype=torch.uint8, device=dev)

        # --- warm stream: 8-bit (centroids 17-256 → remapped to 0-239) -------
        warm_vals = flat[is_warm] - ZeroEscapePacking.TIER_1_MAX
        if n_warm > 0:
            warm_stream = IndexPacking.pack(warm_vals, 8)
        else:
            warm_stream = torch.zeros(0, dtype=torch.uint8, device=dev)

        # --- cold stream: cold_bits-bit (centroids 257+ → remapped) -----------
        cold_vals = flat[is_cold] - ZeroEscapePacking.TIER_2_MAX
        n_cold = int(is_cold.sum().item())
        if n_cold > 0:
            cold_stream = IndexPacking.pack(cold_vals, cold_bits)
        else:
            cold_stream = torch.zeros(0, dtype=torch.uint8, device=dev)

        # --- header (24 bytes, 6 x uint32 LE) ---------------------------------
        header = torch.tensor(
            [ZeroEscapePacking.MAGIC, n, cold_bits, n_zero, n_hot, n_warm],
            dtype=torch.int32,
            device=dev,
        ).view(torch.uint8)

        return torch.cat([header, prefix_stream, hot_stream, warm_stream, cold_stream])

    @staticmethod
    def _parse_header(
        packed: torch.Tensor,
    ) -> tuple[int, int, int, int, int, int]:
        """Parse the zero-escape buffer header.

        Args:
            packed: 1-D ``uint8`` buffer.

        Returns:
            ``(n_indices, cold_bits, n_zero, n_hot, n_warm, header_bytes)``

        Raises:
            ValueError: If the buffer is too short or the magic doesn't match.
        """
        H = ZeroEscapePacking.HEADER_BYTES
        if packed.numel() < H:
            raise ValueError(
                f"ZeroEscapePacking: buffer too short "
                f"({packed.numel()} bytes, need at least {H})."
            )
        hdr = packed[:H].view(torch.int32)
        magic = int(hdr[0].item())
        if magic != ZeroEscapePacking.MAGIC:
            raise ValueError(
                f"ZeroEscapePacking: bad magic 0x{magic:08X}, "
                f"expected 0x{ZeroEscapePacking.MAGIC:08X}."
            )
        n = int(hdr[1].item())
        cold_bits = int(hdr[2].item())
        n_zero = int(hdr[3].item())
        n_hot = int(hdr[4].item())
        n_warm = int(hdr[5].item())
        return n, cold_bits, n_zero, n_hot, n_warm, H

    @staticmethod
    def unpack(packed: torch.Tensor) -> torch.Tensor:
        """Unpack a zero-escape buffer produced by :meth:`pack`.

        Args:
            packed: 1-D ``uint8`` buffer (header + streams).

        Returns:
            1-D ``int64`` tensor of centroid indices.

        Raises:
            ValueError: If the buffer has an invalid header.
        """
        n, cold_bits, n_zero, n_hot, n_warm, H = ZeroEscapePacking._parse_header(packed)
        dev = packed.device
        n_cold = n - n_zero - n_hot - n_warm

        offset = H

        # --- decode prefix stream (2 bits per index) --------------------------
        prefix_bytes = (n * 2 + 7) // 8
        prefix = IndexPacking._unpack_bitstream(
            packed[offset : offset + prefix_bytes], 2, n
        )
        offset += prefix_bytes

        hot_pos = (prefix == 1).nonzero(as_tuple=True)[0]
        warm_pos = (prefix == 2).nonzero(as_tuple=True)[0]
        cold_pos = (prefix == 3).nonzero(as_tuple=True)[0]

        # --- decode hot stream (4-bit, remapped 0-15 → 1-16) -----------------
        hot_bytes = math.ceil(n_hot * 4 / 8)
        if n_hot > 0:
            hot_vals = (
                IndexPacking.unpack(packed[offset : offset + hot_bytes], 4, n_hot) + 1
            )
        else:
            hot_vals = torch.zeros(0, dtype=torch.int64, device=dev)
        offset += hot_bytes

        # --- decode warm stream (8-bit, remapped 0-239 → 17-256) -------------
        if n_warm > 0:
            warm_vals = (
                IndexPacking.unpack(packed[offset : offset + n_warm], 8, n_warm)
                + ZeroEscapePacking.TIER_1_MAX
            )
        else:
            warm_vals = torch.zeros(0, dtype=torch.int64, device=dev)
        offset += n_warm  # 8-bit = 1 byte per index

        # --- decode cold stream (cold_bits-bit, remapped → 257+) -------------
        cold_bytes = math.ceil(n_cold * cold_bits / 8)
        if n_cold > 0:
            cold_vals = (
                IndexPacking.unpack(
                    packed[offset : offset + cold_bytes], cold_bits, n_cold
                )
                + ZeroEscapePacking.TIER_2_MAX
            )
        else:
            cold_vals = torch.zeros(0, dtype=torch.int64, device=dev)

        # --- scatter back into original positions -----------------------------
        result = torch.zeros(n, dtype=torch.int64, device=dev)
        # zero positions: centroid-0, already 0 in result
        if n_hot > 0:
            result[hot_pos] = hot_vals
        if n_warm > 0:
            result[warm_pos] = warm_vals
        if n_cold > 0:
            result[cold_pos] = cold_vals
        return result

    @staticmethod
    def compute_row_offsets(
        packed: torch.Tensor,
        n_rows: int,
        n_groups: int,
    ) -> list[torch.Tensor]:
        """Compute per-row cumulative tier counts from a zero-escape buffer.

        Required by CUDA kernels to jump directly to row ``n``'s slice
        of each tier's data stream.

        Args:
            packed: 1-D ``uint8`` zero-escape buffer.
            n_rows: Number of weight rows (``out_features``).
            n_groups: Number of VQ groups per row.

        Returns:
            List of 4 int32 CPU tensors (zero, hot, warm, cold), each of
            shape ``[n_rows + 1]``.  Entry ``i`` of ``offsets[k]`` is the
            cumulative count of tier-k indices in rows ``0 .. i-1``.

        Raises:
            ValueError: If ``n_rows * n_groups`` doesn't match the header.
        """
        n, _cold_bits, _n_zero, _n_hot, _n_warm, H = ZeroEscapePacking._parse_header(
            packed
        )
        if n != n_rows * n_groups:
            raise ValueError(
                f"compute_row_offsets: n_rows*n_groups="
                f"{n_rows * n_groups} != n_indices={n}."
            )

        # Read prefix stream.
        prefix_bytes = (n * 2 + 7) // 8
        prefix = IndexPacking._unpack_bitstream(
            packed[H : H + prefix_bytes].cpu(), 2, n
        )

        offsets = [torch.zeros(n_rows + 1, dtype=torch.int32) for _ in range(4)]
        for row in range(n_rows):
            start = row * n_groups
            end = start + n_groups
            row_prefix = prefix[start:end]
            for k in range(4):
                count_k = int((row_prefix == k).sum().item())
                offsets[k][row + 1] = offsets[k][row] + count_k

        return offsets

    @staticmethod
    def effective_bits(
        n_zero: int,
        n_hot: int,
        n_warm: int,
        n_cold: int,
        cold_bits: int,
    ) -> float:
        """Average bits per index under the zero-escape flat-2-bit prefix code.

        This is an empirical upper bound on the Shannon entropy ``H(p)``
        of the assignment distribution, computed from the measured tier
        occupancies.  Use it — not ``log2(NC)`` — when estimating bpw::

            bpw = effective_bits(n_zero, n_hot, n_warm, n_cold, cold_bits) / VL
                  + (NC * VL * 16) / total_params

        The result depends on the actual centroid frequency distribution
        (power-law shape), not on NC alone: a more concentrated distribution
        (more zero/hot hits) lowers the returned value and therefore bpw.

        Args:
            n_zero: Count of centroid-0 indices (2 bits total: prefix only).
            n_hot: Count of tier-1 indices, centroids 1–16 (6 bits total).
            n_warm: Count of tier-2 indices, centroids 17–256 (10 bits total).
            n_cold: Count of tier-3 indices, centroids 257+ (2+cold_bits total).
            cold_bits: Flat bit width for the cold tier
                (= ``ceil(log2(NC - 257))``).

        Returns:
            Expected bits per index, including the 2-bit prefix for every index.
            Approaches ``2 + cold_bits`` when almost all indices are cold.
        """
        n_total = n_zero + n_hot + n_warm + n_cold
        if n_total == 0:
            return float(cold_bits)
        # Every index pays 2 bits for the prefix.
        # Tier payloads: 0 (zero), 4 (hot), 8 (warm), cold_bits (cold).
        total_bits = (
            n_zero * 2 + n_hot * (2 + 4) + n_warm * (2 + 8) + n_cold * (2 + cold_bits)
        )
        return total_bits / n_total

    @staticmethod
    def is_zero_escape_buffer(packed: torch.Tensor) -> bool:
        """Check if a packed buffer uses the zero-escape format.

        Args:
            packed: 1-D ``uint8`` buffer.

        Returns:
            ``True`` if the first 4 bytes match :attr:`MAGIC`.
        """
        if packed.numel() < 4:
            return False
        magic = int(packed[:4].view(torch.int32)[0].item())
        return magic == ZeroEscapePacking.MAGIC
