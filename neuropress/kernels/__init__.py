"""JIT-build and expose neuropress CUDA extensions.

Three extensions are available, compiled on first use via
:func:`torch.utils.cpp_extension.load`.

Available functions:

    :func:`q_linear` — generic fused group-quantized dequant + linear.
        Supports any bit-width in ``{1, 2, 4, 8}`` and both symmetric
        (scale-only) and asymmetric (scale + zero_point) schemes.

    :func:`vq_linear` — fused VQ-weight linear (``vq_linear.cu``).
        Unpacks 4, 8, or 12-bit codebook indices, gathers centroids, and
        accumulates — without materialising the float weight matrix.

    :func:`dequantize_vq_weight` — materialise a fp16 weight matrix from
        packed VQ indices and centroids (``vq_linear.cu``).  Handles any
        bit width in ``[4, 20]`` via a generic bitstream unpacker.

    :func:`vq_linear_tiered` — fused VQ-weight linear with two-tier prefix
        decoding (``vq_linear_tiered.cu``).  Uses warp-ballot to decode the
        1-bit hot/cold prefix stream in parallel.  Requires pre-split streams
        and precomputed per-row offsets (see
        :meth:`~neuropress.transforms.vq_packing.TieredIndexPacking.compute_row_offsets`).

    :func:`vq_linear_ntier` — fused VQ-weight linear with N-tier cascaded
        binary splitting (``vq_linear_ntier.cu``).  Generalises
        ``vq_linear_tiered`` to 2–4 tiers.  Uses cascaded ``__ballot_sync``
        to decode tier membership and per-tier ``__popc`` for stream positions.

Importing this module succeeds even without CUDA; errors are raised only
when a kernel function is first called.

Example::

    from neuropress.kernels import q_linear, vq_linear

    # Q4_0 symmetric (group_size=32)
    y = q_linear(x_fp16, w_packed, scale, nbits=4, group_size=32)

    # VQ linear (256 centroids, vector_len=8, 8-bit indices)
    y = vq_linear(x_fp16, packed_indices, centroids, bits=8, vector_len=8)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass  # reserved for type stubs

log = logging.getLogger(__name__)

_ext: object = None  # cached compiled extension
_build_failed: bool = False  # avoid redundant retry after failure


def _load_extension() -> object:
    """JIT-build (once) and return the ``neuropress_kernels`` extension.

    Returns:
        The compiled extension module with a ``q_linear`` function.

    Raises:
        RuntimeError: If CUDA is unavailable or the build fails.
    """
    global _ext, _build_failed
    if _ext is not None:
        return _ext
    if _build_failed:
        raise RuntimeError(
            "neuropress_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda' backend requires a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "q4_linear.cu"
    try:
        log.info("JIT-building neuropress_kernels from %s …", src)
        _ext = load(
            name="neuropress_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_kernels built successfully.")
    except Exception as exc:
        _build_failed = True
        raise RuntimeError(f"Failed to JIT-build neuropress_kernels: {exc}") from exc
    return _ext


def q_linear(
    x: torch.Tensor,
    w_pack: torch.Tensor,
    scale: torch.Tensor,
    nbits: int,
    group_size: int,
    zero_point: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generic fused group-quantized dequant + linear (CUDA kernel).

    Unpacks *nbits*-bit weights packed by :class:`~neuropress.t2n.packing.BitPacking`,
    applies per-group dequantization, and accumulates — all in one kernel pass
    without materialising the full float weight matrix.

    Dequantization formula:

        symmetric (``zero_point=None``):
            ``w = (q - 2**(nbits-1)) * scale``

        asymmetric (``zero_point`` provided):
            ``w = (q - zero_point) * scale``

    Args:
        x: Input tensor ``[..., K]``, float16 or float32, on CUDA.
        w_pack: Packed uint8 weights ``[N, K // (8 // nbits)]``, on CUDA.
        scale: Per-group float16 scales, length ``N * (K // group_size)``, on CUDA.
        nbits: Bits per value — one of ``{1, 2, 4, 8}``.
        group_size: Quantization group size; must divide K.
        zero_point: Optional per-group float16 zero-points (asymmetric scheme),
            same length as *scale*, on CUDA.
        bias: Optional float bias ``[N]`` (any float dtype).

    Returns:
        Output tensor ``[..., N]`` in the same dtype as *x*.

    Raises:
        RuntimeError: If CUDA is unavailable or the extension build fails.
        RuntimeError: If *nbits* is not in ``{1, 2, 4, 8}``.
    """
    return _load_extension().q_linear(
        x, w_pack, scale, nbits, group_size, zero_point, bias
    )


# ---------------------------------------------------------------------------
# VQ linear extension (vq_linear.cu)
# ---------------------------------------------------------------------------

_vq_ext: object = None
_vq_build_failed: bool = False


def _load_vq_extension() -> object:
    """JIT-build (once) and return the ``neuropress_vq_kernels`` extension.

    Returns:
        The compiled extension module with a ``vq_linear`` function.

    Raises:
        RuntimeError: If CUDA is unavailable or the build fails.
    """
    global _vq_ext, _vq_build_failed
    if _vq_ext is not None:
        return _vq_ext
    if _vq_build_failed:
        raise RuntimeError(
            "neuropress_vq_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda' backend requires a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "vq_linear.cu"
    try:
        log.info("JIT-building neuropress_vq_kernels from %s …", src)
        _vq_ext = load(
            name="neuropress_vq_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_vq_kernels built successfully.")
    except Exception as exc:
        _vq_build_failed = True
        raise RuntimeError(f"Failed to JIT-build neuropress_vq_kernels: {exc}") from exc
    return _vq_ext


def vq_linear(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused VQ-weight linear: index unpack + centroid gather + matmul.

    Unpacks *bits*-bit codebook indices from *packed_indices*, gathers the
    corresponding centroid vectors from *centroids*, and accumulates the
    result into the output — without materialising the full float weight
    matrix.

    Index layout matches :class:`~neuropress.transforms.vq_packing.IndexPacking`:
    8-bit → one index per byte; 4-bit → two indices per byte, LSB-first.

    Args:
        x: Input tensor ``[M, K]``, float16 or float32, on CUDA.
        packed_indices: Packed 1-D uint8 index buffer, on CUDA.
            Length = ``N * n_groups`` for ``bits=8``, or
            ``ceil(N * n_groups / 2)`` for ``bits=4``.
        centroids: Codebook ``[num_centroids, vector_len]`` float16, on CUDA.
        bits: Bits per index — 4, 8, or 12.
        vector_len: Number of weight elements per VQ vector.
        bias: Optional float bias ``[N]`` (any float dtype).

    Returns:
        Output tensor ``[M, N]`` in the same dtype as *x*.

    Raises:
        RuntimeError: If CUDA is unavailable or the extension build fails.
    """
    return _load_vq_extension().vq_linear(
        x, packed_indices, centroids, bits, vector_len, bias
    )


def dequantize_vq_weight(
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    vector_len: int,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Materialise a fp16 weight matrix from packed VQ indices and centroids.

    Unpacks every index from *packed_indices* and gathers the corresponding
    centroid vector, writing the result to ``W[out_features, in_features]``.
    Handles any bit width in ``[4, 20]`` via a generic bitstream unpacker.

    *packed_indices* must have at least 4 bytes of zero-padding past the last
    valid byte (safe for the 4-byte window read used internally).

    Args:
        packed_indices: Packed 1-D uint8 index buffer, on CUDA.
        centroids: Codebook ``[num_centroids, vector_len]`` float16, on CUDA.
        bits: Bits per index — any value in ``[4, 20]``.
        vector_len: Number of weight elements per VQ vector (4 or 8).
        out_features: Number of output features (rows of W).
        in_features: Number of input features (columns of W).

    Returns:
        ``[out_features, in_features]`` float16 weight matrix on CUDA.

    Raises:
        RuntimeError: If CUDA is unavailable or the extension build fails.
    """
    return _load_vq_extension().dequantize_vq_weight(
        packed_indices, centroids, bits, vector_len, out_features, in_features
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# VQ linear WMMA extension (vq_linear_wmma.cu)
# ---------------------------------------------------------------------------

_vq_wmma_ext: object = None
_vq_wmma_build_failed: bool = False


def _load_wmma_extension() -> object:
    global _vq_wmma_ext, _vq_wmma_build_failed
    if _vq_wmma_ext is not None:
        return _vq_wmma_ext
    if _vq_wmma_build_failed:
        raise RuntimeError(
            "neuropress_vq_wmma_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda_wmma' backend requires a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "vq_linear_wmma.cu"
    try:
        log.info("JIT-building neuropress_vq_wmma_kernels from %s …", src)
        _vq_wmma_ext = load(
            name="neuropress_vq_wmma_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
                "-arch=native",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_vq_wmma_kernels built successfully.")
    except Exception as exc:
        _vq_wmma_build_failed = True
        raise RuntimeError(
            f"Failed to JIT-build neuropress_vq_wmma_kernels: {exc}"
        ) from exc
    return _vq_wmma_ext


def vq_linear_wmma(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused VQ linear with GEMV/GEMM auto-dispatch (WMMA tensor cores).

    Selects GEMV (smem codebook + warp reduction, M ≤ 8) or GEMM (wmma
    16×16×16 tiles, M > 8) at runtime based on the batch dimension.
    Supports bits ∈ {4, 8, 10, 12, 14, 15, 16}.  NC > 4096 (bits > 12) uses
    L2-cached global memory for the codebook instead of shared memory.
    """
    return _load_wmma_extension().vq_linear_wmma(
        x, packed_indices, centroids, bits, bias
    )


# ---------------------------------------------------------------------------
# VQ LUT gather extension (vq_linear_lut.cu)
# ---------------------------------------------------------------------------

_vq_lut_ext: object = None
_vq_lut_build_failed: bool = False


def _load_lut_extension() -> object:
    """JIT-build (once) and return the ``neuropress_vq_lut_kernels`` extension."""
    global _vq_lut_ext, _vq_lut_build_failed
    if _vq_lut_ext is not None:
        return _vq_lut_ext
    if _vq_lut_build_failed:
        raise RuntimeError(
            "neuropress_vq_lut_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda_lut' backend requires a CUDA device."
        )
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "vq_linear_lut.cu"
    try:
        log.info("JIT-building neuropress_vq_lut_kernels from %s …", src)
        _vq_lut_ext = load(
            name="neuropress_vq_lut_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
                "-arch=native",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_vq_lut_kernels built successfully.")
    except Exception as exc:
        _vq_lut_build_failed = True
        raise RuntimeError(
            f"Failed to JIT-build neuropress_vq_lut_kernels: {exc}"
        ) from exc
    return _vq_lut_ext


def vq_lut_gather(
    lut: torch.Tensor,
    packed_indices: torch.Tensor,
    bits: int,
    out_features: int,
) -> torch.Tensor:
    """LUT gather phase: y[m, j] = sum_g lut[m, g, idx[j, g]].

    Args:
        lut: Precomputed ``[M, n_groups, NC]`` float32 CUDA tensor.
            Built by the caller as ``(x.reshape(M*n_groups, VL) @ centroids.T)
            .reshape(M, n_groups, NC)``.
        packed_indices: 1-D uint8 packed index buffer on CUDA.
            Flat layout: index for output neuron ``j``, group ``g`` is at
            position ``j * n_groups + g`` in the bit-stream.
        bits: Bits per index — 4, 8, 10, 12, 14, 15, or 16.
        out_features: Output dimension N (number of output neurons).

    Returns:
        ``[M, N]`` float32 CUDA tensor.
    """
    return _load_lut_extension().vq_lut_gather(lut, packed_indices, bits, out_features)


# ---------------------------------------------------------------------------
# VQ linear tiered extension (vq_linear_tiered.cu)
# ---------------------------------------------------------------------------

_vq_tiered_ext: object = None
_vq_tiered_build_failed: bool = False


def _load_tiered_extension() -> object:
    """JIT-build (once) and return the ``neuropress_vq_tiered_kernels`` extension.

    Returns:
        The compiled extension module with a ``vq_linear_tiered`` function.

    Raises:
        RuntimeError: If CUDA is unavailable or the build fails.
    """
    global _vq_tiered_ext, _vq_tiered_build_failed
    if _vq_tiered_ext is not None:
        return _vq_tiered_ext
    if _vq_tiered_build_failed:
        raise RuntimeError(
            "neuropress_vq_tiered_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda_tiered' backend requires a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "vq_linear_tiered.cu"
    try:
        log.info("JIT-building neuropress_vq_tiered_kernels from %s …", src)
        _vq_tiered_ext = load(
            name="neuropress_vq_tiered_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_vq_tiered_kernels built successfully.")
    except Exception as exc:
        _vq_tiered_build_failed = True
        raise RuntimeError(
            f"Failed to JIT-build neuropress_vq_tiered_kernels: {exc}"
        ) from exc
    return _vq_tiered_ext


def vq_linear_tiered(
    x: torch.Tensor,
    prefix_stream: torch.Tensor,
    hot_stream: torch.Tensor,
    cold_stream: torch.Tensor,
    hot_row_off: torch.Tensor,
    cold_row_off: torch.Tensor,
    centroids: torch.Tensor,
    cold_bits: int,
    hot_bits: int,
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused VQ tiered-index linear: warp-ballot decode + centroid gather + matmul.

    Decodes a two-tier prefix-coded index buffer using ``__ballot_sync`` to
    identify hot/cold lanes within each warp, then gathers centroid vectors
    and accumulates — without materialising the float weight matrix.

    The caller is responsible for splitting the monolithic tiered buffer
    (produced by
    :meth:`~neuropress.transforms.vq_packing.TieredIndexPacking.pack`) into
    its three constituent streams and supplying precomputed per-row offsets
    (from
    :meth:`~neuropress.transforms.vq_packing.TieredIndexPacking.compute_row_offsets`).

    Args:
        x: Input tensor ``[M, K]``, float16 or float32, on CUDA.
        prefix_stream: 1-bit-per-index prefix flags, ``uint8``, on CUDA.
            Length = ``ceil(N * n_groups / 8)``.
        hot_stream: ``hot_bits``-bit-packed hot indices, ``uint8``, on CUDA.
            Length = ``ceil(n_hot_total * hot_bits / 8)``.
        cold_stream: ``cold_bits``-bit-packed cold indices, ``uint8``, on CUDA.
            Length = ``ceil(n_cold_total * cold_bits / 8)``.
        hot_row_off: Cumulative hot-index counts per row, ``int32``, on CUDA.
            Length = ``N + 1`` (inclusive scan prefix).
        cold_row_off: Cumulative cold-index counts per row, ``int32``, on CUDA.
            Length = ``N + 1``.
        centroids: Codebook ``[num_centroids, vector_len]`` float16, on CUDA.
        cold_bits: Bit width of cold-tier indices (4–20).
        hot_bits: Bit width of hot-tier indices (4 or 8).
        vector_len: Number of weight elements per VQ vector.
        bias: Optional float bias ``[N]`` (any float dtype).

    Returns:
        Output tensor ``[M, N]`` in the same dtype as *x*.

    Raises:
        RuntimeError: If CUDA is unavailable or the extension build fails.
    """
    return _load_tiered_extension().vq_linear_tiered(
        x,
        prefix_stream,
        hot_stream,
        cold_stream,
        hot_row_off,
        cold_row_off,
        centroids,
        cold_bits,
        hot_bits,
        vector_len,
        bias,
    )


# ---------------------------------------------------------------------------
# VQ linear n-tier extension (vq_linear_ntier.cu)
# ---------------------------------------------------------------------------

_vq_ntier_ext: object = None
_vq_ntier_build_failed: bool = False


def _load_ntier_extension() -> object:
    """JIT-build (once) and return the ``neuropress_vq_ntier_kernels`` extension.

    Returns:
        The compiled extension module with a ``vq_linear_ntier`` function.

    Raises:
        RuntimeError: If CUDA is unavailable or the build fails.
    """
    global _vq_ntier_ext, _vq_ntier_build_failed
    if _vq_ntier_ext is not None:
        return _vq_ntier_ext
    if _vq_ntier_build_failed:
        raise RuntimeError(
            "neuropress_vq_ntier_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda_ntier' backend requires a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "vq_linear_ntier.cu"
    try:
        log.info("JIT-building neuropress_vq_ntier_kernels from %s …", src)
        _vq_ntier_ext = load(
            name="neuropress_vq_ntier_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_vq_ntier_kernels built successfully.")
    except Exception as exc:
        _vq_ntier_build_failed = True
        raise RuntimeError(
            f"Failed to JIT-build neuropress_vq_ntier_kernels: {exc}"
        ) from exc
    return _vq_ntier_ext


def vq_linear_ntier(
    x: torch.Tensor,
    prefix_streams: list[torch.Tensor],
    tier_streams: list[torch.Tensor],
    row_offsets: list[torch.Tensor],
    centroids: torch.Tensor,
    tier_bits: list[int],
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused VQ n-tier CBS linear: cascaded ballot decode + centroid gather + matmul.

    Decodes D cascaded prefix streams via ``__ballot_sync`` to determine each
    index's tier, reads the index from the appropriate tier data stream, gathers
    centroids, and accumulates — without materialising the float weight matrix.

    Args:
        x: Input tensor ``[M, K]``, float16 or float32, on CUDA.
        prefix_streams: ``D-1`` uint8 prefix bitstreams on CUDA, one per
            cascade level.
        tier_streams: ``D`` uint8 packed tier data streams on CUDA.
        row_offsets: ``D`` int32 tensors, each ``[N+1]``, cumulative per-tier
            counts per row.
        centroids: Codebook ``[num_centroids, vector_len]`` float16, on CUDA.
        tier_bits: Bit widths per tier (length ``D``).
        vector_len: Number of weight elements per VQ vector.
        bias: Optional float bias ``[N]``.

    Returns:
        Output tensor ``[M, N]`` in the same dtype as *x*.

    Raises:
        RuntimeError: If CUDA is unavailable or the extension build fails.
    """
    return _load_ntier_extension().vq_linear_ntier(
        x,
        prefix_streams,
        tier_streams,
        row_offsets,
        centroids,
        tier_bits,
        vector_len,
        bias,
    )


# ---------------------------------------------------------------------------
# VQ linear zero-escape extension (vq_linear_zero_escape.cu)
# ---------------------------------------------------------------------------

_vq_zesc_ext: object = None
_vq_zesc_build_failed: bool = False


def _load_zesc_extension() -> object:
    """JIT-build (once) and return the ``neuropress_vq_zesc_kernels`` extension.

    Returns:
        The compiled extension module with a ``vq_linear_zero_escape`` function.

    Raises:
        RuntimeError: If CUDA is unavailable or the build fails.
    """
    global _vq_zesc_ext, _vq_zesc_build_failed
    if _vq_zesc_ext is not None:
        return _vq_zesc_ext
    if _vq_zesc_build_failed:
        raise RuntimeError(
            "neuropress_vq_zesc_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda_zero_escape' backend requires "
            "a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "vq_linear_zero_escape.cu"
    try:
        log.info("JIT-building neuropress_vq_zesc_kernels from %s …", src)
        _vq_zesc_ext = load(
            name="neuropress_vq_zesc_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_vq_zesc_kernels built successfully.")
    except Exception as exc:
        _vq_zesc_build_failed = True
        raise RuntimeError(
            f"Failed to JIT-build neuropress_vq_zesc_kernels: {exc}"
        ) from exc
    return _vq_zesc_ext


# ---------------------------------------------------------------------------
# K-means assign extension (kmeans_assign.cu)
# ---------------------------------------------------------------------------

_kmeans_ext: object = None
_kmeans_build_failed: bool = False


def _load_kmeans_extension() -> object:
    """JIT-build (once) and return the ``neuropress_kmeans_kernels`` extension.

    Returns:
        The compiled extension module with a ``kmeans_assign`` function.

    Raises:
        RuntimeError: If CUDA is unavailable or the build fails.
    """
    global _kmeans_ext, _kmeans_build_failed
    if _kmeans_ext is not None:
        return _kmeans_ext
    if _kmeans_build_failed:
        raise RuntimeError(
            "neuropress_kmeans_kernels failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The 'cuda' backend requires a CUDA device."
        )

    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "kmeans_assign.cu"
    try:
        log.info("JIT-building neuropress_kmeans_kernels from %s …", src)
        _kmeans_ext = load(
            name="neuropress_kmeans_kernels",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=True,
        )
        log.info("neuropress_kmeans_kernels built successfully.")
    except Exception as exc:
        _kmeans_build_failed = True
        raise RuntimeError(
            f"Failed to JIT-build neuropress_kmeans_kernels: {exc}"
        ) from exc
    return _kmeans_ext


def kmeans_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused k-means E-step: returns nearest-centroid labels ``[N]`` (int64).

    Args:
        x: ``[N, D]`` float32 tensor on CUDA.
        centroids: ``[k, D]`` float32 tensor on CUDA.
        vector_len: D — must be 4.
        bias: Optional ``[k]`` float32 tensor on CUDA.  When provided, the
            effective distance is ``||x - c||^2 - bias[c]`` before argmin,
            implementing a concentration bonus (e.g. ``beta * freq[c]``).

    Returns:
        ``[N]`` int64 label tensor on the same CUDA device.
    """
    return _load_kmeans_extension().kmeans_assign(x, centroids, vector_len, bias)


def kmeans_assign_grouped(
    x: torch.Tensor,
    C: torch.Tensor,
    imp: torch.Tensor,
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused grouped k-means E-step: all G groups in a single kernel launch.

    Handles all G groups simultaneously using ``blockIdx.y`` as the group
    dimension, eliminating the Python loop and G sequential kernel launches
    of the per-group approach.

    Args:
        x: ``[R, G, D]`` float32 tensor on CUDA (contiguous).
        C: ``[NC, D]`` float32 centroid tensor on CUDA (contiguous).
        imp: ``[G, D]`` float32 sqrt-importance tensor on CUDA (contiguous).
            Applied as a scale to both x and C before distance computation.
        vector_len: D — must be 4.
        bias: Optional ``[NC]`` float32 tensor on CUDA.  When provided, the
            effective distance is ``||x' - c'||^2 - bias[c]`` before argmin.

    Returns:
        ``[R, G]`` int64 label tensor on the same CUDA device.
    """
    return _load_kmeans_extension().kmeans_assign_grouped(x, C, imp, vector_len, bias)


def pack_indices(
    indices: torch.Tensor,
    n_indices: int,
    bits: int,
) -> torch.Tensor:
    """Pack an int64 CUDA tensor into a LSB-first N-bit bit-stream (single kernel launch).

    Replaces the Python ``for b in range(bits)`` loop in
    :func:`~neuropress.transforms.vq_packing.IndexPacking._pack_bitstream`
    for the GPU path.  Reduces ``bits`` sequential CUDA kernel launches to one
    via 32-bit ``atomicOr`` on byte-aligned windows.

    Args:
        indices: ``[n_indices]`` int64 CUDA tensor with values in ``[0, 2**bits)``.
        n_indices: Number of indices to pack.
        bits: Bits per index - must be in ``[4, 16]``.

    Returns:
        ``[ceil(n_indices * bits / 8)]`` uint8 tensor on the same CUDA device.
    """
    return _load_kmeans_extension().pack_indices(
        indices[:n_indices].contiguous(), n_indices, bits
    )


def unpack_indices(
    packed: torch.Tensor,
    n_indices: int,
    bits: int,
) -> torch.Tensor:
    """Unpack a LSB-first N-bit index bit-stream on CUDA (single kernel launch).

    Replaces the Python ``for b in range(bits)`` loop in
    :func:`~neuropress.transforms.vq_packing.IndexPacking._unpack_bitstream`
    for the GPU path.  Reduces ``bits`` sequential CUDA kernel launches to one.

    Args:
        packed: 1-D ``uint8`` CUDA tensor produced by
            :meth:`~neuropress.transforms.vq_packing.IndexPacking.pack`.
            Must have at least 3 zero-padded bytes appended past the last
            valid byte (the wrapper handles this automatically).
        n_indices: Number of indices to unpack.
        bits: Bits per index - must be in ``[4, 16]``.

    Returns:
        ``[n_indices]`` int64 label tensor on the same CUDA device.
    """
    # Append 3 zero-bytes so the 3-byte window read is always in-bounds.
    padded = torch.cat([packed, packed.new_zeros(3)])
    return _load_kmeans_extension().unpack_indices(padded, n_indices, bits)


def vq_linear_zero_escape(
    x: torch.Tensor,
    prefix_stream: torch.Tensor,
    hot_stream: torch.Tensor,
    warm_stream: torch.Tensor,
    cold_stream: torch.Tensor,
    row_off_zero: torch.Tensor,
    row_off_hot: torch.Tensor,
    row_off_warm: torch.Tensor,
    row_off_cold: torch.Tensor,
    centroids: torch.Tensor,
    cold_bits: int,
    vector_len: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused VQ zero-escape linear: flat 2-bit prefix decode + centroid gather + matmul.

    Decodes a flat 2-bit prefix per index to determine tier membership
    (zero/hot/warm/cold), reads the index from the appropriate tier data
    stream using ``__ballot_sync`` + ``__popc``, gathers centroids, and
    accumulates — without materialising the float weight matrix.

    The caller is responsible for splitting the monolithic zero-escape buffer
    (produced by
    :meth:`~neuropress.transforms.vq_packing.ZeroEscapePacking.pack`) into
    its constituent streams and supplying precomputed per-row offsets (from
    :meth:`~neuropress.transforms.vq_packing.ZeroEscapePacking.compute_row_offsets`).

    Args:
        x: Input tensor ``[M, K]``, float16 or float32, on CUDA.
        prefix_stream: 2-bit-per-index prefix flags, ``uint8``, on CUDA.
        hot_stream: 4-bit-packed hot indices, ``uint8``, on CUDA.
        warm_stream: 8-bit warm indices, ``uint8``, on CUDA.
        cold_stream: ``cold_bits``-bit-packed cold indices, ``uint8``, on CUDA.
        row_off_zero: Cumulative zero-index counts per row, ``int32``, on CUDA.
            Length = ``N + 1``.
        row_off_hot: Cumulative hot-index counts per row, ``int32``, on CUDA.
            Length = ``N + 1``.
        row_off_warm: Cumulative warm-index counts per row, ``int32``, on CUDA.
            Length = ``N + 1``.
        row_off_cold: Cumulative cold-index counts per row, ``int32``, on CUDA.
            Length = ``N + 1``.
        centroids: Codebook ``[num_centroids, vector_len]`` float16, on CUDA.
        cold_bits: Bit width of cold-tier indices.
        vector_len: Number of weight elements per VQ vector.
        bias: Optional float bias ``[N]`` (any float dtype).

    Returns:
        Output tensor ``[M, N]`` in the same dtype as *x*.

    Raises:
        RuntimeError: If CUDA is unavailable or the extension build fails.
    """
    return _load_zesc_extension().vq_linear_zero_escape(
        x,
        prefix_stream,
        hot_stream,
        warm_stream,
        cold_stream,
        row_off_zero,
        row_off_hot,
        row_off_warm,
        row_off_cold,
        centroids,
        cold_bits,
        vector_len,
        bias,
    )


# ---------------------------------------------------------------------------
# Fused RMSNorm extension (rms_norm.cu)
# ---------------------------------------------------------------------------

_rms_norm_ext: object = None
_rms_norm_build_failed: bool = False


def _load_rms_norm_extension() -> object:
    """JIT-build (once) and return the ``neuropress_rms_norm`` extension."""
    global _rms_norm_ext, _rms_norm_build_failed
    if _rms_norm_ext is not None:
        return _rms_norm_ext
    if _rms_norm_build_failed:
        raise RuntimeError(
            "neuropress_rms_norm failed to build on a previous attempt. "
            "Ensure CUDA is available and the CUDA toolkit is installed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available.  The fused RMSNorm backend requires a CUDA device."
        )
    from torch.utils.cpp_extension import load

    src = Path(__file__).parent / "rms_norm.cu"
    try:
        log.info("JIT-building neuropress_rms_norm from %s …", src)
        _rms_norm_ext = load(
            name="neuropress_rms_norm",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++20",
                "-Xcompiler=-fpermissive",
                "-Xcompiler=-Wno-missing-template-keyword",
            ],
            extra_cflags=["-O3", "-std=c++20", "-fpermissive"],
            verbose=False,
        )
        log.info("neuropress_rms_norm built successfully.")
    except Exception as exc:
        _rms_norm_build_failed = True
        raise RuntimeError(f"Failed to JIT-build neuropress_rms_norm: {exc}") from exc
    return _rms_norm_ext


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused single-pass RMSNorm (CUDA kernel).

    Computes ``out[m, i] = x[m, i] * rsqrt(mean(x[m,:]²) + eps) * weight[i]``
    in a single kernel pass — replacing HuggingFace's unfused 5-op sequence.

    Args:
        x: Input ``[M, H]`` float16 or bfloat16 CUDA tensor (must be contiguous).
        weight: RMSNorm scale ``[H]`` same dtype as *x*, CUDA.
        eps: Epsilon added to mean-square before rsqrt.

    Returns:
        Normalised ``[M, H]`` tensor, same dtype and device as *x*.
    """
    return _load_rms_norm_extension().rms_norm(x.contiguous(), weight.contiguous(), eps)
