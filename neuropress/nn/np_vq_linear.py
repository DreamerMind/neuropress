"""NPVQLinear: drop-in replacement for ``nn.Linear`` with VQ-compressed weights.

Weight is stored as packed codebook indices plus a shared centroid table held
by a companion :class:`VQProxy` module.  At inference, the weight matrix is
reconstructed on the fly:

    ``W[n, g*VL + v] = centroids[packed_indices[n * n_groups + g], v]``

**Proxy sharing**

All :class:`NPVQLinear` layers that were quantized with the same codebook
reference the *same* :class:`VQProxy` Python object (detected via
``id(qscheme.centroids)`` during :func:`linearize_vq_weights`).  This
means:

- Moving the model to a device propagates centroids exactly **once**.
- The centroid tensor is not duplicated in memory.

**Backends**

``"default"``
    QTensor ``__torch_function__`` path — minimal migration, slow inference.
    Not recommended for production use.

``"compile"``
    A ``torch.compile``'d pure-Python gather + reshape + :func:`F.linear`.
    Compiled once per process; works on CPU and CUDA.  Good baseline.

``"cuda"``
    Custom CUDA kernel (``vq_linear.cu``) that fuses index unpacking,
    centroid gather, and matmul in one pass without materialising the float
    weight matrix.  Requires CUDA; JIT-built on first use.

``"auto"``
    CUDA when available, otherwise compile.

Usage::

    from neuropress.nn import linearize_vq_weights

    # After VPTQGroupEstimator has stored QTensorTractVQ weights:
    linearize_vq_weights(model, backend="auto")
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends import Backend, VQBackend, is_cuda_jit_available

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend resolution (VQ subset — no GEMLITE support for VQ format)
# ---------------------------------------------------------------------------


# Minimum out_features / num_centroids ratio for AUTO to choose CUDA_LUT.
# Below this threshold WMMA is preferred (low centroid reuse, small layers).
_LUT_COV_MIN: float = 4.0


def _resolve_vq_backend(
    backend: str | VQBackend | Backend | None,
    use_tiered: bool = False,
    use_ntier: bool = False,
    use_zero_escape: bool = False,
    bits: int = 8,
    out_features: int = 0,
    num_centroids: int = 0,
    vector_len: int = 4,
) -> VQBackend:
    """Resolve a backend alias to a concrete :class:`VQBackend` for VQ layers.

    The decision is made once at module construction (lowering time) so
    ``self.backend`` always reflects exactly which kernel will run.

    AUTO selection priority (CUDA JIT available):

    1. ``CUDA_ZERO_ESCAPE`` — if *use_zero_escape*.
    2. ``CUDA_NTIER``       — if *tier_bits* set.
    3. ``CUDA_TIERED``      — if *use_tiered*.
    4. ``CUDA_LUT``         — if ``out_features / num_centroids >= _LUT_COV_MIN``
                              (large-vocab layers like lm_head benefit from the
                              LUT decomposition; small attention layers do not).
    5. ``CUDA_WMMA``        — default CUDA path.

    Args:
        backend: A :class:`VQBackend` member, recognised string alias, or
            ``None``.
        use_tiered: Prefer ``CUDA_TIERED`` in AUTO mode.
        use_ntier: Prefer ``CUDA_NTIER`` in AUTO mode.
        use_zero_escape: Prefer ``CUDA_ZERO_ESCAPE`` in AUTO mode.
        bits: Index bit width.
        out_features: Layer output dimension — used for LUT coverage check.
        num_centroids: Codebook size NC — used for LUT coverage check.
        vector_len: Features per VQ vector.  The CUDA kernels are compiled for
            ``vector_len=4`` only, so any other value falls back to ``COMPILE``.

    Returns:
        A concrete :class:`VQBackend` (never ``AUTO``).

    Raises:
        ValueError: If *backend* is an unrecognised string.
    """
    if backend is None or backend == "torch":
        return VQBackend.DEFAULT
    if isinstance(backend, VQBackend | Backend):
        val = backend.value
    elif isinstance(backend, str):
        val = backend
    else:
        val = str(backend)
    if val != "auto":
        return VQBackend(val)  # raises ValueError for unknown strings
    if torch.cuda.is_available() and is_cuda_jit_available():
        # All CUDA inference kernels (WMMA, LUT) only support vector_len=4.
        if vector_len != 4:
            return VQBackend.COMPILE
        if use_zero_escape:
            return VQBackend.CUDA_ZERO_ESCAPE
        if use_ntier:
            return VQBackend.CUDA_NTIER
        if use_tiered:
            return VQBackend.CUDA_TIERED
        if bits > 12:
            # WMMA shared-memory cap is NC=4096 (12-bit); route large NC to LUT.
            return VQBackend.CUDA_LUT
        if num_centroids > 0 and out_features / num_centroids >= _LUT_COV_MIN:
            return VQBackend.CUDA_LUT
        return VQBackend.CUDA_WMMA
    return VQBackend.COMPILE


# ---------------------------------------------------------------------------
# Pure-torch VQ dequant helper — used by the "compile" backend
# ---------------------------------------------------------------------------


def _vq_dequant_linear(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    bits: int,
    n_vectors: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Unpack VQ indices, gather centroids, and apply linear transform.

    This function is the target for ``torch.compile``'d execution on the
    ``"compile"`` backend.  The gather + reshape pattern lets Inductor fuse
    the centroid lookup and the matmul into a single kernel.

    Args:
        x: Input tensor ``[..., in_features]``.
        packed_indices: Packed uint8 index buffer (1-D).
        centroids: Codebook ``[num_centroids, vector_len]``, float16.
        bits: Bits per index — 4 or 8.
        n_vectors: Total number of VQ vectors (= ``out_features * n_groups``).
        out_features: Output dimension N.
        in_features: Input dimension K.
        bias: Optional float bias ``[out_features]``.

    Returns:
        Output tensor ``[..., out_features]`` in the same dtype as *x*.
    """
    from ..transforms.vq_packing import IndexPacking

    indices = IndexPacking.unpack(packed_indices, bits, n_vectors)  # [N*G]
    w = centroids[indices].reshape(out_features, in_features)  # [N, K]
    return F.linear(x, w.to(x.dtype), bias)


_compiled_vq: object | None = None


def _get_compiled_vq() -> object:
    """Return the compiled VQ dequant+linear function (created on first call)."""
    global _compiled_vq
    if _compiled_vq is None:
        _compiled_vq = torch.compile(_vq_dequant_linear)
    return _compiled_vq


def _vq_tiered_dequant_linear(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    n_vectors: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Unpack tiered VQ indices, gather centroids, and apply linear transform.

    Uses :class:`~neuropress.transforms.vq_packing.TieredIndexPacking` to
    decode the two-tier prefix-coded index buffer, then gathers centroid
    vectors and calls :func:`torch.nn.functional.linear`.

    The unpack is device-agnostic: it runs on whichever device
    *packed_indices* lives on (CPU or CUDA).

    Args:
        x: Input tensor ``[..., in_features]``.
        packed_indices: 1-D ``uint8`` tiered buffer (header + streams).
        centroids: Codebook ``[num_centroids, vector_len]``, float16.
        n_vectors: Expected number of VQ vectors (used for shape assertion).
        out_features: Output dimension N.
        in_features: Input dimension K.
        bias: Optional float bias ``[out_features]``.

    Returns:
        Output tensor ``[..., out_features]`` in the same dtype as *x*.
    """
    from ..transforms.vq_packing import TieredIndexPacking

    indices = TieredIndexPacking.unpack(packed_indices)
    if indices.device != x.device:
        indices = indices.to(x.device)
    w = centroids[indices].reshape(out_features, in_features)
    return F.linear(x, w.to(x.dtype), bias)


_compiled_vq_tiered: object | None = None


def _get_compiled_vq_tiered() -> object:
    """Return the compiled tiered VQ dequant+linear function."""
    global _compiled_vq_tiered
    if _compiled_vq_tiered is None:
        _compiled_vq_tiered = torch.compile(_vq_tiered_dequant_linear)
    return _compiled_vq_tiered


def _vq_ntier_dequant_linear(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    n_vectors: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Unpack n-tier CBS VQ indices, gather centroids, apply linear.

    Uses :class:`~neuropress.transforms.vq_packing.NTieredIndexPacking`
    to decode the cascaded binary splitting buffer.

    Args:
        x: Input tensor ``[..., in_features]``.
        packed_indices: 1-D ``uint8`` n-tier buffer (header + streams).
        centroids: Codebook ``[num_centroids, vector_len]``, float16.
        n_vectors: Expected number of VQ vectors.
        out_features: Output dimension N.
        in_features: Input dimension K.
        bias: Optional float bias ``[out_features]``.

    Returns:
        Output tensor ``[..., out_features]`` in the same dtype as *x*.
    """
    from ..transforms.vq_packing import NTieredIndexPacking

    indices = NTieredIndexPacking.unpack(packed_indices)
    if indices.device != x.device:
        indices = indices.to(x.device)
    w = centroids[indices].reshape(out_features, in_features)
    return F.linear(x, w.to(x.dtype), bias)


_compiled_vq_ntier: object | None = None


def _get_compiled_vq_ntier() -> object:
    """Return the compiled n-tier VQ dequant+linear function."""
    global _compiled_vq_ntier
    if _compiled_vq_ntier is None:
        _compiled_vq_ntier = torch.compile(_vq_ntier_dequant_linear)
    return _compiled_vq_ntier


def _vq_zero_escape_dequant_linear(
    x: torch.Tensor,
    packed_indices: torch.Tensor,
    centroids: torch.Tensor,
    n_vectors: int,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Unpack zero-escape VQ indices, gather centroids, apply linear.

    Uses :class:`~neuropress.transforms.vq_packing.ZeroEscapePacking`
    to decode the flat 2-bit prefix buffer.

    Args:
        x: Input tensor ``[..., in_features]``.
        packed_indices: 1-D ``uint8`` zero-escape buffer (header + streams).
        centroids: Codebook ``[num_centroids, vector_len]``, float16.
        n_vectors: Expected number of VQ vectors.
        out_features: Output dimension N.
        in_features: Input dimension K.
        bias: Optional float bias ``[out_features]``.

    Returns:
        Output tensor ``[..., out_features]`` in the same dtype as *x*.
    """
    from ..transforms.vq_packing import ZeroEscapePacking

    indices = ZeroEscapePacking.unpack(packed_indices)
    if indices.device != x.device:
        indices = indices.to(x.device)
    w = centroids[indices].reshape(out_features, in_features)
    return F.linear(x, w.to(x.dtype), bias)


def _vq_gather_linear(
    x: torch.Tensor,
    indices: torch.Tensor,
    centroids: torch.Tensor,
    out_features: int,
    in_features: int,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Gather centroids by pre-unpacked indices and apply linear.

    Compile-friendly: no ``.item()`` calls, no graph breaks.

    Args:
        x: Input tensor ``[..., in_features]``.
        indices: 1-D ``int64`` centroid indices (already unpacked).
        centroids: Codebook ``[num_centroids, vector_len]``, float16.
        out_features: Output dimension N.
        in_features: Input dimension K.
        bias: Optional float bias ``[out_features]``.

    Returns:
        Output tensor ``[..., out_features]`` in the same dtype as *x*.
    """
    w = centroids[indices].reshape(out_features, in_features)
    return F.linear(x, w.to(x.dtype), bias)


_compiled_vq_gather_linear: object | None = None


def _get_compiled_vq_gather_linear() -> object:
    """Return the compiled gather+linear function for zero-escape."""
    global _compiled_vq_gather_linear
    if _compiled_vq_gather_linear is None:
        _compiled_vq_gather_linear = torch.compile(_vq_gather_linear)
    return _compiled_vq_gather_linear


# ---------------------------------------------------------------------------
# VQProxy — shared centroid table
# ---------------------------------------------------------------------------


class VQProxy(nn.Module):
    """Shared VQ codebook holder.

    Holds the ``centroids`` tensor ``[num_centroids, vector_len]`` as a
    registered buffer.  A single :class:`VQProxy` instance is shared by all
    :class:`NPVQLinear` layers that belong to the same quantization group.

    Sharing is achieved by *not* assigning the proxy as a sub-module of each
    :class:`NPVQLinear` (which would make it appear multiple times in
    ``state_dict``).  Instead, the model-owner is responsible for registering
    the proxy as a named child of the top-level model.

    Attributes:
        centroids: Float16 codebook ``[num_centroids, vector_len]``.
        num_centroids: Number of codebook entries.
        vector_len: Number of elements per centroid.
    """

    centroids: torch.Tensor

    def __init__(self, centroids: torch.Tensor) -> None:
        """Initialise VQProxy with a codebook tensor.

        Args:
            centroids: Float codebook ``[num_centroids, vector_len]``.
                Stored as float16.
        """
        super().__init__()
        self.register_buffer("centroids", centroids.to(torch.float16))

    @property
    def num_centroids(self) -> int:
        """Number of entries in the codebook."""
        return self.centroids.shape[0]

    @property
    def vector_len(self) -> int:
        """Number of elements per centroid."""
        return self.centroids.shape[1]

    def extra_repr(self) -> str:
        """Return a concise parameter summary."""
        return f"num_centroids={self.num_centroids}, vector_len={self.vector_len}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """VQProxy has no forward — access centroids directly.

        Args:
            x: Unused.

        Raises:
            RuntimeError: Always — VQProxy is not a callable layer.
        """
        raise RuntimeError(
            "VQProxy.forward() should not be called.  Access centroids "
            "directly via self.proxy.centroids."
        )


# ---------------------------------------------------------------------------
# NPVQLinear
# ---------------------------------------------------------------------------


class NPVQLinear(nn.Module):
    """Linear layer with VQ-compressed weight using a shared codebook proxy.

    The weight matrix is not stored explicitly.  Instead the packed index
    buffer :attr:`packed_indices` and the shared :attr:`proxy` centroids are
    used to reconstruct the weight on the fly during each forward pass.

    Attributes:
        out_features: Output dimension N.
        in_features: Input dimension K.
        vector_len: Number of weight elements per VQ vector.
        bits: Bits per packed index (4 or 8).
        n_vectors: Total number of VQ vectors (``out_features * n_groups``).
        backend: Active :class:`~neuropress.nn.backends.VQBackend`.
        proxy: Shared :class:`VQProxy` holding the centroid table.
        column_perm: Optional int64 permutation ``[K]`` applied to input
            activations before the matmul.  Present when the VQ weight was
            quantized with a Hessian-based column permutation.
    """

    out_features: int
    in_features: int
    vector_len: int
    bits: int
    n_vectors: int
    backend: VQBackend
    proxy: VQProxy
    use_tiered: bool
    cold_bits: int

    def __init__(
        self,
        proxy: VQProxy,
        packed_indices: torch.Tensor,
        bits: int,
        out_features: int,
        in_features: int,
        vector_len: int,
        backend: str | VQBackend | None,
        bias: torch.Tensor | None = None,
        column_perm: torch.Tensor | None = None,
        use_tiered: bool = False,
        cold_bits: int = 8,
        hot_row_offsets: torch.Tensor | None = None,
        cold_row_offsets: torch.Tensor | None = None,
        qtensor_meta: dict[str, str] | None = None,
        tier_bits: list[int] | None = None,
        ntier_row_offsets: list[torch.Tensor] | None = None,
        zero_escape: bool = False,
        zesc_row_offsets: list[torch.Tensor] | None = None,
    ) -> None:
        """Initialise NPVQLinear.

        The *proxy* is stored as a plain Python attribute (not a sub-module)
        so that it is not duplicated in the model's ``state_dict``.  The
        model owner is responsible for registering the proxy via
        ``model.add_module("vq_proxy_<hash>", proxy)``.

        Args:
            proxy: Shared :class:`VQProxy` for this layer.
            packed_indices: 1-D uint8 packed index buffer.  When
                *use_tiered=True* this is the self-contained tiered buffer
                (header + prefix + hot + cold streams).
            bits: Bits per index — 4 or 8 for fixed-width packing, or the
                cold-tier bit width when *use_tiered=True*.
            out_features: Output dimension N.
            in_features: Input dimension K.
            vector_len: Length of each VQ vector.
            backend: Execution backend.
            bias: Optional float bias ``[out_features]``.
            column_perm: Optional int64 column permutation ``[K]``.
                When present, input activations are re-ordered via
                ``x[..., column_perm]`` before the matmul so they align
                with the permuted weight stored in ``packed_indices``.
            use_tiered: If ``True``, ``packed_indices`` holds a tiered buffer
                and :meth:`_forward_default` / :meth:`_forward_cuda_tiered`
                use :class:`~neuropress.transforms.vq_packing.TieredIndexPacking`
                to decode it.
            cold_bits: Cold-tier bit width — used only when *use_tiered=True*.
            hot_row_offsets: Precomputed per-row hot-index prefix sums
                ``[out_features + 1]``, int32.  Required when backend is
                ``CUDA_TIERED``.  Produced by
                :meth:`~neuropress.transforms.vq_packing.TieredIndexPacking.compute_row_offsets`.
            cold_row_offsets: Precomputed per-row cold-index prefix sums
                ``[out_features + 1]``, int32.  Required when backend is
                ``CUDA_TIERED``.
            qtensor_meta: Serde metadata dict for QTensor reconstruction
                via :meth:`to_qtensor`.
            tier_bits: N-tier CBS bit widths, e.g. ``[4, 8, 12, 17]``.
                When provided, the n-tier CBS packing path is used
                instead of the 2-tier path.
            ntier_row_offsets: Precomputed per-tier per-row cumulative
                counts.  List of ``D`` int32 tensors of shape
                ``[out_features + 1]``.  Required for ``CUDA_NTIER``.
            zero_escape: If ``True``, ``packed_indices`` holds a
                zero-escape buffer (flat 2-bit prefix).  Uses
                :class:`~neuropress.transforms.vq_packing.ZeroEscapePacking`
                for decode.
            zesc_row_offsets: Precomputed per-tier per-row cumulative
                counts for zero-escape.  List of 4 int32 tensors
                (zero, hot, warm, cold) of shape ``[out_features + 1]``.
                Required for ``CUDA_ZERO_ESCAPE``.
        """
        super().__init__()
        self._qtensor_meta = qtensor_meta
        self.out_features = out_features
        self.in_features = in_features
        self.vector_len = vector_len
        self.bits = bits
        self.n_vectors = out_features * (in_features // vector_len)
        self.use_tiered = use_tiered
        self.cold_bits = cold_bits
        self.tier_bits: list[int] | None = tier_bits
        self.zero_escape: bool = zero_escape
        self.backend = _resolve_vq_backend(
            backend,
            use_tiered=use_tiered,
            use_ntier=tier_bits is not None,
            use_zero_escape=zero_escape,
            bits=bits,
            out_features=out_features,
            num_centroids=proxy.num_centroids,
            vector_len=vector_len,
        )

        # Store proxy as plain attribute — NOT a sub-module.
        # This prevents duplicate centroid buffers in state_dict.
        object.__setattr__(self, "proxy", proxy)

        # For tiered/ntier, packed_indices is split into separate buffers
        # below (no duplication).  For fixed-width, keep as-is.
        self._packed_is_split = False
        self.register_buffer("packed_indices", packed_indices.contiguous())
        if bias is not None:
            self.register_buffer("bias", bias.contiguous())
        else:
            self.bias: torch.Tensor | None = None
        if column_perm is not None:
            self.register_buffer("column_perm", column_perm.to(torch.int64))
        else:
            self.column_perm: torch.Tensor | None = None
        if hot_row_offsets is not None:
            self.register_buffer("hot_row_offsets", hot_row_offsets.to(torch.int32))
        else:
            self.hot_row_offsets: torch.Tensor | None = None
        if cold_row_offsets is not None:
            self.register_buffer("cold_row_offsets", cold_row_offsets.to(torch.int32))
        else:
            self.cold_row_offsets: torch.Tensor | None = None
        # Zero-escape row offsets: list of 4 int32 tensors.
        if zesc_row_offsets is not None:
            for i, off in enumerate(zesc_row_offsets):
                self.register_buffer(f"zesc_row_off_{i}", off.to(torch.int32))
            self._zesc_n_offsets = len(zesc_row_offsets)
        else:
            self._zesc_n_offsets = 0
        # N-tier row offsets: list of D int32 tensors.
        if ntier_row_offsets is not None:
            for i, off in enumerate(ntier_row_offsets):
                self.register_buffer(f"ntier_row_off_{i}", off.to(torch.int32))
            self._ntier_n_tiers = len(ntier_row_offsets)
        else:
            self._ntier_n_tiers = 0

        # ----- Pre-split stream buffers for CUDA tiered/ntier paths -----
        # Avoids re-parsing the header and re-slicing on every forward call.
        self._tiered_cold_bits: int = 0
        self._tiered_hot_bits: int = 0
        self._nt_n_tiers: int = 0
        self._ze_cold_bits: int = 0
        if zero_escape:
            self._presplit_zero_escape(packed_indices)
        elif use_tiered and tier_bits is None:
            self._presplit_tiered(packed_indices)
        elif tier_bits is not None:
            self._presplit_ntier(packed_indices)

    # ------------------------------------------------------------------
    # Pre-split helpers (called once at construction)
    # ------------------------------------------------------------------

    def _presplit_tiered(self, packed: torch.Tensor) -> None:
        """Split a 2-tier buffer into stream buffers, dropping the original.

        The header is cached (tiny) and the three data streams are registered
        as ``persistent=False`` buffers.  The original ``packed_indices`` is
        replaced with ``None`` to free RAM — :meth:`to_qtensor` reassembles
        it on demand from ``_t_header + _t_prefix + _t_hot + _t_cold``.

        ``persistent=False`` buffers move with ``.to(device)`` and have
        stable CUDA addresses (CUDA-graph safe), but do not appear in
        ``state_dict``.
        """
        from ..transforms.vq_packing import TieredIndexPacking

        n_indices, n_hot, cold_bits, hot_bits, H = TieredIndexPacking._parse_header(
            packed
        )
        n_cold = n_indices - n_hot
        prefix_bytes = (n_indices + 7) // 8
        hot_bytes = math.ceil(n_hot * hot_bits / 8)
        cold_bytes = math.ceil(n_cold * cold_bits / 8)

        # Cache the header (16 bytes) for reassembly in to_qtensor().
        self.register_buffer(
            "_t_header",
            packed[:H].clone(),
            persistent=False,
        )

        o = H
        self.register_buffer(
            "_t_prefix",
            packed[o : o + prefix_bytes].clone(),
            persistent=False,
        )
        o += prefix_bytes
        self.register_buffer(
            "_t_hot",
            packed[o : o + hot_bytes].clone(),
            persistent=False,
        )
        o += hot_bytes
        self.register_buffer(
            "_t_cold",
            packed[o : o + cold_bytes].clone(),
            persistent=False,
        )
        self._tiered_cold_bits = cold_bits
        self._tiered_hot_bits = hot_bits

        # Drop the original monolithic buffer — streams are the only copy.
        self.packed_indices = None
        self._packed_is_split = True

    def _presplit_ntier(self, packed: torch.Tensor) -> None:
        """Split an n-tier CBS buffer into stream buffers, dropping the original.

        Same approach as :meth:`_presplit_tiered` — header cached, streams
        registered as ``persistent=False``, original freed.
        """
        from ..transforms.vq_packing import NTieredIndexPacking

        (n_indices, n_tiers, tier_counts, tier_bits_list, H) = (
            NTieredIndexPacking._parse_header(packed)
        )

        # Cache header for reassembly.
        self.register_buffer(
            "_nt_header",
            packed[:H].clone(),
            persistent=False,
        )

        offset = H
        n_remaining = n_indices
        for k in range(n_tiers - 1):
            pb = (n_remaining + 7) // 8
            self.register_buffer(
                f"_nt_prefix_{k}",
                packed[offset : offset + pb].clone(),
                persistent=False,
            )
            offset += pb
            n_remaining -= tier_counts[k]

        for k in range(n_tiers):
            nb = math.ceil(tier_counts[k] * tier_bits_list[k] / 8)
            self.register_buffer(
                f"_nt_tier_{k}",
                packed[offset : offset + nb].clone(),
                persistent=False,
            )
            offset += nb

        self._nt_n_tiers = n_tiers
        self._nt_tier_bits = tier_bits_list

        # Drop the original monolithic buffer.
        self.packed_indices = None
        self._packed_is_split = True

        # Pre-build forward arg lists (updated on device move).
        self._rebuild_ntier_lists()

    def _presplit_zero_escape(self, packed: torch.Tensor) -> None:
        """Split a zero-escape buffer into stream buffers, dropping the original.

        Parses the 24-byte header, extracts prefix/hot/warm/cold streams
        and registers them as ``persistent=False`` buffers.
        """
        from ..transforms.vq_packing import ZeroEscapePacking

        n, cold_bits, n_zero, n_hot, n_warm, H = ZeroEscapePacking._parse_header(packed)
        n_cold = n - n_zero - n_hot - n_warm

        self.register_buffer(
            "_ze_header",
            packed[:H].clone(),
            persistent=False,
        )

        offset = H
        prefix_bytes = (n * 2 + 7) // 8
        self.register_buffer(
            "_ze_prefix",
            packed[offset : offset + prefix_bytes].clone(),
            persistent=False,
        )
        offset += prefix_bytes

        hot_bytes = math.ceil(n_hot * 4 / 8)
        self.register_buffer(
            "_ze_hot",
            packed[offset : offset + hot_bytes].clone(),
            persistent=False,
        )
        offset += hot_bytes

        warm_bytes = n_warm  # 8-bit = 1 byte each
        self.register_buffer(
            "_ze_warm",
            packed[offset : offset + warm_bytes].clone(),
            persistent=False,
        )
        offset += warm_bytes

        cold_bytes = math.ceil(n_cold * cold_bits / 8)
        self.register_buffer(
            "_ze_cold",
            packed[offset : offset + cold_bytes].clone(),
            persistent=False,
        )

        self._ze_cold_bits = cold_bits

        # Drop the original monolithic buffer.
        self.packed_indices = None
        self._packed_is_split = True

    def _rebuild_ntier_lists(self) -> None:
        """Pre-build the list args for _forward_cuda_ntier.

        Called once after presplit and again after any device move.
        Avoids per-forward getattr + list comprehension overhead.
        """
        D = self._nt_n_tiers
        self._nt_prefix_list: list[torch.Tensor] = [
            getattr(self, f"_nt_prefix_{k}") for k in range(D - 1)
        ]
        self._nt_tier_list: list[torch.Tensor] = [
            getattr(self, f"_nt_tier_{k}") for k in range(D)
        ]
        self._nt_row_off_list: list[torch.Tensor] = [
            getattr(self, f"ntier_row_off_{k}") for k in range(D)
        ]

    def _apply(self, fn: object, recurse: bool = True) -> NPVQLinear:
        """Override to rebuild cached tensor lists after device moves."""
        result = super()._apply(fn, recurse)
        if self._nt_n_tiers > 0:
            self._rebuild_ntier_lists()
        return result

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_qtensor_linear(
        cls,
        linear: nn.Linear,
        proxy: VQProxy,
        backend: str | VQBackend | None = VQBackend.AUTO,
    ) -> NPVQLinear:
        """Build an ``NPVQLinear`` from an ``nn.Linear`` with a VQ weight.

        Args:
            linear: An ``nn.Linear`` whose ``weight`` is a
                :class:`~neuropress.transforms.vq.QTensorTractVQ`.
            proxy: The shared :class:`VQProxy` for this layer's group.
                Must have been constructed from the same centroids tensor
                referenced by ``linear.weight.qscheme.centroids``.
            backend: Target execution backend.

        Returns:
            An ``NPVQLinear`` equivalent to *linear*.

        Raises:
            TypeError: If ``linear.weight`` is not a ``QTensorTractVQ``.
        """
        from ..transforms.vq import QTensorTractVQ

        weight_q = linear.weight
        if not isinstance(weight_q, QTensorTractVQ):
            raise TypeError(
                f"Expected QTensorTractVQ, got {type(weight_q).__name__!r}."
            )

        qscheme = weight_q.qscheme
        bias = linear.bias.detach().clone() if linear.bias is not None else None
        column_perm = (
            qscheme.column_permutation.detach().clone()
            if qscheme.column_permutation is not None
            else None
        )
        use_tiered = qscheme.use_tiered
        cold_bits = qscheme.is_packed_indices
        tier_bits = getattr(qscheme, "tier_bits", None)
        zero_escape = getattr(qscheme, "zero_escape", False)
        hot_row_offsets: torch.Tensor | None = None
        cold_row_offsets: torch.Tensor | None = None
        ntier_row_offsets: list[torch.Tensor] | None = None
        n_groups = linear.in_features // qscheme.vector_len
        zesc_row_offsets: list[torch.Tensor] | None = None
        if zero_escape:
            from ..transforms.vq_packing import ZeroEscapePacking

            zesc_row_offsets = ZeroEscapePacking.compute_row_offsets(
                weight_q.u8_blob, linear.out_features, n_groups
            )
        elif tier_bits is not None:
            from ..transforms.vq_packing import NTieredIndexPacking

            ntier_row_offsets = NTieredIndexPacking.compute_row_offsets(
                weight_q.u8_blob, linear.out_features, n_groups
            )
        elif use_tiered:
            from ..transforms.vq_packing import TieredIndexPacking

            hot_row_offsets, cold_row_offsets = TieredIndexPacking.compute_row_offsets(
                weight_q.u8_blob, linear.out_features, n_groups
            )
        # Capture serde metadata for later QTensor reconstruction.
        from ..t2n.qtensors import _SERDE_REGISTRY

        serde = _SERDE_REGISTRY[type(weight_q).__name__]
        qtensor_meta = serde.to_metadata(weight_q)

        return cls(
            proxy=proxy,
            packed_indices=weight_q.u8_blob.detach().clone(),
            bits=qscheme.is_packed_indices,
            out_features=linear.out_features,
            in_features=linear.in_features,
            vector_len=qscheme.vector_len,
            backend=backend,
            bias=bias,
            column_perm=column_perm,
            use_tiered=use_tiered,
            cold_bits=cold_bits,
            hot_row_offsets=hot_row_offsets,
            cold_row_offsets=cold_row_offsets,
            qtensor_meta=qtensor_meta,
            tier_bits=tier_bits,
            ntier_row_offsets=ntier_row_offsets,
            zero_escape=zero_escape,
            zesc_row_offsets=zesc_row_offsets,
        )

    # ------------------------------------------------------------------
    # QTensor reconstruction
    # ------------------------------------------------------------------

    @property
    def _packed_or_reassembled(self) -> torch.Tensor:
        """Return ``packed_indices`` or reassemble from split streams.

        Used by fallback forward paths (default, compile) that need the
        monolithic buffer.  For the CUDA tiered/ntier paths this is never
        called — they use the split buffers directly.
        """
        if not self._packed_is_split:
            return self.packed_indices
        return self._reassemble_packed()

    def _reassemble_packed(self) -> torch.Tensor:
        """Reassemble the monolithic packed buffer from split streams.

        Called lazily by :meth:`to_qtensor` when the original
        ``packed_indices`` was freed by ``_presplit_tiered`` /
        ``_presplit_ntier``.

        Returns:
            1-D ``uint8`` buffer identical to the original ``packed_indices``.
        """
        if self.zero_escape:
            return torch.cat(
                [
                    self._ze_header,
                    self._ze_prefix,
                    self._ze_hot,
                    self._ze_warm,
                    self._ze_cold,
                ]
            )
        if self.use_tiered and self.tier_bits is None:
            return torch.cat(
                [
                    self._t_header,
                    self._t_prefix,
                    self._t_hot,
                    self._t_cold,
                ]
            )
        # N-tier path.
        D = self._nt_n_tiers
        parts: list[torch.Tensor] = [self._nt_header]
        for k in range(D - 1):
            parts.append(getattr(self, f"_nt_prefix_{k}"))
        for k in range(D):
            parts.append(getattr(self, f"_nt_tier_{k}"))
        return torch.cat(parts)

    def to_qtensor(self) -> torch.Tensor:
        """Reconstruct the original QTensorTractVQ from this module's buffers.

        If the packed buffer was split into streams (tiered/ntier),
        it is reassembled on demand.  This is only called during
        unlower / save — not on the hot path.

        Returns:
            A ``QTensorTractVQ`` matching the original VQ weight.

        Raises:
            RuntimeError: If no serde metadata was stored.
        """
        if self._qtensor_meta is None:
            raise RuntimeError(
                "Cannot reconstruct QTensor: no serde metadata stored. "
                "Build via NPVQLinear.from_qtensor_linear() to enable "
                "to_qtensor()."
            )
        from ..t2n.qtensors import _SERDE_REGISTRY

        serde = _SERDE_REGISTRY["QTensorTractVQ"]
        u8_blob = (
            self._reassemble_packed() if self._packed_is_split else self.packed_indices
        )
        tensors: dict[str, torch.Tensor] = {
            "u8_blob": u8_blob,
            "centroids": self.proxy.centroids,
        }
        if self.column_perm is not None:
            tensors["column_permutation"] = self.column_perm
        return serde.from_parts(tensors, self._qtensor_meta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply VQ-compressed linear with on-the-fly centroid reconstruction.

        If a column permutation is registered, input activations are
        reordered by ``x[..., column_perm]`` before the matmul so they align
        with the permuted weight stored in ``packed_indices``.

        Args:
            x: Input tensor ``[..., in_features]``.

        Returns:
            Output tensor ``[..., out_features]``.
        """
        if self.column_perm is not None:
            x = x[..., self.column_perm]
        if self.backend == VQBackend.CUDA_ZERO_ESCAPE:
            return self._forward_cuda_zero_escape(x)
        if self.backend == VQBackend.CUDA_NTIER:
            return self._forward_cuda_ntier(x)
        if self.backend == VQBackend.CUDA_TIERED:
            return self._forward_cuda_tiered(x)
        if self.backend == VQBackend.CUDA_LUT:
            return self._forward_cuda_lut(x)
        if self.backend == VQBackend.CUDA_WMMA:
            return self._forward_cuda_wmma(x)
        if self.backend == VQBackend.CUDA:
            return self._forward_cuda(x)
        if self.backend == VQBackend.COMPILE:
            if self.zero_escape:
                return self._forward_compile_zero_escape(x)
            if self.tier_bits is not None:
                return self._forward_compile_ntier(x)
            if self.use_tiered:
                return self._forward_compile_tiered(x)
            return self._forward_compile(x)
        return self._forward_default(x)

    def _forward_default(self, x: torch.Tensor) -> torch.Tensor:
        if self.zero_escape:
            return _vq_zero_escape_dequant_linear(
                x,
                self._packed_or_reassembled,
                self.proxy.centroids,
                self.n_vectors,
                self.out_features,
                self.in_features,
                self.bias,
            )
        if self.tier_bits is not None:
            return _vq_ntier_dequant_linear(
                x,
                self._packed_or_reassembled,
                self.proxy.centroids,
                self.n_vectors,
                self.out_features,
                self.in_features,
                self.bias,
            )
        if self.use_tiered:
            return _vq_tiered_dequant_linear(
                x,
                self._packed_or_reassembled,
                self.proxy.centroids,
                self.n_vectors,
                self.out_features,
                self.in_features,
                self.bias,
            )
        from ..transforms.vq_packing import IndexPacking

        indices = IndexPacking.unpack(self.packed_indices, self.bits, self.n_vectors)
        w = self.proxy.centroids[indices].reshape(self.out_features, self.in_features)
        return F.linear(x, w.to(x.dtype), self.bias)

    def _forward_compile(self, x: torch.Tensor) -> torch.Tensor:
        return _get_compiled_vq()(
            x,
            self.packed_indices,
            self.proxy.centroids,
            self.bits,
            self.n_vectors,
            self.out_features,
            self.in_features,
            self.bias,
        )

    def _forward_compile_tiered(self, x: torch.Tensor) -> torch.Tensor:
        return _get_compiled_vq_tiered()(
            x,
            self._packed_or_reassembled,
            self.proxy.centroids,
            self.n_vectors,
            self.out_features,
            self.in_features,
            self.bias,
        )

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        from .. import kernels

        *leading, _ = x.shape
        # Kernel accepts fp16, bf16, and fp32 natively — no cast needed.
        x_2d = x.reshape(-1, self.in_features)
        y_2d = kernels.vq_linear(
            x_2d,
            self.packed_indices,
            self.proxy.centroids,
            self.bits,
            self.vector_len,
            self.bias,
        )
        return y_2d.reshape(*leading, self.out_features)

    # M <= this → LUT gather; M > this → fall through to WMMA GEMM.
    _LUT_M_MAX: int = 16

    def _forward_cuda_lut(self, x: torch.Tensor) -> torch.Tensor:
        """LUT-based VQ forward: fast for large out_features (e.g. lm_head).

        Phase 1 — build LUT via one cuBLAS GEMM:
            lut[m, g, k] = dot(x[m, g*VL:(g+1)*VL], centroids[k])
            shape [M, n_groups, NC], lives in L2 for the gather phase.

        Phase 2 — gather via CUDA kernel:
            y[m, j] = sum_g  lut[m, g, packed_indices[j*n_groups + g]]
            4× fewer FLOPs than dot4-per-group (VL=4 → scalar lookup).

        Falls back to WMMA GEMM when M > _LUT_M_MAX (LUT would spill L2
        and WMMA tensor-core tiles become more efficient).
        """
        from .. import kernels

        *leading, _ = x.shape
        orig_dtype = x.dtype
        if x.dtype == torch.float32:
            x = x.half()
        x_2d = x.reshape(-1, self.in_features)
        M = x_2d.shape[0]

        if M > self._LUT_M_MAX:
            y_2d = kernels.vq_linear_wmma(
                x_2d,
                self.packed_indices,
                self.proxy.centroids,
                self.bits,
                self.vector_len,
                self.bias,
            )
            y = y_2d.reshape(*leading, self.out_features)
            return y.to(orig_dtype) if y.dtype != orig_dtype else y

        n_groups = self.in_features // self.vector_len
        NC = self.proxy.centroids.shape[0]

        # Phase 1: LUT build — [M*n_groups, VL] @ [VL, NC] → [M, n_groups, NC]
        c_f32 = self.proxy.centroids.float()  # [NC, VL]
        lut = (
            x_2d.float()
            .reshape(M * n_groups, self.vector_len)
            .mm(c_f32.T)
            .reshape(M, n_groups, NC)
            .contiguous()
        )

        # Phase 2: LUT gather → [M, N] float32
        y_2d = kernels.vq_lut_gather(
            lut, self.packed_indices, self.bits, self.out_features
        ).to(x.dtype)

        if self.bias is not None:
            y_2d = y_2d + self.bias

        return y_2d.reshape(*leading, self.out_features)

    def _forward_cuda_wmma(self, x: torch.Tensor) -> torch.Tensor:
        from .. import kernels

        *leading, _ = x.shape
        orig_dtype = x.dtype
        if x.dtype == torch.float32:
            x = x.half()
        x_2d = x.reshape(-1, self.in_features)
        y_2d = kernels.vq_linear_wmma(
            x_2d,
            self.packed_indices,
            self.proxy.centroids,
            self.bits,
            self.vector_len,
            self.bias,
        )
        y = y_2d.reshape(*leading, self.out_features)
        return y.to(orig_dtype) if y.dtype != orig_dtype else y

    def _forward_cuda_tiered(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the warp-ballot tiered CUDA kernel.

        All stream buffers are pre-split ``persistent=False`` registered
        buffers with stable addresses — CUDA-graph safe, zero Python
        overhead per call.

        Args:
            x: Input tensor ``[..., in_features]``.

        Returns:
            Output tensor ``[..., out_features]``.
        """
        from .. import kernels

        *leading, _ = x.shape
        orig_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.half()
        x_2d = x.reshape(-1, self.in_features)
        y_2d = kernels.vq_linear_tiered(
            x_2d,
            self._t_prefix,
            self._t_hot,
            self._t_cold,
            self.hot_row_offsets,
            self.cold_row_offsets,
            self.proxy.centroids,
            self._tiered_cold_bits,
            self._tiered_hot_bits,
            self.vector_len,
            self.bias,
        )
        y = y_2d.reshape(*leading, self.out_features)
        return y.to(orig_dtype) if y.dtype != orig_dtype else y

    def _forward_compile_ntier(self, x: torch.Tensor) -> torch.Tensor:
        return _get_compiled_vq_ntier()(
            x,
            self._packed_or_reassembled,
            self.proxy.centroids,
            self.n_vectors,
            self.out_features,
            self.in_features,
            self.bias,
        )

    def _forward_compile_zero_escape(self, x: torch.Tensor) -> torch.Tensor:
        from ..transforms.vq_packing import ZeroEscapePacking

        # Eager unpack (has .item() calls that break torch.compile).
        indices = ZeroEscapePacking.unpack(self._packed_or_reassembled)
        if indices.device != x.device:
            indices = indices.to(x.device)
        # Compiled gather + linear (no graph breaks).
        return _get_compiled_vq_gather_linear()(
            x,
            indices,
            self.proxy.centroids,
            self.out_features,
            self.in_features,
            self.bias,
        )

    def _forward_cuda_zero_escape(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the zero-escape CUDA kernel.

        All stream buffers are pre-split ``persistent=False`` registered
        buffers with stable addresses — CUDA-graph safe, zero Python
        overhead per call.

        Args:
            x: Input tensor ``[..., in_features]``.

        Returns:
            Output tensor ``[..., out_features]``.
        """
        from .. import kernels

        *leading, _ = x.shape
        orig_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.half()
        x_2d = x.reshape(-1, self.in_features)
        y_2d = kernels.vq_linear_zero_escape(
            x_2d,
            self._ze_prefix,
            self._ze_hot,
            self._ze_warm,
            self._ze_cold,
            self.zesc_row_off_0,
            self.zesc_row_off_1,
            self.zesc_row_off_2,
            self.zesc_row_off_3,
            self.proxy.centroids,
            self._ze_cold_bits,
            self.vector_len,
            self.bias,
        )
        y = y_2d.reshape(*leading, self.out_features)
        return y.to(orig_dtype) if y.dtype != orig_dtype else y

    def _forward_cuda_ntier(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the n-tier CBS CUDA kernel.

        Uses pre-built list attributes — zero per-call getattr/list
        overhead.  CUDA-graph safe.

        Args:
            x: Input tensor ``[..., in_features]``.

        Returns:
            Output tensor ``[..., out_features]``.
        """
        from .. import kernels

        *leading, _ = x.shape
        orig_dtype = x.dtype
        if x.dtype == torch.bfloat16:
            x = x.half()
        x_2d = x.reshape(-1, self.in_features)
        y_2d = kernels.vq_linear_ntier(
            x_2d,
            self._nt_prefix_list,
            self._nt_tier_list,
            self._nt_row_off_list,
            self.proxy.centroids,
            self._nt_tier_bits,
            self.vector_len,
            self.bias,
        )
        y = y_2d.reshape(*leading, self.out_features)
        return y.to(orig_dtype) if y.dtype != orig_dtype else y

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise parameter summary."""
        perm_str = ", perm=True" if self.column_perm is not None else ""
        if self.zero_escape:
            tiered_str = ", zero_escape=True"
        elif self.tier_bits is not None:
            tiered_str = f", ntier={self.tier_bits}"
        elif self.use_tiered:
            tiered_str = f", tiered=True(cold={self.cold_bits})"
        else:
            tiered_str = ""
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"vector_len={self.vector_len}, "
            f"num_centroids={self.proxy.num_centroids}, "
            f"bits={self.bits}, backend={self.backend.value}"
            f"{perm_str}{tiered_str}"
        )


# ---------------------------------------------------------------------------
# Model-level transform
# ---------------------------------------------------------------------------


def linearize_vq_weights(
    model: nn.Module,
    backend: str | VQBackend | None = VQBackend.AUTO,
) -> nn.Module:
    """Replace all ``nn.Linear`` with VQ-compressed weights with ``NPVQLinear``.

    Walks *model*, identifies every ``nn.Linear`` whose ``weight`` is a
    :class:`~neuropress.transforms.vq.QTensorTractVQ`, and replaces it with
    an :class:`NPVQLinear`.  Layers sharing the same centroids tensor (same
    Python object identity — ``id(qscheme.centroids)``) receive the *same*
    :class:`VQProxy` instance, preserving the shared-proxy property.

    Each unique proxy is registered on the model root via
    ``model.add_module("_vq_proxy_<i>", proxy)`` so that its buffer is
    tracked by PyTorch's parameter/buffer machinery (device moves,
    ``state_dict``, etc.).

    Args:
        model: The ``nn.Module`` to transform (mutated in-place).
        backend: Execution backend for all created ``NPVQLinear`` layers.
            Defaults to ``VQBackend.AUTO`` (CUDA when available, else compile).

    Returns:
        The same *model* (for chaining).

    Example::

        # After VPTQ quantization:
        linearize_vq_weights(model, backend="auto")
        # All VQ-weight linears now use NPVQLinear — fast gather+matmul.
    """
    from ..transforms.vq import QTensorTractVQ

    # Cache: centroid tensor id → VQProxy
    proxy_cache: dict[int, VQProxy] = {}
    replacements: list[tuple[str, NPVQLinear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weight_q = module.weight
        if not isinstance(weight_q, QTensorTractVQ):
            continue

        centroids = weight_q.qscheme.centroids
        cid = id(centroids)
        if cid not in proxy_cache:
            proxy_cache[cid] = VQProxy(centroids)

        proxy = proxy_cache[cid]
        try:
            np_layer = NPVQLinear.from_qtensor_linear(module, proxy, backend)
        except (TypeError, ValueError) as exc:
            log.warning("linearize_vq_weights: skipping %s — %s", name, exc)
            continue
        replacements.append((name, np_layer))

    if not replacements:
        log.info("linearize_vq_weights: no VQ-weight linear layers found.")
        return model

    # Register each unique proxy on the model root so buffers are tracked.
    for i, proxy in enumerate(proxy_cache.values()):
        model.add_module(f"_vq_proxy_{i}", proxy)

    # Replace linear modules.
    for name, np_layer in replacements:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, np_layer)

    backend_counts: dict[str, int] = {}
    for _, np_layer in replacements:
        key = np_layer.backend.value
        backend_counts[key] = backend_counts.get(key, 0) + 1
    summary = ", ".join(f"{v}x{k}" for k, v in sorted(backend_counts.items()))
    log.info(
        "linearize_vq_weights: replaced %d layer(s) [%s], %d unique proxy/ies",
        len(replacements),
        summary,
        len(proxy_cache),
    )
    return model
