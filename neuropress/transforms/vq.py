"""Vector Quantization scheme and QTensor type for neuropress.

Adapted from ``vptq-explore/vptq_explore/vq_t2n.py`` and integrated into
the neuropress framework.

Reference
---------
VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language
Models — Yang Wang et al., Microsoft Research.
https://github.com/microsoft/VPTQ

How VQ differs from affine quantization
-----------------------------------------
Affine quantization (Q4_0, Q4_1) maps each scalar weight value to a
per-group ``(scale, zero_point)`` pair:

    ``w ≈ (q - zp) * scale``

Vector quantization maps a *group* of ``vector_len`` consecutive weight
values to a single index into a shared codebook (the **proxy**):

    ``w[i : i+vector_len] ≈ centroids[index_i]``

There are no per-weight scales or zero-points.  All compression metadata
lives in the shared centroid table, which can be shared across many layers
(see :class:`VQScheme`).

Index packing
-------------
Codebook indices are stored in packed ``uint8`` form via
:class:`~neuropress.transforms.vq_packing.IndexPacking`.  The number of
bits per index is derived automatically from ``num_centroids``:

    - ``num_centroids <=    16`` →  4 bits/index
    - ``num_centroids <=   256`` →  8 bits/index
    - ``num_centroids <=  1024`` → 10 bits/index
    - (up to 65 536 centroids, 16 bits/index)

The packing level is recorded in :attr:`VQScheme.is_packed_indices` and
stored in the safetensors metadata so that the serde can reconstruct the
correct bit width without ambiguity.

Two-tier prefix coding
----------------------
When ``use_tiered=True`` on :class:`VQScheme`, the index buffer uses
:class:`~neuropress.transforms.vq_packing.TieredIndexPacking` instead of
:class:`~neuropress.transforms.vq_packing.IndexPacking`.  A 1-bit prefix
selects hot (4-bit, centroid 0–15) vs cold (``is_packed_indices``-bit).
Requires frequency-sorted centroids (centroid-0 = most frequent).

Usage::

    from neuropress.transforms.vq import VQScheme, QTensorTractVQ, ProxySelector

    centroids = torch.randn(256, 4, dtype=torch.float16)   # 256-entry codebook
    vq = VQScheme(centroids=centroids, proxy_selector=ProxySelector.L2,
                  original_shape=torch.Size([128, 512]))

    weight = torch.randn(128, 512)
    packed_indices = vq.quantize_as_u8(weight)             # uint8, packed
    w_approx = vq.dequantize(packed_indices, torch.float32)
"""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from pathlib import Path

import torch
from torch_to_nnef.exceptions import T2NErrorNotImplemented
from torch_to_nnef.tensor.quant import QScheme
from torch_to_nnef.tensor.quant.qtract import InferenceTarget, QTensorTract

from .vq_packing import (
    IndexPacking,
    NTieredIndexPacking,
    TieredIndexPacking,
    ZeroEscapePacking,
)


class ProxySelector(StrEnum):
    """Distance metric used to assign weight vectors to codebook entries.

    Attributes:
        L2: Euclidean (L2) distance — fast and suitable for most weights.
        COSINE: Cosine dissimilarity — useful when weight directions matter
            more than magnitudes (e.g. attention projections).
    """

    L2 = "l2"
    COSINE = "cosine"


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the pairwise cosine *dissimilarity* matrix between rows of *a* and *b*.

    Dissimilarity is ``1 - cosine_similarity`` so that the nearest centroid
    still corresponds to the *minimum* value, matching the L2 convention.

    Args:
        a: Tensor of shape ``[N, D]``.
        b: Tensor of shape ``[M, D]``.

    Returns:
        Dissimilarity matrix of shape ``[N, M]``.
    """
    a_norm = torch.nn.functional.normalize(a, dim=1)
    b_norm = torch.nn.functional.normalize(b, dim=1)
    sim = torch.mm(a_norm, b_norm.T)
    return 1.0 - sim


class VQScheme(QScheme):
    """Vector Quantization scheme mapping weight vectors to codebook indices.

    A ``VQScheme`` holds a *shared* centroid table (the **proxy**) that may
    be referenced by multiple :class:`QTensorTractVQ` instances — one per
    quantized ``nn.Linear`` layer in the same quantization group.  This
    amortises the per-layer storage cost of the codebook across all layers
    in the group.

    The index bit width is derived automatically from ``num_centroids`` via
    :class:`~neuropress.transforms.vq_packing.IndexPacking.bits_for` and
    stored in :attr:`is_packed_indices`.

    Attributes:
        centroids: Codebook tensor ``[num_centroids, vector_len]`` (fp16).
            Shared across all layers in the same quantization group.
        proxy_selector: Distance metric for nearest-centroid assignment.
        is_packed_indices: Bits per index (4–16), derived from
            ``num_centroids`` via
            :meth:`~neuropress.transforms.vq_packing.IndexPacking.bits_for`.
            Stored in the serde metadata for unambiguous reconstruction.
            When ``use_tiered=True`` this is the *cold-tier* bit width.
        use_tiered: When ``True``, :meth:`quantize_as_u8` produces a
            :class:`~neuropress.transforms.vq_packing.TieredIndexPacking`
            buffer (header + prefix + hot + cold streams) and
            :meth:`dequantize` decodes it accordingly.  Requires
            frequency-sorted centroids (centroid-0 = most frequent).
        original_shape: Original weight shape, required by
            :meth:`dequantize` to reshape the output.
        column_permutation: Optional int64 tensor of shape ``[K]`` where
            ``K = original_shape[-1]``.  When present, columns of the weight
            matrix are reordered by this permutation before vectorization
            (high-importance features first).  :meth:`dequantize` applies the
            inverse permutation automatically.  ``None`` means no reordering.
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        proxy_selector: ProxySelector,
        original_shape: torch.Size | None = None,
        column_permutation: torch.Tensor | None = None,
        use_tiered: bool = False,
        hot_bits: int | None = None,
        tier_bits: list[int] | None = None,
        zero_escape: bool = False,
    ) -> None:
        """Initialise a VQScheme.

        Args:
            centroids: Codebook tensor ``[num_centroids, vector_len]``.
                Should be fp16 for memory efficiency; any float dtype is
                accepted.
            proxy_selector: Distance metric for centroid assignment.
            original_shape: Weight shape before flattening into vectors.
                Required for :meth:`dequantize`; may be ``None`` when the
                scheme is used only for index assignment.
            column_permutation: Optional int64 permutation of the ``K``
                input columns.  Produced by
                :func:`~neuropress.methods.vptq.hessian_sort_permutation`
                and stored here so that :meth:`quantize_as_u8` and
                :meth:`dequantize` stay consistent.
            use_tiered: If ``True``, use two-tier prefix-coded packing
                (:class:`~neuropress.transforms.vq_packing.TieredIndexPacking`).
                Requires the codebook to be frequency-sorted before any
                :meth:`quantize_as_u8` call.  Ignored when *tier_bits*
                or *zero_escape* is provided.
            hot_bits: Bit width for the hot tier when *use_tiered=True*.
                ``None`` (default) auto-selects: 8 when
                ``num_centroids > 256``, 4 otherwise.  Ignored when
                *tier_bits* or *zero_escape* is provided.
            tier_bits: N-tier CBS bit widths, e.g. ``[4, 8, 12, 17]``.
                When provided, enables n-tier cascaded binary splitting
                via :class:`~neuropress.transforms.vq_packing.NTieredIndexPacking`.
                Overrides *use_tiered* and *hot_bits*.  Must be strictly
                increasing, length 2-4.  Ignored when *zero_escape* is
                provided.
            zero_escape: If ``True``, use the zero-escape 2-bit prefix
                packing via
                :class:`~neuropress.transforms.vq_packing.ZeroEscapePacking`.
                Overrides *use_tiered*, *hot_bits*, and *tier_bits*.
                Requires frequency-sorted centroids and at least 257
                centroids for full benefit (tier-2 covers centroids
                17–256).

        Raises:
            ValueError: If ``centroids`` is not 2-D, has fewer than 2
                rows, or has more than 1M rows.
            ValueError: If ``use_tiered=True`` but
                ``num_centroids < hot_threshold`` (tiered coding requires
                at least 16 centroids to make sense).
            ValueError: If *tier_bits* is invalid.
        """
        if centroids.ndim != 2:
            raise ValueError(
                f"centroids must be 2-D [num_centroids, vector_len], "
                f"got shape {list(centroids.shape)}."
            )
        if centroids.shape[0] < 2:
            raise ValueError(f"num_centroids must be >= 2, got {centroids.shape[0]}.")
        if centroids.shape[0] > IndexPacking._MAX_CENTROIDS:
            raise ValueError(
                f"num_centroids={centroids.shape[0]} exceeds the maximum "
                f"supported value of {IndexPacking._MAX_CENTROIDS}."
            )
        # Zero-escape overrides all other tiered modes.
        if zero_escape:
            use_tiered = False
            tier_bits = None
            hot_bits = 4  # unused but kept for consistency
        # N-tier CBS mode: tier_bits overrides use_tiered/hot_bits.
        elif tier_bits is not None:
            NTieredIndexPacking._validate_tier_bits(tier_bits)
            use_tiered = False  # disable legacy 2-tier path
            hot_bits = tier_bits[0]
        else:
            # Auto-select hot_bits: 8 for large codebooks, 4 otherwise.
            if hot_bits is None:
                hot_bits = 8 if (use_tiered and centroids.shape[0] > 256) else 4
            hot_threshold = TieredIndexPacking.hot_threshold(hot_bits)
            if use_tiered and centroids.shape[0] < hot_threshold:
                raise ValueError(
                    f"use_tiered=True with hot_bits={hot_bits} requires "
                    f"num_centroids >= {hot_threshold}, "
                    f"got {centroids.shape[0]}."
                )
        self.centroids = centroids
        self.proxy_selector = proxy_selector
        self.is_packed_indices: int = IndexPacking.bits_for(centroids.shape[0])
        self.use_tiered: bool = use_tiered
        self.hot_bits: int = hot_bits
        self.tier_bits: list[int] | None = tier_bits
        self.zero_escape: bool = zero_escape
        self.original_shape = original_shape
        self.column_permutation = column_permutation

    # ------------------------------------------------------------------
    # Cached properties
    # ------------------------------------------------------------------

    @cached_property
    def num_centroids(self) -> int:
        """Number of entries in the codebook."""
        return self.centroids.shape[0]

    @cached_property
    def vector_len(self) -> int:
        """Number of weight elements represented by each codebook entry."""
        return self.centroids.shape[1]

    # ------------------------------------------------------------------
    # QScheme interface
    # ------------------------------------------------------------------

    def quantize_as_torch(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Raise — native torch does not support VQ at a tensor level.

        Raises:
            T2NErrorNotImplemented: Always.
        """
        raise T2NErrorNotImplemented(
            "quantize_as_torch is not supported for VQScheme.  "
            "Use quantize_as_u8() to obtain packed indices."
        )

    def quantize_as_u8(self, fp_tensor: torch.Tensor) -> torch.Tensor:
        """Assign each weight vector to its nearest centroid and pack indices.

        Reshapes *fp_tensor* into ``[num_vectors, vector_len]`` vectors,
        computes nearest-centroid distances using :attr:`proxy_selector`,
        and packs the resulting indices via
        :class:`~neuropress.transforms.vq_packing.IndexPacking`.

        Args:
            fp_tensor: Float weight tensor.  Its total element count must be
                divisible by :attr:`vector_len`.

        Returns:
            1-D ``uint8`` packed index buffer.  Unpack with
            :meth:`dequantize` or
            :class:`~neuropress.transforms.vq_packing.IndexPacking.unpack`.

        Raises:
            ValueError: If ``fp_tensor.numel()`` is not divisible by
                :attr:`vector_len`.
        """
        n = fp_tensor.numel()
        if n % self.vector_len != 0:
            raise ValueError(
                f"fp_tensor has {n} elements, not divisible by "
                f"vector_len={self.vector_len}."
            )
        # Apply column permutation (if any) before vectorizing.
        # Permutation reorders the K input features so that high-importance
        # columns are grouped into the same VQ vectors.
        if self.column_permutation is not None:
            perm = self.column_permutation.to(fp_tensor.device)
            K = perm.numel()
            fp_tensor = fp_tensor.reshape(-1, K)[:, perm].reshape(fp_tensor.shape)
        vectors = fp_tensor.reshape(-1, self.vector_len).to(self.centroids.dtype)
        centroids = self.centroids.to(vectors.device)
        # Distance computations require float32; torch.cdist / cosine do not
        # support float16.
        vectors_f32 = vectors.float()
        centroids_f32 = centroids.float()

        # Chunked distance to bound peak memory: a full [N, NC] float32 matrix
        # can be tens of GiB for large-vocabulary layers (e.g. lm_head).
        _CHUNK = 65536
        N = vectors_f32.shape[0]
        if N <= _CHUNK:
            if self.proxy_selector == ProxySelector.L2:
                dists = torch.cdist(vectors_f32, centroids_f32, p=2)
            elif self.proxy_selector == ProxySelector.COSINE:
                dists = _cosine_distance(vectors_f32, centroids_f32)
            else:
                raise T2NErrorNotImplemented(
                    f"ProxySelector {self.proxy_selector!r} is not supported."
                )
            indices = torch.argmin(dists, dim=1)
        else:
            indices_chunks: list[torch.Tensor] = []
            for start in range(0, N, _CHUNK):
                chunk = vectors_f32[start : start + _CHUNK]
                if self.proxy_selector == ProxySelector.L2:
                    d = torch.cdist(chunk, centroids_f32, p=2)
                elif self.proxy_selector == ProxySelector.COSINE:
                    d = _cosine_distance(chunk, centroids_f32)
                else:
                    raise T2NErrorNotImplemented(
                        f"ProxySelector {self.proxy_selector!r} is not supported."
                    )
                indices_chunks.append(torch.argmin(d, dim=1))
            indices = torch.cat(indices_chunks)

        indices = indices  # [num_vectors] int64
        if self.zero_escape:
            return ZeroEscapePacking.pack(indices, self.is_packed_indices)
        if self.tier_bits is not None:
            return NTieredIndexPacking.pack(indices, self.tier_bits)
        if self.use_tiered:
            return TieredIndexPacking.pack(
                indices, self.is_packed_indices, hot_bits=self.hot_bits
            )
        return IndexPacking.pack(indices, self.is_packed_indices)

    def dequantize(
        self, packed_indices: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """Reconstruct the weight tensor from packed indices.

        Unpacks indices, gathers centroid vectors, and reshapes to
        :attr:`original_shape`.

        Args:
            packed_indices: 1-D ``uint8`` buffer produced by
                :meth:`quantize_as_u8`.
            target_dtype: Target float dtype for the output tensor.

        Returns:
            Weight tensor of shape :attr:`original_shape` and dtype
            *target_dtype*.

        Raises:
            RuntimeError: If :attr:`original_shape` is ``None``.
        """
        if self.original_shape is None:
            raise RuntimeError("VQScheme.dequantize requires original_shape to be set.")
        n_vectors = 1
        for d in self.original_shape:
            n_vectors *= d
        n_vectors //= self.vector_len

        if self.zero_escape:
            indices = ZeroEscapePacking.unpack(packed_indices.cpu()).to(
                packed_indices.device
            )
        elif self.tier_bits is not None:
            indices = NTieredIndexPacking.unpack(packed_indices.cpu()).to(
                packed_indices.device
            )
        elif self.use_tiered:
            indices = TieredIndexPacking.unpack(packed_indices.cpu()).to(
                packed_indices.device
            )
        else:
            indices = IndexPacking.unpack(
                packed_indices, self.is_packed_indices, n_indices=n_vectors
            )
        centroids = self.centroids.to(packed_indices.device)
        vectors = centroids[indices]  # [n_vectors, vector_len]
        result = vectors.to(target_dtype).flatten().reshape(self.original_shape)
        # Invert column permutation: the stored indices encode the permuted
        # weight; we must restore the original column order.
        if self.column_permutation is not None:
            perm = self.column_permutation.to(result.device)
            inv_perm = torch.argsort(perm)
            K = perm.numel()
            result = result.reshape(-1, K)[:, inv_perm].reshape(self.original_shape)
        return result

    def __repr__(self) -> str:
        """Return a concise string representation."""
        perm_str = ", perm=True" if self.column_permutation is not None else ""
        if self.zero_escape:
            tiered_str = ", zero_escape=True"
        elif self.tier_bits is not None:
            tiered_str = f", ntier={self.tier_bits}"
        elif self.use_tiered:
            tiered_str = ", tiered=True"
        else:
            tiered_str = ""
        return (
            f"{self.__class__.__name__}("
            f"[{self.num_centroids}, {self.vector_len}], "
            f"bits={self.is_packed_indices}, "
            f"selector={self.proxy_selector.value}"
            f"{perm_str}{tiered_str})"
        )


class QTensorTractVQ(QTensorTract):
    """Quantized tensor backed by a :class:`VQScheme` (codebook + indices).

    Extends :class:`~torch_to_nnef.tensor.quant.qtract.QTensorTract` so
    that VQ-quantized weights are first-class citizens in the neuropress
    pipeline.

    The ``u8_blob`` attribute inherited from ``QTensorTract`` holds the
    packed index buffer produced by :meth:`VQScheme.quantize_as_u8`.
    Dequantization (triggered by ``__torch_function__`` on ``F.linear``
    calls) calls :meth:`VQScheme.dequantize` to reconstruct the float
    weight on the fly.

    NNEF export is **not** currently supported (the VQ binary format is not
    yet finalised).  Use :func:`~neuropress.t2n.qtensors.save_qtensor` /
    :func:`~neuropress.t2n.qtensors.load_qtensor` (safetensors) instead.

    Attributes:
        qscheme: The :class:`VQScheme` shared across all layers in the
            same quantization group.
    """

    qscheme: VQScheme  # type narrowing for mypy / pyright

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Delegate to ``QTensorTract.__init__`` and validate qscheme type."""
        super().__init__(*args, **kwargs)
        if not isinstance(self.qscheme, VQScheme):
            raise TypeError(
                "QTensorTractVQ requires a VQScheme, "
                f"got {type(self.qscheme).__name__}."
            )

    @classmethod
    def from_packed(
        cls,
        indices: torch.Tensor,
        vq: VQScheme,
        dequant_to_dtype: torch.dtype = torch.float16,
    ) -> QTensorTractVQ:
        """Construct from pre-computed centroid indices, bypassing quantize_as_u8.

        This avoids the expensive cdist re-assignment that ``__init__`` would
        trigger via ``VQScheme.quantize_as_u8``.  Valid for all packing
        variants (standard, tiered, zero-escape, n-tiered) because it
        replicates only the packing step, not the distance computation.

        Args:
            indices: ``[N]`` int64 centroid assignments (raw, unpacked).
            vq: The :class:`VQScheme` shared across layers in this group.
            dequant_to_dtype: Weight dtype for dequantization at inference.

        Returns:
            A fully constructed :class:`QTensorTractVQ` instance.
        """
        # Mirror the packing dispatch in VQScheme.quantize_as_u8.
        if vq.zero_escape:
            u8_blob = ZeroEscapePacking.pack(indices, vq.is_packed_indices)
        elif vq.tier_bits is not None:
            u8_blob = NTieredIndexPacking.pack(indices, vq.tier_bits)
        elif vq.use_tiered:
            u8_blob = TieredIndexPacking.pack(
                indices, vq.is_packed_indices, hot_bits=vq.hot_bits
            )
        else:
            u8_blob = IndexPacking.pack(indices, vq.is_packed_indices)

        obj = torch.Tensor.__new__(cls)
        obj.u8_compressors = []
        obj.u8_blob = u8_blob
        obj.qscheme = vq
        obj.dequant_to_dtype = dequant_to_dtype
        obj.nnef_name = None
        obj.requires_grad = False
        return obj

    def write_in_file(
        self,
        dirpath: str | Path,
        label: str,
        inference_target: InferenceTarget | None = None,
    ) -> None:
        """Raise — NNEF export of VQ tensors is not yet implemented.

        Use :func:`~neuropress.t2n.qtensors.save_qtensor` (safetensors)
        for serialization instead.

        Raises:
            T2NErrorNotImplemented: Always.
        """
        raise T2NErrorNotImplemented(
            "QTensorTractVQ.write_in_file: NNEF export of VQ tensors is "
            "not yet implemented.  Use save_qtensor() (safetensors) instead."
        )
