"""NPEmbedding: drop-in replacement for ``nn.Embedding`` with packed QTensor weights.

Three execution backends are available:

``"default"`` (or ``None`` / ``"torch"``)
    The QTensor's ``__torch_function__`` intercepts ``F.embedding`` and calls
    ``decompress()`` (bit-unpack → per-group dequant) on **every** forward
    pass.  Zero migration cost, but materialises the full ``[V, D]`` float
    weight on every call.

``"compile"``
    A ``torch.compile``'d gather-then-dequant path.  Rather than dequantizing
    the entire ``[V, D]`` weight matrix, only the rows selected by the input
    indices are unpacked and dequantized.  Compiled once per process, cached
    across calls.  Works on CPU and CUDA.

``"cuda"``
    Same gather-then-dequant logic as ``"compile"`` but executed as plain
    PyTorch operations on the CUDA device without ``torch.compile`` overhead.
    Useful when avoiding the compilation warmup cost matters more than the
    fused-kernel speedup.

``"auto"``
    Automatically selects the best available backend at construction time.
    For embeddings, GemLite is not applicable (it is a GEMM library), so the
    priority is: ``"cuda"`` when a CUDA device is present (eager PyTorch ops,
    zero warmup), ``"compile"`` on CPU (Inductor fuses the gather + unpack +
    dequant efficiently).

Note:
    ``"gemlite"`` is rejected at construction time for
    :class:`NPEmbedding` — GemLite accelerates dense matrix multiplication
    and has no embedding-lookup primitive.

Supported QTensor types:

    - ``QTensorTractScaleOnly`` (Q4_0 symmetric)
    - ``QTensorScaleZp``        (Q4_1 asymmetric)

Usage::

    from neuropress.nn import NPEmbedding, linearize_qtensor_embeddings

    # Convert all nn.Embedding with QTensor weights in a model:
    linearize_qtensor_embeddings(model, backend="auto")

    # Or wrap a single quantized embedding:
    np_emb = NPEmbedding.from_embedding(embedding_with_qtensor_weight)

    # Switch backend later:
    np_emb.set_backend("cuda")
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backends import Backend, EmbeddingBackend
from .np_linear import (
    _pack_u8,
    _resolve_backend,  # noqa: F401
    _unpack_u8,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend resolution for embeddings
# ---------------------------------------------------------------------------


def _resolve_backend_emb(
    backend: str | EmbeddingBackend | Backend | None,
    nbits: int | None = None,
    group_size: int | None = None,
) -> EmbeddingBackend:
    """Resolve any backend alias to a concrete :class:`EmbeddingBackend`.

    ``"auto"`` resolves to :attr:`EmbeddingBackend.CUDA` on GPU (eager
    PyTorch ops, zero warmup) or :attr:`EmbeddingBackend.COMPILE` on CPU.

    Args:
        backend: An :class:`EmbeddingBackend` member, recognised string
            alias, or ``None``.
        nbits: Unused; kept for API symmetry with
            :func:`~neuropress.nn.np_linear._resolve_backend`.
        group_size: Unused; kept for API symmetry.

    Returns:
        A concrete :class:`EmbeddingBackend` (never ``EmbeddingBackend.AUTO``).

    Raises:
        ValueError: If *backend* is not a valid :class:`EmbeddingBackend`.
    """
    if backend is None or backend == "torch":
        return EmbeddingBackend.DEFAULT
    if isinstance(backend, EmbeddingBackend | Backend):
        val = backend.value
    elif isinstance(backend, str):
        val = backend
    else:
        val = str(backend)
    if val == "auto":
        if torch.cuda.is_available():
            return EmbeddingBackend.CUDA
        return EmbeddingBackend.COMPILE
    return EmbeddingBackend(val)  # raises ValueError for unknown strings


# ---------------------------------------------------------------------------
# Gather-then-dequant helper — used by both "compile" and "cuda" backends
# ---------------------------------------------------------------------------


def _emb_dequant(
    x: torch.Tensor,
    u8_blob: torch.Tensor,
    scale: torch.Tensor,
    nbits: int,
    group_size: int,
    zero_point: torch.Tensor | None,
) -> torch.Tensor:
    """Gather quantized embedding rows by index and dequantize.

    Only the rows selected by *x* are unpacked and dequantized, avoiding
    materialisation of the full ``[V, D]`` float weight matrix.

    Args:
        x: Integer index tensor ``[...]`` (any shape).
        u8_blob: Packed uint8 weights ``[V, D // (8 // nbits)]``.
        scale: Per-group float16 scales, 1-D, length
            ``V * (D // group_size)``.
        nbits: Bits per value — one of ``{1, 2, 4, 8}``.
        group_size: Number of weight elements per group.
        zero_point: Per-group float16 zero-points (asymmetric), same length
            as *scale*, or ``None`` for symmetric dequantization.

    Returns:
        Dequantized embedding tensor ``[..., D]`` with dtype matching *scale*.
    """
    vpb = 8 // nbits
    n_emb = u8_blob.shape[0]
    embedding_dim = u8_blob.shape[1] * vpb
    n_groups = embedding_dim // group_size

    flat_idx = x.reshape(-1)
    n_sel = flat_idx.shape[0]

    # Gather only the rows we need (packed space — much cheaper than full dequant).
    packed = u8_blob[flat_idx]  # [n_sel, D//vpb]
    sc = scale.reshape(n_emb, n_groups)[flat_idx]  # [n_sel, n_groups]

    # Unpack nibbles.
    mask = (1 << nbits) - 1
    shifts = torch.arange(vpb, device=u8_blob.device, dtype=torch.int32) * nbits
    vals = ((packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask).reshape(
        n_sel, embedding_dim
    )  # [n_sel, D]

    # Dequantize in float32, then cast to scale dtype.
    vals_f = vals.to(torch.float32).reshape(n_sel, n_groups, group_size)
    sc_f = sc.to(torch.float32).unsqueeze(-1)  # [n_sel, n_groups, 1]

    if zero_point is not None:
        zp = (
            zero_point.reshape(n_emb, n_groups)[flat_idx]
            .to(torch.float32)
            .unsqueeze(-1)
        )
        out = (vals_f - zp) * sc_f
    else:
        out = (vals_f - float(1 << (nbits - 1))) * sc_f

    return (
        out.reshape(n_sel, embedding_dim)
        .to(scale.dtype)
        .reshape(*x.shape, embedding_dim)
    )


# Single compiled singleton — lazily created on first compile-backend call.
_compiled_emb: object | None = None


def _get_compiled_emb() -> object:
    """Return the compiled gather-dequant function (create on first call)."""
    global _compiled_emb
    if _compiled_emb is None:
        _compiled_emb = torch.compile(_emb_dequant)
    return _compiled_emb


# ---------------------------------------------------------------------------
# NPEmbedding
# ---------------------------------------------------------------------------


class NPEmbedding(nn.Module):
    """Embedding layer with packed n-bit quantized weight and pluggable backend.

    A drop-in replacement for ``nn.Embedding`` whose ``weight`` is a packed
    QTensor.  The forward pass gathers only the requested rows in packed space,
    then dequantizes them — avoiding materialisation of the full float weight
    matrix on every call.

    Supports both symmetric (scale-only) and asymmetric (scale + zero_point)
    group quantization.  Select a backend at construction time; switch later
    with :meth:`set_backend`.

    Attributes:
        num_embeddings: Vocabulary size V.
        embedding_dim: Embedding dimension D.
        group_size: Quantization group size.
        nbits: Bits per quantized value (1, 2, 4, or 8).
        qtype: ``"sym"`` for symmetric or ``"asym"`` for asymmetric.
        backend: Active :class:`EmbeddingBackend` (always concrete, never AUTO).
        padding_idx: Index whose embedding vector is forced to zero, or ``None``.
    """

    num_embeddings: int
    embedding_dim: int
    group_size: int
    nbits: int
    qtype: str
    backend: EmbeddingBackend
    padding_idx: int | None

    def __init__(
        self,
        u8_blob: torch.Tensor,
        scale: torch.Tensor,
        num_embeddings: int,
        embedding_dim: int,
        group_size: int,
        nbits: int,
        qtype: str,
        backend: str | EmbeddingBackend | None,
        padding_idx: int | None = None,
        zero_point: torch.Tensor | None = None,
        weight_q: torch.Tensor | None = None,
        qtensor_meta: dict[str, str] | None = None,
        embed_scale: float = 1.0,
    ) -> None:
        """Initialise NPEmbedding.

        Args:
            u8_blob: Packed uint8 weight ``[num_embeddings, embedding_dim // vpb]``.
            scale: Per-group float16 scales, any shape (flattened to 1-D).
            num_embeddings: Vocabulary size V.
            embedding_dim: Embedding dimension D.
            group_size: Quantization group size.
            nbits: Bits per quantized value.
            qtype: ``"sym"`` (symmetric / scale-only) or
                ``"asym"`` (asymmetric / scale + zero_point).
            backend: ``"default"`` / ``None`` / ``"torch"``, ``"compile"``,
                ``"cuda"``, or ``"auto"``.  ``"gemlite"`` is rejected.
            padding_idx: Optional padding index (embedding zeroed out).
            zero_point: Per-group float16 zero-points for Q4_1 (flattened to 1-D).
            weight_q: Original QTensor, required only for ``"default"`` backend.
            qtensor_meta: Serde metadata dict for QTensor reconstruction
                via :meth:`to_qtensor`.
            embed_scale: Scalar multiplier applied to every output embedding
                row.  Used to replicate the behaviour of custom embedding
                subclasses (e.g. Gemma's ``embed_scale = sqrt(hidden_size)``).
        """
        super().__init__()
        self._qtensor_meta = qtensor_meta
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        self.nbits = nbits
        self.qtype = qtype
        self.padding_idx = padding_idx
        self.embed_scale = embed_scale
        self.backend = _resolve_backend_emb(backend, nbits=nbits, group_size=group_size)

        if self.backend == EmbeddingBackend.DEFAULT:
            if weight_q is None:
                raise ValueError("weight_q is required when backend='default'.")
            object.__setattr__(self, "_weight_q", weight_q)
        else:
            self.register_buffer("u8_blob", u8_blob.contiguous())
            self.register_buffer("scale", scale.reshape(-1).contiguous())
            if zero_point is not None:
                self.register_buffer("zero_point", zero_point.reshape(-1).contiguous())
            else:
                self.zero_point: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        backend: str | EmbeddingBackend | None = EmbeddingBackend.AUTO,
    ) -> NPEmbedding:
        """Build an ``NPEmbedding`` from an ``nn.Embedding`` with a QTensor weight.

        Args:
            embedding: An ``nn.Embedding`` whose ``weight`` attribute is a
                ``QTensorTractScaleOnly`` (Q4_0) or ``QTensorScaleZp`` (Q4_1).
            backend: Target execution backend.  Defaults to ``EmbeddingBackend.AUTO``
                (resolves to ``"compile"`` for embeddings).

        Returns:
            An ``NPEmbedding`` equivalent to *embedding* using the specified
            backend.

        Raises:
            TypeError: If ``embedding.weight`` is not a recognised QTensor type.
        """
        from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

        from ..transforms.weight_only import QTensorScaleZp

        weight_q = embedding.weight
        if isinstance(weight_q, QTensorTractScaleOnly):
            qtype = "sym"
            zero_point = None
        elif isinstance(weight_q, QTensorScaleZp):
            qtype = "asym"
            zero_point = weight_q.qscheme.zero_point.detach().clone()
        else:
            raise TypeError(
                f"Unsupported weight type {type(weight_q).__name__!r}. "
                "Expected QTensorTractScaleOnly (symmetric) or "
                "QTensorScaleZp (asymmetric)."
            )

        bk = _resolve_backend_emb(
            backend,
            nbits=weight_q.qscheme.n_bits,
            group_size=weight_q.qscheme.group_size,
        )

        # Capture serde metadata for later QTensor reconstruction.
        from ..t2n.qtensors import _SERDE_REGISTRY

        serde = _SERDE_REGISTRY[type(weight_q).__name__]
        qtensor_meta = serde.to_metadata(weight_q)

        embed_scale_buf = getattr(embedding, "embed_scale", None)
        embed_scale = (
            float(embed_scale_buf.item()) if embed_scale_buf is not None else 1.0
        )

        # decompress_to_u8() always returns canonical one-value-per-byte data
        # regardless of whether the source QTensor's u8_blob is itself
        # nibble-packed (e.g. via BitPacking4, used by the "q4_0_packed"
        # weight transform) or not (e.g. a QTensor round-tripped through
        # to_qtensor()/from_parts(), which carries no compressor). Repacking
        # ourselves from this canonical form -- rather than assuming
        # weight_q.u8_blob is already in NPEmbedding's packed layout --
        # keeps this correct in both cases.
        u8_blob = _pack_u8(weight_q.decompress_to_u8(), weight_q.qscheme.n_bits)

        return cls(
            u8_blob=u8_blob,
            scale=weight_q.qscheme.scale.detach().clone(),
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            group_size=weight_q.qscheme.group_size,
            nbits=weight_q.qscheme.n_bits,
            qtype=qtype,
            backend=bk,
            padding_idx=embedding.padding_idx,
            zero_point=zero_point,
            weight_q=weight_q if bk == EmbeddingBackend.DEFAULT else None,
            qtensor_meta=qtensor_meta,
            embed_scale=embed_scale,
        )

    # ------------------------------------------------------------------
    # QTensor reconstruction
    # ------------------------------------------------------------------

    def to_qtensor(self) -> torch.Tensor:
        """Reconstruct the original QTensor from this module's buffers.

        Returns:
            A ``QTensorTractScaleOnly`` or ``QTensorScaleZp`` matching the
            original quantized weight.

        Raises:
            RuntimeError: If no serde metadata was stored.
        """
        if self._qtensor_meta is None:
            raise RuntimeError(
                "Cannot reconstruct QTensor: no serde metadata stored. "
                "Build via NPEmbedding.from_embedding() to enable to_qtensor()."
            )
        from ..t2n.qtensors import _SERDE_REGISTRY

        if self.backend == EmbeddingBackend.DEFAULT:
            return self._weight_q

        type_name = "QTensorTractScaleOnly" if self.qtype == "sym" else "QTensorScaleZp"
        serde = _SERDE_REGISTRY[type_name]
        # self.u8_blob is nibble-packed ([V, D // vpb], see class docstring)
        # for GPU-memory efficiency, but the serde's generic dequantize
        # assumes one quantized value per byte (the format nn.Linear weights
        # use, which are never packed) -- unpack before handing off, or every
        # downstream shape is short by the packing factor.
        tensors: dict[str, torch.Tensor] = {
            "u8_blob": _unpack_u8(self.u8_blob, self.nbits),
            "scale": self.scale.reshape(-1, 1),
        }
        if self.zero_point is not None:
            tensors["zero_point"] = self.zero_point.reshape(-1, 1)
        return serde.from_parts(tensors, self._qtensor_meta)

    # ------------------------------------------------------------------
    # Backend switching
    # ------------------------------------------------------------------

    def set_backend(self, backend: str | EmbeddingBackend | None) -> NPEmbedding:
        """Switch to a different backend in-place and return ``self``.

        Args:
            backend: New backend.  ``"auto"`` resolves to ``"compile"``.
                ``"gemlite"`` is rejected.

        Returns:
            ``self`` (for method chaining).

        Raises:
            RuntimeError: If switching to ``"default"`` when the original
                QTensor was not retained.
            RuntimeError: If switching to a kernel backend when ``u8_blob``
                is not available.
            ValueError: If *backend* is not a valid :class:`EmbeddingBackend`.
        """
        bk = _resolve_backend_emb(backend, nbits=self.nbits, group_size=self.group_size)
        if bk == EmbeddingBackend.DEFAULT and not hasattr(self, "_weight_q"):
            raise RuntimeError(
                "Cannot switch to EmbeddingBackend.DEFAULT: the original QTensor was "
                "not retained.  Reconstruct NPEmbedding with backend='default'."
            )
        if bk != EmbeddingBackend.DEFAULT and not hasattr(self, "u8_blob"):
            raise RuntimeError(
                f"Cannot switch to EmbeddingBackend.{bk.name}: u8_blob is not "
                "available.  Reconstruct NPEmbedding with a non-default backend."
            )
        self.backend = bk
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dequantized embeddings for the given indices.

        Args:
            x: Integer index tensor ``[...]`` (any shape).

        Returns:
            Dequantized embedding tensor ``[..., embedding_dim]``.
        """
        if self.backend == EmbeddingBackend.DEFAULT:
            out = F.embedding(x, self._weight_q, padding_idx=self.padding_idx)
        elif self.backend == EmbeddingBackend.COMPILE:
            out = self._forward_compile(x)
        elif self.backend == EmbeddingBackend.CUDA:
            out = self._forward_cuda(x)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")  # pragma: no cover
        if self.embed_scale != 1.0:
            out = out * self.embed_scale
        return out

    def _forward_compile(self, x: torch.Tensor) -> torch.Tensor:
        out = _get_compiled_emb()(
            x,
            self.u8_blob,
            self.scale,
            self.nbits,
            self.group_size,
            self.zero_point,
        )
        return self._apply_padding(out, x)

    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = _emb_dequant(
            x,
            self.u8_blob,
            self.scale,
            self.nbits,
            self.group_size,
            self.zero_point,
        )
        return self._apply_padding(out, x)

    def _apply_padding(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Zero out rows where *x* equals ``padding_idx`` (if set)."""
        if self.padding_idx is not None:
            out = out.clone()
            out[x == self.padding_idx] = 0.0
        return out

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Return a concise parameter summary for ``repr()``."""
        parts = [
            f"num_embeddings={self.num_embeddings}",
            f"embedding_dim={self.embedding_dim}",
            f"nbits={self.nbits}",
            f"group_size={self.group_size}",
            f"qtype={self.qtype}",
            f"backend={self.backend.value}",
        ]
        if self.padding_idx is not None:
            parts.append(f"padding_idx={self.padding_idx}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Model-level transform
# ---------------------------------------------------------------------------


def linearize_qtensor_embeddings(
    model: nn.Module,
    backend: str | EmbeddingBackend | None = EmbeddingBackend.AUTO,
) -> nn.Module:
    """Replace all ``nn.Embedding`` modules with QTensor weights with ``NPEmbedding``.

    Walks *model*, identifies every ``nn.Embedding`` whose ``weight`` is a
    ``QTensorTractScaleOnly`` or ``QTensorScaleZp``, and replaces it with an
    :class:`NPEmbedding` using *backend*.  The model is mutated in-place.

    Modules with unsupported weight types emit a warning and are left unchanged.

    Args:
        model: The ``nn.Module`` to transform.
        backend: Execution backend for all created ``NPEmbedding`` layers.
            Defaults to ``EmbeddingBackend.AUTO`` (resolves to ``"compile"``).

    Returns:
        The same *model* (for chaining).

    Example::

        compress_fn(model, tokenizer=tokenizer)
        linearize_qtensor_embeddings(model, backend="auto")
        linearize_qtensor_weights(model, backend="auto")
        # model now runs inference without QTensor __torch_function__ overhead
    """
    from torch_to_nnef.tensor.quant.base import QTensor
    from torch_to_nnef.tensor.quant.qtract import QTensorTractScaleOnly

    from ..transforms.weight_only import QTensorScaleZp

    _SUPPORTED = (QTensorTractScaleOnly, QTensorScaleZp)

    replacements: list[tuple[str, NPEmbedding]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Embedding) or not isinstance(
            module.weight, QTensor
        ):
            continue
        if not isinstance(module.weight, _SUPPORTED):
            log.debug(
                "linearize_qtensor_embeddings: skipping %s (type=%s)",
                name,
                type(module.weight).__name__,
            )
            continue
        try:
            np_emb = NPEmbedding.from_embedding(module, backend)
        except (TypeError, ValueError) as exc:
            log.warning("linearize_qtensor_embeddings: skipping %s — %s", name, exc)
            continue
        replacements.append((name, np_emb))

    for name, np_emb in replacements:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, np_emb)

    log.info(
        "linearize_qtensor_embeddings: replaced %d layer(s) with NPEmbedding[%s]",
        len(replacements),
        _resolve_backend_emb(backend).value,
    )
    return model
