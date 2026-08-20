"""Generic QTensor serialization and deserialization via safetensors.

Provides a registration-based framework for serializing any QTensor subclass
to the safetensors format — deterministic, header-readable, and pickle-free.

Each concrete handler registers how to decompose a QTensor into:

- **Tensors** — stored as safetensors entries (dtype and shape preserved).
- **Metadata** — stored as safetensors string metadata (key → value, both str).

The framework ships with a built-in handler for
:class:`~torch_to_nnef.tensor.quant.qtract.QTensorTractScaleOnly` (Q4_0).

Usage::

    from neuropress.t2n.qtensors import save_qtensor, load_qtensor

    save_qtensor(q, "checkpoints/layer.safetensors")
    q_loaded = load_qtensor("checkpoints/layer.safetensors")

Registering a handler for a custom QTensor type::

    from neuropress.t2n.qtensors import QTensorSerde, register_qtensor_serde

    class MyQTensorSerde(QTensorSerde):
        def type_name(self) -> str:
            return "MyQTensor"

        def to_tensors(self, q) -> dict[str, torch.Tensor]:
            return {"data": q.packed_data}

        def to_metadata(self, q) -> dict[str, str]:
            return {"bits": str(q.n_bits)}

        def from_parts(self, tensors, metadata):
            return MyQTensor(tensors["data"], int(metadata["bits"]))

    register_qtensor_serde(MyQTensorSerde())
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Shared-proxy cache for VQ centroids
# ---------------------------------------------------------------------------
# When multiple QTensorTractVQ layers share the same codebook, loading them
# from disk would normally produce independent tensor objects.  This cache
# maps the SHA-256 of the centroid bytes to the tensor so that all layers
# with the same content reuse the same Python object — restoring the shared-
# proxy invariant that linearize_vq_weights() depends on.
#
# WeakValueDictionary: entries are evicted automatically once no QTensorTractVQ
# holds a reference, so the cache never keeps centroids alive on its own.
_vq_centroid_cache: WeakValueDictionary[str, torch.Tensor] = WeakValueDictionary()


def _centroid_hash(centroids: torch.Tensor) -> str:
    """Return the SHA-256 hex digest of *centroids* raw bytes (fp16, CPU)."""
    raw = centroids.to(torch.float16).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


# Reserved metadata key that records the QTensor type for deserialization.
_TYPE_KEY = "__qtensor_type__"


# ---------------------------------------------------------------------------
# Abstract serde interface
# ---------------------------------------------------------------------------


class QTensorSerde(ABC):
    """Abstract handler for serializing one QTensor subtype.

    Implement :meth:`type_name`, :meth:`to_tensors`, :meth:`to_metadata`, and
    :meth:`from_parts`, then pass an instance to :func:`register_qtensor_serde`.
    """

    @abstractmethod
    def type_name(self) -> str:
        """Return the type identifier — must match ``type(q).__name__``."""

    @abstractmethod
    def to_tensors(self, q: Any) -> dict[str, torch.Tensor]:
        """Decompose *q* into named tensors for safetensors storage."""

    @abstractmethod
    def to_metadata(self, q: Any) -> dict[str, str]:
        """Extract non-tensor metadata from *q* as string key→value pairs."""

    @abstractmethod
    def from_parts(
        self,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
    ) -> Any:
        """Reconstruct a QTensor from *tensors* and *metadata*."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SERDE_REGISTRY: dict[str, QTensorSerde] = {}


def register_qtensor_serde(serde: QTensorSerde) -> None:
    """Register *serde* under its :meth:`~QTensorSerde.type_name`.

    Later registrations override earlier ones for the same type name.

    Args:
        serde: A fully-constructed :class:`QTensorSerde` instance.
    """
    _SERDE_REGISTRY[serde.type_name()] = serde


# ---------------------------------------------------------------------------
# Public save / load API
# ---------------------------------------------------------------------------


def save_qtensor(q: Any, path: str | Path) -> None:
    """Serialize *q* to a safetensors file at *path*.

    The QTensor type name is stored under the reserved ``__qtensor_type__``
    metadata key so :func:`load_qtensor` can reconstruct the correct type.

    Args:
        q: A QTensor instance whose type has a registered serde handler.
        path: Destination ``.safetensors`` file path.

    Raises:
        ValueError: If no serde is registered for ``type(q).__name__``.
    """
    type_name = type(q).__name__
    serde = _SERDE_REGISTRY.get(type_name)
    if serde is None:
        raise ValueError(
            f"No QTensorSerde registered for type {type_name!r}.  "
            f"Register one via register_qtensor_serde().  "
            f"Available: {sorted(_SERDE_REGISTRY)}"
        )
    tensors = serde.to_tensors(q)
    metadata = {**serde.to_metadata(q), _TYPE_KEY: type_name}
    save_file(tensors, path, metadata=metadata)


def load_qtensor(path: str | Path) -> Any:
    """Deserialize a QTensor from the safetensors file at *path*.

    Reads the ``__qtensor_type__`` metadata key to select the correct serde
    handler, then delegates reconstruction to :meth:`QTensorSerde.from_parts`.

    Args:
        path: Path to a ``.safetensors`` file written by :func:`save_qtensor`.

    Returns:
        A QTensor instance of the type recorded in the file's metadata.

    Raises:
        ValueError: If the file lacks ``__qtensor_type__`` or its type has no
            registered handler.
    """
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    type_name = metadata.get(_TYPE_KEY)
    if type_name is None:
        raise ValueError(
            f"Safetensors file {path!r} is missing the {_TYPE_KEY!r} metadata "
            "key.  Was it written by save_qtensor()?"
        )
    serde = _SERDE_REGISTRY.get(type_name)
    if serde is None:
        raise ValueError(
            f"No QTensorSerde registered for type {type_name!r}.  "
            f"Available: {sorted(_SERDE_REGISTRY)}"
        )
    return serde.from_parts(tensors, metadata)


# ---------------------------------------------------------------------------
# Built-in: QTensorTractScaleOnly (Q4_0)
# ---------------------------------------------------------------------------


class _QTractScaleOnlySerde(QTensorSerde):
    """Serde for :class:`~torch_to_nnef.tensor.quant.qtract.QTensorTractScaleOnly`.

    Safetensors entries:

    =========  ===============  ============================================
    Key        Dtype            Description
    =========  ===============  ============================================
    u8_blob    uint8            Packed 4-bit weights
    scale      float16          Per-group scale factors ``[n_groups, 1]``
    =========  ===============  ============================================

    Metadata keys: ``n_bits``, ``group_size``, ``dequant_to_dtype``,
    ``nnef_name`` (empty string when absent).

    Reconstruction strategy: dequantize ``u8_blob`` with the loaded scale to
    recover an approximate float tensor, then re-quantize with the same scale.
    Because the scale is already calibrated this produces a bitwise-identical
    ``u8_blob`` without requiring the original float weight.
    """

    def type_name(self) -> str:
        """Return ``"QTensorTractScaleOnly"``."""
        return "QTensorTractScaleOnly"

    def to_tensors(self, q: Any) -> dict[str, torch.Tensor]:
        """Extract ``u8_blob`` and ``scale`` from *q*."""
        return {
            "u8_blob": q.u8_blob,
            "scale": q.qscheme.scale,
        }

    def to_metadata(self, q: Any) -> dict[str, str]:
        """Extract quantization parameters as string metadata."""
        return {
            "n_bits": str(q.qscheme.n_bits),
            "group_size": str(q.qscheme.group_size),
            "dequant_to_dtype": str(q.dequant_to_dtype).replace("torch.", ""),
            "nnef_name": q.nnef_name or "",
        }

    def from_parts(
        self,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
    ) -> Any:
        """Reconstruct a ``QTensorTractScaleOnly`` from serialized parts."""
        from torch_to_nnef.tensor.quant.qtract import (
            QScalePerGroupF16,
            QTensorTractScaleOnly,
        )

        scale = tensors["scale"]
        u8_blob = tensors["u8_blob"]
        n_bits = int(metadata["n_bits"])
        group_size = int(metadata["group_size"])
        dtype = getattr(torch, metadata["dequant_to_dtype"])

        qscheme = QScalePerGroupF16(group_size=group_size, scale=scale, n_bits=n_bits)
        # Dequantize u8_blob back to float, then re-quantize with the same
        # calibrated scale — yields bitwise-identical u8_blob.
        fp_approx = qscheme.dequantize(u8_blob, dtype)
        q = QTensorTractScaleOnly(fp_approx, qscheme, dequant_to_dtype=dtype)

        nnef_name = metadata.get("nnef_name")
        if nnef_name:
            q.nnef_name = nnef_name
        return q


# Register the built-in handler on module import.
register_qtensor_serde(_QTractScaleOnlySerde())


# ---------------------------------------------------------------------------
# Built-in: QTensorScaleZp (Q4_1 asymmetric)
# ---------------------------------------------------------------------------


class _QTensorScaleZpSerde(QTensorSerde):
    """Serde for :class:`~neuropress.transforms.weight_only.QTensorScaleZp`.

    Safetensors entries:

    =========  ===============  =============================================
    Key        Dtype            Description
    =========  ===============  =============================================
    u8_blob    uint8            Packed 4-bit weights
    scale      float16          Per-group scale factors ``[n_groups, 1]``
    zero_point float16          Per-group zero-points ``[n_groups, 1]``
    =========  ===============  =============================================

    Metadata keys: ``n_bits``, ``group_size``, ``dequant_to_dtype``,
    ``nnef_name`` (empty string when absent).

    Reconstruction strategy: dequantize ``u8_blob`` with the loaded
    scale/zero_point to recover an approximate float tensor, then re-quantize
    with the same calibrated parameters — yields a bitwise-identical
    ``u8_blob``.
    """

    def type_name(self) -> str:
        """Return ``"QTensorScaleZp"``."""
        return "QTensorScaleZp"

    def to_tensors(self, q: Any) -> dict[str, torch.Tensor]:
        """Extract ``u8_blob``, ``scale``, and ``zero_point`` from *q*."""
        return {
            "u8_blob": q.u8_blob,
            "scale": q.qscheme.scale,
            "zero_point": q.qscheme.zero_point,
        }

    def to_metadata(self, q: Any) -> dict[str, str]:
        """Extract quantization parameters as string metadata."""
        return {
            "n_bits": str(q.qscheme.n_bits),
            "group_size": str(q.qscheme.group_size),
            "dequant_to_dtype": str(q.dequant_to_dtype).replace("torch.", ""),
            "nnef_name": q.nnef_name or "",
        }

    def from_parts(
        self,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
    ) -> Any:
        """Reconstruct a ``QTensorScaleZp`` from serialized parts."""
        from ..transforms.weight_only import QScaleZpPerGroupF16, QTensorScaleZp
        from .packing import BitPacking4

        scale = tensors["scale"]
        zero_point = tensors["zero_point"]
        u8_blob = tensors["u8_blob"]
        n_bits = int(metadata["n_bits"])
        group_size = int(metadata["group_size"])
        dtype = getattr(torch, metadata["dequant_to_dtype"])

        qscheme = QScaleZpPerGroupF16(
            group_size=group_size, scale=scale, zero_point=zero_point, n_bits=n_bits
        )
        # Dequantize u8_blob back to float, then re-quantize with the same
        # calibrated parameters — yields bitwise-identical u8_blob.
        fp_approx = qscheme.dequantize(u8_blob, dtype)
        q = QTensorScaleZp(
            fp_approx, qscheme, dequant_to_dtype=dtype, u8_compressors=[BitPacking4]
        )

        nnef_name = metadata.get("nnef_name")
        if nnef_name:
            q.nnef_name = nnef_name
        return q


register_qtensor_serde(_QTensorScaleZpSerde())


# ---------------------------------------------------------------------------
# Built-in: QTensorTractVQ (Vector Quantization)
# ---------------------------------------------------------------------------


class _QTensorTractVQSerde(QTensorSerde):
    """Serde for :class:`~neuropress.transforms.vq.QTensorTractVQ`.

    Safetensors entries:

    ===================  ===============  =============================================
    Key                  Dtype            Description
    ===================  ===============  =============================================
    u8_blob              uint8            Packed VQ codebook indices
    centroids            float16          Codebook table ``[num_centroids, vector_len]``
    column_permutation   int64            Optional; present only when permutation set
    ===================  ===============  =============================================

    Metadata keys: ``proxy_selector``, ``original_shape``, ``bits_per_index``,
    ``dequant_to_dtype``, ``proxy_content_hash``, ``has_column_perm``,
    ``use_tiered``.

    Reconstruction strategy: indices are stored verbatim in ``u8_blob``;
    centroids are stored as-is.  On load, ``VQScheme`` is reconstructed
    from centroids + metadata, then ``QTensorTractVQ`` is built by
    dequantizing and re-quantizing (which yields a bitwise-identical
    ``u8_blob`` because the same centroids and packed indices are used).
    """

    def type_name(self) -> str:
        """Return ``"QTensorTractVQ"``."""
        return "QTensorTractVQ"

    def to_tensors(self, q: Any) -> dict[str, torch.Tensor]:
        """Extract ``u8_blob``, ``centroids``, and optional ``column_permutation``."""
        # .clone() gives each layer its own storage so safetensors does not
        # reject the file for shared-memory tensors.  On load the
        # _vq_centroid_cache deduplicates identical centroids back to a single
        # shared tensor, restoring the shared-proxy invariant.
        tensors: dict[str, torch.Tensor] = {
            "u8_blob": q.u8_blob.cpu(),
            "centroids": q.qscheme.centroids.to(torch.float16).cpu().clone(),
        }
        if q.qscheme.column_permutation is not None:
            tensors["column_permutation"] = q.qscheme.column_permutation.cpu()
        return tensors

    def to_metadata(self, q: Any) -> dict[str, str]:
        """Extract VQ quantization parameters as string metadata."""
        original_shape = q.qscheme.original_shape
        shape_str = ",".join(str(d) for d in original_shape) if original_shape else ""
        centroids_fp16 = q.qscheme.centroids.to(torch.float16)
        return {
            "proxy_selector": q.qscheme.proxy_selector.value,
            "original_shape": shape_str,
            "bits_per_index": str(q.qscheme.is_packed_indices),
            "dequant_to_dtype": str(q.dequant_to_dtype).replace("torch.", ""),
            "proxy_content_hash": _centroid_hash(centroids_fp16),
            "has_column_perm": "1" if q.qscheme.column_permutation is not None else "0",
            "use_tiered": "1" if q.qscheme.use_tiered else "0",
            "zero_escape": "1" if q.qscheme.zero_escape else "0",
        }

    def from_parts(
        self,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, str],
    ) -> Any:
        """Reconstruct a ``QTensorTractVQ`` from serialized parts.

        Shared-proxy deduplication: if multiple layers were saved with the
        same centroid content hash, ``from_parts`` reuses the same Python
        tensor object so that :func:`~neuropress.nn.np_vq_linear.linearize_vq_weights`
        correctly detects the sharing relationship.
        """
        from ..transforms.vq import ProxySelector, QTensorTractVQ, VQScheme

        # ------------------------------------------------------------------
        # Shared-proxy deduplication (Phase A)
        # ------------------------------------------------------------------
        content_hash = metadata.get("proxy_content_hash")
        raw_centroids = tensors["centroids"]
        if content_hash is not None:
            cached = _vq_centroid_cache.get(content_hash)
            if cached is not None:
                centroids = cached
            else:
                centroids = raw_centroids
                _vq_centroid_cache[content_hash] = centroids
        else:
            centroids = raw_centroids

        u8_blob = tensors["u8_blob"]
        dtype = getattr(torch, metadata["dequant_to_dtype"])

        selector = ProxySelector(metadata["proxy_selector"])
        shape_str = metadata.get("original_shape", "")
        original_shape = (
            torch.Size(int(d) for d in shape_str.split(",")) if shape_str else None
        )
        column_permutation = (
            tensors["column_permutation"]
            if metadata.get("has_column_perm") == "1"
            else None
        )
        use_tiered = metadata.get("use_tiered", "0") == "1"
        zero_escape = metadata.get("zero_escape", "0") == "1"

        qscheme = VQScheme(
            centroids=centroids,
            proxy_selector=selector,
            original_shape=original_shape,
            column_permutation=column_permutation,
            use_tiered=use_tiered,
            zero_escape=zero_escape,
        )
        # Construct QTensorTractVQ directly from the stored u8_blob without
        # dequantizing and re-quantizing. The dequantize → re-quantize round
        # trip is bitwise-identical but materialises the full fp32/fp16 weight
        # on GPU (up to 8 GiB for lm_head), causing OOM on 24 GB cards.
        obj = torch.Tensor.__new__(QTensorTractVQ)
        obj.u8_compressors = []
        obj.u8_blob = u8_blob
        obj.qscheme = qscheme
        obj.dequant_to_dtype = dtype
        obj.nnef_name = None
        obj.requires_grad = False
        return obj


register_qtensor_serde(_QTensorTractVQSerde())


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def dequantize_qtensors(model: torch.nn.Module) -> torch.nn.Module:
    """Replace every QTensor parameter with its dequantized float equivalent.

    QTensors intercept all torch operations via ``__torch_function__`` and
    decompress on **every call** (bit-unpack + per-group scale multiply).
    For a 1024×1024 Q4_0 weight this adds ~0.44 ms per forward pass, making
    single-token autoregressive decode up to 84× slower than full-precision.

    This function eagerly dequantizes all QTensor weights once and replaces
    them with plain ``nn.Parameter`` float tensors.  Memory usage increases
    (packed 4-bit → float16/float32), but inference runs at native speed.

    Call this right after quantization and before running lm-eval::

        compress_fn(model, tokenizer=tokenizer)
        dequantize_qtensors(model)   # ← one-time cost, then fast inference

    The model is mutated in-place; the same object is returned for chaining.

    Args:
        model: Any ``nn.Module`` whose parameters may contain QTensor weights.

    Returns:
        The same *model* with all QTensor parameters replaced by plain tensors.
    """
    from torch_to_nnef.tensor.quant.base import QTensor

    for module in model.modules():
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, QTensor):
                fp = param.decompress().detach()
                setattr(module, param_name, torch.nn.Parameter(fp, requires_grad=False))
    return model
