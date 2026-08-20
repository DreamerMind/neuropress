"""NPVQEmbedding: embedding lookup for VQ-compressed embedding tables.

Pre-unpacks all codebook indices at construction time (vocab × n_groups int16
on CUDA) so each forward pass is just two gather ops + a reshape, with no
bitstream decoding overhead and no CPU→GPU transfers.

Supports the ``embed_scale`` attribute used by
``Gemma3TextScaledWordEmbedding`` (and other scaled embedding classes).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .np_vq_linear import VQProxy

log = logging.getLogger(__name__)


class _WeightProxy:
    """Lazy proxy returned by ``NPVQEmbedding.weight``.

    Supports integer row indexing (``weight[idx]``, ``weight[idx, :]``)
    without materialising the full ``[vocab, embed_dim]`` matrix.
    Models that access ``embed_tokens.weight[pad_token_id]`` (e.g. Gemma4)
    trigger a single-token forward pass instead.
    """

    __slots__ = ("_emb",)

    def __init__(self, emb: NPVQEmbedding) -> None:
        self._emb = emb

    @property
    def shape(self) -> tuple[int, int]:
        return (self._emb.all_indices.shape[0], self._emb.embedding_dim)

    def __getitem__(self, key: object) -> torch.Tensor:
        if isinstance(key, tuple):
            row_key, rest = key[0], key[1:]
        else:
            row_key, rest = key, ()
        scalar = isinstance(row_key, int)
        ids = torch.tensor(
            [row_key] if scalar else list(row_key),
            dtype=torch.long,
            device=self._emb.all_indices.device,
        )
        out = self._emb(ids)  # [1 or N, embed_dim]
        if scalar:
            out = out[0]
        if rest:
            out = out[rest]
        return out


class NPVQEmbedding(nn.Module):
    """VQ-compressed embedding with pre-unpacked indices.

    Stores all codebook indices as ``[vocab_size, n_groups]`` int16 on the
    target device.  Each forward pass gathers centroid rows for the queried
    tokens — no bitstream decoding, no host-device transfers.

    Attributes:
        all_indices: ``[vocab_size, n_groups]`` int16 index table.
        proxy: Shared :class:`~neuropress.nn.np_vq_linear.VQProxy` holding
            the float16 centroid table.
        embedding_dim: Output embedding dimension.
        vector_len: Elements per centroid vector.
        scale: Scalar multiplier applied to every output row (e.g.
            ``sqrt(embedding_dim)`` for Gemma-3).
        inv_perm: Optional ``[embedding_dim]`` int64 inverse-permutation
            tensor.  Present when the VQScheme used column permutation.
        out_dtype: If not ``None``, output is cast to this dtype.
    """

    def __init__(
        self,
        all_indices: torch.Tensor,
        proxy: VQProxy,
        embedding_dim: int,
        vector_len: int,
        scale: float = 1.0,
        inv_perm: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("all_indices", all_indices.to(torch.int16))
        # proxy is NOT a sub-module here; the owner model registers it separately.
        self.proxy = proxy
        self.embedding_dim = embedding_dim
        self.vector_len = vector_len
        self.scale = scale
        if inv_perm is not None:
            self.register_buffer("inv_perm", inv_perm)
        else:
            self.inv_perm: torch.Tensor | None = None
        self.out_dtype = out_dtype
        # Stored by linearize_vq_embeddings so unlower_model / save_quantized_model
        # can reconstruct the original QTensorTractVQ without re-packing indices.
        self._source_qtensor: torch.Tensor | None = None

    def to_qtensor(self) -> torch.Tensor:
        """Return the original QTensorTractVQ stored at lowering time."""
        if self._source_qtensor is None:
            raise RuntimeError(
                "NPVQEmbedding._source_qtensor is None — was this module built "
                "via linearize_vq_embeddings()? Cannot unlower for save."
            )
        return self._source_qtensor

    @property
    def weight(self) -> _WeightProxy:
        """Lazy proxy supporting ``weight[token_id]`` without full materialisation."""
        return _WeightProxy(self)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up *input_ids* by gathering centroids for each token's VQ indices.

        Args:
            input_ids: Integer token ids of any shape ``[...]``.

        Returns:
            Embeddings of shape ``[..., embedding_dim]``.
        """
        leading = input_ids.shape
        flat_ids = input_ids.flatten()  # [T]
        # [T, n_groups] int16 → long for fancy-indexing
        idx = self.all_indices[flat_ids].long()  # [T, n_groups]
        # [T, n_groups, VL] float16
        out = self.proxy.centroids[idx]
        out = out.reshape(flat_ids.shape[0], self.embedding_dim)
        if self.inv_perm is not None:
            out = out[:, self.inv_perm]
        if self.scale != 1.0:
            out = out * self.scale
        if self.out_dtype is not None:
            out = out.to(self.out_dtype)
        return out.reshape(*leading, self.embedding_dim)

    def extra_repr(self) -> str:
        """Return the codebook geometry and any active post-scaling for ``repr()``."""
        scale_str = f", scale={self.scale:.4g}" if self.scale != 1.0 else ""
        perm_str = ", perm=True" if self.inv_perm is not None else ""
        dtype_str = f", out_dtype={self.out_dtype}" if self.out_dtype else ""
        return (
            f"vocab={self.all_indices.shape[0]}, "
            f"dim={self.embedding_dim}, "
            f"vector_len={self.vector_len}, "
            f"num_centroids={self.proxy.num_centroids}"
            f"{scale_str}{perm_str}{dtype_str}"
        )


# ---------------------------------------------------------------------------
# Model-level transform
# ---------------------------------------------------------------------------


def linearize_vq_embeddings(model: nn.Module) -> int:
    """Replace VQ-compressed embedding modules with :class:`NPVQEmbedding`.

    Scans *model* for any module whose ``weight`` attribute is a
    :class:`~neuropress.transforms.vq.QTensorTractVQ` and that is **not** an
    ``nn.Linear`` (linear layers are handled by
    :func:`~neuropress.nn.np_vq_linear.linearize_vq_weights`).

    Handles modules whose class carries an ``embed_scale`` attribute (e.g.
    ``Gemma3TextScaledWordEmbedding``), applying the scale inside
    :class:`NPVQEmbedding` so the forward result is identical.

    Only flat :class:`~neuropress.transforms.vq_packing.IndexPacking` is
    supported.  Tiered / n-tier / zero-escape packing variants are skipped
    with a warning.

    Args:
        model: Model to transform in-place.

    Returns:
        Number of replacements made (0 if none).
    """
    from ..transforms.vq import QTensorTractVQ
    from ..transforms.vq_packing import IndexPacking

    # Reuse VQProxy instances already registered by linearize_vq_weights.
    proxy_cache: dict[int, VQProxy] = {
        id(m.centroids): m for m in model.modules() if isinstance(m, VQProxy)
    }
    new_proxies: dict[int, VQProxy] = {}
    replacements: list[tuple[str, NPVQEmbedding]] = []

    # Detect the model's compute dtype to cast embedding output.
    try:
        out_dtype: torch.dtype | None = next(
            p.dtype
            for p in model.parameters()
            if p.dtype in (torch.float16, torch.bfloat16)
        )
    except StopIteration:
        out_dtype = None

    for name, module in model.named_modules():
        if isinstance(module, (NPVQEmbedding, nn.Linear)):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or not isinstance(weight, QTensorTractVQ):
            continue

        qscheme = weight.qscheme
        if qscheme.use_tiered or qscheme.tier_bits or qscheme.zero_escape:
            log.warning(
                "linearize_vq_embeddings: skipping %s — tiered/ntier/zero-escape "
                "packing not yet supported for embeddings.",
                name,
            )
            continue

        if qscheme.original_shape is None:
            log.warning(
                "linearize_vq_embeddings: skipping %s — qscheme.original_shape is None.",
                name,
            )
            continue
        vocab_size, embed_dim = qscheme.original_shape[0], qscheme.original_shape[1]
        VL = qscheme.vector_len
        n_groups = embed_dim // VL
        bits = qscheme.is_packed_indices
        n_vectors = vocab_size * n_groups

        # Get or create a VQProxy for this centroid table.
        centroids = qscheme.centroids
        cid = id(centroids)
        if cid in proxy_cache:
            proxy = proxy_cache[cid]
        elif cid in new_proxies:
            proxy = new_proxies[cid]
        else:
            proxy = VQProxy(centroids)
            new_proxies[cid] = proxy

        # Pre-unpack all indices → [vocab, n_groups] int16
        indices_flat = IndexPacking.unpack(weight.u8_blob, bits, n_vectors)
        all_indices = indices_flat.reshape(vocab_size, n_groups).to(torch.int16)

        scale = float(getattr(module, "embed_scale", 1.0))

        inv_perm: torch.Tensor | None = None
        if qscheme.column_permutation is not None:
            inv_perm = torch.argsort(qscheme.column_permutation.detach().clone())

        np_emb = NPVQEmbedding(
            all_indices=all_indices,
            proxy=proxy,
            embedding_dim=embed_dim,
            vector_len=VL,
            scale=scale,
            inv_perm=inv_perm,
            out_dtype=out_dtype,
        )
        np_emb._source_qtensor = weight
        replacements.append((name, np_emb))

    if not replacements:
        log.info("linearize_vq_embeddings: no VQ-weight embedding layers found.")
        return 0

    # Register any newly created proxies on the model root.
    existing_proxy_names = {
        name for name, m in model.named_modules() if isinstance(m, VQProxy)
    }
    for i, proxy in enumerate(new_proxies.values()):
        pname = f"_vq_emb_proxy_{i}"
        if pname not in existing_proxy_names:
            model.add_module(pname, proxy)

    # Replace modules in-place.
    for name, np_emb in replacements:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, np_emb)

    log.info("linearize_vq_embeddings: replaced %d layer(s).", len(replacements))
    return len(replacements)
