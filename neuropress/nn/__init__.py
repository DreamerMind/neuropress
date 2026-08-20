"""nn sub-package: enhanced ``nn.Module`` building blocks for quantized inference.

Provides drop-in replacements for ``nn.Linear`` and ``nn.Embedding`` with
packed QTensor weights, and model-level transforms to convert all qualifying
modules in one call.

Backends available in :class:`~neuropress.nn.np_linear.NPLinear`:

    ``"default"``   — QTensor ``__torch_function__`` path (no code change).
    ``"compile"``   — ``torch.compile``'d dequant + ``F.linear``.
    ``"cuda"``      — fused CUDA kernel (JIT-built on first use).
    ``"gemlite"``   — Triton kernels via ``gemlite`` (requires CUDA + gemlite).
    ``"auto"``      — best available per layer: gemlite > cuda > compile.

Backends available in :class:`~neuropress.nn.np_embedding.NPEmbedding`:

    ``"default"``   — QTensor ``__torch_function__`` path (no code change).
    ``"compile"``   — ``torch.compile``'d gather-then-dequant.
    ``"cuda"``      — gather-then-dequant without compile overhead.
    ``"auto"``      — always resolves to ``"compile"`` (gemlite n/a).

Backends available in :class:`~neuropress.nn.np_vq_linear.NPVQLinear`:

    ``"default"``   — index unpack + centroid gather + ``F.linear`` (eager).
    ``"compile"``   — ``torch.compile``'d gather + ``F.linear``.
    ``"cuda"``      — fused CUDA kernel (JIT-built on first use).
    ``"auto"``      — CUDA when available, else compile.
"""

from .backends import Backend, EmbeddingBackend, LinearBackend, NormBackend, VQBackend
from .np_embedding import NPEmbedding, linearize_qtensor_embeddings
from .np_linear import NPLinear, linearize_qtensor_weights
from .np_rms_norm import NPRMSNorm, linearize_rms_norms
from .np_vq_embedding import NPVQEmbedding, linearize_vq_embeddings
from .np_vq_linear import NPVQLinear, VQProxy, linearize_vq_weights

__all__ = [
    "Backend",
    "EmbeddingBackend",
    "LinearBackend",
    "NPEmbedding",
    "NPLinear",
    "NPRMSNorm",
    "NPVQEmbedding",
    "NPVQLinear",
    "NormBackend",
    "VQBackend",
    "VQProxy",
    "linearize_qtensor_embeddings",
    "linearize_qtensor_weights",
    "linearize_rms_norms",
    "linearize_vq_embeddings",
    "linearize_vq_weights",
]
