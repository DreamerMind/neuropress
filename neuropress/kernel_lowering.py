"""KernelLowering: replace QTensor-backed modules with hardware-optimized ones.

Kernel lowering is a distinct pipeline phase that runs *after* weight
assignment and manifest recording.  It replaces generic QTensor-bearing
``nn.Module`` instances (``nn.Linear``, ``nn.Embedding``) with specialised
NP* modules (``NPLinear``, ``NPVQLinear``, ``NPEmbedding``) backed by
optimised kernels (default, torch.compile, custom CUDA, GemLite, …).

The split from the adapter phase ensures that:

1. The manifest records QTensor metadata *before* lowering consumes them.
2. A saved checkpoint with ``kernel_lowering=None`` preserves raw QTensor
   weights that can be re-lowered to any available hardware on load.
3. The pipeline needs no ``isinstance`` checks — the ordering is always
   ``adapter → manifest → kernel_lowering``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

__all__ = [
    "AutoKernelLowering",
    "KernelLowering",
    "LoweringConfig",
    "is_lowered",
    "unlower_model",
]

LOGGER = logging.getLogger(__name__)

_VALID_FIELDS = frozenset(
    {"vq_backend", "linear_backend", "embedding_backend", "norm_backend"}
)


@dataclass
class LoweringConfig:
    """Per-path backend configuration for :class:`AutoKernelLowering`.

    Each field selects the execution backend for one category of quantized
    module.  ``"auto"`` (the default) lets the runtime pick the best available
    option: CUDA JIT when ``nvcc`` is found, ``torch.compile`` otherwise.

    Attributes:
        vq_backend: Backend for VQ-quantized ``nn.Linear`` modules.
        linear_backend: Backend for scalar-quantized ``nn.Linear`` modules.
        embedding_backend: Backend for quantized ``nn.Embedding`` modules.

    Reference configs are stored in ``configs/lowering/`` and can be passed
    directly via ``--lowering-config``.  Sweep entries can also inline the
    config as a YAML mapping under the ``lowering`` key.

    Example YAML::

        vq_backend: cuda        # force CUDA JIT; skip auto-detection
        # linear_backend and embedding_backend default to "auto"
    """

    vq_backend: str = "auto"
    linear_backend: str = "auto"
    embedding_backend: str = "auto"
    norm_backend: str = "auto"

    @classmethod
    def from_dict(cls, d: dict) -> LoweringConfig:
        """Build from a plain dict (e.g. parsed from YAML)."""
        unknown = set(d) - _VALID_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown lowering config key(s): {sorted(unknown)}. "
                f"Valid: {sorted(_VALID_FIELDS)}"
            )
        return cls(**{k: str(v) for k, v in d.items()})

    @classmethod
    def from_file(cls, path: Path | str) -> LoweringConfig:
        """Load from a YAML file."""
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a YAML mapping, got {type(data).__name__}"
            )
        return cls.from_dict(data)

    @classmethod
    def load(cls, src: Path | str | dict | None) -> LoweringConfig:
        """Load from a file path, inline dict, or return defaults when *src* is ``None``."""
        if src is None:
            return cls()
        if isinstance(src, dict):
            return cls.from_dict(src)
        return cls.from_file(src)

    def save(self, path: Path | str) -> None:
        """Write to *path* as a YAML file."""
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as fh:
            yaml.safe_dump(
                dict(asdict(self).items()),
                fh,
                default_flow_style=False,
                sort_keys=True,
            )


class KernelLowering(ABC):
    """Replace QTensor-backed modules with hardware-optimised modules.

    Concrete subclasses implement :meth:`apply` to walk the model tree and
    swap modules in-place.  The method receives no ``QuantizationPlan`` —
    lowering operates purely on the model's current structure and weights.
    """

    @abstractmethod
    def apply(self, model: nn.Module) -> None:
        """Lower *model* in-place to hardware-optimised modules.

        Args:
            model: Model whose QTensor-backed modules will be replaced.
        """


class AutoKernelLowering(KernelLowering):
    """Run all linearize transforms with configurable per-path backends.

    Calls (in order):

    1. :func:`~neuropress.nn.np_embedding.linearize_qtensor_embeddings`
    2. :func:`~neuropress.nn.np_linear.linearize_qtensor_weights`
    3. :func:`~neuropress.nn.np_vq_linear.linearize_vq_weights`

    Each transform is a no-op when no matching weights are found, so this
    lowering pass is safe for any recipe.

    Attributes:
        linear_backend: Backend for scalar-quantized ``nn.Linear``.
        vq_backend: Backend for vector-quantized ``nn.Linear``.
        embedding_backend: Backend for quantized ``nn.Embedding``.
    """

    def __init__(
        self,
        linear_backend: str = "auto",
        vq_backend: str = "auto",
        embedding_backend: str = "auto",
        norm_backend: str = "auto",
    ) -> None:
        """Initialise with per-path backend selections.

        Args:
            linear_backend: Backend for scalar-quantized ``nn.Linear``.
            vq_backend: Backend for vector-quantized ``nn.Linear``.
            embedding_backend: Backend for quantized ``nn.Embedding``.
            norm_backend: Backend for RMSNorm modules.  ``"auto"`` and
                ``"cuda"`` apply the fused single-pass CUDA kernel when JIT
                is available; ``"default"`` skips replacement.
        """
        self.linear_backend = linear_backend
        self.vq_backend = vq_backend
        self.embedding_backend = embedding_backend
        self.norm_backend = norm_backend

    @classmethod
    def from_config(
        cls, src: Path | str | dict | LoweringConfig | None = None
    ) -> AutoKernelLowering:
        """Build from a :class:`LoweringConfig`, file path, inline dict, or ``None`` (all-auto).

        Args:
            src: Config source — a :class:`LoweringConfig` instance, a path to a
                YAML file, an inline dict, or ``None`` for all-``"auto"`` defaults.

        Returns:
            Configured :class:`AutoKernelLowering`.
        """
        if isinstance(src, LoweringConfig):
            cfg = src
        else:
            cfg = LoweringConfig.load(src)
        return cls(
            linear_backend=cfg.linear_backend,
            vq_backend=cfg.vq_backend,
            embedding_backend=cfg.embedding_backend,
            norm_backend=cfg.norm_backend,
        )

    def apply(self, model: nn.Module) -> None:
        """Lower *model* by running all three linearize transforms.

        Args:
            model: Model whose QTensor-backed modules will be replaced.
        """
        from .nn.np_embedding import NPEmbedding, linearize_qtensor_embeddings
        from .nn.np_linear import NPLinear, linearize_qtensor_weights
        from .nn.np_rms_norm import linearize_rms_norms
        from .nn.np_vq_embedding import NPVQEmbedding, linearize_vq_embeddings
        from .nn.np_vq_linear import NPVQLinear, linearize_vq_weights

        linearize_qtensor_embeddings(model, backend=self.embedding_backend)
        linearize_qtensor_weights(model, backend=self.linear_backend)
        linearize_vq_weights(model, backend=self.vq_backend)
        linearize_vq_embeddings(model)

        n_norms = 0
        if self.norm_backend in ("auto", "cuda"):
            n_norms = linearize_rms_norms(model)

        # After lowering, lm_head is replaced by an NP* module and the
        # HuggingFace weight-tie (lm_head.weight is embed_tokens.weight) is
        # physically broken.  Set tie_word_embeddings=False so that any
        # tie_weights() call downstream (e.g. HFLM, generate()) is a no-op
        # rather than trying to re-share tensors that no longer exist.
        if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False

        # Summarise what was lowered.
        counts: dict[str, int] = {}
        for m in model.modules():
            if isinstance(m, NPLinear):
                key = f"NPLinear({m.backend.value})"
                counts[key] = counts.get(key, 0) + 1
            elif isinstance(m, NPVQLinear):
                key = f"NPVQLinear({m.backend.value})"
                counts[key] = counts.get(key, 0) + 1
            elif isinstance(m, NPEmbedding):
                key = f"NPEmbedding({m.backend.value})"
                counts[key] = counts.get(key, 0) + 1
            elif isinstance(m, NPVQEmbedding):
                counts["NPVQEmbedding"] = counts.get("NPVQEmbedding", 0) + 1
        if n_norms:
            counts["NPRMSNorm(cuda)"] = n_norms
        if counts:
            summary = ", ".join(f"{v}x {k}" for k, v in sorted(counts.items()))
            LOGGER.warning("Kernel lowering applied: %s", summary)
        else:
            LOGGER.warning("Kernel lowering: no modules were lowered.")


# ---------------------------------------------------------------------------
# Query / inverse
# ---------------------------------------------------------------------------


def is_lowered(model: nn.Module) -> bool:
    """Return ``True`` if *model* contains any hardware-lowered NP* modules.

    Args:
        model: Model to inspect.

    Returns:
        ``True`` when at least one :class:`~neuropress.nn.np_linear.NPLinear`,
        :class:`~neuropress.nn.np_vq_linear.NPVQLinear`, or
        :class:`~neuropress.nn.np_embedding.NPEmbedding` is found.
    """
    from .nn.np_embedding import NPEmbedding
    from .nn.np_linear import NPLinear
    from .nn.np_vq_linear import NPVQLinear

    return any(
        isinstance(m, NPLinear | NPVQLinear | NPEmbedding) for m in model.modules()
    )


def unlower_model(model: nn.Module) -> None:
    """Replace NP* modules with ``nn.Linear`` / ``nn.Embedding`` + QTensor weights.

    The inverse of :meth:`KernelLowering.apply`.  Each NP* module's
    :meth:`to_qtensor` reconstructs the original QTensor from the module's
    buffers and stored serde metadata.  The QTensor is then placed back into a
    standard ``nn.Linear`` or ``nn.Embedding``.

    Also removes any ``_vq_proxy_*`` modules registered on the model root by
    :func:`~neuropress.nn.np_vq_linear.linearize_vq_weights`.

    Args:
        model: Model to unlower in-place.
    """
    from .nn.np_embedding import NPEmbedding
    from .nn.np_linear import NPLinear
    from .nn.np_vq_embedding import NPVQEmbedding
    from .nn.np_vq_linear import NPVQLinear

    replacements: list[tuple[str, nn.Module]] = []

    for name, module in model.named_modules():
        if isinstance(module, NPLinear | NPVQLinear):
            qtensor = module.to_qtensor()
            has_bias = module.bias is not None
            # Use device="meta" so no weight tensor is allocated — we
            # immediately replace _parameters["weight"] with the real qtensor.
            # Without this, nn.Linear allocates a full fp32 weight on whatever
            # the current default device is, which OOMs for large layers
            # (e.g. lm_head [262144, 8192] = 8 GiB) when the model is on GPU.
            with torch.device("meta"):
                linear = nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=has_bias,
                )
            linear._parameters["weight"] = qtensor
            if has_bias:
                linear._parameters["bias"] = module.bias
            replacements.append((name, linear))
        elif isinstance(module, NPEmbedding):
            qtensor = module.to_qtensor()
            with torch.device("meta"):
                emb = nn.Embedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    padding_idx=module.padding_idx,
                )
            emb._parameters["weight"] = qtensor
            replacements.append((name, emb))
        elif isinstance(module, NPVQEmbedding):
            qtensor = module.to_qtensor()
            with torch.device("meta"):
                emb = nn.Embedding(
                    module.all_indices.shape[0],
                    module.embedding_dim,
                )
            emb._parameters["weight"] = qtensor
            replacements.append((name, emb))

    for name, new_mod in replacements:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, attr = model, name
        setattr(parent, attr, new_mod)

    # Remove VQ proxy modules registered on the model root (both linear and embedding).
    proxy_names = [
        n
        for n, _ in model.named_children()
        if n.startswith("_vq_proxy_") or n.startswith("_vq_emb_proxy_")
    ]
    for pname in proxy_names:
        delattr(model, pname)

    LOGGER.info(
        "unlower_model: restored %d module(s) to QTensor form",
        len(replacements),
    )
