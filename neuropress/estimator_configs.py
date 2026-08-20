"""Generic estimator configuration classes for the modular recipe framework.

Each class is self-registered via :func:`.recipe_config.estimator_config` so
that :meth:`.QuantizationRecipeConfig.from_json` can resolve them by name::

    {"estimator": {"type": "gptq", "target": "linear", "nbits": 4}}

The ``target`` field accepts a module class name (``"linear"``, ``"conv2d"``,
``"embedding"``) that is mapped to the corresponding :mod:`torch.nn` class.

Available estimator types:

* ``"gptq"`` → :class:`.GPTQEstimator`
* ``"gptq_panel"`` → :class:`.GPTQPanelEstimator`
* ``"hqq"`` → :class:`.HQQEstimator`
* ``"min_max_zp_weight"`` → :class:`.MinMaxZpWeightEstimator`
* ``"composite"`` → :class:`.CompositeEstimator` (defined in :mod:`.recipe_config`)

Weight-transform plug-ins (e.g. tract Q4_0) are registered in
:data:`WEIGHT_TRANSFORM_REGISTRY` by external modules and referenced by name
in :class:`GptqEstimatorConfig` via the ``weight_transform`` field::

    {"type": "gptq", "weight_transform": "q4_0", "group_size": 32}
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import Field
from torch import nn

from .estimators import (
    GPTQEstimator,
    GPTQPanelEstimator,
    HQQEstimator,
    MinMaxZpWeightEstimator,
    VPTQGroupEstimator,
)
from .methods.correction_mode import CorrectionMode
from .recipe_config import EstimatorConfig, estimator_config

# ---------------------------------------------------------------------------
# Single source of truth for VPTQGroupEstimator field defaults.
# VptqGroupEstimatorConfig derives its defaults from VPTQGroupEstimator so
# there is never a silent divergence between the two classes.
# ---------------------------------------------------------------------------
_VPTQ_FIELD_DEFAULTS: dict[str, Any] = {
    f.name: f.default
    for f in dataclasses.fields(VPTQGroupEstimator)
    if f.default is not dataclasses.MISSING
}


def _vd(name: str) -> Any:
    """Return VPTQGroupEstimator's canonical default for *name*."""
    return _VPTQ_FIELD_DEFAULTS[name]


# ---------------------------------------------------------------------------
# Module class lookup table
# ---------------------------------------------------------------------------

_MODULE_CLASS_MAP: dict[str, type[nn.Module]] = {
    "linear": nn.Linear,
    "conv2d": nn.Conv2d,
    "embedding": nn.Embedding,
}


def _resolve_module_class(target: str) -> type[nn.Module]:
    """Return the :mod:`torch.nn` class for *target*.

    Args:
        target: One of ``"linear"``, ``"conv2d"``, ``"embedding"``.

    Returns:
        The corresponding :class:`torch.nn.Module` subclass.

    Raises:
        ValueError: If *target* is not in :data:`_MODULE_CLASS_MAP`.
    """
    mod_cls = _MODULE_CLASS_MAP.get(target)
    if mod_cls is None:
        raise ValueError(
            f"Unknown target module {target!r}. Available: {sorted(_MODULE_CLASS_MAP)}"
        )
    return mod_cls


# ---------------------------------------------------------------------------
# Weight-transform registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightTransformEntry:
    """Registry entry pairing a weight-transform callable with an optional filter.

    Attributes:
        transform: Callable ``(weight, name) -> value | None`` applied to the
            quantization-corrected weight after each layer.  Return *None* to
            skip a layer from the plan.
        filter_fn: Optional extra filter ``(name, module) -> bool`` applied in
            addition to the estimator's ``module_classes`` check.  Use this to
            enforce backend-specific shape constraints (e.g. group-size
            divisibility for tract Q4_0).
        storage_bits: Effective bits per weight on disk, *including* all
            quantization-metadata overhead (scales, zero-points).  For example,
            4-bit packed weights with one f16 scale per 32-element group cost
            4 + 16/32 = 4.5 effective bpw, so ``storage_bits=4.5``.  ``None``
            means unknown; size estimation falls back to the estimator's
            quantization ``nbits``.
        group_size: Per-group quantization group size used by this transform
            (e.g. 32 for all Q4_0 variants).  Used by module filters to
            enforce divisibility constraints.  ``None`` means the transform
            uses per-channel granularity.
    """

    transform: Callable[[torch.Tensor, str], Any | None]
    filter_fn: Callable[[str, nn.Module], bool] | None = None
    storage_bits: float | None = None
    group_size: int | None = None


WEIGHT_TRANSFORM_REGISTRY: dict[str, WeightTransformEntry] = {}
"""Registry mapping weight-transform keys to :class:`WeightTransformEntry` instances.

External modules populate this dict before calling
:meth:`GptqEstimatorConfig.build`.  The built-in tract Q4_X transforms are
registered by :mod:`neuropress.transforms.weight_only` on import::

    from neuropress.estimator_configs import (
        WEIGHT_TRANSFORM_REGISTRY, WeightTransformEntry)

    WEIGHT_TRANSFORM_REGISTRY["my_format"] = WeightTransformEntry(
        transform=my_transform_fn,
        filter_fn=my_filter_fn,
        storage_bits=4.5,
        group_size=32,
    )
"""


# ---------------------------------------------------------------------------
# Concrete estimator configs
# ---------------------------------------------------------------------------


@estimator_config("gptq")
class GptqEstimatorConfig(EstimatorConfig):
    """JSON-serialisable config for :class:`.GPTQEstimator`.

    Attributes:
        type: Discriminator field, always ``"gptq"``.
        target: Module class to quantize (``"linear"``, ``"conv2d"``,
            ``"embedding"``).
        group_size: Preferred group size for per-group granularity.
        nbits: Quantization bit-width.
        symmetric: Whether to use symmetric quantization.
        damp_percent: Initial Hessian dampening fraction.
        damp_auto_increment: Dampening increment on Cholesky failure.
        weight_transform: Optional key into :data:`WEIGHT_TRANSFORM_REGISTRY`.
            When set, the matching :class:`WeightTransformEntry` supplies both
            the ``weight_transform`` callable (stored in the plan) and the
            optional ``filter_fn`` (extra module-selection constraint).
    """

    type: Literal["gptq"] = "gptq"
    target: str = "linear"
    group_size: int = 128
    nbits: int = 4
    symmetric: bool = True
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025
    weight_transform: str | None = None

    def build(self) -> GPTQEstimator:
        """Build a :class:`.GPTQEstimator` for the configured target class.

        Returns:
            A :class:`.GPTQEstimator` targeting the module class named by
            :attr:`target`, optionally wired with a weight-transform from
            :data:`WEIGHT_TRANSFORM_REGISTRY`.

        Raises:
            ValueError: If :attr:`weight_transform` is set but not registered.
        """
        entry: WeightTransformEntry | None = None
        if self.weight_transform is not None:
            entry = WEIGHT_TRANSFORM_REGISTRY.get(self.weight_transform)
            if entry is None:
                raise ValueError(
                    f"Unknown weight transform {self.weight_transform!r}. "
                    f"Available: {sorted(WEIGHT_TRANSFORM_REGISTRY)}"
                )
        return GPTQEstimator(
            module_classes=(_resolve_module_class(self.target),),
            filter_fn=entry.filter_fn if entry is not None else None,
            weight_transform=entry.transform if entry is not None else None,
            group_size=self.group_size,
            nbits=self.nbits,
            symmetric=self.symmetric,
            damp_percent=self.damp_percent,
            damp_auto_increment=self.damp_auto_increment,
        )


@estimator_config("gptq_panel")
class GptqPanelEstimatorConfig(EstimatorConfig):
    """JSON-serialisable config for :class:`.GPTQPanelEstimator`.

    Attributes:
        type: Discriminator field, always ``"gptq_panel"``.
        target: Module class to quantize (``"linear"``, ``"conv2d"``,
            ``"embedding"``).
        group_size: Column-group size for panel scoring and quantization.
        possible_nbits: Candidate bit-widths (must contain at least two).
        target_bpw: Average bits-per-weight budget for mixed allocation.
        symmetric: Whether to use symmetric quantization.
        damp_percent: Initial Hessian dampening fraction.
        damp_auto_increment: Dampening increment on Cholesky failure.
    """

    type: Literal["gptq_panel"] = "gptq_panel"
    target: str = "conv2d"
    group_size: int = 32
    possible_nbits: list[int] = Field(default=[4, 8])
    target_bpw: float = 5.0
    symmetric: bool = True
    damp_percent: float = 0.01
    damp_auto_increment: float = 0.0025

    def build(self) -> GPTQPanelEstimator:
        """Build a :class:`.GPTQPanelEstimator` for the configured target class.

        Returns:
            A :class:`.GPTQPanelEstimator` targeting the module class named by
            :attr:`target`.
        """
        return GPTQPanelEstimator(
            module_classes=(_resolve_module_class(self.target),),
            group_size=self.group_size,
            possible_nbits=self.possible_nbits,
            target_bpw=self.target_bpw,
            symmetric=self.symmetric,
            damp_percent=self.damp_percent,
            damp_auto_increment=self.damp_auto_increment,
        )


@estimator_config("min_max_zp_weight")
class MinMaxZpWeightEstimatorConfig(EstimatorConfig):
    """JSON-serialisable config for :class:`.MinMaxZpWeightEstimator`.

    Attributes:
        type: Discriminator field, always ``"min_max_zp_weight"``.
        target: Module class to quantize (``"linear"``, ``"conv2d"``,
            ``"embedding"``).
        nbits: Quantization bit-width.
        symmetric: Whether to use symmetric quantization.
        weight_transform: Optional key into :data:`WEIGHT_TRANSFORM_REGISTRY`.
            When set, the matching :class:`WeightTransformEntry` supplies both
            the ``weight_transform`` callable (stored in the plan) and the
            optional ``filter_fn`` (extra module-selection constraint).
        group_size: Group size used by the weight transform for per-group
            quantization scale storage.  Ignored when *weight_transform* is
            ``None``.  Should match the transform's own group size (e.g. 32
            for all Q4_0 variants).
    """

    type: Literal["min_max_zp_weight"] = "min_max_zp_weight"
    target: str = "embedding"
    nbits: int = 8
    symmetric: bool = True
    weight_transform: str | None = None
    group_size: int = 32

    def build(self) -> MinMaxZpWeightEstimator:
        """Build a :class:`.MinMaxZpWeightEstimator` for the configured target.

        Returns:
            A :class:`.MinMaxZpWeightEstimator` targeting the module class
            named by :attr:`target`, optionally wired with a weight-transform
            from :data:`WEIGHT_TRANSFORM_REGISTRY`.

        Raises:
            ValueError: If :attr:`weight_transform` is set but not registered.
        """
        entry: WeightTransformEntry | None = None
        if self.weight_transform is not None:
            entry = WEIGHT_TRANSFORM_REGISTRY.get(self.weight_transform)
            if entry is None:
                raise ValueError(
                    f"Unknown weight transform {self.weight_transform!r}. "
                    f"Available: {sorted(WEIGHT_TRANSFORM_REGISTRY)}"
                )
        return MinMaxZpWeightEstimator(
            module_classes=(_resolve_module_class(self.target),),
            filter_fn=entry.filter_fn if entry is not None else None,
            weight_transform=entry.transform if entry is not None else None,
            nbits=self.nbits,
            symmetric=self.symmetric,
            group_size=self.group_size,
        )


@estimator_config("hqq")
class HqqEstimatorConfig(EstimatorConfig):
    """JSON-serialisable config for :class:`.HQQEstimator`.

    Attributes:
        type: Discriminator field, always ``"hqq"``.
        target: Module class to quantize (``"linear"``, ``"conv2d"``,
            ``"embedding"``).
        group_size: Preferred group size for per-group granularity.  Layers
            whose ``in_features`` is not divisible by this value fall back to
            per-channel granularity.
        nbits: Quantization bit-width.
        symmetric: Whether to use symmetric quantization.
        num_iter: Number of HQQ optimization iterations.  ``0`` = min-max only.
        beta: Initial penalty strength for half-quadratic splitting.
        kappa: Penalty schedule multiplier per iteration.
        lp_norm: Lp exponent for the proximal operator.
        weight_transform: Optional key into :data:`WEIGHT_TRANSFORM_REGISTRY`.
    """

    type: Literal["hqq"] = "hqq"
    target: str = "linear"
    group_size: int = 64
    nbits: int = 4
    symmetric: bool = False
    num_iter: int = 20
    beta: float = 1e4
    kappa: float = 1.01
    lp_norm: float = 0.5
    weight_transform: str | None = None

    def build(self) -> HQQEstimator:
        """Build a :class:`.HQQEstimator` for the configured target class.

        Returns:
            A :class:`.HQQEstimator` targeting the module class named by
            :attr:`target`, optionally wired with a weight-transform from
            :data:`WEIGHT_TRANSFORM_REGISTRY`.

        Raises:
            ValueError: If :attr:`weight_transform` is set but not registered.
        """
        entry: WeightTransformEntry | None = None
        if self.weight_transform is not None:
            entry = WEIGHT_TRANSFORM_REGISTRY.get(self.weight_transform)
            if entry is None:
                raise ValueError(
                    f"Unknown weight transform {self.weight_transform!r}. "
                    f"Available: {sorted(WEIGHT_TRANSFORM_REGISTRY)}"
                )
        return HQQEstimator(
            module_classes=(_resolve_module_class(self.target),),
            filter_fn=entry.filter_fn if entry is not None else None,
            weight_transform=entry.transform if entry is not None else None,
            group_size=self.group_size,
            nbits=self.nbits,
            symmetric=self.symmetric,
            num_iter=self.num_iter,
            beta=self.beta,
            kappa=self.kappa,
            lp_norm=self.lp_norm,
        )


@estimator_config("vptq")
@estimator_config("vptq_group")  # backward-compat alias
def _resolve_cluster_backend(name: str, n_iter: int):
    """Resolve a backend name to an instance, deferring the import.

    ``neuropress.transforms`` pulls in ``weight_only``, which imports back into
    this module, so the import must not happen at module scope.
    """
    from .transforms.cluster_backends import backend_by_name

    return backend_by_name(name, n_iter=n_iter)


class VptqGroupEstimatorConfig(EstimatorConfig):
    """JSON-serialisable config for :class:`.VPTQGroupEstimator`.

    Builds a vector-quantization estimator that trains a shared codebook
    across all target layers and assigns each weight vector to the nearest
    centroid (optionally importance-weighted by the Hessian diagonal).

    Attributes:
        type: Discriminator field — ``"vptq"`` (preferred) or ``"vptq_group"``
            (backward-compat alias).
        target: Module class to quantize (``"linear"``, ``"conv2d"``,
            ``"embedding"``).  Pass a list to target multiple types
            (e.g. ``"linear"`` and ``"embedding"``).
        num_centroids: Codebook size.  ``<= 16`` → 4-bit indices,
            ``<= 256`` → 8-bit indices.
        vector_len: Length of each VQ vector (consecutive weight elements).
        proxy_selector: Distance metric — ``"l2"`` or ``"cosine"``.
        grouping: Centroid-sharing strategy — ``"all"`` (one shared
            codebook), ``"by_type"`` (one codebook per module class — not
            useful for distinguishing weight roles within one class, e.g.
            all ``nn.Linear`` layers), ``"by_leaf_name"`` (one codebook
            per trailing attribute name, e.g. ``down_proj`` vs.
            ``gate_proj``/``up_proj`` — use when different structural roles
            have different weight-magnitude distributions and pooling them
            starves one role of centroids), or ``"per_layer"`` (one
            codebook per individual target module — no sharing at all;
            the original VPTQ paper's design.  Every group needs at least
            *num_centroids* training vectors, so very small individual
            layers must be excluded first, e.g. via *exclude_name_suffixes*).
        use_hessian_weighting: Enable importance-weighted assignment using
            ``hessian_diag`` statistics collected during calibration.
        use_tiered: Emit tiered (two-tier prefix-coded) index streams via
            :class:`~neuropress.transforms.vq_packing.TieredIndexPacking`.
            Reduces effective bpw when assignment distribution is skewed.
        zero_escape: Emit zero-escape (flat 2-bit prefix) index streams
            via :class:`~neuropress.transforms.vq_packing.ZeroEscapePacking`.
            Overrides *use_tiered*.  Most effective when centroid-0
            accounts for >= 5% of assignments.
        num_centroids_per_group: Optional per-group codebook sizes, keyed by
            group key (the leaf module name under ``grouping="by_leaf_name"``).
            Groups not listed use ``num_centroids``.  Sizing per type matches
            measured index entropy, which ranges 7.85-13.69 bits: a uniform NC
            spends 14 bits addressing ~230 effective centroids on ``down_proj``
            (22% of the model) while ``in_proj_qkv`` needs all 16384.
        cluster_backend_name: Clustering backend: ``"auto"`` (default),
            ``"torch"``, ``"ivf"``, ``"cuda"`` or ``"sklearn"``.  Pin
            ``"torch"`` for large codebooks on constrained VRAM: the auto
            selector routes ``num_centroids > 4096`` to the IVF backend, whose
            compiled kernel materialises a slab far wider than its own chunk
            budget assumes and OOM'd both 24 GiB and 40 GiB cards at NC=8192.
        max_train_vectors: Total k-means training vectors, spread across the
            layers in a group as ``max(num_centroids * 10, max_train_vectors //
            n_layers)``.  Note the divisor is the group's layer count, so with
            ``grouping="per_layer"`` (n_layers=1) the cap does not bind at the
            50M default.  Note this is usually moot on GPU: a separate VRAM
            budget (``vram_budget``, default 50% of free memory divided by the
            distance-matrix row size) caps the fit far lower — measured
            146K-319K vectors on a 24 GiB card — so ``max_train_vectors`` only
            binds on CPU or on very large GPUs.
        max_n_iter: Maximum number of Lloyd's refinement iterations; the loop
            exits early when centroid shift falls below *tol*.  ``0`` disables
            refinement entirely.
        tol: Convergence threshold — stop when the mean centroid shift falls
            below this value.
        centroid_weighting: Strategy for weighting vectors during centroid
            training (k-means M-step).  ``"uniform"`` treats all vectors
            equally (baseline).  ``"hessian"`` weights each vector by its
            Hessian diagonal importance so centroids become denser in
            high-sensitivity regions.
        weighting_alpha: Power exponent applied to per-vector Hessian
            importance when ``centroid_weighting="hessian"``.  ``0.5``
            (sqrt, gentler), ``1.0`` (linear), ``2.0`` (quadratic,
            aggressive).  Ignored when ``centroid_weighting="uniform"``.
        coverage_beta: Sharpness of the coverage sigmoid that blends
            importance weights toward uniform for oversized clusters.
            Higher values create a sharper transition.  ``0`` disables
            coverage regularization.  Only active when
            ``centroid_weighting="hessian"``.
        coverage_r_max: Maximum acceptable mean cluster radius before the
            coverage penalty activates.  ``None`` (default) auto-tunes
            from the median cluster radius on the first refinement
            iteration.
        exclude_lm_head: When ``True`` (default ``False``), skip any
            module whose leaf name is ``lm_head``.  Useful for models
            with large vocabularies (e.g. Gemma-3 with vocab=262 144)
            where ``lm_head`` dominates the weight-vector pool and
            pollutes the shared codebook.  Set to ``False`` to quantize
            ``lm_head`` along with every other linear layer.
        exclude_name_suffixes: Skip any module whose leaf name matches one
            of these strings, leaving it unquantized (native ``nn.Linear``).
            Useful for small, architecturally-special projections that
            don't belong in a shared codebook with the bulk of the model's
            linear layers — e.g. the GatedDeltaNet decay/gate projections
            (``in_proj_a``, ``in_proj_b``) in hybrid linear-attention
            models, whose weight shape and scale differ drastically from
            the attention/MLP matrices that dominate the codebook fit.
            Empty tuple (default) quantizes every matching module.
        include_name_suffixes: When non-empty, restrict to modules whose
            leaf name matches one of these strings — the inverse of
            *exclude_name_suffixes*.  Use this to carve out a single
            structural role (e.g. ``("down_proj",)``) into its own
            :class:`CompositeEstimatorConfig` entry with a dedicated
            codebook, while a second entry (with that same name in
            *exclude_name_suffixes*) pools everything else — cheaper and
            less failure-prone than ``grouping="by_leaf_name"`` across
            every role at once, since only two codebook fits happen
            instead of one per distinct leaf name.  Empty tuple (default)
            applies no inclusion filter.
        temperature_annealing: Add Gaussian noise to weight vectors before
            assignment so near-boundary vectors can escape local minima.
            T₀ is auto-computed from the mean initial cluster radius;
            temperature decays geometrically as ``T₀ × decay^i``.
        temperature_decay: Geometric decay factor per Lloyd iteration when
            ``temperature_annealing=True``.
        concentration_beta: Strength of the normalized frequency-attraction
            bonus during the E-step.  Each centroid's assignment cost is
            reduced by ``β × mean_L2_layer × freq[c]``, making globally
            popular centroids more attractive without a per-layer tuning
            surface.  ``0`` (default) disables the bonus.
        gptq_correction_iters: Number of GPTQ sequential error-propagation
            passes.  ``0`` disables correction entirely.  ``1`` (default)
            runs one pass after Lloyd's convergence.  ``N > 1`` alternates
            GPTQ E-steps with centroid M-steps (VPTQ-like iterative mode).
        gptq_damp_percent: Hessian dampening fraction for Cholesky inversion.
        centroid_attraction: Strength of the centroid-attraction incentive
            applied after each GPTQ group propagation.  ``0`` (default)
            disables the dual-objective nudge.  Values around ``0.05-0.2``
            add a small bias toward the nearest centroid in proportion to
            inverse Hessian sensitivity.
        centroid_attraction_window: Number of future column groups to nudge
            per GPTQ step.  Larger windows capture more cascading benefit
            at proportional compute cost.
    """

    type: Literal["vptq", "vptq_group"] = "vptq"
    target: str | list[str] = "linear"
    proxy_selector: str = "l2"
    exclude_lm_head: bool = False
    exclude_name_suffixes: tuple[str, ...] = ()
    include_name_suffixes: tuple[str, ...] = ()
    num_centroids: int = _vd("num_centroids")
    vector_len: int = _vd("vector_len")
    grouping: str = _vd("grouping")
    use_hessian_weighting: bool = _vd("use_hessian_weighting")
    use_tiered: bool = _vd("use_tiered")
    zero_escape: bool = _vd("zero_escape")
    max_n_iter: int = _vd("max_n_iter")
    max_train_vectors: int = _vd("max_train_vectors")
    cluster_backend_name: str = "auto"
    num_centroids_per_group: dict[str, int] | None = None
    tol: float = _vd("tol")
    centroid_weighting: str = _vd("centroid_weighting")
    weighting_alpha: float = _vd("weighting_alpha")
    coverage_beta: float = _vd("coverage_beta")
    coverage_r_max: float | None = _vd("coverage_r_max")
    temperature_annealing: bool = _vd("temperature_annealing")
    temperature_decay: float = _vd("temperature_decay")
    concentration_beta: float = _vd("concentration_beta")
    gptq_correction_iters: int = _vd("gptq_correction_iters")
    gptq_correction_alpha: float = _vd("gptq_correction_alpha")
    gptq_damp_percent: float = _vd("gptq_damp_percent")
    centroid_attraction: float = _vd("centroid_attraction")
    centroid_attraction_window: int = _vd("centroid_attraction_window")
    correction_mode: str = _vd("correction_mode")
    ecvq_beta_schedule: tuple[float, ...] = _vd("ecvq_beta_schedule")
    ecvq_iters_per_step: int = _vd("ecvq_iters_per_step")
    ecvq_beta_in_loop: float = _vd("ecvq_beta_in_loop")
    ecvq_warmup_iters: int = _vd("ecvq_warmup_iters")

    def build(self) -> VPTQGroupEstimator:
        """Build a :class:`.VPTQGroupEstimator` for the configured target class(es).

        Returns:
            A :class:`.VPTQGroupEstimator` targeting the module class(es)
            named by :attr:`target`.

        Raises:
            ValueError: If any target name is not in :data:`_MODULE_CLASS_MAP`
                or if :attr:`proxy_selector` is unknown.
        """
        from .transforms.vq import ProxySelector

        targets = [self.target] if isinstance(self.target, str) else self.target
        module_classes = tuple(_resolve_module_class(t) for t in targets)

        try:
            selector = ProxySelector(self.proxy_selector)
        except ValueError as exc:
            raise ValueError(
                f"Unknown proxy_selector {self.proxy_selector!r}. "
                f"Available: {[e.value for e in ProxySelector]}"
            ) from exc

        exclude_leaf_names = set(self.exclude_name_suffixes)
        if self.exclude_lm_head:
            exclude_leaf_names.add("lm_head")
        include_leaf_names = set(self.include_name_suffixes)
        filter_fn = None
        if exclude_leaf_names or include_leaf_names:

            def filter_fn(name: str, mod: nn.Module) -> bool:
                leaf = name.split(".")[-1]
                if include_leaf_names and leaf not in include_leaf_names:
                    return False
                return leaf not in exclude_leaf_names

        return VPTQGroupEstimator(
            module_classes=module_classes,
            num_centroids=self.num_centroids,
            vector_len=self.vector_len,
            proxy_selector=selector,
            grouping=self.grouping,
            use_hessian_weighting=self.use_hessian_weighting,
            use_tiered=self.use_tiered,
            zero_escape=self.zero_escape,
            max_n_iter=self.max_n_iter,
            max_train_vectors=self.max_train_vectors,
            num_centroids_per_group=self.num_centroids_per_group,
            cluster_backend=_resolve_cluster_backend(
                self.cluster_backend_name, self.max_n_iter
            ),
            tol=self.tol,
            centroid_weighting=self.centroid_weighting,
            weighting_alpha=self.weighting_alpha,
            coverage_beta=self.coverage_beta,
            coverage_r_max=self.coverage_r_max,
            temperature_annealing=self.temperature_annealing,
            temperature_decay=self.temperature_decay,
            concentration_beta=self.concentration_beta,
            gptq_correction_iters=self.gptq_correction_iters,
            gptq_correction_alpha=self.gptq_correction_alpha,
            gptq_damp_percent=self.gptq_damp_percent,
            centroid_attraction=self.centroid_attraction,
            centroid_attraction_window=self.centroid_attraction_window,
            correction_mode=CorrectionMode(self.correction_mode),
            ecvq_beta_schedule=tuple(self.ecvq_beta_schedule),
            ecvq_iters_per_step=self.ecvq_iters_per_step,
            ecvq_beta_in_loop=self.ecvq_beta_in_loop,
            ecvq_warmup_iters=self.ecvq_warmup_iters,
            filter_fn=filter_fn,
        )
