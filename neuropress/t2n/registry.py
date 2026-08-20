"""torch_to_nnef compression registry for neuropress.

Each entry in ``NEUROPRESS_COMPRESSION`` is a :func:`functools.partial` of
:func:`run_quantization_pipeline` with a pre-built :class:`.QuantizationRecipe`
bound via the *recipe* keyword::

    NEUROPRESS_COMPRESSION = {
        "open_text_c4_gptq_q4_0": partial(run_quantization_pipeline, recipe=...),
        ...,
    }

Registry keys encode the calibration dataset and compression method so the
intended configuration is unambiguous at a glance:
``<dataset>_<method>`` — e.g. ``open_text_c4_gptq_q4_0``.

To build an arbitrary recipe from JSON use :class:`.QuantizationRecipeConfig`,
which resolves all component types from the registries at deserialisation time::

    recipe = QuantizationRecipeConfig.from_json(json_str).build()
    run_quantization_pipeline(model, tokenizer=tok, recipe=recipe)

The ``"q4_0"`` weight-transform key is registered into
:data:`.WEIGHT_TRANSFORM_REGISTRY` when this module is imported.  Reference it
in any :class:`.GptqEstimatorConfig` via the ``weight_transform`` field::

    {"type": "gptq", "weight_transform": "q4_0", "group_size": 32}

Pass the registry path to ``LLMExporter.prepare``::

    exporter.prepare(
        compression_method="open_text_c4_gptq_q4_0",
        compression_registry="neuropress.t2n.registry.NEUROPRESS_COMPRESSION",
    )

:func:`run_quantization_pipeline` contract::

    fn(model, *, tokenizer, recipe, **kwargs) -> model

It reads :attr:`.QuantizationRecipe.calibration` to load batches from
:data:`.calibration.DATASET_REGISTRY`, transfers them to the model device
lazily (one batch at a time — no OOM), then delegates to
:class:`.QuantizationPipeline`.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .. import transforms as _transforms  # noqa: F401  (registers weight transforms)
from ..calibration import DATASET_REGISTRY
from ..estimator_configs import (
    GptqEstimatorConfig,
    GptqPanelEstimatorConfig,
    HqqEstimatorConfig,
    MinMaxZpWeightEstimatorConfig,
    VptqGroupEstimatorConfig,
)
from ..pipeline import QuantizationPipeline
from ..quantization.cache import DiskCacheManager, _cal_fingerprint, _recipe_fingerprint
from ..quantization.spans import (
    LoggingSpanCollector,
    SpanCollector,
)
from ..recipe import CalibrationConfig, QuantizationRecipe
from ..recipe_config import CompositeEstimatorConfig, QuantizationRecipeConfig
from ..transforms import Q4_X_GROUP_SIZE
from .model_io import attach_manifest

LOGGER = logging.getLogger(__name__)

_LIBRISPEECH_SEQ_LEN = 480_000  # 30 s at 16 kHz — Whisper's native window


# ---------------------------------------------------------------------------
# Generic pipeline runner
# ---------------------------------------------------------------------------


def run_quantization_pipeline(
    model: nn.Module,
    *,
    tokenizer: Any,
    recipe: QuantizationRecipe,
    recipe_config_json: str | None = None,
    cache_dir: Path | str | None = None,
    span_collector: SpanCollector | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Run a :class:`.QuantizationPipeline` driven entirely by *recipe*.

    Calibration data is loaded from :data:`.calibration.DATASET_REGISTRY`
    using the parameters stored in ``recipe.calibration``.  Batches are
    transferred to the model device lazily — only one batch is on the GPU at
    a time, avoiding the OOM that eager loading would cause.

    The pipeline applies weight assignment, manifest recording, and kernel
    lowering (when configured) in a single pass.  After the pipeline
    completes, :func:`~neuropress.t2n.model_io.attach_manifest` is called
    to store the ``recipe_config_json``.

    When the recipe configures a kernel lowering pass, it is applied eagerly
    so the returned model is always in the fast inference state.  The
    lowering object is stored on the model as
    ``model.__neuropress_kernel_lowering__`` so that
    :func:`~neuropress.t2n.model_io.save_quantized_model` can auto-unlower
    before saving and re-lower afterwards.

    Args:
        model: Float model to quantize (modified in-place).
        tokenizer: HuggingFace-compatible tokenizer passed to the dataset
            loader.
        recipe: Self-contained quantization recipe carrying both the
            calibration configuration and the estimator / adapter.
        recipe_config_json: Optional JSON serialisation of the
            :class:`~neuropress.recipe_config.QuantizationRecipeConfig` used to
            build *recipe*.  Stored in the manifest for auditing / round-trip
            deserialization context.  Registry entries supply this
            automatically via :func:`_make_entry`.
        cache_dir: Root directory for the estimator's cache stages.  When
            ``None`` no caching happens (a :class:`NullCacheManager` is used);
            model-level isolation is the caller's responsibility.
        span_collector: Optional sink receiving one timing span per pipeline
            stage.  Defaults to logging spans at INFO level.
        **kwargs: Extra kwargs forwarded by ``LLMExporter`` (e.g.
            ``export_dirpath``) are silently ignored.  ``_load_dataset`` is
            extracted and forwarded to the loader for testing purposes.

    Returns:
        The same *model*, modified according to *recipe* and with a
        :class:`~neuropress.t2n.model_io.QuantizationManifest` attached as
        ``model.__neuropress_manifest__``.
    """
    _load_dataset = kwargs.pop("_load_dataset", None)
    device = next(model.parameters()).device
    cal = recipe.calibration
    loader = DATASET_REGISTRY[cal.dataset]
    loader_kwargs = (
        {"_load_dataset": _load_dataset} if _load_dataset is not None else {}
    )
    batches_raw = loader(
        tokenizer,
        seq_len=cal.seq_len,
        n_samples=cal.n_samples,
        batch_size=cal.batch_size,
        seed=cal.seed,
        **loader_kwargs,
    )
    # Wire up cache and span collector before the pipeline runs.
    estimator = recipe.param_estimator
    if cache_dir is not None:
        cal_fp = _cal_fingerprint(recipe.calibration)
        recipe_fp = _recipe_fingerprint(recipe_config_json or "")
        stage_scopes = {d.name: d.scope for d in estimator.cache_stages()}
        estimator.set_cache(
            DiskCacheManager(cache_dir, cal_fp, recipe_fp, stage_scopes)
        )
    collector = span_collector if span_collector is not None else LoggingSpanCollector()
    estimator.set_span_collector(collector)
    # Lazy device transfer — one batch on the GPU at a time.
    batches = ({k: v.to(device) for k, v in b.items()} for b in batches_raw)
    QuantizationPipeline(recipe).run(model, batches)
    attach_manifest(model, recipe_config_json=recipe_config_json)
    # Apply kernel lowering eagerly so the model is always in the fast
    # inference state.  save_quantized_model will unlower transparently.
    if recipe.kernel_lowering is not None:
        model.__neuropress_kernel_lowering__ = recipe.kernel_lowering  # type: ignore[attr-defined]
        recipe.kernel_lowering.apply(model)
    _guard_unquantized_linear_dtype(model)
    return model


def _guard_unquantized_linear_dtype(model: nn.Module) -> None:
    """Cast activations to match any plain ``nn.Linear`` left in its native dtype.

    Estimator filters (e.g. ``exclude_lm_head``, ``exclude_name_suffixes``)
    can leave individual ``nn.Linear`` modules unquantized and un-lowered,
    keeping their original checkpoint dtype (e.g. bfloat16).  Everything
    else in the model may run in a different compute dtype after kernel
    lowering (e.g. float16), so activations reaching an excluded layer can
    mismatch its weight dtype.  Mirrors the Conv1d dtype hook in
    ``eval/llm_cli.py``: cast input to the layer's weight dtype, cast the
    output back so downstream ops see the dtype they expect.

    Only touches modules whose ``.weight`` is a plain ``nn.Parameter`` —
    quantized-but-not-yet-lowered layers carry a QTensor assigned directly
    to ``._parameters["weight"]`` (see ``WeightAssignmentAdapter``), which
    this must not disturb.

    Note: forward hooks registered via ``register_forward_hook`` see
    ``input`` *after* any forward-pre-hook has already modified it — so the
    original caller dtype must be stashed explicitly (on the module) rather
    than re-read from ``input`` inside the post-hook, or the "cast back"
    becomes a no-op.
    """

    def _pre(mod: nn.Module, args: tuple) -> tuple | None:
        x = args[0]
        mod._np_dtype_guard_orig = x.dtype  # type: ignore[attr-defined]
        if x.dtype != mod.weight.dtype:
            return (x.to(mod.weight.dtype), *args[1:])
        return None

    def _post(mod: nn.Module, args: tuple, out: torch.Tensor) -> torch.Tensor:
        orig = mod._np_dtype_guard_orig  # type: ignore[attr-defined]
        return out.to(orig) if out.dtype != orig else out

    for m in model.modules():
        # Exact type check on purpose: QTensor types subclass Parameter, so
        # isinstance() would also match the quantized layers this guard must
        # skip.  Do not "simplify" this to isinstance().
        if isinstance(m, nn.Linear) and type(m.weight) is nn.Parameter:
            m.register_forward_pre_hook(_pre)
            m.register_forward_hook(_post)


def apply_kernel_lowering(model: nn.Module) -> nn.Module:
    """Apply the stored kernel lowering pass to *model*.

    After :func:`run_quantization_pipeline` runs, the recipe's
    :class:`~neuropress.kernel_lowering.KernelLowering` is stored on the
    model as ``model.__neuropress_kernel_lowering__``.  This function
    retrieves and applies it, replacing QTensor-backed modules with
    hardware-optimised NP* modules.

    Typical usage (re-lower after manual unlower)::

        from neuropress.kernel_lowering import unlower_model
        unlower_model(model)      # back to raw QTensors
        apply_kernel_lowering(model)  # NP* modules again

    Args:
        model: Quantized model with a stored kernel lowering pass.

    Returns:
        The same *model* with NP* modules replacing QTensor-backed ones.
        Returns *model* unchanged if no lowering was stored.
    """
    lowering = getattr(model, "__neuropress_kernel_lowering__", None)
    if lowering is None:
        return model
    lowering.apply(model)
    return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NEUROPRESS_COMPRESSION_REGISTRY = "neuropress.t2n.registry.NEUROPRESS_COMPRESSION"


def _make_entry(config: QuantizationRecipeConfig) -> partial:
    """Build a registry entry from *config*, capturing its JSON for the manifest.

    The returned callable quantizes the model **and** applies kernel lowering
    (when the recipe configures one).  The lowering pass is stored on the
    model so :func:`~neuropress.t2n.model_io.save_quantized_model` can
    auto-unlower before saving and re-lower afterwards.

    The *config* is stored on the returned partial as ``.__config__``
    so callers can retrieve it if needed.
    """
    entry = partial(
        run_quantization_pipeline,
        recipe=config.build(),
        recipe_config_json=config.model_dump_json(),
    )
    entry.__config__ = config  # type: ignore[attr-defined]
    return entry


# Keys encode <dataset>_<method> so the configuration is explicit at a glance.
# Each entry uses a freshly built recipe so that GPTQEstimator.prepare() has
# its own mutable state and entries do not interfere with each other.
NEUROPRESS_COMPRESSION: dict[str, Any] = {
    "open_text_c4_gptq_q4_0": _make_entry(
        QuantizationRecipeConfig(
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed",
            ),
        )
    ),
    "open_text_c4_gptq_q4_0_32samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=32),
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed",
            ),
        )
    ),
    "open_text_c4_gptq_q4_0_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed",
            ),
        )
    ),
    # Q4_0 linear-only, lm_head excluded — avoids corrupting embed_tokens via
    # the tied-weight reference that GPTQ correction modifies in-place.
    "open_text_c4_gptq_q4_0_no_lm_head_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed_no_lm_head",
            ),
        )
    ),
    # --- Speech / audio entries (require neuropress[audio]) -----------------
    # dataset="librispeech" → load_librispeech_calibration; seq_len is in
    # raw waveform samples at 16 kHz (480_000 = 30 s, Whisper's native window).
    "speech_librispeech_gptq_q4_0": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(
                dataset="librispeech",
                seq_len=_LIBRISPEECH_SEQ_LEN,
                n_samples=128,
                batch_size=8,
            ),
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed",
            ),
        )
    ),
    "speech_librispeech_gptq_q4_0_32samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(
                dataset="librispeech",
                seq_len=_LIBRISPEECH_SEQ_LEN,
                n_samples=32,
                batch_size=8,
            ),
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed",
            ),
        )
    ),
    "speech_librispeech_gptq_q4_0_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(
                dataset="librispeech",
                seq_len=_LIBRISPEECH_SEQ_LEN,
                n_samples=256,
                batch_size=8,
            ),
            estimator=GptqEstimatorConfig(
                group_size=Q4_X_GROUP_SIZE,
                nbits=4,
                weight_transform="q4_0_packed",
            ),
        )
    ),
    # --- Composite: GPTQ Q4_0 linear + MinMax Q4_0 embedding ---------------
    # Applies GPTQ-corrected Q4_0 (symmetric, group_size=32) to all Linear
    # layers and min-max symmetric 4-bit packed Q4_0 to Embedding layers.
    "open_text_c4_gptq_q4_0_linear_minmax_q4_0_emb_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=CompositeEstimatorConfig(
                methods=[
                    GptqEstimatorConfig(
                        target="linear",
                        group_size=Q4_X_GROUP_SIZE,
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                    ),
                    MinMaxZpWeightEstimatorConfig(
                        target="embedding",
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                        group_size=Q4_X_GROUP_SIZE,
                    ),
                ]
            ),
        )
    ),
    # --- Composite: GPTQ Q4_0 linear + HQQ asymmetric Q4_1 embedding -------
    # Applies GPTQ-corrected Q4_0 (symmetric, group_size=32) to all Linear
    # layers and HQQ asymmetric 4-bit packed Q4_1 to Embedding layers.
    # Q4_1 uses per-group f16 scale + zero-point (5.0 effective bpw).
    "open_text_c4_gptq_q4_0_linear_hqq_q41_emb_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=CompositeEstimatorConfig(
                methods=[
                    GptqEstimatorConfig(
                        target="linear",
                        group_size=Q4_X_GROUP_SIZE,
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                    ),
                    HqqEstimatorConfig(
                        target="embedding",
                        nbits=4,
                        symmetric=False,
                        weight_transform="q4_1_packed",
                        group_size=Q4_X_GROUP_SIZE,
                    ),
                ]
            ),
        )
    ),
    # --- Composite: GPTQ Q4_0 linear + MinMax Q4_0 embedding (32 samples) --
    "open_text_c4_gptq_q4_0_linear_minmax_q4_0_emb_32samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=32),
            estimator=CompositeEstimatorConfig(
                methods=[
                    GptqEstimatorConfig(
                        target="linear",
                        group_size=Q4_X_GROUP_SIZE,
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                    ),
                    MinMaxZpWeightEstimatorConfig(
                        target="embedding",
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                        group_size=Q4_X_GROUP_SIZE,
                    ),
                ]
            ),
        )
    ),
    # --- Composite: GPTQ Q4_0 linear + HQQ asymmetric Q4_1 emb (32 samples)
    "open_text_c4_gptq_q4_0_linear_hqq_q41_emb_32samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=32),
            estimator=CompositeEstimatorConfig(
                methods=[
                    GptqEstimatorConfig(
                        target="linear",
                        group_size=Q4_X_GROUP_SIZE,
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                    ),
                    HqqEstimatorConfig(
                        target="embedding",
                        nbits=4,
                        symmetric=False,
                        weight_transform="q4_1_packed",
                        group_size=Q4_X_GROUP_SIZE,
                    ),
                ]
            ),
        )
    ),
    # --- GPTQ panel mixed Q4/Q2 linear ------------------------------------
    # Mixed-precision GPTQ on Linear layers: per-column-group sensitivity
    # scoring allocates 4-bit to important groups and 2-bit to the rest,
    # targeting 3.0 average bits-per-weight (group_size=32).
    "open_text_c4_gptq_panel_q4q2_linear_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=GptqPanelEstimatorConfig(
                target="linear",
                group_size=Q4_X_GROUP_SIZE,
                possible_nbits=[2, 4],
                target_bpw=3.0,
            ),
        )
    ),
    # --- Composite: GPTQ panel Q4/Q2 linear + MinMax Q4_0 embedding --------
    # Panel mixed-precision (3.0 avg bpw) on Linear layers combined with
    # symmetric min-max Q4_0 on Embedding layers.
    "open_text_c4_gptq_panel_q4q2_linear_minmax_q4_0_emb_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=CompositeEstimatorConfig(
                methods=[
                    GptqPanelEstimatorConfig(
                        target="linear",
                        group_size=Q4_X_GROUP_SIZE,
                        possible_nbits=[2, 4],
                        target_bpw=3.0,
                    ),
                    MinMaxZpWeightEstimatorConfig(
                        target="embedding",
                        nbits=4,
                        symmetric=True,
                        weight_transform="q4_0_packed",
                        group_size=Q4_X_GROUP_SIZE,
                    ),
                ]
            ),
        )
    ),
    # --- Composite: GPTQ panel Q4/Q2 linear + HQQ asymmetric Q4_1 embedding
    # Panel mixed-precision (3.0 avg bpw) on Linear layers combined with
    # HQQ asymmetric Q4_1 on Embedding layers.
    "open_text_c4_gptq_panel_q4q2_linear_hqq_q41_emb_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=CompositeEstimatorConfig(
                methods=[
                    GptqPanelEstimatorConfig(
                        target="linear",
                        group_size=Q4_X_GROUP_SIZE,
                        possible_nbits=[2, 4],
                        target_bpw=3.0,
                    ),
                    HqqEstimatorConfig(
                        target="embedding",
                        nbits=4,
                        symmetric=False,
                        weight_transform="q4_1_packed",
                        group_size=Q4_X_GROUP_SIZE,
                    ),
                ]
            ),
        )
    ),
    # --- VPTQ: shared-codebook vector quantization --------------------------
    # All nn.Linear layers share one codebook trained on pooled weight vectors.
    # Importance-weighted assignment uses hessian_diag collected in one pass.
    #
    # Approximate bits-per-weight (indices only, no per-layer overhead):
    #   256 centroids, VL=4 → log2(256)/4 = 2.0 bpw
    #   256 centroids, VL=8 → log2(256)/8 = 1.0 bpw
    #   16  centroids, VL=8 → log2(16)/8  = 0.5 bpw
    "open_text_c4_vptq_256c_v4_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
            ),
        )
    ),
    "open_text_c4_vptq_256c_v4_32samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=32),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
            ),
        )
    ),
    "open_text_c4_vptq_16c_v8_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=16,
                vector_len=8,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
            ),
        )
    ),
    # VPTQ with iterative refinement until convergence (tol=1e-4, max_n_iter=50).
    # Same codebook size and vector length as the 256-sample entry above;
    # iterative M+E steps improve centroid placement after initial assignment.
    "open_text_c4_vptq_256c_v4_256samples_iter3": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
            ),
        )
    ),
    # --- VPTQ with large codebooks (Phase 3: non-power-of-2 bitpacking) -----
    # 1024 centroids → 10-bit indices → 2.5 bpw (VL=4).
    # More expressive codebook at the cost of +0.5 bpw vs 256c/8-bit.
    "open_text_c4_vptq_1024c_v4_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
            ),
        )
    ),
    # 4096 centroids → 12-bit indices → 3.0 bpw (VL=4).
    # High-quality VQ at the cost of 1 extra bpw vs 256c.
    "open_text_c4_vptq_4096c_v4_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=4096,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
            ),
        )
    ),
    # 16384 centroids → 14-bit indices → 3.5 bpw (VL=4).
    # Large codebook — test whether more centroids improve quality beyond 4096c.
    "open_text_c4_vptq_16384c_v4_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=16384,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
            ),
        )
    ),
    # 65536 centroids → 16-bit indices → 4.0 bpw (VL=4).
    # Iso-bpw comparison with GPTQ Q4_0 — same bit budget, VQ vs scalar.
    "open_text_c4_vptq_65536c_v4_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=65536,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
            ),
        )
    ),
    # 1024 centroids, VL=8 → 1.25 bpw.  Ultra-low bpw at larger vector stride.
    "open_text_c4_vptq_1024c_v8_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=8,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
            ),
        )
    ),
    # --- Tiered VPTQ entries (Phase 4: TieredIndexPacking) -------------------
    # Two-tier prefix-coded index streams: 1-bit prefix selects hot (4-bit)
    # or cold (N-bit) index.  Effective bpw ≈ 1 + hot_frac*4 + cold_frac*N
    # where hot_frac is the top-16-centroid share of assignments.
    #
    # 256 centroids, VL=4, tiered → fixed 2.0 bpw; at 30% hot fraction the
    # effective rate drops to ≈1.7 bpw.
    "open_text_c4_vptq_256c_v4_tiered_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                use_tiered=True,
                tol=1e-4,
            ),
        )
    ),
    # 1024 centroids, VL=4, tiered → fixed 2.5 bpw; cold tier uses 10 bits.
    # At 30% hot fraction effective rate ≈2.1 bpw — richer codebook same cost
    # as 256c/tiered at high skew.
    "open_text_c4_vptq_1024c_v4_tiered_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                use_tiered=True,
                tol=1e-4,
            ),
        )
    ),
    # 256 centroids, VL=8, tiered → fixed 1.0 bpw; at 30% hot fraction ≈0.85
    # bpw.  Aggressive compression for bandwidth-limited inference.
    "open_text_c4_vptq_256c_v8_tiered_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=8,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                use_tiered=True,
                tol=1e-4,
            ),
        )
    ),
    # --- Zero-escape VPTQ entries (Phase 5: ZeroEscapePacking) ---------------
    # Flat 2-bit prefix: 00→centroid-0 (2 bpi), 01+4bit→1-16 (6 bpi),
    # 10+8bit→17-256 (10 bpi), 11+Nbit→full range (2+N bpi).
    # Most effective when centroid-0 ≥ 5% of assignments (Zipf s ≥ 1.0).
    #
    # 1024 centroids, VL=4, zero-escape → fixed 2.5 bpw; with 10% centroid-0
    # and 60% top-16 the effective rate drops to ≈1.6 bpw.
    "open_text_c4_vptq_1024c_v4_zesc_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                zero_escape=True,
                tol=1e-4,
            ),
        )
    ),
    # 4096 centroids, VL=4, zero-escape → fixed 3.0 bpw; richer codebook
    # with zero-escape to keep effective bpw competitive.
    "open_text_c4_vptq_4096c_v4_zesc_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=4096,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                zero_escape=True,
                tol=1e-4,
            ),
        )
    ),
    # 1024 centroids, VL=8, zero-escape → fixed 1.25 bpw; aggressive
    # compression with rich codebook.
    "open_text_c4_vptq_1024c_v8_zesc_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=8,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                zero_escape=True,
                tol=1e-4,
            ),
        )
    ),
    # --- VPTQ with hessian-weighted centroid training (importance-aware M-step) -
    # Same codebook sizes as the uniform baselines above, but with
    # centroid_weighting="hessian" so the k-means M-step weights vectors by
    # their Hessian importance.  alpha=0.5 (sqrt) is the default gentler curve.
    #
    # 256c hessian-weighted → 2.0 bpw (VL=4).
    "open_text_c4_vptq_256c_v4_hwt_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
            ),
        )
    ),
    # 1024c hessian-weighted → 2.5 bpw (VL=4).
    "open_text_c4_vptq_1024c_v4_hwt_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
            ),
        )
    ),
    # 4096c hessian-weighted → 3.0 bpw (VL=4).
    "open_text_c4_vptq_4096c_v4_hwt_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=4096,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
            ),
        )
    ),
    # --- VPTQ with HWT + coverage regularization (beta=5.0, r_max=auto) ------
    # Same as HWT above but with soft radius penalty to prevent centroid
    # concentration in high-importance regions.
    #
    # 256c HWT+coverage → 2.0 bpw (VL=4).
    "open_text_c4_vptq_256c_v4_hwt_cov_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=256,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
                coverage_beta=5.0,
            ),
        )
    ),
    # 1024c HWT+coverage → 2.5 bpw (VL=4).
    "open_text_c4_vptq_1024c_v4_hwt_cov_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
                coverage_beta=5.0,
            ),
        )
    ),
    # --- FineWeb calibration variants (Track 1: diverse generic corpus) ------
    # Same estimator configs as the C4 equivalents; only the calibration
    # dataset changes to FineWeb (higher quality, more diverse than C4,
    # still fully generic — blind to any downstream eval task).
    # 1024c uniform → 2.5 bpw (VL=4), FineWeb calibration.
    "open_text_fineweb_vptq_1024c_v4_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(dataset="fineweb", n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=False,
                tol=1e-4,
            ),
        )
    ),
    # 1024c HWT+coverage → 2.5 bpw (VL=4), FineWeb calibration.
    "open_text_fineweb_vptq_1024c_v4_hwt_cov_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(dataset="fineweb", n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=1024,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
                coverage_beta=5.0,
            ),
        )
    ),
    # 4096c HWT+coverage → 3.0 bpw (VL=4).
    "open_text_c4_vptq_4096c_v4_hwt_cov_256samples": _make_entry(
        QuantizationRecipeConfig(
            calibration=CalibrationConfig(n_samples=256),
            estimator=VptqGroupEstimatorConfig(
                target="linear",
                num_centroids=4096,
                vector_len=4,
                proxy_selector="l2",
                grouping="all",
                use_hessian_weighting=True,
                tol=1e-4,
                centroid_weighting="hessian",
                weighting_alpha=0.5,
                coverage_beta=5.0,
            ),
        )
    ),
}

try:  # optional local extension module
    from .research_registry import RESEARCH_ENTRIES
except ImportError:  # pragma: no cover - not present in every checkout
    pass
else:
    NEUROPRESS_COMPRESSION.update(RESEARCH_ENTRIES)
