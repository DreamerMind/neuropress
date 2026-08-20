"""Static model-size estimation without running compression or calibration.

Given a HuggingFace model slug and a ``NEUROPRESS_COMPRESSION`` key, loads
the model architecture on the PyTorch *meta* device (zero-byte allocation,
no weight download) and estimates the quantized storage size from the
recipe's already-built estimator.

Public entry points:

- :func:`load_model_meta` — loads HF config and instantiates on meta device.
- :func:`collect_estimates` — walks the model and builds per-module estimates.
- :func:`format_size_report` — formats the estimates as a human-readable table.
"""

from __future__ import annotations

import functools
import logging
import math
import operator
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch import nn

if TYPE_CHECKING:
    from neuropress.estimator import QuantizationParamEstimator
    from neuropress.estimator_configs import WeightTransformEntry

LOGGER = logging.getLogger(__name__)

# Bytes per parameter for each baseline float dtype the CLI accepts.  Keys match
# the ``--dtype`` choices in :mod:`neuropress.eval.llm_cli` ("auto" is absent on
# purpose: it falls through to the value inferred from the model config).
_DTYPE_BYTES: dict[str, int] = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}


# ---------------------------------------------------------------------------
# Estimator spec extraction (from live estimator instances)
# ---------------------------------------------------------------------------


@dataclass
class _EstimatorSpec:
    """Targeting and storage description for one leaf estimator."""

    module_class: type
    label: str
    storage_bits: float
    group_size: int | None
    is_packed: bool
    filter_fn: Callable[[str, nn.Module], bool] | None


def _get_transform_entry(
    transform_fn: Callable[..., Any],
) -> WeightTransformEntry | None:
    """Look up the ``WeightTransformEntry`` for *transform_fn*.

    Args:
        transform_fn: The callable stored in the estimator's
            ``weight_transform`` field.

    Returns:
        The matching ``WeightTransformEntry``, or ``None`` if not found.
    """
    from neuropress.estimator_configs import WEIGHT_TRANSFORM_REGISTRY

    for entry in WEIGHT_TRANSFORM_REGISTRY.values():
        if entry.transform is transform_fn:
            return entry
    return None


def _combine_filters(
    f1: Callable[[str, nn.Module], bool] | None,
    f2: Callable[[str, nn.Module], bool] | None,
) -> Callable[[str, nn.Module], bool] | None:
    """Return an AND-combination of two optional module-filter callables.

    Args:
        f1: First filter, or ``None`` to accept all.
        f2: Second filter, or ``None`` to accept all.

    Returns:
        A callable that returns ``True`` only when both filters pass, or
        ``None`` when both inputs are ``None``.
    """
    if f1 is None and f2 is None:
        return None
    if f1 is None:
        return f2
    if f2 is None:
        return f1

    def _combined(name: str, mod: nn.Module, _f1: Any = f1, _f2: Any = f2) -> bool:
        return _f1(name, mod) and _f2(name, mod)

    return _combined


def _specs_for_transform_estimator(
    estimator: Any,
    method_label: str,
) -> list[_EstimatorSpec]:
    """Build :class:`_EstimatorSpec` entries for any transform-bearing estimator.

    Handles estimators that expose ``module_classes``, ``filter_fn``,
    ``weight_transform``, and ``nbits`` — concretely
    :class:`~neuropress.estimators.GPTQEstimator`,
    :class:`~neuropress.estimators.MinMaxZpWeightEstimator`, and
    :class:`~neuropress.estimators.HQQEstimator`.

    Sizing and filtering information is taken exclusively from the matching
    :class:`~neuropress.estimator_configs.WeightTransformEntry` in
    :data:`~neuropress.estimator_configs.WEIGHT_TRANSFORM_REGISTRY`:

    * ``entry.filter_fn`` is AND-combined with ``estimator.filter_fn`` so that
      backend shape constraints (e.g. group-size divisibility) are respected.
    * ``entry.group_size`` is used as-is; ``None`` means per-channel scale
      overhead (the estimator's own ``group_size`` is *not* substituted).
    * ``entry.storage_bits`` is used as-is; falls back to ``estimator.nbits``
      only when the entry has no storage annotation.

    Args:
        estimator: A built estimator instance with the fields listed above.
        method_label: Short human-readable method name (e.g. ``"GPTQ"``).

    Returns:
        One :class:`_EstimatorSpec` per entry in ``estimator.module_classes``.
    """
    wt = estimator.weight_transform
    specs: list[_EstimatorSpec] = []
    for mod_cls in estimator.module_classes:
        if wt is not None:
            entry = _get_transform_entry(wt)
            if entry is None:
                # Unregistered transform — cannot estimate storage size.
                continue
            sbits: float = float(
                entry.storage_bits
                if entry.storage_bits is not None
                else estimator.nbits
            )
            group_size: int | None = entry.group_size  # None = per-channel
            filter_fn = _combine_filters(estimator.filter_fn, entry.filter_fn)
            label = (
                f"Q{estimator.nbits} packed ({method_label}, {int(sbits)}-bit storage)"
            )
            is_packed = True
        else:
            sbits = 16.0
            group_size = None
            filter_fn = estimator.filter_fn
            label = (
                f"float ({method_label} {estimator.nbits}-bit, no storage reduction)"
            )
            is_packed = False
        specs.append(
            _EstimatorSpec(
                module_class=mod_cls,
                label=label,
                storage_bits=sbits,
                group_size=group_size,
                is_packed=is_packed,
                filter_fn=filter_fn,
            )
        )
    return specs


def _parse_specs(estimator: QuantizationParamEstimator) -> list[_EstimatorSpec]:
    """Recursively extract a flat list of :class:`_EstimatorSpec` from *estimator*.

    Works directly on built estimator instances so the result is always
    consistent with what the compression pipeline would actually do.

    Args:
        estimator: A built :class:`~neuropress.estimator.QuantizationParamEstimator`
            instance (may be a :class:`~neuropress.estimators.CompositeEstimator`).

    Returns:
        Flat list of :class:`_EstimatorSpec`, one per
        (leaf estimator, module class) pair.
    """
    from neuropress.estimators import (
        CompositeEstimator,
        GPTQEstimator,
        GPTQPanelEstimator,
        HQQEstimator,
        MinMaxZpWeightEstimator,
    )

    if isinstance(estimator, CompositeEstimator):
        specs: list[_EstimatorSpec] = []
        for sub in estimator.estimators:
            specs.extend(_parse_specs(sub))
        return specs

    if isinstance(estimator, GPTQEstimator):
        return _specs_for_transform_estimator(estimator, "GPTQ")

    if isinstance(estimator, GPTQPanelEstimator):
        nbits_str = "/".join(str(n) for n in sorted(estimator.possible_nbits))
        group_size = estimator.group_size

        def _panel_filter(name: str, mod: nn.Module, _gs: int = group_size) -> bool:
            w = getattr(mod, "weight", None)
            if w is None:
                return False
            weight_cols = functools.reduce(operator.mul, w.shape[1:])
            return weight_cols >= _gs and weight_cols % _gs == 0

        # target_bpw is the average weight-quantization bpw (mix of Q4/Q2);
        # add one f16 scale per group (16 bits / group_size weights).
        scale_bpw = 16.0 / group_size
        effective_bpw = estimator.target_bpw + scale_bpw
        specs = []
        for mod_cls in estimator.module_classes:
            specs.append(
                _EstimatorSpec(
                    module_class=mod_cls,
                    label=f"Q{nbits_str} panel ({estimator.target_bpw:.1f} avg bpw)",
                    storage_bits=effective_bpw,
                    group_size=group_size,
                    is_packed=True,
                    filter_fn=_panel_filter,
                )
            )
        return specs

    if isinstance(estimator, MinMaxZpWeightEstimator):
        return _specs_for_transform_estimator(estimator, "MinMax")

    if isinstance(estimator, HQQEstimator):
        return _specs_for_transform_estimator(estimator, "HQQ")

    LOGGER.warning(
        "Unknown estimator type %s; size estimation will be incomplete",
        type(estimator).__name__,
    )
    return []


# ---------------------------------------------------------------------------
# Size calculation
# ---------------------------------------------------------------------------


@dataclass
class _LayerEstimate:
    """Per-module weight size estimate."""

    name: str
    module_type: str
    description: str
    n_weight: int
    float_bytes: int
    quant_bytes: int

    @property
    def bpw(self) -> float:
        """Effective storage bits per weight parameter."""
        return 0.0 if self.n_weight == 0 else self.quant_bytes * 8 / self.n_weight


def _packed_bytes(n_params: int, storage_bits: float) -> int:
    """Compute expected packed-storage bytes for a quantized weight tensor.

    ``storage_bits`` is the *effective* bits per weight — it must already
    include any scale / zero-point overhead (e.g. 4.5 for Q4_0 with a f16
    scale per 32 weights, 5.0 for Q4_1 with scale + zero-point).  Registry
    entries in :data:`~neuropress.estimator_configs.WEIGHT_TRANSFORM_REGISTRY`
    encode this directly in their ``storage_bits`` field.

    Args:
        n_params: Total number of weight elements.
        storage_bits: Effective bits per weight including all quantization
            metadata overhead.

    Returns:
        Estimated storage bytes.
    """
    return math.ceil(n_params * storage_bits / 8)


def collect_estimates(
    model: nn.Module,
    param_estimator: QuantizationParamEstimator,
    dtype_bytes: int = 2,
) -> tuple[list[_LayerEstimate], int]:
    """Walk *model* and build per-module weight size estimates.

    The model may reside on the PyTorch *meta* device — only parameter
    shapes (not data) are needed.

    Args:
        model: PyTorch model to inspect (meta device is fine).
        param_estimator: The built estimator from the compression recipe
            (e.g. ``recipe.param_estimator``).  Its ``module_classes`` and
            ``filter_fn`` determine which modules are quantized.
        dtype_bytes: Bytes per parameter in the baseline float model
            (2 for fp16 / bfloat16, 4 for fp32).

    Returns:
        Tuple ``(estimates, other_bytes)`` where *estimates* is a list of
        :class:`_LayerEstimate` (one per unique weight tensor) and
        *other_bytes* is the total baseline bytes for parameters not
        attached to a weight-bearing module (biases, LayerNorm scales, etc.).
    """
    specs = _parse_specs(param_estimator)

    estimates: list[_LayerEstimate] = []
    seen_weight_ids: set[int] = set()
    seen_param_ids: set[int] = set()

    for mod_name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if w is None:
            continue

        w_id = id(w)
        if w_id in seen_weight_ids:
            continue  # weight-tied — already counted once
        seen_weight_ids.add(w_id)
        seen_param_ids.add(w_id)

        bias = getattr(mod, "bias", None)
        if bias is not None:
            seen_param_ids.add(id(bias))

        n_weight = w.numel()
        float_b = n_weight * dtype_bytes
        quant_b = float_b
        matched_label = "float (untargeted)"

        for spec in specs:
            if isinstance(mod, spec.module_class):
                if spec.filter_fn is None or spec.filter_fn(mod_name, mod):
                    matched_label = spec.label
                    if spec.is_packed:
                        quant_b = _packed_bytes(n_weight, spec.storage_bits)
                    break

        estimates.append(
            _LayerEstimate(
                name=mod_name,
                module_type=type(mod).__name__,
                description=matched_label,
                n_weight=n_weight,
                float_bytes=float_b,
                quant_bytes=quant_b,
            )
        )

    other_bytes = sum(
        p.numel() * dtype_bytes
        for p in model.parameters()
        if id(p) not in seen_param_ids
    )
    return estimates, other_bytes


# ---------------------------------------------------------------------------
# Model loading (meta device)
# ---------------------------------------------------------------------------


def load_model_meta(model_id: str) -> tuple[nn.Module, int]:
    """Load a HuggingFace model on the meta device and return its default dtype size.

    Downloads only the model config JSON (a few KB) — no weights.

    Args:
        model_id: HuggingFace model ID or local path.

    Returns:
        Tuple ``(model, dtype_bytes)`` where *model* is instantiated on the
        meta device and *dtype_bytes* is the baseline bytes-per-parameter
        inferred from the config's ``torch_dtype`` (defaults to 2 for fp16).

    Raises:
        ImportError: If ``transformers`` is not installed.
    """
    try:
        import torch
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "transformers is required for model loading. Install it with:\n"
            "  pip install transformers"
        ) from exc

    config = AutoConfig.from_pretrained(model_id)

    import torch as _torch

    _dtype_bytes_map_torch = {
        _torch.float16: 2,
        _torch.bfloat16: 2,
        _torch.float32: 4,
    }
    # transformers 5.x renamed config.torch_dtype -> config.dtype and warns on the
    # old name, so prefer the new one and keep the old as a fallback.
    cfg_dtype = getattr(config, "dtype", None)
    if cfg_dtype is None:
        cfg_dtype = getattr(config, "torch_dtype", None)
    dtype_bytes = _DTYPE_BYTES.get(
        str(cfg_dtype).replace("torch.", ""),
        _dtype_bytes_map_torch.get(cfg_dtype, 2),
    )

    with torch.device("meta"):
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_config(config)
        except Exception:
            from transformers import AutoModel

            model = AutoModel.from_config(config)

    model.eval()
    return model, dtype_bytes


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        String with MB or GB suffix.
    """
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    return f"{n / 1e6:.1f} MB"


def _fmt_params(n: int) -> str:
    """Format a parameter count as a human-readable string.

    Args:
        n: Number of parameters.

    Returns:
        String with M or B suffix.
    """
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} B"
    return f"{n / 1e6:.1f} M"


def format_size_report(
    model_id: str,
    compression: str,
    estimates: list[_LayerEstimate],
    other_bytes: int,
    dtype_bytes: int = 2,
) -> Any:
    """Format a Rich renderable size estimation report.

    Row colour coding:

    * **green** — layers with genuine storage reduction (quantized < float).
    * **yellow** — targeted layers with no storage reduction (float-sim).
    * **dim** — untargeted or non-weight parameters (bias, norms).
    * **bold** — totals row.

    Args:
        model_id: HuggingFace model slug or local path.
        compression: Compression method key from ``NEUROPRESS_COMPRESSION``.
        estimates: Per-module estimates from :func:`collect_estimates`.
        other_bytes: Extra float bytes for non-weight params (biases, norms).
        dtype_bytes: Bytes per element in the baseline float model.

    Returns:
        A Rich ``Group`` renderable — pass to ``Console().print()``.
    """
    from rich import box
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    # --- aggregate by (module_type, description) ----------------------------
    groups: dict[tuple[str, str], list[_LayerEstimate]] = defaultdict(list)
    for est in estimates:
        groups[(est.module_type, est.description)].append(est)

    def _row_key(item: tuple[tuple[str, str], list[_LayerEstimate]]) -> tuple:
        (mod_type, _), group = item
        any_packed = any(e.quant_bytes < e.float_bytes for e in group)
        return (not any_packed, mod_type)

    rows = []
    for (mod_type, desc), group in sorted(groups.items(), key=_row_key):
        n_params = sum(e.n_weight for e in group)
        float_b = sum(e.float_bytes for e in group)
        quant_b = sum(e.quant_bytes for e in group)
        bpw = quant_b * 8 / n_params if n_params else 0.0
        rows.append((mod_type, desc, len(group), n_params, float_b, quant_b, bpw))

    # --- totals -------------------------------------------------------------
    total_w_params = sum(e.n_weight for e in estimates)
    total_w_float = sum(e.float_bytes for e in estimates)
    total_w_quant = sum(e.quant_bytes for e in estimates)
    total_float = total_w_float + other_bytes
    total_quant = total_w_quant + other_bytes
    total_params = total_w_params + other_bytes // dtype_bytes
    avg_bpw = total_w_quant * 8 / total_w_params if total_w_params else 0.0
    ratio = total_float / total_quant if total_quant else 1.0

    # --- header -------------------------------------------------------------
    header = Text()
    header.append("Model:       ", style="dim")
    header.append(model_id, style="bold white")
    header.append("\nCompression: ", style="dim")
    header.append(compression, style="bold cyan")

    # --- table --------------------------------------------------------------
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        show_footer=False,
        expand=False,
    )
    table.add_column("Layer type", min_width=12, no_wrap=True)
    table.add_column("Quantization", min_width=40)
    table.add_column("Layers", justify="right", min_width=6)
    table.add_column("Params", justify="right", min_width=8)
    table.add_column("Float", justify="right", min_width=10)
    table.add_column("Quantized", justify="right", min_width=10)
    table.add_column("bpw", justify="right", min_width=5)

    for mod_type, desc, n_layers, n_params, float_b, quant_b, bpw in rows:
        is_compressed = quant_b < float_b
        is_untargeted = "untargeted" in desc
        if is_compressed:
            row_style = "green"
        elif is_untargeted:
            row_style = "dim"
        else:
            row_style = "yellow"  # targeted but no storage reduction (float-sim)
        table.add_row(
            mod_type,
            desc,
            str(n_layers),
            _fmt_params(n_params),
            _fmt_bytes(float_b),
            _fmt_bytes(quant_b),
            f"{bpw:.2f}",
            style=row_style,
        )

    if other_bytes > 0:
        n_other = other_bytes // dtype_bytes
        table.add_row(
            "—",
            "float (bias, norm, …)",
            "—",
            _fmt_params(n_other),
            _fmt_bytes(other_bytes),
            _fmt_bytes(other_bytes),
            "—",
            style="dim",
        )

    table.add_section()
    table.add_row(
        "Total",
        "",
        "",
        _fmt_params(total_params),
        _fmt_bytes(total_float),
        _fmt_bytes(total_quant),
        "",
        style="bold",
    )

    # --- summary line -------------------------------------------------------
    summary = Text()
    summary.append("  Compression ratio: ")
    summary.append(f"{ratio:.2f}x", style="bold cyan")
    summary.append("   |   Avg weight storage bpw: ")
    summary.append(f"{avg_bpw:.2f}", style="bold cyan")

    return Group(header, Text(""), table, summary)
