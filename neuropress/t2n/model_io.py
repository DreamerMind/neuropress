"""Quantized model serialization and deserialization via safetensors.

Saves the full model state — regular float parameters *and* QTensor
parameters — to a single ``.safetensors`` file.  A JSON manifest embedded
in the file's metadata records an ordered list of *transforms* that were
applied to the model, plus the optional
:class:`~neuropress.recipe_config.QuantizationRecipeConfig` JSON used to
produce it.

This enables lossless roundtrip serialization without re-running calibration::

    # After quantization —
    save_quantized_model(quantized_model, "checkpoint.safetensors")

    # On reload —
    fresh_model = MyModel()
    manifest = load_quantized_model(fresh_model, "checkpoint.safetensors")
    # fresh_model now carries the same quantized weights as quantized_model.

Manifest attachment
-------------------
:func:`.run_quantization_pipeline` and :class:`.QuantizationPipeline` both
call :func:`attach_manifest` automatically after running.  The manifest is
stored on the model as ``model.__neuropress_manifest__`` so
:func:`save_quantized_model` can include the full recipe JSON without any
extra arguments.

Calling :func:`attach_manifest` manually is also supported — useful when
bypassing the registry entry-point or when adding ``recipe_config_json``
after the fact::

    attach_manifest(model, recipe_config_json=cfg.model_dump_json())
    save_quantized_model(model, "checkpoint.safetensors")

Transform list
--------------
``QuantizationManifest.transforms`` is an ordered list of dicts.  Each entry
has at least a ``"type"`` key.  Known types:

``"qtensor_assignment"``
    Records which parameters were replaced with QTensors and the serde
    metadata needed to reconstruct them on load.

``"mixed_precision_split"``
    Records which ``nn.Linear`` modules were replaced with
    :class:`~neuropress.transforms.split_mixed_linear.MixedPrecisionLinear`
    and the structure needed to recreate them before loading weights.

Safetensors key convention
--------------------------
==========================================  ==============================
Key pattern                                 Contents
==========================================  ==============================
``{param_path}``                            Regular float/int parameter
``{param_path}::{serde_tensor_key}``        QTensor sub-tensor component
==========================================  ==============================

PyTorch parameter paths are dotted Python identifiers and never contain
``::``; the separator is therefore unambiguous.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from .qtensors import _SERDE_REGISTRY

_log = logging.getLogger(__name__)

_MANIFEST_META_KEY = "__neuropress_manifest__"
_QT_SEP = "::"  # separates param_path from serde_tensor_key; never in PyTorch paths
_FORMAT_VERSION = "2"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class QtensorEntry(BaseModel):
    """Serde descriptor for one QTensor parameter."""

    type: str
    tensor_keys: list[str]
    metadata: dict[str, Any]


class QtensorAssignmentTransform(BaseModel):
    """Records which parameters were replaced with QTensors."""

    type: Literal["qtensor_assignment"]
    qtensor_params: dict[str, QtensorEntry]


class MixedPrecisionModuleSpec(BaseModel):
    """Structure needed to reconstruct one MixedPrecisionLinear on load."""

    out_features: int
    split_sizes: list[int]
    precision_nbits: list[int]
    has_bias: bool


class MixedPrecisionSplitTransform(BaseModel):
    """Records which nn.Linear modules were split for mixed-precision."""

    type: Literal["mixed_precision_split"]
    modules: dict[str, MixedPrecisionModuleSpec]


# Ordered: typed variants first, then a plain-dict fallback for unknown types.
ManifestTransform = (
    QtensorAssignmentTransform | MixedPrecisionSplitTransform | dict[str, Any]
)


class QuantizationManifest(BaseModel):
    """Records all transformations applied to a model.

    Attached to the model as ``model.__neuropress_manifest__`` by
    :func:`attach_manifest`, and embedded in checkpoint files by
    :func:`save_quantized_model`.
    """

    transforms: list[ManifestTransform] = Field(default_factory=list)
    recipe_config_json: str | None = None

    @property
    def qtensor_params(self) -> dict[str, str]:
        """Map from parameter path to QTensor type name."""
        for t in self.transforms:
            if isinstance(t, QtensorAssignmentTransform):
                return {k: v.type for k, v in t.qtensor_params.items()}
        return {}


def attach_manifest(
    model: nn.Module,
    recipe_config_json: str | None = None,
) -> QuantizationManifest:
    """Scan *model* for QTensor parameters and update its manifest.

    Inspects every named parameter; any whose type is registered in
    :data:`~neuropress.t2n.qtensors._SERDE_REGISTRY` is recorded in the
    ``"qtensor_assignment"`` transform entry.  If a
    :class:`QuantizationManifest` is already attached to the model (e.g. by a
    prior call or by :func:`apply_split_to_model`), it is updated in-place so
    that other transform entries (such as ``"mixed_precision_split"``) are
    preserved.

    Args:
        model: Quantized model to inspect.
        recipe_config_json: Optional JSON string of the
            :class:`~neuropress.recipe_config.QuantizationRecipeConfig` used to
            produce *model*.  Stored verbatim in the manifest.

    Returns:
        The :class:`QuantizationManifest` attached to *model*.
    """
    manifest = getattr(model, "__neuropress_manifest__", None)
    if manifest is None:
        manifest = QuantizationManifest()

    # Scan QTensor parameters and build the qtensor_assignment entry.
    qt_entries: dict[str, QtensorEntry] = {}
    for name, param in model.named_parameters():
        type_name = type(param).__name__
        serde = _SERDE_REGISTRY.get(type_name)
        if serde is not None:
            qt_entries[name] = QtensorEntry(
                type=type_name,
                tensor_keys=list(serde.to_tensors(param).keys()),
                metadata=serde.to_metadata(param),
            )

    # Replace (or add) the qtensor_assignment transform; preserve all others.
    # When the scan finds QTensor params, always update; when it finds none,
    # keep any existing entry (linearization may have consumed the QTensors
    # after a prior attach_manifest recorded them).
    if qt_entries:
        manifest.transforms = [
            t
            for t in manifest.transforms
            if not isinstance(t, QtensorAssignmentTransform)
        ]
        manifest.transforms.append(
            QtensorAssignmentTransform(
                type="qtensor_assignment", qtensor_params=qt_entries
            )
        )

    if recipe_config_json is not None:
        manifest.recipe_config_json = recipe_config_json

    model.__neuropress_manifest__ = manifest  # type: ignore[attr-defined]
    return manifest


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_quantized_model(
    model: nn.Module,
    path: str | Path,
    recipe_config_json: str | None = None,
) -> None:
    """Serialise *model* to a safetensors checkpoint at *path*.

    If the model has been kernel-lowered (NP* modules replacing QTensor-backed
    ``nn.Linear`` / ``nn.Embedding``), it is automatically unlowered before
    saving and re-lowered afterwards.  This makes saving transparent — callers
    never need to manage the lower/unlower cycle manually.

    Regular parameters are stored under their standard ``state_dict`` paths.
    QTensor parameters (those whose type is registered in
    :data:`~neuropress.t2n.qtensors._SERDE_REGISTRY`) are decomposed into named
    sub-tensors via :class:`~neuropress.t2n.qtensors.QTensorSerde` and stored
    under ``{param_path}::{serde_tensor_key}`` keys; their non-tensor metadata
    is embedded in the JSON manifest.

    Before saving, :func:`attach_manifest` is called to ensure the manifest
    reflects the current QTensor state.  Any transforms already recorded on the
    model (e.g. a ``"mixed_precision_split"`` from
    :func:`~neuropress.transforms.split_mixed_linear.apply_split_to_model`) are
    preserved in the saved manifest.

    Args:
        model: A (quantized) model to serialise.
        path: Destination ``.safetensors`` file path.
        recipe_config_json: Optional JSON of the
            :class:`~neuropress.recipe_config.QuantizationRecipeConfig` used to
            quantize the model.  Overrides any manifest already attached to
            *model*.
    """
    from ..kernel_lowering import is_lowered, unlower_model

    lowering = getattr(model, "__neuropress_kernel_lowering__", None)
    was_lowered = is_lowered(model)

    if was_lowered:
        # Unlower on the current device (GPU).  unlower_model reconstructs
        # compact QTensorTractVQ objects (centroids + packed indices) without
        # materialising fp16 weights.  nn.Linear / nn.Embedding shells are
        # built with device="meta" to avoid allocating a throwaway weight
        # tensor that would OOM for large layers (e.g. lm_head [262144, 8192]
        # fp32 = 8 GiB).  Moving the model to CPU first (old approach) caused
        # multi-hour hangs under Python GC pressure with 29+ GB RSS (CUDA
        # synchronisation deadlock during model._apply across 236+ threads).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        unlower_model(model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        # Ensure manifest is current (scans QTensors, preserves other transforms).
        attach_manifest(model, recipe_config_json=recipe_config_json)
        manifest = model.__neuropress_manifest__  # type: ignore[attr-defined]

        tensors: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            type_name = type(param).__name__
            serde = _SERDE_REGISTRY.get(type_name)
            if serde is not None:
                for tensor_key, tensor in serde.to_tensors(param).items():
                    tensors[f"{name}{_QT_SEP}{tensor_key}"] = tensor
            else:
                tensors[name] = param.data

        # Include non-parameter buffers (e.g. BatchNorm running stats).
        for name, buf in model.named_buffers():
            if buf is not None and name not in tensors:
                tensors[name] = buf

        rc_json = manifest.recipe_config_json
        rc = json.loads(rc_json) if rc_json else None
        manifest_dict: dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "transforms": [
                t.model_dump() if isinstance(t, BaseModel) else t
                for t in manifest.transforms
            ],
            "recipe_config": rc,
        }
        # Move to CPU before serialising — safetensors pins GPU memory for the
        # full tensor dict, which OOMs on 24 GB cards after a long eval pass.
        tensors_cpu = {k: v.cpu() for k, v in tensors.items()}
        save_file(
            tensors_cpu,
            path,
            metadata={_MANIFEST_META_KEY: json.dumps(manifest_dict)},
        )
    finally:
        # Re-lower so the model remains in the fast-inference state,
        # even if save_file raised (disk full, permissions, etc.).
        if was_lowered and lowering is not None:
            lowering.apply(model)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _set_param(model: nn.Module, param_path: str, tensor: Any) -> None:
    r"""Assign *tensor* to the parameter at *param_path* in *model*.

    Navigates the module hierarchy by splitting *param_path* on ``"."``, then
    assigns directly via ``_parameters`` or ``_buffers`` — bypassing
    ``nn.Module.__setattr__`` which would reject non-``Parameter`` tensors
    (such as QTensors).

    Args:
        model: Root module.
        param_path: Dotted path, e.g. ``"layers.0.self_attn.q_proj.weight"``.
        tensor: Replacement tensor (may be a QTensor).
    """
    parts = param_path.split(".")
    mod: nn.Module = model
    for part in parts[:-1]:
        mod = getattr(mod, part)
    attr_name = parts[-1]
    if attr_name in mod._parameters:
        mod._parameters[attr_name] = tensor
    elif attr_name in mod._buffers:
        mod._buffers[attr_name] = tensor
    else:
        setattr(mod, attr_name, tensor)


def _set_module(model: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """Replace the sub-module at *module_path* with *new_module*.

    Args:
        model: Root module.
        module_path: Dotted path, e.g. ``"layers.0.fc"``.
        new_module: Replacement module.
    """
    parts = module_path.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _replay_mixed_precision_split(
    model: nn.Module, transform: MixedPrecisionSplitTransform
) -> None:
    """Recreate :class:`MixedPrecisionLinear` modules from a saved transform.

    Replaces the ``nn.Linear`` at each recorded path with a
    :class:`~neuropress.transforms.split_mixed_linear.MixedPrecisionLinear`
    whose sub-linears have placeholder weights.  The actual weights and
    ``column_perm`` buffer are filled in by the subsequent
    :meth:`~torch.nn.Module.load_state_dict` call.

    Args:
        model: Target model (must have the same architecture as when saved).
        transform: The ``"mixed_precision_split"`` transform from the manifest.
    """
    from ..transforms.split_mixed_linear import MixedPrecisionLinear

    for path, spec in transform.modules.items():
        out = spec.out_features
        split_sizes: list[int] = spec.split_sizes
        precision_nbits: list[int] = spec.precision_nbits
        has_bias: bool = spec.has_bias

        sub_linears = nn.ModuleList(
            [nn.Linear(sz, out, bias=False) for sz in split_sizes]
        )
        # Placeholder tensors — overwritten by load_state_dict.
        column_perm = torch.zeros(sum(split_sizes), dtype=torch.long)
        bias = torch.zeros(out) if has_bias else None

        mpl = MixedPrecisionLinear(
            column_perm=column_perm,
            sub_linears=sub_linears,
            split_sizes=split_sizes,
            precision_nbits=precision_nbits,
            bias=bias,
        )
        _set_module(model, path, mpl)


# ---------------------------------------------------------------------------
# Kernel lowering resolution for load
# ---------------------------------------------------------------------------


def _resolve_kernel_lowering(
    kernel_lowering: str,
    recipe_config_raw: dict[str, Any] | None,
) -> Any:
    """Resolve a kernel-lowering specifier to a runtime object.

    When *kernel_lowering* is ``"auto"``, the function first tries to
    reconstruct the :class:`~neuropress.kernel_lowering.KernelLowering`
    from the saved ``recipe_config`` dict.  If the config is missing or
    has no ``kernel_lowering`` key, an
    :class:`~neuropress.kernel_lowering.AutoKernelLowering` with default
    backends is returned as a safe fallback.

    Args:
        kernel_lowering: ``"auto"`` to auto-detect.
        recipe_config_raw: The ``"recipe_config"`` dict from the manifest,
            or ``None`` when the checkpoint has no recipe config.

    Returns:
        A ready-to-use :class:`~neuropress.kernel_lowering.KernelLowering`.
    """
    from ..kernel_lowering import AutoKernelLowering

    if kernel_lowering != "auto":
        raise ValueError(
            f"kernel_lowering must be 'auto' or None, got {kernel_lowering!r}"
        )

    # Always use fresh AutoKernelLowering with "auto" backends so the
    # resolver picks the best backend for the *current* hardware — not
    # whatever was saved in the checkpoint (which may have been a different
    # machine or lacked nvcc at save time).
    return AutoKernelLowering()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_quantized_model(
    model: nn.Module,
    path: str | Path,
    kernel_lowering: str | None = "auto",
) -> QuantizationManifest:
    """Load a safetensors checkpoint into *model*, restoring quantized state.

    Processing order:

    1. Structural transforms (e.g. ``"mixed_precision_split"``) are replayed
       first to recreate the correct module hierarchy.
    2. Regular tensors are loaded via
       :meth:`~torch.nn.Module.load_state_dict` (``strict=False``).
    3. QTensor parameters are reconstructed from their stored sub-tensors using
       the registered :class:`~neuropress.t2n.qtensors.QTensorSerde` and
       assigned directly to the model.
    4. If *kernel_lowering* is not ``None``, QTensor-backed modules are
       replaced with hardware-optimised NP* modules for fast inference.

    *model* should be the same architecture as when the checkpoint was saved
    (typically a freshly initialised float model).  The
    :class:`QuantizationManifest` from the checkpoint is re-attached as
    ``model.__neuropress_manifest__``.

    Args:
        model: Target model (float/unquantized) to load the checkpoint into.
        path: Path to a ``.safetensors`` file written by
            :func:`save_quantized_model`.
        kernel_lowering: Controls hardware kernel lowering after loading
            weights.  ``"auto"`` (default) reconstructs the lowering pass
            from the saved recipe config, falling back to
            :class:`~neuropress.kernel_lowering.AutoKernelLowering`.
            ``None`` skips lowering — useful for inspection or re-saving.

    Returns:
        The :class:`QuantizationManifest` stored in the checkpoint.

    Raises:
        ValueError: If the file is missing ``__neuropress_manifest__``, or a
            QTensor type has no registered serde handler.
    """
    with safe_open(str(path), framework="pt") as f:
        raw_meta = f.metadata() or {}
        all_keys = list(f.keys())
        all_tensors = {k: f.get_tensor(k) for k in all_keys}

    manifest_str = raw_meta.get(_MANIFEST_META_KEY)
    if manifest_str is None:
        raise ValueError(
            f"File {path!r} is missing the {_MANIFEST_META_KEY!r} metadata key.  "
            "Was it written by save_quantized_model()?"
        )
    manifest_dict = json.loads(manifest_str)
    manifest_obj = QuantizationManifest.model_validate(
        {
            "transforms": manifest_dict.get("transforms", []),
        }
    )
    transforms_list = manifest_obj.transforms

    # -----------------------------------------------------------------------
    # Step 1: Structural transforms — must run before loading state.
    # -----------------------------------------------------------------------
    for t in transforms_list:
        if isinstance(t, MixedPrecisionSplitTransform):
            _replay_mixed_precision_split(model, t)

    # -----------------------------------------------------------------------
    # Step 2: Extract QTensor parameter info.
    # -----------------------------------------------------------------------
    qt_entries: dict[str, QtensorEntry] = {}
    for t in transforms_list:
        if isinstance(t, QtensorAssignmentTransform):
            qt_entries = t.qtensor_params
            break
    qt_param_paths = set(qt_entries)

    # -----------------------------------------------------------------------
    # Step 3: Regular parameters and buffers — keys without the QT separator.
    # -----------------------------------------------------------------------
    regular_state: dict[str, torch.Tensor] = {
        k: v
        for k, v in all_tensors.items()
        if _QT_SEP not in k and k not in qt_param_paths
    }
    model.load_state_dict(regular_state, strict=False)

    # -----------------------------------------------------------------------
    # Step 4: QTensor parameters — reconstruct and assign directly.
    # -----------------------------------------------------------------------
    for param_path, entry in qt_entries.items():
        serde = _SERDE_REGISTRY.get(entry.type)
        if serde is None:
            raise ValueError(
                f"No QTensorSerde registered for type {entry.type!r} "
                f"(needed for parameter {param_path!r}).  "
                f"Available: {sorted(_SERDE_REGISTRY)}"
            )
        sub_tensors = {
            tk: all_tensors[f"{param_path}{_QT_SEP}{tk}"] for tk in entry.tensor_keys
        }
        q = serde.from_parts(sub_tensors, entry.metadata)
        _set_param(model, param_path, q)

    # -----------------------------------------------------------------------
    # Step 5: Re-attach manifest.
    # -----------------------------------------------------------------------
    rc_raw = manifest_dict.get("recipe_config")
    manifest_obj.recipe_config_json = json.dumps(rc_raw) if rc_raw is not None else None
    model.__neuropress_manifest__ = manifest_obj  # type: ignore[attr-defined]

    # -----------------------------------------------------------------------
    # Step 6: Kernel lowering (optional, on by default).
    # -----------------------------------------------------------------------
    if kernel_lowering is not None:
        _log.warning("Weights loaded — applying kernel lowering…")
        lowering = _resolve_kernel_lowering(kernel_lowering, rc_raw)
        model.__neuropress_kernel_lowering__ = lowering  # type: ignore[attr-defined]
        lowering.apply(model)

    return manifest_obj
