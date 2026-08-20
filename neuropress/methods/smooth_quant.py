"""SmoothQuant weight/activation co-migration preprocessing.

SmoothQuant is a structural transformation (not a :class:`.CalibrationMethod`)
that operates on pairs of layers.  It migrates quantization difficulty from
activations to weights by computing per-channel scales from calibration data
and absorbing them into the weight matrix and the preceding layer.

The per-channel scale formula (Xiao et al., ICML 2023) is::

    s_j = max|x_j|^α / max|w_j|^(1-α)

where ``x_j`` is the maximum absolute activation for input channel ``j``,
``w_j`` is the maximum absolute weight for that channel (across output rows),
and ``α ∈ [0, 1]`` controls the migration balance (0.5 = equal split).

Activation statistics are collected through :class:`.StatisticsRegistry` using
the ``"channel_max"`` stat type, keeping all hook lifecycle management
centralised in the registry.

Reference:
    SmoothQuant: Accurate and Efficient Post-Training Quantization for
    Large Language Models.
    Xiao et al., ICML 2023 – https://arxiv.org/abs/2211.10438
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from ..calibration.probe import StatisticRequest, StatisticsRegistry


def compute_smooth_scales(
    activation_max: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Compute per-channel SmoothQuant scales.

    Args:
        activation_max: Max absolute activation per input channel, shape
            ``[in_features]``.
        weight: Linear weight matrix of shape ``[out_features, in_features]``.
        alpha: Migration strength in ``[0, 1]``. ``0.5`` migrates equally
            between activations and weights.

    Returns:
        Scale tensor of shape ``[in_features]``.

    Raises:
        ValueError: If ``activation_max`` and the input-channel dimension of
            ``weight`` do not match.
    """
    in_features = weight.shape[1]
    if activation_max.shape != (in_features,):
        raise ValueError(
            f"activation_max shape {tuple(activation_max.shape)} does not match "
            f"weight in_features {in_features}"
        )
    weight_max = weight.abs().amax(dim=0)  # [in_features]
    scale = activation_max.pow(alpha) / weight_max.pow(1 - alpha).clamp(min=1e-8)
    return scale


def apply_scale_to_linear(linear: nn.Linear, scale: torch.Tensor) -> None:
    """Divide each column of ``linear.weight`` by ``scale`` in-place.

    Args:
        linear: The ``nn.Linear`` layer whose weight columns are rescaled.
        scale: Per-input-channel scale tensor of shape ``[in_features]``.
    """
    linear.weight.data /= scale.unsqueeze(0)  # broadcast over out_features


def absorb_scale_into_prev(
    prev: nn.Linear | nn.LayerNorm,
    scale: torch.Tensor,
) -> None:
    """Multiply ``prev``'s output-scaling parameters by ``scale`` in-place.

    Args:
        prev: The preceding layer. Supported types are ``nn.Linear`` and
            ``nn.LayerNorm``.
        scale: Per-channel scale tensor aligned with ``prev``'s output channels.

    Raises:
        TypeError: If ``prev`` is not an ``nn.Linear`` or ``nn.LayerNorm``.
    """
    if isinstance(prev, nn.Linear):
        prev.weight.data *= scale.unsqueeze(1)  # rows x 1 broadcast
        if prev.bias is not None:
            prev.bias.data *= scale
    elif isinstance(prev, nn.LayerNorm):
        prev.weight.data *= scale
        if prev.bias is not None:
            prev.bias.data *= scale
    else:
        raise TypeError(f"Unsupported preceding layer type: {type(prev)}")


def apply_smooth_quant(
    model: nn.Module,
    calibration_data: list[dict[str, torch.Tensor]],
    alpha: float = 0.5,
    prev_modules: dict[str, nn.Linear | nn.LayerNorm] | None = None,
    module_filter: Callable[[str, nn.Linear], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Apply SmoothQuant to all qualifying ``nn.Linear`` layers.

    Activation statistics are collected via :class:`.StatisticsRegistry` using
    the ``"channel_max"`` stat type, which ensures a single hook-managed
    forward pass with automatic cleanup — consistent with how all other
    calibration statistics are gathered in the pipeline.

    Args:
        model: The model to transform.
        calibration_data: List of input dicts passed as ``model(**batch)``.
        alpha: Migration strength in ``[0, 1]``. Defaults to ``0.5``.
        prev_modules: Optional mapping from linear layer name to its preceding
            layer. When provided, :func:`absorb_scale_into_prev` is called for
            each matching entry.
        module_filter: Optional callable ``(name, module) -> bool``. Only
            layers for which this returns ``True`` are processed. When
            ``None``, all ``nn.Linear`` layers are processed.

    Returns:
        Mapping from layer name to computed scale tensor of shape
        ``[in_features]``.

    Raises:
        RuntimeError: If no channel-max statistics were collected for a layer
            (e.g. because ``calibration_data`` is empty).
    """
    target_modules: dict[str, nn.Linear] = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (module_filter is None or module_filter(name, module))
    }

    registry = StatisticsRegistry()
    registry.register_requests(
        [
            StatisticRequest(target_key=name, stat_type="channel_max")
            for name in target_modules
        ]
    )
    registry.collect(model, calibration_data)

    scales: dict[str, torch.Tensor] = {}
    for name, module in target_modules.items():
        req = StatisticRequest(target_key=name, stat_type="channel_max")
        result = registry.get(req)
        act_max = result.get("channel_max") if isinstance(result, dict) else None
        if act_max is None:
            raise RuntimeError(
                f"No channel_max statistics collected for layer '{name}'. "
                "Ensure calibration_data is non-empty."
            )

        scale = compute_smooth_scales(
            act_max.cpu(), module.weight.data.cpu(), alpha=alpha
        )
        scales[name] = scale

        apply_scale_to_linear(module, scale.to(module.weight.device))

        if prev_modules is not None and name in prev_modules:
            absorb_scale_into_prev(prev_modules[name], scale.to(module.weight.device))

    return scales


def smooth_down_proj(
    model: nn.Module,
    calibration_data: list[dict[str, torch.Tensor]],
    alpha: float = 0.5,
    normalize_geomean: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply SmoothQuant scale migration to MLP ``down_proj`` layers only.

    ``down_proj`` is the layer where activation outliers concentrate - measured
    channel outlier ratios of 19-103x on Qwen3.5-2B - and water-filling against
    Hessian spectra shows a per-channel scale recovers 40-90% of the available
    anisotropy gain there.  It is also the one MLP layer with no norm or linear
    directly preceding it: its input is ``silu(gate) * up``.

    The migration is still exact, because ``up`` enters that product
    *linearly*.  Scaling ``up_proj``'s output rows by ``s`` multiplies
    ``down_proj``'s input by ``s``, exactly offsetting the division applied to
    ``down_proj``'s columns.  No norm is touched, so the ``1 + weight``
    convention is irrelevant here.

    Contrast :func:`apply_smooth_quant` with ``prev_modules=None``, which
    rescales weights with nothing absorbing the compensating factor and so
    silently changes the model's function.

    Args:
        model: Model to transform in place.
        calibration_data: Batches passed as ``model(**batch)``.
        alpha: Migration balance in ``[0, 1]``.
        normalize_geomean: Rescale each layer's scales to geometric mean 1, so
            only the *relative* per-channel structure changes and the layer's
            overall magnitude is preserved.  Required whenever a single codebook
            is shared across layers (``grouping="all"``): centroids are fitted to
            the pooled vector distribution, and an unnormalized scale pushes a
            layer's VQ-vector RMS off that pool, which costs real quality.
            The factor is a scalar per layer, so
            removing it cannot break exactness - it cancels between ``down_proj``
            and ``up_proj``.

    Returns:
        Mapping from ``down_proj`` layer name to its per-channel scale (after
        normalization, when enabled).
    """
    prev_modules: dict[str, nn.Linear] = {}
    for name, module in model.named_modules():
        if not (name.endswith("mlp.down_proj") and isinstance(module, nn.Linear)):
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        up_proj = getattr(parent, "up_proj", None)
        if isinstance(up_proj, nn.Linear):
            prev_modules[name] = up_proj

    scales = apply_smooth_quant(
        model,
        calibration_data,
        alpha=alpha,
        prev_modules=prev_modules,
        module_filter=lambda name, _module: name in prev_modules,
    )
    if not normalize_geomean:
        return scales

    for name, scale in scales.items():
        geomean = scale.double().clamp_min(1e-12).log().mean().exp().to(scale.dtype)
        down = model.get_submodule(name)
        up = prev_modules[name]
        # scales were applied as down /= s and up *= s; fold out the scalar so
        # the effective scale becomes s / geomean.  Exactness is unaffected.
        down.weight.data *= geomean.to(down.weight.device)
        up.weight.data /= geomean.to(up.weight.device)
        if up.bias is not None:
            up.bias.data /= geomean.to(up.bias.device)
        scales[name] = scale / geomean

    return scales
