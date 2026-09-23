"""Hessian accumulator — standalone statistics collector.

Serves as both the registry ``Collector`` for "hessian" stat type and as the
composable accumulator inside ``GPTQCalibration``.
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.types import Device

from .collectors import Collector

LOGGER = logging.getLogger(__name__)

_HAS_CUDA: bool = torch.cuda.is_available()
try:
    _HAS_MPS: bool = torch.backends.mps.is_available()
except AttributeError:
    _HAS_MPS = False


def _is_conv1d_transpose(module_class: type | None) -> bool:
    """Return True for Conv1d-like transposed weight modules (e.g. GPT-2 Conv1D)."""
    if module_class is None:
        return False
    return module_class.__name__ == "Conv1D"


@dataclass
class HessianAccumulator(Collector):
    """Accumulates an empirical Hessian approximation from input activations.

    Implements the ``Collector`` contract so it can be used directly by
    ``StatisticsRegistry`` for the "hessian" stat type.  When ``module_class``
    is set, module-specific preprocessing (Linear transpose, Conv2d unfold,
    Conv1D transpose) is applied; otherwise a generic [batch, features] →
    [features, batch] transposition is used.

    Can also be composed inside calibration methods (e.g. ``GPTQCalibration``)
    for finer-grained weight-aware accumulation.
    """

    # Context (set by the owning CalibrationMethod before use; optional for generic use)
    weight_tensor: torch.Tensor | None = field(default=None, repr=False)
    module_class: type | None = field(default=None, repr=False)
    module: nn.Module | None = field(default=None, repr=False)
    name: str = field(default="")
    device: Device | None = field(default=None)
    offload_intermediate_tensor_on_disk: bool = field(default=False)
    # Class-level shared offload directory (all instances share one temp dir).
    _offload_dir: Path | None = field(default=None, repr=False)

    # Accumulation state
    hessian: torch.Tensor | None = field(default=None, repr=False)
    nsamples: int = field(default=0)
    observer_n_call: int = field(default=0)

    # ------------------------------------------------------------------
    # Cached weight properties (used by GPTQCalibration delegation)
    # ------------------------------------------------------------------

    @cached_property
    def rows(self) -> int:
        """Number of output rows of the weight matrix."""
        return self._clone_weights().shape[0]

    @cached_property
    def columns(self) -> int:
        """Number of input columns of the weight matrix."""
        return self._clone_weights().shape[1]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset accumulation state and invalidate cached properties."""
        self.hessian = None
        self.nsamples = 0
        self.observer_n_call = 0
        self.__dict__.pop("rows", None)
        self.__dict__.pop("columns", None)

    # ------------------------------------------------------------------
    # Collector contract
    # ------------------------------------------------------------------

    def update(self, data: torch.Tensor) -> None:
        """Accumulate Hessian from one data batch.

        When ``module_class`` is set, applies module-specific preprocessing:
        - ``nn.Linear`` / Conv1D-transposed: reshape + transpose to [features, batch].
        - ``nn.Conv2d``: unfold + permute + flatten.

        When ``module_class`` is ``None``, falls back to a generic path:
        - 1-D → unsqueeze to [1, features]
        - 3-D [batch, seq, features] → reshape to [batch*seq, features]
        - Transpose to [features, batch]

        In both cases ``nsamples`` accumulates the effective batch dimension
        contributed by this call.
        """
        self.observer_n_call += 1

        if self.device:
            data = data.to(device=self.device)

        if self.module_class is not None and (
            issubclass(self.module_class, nn.Linear)
            or _is_conv1d_transpose(self.module_class)
        ):
            if data.dim() == 2:
                data = data.unsqueeze(0)
            batch_size = data.shape[0]
            if data.dim() in (3, 4):
                data = data.reshape(-1, data.shape[-1])
            data = data.t()

        elif self.module_class is not None and issubclass(self.module_class, nn.Conv2d):
            if data.dim() == 2:
                data = data.unsqueeze(0)
            batch_size = data.shape[0]
            assert self.module is not None
            unfold = nn.Unfold(
                self.module.kernel_size,  # type: ignore[attr-defined]
                dilation=self.module.dilation,  # type: ignore[attr-defined]
                padding=self.module.padding,  # type: ignore[attr-defined]
                stride=self.module.stride,  # type: ignore[attr-defined]
            )
            data = unfold(data)
            data = data.permute([1, 0, 2]).flatten(1)

        else:
            # Generic path: no module type info available.
            if data.dim() == 1:
                data = data.unsqueeze(0)
            if data.dim() == 3:
                data = data.reshape(-1, data.shape[-1])
            data = data.t()
            batch_size = data.shape[1]  # actual number of samples

        n_features = data.shape[0]
        dev = self.device or data.device

        # Fall back to CPU accumulation when [K, K] fp32 won't fit on GPU.
        # _get_full_hessian loads it back to GPU explicitly when needed.
        if dev.type == "cuda":
            hessian_bytes = n_features * n_features * 4
            free_bytes, _ = torch.cuda.mem_get_info(dev)
            if hessian_bytes * 2 > free_bytes * 0.3:
                dev = torch.device("cpu")

        if self.hessian is None:
            hessian = torch.zeros(n_features, n_features, device=dev)
        else:
            hessian = self.hessian * self.nsamples / (self.nsamples + batch_size)
            if not isinstance(hessian, torch.Tensor):
                hessian = hessian.to_base_tensor()
            hessian = hessian.to(dev)

        self.nsamples += batch_size
        scale = math.sqrt(2 * batch_size / self.nsamples)
        data_f = (scale * data.to(dev)).float()
        hessian = hessian + data_f.matmul(data_f.t())

        if self.offload_intermediate_tensor_on_disk:
            from torch_to_nnef.tensor.offload import OffloadedTensor

            if isinstance(self.hessian, OffloadedTensor):
                self.hessian.update_values(hessian)
            else:
                self.hessian = OffloadedTensor.from_original_tensor(
                    hessian, f"{self.name}.hessian", self.offload_dir
                )
        else:
            self.hessian = hessian

        LOGGER.debug("update '%s' call #%d OK", self.name, self.observer_n_call)

    def result(self) -> dict[str, Any]:
        """Return accumulated statistics as a dict."""
        return {"hessian": self.hessian, "nsamples": self.nsamples}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def offload_dir(self) -> Path:
        """Shared temporary directory used for tensor offloading."""
        if HessianAccumulator._offload_dir is None:
            HessianAccumulator._offload_dir = Path(tempfile.mkdtemp(prefix="hessian_"))
        return HessianAccumulator._offload_dir

    def _clone_weights(self) -> torch.Tensor:
        assert self.weight_tensor is not None
        clone = self.weight_tensor.data.clone()
        if self.module_class is not None:
            if issubclass(self.module_class, nn.Conv2d):
                clone = clone.flatten(1)
            elif _is_conv1d_transpose(self.module_class):
                clone = clone.t()
        return clone.float()
