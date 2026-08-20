"""Monkey-patches for ``OpaqueTensor`` and ``QTensor`` device handling.

Importing this module patches two upstream issues in ``torch_to_nnef``:

1. ``OpaqueTensor.data`` setter — raises unconditionally, preventing
   ``nn.Module.to(device)`` from working (``_apply`` does
   ``param.data = fn(param)``).

2. ``QTensor.to()`` — performs an expensive dequant/requant round-trip
   even for device-only moves.

These patches are intended as prototypes for eventual upstream fixes.
"""

from __future__ import annotations

import logging

import torch
from torch_to_nnef.tensor.opaque import OpaqueTensor
from torch_to_nnef.tensor.quant import QTensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. data property — permissive setter
# ---------------------------------------------------------------------------
# Both OpaqueTensor and QTensor define a ``data`` property whose setter
# raises unconditionally.  ``nn.Module._apply`` (used by ``.to()``) does
# ``param.data = fn(param)``, so the setter must accept the new value.
# We patch both classes: QTensor (which overrides OpaqueTensor) for all
# QTensor instances, and OpaqueTensor for any future non-QTensor opaques.


def _make_data_property(cls: type) -> property:
    """Build a ``data`` property for *cls* with a permissive setter."""
    original_fget = cls.__dict__["data"].fget

    def _data_get(self: OpaqueTensor) -> OpaqueTensor:
        return original_fget(self)

    def _data_set(self: OpaqueTensor, new_data: torch.Tensor) -> None:
        if new_data is self:
            return

        if isinstance(new_data, QTensor):
            # Copy internal QTensor state from *new_data* into *self*.
            self.u8_blob = new_data.u8_blob
            self.qscheme = new_data.qscheme
            self.u8_compressors = new_data.u8_compressors
            self.dequant_to_dtype = new_data.dequant_to_dtype
            self.nnef_name = new_data.nnef_name
            return

        # Regular tensor — delegate to torch.Tensor.data setter.
        torch.Tensor.data.__set__(self, new_data)

    return property(_data_get, _data_set)


OpaqueTensor.data = _make_data_property(OpaqueTensor)  # type: ignore[assignment]
QTensor.data = _make_data_property(QTensor)  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# 2. QTensor.to() — efficient device-only moves
# ---------------------------------------------------------------------------

_original_to = QTensor.to


def _qtensor_to(self: QTensor, *args: object, **kwargs: object) -> QTensor:
    """Move QTensor to a device without dequant/requant when possible."""
    new_dtype = kwargs.get("dtype")
    new_device = kwargs.get("device")

    if args:
        for arg in args:
            if isinstance(arg, torch.device | str):
                new_device = arg
            elif isinstance(arg, torch.dtype):
                new_dtype = arg
            elif isinstance(arg, torch.Tensor):
                new_dtype = arg.dtype
                new_device = arg.device
            elif isinstance(arg, type(None) | bool):
                continue

    # Dtype change requires dequant/requant — delegate to original.
    if new_dtype is not None and new_dtype != self.dtype:
        return _original_to(self, *args, **kwargs)

    # Normalise device for comparison.
    if new_device is not None:
        new_device = torch.tensor(0.0).to(new_device).device
        if new_device != self.device:
            self.to_device(new_device)

    return self


QTensor.to = _qtensor_to  # type: ignore[assignment]

logger.debug("Patched OpaqueTensor.data and QTensor.to for device handling")
