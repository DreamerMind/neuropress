"""Tensor-tree batches shared by calibration collection and block execution."""

from typing import Any

import torch
from torch import nn


def move_batch(
    value: Any,
    device: torch.device | str,
    *,
    copy: bool = False,
    tensor_subclasses: bool = True,
) -> Any:
    """Detach and move tensors inside dictionaries, tuples and lists.

    Containers come back as plain ``dict``/``tuple``/``list``: a Hugging Face
    ``ModelOutput`` becomes a ``dict`` (index it by key, not attribute) and a
    namedtuple becomes a ``tuple``. This is deliberate: block streams are
    checkpointed and reloaded with ``torch.load(weights_only=True)``, which
    rejects container subclasses, so preserving them would make a resumed run
    see different types than a fresh one.

    With ``tensor_subclasses=False``, only plain ``torch.Tensor`` and
    ``nn.Parameter`` leaves move and subclasses are returned untouched: for
    ``OffloadedTensor``, even ``.detach()`` reloads the data from disk.
    """
    if isinstance(value, torch.Tensor):
        if not tensor_subclasses and type(value) not in (torch.Tensor, nn.Parameter):
            return value
        return value.detach().to(device, copy=copy)
    options = {"copy": copy, "tensor_subclasses": tensor_subclasses}
    if isinstance(value, dict):
        return {key: move_batch(item, device, **options) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch(item, device, **options) for item in value)
    if isinstance(value, list):
        return [move_batch(item, device, **options) for item in value]
    return value


def call_batch(model: nn.Module, batch: Any) -> Any:
    """Call a model with a tensor, positional arguments, or keyword arguments."""
    if isinstance(batch, dict):
        return model(**batch)
    if isinstance(batch, list | tuple):
        return model(*batch)
    return model(batch)
