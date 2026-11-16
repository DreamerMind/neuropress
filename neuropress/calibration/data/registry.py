"""Calibration dataset registry.

Each entry maps a string key to a loader callable with the contract::

    loader(tokenizer, *, seq_len, n_samples, batch_size, seed, ...) -> Iterator[dict]

The registry is populated explicitly by :mod:`neuropress.calibration.data`::

    DATASET_REGISTRY.update({"c4": load_c4_calibration})

Use the ``dataset`` keyword in compression functions to select a loader::

    NEUROPRESS_COMPRESSION["open_text_c4_gptq_q4_0"](model, tokenizer=tok, dataset="c4")

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DATASET_REGISTRY: dict[str, Callable[..., Any]] = {}


TOKEN_FILE_PREFIX = "tokens:"


def resolve_calibration_loader(name: str) -> Callable[..., Any]:
    """Return the loader for a calibration dataset name.

    ``"tokens:<path>"`` selects :func:`~.llm.load_token_file_calibration` on
    *path*; any other name is looked up in :data:`DATASET_REGISTRY`.

    Raises:
        KeyError: If *name* is neither registered nor a ``tokens:`` path.
    """
    if name.startswith(TOKEN_FILE_PREFIX):
        from functools import partial

        from .llm import load_token_file_calibration

        return partial(load_token_file_calibration, path=name[len(TOKEN_FILE_PREFIX) :])
    return DATASET_REGISTRY[name]
