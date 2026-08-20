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
