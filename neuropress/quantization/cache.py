"""Pluggable cache manager for quantization intermediate results.

Each estimator declares which stages it can cache via :meth:`cache_stages`.
The pipeline wires up a :class:`CacheManager` and calls :meth:`set_cache` on
the estimator before running.  Concrete estimators use ``self._cache``
transparently; without configuration a :class:`NullCacheManager` is used and
behaviour is identical to having no cache at all.

Layout on disk (:class:`DiskCacheManager`)::

    {cache_dir}/
      cal_{cal_fingerprint}/      # stages whose scope is "calibration"
        hessian/
          {key}.safetensors
        hessian_diag/
          {key}.safetensors
      recipe_{recipe_fingerprint}/  # stages whose scope is "recipe"
        centroids/
          {key}.safetensors
        assignments/
          {key}.safetensors

Two fingerprints are computed automatically from the recipe:

* **cal_fingerprint** — SHA-256 of the calibration config fields only
  (dataset, n_samples, seq_len, seed, batch_size).  Changing quantization
  hyperparameters (n_iter, num_centroids, …) does not change this; all runs
  sharing the same calibration config share the same hessian cache.

* **recipe_fingerprint** — SHA-256 of the full ``recipe_config_json`` string.
  Any change to any hyperparameter (including calibration) produces a new
  fingerprint and a fresh centroid/assignment cache, while the hessian cache
  (keyed on cal_fingerprint) remains reusable.

Key sanitisation: dots → ``__``, slashes → ``_`` so module names like
``model.layers.0.mlp.down_proj`` map to safe filenames.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import torch

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheStageDescriptor:
    """Metadata describing one cacheable computation stage.

    Attributes:
        name: Stage identifier used as a directory name and in ``has``/``load``/``save`` calls.
        description: Human-readable description shown in logs.
        scope: Determines which fingerprint scopes this stage.

            * ``"calibration"`` — keyed on calibration config only.
              Shared across runs that differ only in quantization HP.
              Use for hessians.
            * ``"recipe"`` — keyed on the full recipe config.
              Isolated whenever any HP changes.
              Use for centroids and assignments.
    """

    name: str
    description: str
    scope: Literal["calibration", "recipe"] = "recipe"


def _cal_fingerprint(cal_config: Any) -> str:
    """Short SHA-256 of calibration-only fields (dataset, n_samples, seq_len, seed, batch_size)."""
    data = json.dumps(
        {
            "dataset": getattr(cal_config, "dataset", None),
            "n_samples": getattr(cal_config, "n_samples", None),
            "seq_len": getattr(cal_config, "seq_len", None),
            "seed": getattr(cal_config, "seed", None),
            "batch_size": getattr(cal_config, "batch_size", None),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(data).hexdigest()[:12]


def _recipe_fingerprint(recipe_config_json: str) -> str:
    """Short SHA-256 of the full recipe config JSON string."""
    return hashlib.sha256(recipe_config_json.encode()).hexdigest()[:12]


@runtime_checkable
class CacheManager(Protocol):
    """Protocol for cache backends."""

    def has(self, stage: str, key: str) -> bool:
        """Return True if the given stage/key is present in the cache."""
        ...

    def load(self, stage: str, key: str) -> dict[str, torch.Tensor] | None:
        """Return the cached tensor dict, or None on miss."""
        ...

    def save(self, stage: str, key: str, data: dict[str, torch.Tensor]) -> None:
        """Persist *data* under *stage*/*key*."""
        ...


class NullCacheManager:
    """No-op cache: every ``has`` returns False, ``save`` is a no-op."""

    def has(self, stage: str, key: str) -> bool:
        """Return ``False``: a null cache never holds anything."""
        return False

    def load(self, stage: str, key: str) -> dict[str, torch.Tensor] | None:
        """Return ``None``: a null cache never holds anything."""
        return None

    def save(self, stage: str, key: str, data: dict[str, torch.Tensor]) -> None:
        """Discard *data*: a null cache does not persist."""


# Module-level singleton — avoids allocating one per estimator instantiation.
_NULL_CACHE = NullCacheManager()


class DiskCacheManager:
    """Persists cache stages as safetensors files under *cache_dir*.

    Stages are routed to one of two fingerprinted subdirectories depending on
    their declared :attr:`~CacheStageDescriptor.scope`:

    * ``"calibration"`` stages → ``cal_{cal_fingerprint}/``
    * ``"recipe"`` stages → ``recipe_{recipe_fingerprint}/``

    Args:
        cache_dir: Root directory.  Model-level isolation is the caller's
            responsibility (e.g. ``data/cache/gemma4_e2b/``).
        cal_fingerprint: Short hash of the calibration config; computed via
            :func:`_cal_fingerprint`.
        recipe_fingerprint: Short hash of the full recipe config; computed via
            :func:`_recipe_fingerprint`.
        stage_scopes: Mapping ``{stage_name: scope}`` built from the estimator's
            :meth:`cache_stages` descriptors.  Stages absent from this dict
            default to ``"recipe"`` scope.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        cal_fingerprint: str,
        recipe_fingerprint: str,
        stage_scopes: dict[str, Literal["calibration", "recipe"]],
    ) -> None:
        self._root = Path(cache_dir)
        self._cal_dir = self._root / f"cal_{cal_fingerprint}"
        self._recipe_dir = self._root / f"recipe_{recipe_fingerprint}"
        self._stage_scopes = stage_scopes
        LOGGER.info(
            "DiskCacheManager  root=%s  cal=%s  recipe=%s",
            self._root,
            cal_fingerprint,
            recipe_fingerprint,
        )

    def _path(self, stage: str, key: str) -> Path:
        scope = self._stage_scopes.get(stage, "recipe")
        base = self._cal_dir if scope == "calibration" else self._recipe_dir
        safe = key.replace("/", "_").replace(".", "__")
        return base / stage / f"{safe}.safetensors"

    def has(self, stage: str, key: str) -> bool:
        """Return ``True`` if a safetensors file exists for *stage*/*key*."""
        return self._path(stage, key).exists()

    def load(self, stage: str, key: str) -> dict[str, torch.Tensor] | None:
        """Return the cached tensor dict, or ``None`` on miss or read failure."""
        from safetensors.torch import load_file

        path = self._path(stage, key)
        if not path.exists():
            return None
        try:
            data = dict(load_file(str(path)))
            LOGGER.debug("cache hit  %s/%s", stage, key)
            return data
        except Exception as exc:
            LOGGER.warning("cache load failed  %s/%s: %s", stage, key, exc)
            return None

    def save(self, stage: str, key: str, data: dict[str, torch.Tensor]) -> None:
        """Persist *data* under *stage*/*key*; log and continue on failure."""
        from safetensors.torch import save_file

        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cpu_data = {k: v.detach().contiguous().cpu() for k, v in data.items()}
            save_file(cpu_data, str(path))
            size_mb = path.stat().st_size / (1024 * 1024)
            LOGGER.info("cache save  %s/%s  %.1f MB", stage, key, size_mb)
        except Exception as exc:
            LOGGER.warning("cache save failed  %s/%s: %s", stage, key, exc)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
