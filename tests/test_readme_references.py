"""The README's documented dotted paths must actually resolve.

`compression_registry` is a string that torch_to_nnef imports at runtime, so a
stale value fails only for the user copying the example. The registry moved from
neuropress/compress.py to neuropress/t2n/registry.py and the README kept the old
path; this pins it.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

_DOTTED = re.compile(r'compression_registry="([\w.]+)"')


def _documented_registry_paths() -> list[str]:
    return _DOTTED.findall(README.read_text())


def test_readme_documents_a_registry_path():
    """Guard the guard: a regex that matches nothing would pass silently."""
    assert _documented_registry_paths(), "no compression_registry= example in README"


@pytest.mark.parametrize("dotted", _documented_registry_paths())
def test_documented_registry_path_resolves(dotted):
    module_name, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    registry = getattr(module, attr, None)
    assert registry is not None, f"{dotted} does not resolve"
    assert registry, f"{dotted} resolves to an empty registry"


@pytest.mark.parametrize("dotted", _documented_registry_paths())
def test_documented_methods_exist_in_that_registry(dotted):
    """Method names quoted next to the registry path must be keys in it."""
    module_name, _, attr = dotted.rpartition(".")
    registry = getattr(importlib.import_module(module_name), attr)
    quoted = re.findall(r'compression_method="([\w]+)"', README.read_text())
    assert quoted, "no compression_method= example in README"
    missing = [m for m in quoted if m not in registry]
    assert not missing, f"README documents unknown methods: {missing}"
