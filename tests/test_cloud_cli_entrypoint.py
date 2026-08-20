"""The neuropress-cloud entry point must explain a missing 'cloud' extra.

Console scripts are installed regardless of extras, so `pip install neuropress`
followed by `neuropress-cloud` used to raise a bare ModuleNotFoundError.
"""

from __future__ import annotations

import builtins
import sys
import tomllib
from pathlib import Path

import pytest

from neuropress.cloud import cli

# Modules whose import must be re-attempted rather than served from cache.
_RELOAD = ("neuropress.cloud.vast_runner", "neuropress.cloud.runner")


def _blocking_import(blocked: str):
    """Return an __import__ raising ModuleNotFoundError for *blocked*.

    Matches the full dotted name or its root package, so it can stand in for a
    missing third-party distribution or for breakage inside a submodule.
    """
    real = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name == blocked or name.split(".")[0] == blocked:
            raise ModuleNotFoundError(f"No module named {blocked!r}", name=blocked)
        return real(name, globals, locals, fromlist, level)

    return fake


def _arm(monkeypatch, blocked: str) -> None:
    for name in _RELOAD:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import(blocked))


def test_missing_cloud_extra_exits_with_install_hint(monkeypatch):
    _arm(monkeypatch, "rich")

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    message = str(exc_info.value)
    assert "neuropress[cloud]" in message
    assert "rich" in message


def test_breakage_inside_the_cloud_package_is_not_swallowed(monkeypatch):
    """A guard that hides real breakage is worse than the traceback it replaces.

    An import cycle that leaves a module partially initialised surfaces as an
    ImportError too; translating that into "install the cloud extra" would send
    the user chasing a dependency that is already present.
    """
    _arm(monkeypatch, "neuropress.cloud.runner")

    with pytest.raises(ModuleNotFoundError) as exc_info:
        cli.main()

    assert exc_info.value.name == "neuropress.cloud.runner"


def test_entry_point_is_wired_to_the_guard():
    """pyproject must point neuropress-cloud at this module, not vast_runner."""
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    assert (
        pyproject["project"]["scripts"]["neuropress-cloud"]
        == "neuropress.cloud.cli:main"
    )
