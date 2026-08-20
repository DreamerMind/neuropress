"""Entry point for the ``neuropress-cloud`` console script.

Console scripts are installed regardless of which extras were selected, so
``pip install neuropress`` puts ``neuropress-cloud`` on PATH even though the
cloud runner needs the ``cloud`` extra.  :mod:`.vast_runner` imports ``rich``
and ``textual`` at module scope, so the check has to happen here, before that
import, to turn a bare ``ModuleNotFoundError`` traceback into something the
user can act on.
"""

from __future__ import annotations

import sys

#: Import names provided by the ``cloud`` extra.  Only ``rich`` is reachable
#: today -- ``vastai_sdk`` and ``textual`` are imported lazily, inside the
#: functions that need them -- but list them all so the message stays correct if
#: one of those imports ever moves to module scope.
_CLOUD_MODULES = frozenset({"rich", "textual", "vastai_sdk"})


def main() -> None:
    """Run the cloud runner, reporting a missing ``cloud`` extra clearly."""
    try:
        from .vast_runner import main as _run
    except ImportError as exc:
        # Translate *only* a missing optional dependency.  A guard that
        # swallows every ImportError also hides real breakage inside
        # vast_runner -- a partial initialisation from an import cycle, say --
        # which is far harder to diagnose than the traceback it replaced.
        if (exc.name or "").split(".")[0] not in _CLOUD_MODULES:
            raise
        sys.exit(
            f"neuropress-cloud requires the 'cloud' extra ({exc.name} is not installed).\n"
            "Install it with:\n"
            "  pip install 'neuropress[cloud]'"
        )
    _run()
