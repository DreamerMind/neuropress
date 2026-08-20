"""Tests that the ``neuropress-eval size`` CLI wiring stays intact (no network).

``_cmd_size`` imports its helpers from :mod:`neuropress.eval.size_estimate` lazily,
inside the function body, so a renamed or removed symbol there is invisible until
someone actually runs the command.  These tests pin that contract.
"""

import argparse
import ast
import inspect
from pathlib import Path

import neuropress.eval.llm_cli as llm_cli
from neuropress.eval import size_estimate


def _size_dtype_choices() -> list[str]:
    """Return the ``--dtype`` choices declared for the ``size`` subcommand."""
    parser = llm_cli.build_parser()
    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    assert subparsers, "expected the CLI to define subcommands"
    size_parser = subparsers[0].choices["size"]
    for action in size_parser._actions:
        if action.dest == "dtype":
            return list(action.choices or [])
    raise AssertionError("size subcommand has no --dtype argument")


def test_cmd_size_lazy_imports_all_resolve():
    """Every name ``_cmd_size`` imports from size_estimate must exist."""
    source = inspect.getsource(llm_cli._cmd_size)
    tree = ast.parse(source.strip())

    imported: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "neuropress.eval.size_estimate"
        ):
            imported.extend(alias.name for alias in node.names)

    assert imported, "expected _cmd_size to import from size_estimate"
    missing = [name for name in imported if not hasattr(size_estimate, name)]
    assert not missing, f"_cmd_size imports names absent from size_estimate: {missing}"


def test_dtype_bytes_covers_cli_choices():
    """``_DTYPE_BYTES`` must cover every ``--dtype`` choice except ``auto``."""
    choices = set(_size_dtype_choices())
    assert "auto" in choices, "expected 'auto' to remain a --dtype choice"

    # "auto" is deliberately absent: it falls back to the config-inferred value.
    assert "auto" not in size_estimate._DTYPE_BYTES
    assert choices - {"auto"} == set(size_estimate._DTYPE_BYTES)
    assert all(v > 0 for v in size_estimate._DTYPE_BYTES.values())


def test_size_estimate_prefers_new_dtype_config_attr():
    """The dtype lookup must read ``config.dtype`` before legacy ``torch_dtype``.

    transformers 5.x renamed the attribute and warns on the old name.
    """
    source = Path(inspect.getfile(size_estimate)).read_text()
    load_meta = source.split("def load_model_meta")[1]
    new_pos = load_meta.find('getattr(config, "dtype"')
    old_pos = load_meta.find('getattr(config, "torch_dtype"')
    assert new_pos != -1, "load_model_meta should read the modern config.dtype"
    assert old_pos != -1, "the legacy torch_dtype fallback should be kept"
    assert new_pos < old_pos, "config.dtype must be tried before torch_dtype"
