"""--deterministic turns on deterministic algorithms before any CUDA work."""

import os

import torch

from neuropress.eval.llm_cli import _enable_deterministic, build_parser


def test_flag_parses():
    args = build_parser().parse_args(
        ["compare", "m", "-t", "wikitext", "--compression", "x", "--deterministic"]
    )
    assert args.deterministic is True
    default = build_parser().parse_args(
        ["compare", "m", "-t", "wikitext", "--compression", "x"]
    )
    assert default.deterministic is False


def test_enable_deterministic(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        _enable_deterministic()
        assert torch.are_deterministic_algorithms_enabled()
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    finally:
        torch.use_deterministic_algorithms(False)
