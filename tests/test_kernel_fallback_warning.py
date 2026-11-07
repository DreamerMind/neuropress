"""Kernel fallbacks must warn once, with the cause and the fix."""

import logging

from neuropress.kernels import warn_kernel_fallback


def test_warns_once_with_cause_and_hint(caplog):
    caplog.set_level(logging.WARNING, logger="neuropress.kernels")
    exc = RuntimeError("ninja is required to load C++ extensions")
    for _ in range(3):
        warn_kernel_fallback("test_kernel_a", "slow path", exc)
    records = [r for r in caplog.records if "test_kernel_a" in r.getMessage()]
    assert len(records) == 1
    msg = records[0].getMessage()
    assert "ninja is required" in msg
    assert "slow path" in msg
    assert "pip install ninja" in msg and "nvcc" in msg


def test_each_kernel_warns_separately(caplog):
    caplog.set_level(logging.WARNING, logger="neuropress.kernels")
    warn_kernel_fallback("test_kernel_b", "x", RuntimeError("b"))
    warn_kernel_fallback("test_kernel_c", "x", RuntimeError("c"))
    names = {
        n
        for n in ("test_kernel_b", "test_kernel_c")
        for r in caplog.records
        if n in r.getMessage()
    }
    assert names == {"test_kernel_b", "test_kernel_c"}
