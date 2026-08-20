"""Backend enumerations for quantized inference modules.

Each quantization path (scalar linear, vector-quantized linear, embedding)
supports a different subset of execution backends.  The enums defined here
are ``str`` enums so they work natively with pydantic validation, JSON
serialisation, and string comparisons in resolve helpers.

The superset :class:`Backend` enum is kept for backward compatibility — it
was the single shared enum used before the per-path split.
"""

from __future__ import annotations

import enum
import glob
import logging
import os
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

_cuda_jit_ok: bool | None = None


def is_cuda_jit_available() -> bool:
    """Return ``True`` when CUDA JIT compilation is possible.

    Checks that both ``ninja`` (build system) and ``nvcc`` (CUDA compiler)
    are available.  The result is cached after the first call.

    Set ``NEUROPRESS_FORCE_CUDA_JIT=1`` to skip detection and return ``True``
    unconditionally (useful when the auto-detection fails on non-standard
    CUDA installs but the toolchain is actually present).
    """
    global _cuda_jit_ok
    if _cuda_jit_ok is not None:
        return _cuda_jit_ok

    if os.environ.get("NEUROPRESS_FORCE_CUDA_JIT", "").strip() in ("1", "true", "yes"):
        _cuda_jit_ok = True
        print(
            "[neuropress] NEUROPRESS_FORCE_CUDA_JIT=1 — skipping CUDA JIT detection",
            flush=True,
        )
        return True

    # 1. Check ninja
    try:
        from torch.utils.cpp_extension import verify_ninja_availability

        verify_ninja_availability()
    except RuntimeError:
        _cuda_jit_ok = False
        msg = "ninja not found — CUDA JIT backends disabled; falling back to torch.compile"
        _log.warning(msg)
        print(f"[neuropress] {msg}", flush=True)
        return False

    # 2. Check nvcc — four locations in priority order:
    #    a) PATH (standard)
    #    b) CUDA_HOME / CUDA_PATH env vars
    #    c) /usr/local/cuda (distro symlink, always present on CUDA images)
    #    d) /usr/local/cuda-X.Y/bin/nvcc glob (apt-installed packages on
    #       vast.ai / Lambda / bare-metal hosts where the symlink is absent
    #       and PATH isn't patched in the subprocess environment)
    nvcc = shutil.which("nvcc")
    if nvcc:
        print(f"[neuropress] nvcc found in PATH: {nvcc} — CUDA JIT enabled", flush=True)
        _cuda_jit_ok = True
        return True
    cuda_home = (
        os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    )
    candidate_home = Path(cuda_home) / "bin" / "nvcc"
    if candidate_home.is_file():
        print(
            f"[neuropress] nvcc found at {candidate_home} — CUDA JIT enabled",
            flush=True,
        )
        _cuda_jit_ok = True
        return True
    for candidate in sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True):
        if Path(candidate).is_file():
            print(
                f"[neuropress] nvcc found at {candidate} — CUDA JIT enabled", flush=True
            )
            _cuda_jit_ok = True
            return True

    _cuda_jit_ok = False
    msg = (
        f"nvcc not found (PATH={shutil.which('nvcc')!r}, CUDA_HOME={cuda_home!r}, "
        f"glob=/usr/local/cuda-*/bin/nvcc→{glob.glob('/usr/local/cuda-*/bin/nvcc')}) "
        "— CUDA JIT backends disabled; falling back to torch.compile"
    )
    _log.warning(msg)
    print(f"[neuropress] {msg}", flush=True)
    return False


class LinearBackend(enum.StrEnum):
    """Valid backends for scalar-quantized ``nn.Linear`` (:class:`.NPLinear`)."""

    AUTO = "auto"
    DEFAULT = "default"
    COMPILE = "compile"
    CUDA = "cuda"
    GEMLITE = "gemlite"


class VQBackend(enum.StrEnum):
    """Valid backends for vector-quantized ``nn.Linear`` (:class:`.NPVQLinear`)."""

    AUTO = "auto"
    DEFAULT = "default"
    COMPILE = "compile"
    CUDA = "cuda"
    CUDA_WMMA = "cuda_wmma"
    CUDA_LUT = "cuda_lut"
    CUDA_TIERED = "cuda_tiered"
    CUDA_NTIER = "cuda_ntier"
    CUDA_ZERO_ESCAPE = "cuda_zero_escape"


class EmbeddingBackend(enum.StrEnum):
    """Valid backends for quantized ``nn.Embedding`` (:class:`.NPEmbedding`)."""

    AUTO = "auto"
    DEFAULT = "default"
    COMPILE = "compile"
    CUDA = "cuda"


class NormBackend(enum.StrEnum):
    """Valid backends for RMSNorm modules (:class:`.NPRMSNorm`)."""

    AUTO = "auto"
    DEFAULT = "default"
    CUDA = "cuda"


class Backend(enum.StrEnum):
    """Superset backend enum (backward compatibility).

    Prefer the per-path enums (:class:`LinearBackend`, :class:`VQBackend`,
    :class:`EmbeddingBackend`) for new code — they provide tighter validation.
    """

    DEFAULT = "default"
    COMPILE = "compile"
    CUDA = "cuda"
    CUDA_WMMA = "cuda_wmma"
    CUDA_LUT = "cuda_lut"
    CUDA_TIERED = "cuda_tiered"
    CUDA_NTIER = "cuda_ntier"
    CUDA_ZERO_ESCAPE = "cuda_zero_escape"
    GEMLITE = "gemlite"
    AUTO = "auto"
