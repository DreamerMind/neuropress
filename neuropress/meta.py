"""Run metadata collection — git commit, hardware, timestamps.

Shared by the benchmark and evaluation tools so every JSON output
can be traced back to the exact code version that produced it.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from datetime import UTC, datetime

import torch
from pydantic import BaseModel


def git_commit_hash() -> str | None:
    """Return the full HEAD commit hash, or ``None`` outside a git repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def git_dirty() -> bool | None:
    """Return ``True`` if the working tree has uncommitted changes.

    Returns ``None`` when not inside a git repository.
    """
    try:
        rc = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return rc != 0
    except Exception:
        return None


def collect_run_meta() -> dict[str, object]:
    """Collect version and environment metadata for a run.

    Returns:
        Dict with ``git_commit``, ``git_dirty``, ``timestamp``,
        ``hostname``, and ``platform`` keys.
    """
    meta: dict[str, object] = {
        "git_commit": git_commit_hash(),
        "git_dirty": git_dirty(),
        "timestamp": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
    }

    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_count"] = torch.cuda.device_count()
        _free, total = torch.cuda.mem_get_info(0)
        meta["gpu_vram_gb"] = round(total / 2**30, 1)

    return meta


class RunMeta(BaseModel):
    """Run environment snapshot — git state, hardware, and process info."""

    git_commit: str | None = None
    git_dirty: bool | None = None
    timestamp: str
    hostname: str
    platform: str
    python: str
    pid: int
    gpu: str | None = None
    gpu_count: int | None = None
    gpu_vram_gb: float | None = None

    @classmethod
    def collect(cls) -> RunMeta:
        """Snapshot the current environment."""
        return cls(**collect_run_meta())
