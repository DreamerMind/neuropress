"""Direct-SSH backend for neuropress cloud jobs.

Connects to any SSH-accessible machine you already own (home PC, university
cluster, …) without renting or destroying anything.  No vast.ai account or
API key is required.

Usage::

    from neuropress.cloud.direct_runner import DirectRunner

    runner = DirectRunner(host="192.168.1.42", port=22, user="myuser")
    runner.run(job)
"""

from __future__ import annotations

import re
from pathlib import Path

from neuropress.cloud.runner import SshRunner, VastJob

_DIRECT_WORKDIR_PREFIX = "~/neuropress-cloud"
"""Default remote workdir prefix for direct SSH.

Vast.ai containers use ``/workspace`` (root-owned); personal machines
typically lack that path, so we default to a subdirectory under ``$HOME``.
"""


_APT_RE = re.compile(r"\bapt-(?:get|cache)\b")


def _sudo_apt(cmd: str) -> str:
    """Prepend ``sudo`` to bare ``apt-get`` / ``apt-cache`` in *cmd*.

    Already-prefixed ``sudo apt-get`` is left untouched.

    Args:
        cmd: Shell command string.

    Returns:
        Modified command with ``sudo`` inserted where needed.
    """
    return _APT_RE.sub(lambda m: f"sudo {m.group()}", cmd)


class DirectRunner(SshRunner):
    """Runs jobs on an SSH-accessible machine you already own.

    Nothing is rented, billed, or destroyed.  The lifecycle is identical to
    :class:`~neuropress.cloud.vast_runner.VastRunner` except that connection
    info (host / port / user) is supplied up-front rather than obtained from
    the vast.ai API.

    Job workdirs that start with ``/workspace`` are automatically remapped to
    ``~/neuropress-cloud`` so that no root-owned paths are required.

    Args:
        host: Hostname or IP address of the target machine.
        port: SSH port (default: ``22``).
        user: SSH login user (default: ``"root"``).
        ssh_key: Path to SSH private key (default: ``~/.ssh/id_rsa``).
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        ssh_key: Path | None = None,
    ) -> None:
        super().__init__(ssh_key=ssh_key)
        self._direct_host = host
        self._direct_port = port
        self._direct_user = user

    # ---------------------------------------------------------------- abstract

    def _connect(self, job: VastJob) -> None:
        """Set SSH connection fields and adapt job for personal machines.

        Performs two adaptations:

        1. Remaps ``/workspace``-based workdirs to ``~/neuropress-cloud``.
        2. Prepends ``sudo`` to bare ``apt-get`` / ``apt-cache`` calls in
           setup commands (vast.ai containers run as root; personal machines
           typically do not).

        Args:
            job: Job whose ``remote_workdir`` and ``setup`` may be adapted.
        """
        self._host = self._direct_host
        self._port = self._direct_port
        self._user = self._direct_user

        # Remap /workspace-based paths to ~/neuropress-cloud.
        if job.remote_workdir.startswith("/workspace"):
            suffix = job.remote_workdir[len("/workspace") :]
            job.remote_workdir = f"{_DIRECT_WORKDIR_PREFIX}{suffix}"
            self._log(f"Remapped workdir → {job.remote_workdir}")

        # Prefix apt commands with sudo for non-root users.
        if self._direct_user != "root":
            job.setup = [_sudo_apt(cmd) for cmd in job.setup]

        self._log(f"Direct SSH: {self._user}@{self._host}:{self._port}")

    def _cleanup(self) -> None:
        """No-op — the machine is never destroyed by this runner."""

    # ----------------------------------------------------------------- public

    def dry_run(self, job: VastJob) -> None:
        """Print connection info without executing anything.

        Args:
            job: Job whose name is shown in the preview.
        """
        self._log(
            f"Dry run: '{job.name}' would run on "
            f"{self._direct_user}@{self._direct_host}:{self._direct_port} "
            f"(SSH key: {self._ssh_key})"
        )
