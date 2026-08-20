"""Shared SSH infrastructure for neuropress cloud backends.

Defines :class:`SshRunner`, the abstract base class used by all backends.
Concrete subclasses supply two hooks:

* :meth:`~SshRunner._connect` — establish SSH connectivity for a job
  (e.g. rent a GPU instance and wait for SSH, or look up a pre-owned host).
* :meth:`~SshRunner._cleanup` — release resources when the job finishes
  (e.g. destroy a cloud instance, or no-op for machines you own).

The :class:`GpuSpec` and :class:`VastJob` dataclasses describing GPU offers
and job specifications are also defined here so they can be imported without
pulling in any backend-specific dependencies (e.g. ``vastai-sdk``).
"""

from __future__ import annotations

import atexit
import io
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.ansi import AnsiDecoder
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.text import Text

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

_POLL_INTERVAL_S: int = 15  # seconds between instance-status polls
_SSH_TIMEOUT_S: int = 2400  # max wait for SSH readiness (image pull can take 30+ min)
_DESTROY_RETRIES: int = 5  # retries before giving up on destroy
_RETRY_BACKOFF_S: int = 5  # base back-off between destroy retries
_LOG_POLL_S: int = 5  # seconds between remote log fetches
_SYNC_INTERVAL_POLLS: int = 6  # sync intermediate artifacts every N log polls (~30s)
_RECONNECT_WAIT_S: int = 15  # pause before SSH reconnect after drop
_MAX_RECONNECTS: int = 10  # consecutive SSH drops before aborting

# rsync exit codes that are acceptable (not errors):
# 0  = success
# 23 = partial transfer (special files, broken symlinks)
# 24 = partial transfer (files vanished during transfer)
_RSYNC_OK_CODES: tuple[int, ...] = (0, 23, 24)

# Remote paths used by the tmux job executor
_TMUX_SESSION: str = "neuropress"
_JOB_SCRIPT: str = "/tmp/neuropress_job.sh"
_JOB_LOG: str = "/tmp/neuropress_job.log"
_JOB_EXIT: str = "/tmp/neuropress_exit"

# Matches rsync --info=progress2 progress lines, e.g.:
#   "      1,234,567  45%  12.34MB/s    0:00:05 (xfr#3, to-chk=45/100)"
# Groups: (1) pct, (2) speed, (3) remaining, (4) total_files
_RSYNC_PROGRESS_RE: re.Pattern[str] = re.compile(
    r"^\s*[\d,]+\s+(\d+)%\s+(\S+)\s+\S+\s+\((?:xfr|ir)-chk=(\d+)/(\d+)\)"
)

# Strips ANSI/VT100 escape sequences when writing to the plain-text run log.
_ANSI_ESCAPE_RE: re.Pattern[str] = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class _CRCollapsingWriter:
    r"""Wrap a text file to emulate terminal ``\r`` (carriage return) behaviour.

    On each ``\r``, seeks back to the start of the current line and
    truncates, so the next write overwrites it in-place — exactly like a
    terminal.  The log file only ever contains the last update of each
    progress bar line.
    """

    def __init__(self, f: Any) -> None:
        self._f = f
        self._line_start: int = f.tell()

    def write(self, data: str) -> None:
        r"""Write *data*, seeking back to line start on each ``\r``."""
        parts = data.split("\r")
        for i, part in enumerate(parts):
            if i > 0:
                self._f.seek(self._line_start)
                self._f.truncate()
            if not part:
                continue
            self._f.write(part)
            nl = part.rfind("\n")
            if nl >= 0:
                after_nl = part[nl + 1 :]
                self._line_start = self._f.tell() - len(after_nl.encode("utf-8"))

    def flush(self) -> None:
        """Flush the underlying file."""
        self._f.flush()

    def close(self) -> None:
        """Close the underlying file."""
        self._f.close()


# ---------------------------------------------------------------------------
# Job specification
# ---------------------------------------------------------------------------


@dataclass
class GpuSpec:
    """GPU offer selection and Docker image configuration.

    Attributes:
        query: vast.ai offer search query string.
        image: Docker image to launch on the instance.
        disk_gb: Root disk size in GB.
        max_dph: Hard cap on $/hr; raises if nothing cheaper is found.
        datacenter: Restrict search to datacenter/colocation hosts.
        verified: Restrict search to identity-verified hosts only.
        min_inet_down_mbps: Minimum measured download bandwidth in Mbps.
        min_reliability: Minimum vast.ai reliability score (0-1).
    """

    query: str = "num_gpus=1 rented=False rentable=True cuda_vers>=12.1 dph<0.5"
    image: str = "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
    disk_gb: int = 50
    max_dph: float = 0.80
    gpu_name: str | None = None
    """Override the ``gpu_name=X`` filter in *query*.

    When set, replaces any existing ``gpu_name=...`` token in *query*
    or appends one if absent.  Use to switch GPU model without rewriting
    the whole query string (e.g. ``"RTX_3090"``, ``"A100_SXM4_80GB"``).
    """
    datacenter: bool = True
    """Restrict search to datacenter/colocation hosts (better uptime, more private)."""
    verified: bool = False
    """Restrict search to identity-verified hosts only."""
    min_inet_down_mbps: int = 500
    """Minimum measured download bandwidth in Mbps.

    The PyTorch Docker image is ~6.5 GB; at 500 Mbps it pulls in ~1.5 min.
    At 100 Mbps it takes ~8 min — fully billed idle time.  Set to 0 to disable.
    """
    min_reliability: float = 0.9
    """Minimum vast.ai reliability score (0-1).  Filters out flaky hosts.

    Set to 0.0 to disable.
    """
    location: str | None = "eu"
    """Location filter applied to the ``geolocation`` offer field.

    Defaults to ``"eu"`` (European datacenters).  Set to ``None`` or pass
    ``--location any`` on the CLI to disable.

    ``"eu"`` expands to a built-in list of European country codes.  Any other
    value is matched as a case-insensitive substring of each offer's
    ``geolocation`` string (e.g. ``"de"``, ``"germany"``, ``"nl"``).
    Set to ``None`` (default) to allow any location.
    """


@dataclass
class VastJob:
    """Complete description of a GPU job.

    Attributes:
        name: Human-readable name (shown in logs and output path).
        gpu: GPU offer selection criteria and Docker image.
        uploads: Local paths rsync'd to *remote_workdir* before the job.
        remote_workdir: Absolute working directory on the remote instance.
            All *setup* and *run* commands execute here.
        setup: Shell commands run once after upload (installs, env setup).
            Each must exit 0; the runner aborts on the first failure.
        run: Shell commands that constitute the actual job.
        artifacts: Remote paths rsync'd back to *output_dir* after the job.
        output_dir: Local directory that receives downloaded artifacts.
        sync_dir: Remote directory to periodically rsync during the run
            phase for crash resilience (per-task eval checkpoints).
        resume_files: Mapping of local-relative → remote-absolute paths.
            On retry, existing local files are uploaded before the run so
            the remote job can skip expensive re-computation.
        env_vars: Environment variables exported at the start of the run script.
        required_env_vars: Env var names that must be present before the job starts.
        rsync_excludes: Patterns passed to ``rsync --exclude``.
    """

    name: str
    gpu: GpuSpec
    uploads: list[Path] = field(default_factory=list)
    remote_workdir: str = "/workspace"
    setup: list[str] = field(default_factory=list)
    run: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    output_dir: Path = Path("./vast_output")
    env_vars: dict[str, str] = field(default_factory=dict)
    """Environment variables exported at the start of the run script.

    Values are shell-quoted before insertion.  Not echoed to the log.
    Typical use: ``{"HF_TOKEN": "hf_…"}`` for gated HuggingFace models.
    """
    required_env_vars: list[str] = field(default_factory=list)
    """Env var names that must be present in *env_vars* before the job starts.

    The runner checks these before renting any instance and exits with a clear
    error if any are missing.  Typical use: ``["HF_TOKEN"]``.
    """
    rsync_excludes: list[str] = field(
        default_factory=lambda: [
            # .git is excluded explicitly: it is never listed in .gitignore
            # (git ignores its own internals regardless) but must not be
            # uploaded to remote instances.  All other patterns live in
            # .gitignore and are picked up automatically via the
            # --filter=':- .gitignore' rsync flag in _upload().
            ".git",
        ]
    )
    sync_dir: str | None = None
    """Remote directory to periodically rsync during the run phase.

    When set, the runner rsyncs this remote directory to *output_dir* every
    ``_SYNC_INTERVAL_POLLS`` log-poll cycles during :meth:`_stream_until_done`.
    This enables progressive download of intermediate results (per-task eval
    checkpoints, saved models) for crash resilience.
    """
    sync_files: dict[str, str] = field(default_factory=dict)
    """Mapping of ``{remote_path: local_relative_path}`` synced periodically.

    Like *sync_dir* but for individual files.  Each remote path is rsync'd
    to ``output_dir / local_relative_path`` during the periodic intermediate
    sync.  Use this for expensive artifacts (e.g. quantized model checkpoint)
    that should be downloaded as soon as they appear on the remote, without
    waiting for the job to complete.
    """
    resume_files: dict[str, str] = field(default_factory=dict)
    """Mapping of ``{local_path: remote_path}`` to upload before the run.

    On retry, the runner checks each local path; if the file exists it is
    uploaded to the corresponding remote path so the job can skip expensive
    re-computation (e.g. re-quantization).  Paths are relative to
    *output_dir* on the local side and absolute on the remote side.
    """
    resume_dirs: dict[str, str] = field(default_factory=dict)
    """Mapping of ``{local_abs_path: remote_abs_path}`` for directory uploads before run.

    Like *resume_files* but for directories.  Keys are absolute local paths;
    values are absolute remote paths.  Contents are rsync'd with a trailing
    slash so the directory tree is mirrored (not wrapped in a subdirectory).
    Useful for cache directories that should be seeded from a previous run.
    """
    sync_dirs: dict[str, str] = field(default_factory=dict)
    """Mapping of ``{remote_abs_path: local_abs_path}`` for directory sync during run.

    Like *sync_files* but for directories.  Synced periodically during the
    run and once more after completion.  Use for cache directories that grow
    during quantization and should be preserved for future runs.
    """
    meta: dict[str, Any] = field(default_factory=dict)
    """Optional human-readable metadata shown in the TUI configuration modal.

    Populated by job factories with high-level info (model, compression,
    tasks, …) that is too verbose to embed in the job *name* but useful to
    see at a glance.  Keys and values are displayed as-is; the dict is not
    used by the runner logic itself.
    """


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------


class _StatusClearingFile:
    r"""Wraps a text stream; prepends ``\r\033[2K`` before each write.

    The clear escape is only emitted when the status bar is active and
    not paused.  This prevents the status bar from leaving a stale partial
    line on the terminal when normal console output follows.

    :class:`_StatusBar` writes directly to :attr:`underlying` to bypass
    this wrapper (so the status bar never clears itself before writing).
    """

    def __init__(self, underlying: Any) -> None:
        self._underlying = underlying
        self.status_bar: _StatusBar | None = None
        self.log_file: Any = None  # text-mode IO; receives ANSI-stripped output

    @property
    def underlying(self) -> Any:
        """Direct access to the underlying stream (bypasses clearing)."""
        return self._underlying

    def write(self, data: str) -> int:
        """Clear the status line before writing, then write *data*."""
        if self.status_bar and not self.status_bar._paused:
            self._underlying.write("\r\033[2K")
            # The clear escape is terminal-only; do not forward to the log.
        if self.log_file is not None:
            self.log_file.write(_ANSI_ESCAPE_RE.sub("", data))
        return self._underlying.write(data)

    def flush(self) -> None:
        """Flush the underlying stream and log file."""
        self._underlying.flush()
        if self.log_file is not None:
            self.log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._underlying, name)


class _StatusBar:
    r"""Sticky one-line status rendered via carriage-return overwriting.

    Runs in a daemon thread throughout the job lifecycle.  Because it uses
    ``\r\033[2K`` (overwrite, no newline), the line is never added to the
    terminal scrollback buffer or to any log file that captures stdout.

    Writes directly to the *file* underlying stream (bypassing any
    :class:`_StatusClearingFile` wrapper) so it does not trigger its own
    clear-on-write logic.

    Call :meth:`pause` / :meth:`resume` around rich ``Progress`` contexts to
    prevent cursor-positioning conflicts.
    """

    def __init__(self, label: str, get_phase: Callable[[], str], file: Any) -> None:
        self._label = label
        self._get_phase = get_phase
        self._file = file
        self._start = time.monotonic()
        self._active = True
        self._paused = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start the background render thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and erase the status line."""
        self._active = False
        self._thread.join(timeout=2)
        self._file.write("\r\033[2K")
        self._file.flush()

    def pause(self) -> None:
        """Clear and suppress the status line (call before rich Progress)."""
        self._paused = True
        self._file.write("\r\033[2K")
        self._file.flush()

    def resume(self) -> None:
        """Re-enable status rendering after :meth:`pause`."""
        self._paused = False

    def _loop(self) -> None:
        while self._active:
            if not self._paused:
                elapsed = time.monotonic() - self._start
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                phase = self._get_phase()
                text = f"  \u25cf {self._label}  {phase}  {mins}:{secs:02d}"
                # \r: go to line start  \033[2K: clear line  \033[2m: dim
                self._file.write(f"\r\033[2K\033[2m{text}\033[0m")
                self._file.flush()
            time.sleep(1)


# ---------------------------------------------------------------------------
# TUI file bridge
# ---------------------------------------------------------------------------


class _TuiFile:
    r"""File-like object that bridges a Rich :class:`~rich.console.Console` to the TUI.

    Rich writes ANSI-formatted text to this object.  Each complete line is
    decoded from ANSI escape sequences back to a :class:`rich.text.Text`
    object (preserving colour/style) and forwarded to *callback*, which
    routes it to the :class:`~neuropress.cloud.tui.JobApp` log widget.

    Used in TUI mode to replace the normal ``_StatusClearingFile`` so that
    **all** :meth:`~rich.console.Console.print` / :meth:`~rich.console.Console.rule`
    output appears inside the TUI rather than being written directly to
    ``sys.stdout``.
    """

    def __init__(self, callback: Callable[[Text], None]) -> None:
        self._callback = callback
        self._buf = ""
        self._decoder = AnsiDecoder()

    def write(self, data: str) -> int:
        """Buffer *data* and flush complete lines to the TUI callback."""
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            text = self._decoder.decode_line(line)
            if str(text).strip():
                self._callback(text)
        return len(data)

    def flush(self) -> None:
        """Flush any remaining buffered content."""
        if self._buf.strip():
            text = self._decoder.decode_line(self._buf)
            self._callback(text)
            self._buf = ""

    @property
    def encoding(self) -> str:
        """Report UTF-8 encoding to callers that probe the stream."""
        return "utf-8"

    def isatty(self) -> bool:
        """Report as non-TTY to suppress interactive Rich features."""
        return False

    def fileno(self) -> int:
        """Raise so callers know this is not a real file descriptor."""
        raise io.UnsupportedOperation("not a real file")


# ---------------------------------------------------------------------------
# Abstract SSH runner
# ---------------------------------------------------------------------------


class SshRunner(ABC):
    """Abstract base: shared SSH upload / setup / run / download lifecycle.

    Subclasses implement :meth:`_connect` (establish a connection for a job)
    and :meth:`_cleanup` (release resources — destroy cloud instance or no-op).

    All SSH, rsync, tmux, and display helpers live here and are shared across
    backends.

    Args:
        ssh_key: Path to SSH private key (default: ``~/.ssh/id_rsa``).
        keep: If ``True``, skip cleanup after the job and preserve state for
            reuse.  Only meaningful for backends that manage cloud instances.
    """

    def __init__(
        self,
        ssh_key: Path | None = None,
        *,
        keep: bool = False,
        resume: bool = False,
    ) -> None:
        self._ssh_key = ssh_key or Path("~/.ssh/id_rsa").expanduser()
        self._host: str | None = None
        self._port: int | None = None
        self._user: str = "root"
        self._clearing_file = _StatusClearingFile(sys.stdout)
        self._console = Console(file=self._clearing_file)
        self._current_phase: str = ""
        self._keep = keep
        self._resume = resume  # skip upload+setup; re-attach to live tmux session
        self._status_bar: _StatusBar | None = None
        self._dph: float | None = None  # $/hr set by backend (None for own machines)
        self._log_file: Any = None  # open text file for the local run log
        self._in_tui: bool = False
        self._phase_cb: Callable[[str], None] | None = None
        self._job_failed: bool = False  # set True when the remote job itself fails

        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
        signal.signal(signal.SIGINT, lambda *_: sys.exit(1))

    # ---------------------------------------------------------------- abstract

    @abstractmethod
    def _connect(self, job: VastJob) -> None:
        """Establish SSH connectivity for *job*; populate host/port/user.

        Called at the start of :meth:`run` before any uploads or commands.
        On return, ``self._host``, ``self._port``, and ``self._user`` must
        be set and SSH must be ready.

        Args:
            job: The job about to run (may be used to pick a machine/image).
        """

    @abstractmethod
    def _cleanup(self) -> None:
        """Release resources after the job finishes (or on crash).

        Registered with ``atexit`` so it is always called, even on
        ``KeyboardInterrupt`` or unhandled exceptions.  Implementations must
        be idempotent (safe to call multiple times).
        """

    # ----------------------------------------------------------------- public

    def dry_run(self, job: VastJob) -> None:
        """Preview what would happen without executing (no-op by default).

        Override in subclasses that can show meaningful pre-flight info
        (e.g. :class:`VastRunner` lists matching GPU offers).

        Args:
            job: Job whose spec is used for the preview.
        """
        self._log(f"Dry run for '{job.name}' — nothing to preview for this backend.")

    def run(self, job: VastJob) -> bool:
        """Execute the full lifecycle: connect → upload → setup → run → pull.

        Launches the Textual TUI when ``stdout`` is a TTY; falls back to
        plain terminal output otherwise (e.g. sweep subprocesses, CI).

        Args:
            job: Job to execute.

        Returns:
            ``True`` if all run commands succeeded, ``False`` otherwise.
        """
        if sys.stdout.isatty():
            return self._run_with_tui(job)
        return self._run_core(job)

    def _run_with_tui(self, job: VastJob) -> bool:
        """Wrap *_run_core* in a Textual full-screen TUI.

        Replaces the Rich console with a :class:`_TuiFile` bridge so all
        ``self._console.print()`` calls route to the TUI log widget.
        Restores the original console after the TUI exits.

        Args:
            job: Job to execute.

        Returns:
            ``True`` if the job completed successfully.
        """
        try:
            from .tui import JobApp
        except ImportError as exc:
            self._log(f"textual not installed ({exc}); falling back to plain output.")
            return self._run_core(job)

        # Lazily-resolved app reference: needed by the _write callback but
        # the app object doesn't exist yet when we create the closure.
        _app: JobApp | None = None

        def _write(text: Text) -> None:
            if _app is not None:
                _app.write(text)

        tui_file = _TuiFile(_write)
        old_console = self._console
        # force_terminal=True so rich renders colour + box-drawing chars;
        # width=120 is wide enough for typical log lines.
        self._console = Console(file=tui_file, force_terminal=True, width=120)
        self._in_tui = True

        def _phase_cb(phase: str) -> None:
            if _app is not None:
                _app.set_phase(phase)

        self._phase_cb = _phase_cb

        config: dict[str, Any] = {"name": job.name, **job.meta}
        if job.output_dir != Path("./vast_output"):
            config["output"] = str(job.output_dir)

        _app = JobApp(job.name, config, lambda: self._run_core(job))
        try:
            result = _app.run()
            return bool(result)
        finally:
            self._console = old_console
            self._in_tui = False
            self._phase_cb = None

    def _run_core(self, job: VastJob) -> bool:
        """Core run lifecycle used by both plain and TUI modes.

        Args:
            job: Job to execute.

        Returns:
            ``True`` if all run commands succeeded, ``False`` otherwise.
        """
        self._log(f"Starting job: {job.name}")

        missing = [k for k in job.required_env_vars if k not in job.env_vars]
        if missing:
            self._console.print(
                f"\n[bold red][ERROR][/bold red] Missing required environment "
                f"variable(s): [bold]{', '.join(missing)}[/bold]\n"
                + "\n".join(
                    f"  Pass it with [bold]--hf-token[/bold] or set "
                    f"[bold]${k}[/bold] in your shell."
                    if k == "HF_TOKEN"
                    else f"  Pass [bold]${k}[/bold] in your shell environment."
                    for k in missing
                )
            )
            return False

        job.output_dir.mkdir(parents=True, exist_ok=True)

        log_path = job.output_dir / "run.log"
        self._log_file = _CRCollapsingWriter(
            log_path.open("w", encoding="utf-8", errors="replace")
        )
        if not self._in_tui:
            self._clearing_file.log_file = self._log_file

        run_start_s = time.monotonic()

        if not self._in_tui and sys.stdout.isatty():
            self._status_bar = _StatusBar(
                job.name, lambda: self._current_phase, self._clearing_file.underlying
            )
            self._clearing_file.status_bar = self._status_bar
            self._status_bar.start()

        try:
            self._connect(job)

            # Write instance ID immediately so it survives local disconnects.
            self._save_instance_id(job.output_dir)

            if self._resume:
                self._phase("Resume")
                self._console.print(
                    "  [yellow]⏩[/yellow]  [bold]--resume[/bold]: "
                    "skipping upload + setup, re-attaching to live tmux session."
                )
                if job.run:
                    self._phase("Run (resumed)")
                    self._console.print(
                        f"  Job running in tmux [bold]{escape(_TMUX_SESSION)!r}[/bold]"
                        " — streaming output [dim](auto-reconnects on SSH drop)[/dim]:"
                    )
                    self._stream_until_done(job)
            else:
                self._phase("Upload")
                self._upload(job)

                self._phase("Setup")
                for cmd in job.setup:
                    self._ssh(cmd, workdir=job.remote_workdir)

                self._upload_partials(job)

                self._phase("Run")
                if job.run:
                    self._run_in_tmux(
                        job.run,
                        workdir=job.remote_workdir,
                        env_vars=job.env_vars,
                        job=job,
                    )

            # Final intermediate sync before downloading full artifacts.
            self._sync_intermediate(job)

            self._phase("Download artifacts")
            self._download(job)

            self._log(f"Job '{job.name}' completed successfully.")
            self._job_failed = False
            return True

        except subprocess.CalledProcessError as exc:
            phase = (
                f" during [bold]{escape(self._current_phase)}[/bold]"
                if self._current_phase
                else ""
            )
            if exc.returncode == 255:
                self._console.print(
                    f"\n[bold red]\\[ERROR][/bold red] SSH connection failed"
                    f"{phase} (exit 255)\n"
                    "  The instance may have terminated, the Docker image failed\n"
                    "  to start, or the network dropped.\n"
                    "  → Check [link=https://cloud.vast.ai/instances/]"
                    "cloud.vast.ai/instances[/link] for instance health."
                )
                # SSH drop ≠ remote job failure — preserve the instance.
                self._keep = True
            else:
                self._job_failed = True
                self._console.print(
                    f"\n[bold red]\\[ERROR][/bold red] Command failed"
                    f"{phase} (exit {exc.returncode})"
                    " — see output above for details."
                )
            self._try_save_log(job.output_dir)
            return False

        except TimeoutError as exc:
            phase = (
                f" during [bold]{escape(self._current_phase)}[/bold]"
                if self._current_phase
                else ""
            )
            self._console.print(
                f"\n[bold red]\\[ERROR][/bold red] Timeout{phase}: {escape(str(exc))}"
            )
            # Timeout (e.g. SSH not ready) — destroy the instance, don't keep.
            self._try_save_log(job.output_dir)
            return False

        except Exception as exc:
            phase = (
                f" during [bold]{escape(self._current_phase)}[/bold]"
                if self._current_phase
                else ""
            )
            self._console.print(
                f"\n[bold red]\\[ERROR][/bold red] Unexpected error"
                f"{phase}: {escape(str(exc))}"
            )
            # Unknown error — preserve the instance so the user can inspect it.
            self._keep = True
            self._try_save_log(job.output_dir)
            return False

        finally:
            elapsed_s = time.monotonic() - run_start_s
            if self._status_bar:
                self._status_bar.stop()
                self._status_bar = None
            if not self._in_tui:
                self._clearing_file.status_bar = None
            self._print_cost_summary(elapsed_s)
            self._cleanup()
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
                if not self._in_tui:
                    self._clearing_file.log_file = None
                self._console.print(
                    f"  [dim]Run log → "
                    f"[bold]{escape(str(log_path.resolve()))}[/bold][/dim]"
                )

    # ----------------------------------------------------------------- private

    # -- Instance state file -------------------------------------------------

    def _save_instance_id(self, output_dir: Path) -> None:
        """Write the instance ID to *output_dir/instance_id.txt*.

        Called immediately after :meth:`_connect` so that the ID is persisted
        even if the local process dies mid-job.  The user can recover it with::

            cat <output_dir>/instance_id.txt

        And reconnect with ``--instance-id <N> --resume``.  A no-op when the
        backend has no notion of an instance ID (e.g. own-machine backend
        where ``_instance_id`` is not defined).

        Args:
            output_dir: Local output directory for this job.
        """
        iid = getattr(self, "_instance_id", None)
        if iid is None:
            return
        id_file = output_dir / "instance_id.txt"
        try:
            id_file.write_text(str(iid))
            self._console.print(
                f"  [dim]Instance ID → "
                f"[bold]{escape(str(id_file.resolve()))}[/bold]  "
                f"(reconnect: [bold]--instance-id {iid} --resume[/bold])[/dim]"
            )
        except OSError:
            pass  # non-fatal; user still sees the ID in the TUI header

    # -- Upload / download ---------------------------------------------------

    def _upload(self, job: VastJob) -> None:
        if not job.uploads:
            self._console.print("  No uploads specified.")
            return
        self._ssh(f"mkdir -p {job.remote_workdir}")
        exclude_args = [arg for pat in job.rsync_excludes for arg in ("--exclude", pat)]
        # --delete mirrors the source exactly: stale/renamed files on the
        # remote are removed.  Excluded paths (.venv, __pycache__, …) are
        # never deleted regardless of --delete.
        for local in job.uploads:
            src = str(local) + ("/" if local.is_dir() else "")
            self._rsync(
                description=(f"{escape(str(local))} → {escape(job.remote_workdir)}/"),
                src=src,
                dst=f"{self._user}@{self._host}:{job.remote_workdir}/",
                extra_args=[
                    "--delete",
                    # Respect every .gitignore in the tree as per-directory
                    # exclude rules (same behaviour as `git ls-files`).
                    "--filter=:- .gitignore",
                    # Skip socket files, FIFOs, and device files — these
                    # exist in Python dev environments (.sock, __pycache__
                    # lock files, etc.) and cause rsync to exit 23 because
                    # it cannot transfer them.  --archive implies --specials
                    # and --devices; these flags undo just those two parts.
                    "--no-specials",
                    "--no-devices",
                    *exclude_args,
                ],
            )

    def _download(self, job: VastJob) -> None:
        if not job.artifacts:
            self._console.print("  No artifacts specified.")
            return
        for remote in job.artifacts:
            self._rsync(
                description=f"{escape(remote)} → {escape(str(job.output_dir))}/",
                src=f"{self._user}@{self._host}:{remote}",
                dst=str(job.output_dir) + "/",
            )
        self._console.print(
            f"  [green]✓[/green] Artifacts saved to "
            f"[bold]{escape(str(job.output_dir.resolve()))}[/bold]"
        )

    def _upload_partials(self, job: VastJob) -> None:
        """Upload cached partial results and resume files to the remote.

        Handles two kinds of cached data:

        * **sync_dir partials** — per-task eval JSON checkpoints previously
          downloaded via :meth:`_sync_intermediate`.
        * **resume_files** — expensive artifacts (e.g. quantized model
          checkpoint) that let the remote skip re-computation.

        Args:
            job: Job whose *sync_dir*, *resume_files*, and *output_dir*
                are used.
        """
        # 1. Upload eval partial JSONs.
        if job.sync_dir:
            local_partial = job.output_dir / Path(job.sync_dir).name
            if local_partial.is_dir() and any(local_partial.iterdir()):
                n = sum(1 for _ in local_partial.iterdir())
                self._console.print(
                    f"  Uploading {n} cached partial(s) to "
                    f"[bold]{escape(job.sync_dir)}[/bold]"
                )
                self._ssh(f"mkdir -p {job.sync_dir}")
                try:
                    subprocess.run(
                        [
                            "rsync",
                            "--archive",
                            "--compress",
                            "-e",
                            self._ssh_e_flag(),
                            str(local_partial) + "/",
                            f"{self._user}@{self._host}:{job.sync_dir}/",
                        ],
                        capture_output=True,
                        check=True,
                        timeout=120,
                    )
                except Exception as exc:
                    self._console.print(
                        f"  [yellow]Warning: partial upload failed: {exc}[/yellow]"
                    )

        # 2. Upload resume files (e.g. quantized model).
        for local_rel, remote_path in job.resume_files.items():
            local_abs = job.output_dir / local_rel
            if not local_abs.exists():
                continue
            size_mb = local_abs.stat().st_size / (1024 * 1024)
            self._console.print(
                f"  Uploading [bold]{escape(local_rel)}[/bold] "
                f"({size_mb:.0f} MB) → {escape(remote_path)}"
            )
            try:
                subprocess.run(
                    [
                        "rsync",
                        "--archive",
                        "--compress",
                        "-e",
                        self._ssh_e_flag(),
                        str(local_abs),
                        f"{self._user}@{self._host}:{remote_path}",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=600,
                )
            except Exception as exc:
                self._console.print(
                    f"  [yellow]Warning: resume file upload failed: {exc}[/yellow]"
                )

        # 3. Upload resume dirs (e.g. hessian cache from a prior run).
        for local_abs_str, remote_path in job.resume_dirs.items():
            local_abs = Path(local_abs_str)
            if not local_abs.exists() or not local_abs.is_dir():
                continue
            n_files = sum(1 for _ in local_abs.rglob("*") if _.is_file())
            self._console.print(
                f"  Uploading cache dir [bold]{escape(local_abs_str)}[/bold] "
                f"({n_files} file(s)) → {escape(remote_path)}"
            )
            self._ssh(f"mkdir -p {remote_path}")
            try:
                subprocess.run(
                    [
                        "rsync",
                        "--archive",
                        "--compress",
                        "-e",
                        self._ssh_e_flag(),
                        str(local_abs) + "/",
                        f"{self._user}@{self._host}:{remote_path}/",
                    ],
                    capture_output=True,
                    check=True,
                    timeout=1200,
                )
            except Exception as exc:
                self._console.print(
                    f"  [yellow]Warning: cache dir upload failed: {exc}[/yellow]"
                )

    def _sync_intermediate(self, job: VastJob) -> None:
        """Rsync the job's *sync_dir* and *sync_files* (best-effort).

        Called periodically during :meth:`_stream_until_done` to download
        intermediate results for crash resilience.  Directory contents are
        placed in a subdirectory named after the last component of *sync_dir*
        (e.g. ``eval_partial/``).  Individual *sync_files* are placed at
        their configured local-relative paths under *output_dir*.

        Failures are silently ignored so they don't interrupt the main
        streaming loop.

        Args:
            job: Running job whose *sync_dir* / *sync_files* are rsynced.
        """
        if job.sync_dir:
            local_dest = job.output_dir / Path(job.sync_dir).name
            local_dest.mkdir(parents=True, exist_ok=True)
            try:
                cmd = [
                    "rsync",
                    "--archive",
                    "--compress",
                    "--no-specials",
                    "--no-devices",
                    "-e",
                    self._ssh_e_flag(),
                    f"{self._user}@{self._host}:{job.sync_dir}/",
                    str(local_dest) + "/",
                ]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=60,
                )
            except Exception:
                pass  # best-effort; don't interrupt the job

        for remote_path, local_rel in job.sync_files.items():
            local_path = job.output_dir / local_rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                cmd = [
                    "rsync",
                    "--archive",
                    "--compress",
                    "--partial",  # keep partial file so next attempt resumes
                    "-e",
                    self._ssh_e_flag(),
                    f"{self._user}@{self._host}:{remote_path}",
                    str(local_path),
                ]
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=600,  # 10 min for large models
                )
            except Exception:
                pass  # best-effort

        for remote_path, local_abs_str in job.sync_dirs.items():
            local_abs = Path(local_abs_str)
            local_abs.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    [
                        "rsync",
                        "--archive",
                        "--compress",
                        "-e",
                        self._ssh_e_flag(),
                        f"{self._user}@{self._host}:{remote_path}/",
                        str(local_abs) + "/",
                    ],
                    capture_output=True,
                    timeout=1200,
                )
            except Exception:
                pass  # best-effort

    def _rsync(
        self,
        description: str,
        src: str,
        dst: str,
        extra_args: list[str] | None = None,
    ) -> None:
        """Run rsync with a rich progress bar showing overall % and current file.

        Uses ``--info=progress2,name1 --no-inc-recursive`` so rsync scans the
        whole tree first, giving accurate totals from the start.

        Args:
            description: Short label shown in the progress bar (may contain
                rich markup already escaped by the caller).
            src: rsync source path (local or ``user@host:path``).
            dst: rsync destination path (local or ``user@host:path``).
            extra_args: Extra rsync arguments inserted before ``-e``
                (e.g. ``["--exclude", ".git"]``).

        Raises:
            subprocess.CalledProcessError: If rsync exits non-zero.
        """
        cmd = [
            "rsync",
            "--archive",
            "--info=progress2,name1",
            "--no-inc-recursive",
            *(extra_args or []),
            "-e",
            self._ssh_e_flag(),
            src,
            dst,
        ]

        if self._in_tui:
            # TUI mode: no rich Progress widget — log periodic % updates instead.
            self._console.print(f"  Syncing [dim]{description}[/dim]…")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            assert proc.stdout is not None
            buf = ""
            last_pct = -1
            while True:
                chunk = proc.stdout.read(512)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                tokens = re.split(r"[\r\n]", buf)
                buf = tokens.pop()
                for token in tokens:
                    token = token.strip()
                    if not token:
                        continue
                    m = _RSYNC_PROGRESS_RE.match(token)
                    if m:
                        pct = int(m.group(1))
                        if pct // 10 > last_pct // 10:
                            self._console.print(f"  [dim]{pct}%  {m.group(2)}[/dim]")
                            last_pct = pct
            proc.wait()
            self._console.print(f"  [green]✓[/green] {description}")
            if proc.returncode not in _RSYNC_OK_CODES:
                raise subprocess.CalledProcessError(proc.returncode, cmd[0])
            return

        if self._status_bar:
            self._status_bar.pause()
        try:
            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("[dim]{task.fields[files]}[/dim]"),
                TextColumn("{task.description}"),
                console=self._console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    description=f"[dim]{description}[/dim]",
                    total=100,
                    files="",
                )

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                assert proc.stdout is not None

                buf = ""
                while True:
                    data = proc.stdout.read(512)
                    if not data:
                        break
                    buf += data.decode("utf-8", errors="replace")
                    # Split on \r or \n; keep the last incomplete chunk.
                    tokens = re.split(r"[\r\n]", buf)
                    buf = tokens.pop()
                    for token in tokens:
                        token = token.strip()
                        if not token:
                            continue
                        m = _RSYNC_PROGRESS_RE.match(token)
                        if m:
                            pct = int(m.group(1))
                            speed = m.group(2)
                            remaining = int(m.group(3))
                            total_files = int(m.group(4))
                            done = total_files - remaining
                            files_note = (
                                f"{done}/{total_files} files  {speed}"
                                if total_files
                                else speed
                            )
                            progress.update(task, completed=pct, files=files_note)
                        else:
                            # filename line — show just the basename
                            fname = Path(token).name if token else ""
                            progress.update(
                                task,
                                description=f"[dim]{escape(fname)}[/dim]",
                            )

                proc.wait()
        finally:
            if self._status_bar:
                self._status_bar.resume()

        self._console.print(f"  [green]✓[/green] {description}")
        if proc.returncode not in _RSYNC_OK_CODES:
            raise subprocess.CalledProcessError(proc.returncode, cmd[0])

    # -- SSH helpers ---------------------------------------------------------

    def _ssh_flags(self) -> list[str]:
        return [
            "-i",
            str(self._ssh_key),
            "-p",
            str(self._port),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
        ]

    def _ssh_e_flag(self) -> str:
        """Return the rsync ``-e`` value: ``'ssh <flags>'``."""
        return "ssh " + " ".join(self._ssh_flags())

    def _ssh(self, cmd: str, workdir: str | None = None) -> None:
        if workdir:
            cmd = f"cd {workdir} && {cmd}"
        self._console.print(f"  [dim]$[/dim] {escape(cmd)}")
        if self._in_tui:
            # TUI mode: capture output line-by-line and route to the TUI log.
            proc = subprocess.Popen(
                ["ssh", *self._ssh_flags(), f"{self._user}@{self._host}", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                clean = _ANSI_ESCAPE_RE.sub("", line.rstrip())
                if clean:
                    self._console.print(f"  {escape(clean)}")
                if self._log_file is not None:
                    self._log_file.write(_ANSI_ESCAPE_RE.sub("", line))
            proc.wait()
            if self._log_file is not None:
                self._log_file.flush()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        else:
            # Plain mode: stdout and stderr both inherited so tools like
            # uv/apt-get can stream their own progress in real-time.
            result = subprocess.run(
                ["ssh", *self._ssh_flags(), f"{self._user}@{self._host}", cmd],
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

    def _ssh_probe(self) -> bool:
        r = subprocess.run(
            ["ssh", *self._ssh_flags(), f"{self._user}@{self._host}", "true"],
            capture_output=True,
        )
        return r.returncode == 0

    # -- tmux job executor ---------------------------------------------------

    def _run_in_tmux(
        self,
        cmds: list[str],
        workdir: str | None = None,
        env_vars: dict[str, str] | None = None,
        job: VastJob | None = None,
    ) -> None:
        """Run *cmds* inside a detached tmux session; survives SSH disconnects.

        All commands are written to a remote bash script via stdin (no shell
        quoting issues). Output is tee'd to ``_JOB_LOG``; the exit code is
        written to ``_JOB_EXIT`` when the session finishes.  The local side
        streams the log and reconnects automatically on any SSH drop.

        Args:
            cmds: Shell commands to run in sequence.
            workdir: Remote working directory; prepended to each command.
            env_vars: Environment variables exported before the first command.
                Values are shell-quoted.  Not echoed to the log.
            job: Running job; passed to :meth:`_stream_until_done` for
                intermediate artifact sync.

        Raises:
            subprocess.CalledProcessError: If the remote script exits non-zero.
        """
        lines = ["#!/bin/bash", "set -euo pipefail"]
        # Export env vars before any command; do NOT echo them to avoid leaking
        # secrets (e.g. HF_TOKEN) into the log.
        for key, val in sorted((env_vars or {}).items()):
            lines.append(f"export {key}={shlex.quote(val)}")
        for cmd in cmds:
            full = f"cd {workdir} && {cmd}" if workdir else cmd
            # Use single-quoted echo so shell variables (e.g. ${PIPESTATUS[0]})
            # are printed verbatim instead of being expanded at echo time.
            echo_safe = cmd.replace("'", r"'\''")
            lines.append(f"echo '$ {echo_safe}'")
            lines.append(full)
        script = "\n".join(lines) + "\n"

        # Write script via stdin — avoids all shell-quoting pitfalls
        subprocess.run(
            [
                "ssh",
                *self._ssh_flags(),
                f"{self._user}@{self._host}",
                f"cat > {_JOB_SCRIPT}",
            ],
            input=script.encode(),
            check=True,
        )

        # Clean up any stale session / logs from previous runs
        self._ssh(f"tmux kill-session -t {_TMUX_SESSION} 2>/dev/null || true")
        self._ssh(f"rm -f {_JOB_LOG} {_JOB_EXIT}")

        # Launch: pipe all output through tee; record exit code when done
        self._ssh(
            f"tmux new-session -d -s {_TMUX_SESSION} "
            f"'bash {_JOB_SCRIPT} 2>&1 | tee {_JOB_LOG}; "
            f"echo ${{PIPESTATUS[0]}} > {_JOB_EXIT}'"
        )

        self._console.print(
            f"  Job running in tmux [bold]{escape(_TMUX_SESSION)!r}[/bold] — "
            "streaming output [dim](auto-reconnects on SSH drop)[/dim]:"
        )
        self._stream_until_done(job)

    def _stream_until_done(self, job: VastJob | None = None) -> None:
        """Stream ``_JOB_LOG`` until the tmux session exits; reconnect on drop.

        When *job* has a :attr:`~VastJob.sync_dir`, intermediate artifacts are
        rsynced to the local output directory every ``_SYNC_INTERVAL_POLLS``
        poll cycles (~30 s) for crash resilience.

        Args:
            job: Running job (used for intermediate sync).  May be ``None``
                for backward compatibility (resume mode).

        Raises:
            RuntimeError: If SSH drops more than ``_MAX_RECONNECTS`` times.
            subprocess.CalledProcessError: If the remote job exits non-zero.
        """
        byte_offset = 0
        reconnects = 0
        poll_count = 0
        while True:
            try:
                byte_offset = self._fetch_log_from(byte_offset)

                r = subprocess.run(
                    [
                        "ssh",
                        *self._ssh_flags(),
                        f"{self._user}@{self._host}",
                        f"tmux has-session -t {_TMUX_SESSION} 2>/dev/null"
                        " && echo alive || echo done",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if "done" in r.stdout:
                    self._fetch_log_from(byte_offset)  # drain remainder
                    break
                reconnects = 0  # successful poll — reset counter

                # Periodic intermediate artifact sync.
                poll_count += 1
                if (
                    job is not None
                    and (job.sync_dir or job.sync_dirs)
                    and poll_count % _SYNC_INTERVAL_POLLS == 0
                ):
                    self._sync_intermediate(job)

            except Exception as exc:
                reconnects += 1
                if reconnects > _MAX_RECONNECTS:
                    self._console.print(
                        f"\n  [bold red]\\[ERROR][/bold red] SSH dropped "
                        f"{reconnects} consecutive times — aborting."
                    )
                    self._show_log_tail()
                    raise RuntimeError(
                        f"SSH dropped {reconnects} consecutive times; aborting."
                    ) from exc
                self._console.print(
                    f"\n  [yellow]⚠[/yellow]  SSH drop "
                    f"({reconnects}/{_MAX_RECONNECTS}): {escape(str(exc))}"
                    f" — reconnecting in {_RECONNECT_WAIT_S}s…"
                )
                time.sleep(_RECONNECT_WAIT_S)
                continue

            time.sleep(_LOG_POLL_S)

        r = subprocess.run(
            [
                "ssh",
                *self._ssh_flags(),
                f"{self._user}@{self._host}",
                f"cat {_JOB_EXIT} 2>/dev/null",
            ],
            capture_output=True,
            text=True,
        )
        try:
            code = int(r.stdout.strip())
        except ValueError:
            code = 1  # exit file missing or corrupt → treat as failure
        if code != 0:
            self._show_log_tail()
            raise subprocess.CalledProcessError(code, "remote job")

    def _fetch_log_from(self, byte_offset: int) -> int:
        """Print remote log bytes from *byte_offset* and return the new offset.

        Uses ``tail -c +N`` (1-indexed), so offset 0 maps to N=1 (whole file).

        Args:
            byte_offset: Number of bytes already consumed.

        Returns:
            Updated byte offset after printing new content.
        """
        r = subprocess.run(
            [
                "ssh",
                *self._ssh_flags(),
                f"{self._user}@{self._host}",
                f"tail -c +{byte_offset + 1} {_JOB_LOG} 2>/dev/null",
            ],
            capture_output=True,
        )
        if r.returncode == 0 and r.stdout:
            self._write_raw_output(r.stdout)
            byte_offset += len(r.stdout)
        return byte_offset

    def _write_raw_output(self, data: bytes) -> None:
        """Write raw bytes to the TUI log or directly to the terminal.

        Also mirrors to the run log file (ANSI-stripped).

        Args:
            data: Raw bytes (possibly containing ANSI escape sequences).
        """
        text = data.decode("utf-8", errors="replace")
        if self._log_file is not None:
            self._log_file.write(_ANSI_ESCAPE_RE.sub("", text))
            self._log_file.flush()
        if self._in_tui:
            # Route through the _TuiFile bridge: ANSI → Text → RichLog.
            self._console.file.write(text)
            self._console.file.flush()
        else:
            # Flush rich's text buffer before raw-byte write so ordering
            # is preserved (both share the same underlying stdout fd).
            self._console.file.flush()
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    # -- Display helpers -----------------------------------------------------

    def _print_cost_summary(self, elapsed_s: float) -> None:
        """Print a billing cost-estimate panel at the end of a run.

        Args:
            elapsed_s: Wall-clock seconds elapsed since ``run()`` started.
        """
        total_mins, secs = divmod(int(elapsed_s), 60)
        hours, mins = divmod(total_mins, 60)
        if hours:
            duration_str = f"{hours}h {mins:02d}m {secs:02d}s"
        else:
            duration_str = f"{mins}m {secs:02d}s"

        lines = [f"Duration  {duration_str}"]
        if self._dph is not None:
            cost = (elapsed_s / 3600) * self._dph
            lines.append(f"Rate      ${self._dph:.3f}/hr")
            if self._keep:
                lines.append(
                    f"This run  ~${cost:.3f}  "
                    "[dim](instance still running — billing continues)[/dim]"
                )
            else:
                lines.append(f"Total     ~${cost:.3f}")
        self._console.print()
        self._console.print(
            Panel(
                "\n".join(lines),
                title="[bold]Cost estimate[/bold]",
                border_style="dim",
                padding=(0, 2),
            )
        )

    def _log(self, msg: str) -> None:
        self._console.print()
        self._console.print(
            Panel(f"[bold]{escape(msg)}[/bold]", border_style="blue", padding=(0, 2))
        )

    def _phase(self, label: str) -> None:
        self._current_phase = label
        if self._phase_cb is not None:
            self._phase_cb(label)
        self._console.print()
        self._console.rule(f"[bold cyan]{escape(label)}[/bold cyan]")

    def _show_log_tail(self, lines: int = 30) -> None:
        """Print the last *lines* of the remote job log for diagnostics.

        Args:
            lines: Number of tail lines to fetch and display.
        """
        if not self._host or not self._port:
            return
        r = subprocess.run(
            [
                "ssh",
                *self._ssh_flags(),
                f"{self._user}@{self._host}",
                f"tail -n {lines} {_JOB_LOG} 2>/dev/null",
            ],
            capture_output=True,
            timeout=15,
        )
        if r.returncode == 0 and r.stdout:
            self._console.print(
                f"\n  [yellow]─── last {lines} lines of remote log ───[/yellow]"
            )
            self._write_raw_output(r.stdout)

    def _try_save_log(self, output_dir: Path) -> None:
        """Fetch the full remote job log and save it locally (best effort).

        Args:
            output_dir: Local directory to write ``debug_job.log`` into.
        """
        if not self._host or not self._port:
            return
        try:
            r = subprocess.run(
                [
                    "ssh",
                    *self._ssh_flags(),
                    f"{self._user}@{self._host}",
                    f"cat {_JOB_LOG} 2>/dev/null",
                ],
                capture_output=True,
                timeout=15,
            )
            if r.returncode == 0 and r.stdout:
                output_dir.mkdir(parents=True, exist_ok=True)
                log_path = output_dir / "debug_job.log"
                log_path.write_bytes(r.stdout)
                self._console.print(
                    f"\n  [dim]Full remote log saved → "
                    f"[bold]{escape(str(log_path))}[/bold][/dim]"
                )
        except Exception:
            pass  # best-effort; don't obscure the original error


# Default Docker image for GPU jobs.  Override with NEUROPRESS_DOCKER_IMAGE
# to use a custom pre-baked image (see Dockerfile.vast) that cuts cold-start
# time from ~15 min down to ~3 min by pre-warming the uv package cache.
_DEFAULT_IMAGE: str = os.environ.get(
    "NEUROPRESS_DOCKER_IMAGE",
    "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
)


def _uv_setup_eval() -> list[str]:
    """Lighter setup for eval-only jobs: no dev tools, no pytest/ruff.

    Returns:
        List of shell commands to run during the setup phase.
        Each command is executed by :func:`SshRunner._ssh` with the job's
        remote workdir prepended automatically.
    """
    return [
        # Same pattern as _uv_setup: check tmux, g++, ninja, nvcc + dev headers.
        "command -v tmux >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1"
        " && command -v ninja >/dev/null 2>&1"
        " && command -v nvcc >/dev/null 2>&1 || {"
        " n=0;"
        " until apt-get update -qq"
        " && NVCC_PKG=$(apt-cache search --names-only '^cuda-nvcc-[0-9]' "
        "| sort -V | tail -1 | awk '{print $1}') && [ -n \"$NVCC_PKG\" ]"
        " && CUDA_DEV_PKG=$(apt-cache search --names-only"
        " '^cuda-libraries-dev-[0-9]' "
        "| sort -V | tail -1 | awk '{print $1}') && [ -n \"$CUDA_DEV_PKG\" ]"
        " && apt-get install -y --no-install-recommends"
        ' tmux build-essential ninja-build "$NVCC_PKG" "$CUDA_DEV_PKG";'
        " do n=$((n+1)); [ $n -ge 8 ] && exit 1; sleep 15; done; }",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # Only llm + eval extras — skips pytest/ruff/ty (~1 min saved).
        # No leading "cd": _ssh() prepends workdir automatically.
        "~/.local/bin/uv sync --extra llm --extra eval --extra gemlite",
    ]
