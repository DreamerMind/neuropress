r"""Cloud GPU job runner for vast.ai: rent → upload → setup → run → destroy.

Provides :class:`VastRunner`, a concrete :class:`~neuropress.cloud.runner.SshRunner`
subclass that rents a GPU on `vast.ai`_, runs the job, then destroys the
instance (guaranteed even on crash/Ctrl-C via ``atexit``).

Select the backend with ``--cloud``:

* ``vastai`` *(default)* — :class:`VastRunner`: rents a GPU, runs the job,
  destroys the instance.  Requires ``VAST_API_KEY``.
* ``own`` — :class:`~neuropress.cloud.direct_runner.DirectRunner`: connects
  via SSH to a machine you already own.  Nothing is rented or destroyed.
  Requires ``--ssh-host``.

Requires the ``cloud`` extra::

    uv sync --extra cloud

Environment variables::

    VAST_API_KEY              vast.ai API key (--cloud vastai)
    NEUROPRESS_DOCKER_IMAGE   override the Docker image used on vast.ai
                              (see Dockerfile.vast for a pre-baked image that
                              cuts cold-start from ~15 min to ~3 min)

Usage::

    neuropress-cloud --list-jobs

    # vast.ai (rents a GPU, destroys it afterwards)
    neuropress-cloud --cloud vastai --job neuropress-integration
    neuropress-cloud --cloud vastai --job neuropress-eval-compare \
        --model google/gemma-3-270m \
        --compression open_text_c4_gptq_q4_0_256samples \
        --tasks hellaswag --limit 100

    # own machine (no rent, no destroy)
    neuropress-cloud --cloud own --ssh-host 192.168.1.42 --ssh-user myuser \
        --job neuropress-eval-compare \
        --model google/gemma-3-270m \
        --compression open_text_c4_gptq_q4_0_256samples \
        --tasks hellaswag --limit 100

.. _vast.ai: https://cloud.vast.ai

API note: vastai-sdk wraps the CLI dynamically; return-value shapes are not
formally documented.  ``VastRunner._parse_*`` methods handle all known
variants (JSON string, dict, list, or pandas DataFrame).
"""

from __future__ import annotations

import argparse
import atexit
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from neuropress.cloud.runner import (
    _DEFAULT_IMAGE,
    _DESTROY_RETRIES,
    _POLL_INTERVAL_S,
    _RETRY_BACKOFF_S,
    _SSH_TIMEOUT_S,
    GpuSpec,
    SshRunner,
    VastJob,
    _uv_setup_eval,
)

# ISO 3166-1 alpha-2 country codes considered "EU" for --location eu.
# Covers EU member states + Iceland, Norway, Switzerland, and the UK.
_EU_COUNTRY_CODES: frozenset[str] = frozenset(
    c.upper()
    for c in [
        "at",
        "be",
        "bg",
        "cy",
        "cz",
        "de",
        "dk",
        "ee",
        "es",
        "fi",
        "fr",
        "gr",
        "hr",
        "hu",
        "ie",
        "is",
        "it",
        "lt",
        "lu",
        "lv",
        "mt",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "se",
        "si",
        "sk",
        "ch",
        "gb",  # Switzerland + UK
    ]
)


#: Remote cache path on the rented instance.  A sweep's ``cache_dir`` is the
#: LOCAL counterpart that is synced against this, never this path itself.
REMOTE_CACHE = "/workspace/cache"


def check_local_cache_dir(cache_dir: Path | str | None | bool) -> str | None:
    """Return an error message if *cache_dir* is not usable locally.

    A sweep entry's ``cache_dir`` names a **local** directory: the remote path
    is always :data:`_REMOTE_CACHE`, and the local one is seeded to the remote
    before the run and synced back during it.  Writing a remote-looking path
    such as ``/workspace/cache`` therefore fails on every sync with
    ``PermissionError``, aborting the run before the report is written and
    before teardown - hours after the mistake was made.

    Args:
        cache_dir: The configured local cache path, or ``None``/``False`` when
            caching is auto-derived or disabled.

    Returns:
        ``None`` if usable, else a message explaining what to change.
    """
    if cache_dir is None or cache_dir is False:
        return None
    path = Path(cache_dir)
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not os.access(probe, os.W_OK | os.X_OK):
        return (
            f"cache_dir {str(path)!r} is not writable (blocked at {str(probe)!r}). "
            "cache_dir is a LOCAL path - the remote cache is always "
            f"{REMOTE_CACHE!r} - so remote-looking paths cannot work.  Use a "
            "path under the project or home directory, or omit cache_dir to "
            "auto-derive one."
        )
    return None


def select_offer(
    offers: list[dict[str, Any]], exclude: set[int]
) -> dict[str, Any] | None:
    """Return the cheapest offer whose id is not in *exclude*.

    ``offers`` is assumed pre-sorted by price (``_search_offers`` sorts it).
    Exclusion exists because the offer search is deterministic: without it, a
    retry after an SSH timeout re-rents the machine that just failed, so every
    attempt burns the full SSH timeout on the same unreachable host.

    Args:
        offers: Offer dicts from the vast.ai search, cheapest first.
        exclude: Offer ids already known to be bad.

    Returns:
        The chosen offer, or ``None`` if every offer is excluded (or there are
        none).  The caller decides whether that is fatal.
    """
    for offer in offers:
        try:
            oid = int(offer["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if oid not in exclude:
            return offer
    return None


def _location_matches(geolocation: str | None, location: str) -> bool:
    """Return True if *geolocation* matches the user-supplied *location* filter.

    ``"eu"`` is expanded to the :data:`_EU_COUNTRY_CODES` set; the
    ``geolocation`` string is checked for any two-letter suffix that is an EU
    country code (e.g. ``"Germany, DE"`` → matches ``"DE"``).  All other
    values are matched as a case-insensitive substring.

    Args:
        geolocation: The ``geolocation`` field from a vast.ai offer dict.
            May be ``None`` or empty for unknown locations.
        location: The user-supplied filter string (e.g. ``"eu"``, ``"de"``).

    Returns:
        ``True`` iff the offer's geolocation passes the filter.
    """
    geo = (geolocation or "").strip()
    if not geo:
        return False
    loc = location.strip().lower()
    if loc == "eu":
        # Extract the last 2-letter token (country code) from strings like
        # "Germany, DE" or "Netherlands, NL".
        parts = geo.replace(",", " ").split()
        for part in reversed(parts):
            if len(part) == 2 and part.isalpha():
                return part.upper() in _EU_COUNTRY_CODES
        return False
    return loc in geo.lower()


# ---------------------------------------------------------------------------
# VastRunner
# ---------------------------------------------------------------------------


class VastRunner(SshRunner):
    """Manages a GPU job on vast.ai: rent → upload → setup → run → pull → destroy.

    Guarantees ``destroy_instance`` is called via ``atexit`` and SIGTERM/
    SIGINT handlers — the instance is always cleaned up, even on
    ``KeyboardInterrupt`` or unhandled exceptions.

    Args:
        api_key: vast.ai API key.
        ssh_key: Path to SSH private key; defaults to ``~/.ssh/id_rsa``.
        keep: If ``True``, skip ``destroy_instance`` at the end and print the
            instance ID for reuse with ``instance_id``.
        instance_id: Attach to an existing instance instead of renting a new
            one (skips rent and wait-for-SSH phases).

    Raises:
        ImportError: If ``vastai-sdk`` is not installed.
    """

    def __init__(
        self,
        api_key: str = "",
        ssh_key: Path | None = None,
        *,
        keep: bool = False,
        resume: bool = False,
        instance_id: int | None = None,
        ssh_timeout: int | None = None,
        ssh_retries: int = 3,
    ) -> None:
        # Must be initialised before super().__init__() registers atexit so
        # that _cleanup() never sees a missing attribute on early failure.
        self._instance_id: int | None = None
        super().__init__(ssh_key=ssh_key, keep=keep, resume=resume)
        try:
            from vastai_sdk import VastAI
        except ImportError:
            raise ImportError(
                "vastai-sdk is not installed.\n"
                "Run: uv sync --extra cloud   or   pip install vastai-sdk"
            ) from None
        self._client: Any = VastAI(api_key=api_key)
        self._instance_id = instance_id
        self._ssh_timeout = ssh_timeout or _SSH_TIMEOUT_S
        self._ssh_retries = max(1, ssh_retries)
        self._last_offer_id: int | None = None
        self._failed_offers: set[int] = set()

    # ---------------------------------------------------------------- abstract

    def _connect(self, job: VastJob) -> None:
        """Rent a new instance or attach to an existing one.

        On SSH timeout, destroys the failed instance and retries with a fresh
        one up to :attr:`_ssh_retries` times.

        Args:
            job: Job whose ``gpu`` spec is used for offer search (new rent).
        """
        if self._instance_id is not None:
            self._phase("Attach to existing instance")
            self._attach()
            return

        last_exc: Exception | None = None
        for attempt in range(1, self._ssh_retries + 1):
            self._phase(f"Rent instance (attempt {attempt}/{self._ssh_retries})")
            self._rent(job.gpu)
            self._phase(f"Wait for SSH (attempt {attempt}/{self._ssh_retries})")
            try:
                self._wait_for_ssh()
                return  # success
            except TimeoutError as exc:
                last_exc = exc
                iid = self._instance_id
                self._console.print(
                    f"\n  [yellow]⚠[/yellow] SSH timeout on instance "
                    f"[bold]{iid}[/bold] (attempt {attempt}/{self._ssh_retries})"
                    " — destroying and retrying…"
                    if attempt < self._ssh_retries
                    else f"\n  [red]✗[/red] SSH timeout on instance "
                    f"[bold]{iid}[/bold] (attempt {attempt}/{self._ssh_retries})"
                    " — no retries left."
                )
                # Blacklist the offer, or the next _rent picks the same host:
                # the offer search is deterministic and takes the cheapest.
                if self._last_offer_id is not None:
                    self._failed_offers.add(self._last_offer_id)
                # Destroy the stuck instance before retrying.
                self._destroy_instance_now()

        raise last_exc  # type: ignore[misc]

    def _destroy_instance_now(self) -> None:
        """Destroy the current instance immediately and reset state.

        Used by :meth:`_connect` retry logic to clean up a stuck instance
        before renting a new one.  Idempotent.
        """
        iid = self._instance_id
        if iid is None:
            return
        self._instance_id = None
        self._host = None
        self._port = None
        try:
            self._client.destroy_instance(id=iid)
            self._console.print(
                f"  [green]✓[/green] Instance [bold]{iid}[/bold] destroyed."
            )
        except Exception as exc:
            self._console.print(
                f"  [red]Could not destroy instance {iid}:[/red] {escape(str(exc))}"
            )

    def _cleanup(self) -> None:
        """Destroy the rented instance (or preserve it when ``--keep`` is set).

        Idempotent: the ``self._instance_id = None`` guard prevents double
        destruction even when called by both ``atexit`` and the ``finally``
        block in :meth:`~SshRunner.run`.
        """
        if self._instance_id is None:
            return
        if self._keep:
            self._console.print(
                f"\n[yellow][keep][/yellow] Instance "
                f"[bold]{self._instance_id}[/bold] preserved.\n"
                f"  Reconnect:   "
                f"[bold]--instance-id {self._instance_id} --resume[/bold]\n"
                f"  Re-run full: "
                f"[bold]--instance-id {self._instance_id}[/bold]"
            )
            self._instance_id = None  # prevent re-entry
            return
        iid, self._instance_id = self._instance_id, None  # prevent re-entry
        self._console.print(
            f"\n[yellow]\\[cleanup][/yellow] Destroying instance [bold]{iid}[/bold]…"
        )
        for attempt in range(1, _DESTROY_RETRIES + 1):
            try:
                self._client.destroy_instance(id=iid)
                self._console.print(
                    f"  [green]✓[/green] Instance [bold]{iid}[/bold] destroyed."
                )
                return
            except Exception as exc:
                wait = _RETRY_BACKOFF_S * attempt
                self._console.print(
                    f"  [red]Attempt {attempt}/{_DESTROY_RETRIES} failed:[/red] "
                    f"{escape(str(exc))}  (retry in {wait}s)"
                )
                time.sleep(wait)
        self._console.print(
            f"\n  [bold red]!! Could not destroy instance {iid} !![/bold red]\n"
            "  Go to [link]https://cloud.vast.ai/instances/[/link] "
            "and destroy it manually."
        )

    # ----------------------------------------------------------------- public

    def dry_run(self, job: VastJob) -> None:
        """Search for matching offers without renting anything.

        Args:
            job: Job whose ``gpu`` spec is used for the search.
        """
        active = [
            name
            for name, on in [
                ("datacenter", job.gpu.datacenter),
                ("verified", job.gpu.verified),
            ]
            if on
        ]
        if job.gpu.min_inet_down_mbps > 0:
            active.append(f"inet_down≥{job.gpu.min_inet_down_mbps} Mbps")
        if job.gpu.min_reliability > 0:
            active.append(f"reliability≥{job.gpu.min_reliability}")
        if job.gpu.location:
            active.append(f"location={job.gpu.location}")
        filters = ", ".join(active)
        title = "DRY RUN — searching for offers, nothing will be rented"
        if filters:
            title += f"  [dim]({filters})[/dim]"
        self._log(title)
        offers = self._search_offers(job.gpu)
        if not offers:
            self._console.print("  No matching offers found.")
            return

        table = Table(title="Matching GPU Offers (sorted by price)", border_style="dim")
        table.add_column("ID", justify="right", style="dim")
        table.add_column("Location", style="bold cyan")
        table.add_column("GPU", style="bold")
        table.add_column("$/hr", justify="right", style="green")
        table.add_column("CUDA", justify="right")
        table.add_column("↓ Mbps", justify="right")
        table.add_column("Rel", justify="right")
        table.add_column("Disk GB", justify="right")
        for o in offers[:10]:
            inet = o.get("inet_down")
            rel = o.get("reliability2")
            table.add_row(
                str(o["id"]),
                str(o.get("geolocation") or "?"),
                o.get("gpu_name", "?"),
                f"{float(o.get('dph_total', 0)):.3f}",
                str(o.get("cuda_max_good", "?")),
                f"{int(inet)}" if inet is not None else "?",
                f"{float(rel):.2f}" if rel is not None else "?",
                str(o.get("disk_space", "?")),
            )
        self._console.print(table)

    # ----------------------------------------------------------------- private

    # -- vast.ai API ---------------------------------------------------------

    def _rent(self, gpu: GpuSpec) -> None:
        offers = self._search_offers(gpu)
        if not offers:
            raise RuntimeError(f"No GPU offers match query: {gpu.query!r}")

        best = select_offer(offers, self._failed_offers)
        if best is None:
            raise RuntimeError(
                f"All {len(offers)} matching offers already failed SSH: "
                f"{sorted(self._failed_offers)}"
            )
        price = float(best.get("dph_total", 9999))
        self._dph = price
        if price > gpu.max_dph:
            raise RuntimeError(
                f"Selected offer is ${price:.3f}/hr, exceeds max_dph=${gpu.max_dph:.2f}"
            )

        oid = int(best["id"])
        self._last_offer_id = oid
        location = escape(str(best.get("geolocation") or "?"))
        inet_down = best.get("inet_down")
        reliability = best.get("reliability2")
        parts = [
            f"  [dim]offer[/dim] [bold]{oid}[/bold]",
            f"[bold cyan]{location}[/bold cyan]",
            escape(str(best.get("gpu_name", "?"))),
            f"[green]${price:.3f}/hr[/green]",
            f"CUDA {escape(str(best.get('cuda_max_good', '?')))}",
        ]
        if inet_down is not None:
            parts.append(f"[dim]↓{int(inet_down)} Mbps[/dim]")
        if reliability is not None:
            parts.append(f"[dim]rel {float(reliability):.2f}[/dim]")
        self._console.print("  ".join(parts))

        # Snapshot existing IDs before creation; the SDK may return None.
        before = self._current_instance_ids()

        raw = self._client.create_instance(
            id=oid,
            image=gpu.image,
            disk=gpu.disk_gb,
        )

        # Prefer the return value; fall back to diffing show_instances.
        try:
            if raw is not None:
                self._instance_id = self._extract_id(raw, keys=["new_contract", "id"])
        except ValueError:
            pass

        if self._instance_id is None:
            after = self._current_instance_ids()
            new_ids = after - before
            if not new_ids:
                raise RuntimeError(
                    "create_instance returned no usable ID and "
                    "show_instances found no new instance."
                )
            self._instance_id = max(new_ids)  # newest = highest ID

        self._console.print(
            f"  [dim]instance ID:[/dim] [bold]{self._instance_id}[/bold]"
        )

    def _search_offers(self, gpu: GpuSpec) -> list[dict[str, Any]]:
        import re

        query = gpu.query
        if gpu.gpu_name is not None:
            if re.search(r"\bgpu_name=\S+", query):
                query = re.sub(r"\bgpu_name=\S+", f"gpu_name={gpu.gpu_name}", query)
            else:
                query += f" gpu_name={gpu.gpu_name}"
        if gpu.datacenter:
            query += " datacenter=True"
        if gpu.verified:
            query += " verified=True"
        if gpu.min_inet_down_mbps > 0:
            query += f" inet_down>={gpu.min_inet_down_mbps}"
        if gpu.min_reliability > 0:
            query += f" reliability2>={gpu.min_reliability}"
        raw = self._client.search_offers(query=query)
        offers = self._parse_collection(raw, key="offers")
        if gpu.location:
            offers = [
                o
                for o in offers
                if _location_matches(o.get("geolocation"), gpu.location)
            ]
        offers.sort(key=lambda o: float(o.get("dph_total", 9999)))
        return offers

    def _wait_for_ssh(self) -> None:
        timeout = self._ssh_timeout
        deadline = time.monotonic() + timeout
        prev_status = ""

        if self._in_tui:
            # TUI mode: no rich.Progress (its Live background thread conflicts
            # with Textual).  Log status changes directly to the TUI console.
            while time.monotonic() < deadline:
                try:
                    info = self._instance_info()
                except Exception:
                    info = {}

                status = info.get("actual_status", "")
                host = info.get("ssh_host") or info.get("public_ipaddr")
                port = info.get("ssh_port")

                if status and status != prev_status:
                    arrow = (
                        f"[dim]{escape(prev_status)}[/dim] → " if prev_status else ""
                    )
                    self._console.print(
                        f"  [dim]●[/dim] {arrow}[bold]{escape(status)}[/bold]"
                    )
                    prev_status = status

                if status == "running" and host and port:
                    self._host, self._port = host, int(port)
                    if self._ssh_probe():
                        self._console.print(
                            f"  [green]✓[/green] SSH ready — "
                            f"[bold]{escape(str(host))}[/bold]:{port}"
                        )
                        return

                remaining_s = max(0, int(deadline - time.monotonic()))
                mins, secs = divmod(remaining_s, 60)
                self._console.print(
                    f"  [dim]waiting… {mins}:{secs:02d} remaining[/dim]"
                )
                time.sleep(_POLL_INTERVAL_S)

            raise TimeoutError(
                f"Instance {self._instance_id} was not SSH-ready within {timeout}s"
            )

        if self._status_bar:
            self._status_bar.pause()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                TextColumn("[dim]{task.fields[remaining]}[/dim]"),
                console=self._console,
                transient=False,
            ) as progress:
                task = progress.add_task(
                    description="[dim]starting…[/dim]",
                    total=None,
                    remaining=f"≤{timeout // 60} min",
                )

                while time.monotonic() < deadline:
                    try:
                        info = self._instance_info()
                    except Exception:
                        info = {}  # transient API error — retry

                    status = info.get("actual_status", "")
                    host = info.get("ssh_host") or info.get("public_ipaddr")
                    port = info.get("ssh_port")

                    remaining_s = max(0, int(deadline - time.monotonic()))
                    mins, secs = divmod(remaining_s, 60)

                    # Status change → print a new permanent line above the spinner
                    if status and status != prev_status:
                        arrow = (
                            f"[dim]{escape(prev_status)}[/dim] → "
                            if prev_status
                            else ""
                        )
                        progress.console.print(
                            f"  [dim]●[/dim] {arrow}[bold]{escape(status)}[/bold]"
                        )
                        prev_status = status

                    progress.update(
                        task,
                        description=f"[bold]{escape(status or 'starting')}[/bold]",
                        remaining=f"{mins}:{secs:02d} remaining",
                    )

                    if status == "running" and host and port:
                        self._host, self._port = host, int(port)
                        if self._ssh_probe():
                            progress.console.print(
                                f"  [green]✓[/green] SSH ready — "
                                f"[bold]{escape(str(host))}[/bold]:{port}"
                            )
                            return

                    time.sleep(_POLL_INTERVAL_S)
        finally:
            if self._status_bar:
                self._status_bar.resume()

        raise TimeoutError(
            f"Instance {self._instance_id} was not SSH-ready within {timeout}s"
        )

    def _attach(self) -> None:
        """Attach to an existing instance: look up host/port and probe SSH.

        Raises:
            RuntimeError: If the instance is not running or SSH is unreachable.
        """
        info = self._instance_info()
        status = info.get("actual_status", "")
        if status != "running":
            raise RuntimeError(
                f"Instance {self._instance_id} is not running (status={status!r}).\n"
                "  It may still be loading — wait a moment and retry."
            )
        try:
            self._dph = float(info["dph_total"])
        except (KeyError, TypeError, ValueError):
            pass
        host = info.get("ssh_host") or info.get("public_ipaddr")
        port = info.get("ssh_port")
        if not host or not port:
            raise RuntimeError(
                f"Instance {self._instance_id} has no SSH address in API response."
            )
        self._host, self._port = host, int(port)
        if not self._ssh_probe():
            raise RuntimeError(
                f"Instance {self._instance_id} is running but SSH is not responding "
                f"at {host}:{port}."
            )
        self._console.print(
            f"  [green]✓[/green] Attached to instance "
            f"[bold]{self._instance_id}[/bold] — "
            f"[bold]{escape(str(host))}[/bold]:{port}"
        )

    # -- Instance info -------------------------------------------------------

    def list_instances_info(self) -> list[dict[str, Any]]:
        """Return info dicts for all instances on this account.

        Returns:
            List of dicts with at least ``"id"`` and ``"actual_status"`` keys,
            sorted by instance ID descending (newest first).
        """
        raw = self._client.show_instances()
        instances = self._parse_collection(raw, key="instances")
        return sorted(
            [i for i in instances if "id" in i],
            key=lambda i: int(i["id"]),
            reverse=True,
        )

    def _current_instance_ids(self) -> set[int]:
        """Return the set of all instance IDs currently on this account."""
        return {int(i["id"]) for i in self.list_instances_info()}

    def _instance_info(self) -> dict[str, Any]:
        raw = self._client.show_instance(id=self._instance_id)
        items = self._parse_collection(raw, key="instances")
        return items[0] if items else self._coerce_dict(raw)

    # -- Response parsing ----------------------------------------------------
    # vastai-sdk return shapes are undocumented and vary by version:
    # they may be a JSON string, a dict, a list, or a pandas DataFrame.

    @staticmethod
    def _parse_collection(raw: Any, key: str) -> list[dict[str, Any]]:
        """Normalise a collection response to a list of dicts."""
        if hasattr(raw, "to_dict"):  # pandas DataFrame
            return raw.to_dict(orient="records")
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return []  # empty or malformed body — treat as no results
        if isinstance(raw, dict):
            val = raw.get(key, [raw])
            return val if isinstance(val, list) else []
        return []

    @staticmethod
    def _coerce_dict(raw: Any) -> dict[str, Any]:
        """Normalise a single-object response to a dict."""
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                d = json.loads(raw)
                return d if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                return {}  # empty or malformed body
        return {}

    @staticmethod
    def _extract_id(raw: Any, keys: list[str]) -> int:
        """Extract an integer ID from a response using candidate key names."""
        d = VastRunner._coerce_dict(raw)
        for k in keys:
            if k in d:
                return int(d[k])
        raise ValueError(f"Cannot find instance ID (tried {keys}) in response: {raw!r}")


# ---------------------------------------------------------------------------
# Pre-built job definitions
# ---------------------------------------------------------------------------


def _uv_setup() -> list[str]:
    """Full setup: build-essential + ninja + uv + all extras including dev tools.

    Returns:
        List of shell commands to run during the setup phase.
        Each command is executed by :func:`SshRunner._ssh` with the job's
        remote workdir prepended automatically.
    """
    return [
        # Install tmux + build-essential + ninja + CUDA toolkit if missing.
        # g++ is needed to compile the nnef C++ extension via uv sync;
        # ninja + nvcc are needed for torch CUDA JIT kernels.
        # cuda-nvcc-XX-Y provides nvcc; cuda-libraries-dev-XX-Y provides
        # dev headers (cusparse.h, cublas.h, etc.) needed by torch JIT.
        # Package names are version-suffixed (e.g. cuda-nvcc-12-4) — we
        # discover the right suffix dynamically via apt-cache search.
        # Pre-baked images (NEUROPRESS_DOCKER_IMAGE) have all → skip entirely.
        # Retry loop caps at 8 x 15 s = 2 min to avoid infinite hangs.
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
        # Install uv (writes to ~/.local/bin/uv)
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # uv sync auto-downloads Python 3.13 if needed; extras cover all deps.
        # No leading "cd" here: _ssh() prepends workdir automatically.
        "~/.local/bin/uv sync --extra llm --extra eval --extra dev --extra gemlite",
    ]


def _auto_cache_dir(model_id: str) -> Path:
    """Derive a local cache directory from the HuggingFace model ID.

    Uses ``$NEUROPRESS_DATA_DIR/cache/{safe_model}`` when the env var is set,
    otherwise falls back to ``./data/cache/{safe_model}``.
    """
    data_dir = os.environ.get("NEUROPRESS_DATA_DIR")
    base = Path(data_dir) if data_dir else Path("./data")
    safe_model = model_id.replace("/", "__")
    return base / "cache" / safe_model


def job_neuropress_integration(project_root: Path, output_dir: Path) -> VastJob:
    """Integration test job: GPTQ calibration + NNEF export.

    Runs ``RUN_INTEGRATION=1 uv run pytest tests/integration/ -v``.
    Pulls back the pytest log.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"
    return VastJob(
        name="neuropress-integration",
        gpu=GpuSpec(
            # RTX 3090/4090 have 24 GB — enough for Gemma-3-270M + GPTQ
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.55"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=60,  # model cache + NNEF export
            max_dph=0.60,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup(),
        run=[
            (
                "RUN_INTEGRATION=1 ~/.local/bin/uv run pytest tests/integration/ -v "
                "--tb=short 2>&1 | tee /tmp/integration.log ; "
                # propagate pytest exit code even through tee
                "exit ${PIPESTATUS[0]}"
            ),
        ],
        artifacts=["/tmp/integration.log"],
        output_dir=output_dir / "integration",
        required_env_vars=["HF_TOKEN"],
    )


def job_neuropress_eval_compare(
    project_root: Path,
    output_dir: Path,
    *,
    model: str,
    compression: str,
    tasks: list[str],
    limit: float | None = None,
    num_fewshot: int | None = None,
    device: str = "cuda",
    save_model: bool = False,
    skip_baseline: bool = False,
    batch_size: str | int = "auto",
    cache_dir: Path | str | None | bool = None,
) -> VastJob:
    """lm-eval harness compare job: float vs quantized.

    Runs ``neuropress-eval compare`` and pulls back the JSON report.
    Optionally also saves the quantized model as a safetensors checkpoint
    and downloads it as an artifact.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        model: HuggingFace model ID (e.g. ``"google/gemma-3-270m"``).
        compression: Key in ``NEUROPRESS_COMPRESSION``
            (e.g. ``"open_text_c4_gptq_q4_0"``).
        tasks: lm-eval task names (e.g. ``["hellaswag", "arc_easy"]``).
        limit: Max examples per task (float fraction or int count).
        num_fewshot: Few-shot examples; ``None`` uses the task default.
        device: PyTorch device string passed to ``neuropress-eval``
            (default: ``"cuda"``).
        save_model: If ``True``, save the quantized model to
            ``/tmp/quantized_model.safetensors`` on the remote and download
            it as an artifact alongside the eval report.
        skip_baseline: If ``True``, skip baseline (float model) evaluation.
            The report will contain ``null`` for baseline results.
        batch_size: Batch size for lm-eval (default: ``"auto"``).
            ``"auto"`` lets lm-eval pick the largest batch that fits in GPU
            memory.  An integer fixes the batch size explicitly.
        cache_dir: Local path to the quantization cache directory.  ``None``
            (default) auto-derives the path from *model* using
            :func:`_auto_cache_dir`.  ``False`` explicitly disables caching.
            When a path is used, the directory is uploaded to
            ``/workspace/cache`` before the run and synced back after.

    Returns:
        Configured :class:`VastJob`.
    """
    # Resolve cache_dir: None → auto-derive, False → disable, path → use as-is.
    if cache_dir is False:
        cache_dir = None
    elif cache_dir is None:
        cache_dir = _auto_cache_dir(model)

    remote = "/workspace/neuropress"
    _REMOTE_MODEL = "/tmp/quantized_model.safetensors"
    _REMOTE_CACHE = REMOTE_CACHE
    # Partial dir is auto-derived from --output's parent by _cmd_compare:
    # /tmp/eval_report.json → /tmp/eval_partial/
    _REMOTE_PARTIAL = "/tmp/eval_partial"

    eval_args = [
        "compare",
        model,
        "--tasks",
        *tasks,
        "--compression",
        compression,
        "--device",
        device,
        "--output",
        "/tmp/eval_report.json",
    ]
    if limit is not None:
        eval_args += ["--limit", str(limit)]
    if num_fewshot is not None:
        eval_args += ["--num-fewshot", str(num_fewshot)]
    if save_model:
        eval_args += ["--save-model", _REMOTE_MODEL]
    if skip_baseline:
        eval_args += ["--skip-baseline"]
    eval_args += ["--batch-size", str(batch_size)]
    if cache_dir is not None:
        eval_args += ["--cache-dir", _REMOTE_CACHE]

    eval_cmd = (
        f"~/.local/bin/uv run neuropress-eval {' '.join(eval_args)} "
        "2>&1 | tee /tmp/eval.log ; exit ${PIPESTATUS[0]}"
    )

    safe_model = model.replace("/", "_")
    safe_tasks = "_".join(tasks)

    artifacts = ["/tmp/run_config.json", "/tmp/eval_report.json", "/tmp/eval.log"]
    sync_files: dict[str, str] = {}
    resume_files: dict[str, str] = {}
    resume_dirs: dict[str, str] = {}
    sync_dirs: dict[str, str] = {}
    if save_model:
        # Sync the model incrementally so it's downloaded as soon as
        # quantization finishes, not deferred to job completion.
        sync_files[_REMOTE_MODEL] = "quantized_model.safetensors"
        # On retry, upload the locally-cached model so quantization is skipped.
        resume_files["quantized_model.safetensors"] = _REMOTE_MODEL
    if cache_dir is not None:
        local_cache_abs = str(Path(cache_dir).resolve())
        # Seed remote cache from local before run; download back during/after.
        resume_dirs[local_cache_abs] = _REMOTE_CACHE
        sync_dirs[_REMOTE_CACHE] = local_cache_abs

    meta: dict[str, Any] = {
        "model": model,
        "compression": compression,
        "tasks": tasks,
        "device": device,
    }
    if limit is not None:
        meta["limit"] = limit
    if num_fewshot is not None:
        meta["num_fewshot"] = num_fewshot
    if save_model:
        meta["save_model"] = True

    return VastJob(
        name=f"neuropress-eval-{safe_model}-{compression}-{safe_tasks}",
        gpu=GpuSpec(
            query=(
                "num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.55 gpu_ram>=20"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=60,
            max_dph=0.60,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[eval_cmd],
        artifacts=artifacts,
        output_dir=output_dir / "eval",
        sync_dir=_REMOTE_PARTIAL,
        sync_files=sync_files,
        sync_dirs=sync_dirs,
        resume_files=resume_files,
        resume_dirs=resume_dirs,
        required_env_vars=["HF_TOKEN"],
        env_vars={
            # Enable Rust-based multi-threaded HuggingFace downloader.
            # Requires hf-transfer (installed via --extra eval).
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # Reduce CUDA allocator fragmentation — important for large VQ
            # models where repeated allocation/free of varied-size tensors
            # (distance matrices, index buffers) causes free-but-reserved bloat.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        meta=meta,
    )


def job_neuropress_bench(
    project_root: Path,
    output_dir: Path,
    *,
    module: str = "all",
    shapes: str = "all",
    batch_sizes: str = "1,16,256",
    warmup: int = 5,
    iters: int = 20,
    device: str = "cuda:0",
) -> VastJob:
    """Synthetic kernel micro-benchmark on a remote GPU.

    Runs ``neuropress-bench`` with the given options and pulls back a JSON
    report with per-backend timings for every shape/batch combination.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        module: Module type — ``"linear"``, ``"vq"``, ``"embedding"``,
            or ``"all"`` (default).
        shapes: Shape preset — ``"small"``, ``"medium"``, ``"large"``,
            ``"xlarge"``, or ``"all"`` (default).
        batch_sizes: Comma-separated batch sizes (default: ``"1,16,256"``).
        warmup: Warmup iterations (default: 5).
        iters: Timed iterations (default: 20).
        device: PyTorch device string (default: ``"cuda:0"``).

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"
    bench_args = [
        "--module",
        module,
        "--shapes",
        shapes,
        "--batch-sizes",
        batch_sizes,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--device",
        device,
        "--output-file",
        "/tmp/bench_results.json",
    ]
    bench_cmd = (
        "~/.local/bin/uv run neuropress-bench "
        f"{' '.join(bench_args)} "
        "2>&1 | tee /tmp/bench.log ; exit ${PIPESTATUS[0]}"
    )

    meta: dict[str, Any] = {
        "module": module,
        "shapes": shapes,
        "batch_sizes": batch_sizes,
        "warmup": warmup,
        "iters": iters,
    }

    return VastJob(
        name=f"neuropress-bench-{module}-{shapes}",
        gpu=GpuSpec(
            query=(
                "num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.55 gpu_ram>=16"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=30,
            max_dph=0.60,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[bench_cmd],
        artifacts=["/tmp/bench_results.json", "/tmp/bench.log"],
        output_dir=output_dir / "bench",
        meta=meta,
    )


def job_neuropress_bench_inference(
    project_root: Path,
    output_dir: Path,
    *,
    model: str,
    compression: str | None = None,
    checkpoint: str | None = None,
    pp: list[int] | None = None,
    tg: int = 128,
    warmup: int = 3,
    iters: int = 10,
    device: str = "cuda",
    lowering_config: str | None = None,
) -> VastJob:
    """End-to-end inference throughput benchmark on a remote RTX 4090.

    Runs ``scripts/benchmark_inference.py`` to measure pp=512, pp=1024, and
    tg=128 throughput for both the float baseline and the quantized model,
    reporting RTFx = quantized / float for each scenario.

    Either *checkpoint* (path to a pre-saved local checkpoint to upload) or
    *compression* (key in ``NEUROPRESS_COMPRESSION`` to quantize on the remote)
    must be provided — not both.  Prefer *checkpoint* when a model has already
    been quantized and saved locally: it skips ~35 min of re-quantization.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        model: HuggingFace model ID (e.g. ``"google/gemma-3-270m"``).
        compression: Key in ``NEUROPRESS_COMPRESSION`` — quantizes on the
            remote.  Mutually exclusive with *checkpoint*.
        checkpoint: Local path to a pre-saved ``.safetensors`` checkpoint or
            directory.  Uploaded alongside the repo and passed via
            ``--checkpoint``.  Mutually exclusive with *compression*.
        pp: Prompt lengths to benchmark (default: ``[512, 1024]``).
        tg: Number of tokens to generate for the tg scenario (default: 128).
        warmup: Warmup iterations per scenario (default: 3).
        iters: Timed iterations per scenario (default: 10).
        device: PyTorch device string (default: ``"cuda"``).
        lowering_config: Path to a YAML kernel-lowering config (see
            ``configs/lowering/``).  Omit to use all-auto defaults.  When the
            path is inside the project repo it is available on the remote
            automatically; external files are uploaded separately.

    Returns:
        Configured :class:`VastJob`.

    Raises:
        ValueError: If neither or both of *compression* / *checkpoint* are set.
    """
    if (compression is None) == (checkpoint is None):
        raise ValueError("Exactly one of compression or checkpoint must be set.")
    if pp is None:
        pp = [512, 1024]

    remote = "/workspace/neuropress"
    pp_args = " ".join(str(p) for p in pp)

    uploads: list[Path] = [project_root]
    if checkpoint is not None:
        ckpt_local = Path(checkpoint).resolve()
        uploads.append(ckpt_local)
        # Place the checkpoint under the remote workdir so the script can
        # address it with a stable relative path.
        remote_ckpt = f"{remote}/{ckpt_local.name}"
        model_arg = f"--checkpoint {remote_ckpt}"
    else:
        model_arg = f"--compression {compression}"

    lowering_arg = ""
    if lowering_config is not None:
        lc_path = Path(lowering_config)
        # If outside the project tree, upload it as a standalone file.
        try:
            lc_path.relative_to(project_root.resolve())
            remote_lc = f"{remote}/{lc_path.relative_to(project_root.resolve())}"
        except ValueError:
            uploads.append(lc_path.resolve())
            remote_lc = f"{remote}/{lc_path.name}"
        lowering_arg = f"--lowering-config {remote_lc} "

    bench_cmd = (
        f"~/.local/bin/uv run python scripts/benchmark_inference.py "
        f"--model {model} "
        f"{model_arg} "
        f"--pp {pp_args} "
        f"--tg {tg} "
        f"--warmup {warmup} "
        f"--iters {iters} "
        f"--device {device} "
        f"{lowering_arg}"
        "--output /tmp/bench_inference.json "
        "2>&1 | tee /tmp/bench_inference.log ; exit ${PIPESTATUS[0]}"
    )

    safe_model = model.replace("/", "_")
    label = Path(checkpoint).name if checkpoint else compression
    meta: dict[str, Any] = {
        "model": model,
        "pp": pp,
        "tg": tg,
        "warmup": warmup,
        "iters": iters,
    }
    if compression:
        meta["compression"] = compression
    if checkpoint:
        meta["checkpoint"] = checkpoint

    return VastJob(
        name=f"neuropress-bench-inference-{safe_model}-{label}",
        gpu=GpuSpec(
            # RTX 4090 — community reference GPU for inference benchmarks.
            # 24 GB VRAM fits float + quantized 270M-1B models side-by-side.
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=60,
            max_dph=0.70,
        ),
        uploads=uploads,
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[bench_cmd],
        artifacts=["/tmp/bench_inference.json", "/tmp/bench_inference.log"],
        output_dir=output_dir / "bench_inference",
        required_env_vars=["HF_TOKEN"],
        env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        meta=meta,
    )


def job_neuropress_bench_kmeans(
    project_root: Path,
    output_dir: Path,
    *,
    n_values: list[int] | None = None,
    k_values: list[int] | None = None,
    d_values: list[int] | None = None,
    backends: list[str] | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> VastJob:
    """K-means backend scaling benchmark on a remote RTX 4090.

    Runs ``scripts/bench_kmeans.py`` to measure E-step and full-iteration
    throughput across (N, k, D) combinations, comparing TorchKMeansBackend
    vs CudaKMeansBackend (the built-in CUDA kernel).

    The CUDA kernel is JIT-compiled on first run; nvcc is required and is
    installed via the standard setup sequence.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        n_values: Number-of-vectors sweep (default: 10k 50k 100k 500k 1M).
        k_values: Codebook-size sweep (default: 256 1024 4096 16384 65536).
        d_values: Vector-length sweep (default: 4).
        backends: Backends to compare (default: torch cuda).
        warmup: Warmup iterations per measurement (default: 3).
        iters: Timed iterations per measurement (default: 10).

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"

    def _flag(name: str, values: list) -> str:
        return f"--{name} " + " ".join(str(v) for v in values)

    args_parts = []
    if n_values:
        args_parts.append(_flag("n-values", n_values))
    if k_values:
        args_parts.append(_flag("k-values", k_values))
    if d_values:
        args_parts.append(_flag("d-values", d_values))
    if backends:
        args_parts.append(_flag("backends", backends))
    args_parts.append(f"--warmup {warmup}")
    args_parts.append(f"--iters {iters}")
    args_parts.append("--output /tmp/bench_kmeans.json")

    bench_cmd = (
        "~/.local/bin/uv run python scripts/bench_kmeans.py "
        + " ".join(args_parts)
        + " 2>&1 | tee /tmp/bench_kmeans.log ; exit ${PIPESTATUS[0]}"
    )

    return VastJob(
        name="neuropress-bench-kmeans",
        gpu=GpuSpec(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=40,
            max_dph=0.70,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[bench_cmd],
        artifacts=["/tmp/bench_kmeans.json", "/tmp/bench_kmeans.log"],
        output_dir=output_dir / "bench_kmeans",
        meta={
            "n_values": n_values,
            "k_values": k_values,
            "d_values": d_values,
            "backends": backends,
            "warmup": warmup,
            "iters": iters,
        },
    )


def job_neuropress_profile_layers(
    project_root: Path,
    output_dir: Path,
    *,
    model: str,
    checkpoint: str,
    lowering_config: str | None = None,
    seq_len: int = 1,
    warmup: int = 3,
    iters: int = 10,
    top: int = 30,
    device: str = "cuda",
    attn_implementation: str | None = None,
) -> VastJob:
    """Per-layer forward timing to identify inference bottlenecks.

    Runs ``scripts/profile_layers.py`` on a remote RTX 4090, timing each
    leaf module's forward pass and reporting a ranked table by latency.
    Also reports total wall-clock time vs tracked module time so the
    untracked SDPA gap is visible.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        model: HuggingFace model ID.
        checkpoint: Local path to a pre-saved ``.safetensors`` checkpoint.
        lowering_config: Path to a YAML kernel-lowering config.
        seq_len: Sequence length — 1 for TG, 512 for PP.
        warmup: Warmup iterations.
        iters: Timed iterations.
        top: Number of slowest layers to show.
        device: PyTorch device string.
        attn_implementation: Override HF attention backend — one of
            ``"eager"``, ``"sdpa"``, or ``"flash_attention_2"``.
            ``None`` uses the model's own default.

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"
    ckpt_local = Path(checkpoint).resolve()
    remote_ckpt = f"{remote}/{ckpt_local.name}"
    uploads: list[Path] = [project_root, ckpt_local]

    lowering_arg = ""
    if lowering_config is not None:
        lc_path = Path(lowering_config)
        try:
            lc_path.relative_to(project_root.resolve())
            remote_lc = f"{remote}/{lc_path.relative_to(project_root.resolve())}"
        except ValueError:
            uploads.append(lc_path.resolve())
            remote_lc = f"{remote}/{lc_path.name}"
        lowering_arg = f"--lowering-config {remote_lc} "

    attn_arg = (
        f"--attn-implementation {attn_implementation} " if attn_implementation else ""
    )

    cmd = (
        f"~/.local/bin/uv run python scripts/profile_layers.py "
        f"--model {model} "
        f"--checkpoint {remote_ckpt} "
        f"{lowering_arg}"
        f"{attn_arg}"
        f"--seq-len {seq_len} "
        f"--warmup {warmup} "
        f"--iters {iters} "
        f"--top {top} "
        f"--device {device} "
        "2>&1 | tee /tmp/profile_layers.log"
    )

    return VastJob(
        name=f"neuropress-profile-layers-{model.replace('/', '_')}-seq{seq_len}",
        gpu=GpuSpec(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=60,
            max_dph=0.70,
        ),
        uploads=uploads,
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[cmd],
        artifacts=["/tmp/profile_layers.log"],
        output_dir=output_dir / "profile_layers",
        required_env_vars=["HF_TOKEN"],
        env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        meta={"model": model, "checkpoint": checkpoint, "seq_len": seq_len},
    )


def job_neuropress_test_save(
    project_root: Path,
    output_dir: Path,
    *,
    nc: int = 32768,
    no_embeddings: bool = False,
) -> VastJob:
    """Smoke test: random VQ on Gemma 4 E2B → adapter apply → save_quantized_model.

    Loads the real google/gemma-4-E2B model, assigns random QTensorTractVQ
    to every eligible layer (matching production config), runs adapter.apply()
    and save_quantized_model to surface any remaining OOM sites.  No
    calibration or k-means — completes in ~5 min on RTX 4090.

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        nc: Codebook size (default: 32768, matching production).
        no_embeddings: Quantize linear layers only (skip nn.Embedding).

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"
    args_parts = [f"--nc {nc}"]
    if no_embeddings:
        args_parts.append("--no-embeddings")
    args_parts.append("--out /tmp/test_save.safetensors")

    cmd = (
        "~/.local/bin/uv run python scripts/test_save.py "
        + " ".join(args_parts)
        + " 2>&1 | tee /tmp/test_save.log ; exit ${PIPESTATUS[0]}"
    )

    return VastJob(
        name="neuropress-test-save",
        gpu=GpuSpec(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=60,
            max_dph=0.70,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[cmd],
        artifacts=["/tmp/test_save.log"],
        output_dir=output_dir / "test_save",
        required_env_vars=["HF_TOKEN"],
        env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        meta={"nc": nc, "no_embeddings": no_embeddings},
    )


def job_neuropress_bench_assign(
    project_root: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
    warmup: int = 3,
    iters: int = 10,
) -> VastJob:
    """Grouped vs per-group-loop k-means assignment benchmark on a remote RTX 4090.

    Runs ``scripts/bench_assign.py`` comparing the old Python G-loop against the
    new ``kmeans_assign_grouped`` single-launch kernel for production-size layers
    (lm_head R=262144 G=384 NC=32768, q_proj R=1024 G=256 NC=32768).

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        smoke: Run tiny smoke config only (R=64 G=8 NC=256, <5s).
        warmup: Warmup iterations per measurement (default: 3).
        iters: Timed iterations per measurement (default: 10).

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"

    args_parts = []
    if smoke:
        args_parts.append("--smoke")
    else:
        args_parts.append(f"--warmup {warmup}")
        args_parts.append(f"--iters {iters}")
    args_parts.append("--output /tmp/bench_assign.json")

    bench_cmd = (
        "~/.local/bin/uv run python scripts/bench_assign.py "
        + " ".join(args_parts)
        + " 2>&1 | tee /tmp/bench_assign.log ; exit ${PIPESTATUS[0]}"
    )

    return VastJob(
        name="neuropress-bench-assign",
        gpu=GpuSpec(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=40,
            max_dph=0.70,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[bench_cmd],
        artifacts=["/tmp/bench_assign.json", "/tmp/bench_assign.log"],
        output_dir=output_dir / "bench_assign",
        meta={"smoke": smoke, "warmup": warmup, "iters": iters},
    )


def job_neuropress_profile_mstep(
    project_root: Path,
    output_dir: Path,
    *,
    nc: int = 32768,
    n_iters: int = 3,
    n_layers: int | None = None,
) -> VastJob:
    """M-step per-op timing profiler on a remote RTX 4090.

    Runs ``scripts/profile_mstep.py`` with ``VPTQ_PROFILE_MSTEP=1``.
    Creates synthetic Gemma 4 E2B-like layers (no HF download needed) and
    runs *n_iters* Lloyd iterations, emitting a per-op breakdown table from
    ``_refine_centroids`` on iteration 1 (warm cache, warm GPU).

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        nc: Codebook size (default 32768).
        n_iters: Lloyd iterations to run (default 3; profile fires on iter 1).
        n_layers: Override layer count (default: Gemma 4 E2B-like ~126).

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"

    args_parts = [f"--nc {nc}", f"--n-iters {n_iters}"]
    if n_layers is not None:
        args_parts.append(f"--n-layers {n_layers}")

    cmd = (
        "VPTQ_PROFILE_MSTEP=1 ~/.local/bin/uv run python scripts/profile_mstep.py "
        + " ".join(args_parts)
        + " 2>&1 | tee /tmp/profile_mstep.log ; exit ${PIPESTATUS[0]}"
    )

    return VastJob(
        name="neuropress-profile-mstep",
        gpu=GpuSpec(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=20,
            max_dph=0.70,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[cmd],
        artifacts=["/tmp/profile_mstep.log"],
        output_dir=output_dir / "profile_mstep",
        meta={"nc": nc, "n_iters": n_iters},
    )


def job_neuropress_bench_mstep(
    project_root: Path,
    output_dir: Path,
    *,
    smoke: bool = False,
    warmup: int = 3,
    iters: int = 10,
) -> VastJob:
    """M-step optimization benchmark on a remote RTX 4090.

    Measures three independent speedups introduced by the M-step fixes:
    - Fix 3: ``unpack_indices`` CUDA kernel vs Python bit-loop
    - Fix 2: ``.zero_()`` on pre-allocated tensors vs ``torch.zeros`` per call
    - ``index_add_`` accumulation throughput at NC=32768 scale

    Args:
        project_root: Local path to the neuropress repository root.
        output_dir: Local directory for artifacts.
        smoke: Run small NC=256 config only (<5s).
        warmup: Warmup iterations per measurement (default: 3).
        iters: Timed iterations per measurement (default: 10).

    Returns:
        Configured :class:`VastJob`.
    """
    remote = "/workspace/neuropress"

    args_parts = []
    if smoke:
        args_parts.append("--smoke")

    bench_cmd = (
        "~/.local/bin/uv run python scripts/bench_mstep.py "
        + " ".join(args_parts)
        + " 2>&1 | tee /tmp/bench_mstep.log ; exit ${PIPESTATUS[0]}"
    )

    return VastJob(
        name="neuropress-bench-mstep",
        gpu=GpuSpec(
            query=(
                "gpu_name=RTX_4090 num_gpus=1 rented=False rentable=True "
                "cuda_vers>=13.0 dph<0.65"
            ),
            image=_DEFAULT_IMAGE,
            disk_gb=20,
            max_dph=0.70,
        ),
        uploads=[project_root],
        remote_workdir=remote,
        setup=_uv_setup_eval(),
        run=[bench_cmd],
        artifacts=["/tmp/bench_mstep.log"],
        output_dir=output_dir / "bench_mstep",
        meta={"smoke": smoke, "warmup": warmup, "iters": iters},
    )


# Registry: name → factory function.
# Each factory has signature (project_root, output_dir, **extra_kwargs).
_JOB_REGISTRY: dict[str, Any] = {
    "neuropress-integration": job_neuropress_integration,
    "neuropress-eval-compare": job_neuropress_eval_compare,
    "neuropress-bench": job_neuropress_bench,
    "neuropress-bench-inference": job_neuropress_bench_inference,
    "neuropress-bench-kmeans": job_neuropress_bench_kmeans,
    "neuropress-bench-assign": job_neuropress_bench_assign,
    "neuropress-bench-mstep": job_neuropress_bench_mstep,
    "neuropress-profile-mstep": job_neuropress_profile_mstep,
    "neuropress-test-save": job_neuropress_test_save,
    "neuropress-profile-layers": job_neuropress_profile_layers,
}

try:  # optional local extension module
    from .research_jobs import RESEARCH_JOBS
except ImportError:  # pragma: no cover - not present in every checkout
    pass
else:
    _JOB_REGISTRY.update(RESEARCH_JOBS)


# Project root: neuropress/cloud/vast_runner.py → ../../.. → repo root
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def _run_sweep(
    sweep_file: Path,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    """Read a YAML sweep file and run all entries as parallel cloud jobs.

    Each entry is launched as an independent ``neuropress-cloud`` subprocess
    so that jobs rent their own GPU instances and run concurrently.  Output
    from each job is streamed to a per-job ``run.log`` in its artifact
    directory.  A summary table is printed once all jobs have completed.

    The YAML file must be a list of mappings.  Required keys per entry:

    * ``model`` — HuggingFace model ID.
    * ``compression`` — key in :data:`NEUROPRESS_COMPRESSION`.
    * ``tasks`` — list of lm-eval task names.

    Optional keys mirror ``neuropress-eval compare`` flags: ``limit``,
    ``num_fewshot``, ``device``, ``save_model``, ``cache_dir``, ``name``
    (friendly label used for the output subdirectory; defaults to the
    compression key).

    Args:
        sweep_file: Path to the YAML sweep configuration file.
        args: Parsed CLI namespace carrying cloud / infra settings shared
            across all jobs (api_key, cloud, ssh_key, datacenter, …).
        parser: Argument parser used for structured error reporting.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        parser.error(
            "PyYAML is required for --sweep.\n"
            "Install it with: uv sync --extra cloud   or   pip install pyyaml"
        )

    with sweep_file.open() as fh:
        raw = yaml.safe_load(fh)

    # Two supported formats:
    #   Plain list (original):  [entry, entry, ...]
    #   Dict with defaults:     {defaults: {...}, entries: [entry, ...]}
    defaults: dict[str, Any] = {}
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict) and "entries" in raw:
        defaults = raw.get("defaults") or {}
        entries = raw["entries"]
    else:
        parser.error(
            f"{sweep_file}: expected a YAML list or a mapping with an 'entries' key, "
            f"got {type(raw).__name__}. "
            "Dict-style sweeps use: defaults: {{...}} / entries: [...]"
        )

    if not isinstance(entries, list) or not entries:
        parser.error(f"{sweep_file}: 'entries' must be a non-empty list")

    # Merge defaults into each entry; entry-level keys win over defaults.
    if defaults:
        entries = [{**defaults, **e} for e in entries]

    console = Console()
    console.print(
        f"\n[bold]Sweep:[/bold] {len(entries)} job(s) from [cyan]{sweep_file}[/cyan]\n"
    )

    # Pre-flight: verify vastai-sdk is importable before spawning anything,
    # so the user gets a clear error immediately rather than discovering it
    # from every log file after all subprocesses have been launched.
    if args.cloud == "vastai":
        try:
            import vastai_sdk as _vastai_sdk  # noqa: F401
        except ImportError:
            parser.error(
                "vastai-sdk is not installed in the current environment.\n"
                "Run: uv sync --extra cloud   or   pip install vastai-sdk"
            )

    # Resolve the interpreter: prefer the venv Python (via $VIRTUAL_ENV) so
    # that the subprocess uses the correct site-packages even when
    # sys.executable is the uv-managed base Python rather than the venv
    # wrapper.  Falls back to sys.executable for non-venv installs.
    _venv_dir = os.environ.get("VIRTUAL_ENV")
    _venv_py = Path(_venv_dir) / "bin" / "python" if _venv_dir else None
    _interp = str(_venv_py) if _venv_py and _venv_py.exists() else sys.executable
    base_argv: list[str] = [
        _interp,
        "-c",
        "from neuropress.cloud.vast_runner import main; main()",
    ]
    base_argv += ["--cloud", args.cloud]
    if args.api_key:
        base_argv += ["--api-key", args.api_key]
    if args.ssh_key:
        base_argv += ["--ssh-key", str(args.ssh_key)]
    if args.keep:
        base_argv += ["--keep"]
    if not args.datacenter:
        base_argv += ["--no-datacenter"]
    if args.verified:
        base_argv += ["--verified"]
    if args.min_inet_down is not None:
        base_argv += ["--min-inet-down", str(args.min_inet_down)]
    if args.min_reliability is not None:
        base_argv += ["--min-reliability", str(args.min_reliability)]
    if args.location:
        base_argv += ["--location", args.location]
    if args.gpu_name is not None:
        base_argv += ["--gpu-name", args.gpu_name]
    if args.hf_token:
        base_argv += ["--hf-token", args.hf_token]
    if args.ssh_timeout is not None:
        base_argv += ["--ssh-timeout", str(args.ssh_timeout)]
    if args.ssh_retries != 3:
        base_argv += ["--ssh-retries", str(args.ssh_retries)]
    if args.dry_run:
        base_argv += ["--dry-run"]
    base_argv += ["--project-root", str(args.project_root)]

    # Validate entries, compute output dirs, and build per-entry argv lists.
    # Tuple: (name, argv, output_dir, gpu_name_override)
    planned: list[tuple[str, list[str], Path, str | None]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            parser.error(f"Entry {i}: expected a mapping, got {entry!r}")

        job_type = str(entry.get("job", "eval-compare"))
        if job_type not in ("eval-compare", "bench-inference"):
            parser.error(
                f"Entry {i}: unknown job type {job_type!r}. "
                "Valid: 'eval-compare', 'bench-inference'."
            )

        model = entry.get("model")
        if not model:
            parser.error(f"Entry {i}: missing required field 'model'")

        entry_name = str(
            entry.get("name")
            or entry.get("compression")
            or entry.get("checkpoint")
            or job_type
        )
        date_prefix = datetime.date.today().strftime("%Y_%m")
        slug = f"{date_prefix}_{entry_name}"
        if args.output is not None:
            output = args.output / entry_name
        else:
            data_dir = os.environ.get("NEUROPRESS_DATA_DIR")
            base = Path(data_dir) if data_dir else Path("./data")
            subdir = "bench" if job_type == "bench-inference" else "eval"
            output = base / subdir / slug

        entry_argv = [
            *list(base_argv),
            "--job",
            f"neuropress-{job_type}",
            "--output",
            str(output),
            "--model",
            model,
        ]

        if job_type == "eval-compare":
            # --- eval-compare branch (original behaviour) ----------------
            compression = entry.get("compression")
            if not compression:
                parser.error(f"Entry {i}: eval-compare requires 'compression'")

            tasks_raw = entry.get("tasks", [])
            if isinstance(tasks_raw, str):
                tasks_raw = [tasks_raw]
            tasks_raw = [str(t) for t in tasks_raw]

            from neuropress.eval.harness import normalize_tasks, validate_tasks

            tasks = normalize_tasks(tasks_raw)
            if not tasks:
                parser.error(f"Entry {i}: missing required field 'tasks'")
            unknown = validate_tasks(tasks)
            if unknown:
                parser.error(
                    f"Entry {i}: unknown lm-eval task(s): {', '.join(unknown)}"
                )

            # Skip when report already exists (crash recovery).
            # Set force: true in the sweep entry to override (e.g. to re-run
            # and save a model when eval already exists but model was lost).
            report_path = output / "eval" / "eval_report.json"
            if report_path.exists() and not entry.get("force"):
                console.print(
                    f"  [dim]{i + 1:2d}.[/dim] [yellow]skip[/yellow] "
                    f"[cyan]{entry_name}[/cyan]  "
                    f"[dim](report already exists: {report_path})[/dim]"
                )
                continue

            entry_argv += ["--compression", compression, "--tasks", *tasks]
            if entry.get("limit") is not None:
                entry_argv += ["--limit", str(entry["limit"])]
            if entry.get("num_fewshot") is not None:
                entry_argv += ["--num-fewshot", str(entry["num_fewshot"])]
            if entry.get("device") is not None:
                entry_argv += ["--device", str(entry["device"])]
            if entry.get("save_model"):
                entry_argv += ["--save-model"]
            if entry.get("skip_baseline"):
                entry_argv += ["--skip-baseline"]
            if entry.get("batch_size") is not None:
                entry_argv += ["--batch-size", str(entry["batch_size"])]
            cache_dir_val = entry.get("cache_dir")
            if cache_dir_val is False or (
                cache_dir_val is not None
                and str(cache_dir_val).lower() in ("false", "none", "off")
            ):
                entry_argv += ["--no-cache-dir"]
            elif cache_dir_val is not None:
                # Fail here, not on the first sync two hours into the run.
                problem = check_local_cache_dir(cache_dir_val)
                if problem is not None:
                    raise ValueError(
                        f"sweep entry {entry.get('name', '?')!r}: {problem}"
                    )
                entry_argv += ["--cache-dir", str(cache_dir_val)]
            # else: not set → auto-derive in job factory

        else:
            # --- bench-inference branch (new) ----------------------------
            checkpoint = entry.get("checkpoint")
            compression = entry.get("compression")
            if bool(checkpoint) == bool(compression):
                parser.error(
                    f"Entry {i}: bench-inference requires exactly one of "
                    "'checkpoint' or 'compression'."
                )

            # Skip when bench result already exists (crash recovery).
            bench_path = output / "bench_inference" / "bench_inference.json"
            if bench_path.exists():
                console.print(
                    f"  [dim]{i + 1:2d}.[/dim] [yellow]skip[/yellow] "
                    f"[cyan]{entry_name}[/cyan]  "
                    f"[dim](result already exists: {bench_path})[/dim]"
                )
                continue

            if checkpoint:
                entry_argv += ["--bench-checkpoint", checkpoint]
            else:
                entry_argv += ["--compression", compression]

            if entry.get("pp") is not None:
                pp_vals = entry["pp"]
                if isinstance(pp_vals, int):
                    pp_vals = [pp_vals]
                entry_argv += ["--bench-pp"] + [str(p) for p in pp_vals]
            if entry.get("tg") is not None:
                entry_argv += ["--bench-tg", str(entry["tg"])]
            if entry.get("warmup") is not None:
                entry_argv += ["--bench-warmup", str(entry["warmup"])]
            if entry.get("iters") is not None:
                entry_argv += ["--bench-iters", str(entry["iters"])]
            if entry.get("device") is not None:
                entry_argv += ["--device", str(entry["device"])]

            # Handle lowering key: inline dict → write to output dir; string → pass directly.
            lowering = entry.get("lowering")
            if lowering is not None:
                if isinstance(lowering, dict):
                    from neuropress.kernel_lowering import LoweringConfig

                    lc = LoweringConfig.from_dict(lowering)
                    lc_path = output / "lowering.yaml"
                    lc_path.parent.mkdir(parents=True, exist_ok=True)
                    lc.save(lc_path)
                    entry_argv += ["--bench-lowering-config", str(lc_path)]
                else:
                    entry_argv += ["--bench-lowering-config", str(lowering)]

        entry_gpu_name: str | None = None
        if entry.get("gpu") is not None:
            entry_gpu_name = str(entry["gpu"])
            entry_argv += ["--gpu-name", entry_gpu_name]
        if entry.get("ssh_timeout") is not None:
            entry_argv += ["--ssh-timeout", str(entry["ssh_timeout"])]
        if entry.get("ssh_retries") is not None:
            entry_argv += ["--ssh-retries", str(entry["ssh_retries"])]

        planned.append((entry_name, entry_argv, output, entry_gpu_name))
        console.print(
            f"  [dim]{i + 1:2d}.[/dim] [bold cyan]{job_type}[/bold cyan] "
            f"[cyan]{entry_name}[/cyan]\n"
            f"       output → {output}"
        )

    console.print()

    if not planned:
        console.print("[green]All entries already completed — nothing to run.[/green]")
        return

    # Track all child processes so we can terminate them on parent exit.
    procs: list[tuple[str, subprocess.Popen[bytes], Path | None]] = []
    # Pre-allocated instance IDs (vast.ai only) for cleanup on failure.
    preallocated_ids: list[int] = []
    vast_client: Any = None

    def _terminate_children(sig: int | None = None, frame: Any = None) -> None:
        """Send SIGTERM to all children and destroy orphaned instances.

        Registered with atexit and SIGTERM/SIGINT so that child processes
        (and their vast.ai instances) are cleaned up even when the parent
        sweep process is killed.
        """
        for _name, p, _log in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        # Give children a moment to run their atexit handlers.
        for _name, p, _log in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        # Destroy any pre-allocated instances that children may not have
        # claimed or whose subprocess exited before cleanup ran.
        if vast_client is not None:
            for iid in preallocated_ids:
                try:
                    vast_client.destroy_instance(id=iid)
                except Exception:
                    pass
        if sig is not None:
            sys.exit(1)

    atexit.register(_terminate_children)
    prev_sigterm = signal.signal(signal.SIGTERM, _terminate_children)
    prev_sigint = signal.signal(signal.SIGINT, _terminate_children)

    try:
        if args.dry_run:
            # Dry-run: each subprocess will search for GPU offers and exit.
            # Log to terminal directly (no output files created).
            procs.extend(
                [
                    (name, subprocess.Popen(argv), None)  # type: ignore[arg-type]
                    for name, argv, _, _gpu in planned
                ]
            )
        elif args.cloud == "vastai" and not args.dry_run:
            # Rent instances *sequentially* in the parent process to avoid
            # the race where parallel subprocesses all rent the same offer
            # and end up sharing a single instance ID.  Each subprocess is
            # launched as soon as its instance is SSH-ready so already-up
            # instances start working immediately.
            #
            # base_gpu_spec: shared infra constraints; gpu_name is applied
            # per-entry so each job rents the GPU type it actually requires.
            base_gpu_spec = GpuSpec(
                query=(
                    "num_gpus=1 rented=False rentable=True "
                    "cuda_vers>=13.0 dph<0.75 gpu_ram>=20"
                ),
                image=_DEFAULT_IMAGE,
                disk_gb=60,
                max_dph=0.80,
                datacenter=args.datacenter,
                verified=args.verified,
                min_inet_down_mbps=(
                    args.min_inet_down if args.min_inet_down is not None else 500
                ),
                min_reliability=(
                    args.min_reliability if args.min_reliability is not None else 0.9
                ),
                location=args.location,
            )
            console.rule(f"[bold]Renting {len(planned)} instance(s)[/bold]")
            for i, (name, argv, output, gpu_name) in enumerate(planned):
                # Apply the GPU model override so the parent rents the same GPU
                # type the subprocess will use — not just any GPU satisfying the
                # base spec.  The parent does the renting here, so --gpu-name
                # must be honoured too, not only the per-entry `gpu:` field.
                effective_gpu_name = resolve_gpu_name(gpu_name, args.gpu_name)
                entry_gpu_spec = (
                    dc_replace(base_gpu_spec, gpu_name=effective_gpu_name)
                    if effective_gpu_name is not None
                    else base_gpu_spec
                )
                console.print(
                    f"\n  [dim]{i + 1}/{len(planned)}[/dim] "
                    f"Renting instance for [cyan]{name}[/cyan]…"
                    + (f"  [dim]({gpu_name})[/dim]" if gpu_name else "")
                )
                runner = VastRunner(
                    api_key=args.api_key or "",
                    keep=True,  # subprocess manages lifecycle
                    ssh_timeout=args.ssh_timeout or _SSH_TIMEOUT_S,
                    ssh_retries=args.ssh_retries,
                )
                dummy_job = VastJob(
                    name=f"rent-{name}",
                    gpu=entry_gpu_spec,
                    uploads=[],
                    remote_workdir="/workspace",
                    setup=[],
                    run=[],
                    artifacts=[],
                    output_dir=Path("/dev/null"),
                )
                # Honour --instance-id: attach to an existing (e.g. preserved)
                # instance instead of renting.  Ignoring it here previously
                # turned --resume into a fresh rent attempt.
                _override = sweep_instance_override(len(planned), args.instance_id)
                if _override is not None:
                    console.print(
                        f"  [dim]attaching to existing instance {_override} "
                        f"(--instance-id)[/dim]"
                    )
                    iid = _override
                else:
                    runner._connect(dummy_job)
                    iid = runner._instance_id
                assert iid is not None
                # Detach so VastRunner's atexit doesn't destroy it.
                runner._instance_id = None
                preallocated_ids.append(iid)
                vast_client = runner._client

                # Launch subprocess immediately with this instance.
                output.mkdir(parents=True, exist_ok=True)
                log_path = output / "run.log"
                log_fh = log_path.open("w")
                entry_argv = [*argv, "--instance-id", str(iid)]
                p = subprocess.Popen(
                    entry_argv,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
                procs.append((name, p, log_path))
                console.print(
                    f"  [green]▶[/green] [bold]{name}[/bold]  "
                    f"[dim](pid {p.pid}  instance {iid}  "
                    f"log → {log_path})[/dim]"
                )
            console.print(
                f"\n[dim]All {len(procs)} job(s) running — "
                "tail each log file to follow progress.[/dim]\n"
            )
        else:
            # Non-vastai live run: each subprocess logs to its artifact dir.
            for name, argv, output, _gpu in planned:
                output.mkdir(parents=True, exist_ok=True)
                log_path = output / "run.log"
                log_fh = log_path.open("w")
                p = subprocess.Popen(argv, stdout=log_fh, stderr=subprocess.STDOUT)
                procs.append((name, p, log_path))
                console.print(
                    f"  [green]▶[/green] [bold]{name}[/bold]  "
                    f"[dim](pid {p.pid}  log → {log_path})[/dim]"
                )
            console.print(
                f"\n[dim]All {len(procs)} job(s) running — "
                "tail each log file to follow progress.[/dim]\n"
            )

        results: list[tuple[str, bool, Path | None]] = []
        for name, p, log_path in procs:
            rc = p.wait()
            results.append((name, rc == 0, log_path))

        # All children have exited — they each ran their own cleanup.
        # Clear preallocated_ids so the atexit handler is a no-op.
        preallocated_ids.clear()

        console.rule("[bold]Sweep Summary[/bold]")
        all_ok = True
        for name, success, log_path in results:
            icon = "[green]✓[/green]" if success else "[red]✗[/red]"
            suffix = f"  [dim]{log_path}[/dim]" if log_path else ""
            console.print(f"  {icon} {name}{suffix}")
            all_ok = all_ok and success
        console.print()
        sys.exit(0 if all_ok else 1)

    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)
        atexit.unregister(_terminate_children)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def sweep_instance_override(
    n_entries: int,
    cli_instance_id: int | None,
) -> int | None:
    """Return an existing instance to attach to in sweep mode, or ``None``.

    ``_run_sweep`` rents an instance per entry, so ``--instance-id`` was
    previously ignored there: a ``--resume`` against a preserved instance became
    a fresh rent attempt, which then failed on "No GPU offers match query" while
    the preserved instance kept billing.

    A single instance cannot serve several entries, so that combination raises
    rather than silently renting - failing loudly is cheaper than a surprise
    rental.

    Args:
        n_entries: Number of sweep entries planned to run.
        cli_instance_id: Value of ``--instance-id``, if given.

    Returns:
        The instance id to attach to, or ``None`` to rent normally.

    Raises:
        ValueError: If an instance id is given for a multi-entry sweep.
    """
    if cli_instance_id is None:
        return None
    if n_entries != 1:
        raise ValueError(
            f"--instance-id can only be used with a single-entry sweep; "
            f"{n_entries} entries are planned. Filter with --job NAME."
        )
    return cli_instance_id


def resolve_gpu_name(
    entry_gpu_name: str | None,
    cli_gpu_name: str | None,
) -> str | None:
    """Resolve which GPU model a sweep entry should rent.

    In sweep mode the parent process rents the instance, so whatever this
    returns is the GPU the job actually runs on.  A per-entry ``gpu:`` field is
    more specific than the sweep-wide ``--gpu-name`` flag and therefore wins;
    the flag applies to entries that do not name a GPU themselves.

    Args:
        entry_gpu_name: The sweep entry's ``gpu`` field, if any.
        cli_gpu_name: The ``--gpu-name`` command-line value, if any.

    Returns:
        The GPU model to rent, or ``None`` to accept any GPU matching the base
        query.
    """
    return entry_gpu_name if entry_gpu_name is not None else cli_gpu_name


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neuropress-cloud",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cloud",
        choices=["vastai", "own"],
        default="vastai",
        metavar="BACKEND",
        help=(
            "Cloud backend to use.  "
            "'vastai': rent a GPU via vast.ai (default).  "
            "'own': connect via SSH to a machine you already have "
            "(home PC, cluster, …) — nothing is rented or destroyed."
        ),
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("VAST_API_KEY"),
        metavar="KEY",
        help="vast.ai API key (default: $VAST_API_KEY).  Required for --cloud vastai.",
    )
    p.add_argument(
        "--ssh-key",
        type=Path,
        default=None,
        metavar="PATH",
        help="SSH private key path (default: ~/.ssh/id_rsa).",
    )
    p.add_argument(
        "--job",
        metavar="NAME",
        help="Job to run.  Use --list-jobs to see available names.",
    )
    p.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print available job names and exit.",
    )
    p.add_argument(
        "--sweep",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "YAML file listing multiple jobs to run in parallel.  "
            "Each entry is a mapping with keys 'model', 'compression', "
            "'tasks', and optional 'limit', 'num_fewshot', 'device', "
            "'save_model', 'name'.  Cloud/infra flags are shared across "
            "all entries.  Requires PyYAML (included in the cloud extra)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Search for matching GPU offers without renting anything.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Local directory for downloaded artifacts. "
            "Default: $NEUROPRESS_DATA_DIR/eval/YYYY_mm_{compression} when "
            "$NEUROPRESS_DATA_DIR is set, otherwise "
            "./data/eval/YYYY_mm_{compression}.  The compression key is used "
            "instead of the job name so that two runs with different --compression "
            "values produce separate output directories."
        ),
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=_DEFAULT_PROJECT_ROOT,
        metavar="DIR",
        help="Local neuropress project root (default: repo root).",
    )
    # GPU host filtering (applied to any job)
    p.add_argument(
        "--datacenter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict to datacenter/colocation hosts only (default: on).  "
            "Better uptime and physical security than residential hosts; "
            "the host is a business rather than an individual.  "
            "Use --no-datacenter to allow private/residential hosts."
        ),
    )
    p.add_argument(
        "--verified",
        action="store_true",
        help=(
            "Restrict to identity-verified hosts only (vast.ai KYC check).  "
            "Can be combined with --datacenter."
        ),
    )
    p.add_argument(
        "--keep",
        action="store_true",
        help=(
            "Do not destroy the instance after the job.  "
            "Prints the instance ID so you can reuse it with --instance-id."
        ),
    )
    p.add_argument(
        "--instance-id",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Attach to an existing instance instead of renting a new one.  "
            "Skips the Rent and Wait-for-SSH phases (Docker image already pulled)."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Re-attach to a running job after a local disconnection.  "
            "Skips upload and setup; streams output from the live tmux session.  "
            "Requires --instance-id, or a valid instance_id.txt in the output "
            "directory (written automatically at job start).  "
            "Example: neuropress-cloud --cloud vastai --resume --output ./my_run"
        ),
    )
    p.add_argument(
        "--ssh-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            f"Maximum wait for SSH readiness per attempt "
            f"(default: {_SSH_TIMEOUT_S}s).  "
            "Increase when using large Docker images on slow hosts."
        ),
    )
    p.add_argument(
        "--ssh-retries",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Maximum number of rent+SSH attempts before giving up (default: 3).  "
            "On SSH timeout, the stuck instance is destroyed and a new one is "
            "rented.  Set to 1 to disable retries."
        ),
    )
    p.add_argument(
        "--min-inet-down",
        type=int,
        default=None,
        metavar="MBPS",
        help=(
            "Minimum host download bandwidth in Mbps (default: 500).  "
            "Avoids slow hosts that burn billing time pulling the Docker image."
        ),
    )
    p.add_argument(
        "--min-reliability",
        type=float,
        default=None,
        metavar="SCORE",
        help="Minimum vast.ai reliability score 0-1 (default: 0.9).",
    )
    p.add_argument(
        "--location",
        default="eu",
        metavar="REGION",
        help=(
            "Restrict to offers whose geolocation matches REGION (default: eu).  "
            "'eu' filters to European countries (EU member states + IS, NO, CH, GB).  "
            "'any' disables the location filter entirely.  "
            "Any other value is matched as a case-insensitive substring of the "
            "offer's geolocation field (e.g. 'de', 'germany', 'nl')."
        ),
    )
    p.add_argument(
        "--gpu-name",
        default=None,
        metavar="MODEL",
        help=(
            "Override the GPU model filter (e.g. RTX_4090, RTX_3090, A100_SXM4_80GB).  "
            "Replaces the gpu_name=X token in the job's default query string.  "
            "Omit to use the job's built-in default."
        ),
    )
    p.add_argument(
        "--hf-token",
        default=(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ),
        metavar="TOKEN",
        help=(
            "HuggingFace API token for gated models "
            "(default: $HF_TOKEN or $HUGGING_FACE_HUB_TOKEN).  "
            "Required for models like Gemma that need explicit access approval."
        ),
    )
    # Extra arguments forwarded to job factories that support them
    p.add_argument("--model", default=None, metavar="HF_ID")
    p.add_argument("--compression", default=None, metavar="METHOD")
    p.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    p.add_argument("--limit", type=float, default=None, metavar="N")
    p.add_argument("--num-fewshot", type=int, default=None, metavar="K")
    p.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help=(
            "Save the quantized model as a safetensors checkpoint and "
            "download it with the other artifacts after the job completes."
        ),
    )
    p.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help=(
            "Skip baseline (float model) evaluation.  The report will "
            "contain null for baseline.  Useful when baseline data already "
            "exists from a prior sweep."
        ),
    )
    p.add_argument(
        "--batch-size",
        "-b",
        default=None,
        metavar="B",
        help=(
            "Batch size for lm-eval.  'auto' lets lm-eval pick the largest "
            "batch that fits in GPU memory.  Default: 'auto' (via job factory)."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Local directory for quantization stage caching (hessians, centroids, "
            "assignments).  Defaults to $NEUROPRESS_DATA_DIR/cache/{model} when "
            "$NEUROPRESS_DATA_DIR is set.  The directory is uploaded to the remote "
            "before the run and synced back during and after."
        ),
    )
    p.add_argument(
        "--no-cache-dir",
        action="store_true",
        default=False,
        help=(
            "Disable quantization stage caching even when $NEUROPRESS_DATA_DIR "
            "is set.  Overrides the auto-derived cache_dir default."
        ),
    )
    p.add_argument(
        "--device",
        "-d",
        default=None,
        metavar="DEVICE",
        help=(
            "PyTorch device for eval jobs (e.g. cpu, cuda, cuda:1). "
            "Default: cuda (eval jobs run on GPU instances by default)."
        ),
    )
    # -- Bench-specific arguments -----------------------------------------------
    bench = p.add_argument_group(
        "Benchmark (--job neuropress-bench)",
        "Options forwarded to neuropress-bench on the remote instance.",
    )
    bench.add_argument(
        "--bench-module",
        choices=["linear", "vq", "embedding", "all"],
        default=None,
        metavar="TYPE",
        help="Module type to benchmark (default: all).",
    )
    bench.add_argument(
        "--bench-shapes",
        choices=["small", "medium", "large", "xlarge", "all"],
        default=None,
        metavar="PRESET",
        help="Shape preset (default: all).",
    )
    bench.add_argument(
        "--bench-batch-sizes",
        default=None,
        metavar="LIST",
        help="Comma-separated batch sizes (default: 1,16,256).",
    )
    bench.add_argument(
        "--bench-warmup",
        type=int,
        default=None,
        metavar="N",
        help="Warmup iterations (default: 5 for synthetic, 3 for inference).",
    )
    bench.add_argument(
        "--bench-iters",
        type=int,
        default=None,
        metavar="N",
        help="Timed iterations (default: 20 for synthetic, 10 for inference).",
    )
    bench.add_argument(
        "--bench-pp",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Prompt lengths for neuropress-bench-inference (default: 512 1024).",
    )
    bench.add_argument(
        "--bench-tg",
        type=int,
        default=None,
        metavar="N",
        help="Generation tokens for neuropress-bench-inference (default: 128).",
    )
    bench.add_argument(
        "--bench-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Local path to a pre-saved checkpoint for neuropress-bench-inference. "
            "Uploaded alongside the repo — skips re-quantization on the remote. "
            "Mutually exclusive with --compression."
        ),
    )
    bench.add_argument(
        "--bench-lowering-config",
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML kernel-lowering config for neuropress-bench-inference "
            "(see configs/lowering/). Controls vq_backend, linear_backend, "
            "embedding_backend. Omit for all-auto defaults."
        ),
    )
    bench.add_argument(
        "--bench-n-values",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Vector counts for neuropress-bench-kmeans (default: 10k 50k 100k 500k 1M).",
    )
    bench.add_argument(
        "--bench-k-values",
        nargs="+",
        type=int,
        default=None,
        metavar="K",
        help="Codebook sizes for neuropress-bench-kmeans (default: 256 1024 4096 16384 65536).",
    )
    bench.add_argument(
        "--bench-d-values",
        nargs="+",
        type=int,
        default=None,
        metavar="D",
        help="Vector lengths for neuropress-bench-kmeans (default: 4).",
    )
    bench.add_argument(
        "--bench-backends",
        nargs="+",
        default=None,
        metavar="BACKEND",
        choices=["torch", "cuda"],
        help="Backends to compare for neuropress-bench-kmeans (default: torch cuda).",
    )
    bench.add_argument(
        "--bench-smoke",
        action="store_true",
        default=False,
        help="Run tiny smoke config only for neuropress-bench-assign (R=64 G=8 NC=256, <5s).",
    )
    bench.add_argument(
        "--bench-seq-len",
        type=int,
        default=None,
        metavar="N",
        help="Sequence length for neuropress-profile-layers (1=TG, 512=PP, default: 1).",
    )
    bench.add_argument(
        "--attn-implementation",
        default=None,
        metavar="IMPL",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Override HF attention implementation for profile-layers job.",
    )
    # -- Direct SSH (--cloud own) ---------------------------------------------
    direct = p.add_argument_group(
        "Direct SSH (--cloud own)",
        "Used when --cloud own is selected.  Connect to a machine you already "
        "own — nothing is rented or destroyed.  VAST_API_KEY is not required.",
    )
    direct.add_argument(
        "--ssh-host",
        default=None,
        metavar="HOST",
        help="Hostname or IP of the target machine.",
    )
    direct.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        metavar="PORT",
        help="SSH port (default: 22).",
    )
    direct.add_argument(
        "--ssh-user",
        default="root",
        metavar="USER",
        help="SSH login user (default: root).",
    )
    return p


def _resolve_instance_id(
    runner: VastRunner,
    instance_id: int,
    console: Console,
) -> int | None:
    """Validate *instance_id* and return it, or let the user pick one via TUI.

    When *instance_id* is not found among the account's instances, this
    function lists all available instances and — if stdout is a TTY — shows
    :class:`~neuropress.cloud.tui.InstanceSelectApp` so the user can pick.
    In non-TTY mode a plain table is printed and ``None`` is returned.

    Args:
        runner: Authenticated :class:`VastRunner` used to query instances.
        instance_id: The value supplied via ``--instance-id``.
        console: Rich console for non-TUI output.

    Returns:
        The resolved instance ID, or ``None`` if the user cancelled / no
        instances are available.
    """
    all_instances = runner.list_instances_info()
    valid_ids = {int(i["id"]) for i in all_instances}
    if instance_id in valid_ids:
        return instance_id

    console.print(
        f"[yellow]Warning:[/yellow] Instance [bold]{instance_id}[/bold] "
        "not found on this account."
    )
    if not all_instances:
        console.print("[red]No instances found on this account.[/red]")
        return None

    if sys.stdout.isatty():
        try:
            from neuropress.cloud.tui import InstanceSelectApp
        except ImportError:
            pass
        else:
            selected: int | None = InstanceSelectApp(all_instances).run()
            if selected is None:
                console.print("No instance selected.")
            return selected

    # Non-TTY fallback: print a plain list.
    table = Table(title="Available instances", show_lines=False)
    table.add_column("ID", style="bold cyan")
    table.add_column("Status")
    table.add_column("GPU")
    table.add_column("#")
    table.add_column("$/hr")
    table.add_column("Location")
    for inst in all_instances:
        dph = inst.get("dph_total")
        table.add_row(
            str(inst.get("id", "?")),
            str(inst.get("actual_status", "?")),
            str(inst.get("gpu_name", "?")),
            str(inst.get("num_gpus", 1)),
            f"${float(dph):.3f}" if dph is not None else "?",
            str(inst.get("geolocation") or "?"),
        )
    console.print(table)
    console.print("Re-run with a valid [bold]--instance-id[/bold].")
    return None


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``neuropress-cloud`` CLI.

    Args:
        argv: Argument list (defaults to :data:`sys.argv[1:]`).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_jobs:
        console = Console()
        console.print("\n[bold]Available jobs:[/bold]")
        for name in sorted(_JOB_REGISTRY):
            console.print(f"  [cyan]•[/cyan] {name}")
        console.print()
        return

    if args.sweep:
        _run_sweep(args.sweep, args, parser)
        return  # _run_sweep calls sys.exit internally

    if not args.job:
        parser.error("specify --job NAME, --list-jobs, or --sweep FILE")

    if args.job not in _JOB_REGISTRY:
        parser.error(f"unknown job {args.job!r}; available: {', '.join(_JOB_REGISTRY)}")

    if args.output is None:
        date_prefix = datetime.date.today().strftime("%Y_%m")
        # Use the compression key (when set) so that two CLI runs with
        # different --compression values produce separate output directories
        # and don't overwrite each other's eval_report.json.
        label = args.compression if args.compression is not None else args.job
        slug = f"{date_prefix}_{label}"
        data_dir = os.environ.get("NEUROPRESS_DATA_DIR")
        base = Path(data_dir) if data_dir else Path("./data")
        if args.job == "neuropress-bench":
            # Bench results are hardware-specific: group by host.
            host = getattr(args, "ssh_host", None) or args.cloud
            args.output = base / "bench" / host / slug
        else:
            args.output = base / "eval" / slug

    # Normalise and validate tasks locally before touching the cloud.
    if args.tasks is not None:
        from neuropress.eval.harness import normalize_tasks, validate_tasks

        args.tasks = normalize_tasks(args.tasks)
        unknown = validate_tasks(args.tasks)
        if unknown:
            parser.error(
                f"Unknown lm-eval task(s): {', '.join(unknown)}\n"
                "Run `neuropress-eval list-tasks` or check the lm-eval docs "
                "for valid task names."
            )

    from neuropress.cloud.direct_runner import DirectRunner

    runner: SshRunner
    if args.cloud == "own":
        if not args.ssh_host:
            parser.error("--ssh-host is required when --cloud own")
        runner = DirectRunner(
            host=args.ssh_host,
            port=args.ssh_port,
            user=args.ssh_user,
            ssh_key=args.ssh_key,
        )
    else:  # vastai
        if not args.api_key:
            parser.error(
                "VAST_API_KEY environment variable or --api-key required "
                "for --cloud vastai"
            )
        resume = getattr(args, "resume", False)
        if resume and args.instance_id is None:
            # Try to read instance_id.txt from the output directory.
            id_file = args.output / "instance_id.txt"
            if id_file.exists():
                try:
                    args.instance_id = int(id_file.read_text().strip())
                    Console().print(
                        f"[dim]Auto-loaded instance ID {args.instance_id} "
                        f"from {id_file}[/dim]"
                    )
                except (ValueError, OSError):
                    pass
            if args.instance_id is None:
                parser.error(
                    "--resume requires --instance-id or a valid instance_id.txt "
                    "in the output directory."
                )
        vast_runner = VastRunner(
            api_key=args.api_key,
            ssh_key=args.ssh_key,
            keep=args.keep,
            resume=resume,
            instance_id=args.instance_id,
            ssh_timeout=args.ssh_timeout,
            ssh_retries=args.ssh_retries,
        )
        if args.instance_id is not None:
            args.instance_id = _resolve_instance_id(
                vast_runner, args.instance_id, Console()
            )
            if args.instance_id is None:
                sys.exit(1)
            vast_runner._instance_id = args.instance_id
        runner = vast_runner

    factory = _JOB_REGISTRY[args.job]
    factory_kwargs: dict[str, Any] = {}
    if args.model is not None:
        factory_kwargs["model"] = args.model
    if args.compression is not None:
        factory_kwargs["compression"] = args.compression
    if args.tasks is not None:
        factory_kwargs["tasks"] = args.tasks
    if args.limit is not None:
        factory_kwargs["limit"] = args.limit
    if args.num_fewshot is not None:
        factory_kwargs["num_fewshot"] = args.num_fewshot
    if args.device is not None:
        factory_kwargs["device"] = args.device
    if args.save_model:
        factory_kwargs["save_model"] = True
    if args.skip_baseline:
        factory_kwargs["skip_baseline"] = True
    if args.batch_size is not None:
        factory_kwargs["batch_size"] = args.batch_size
    if hasattr(args, "no_cache_dir") and args.no_cache_dir:
        factory_kwargs["cache_dir"] = False  # explicitly disable auto-derive
    elif hasattr(args, "cache_dir") and args.cache_dir is not None:
        factory_kwargs["cache_dir"] = args.cache_dir
    # Bench-specific kwargs.
    if args.bench_module is not None:
        factory_kwargs["module"] = args.bench_module
    if args.bench_shapes is not None:
        factory_kwargs["shapes"] = args.bench_shapes
    if args.bench_batch_sizes is not None:
        factory_kwargs["batch_sizes"] = args.bench_batch_sizes
    if args.bench_warmup is not None:
        factory_kwargs["warmup"] = args.bench_warmup
    if args.bench_iters is not None:
        factory_kwargs["iters"] = args.bench_iters
    if args.bench_pp is not None:
        factory_kwargs["pp"] = args.bench_pp
    if args.bench_tg is not None:
        factory_kwargs["tg"] = args.bench_tg
    if args.bench_checkpoint is not None:
        factory_kwargs["checkpoint"] = args.bench_checkpoint
    if (
        hasattr(args, "bench_lowering_config")
        and args.bench_lowering_config is not None
    ):
        factory_kwargs["lowering_config"] = args.bench_lowering_config
    if hasattr(args, "bench_n_values") and args.bench_n_values is not None:
        factory_kwargs["n_values"] = args.bench_n_values
    if hasattr(args, "bench_k_values") and args.bench_k_values is not None:
        factory_kwargs["k_values"] = args.bench_k_values
    if hasattr(args, "bench_d_values") and args.bench_d_values is not None:
        factory_kwargs["d_values"] = args.bench_d_values
    if hasattr(args, "bench_backends") and args.bench_backends is not None:
        factory_kwargs["backends"] = args.bench_backends
    if hasattr(args, "bench_seq_len") and args.bench_seq_len is not None:
        factory_kwargs["seq_len"] = args.bench_seq_len
    if hasattr(args, "bench_smoke") and args.bench_smoke:
        factory_kwargs["smoke"] = True
    if hasattr(args, "attn_implementation") and args.attn_implementation is not None:
        factory_kwargs["attn_implementation"] = args.attn_implementation

    try:
        job = factory(args.project_root, args.output, **factory_kwargs)
    except TypeError as exc:
        parser.error(f"Job {args.job!r} is missing required arguments: {exc}")

    # Apply host-filtering flags universally — patch GpuSpec after factory runs.
    gpu_overrides: dict[str, Any] = {}
    gpu_overrides["datacenter"] = args.datacenter
    if args.verified:
        gpu_overrides["verified"] = True
    if args.min_inet_down is not None:
        gpu_overrides["min_inet_down_mbps"] = args.min_inet_down
    if args.min_reliability is not None:
        gpu_overrides["min_reliability"] = args.min_reliability
    loc = args.location
    gpu_overrides["location"] = None if loc and loc.lower() == "any" else loc
    if args.gpu_name is not None:
        gpu_overrides["gpu_name"] = args.gpu_name
    if gpu_overrides:
        job = dc_replace(job, gpu=dc_replace(job.gpu, **gpu_overrides))

    if args.hf_token:
        job = dc_replace(job, env_vars={**job.env_vars, "HF_TOKEN": args.hf_token})

    if args.dry_run:
        runner.dry_run(job)
        return

    success = runner.run(job)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
