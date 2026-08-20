"""Textual TUI for neuropress cloud job execution.

Provides :class:`JobApp`, a full-screen terminal application that shows the
experiment name and current phase at the top, a scrollable log in the body,
and a [Config] button (or ``c`` key) that opens a modal with the full job
configuration.

Also provides :class:`InstanceSelectApp`, a pre-job instance picker shown when
``--instance-id`` is missing or invalid.

Usage from :class:`~neuropress.cloud.runner.SshRunner`::

    from neuropress.cloud.tui import JobApp

    app = JobApp(job.name, config_dict, run_fn)
    success: bool = app.run()
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from typing import Any, ClassVar

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, RichLog, Static, TextArea


def _copy_to_clipboard(text: str) -> bool:
    """Attempt to copy *text* to the system clipboard.

    Tries ``xclip``, ``xsel``, and ``wl-copy`` in order.  Returns ``True``
    if one of them succeeds, ``False`` if none are available.

    Args:
        text: Plain-text content to copy.

    Returns:
        ``True`` on success, ``False`` when no clipboard tool is found.
    """
    for cmd in (
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["wl-copy"],
    ):
        try:
            subprocess.run(
                cmd,
                input=text.encode(),
                check=True,
                capture_output=True,
                timeout=5,
            )
            return True
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            continue
    return False


class InstanceSelectApp(App[int | None]):
    """Pre-job TUI to pick a running vast.ai instance from a table.

    Shown when ``--instance-id`` is absent or points to an unknown instance.
    Returns the selected integer instance ID, or ``None`` if the user cancels.

    Args:
        instances: List of instance dicts from the vast.ai API
            (must contain at least ``"id"`` and ``"actual_status"``).
    """

    CSS = """
    Screen {
        align: center middle;
    }
    #outer {
        width: 92%;
        height: 80%;
        border: round $primary;
        background: $surface;
    }
    #sel-title {
        text-align: center;
        text-style: bold;
        padding: 1 2;
        border-bottom: solid $primary-darken-1;
    }
    #sel-table {
        height: 1fr;
    }
    #sel-hint {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "Cancel"),
        Binding("q", "cancel", "Cancel"),
    ]

    def __init__(self, instances: list[dict[str, Any]]) -> None:
        super().__init__()
        self._instances = instances

    def compose(self) -> ComposeResult:
        """Compose the instance-picker layout."""
        with Vertical(id="outer"):
            yield Static(
                "Select an instance  (no valid --instance-id was given)",
                id="sel-title",
                markup=False,
            )
            yield DataTable(id="sel-table", cursor_type="row")
            yield Static(
                "↑ ↓  navigate    Enter  select    Esc  cancel",
                id="sel-hint",
                markup=False,
            )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the DataTable with running instances."""
        table = self.query_one("#sel-table", DataTable)
        table.add_columns("ID", "Status", "GPU", "#GPUs", "$/hr", "Location")
        for inst in self._instances:
            iid = int(inst.get("id", 0))
            status = str(inst.get("actual_status", "?"))
            gpu = str(inst.get("gpu_name", "?"))
            num_gpus = str(inst.get("num_gpus", 1))
            dph = inst.get("dph_total")
            dph_str = f"${float(dph):.3f}" if dph is not None else "?"
            geo = str(inst.get("geolocation") or "?")
            table.add_row(str(iid), status, gpu, num_gpus, dph_str, geo, key=str(iid))
        if self._instances:
            table.focus()

    @on(DataTable.RowSelected, "#sel-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        key_val = event.row_key.value
        if key_val is not None:
            self.exit(int(key_val))

    def action_cancel(self) -> None:
        """Exit without selecting an instance."""
        self.exit(None)


class ConfigModal(ModalScreen[None]):
    """Modal dialog showing the full job configuration.

    Opened by pressing ``c`` or clicking [Config].  Dismissed by pressing
    ``ESC`` or clicking *Close*.

    Args:
        config: Ordered dict of configuration items to display.
    """

    BINDINGS: ClassVar = [("escape", "dismiss", "Close")]

    DEFAULT_CSS = """
    ConfigModal {
        align: center middle;
    }
    ConfigModal > Vertical {
        width: 70;
        max-width: 90%;
        max-height: 80%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #modal-title {
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
        border-bottom: solid $primary-darken-1;
        margin-bottom: 1;
    }
    #modal-scroll {
        height: 1fr;
    }
    #modal-close-row {
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._config = config

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Vertical():
            yield Static("Configuration", id="modal-title")
            with ScrollableContainer(id="modal-scroll"):
                yield Static(Text.from_markup(self._build_markup()))
            with Horizontal(id="modal-close-row"):
                yield Button("Close", variant="primary", id="btn-close")

    def _build_markup(self) -> str:
        lines: list[str] = []
        for key, value in self._config.items():
            k = escape(str(key))
            if isinstance(value, list):
                items = "[dim],[/dim] ".join(escape(str(x)) for x in value)
                lines.append(f"[bold cyan]{k}:[/bold cyan] {items}")
            elif isinstance(value, dict):
                lines.append(f"[bold cyan]{k}:[/bold cyan]")
                for sub_k, sub_v in value.items():
                    lines.append(
                        f"  [dim]{escape(str(sub_k))}:[/dim] {escape(str(sub_v))}"
                    )
            else:
                lines.append(f"[bold cyan]{k}:[/bold cyan] {escape(str(value))}")
        return "\n".join(lines)

    @on(Button.Pressed, "#btn-close")
    def _close(self) -> None:
        self.dismiss()


class CopyModeModal(ModalScreen[None]):
    """Full-log viewer that allows mouse selection and clipboard copy.

    Opened by pressing ``y`` in :class:`JobApp`.  Shows the entire
    accumulated log in a read-only :class:`~textual.widgets.TextArea` where
    the user can click-and-drag to select any range of lines, then press
    ``y`` to copy the selection (or the whole log when nothing is selected).
    Press ``Esc`` to return to the live coloured view.

    Args:
        log_text: Plain-text snapshot of the log at the time ``y`` was pressed.
    """

    BINDINGS: ClassVar = [
        Binding("escape", "dismiss", "Close"),
        Binding("y", "copy_selection", "Copy"),
    ]

    DEFAULT_CSS = """
    CopyModeModal {
        align: center middle;
    }
    CopyModeModal > Vertical {
        width: 98%;
        height: 95%;
        border: round $accent;
        background: $surface;
    }
    #copy-title {
        text-align: center;
        text-style: bold;
        padding: 0 2;
        background: $accent-darken-2;
        color: $text;
    }
    #copy-area {
        height: 1fr;
    }
    #copy-hint {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    def __init__(self, log_text: str) -> None:
        super().__init__()
        self._log_text = log_text

    def compose(self) -> ComposeResult:
        """Compose the copy-mode overlay."""
        with Vertical():
            yield Static(
                "Copy mode — click-drag to select, y to copy, Esc to close",
                id="copy-title",
                markup=False,
            )
            yield TextArea(
                self._log_text,
                id="copy-area",
                read_only=True,
                show_line_numbers=True,
                soft_wrap=False,
            )
            yield Static(
                "y  copy selection (or full log)    Esc  return to live view",
                id="copy-hint",
                markup=False,
            )

    def on_mount(self) -> None:
        """Focus the text area and scroll to the bottom."""
        ta = self.query_one("#copy-area", TextArea)
        ta.focus()
        ta.scroll_end(animate=False)

    def action_copy_selection(self) -> None:
        """Copy selected text, or the full log if nothing is selected."""
        ta = self.query_one("#copy-area", TextArea)
        selected = ta.selected_text
        text = selected if selected.strip() else self._log_text
        label = "Selection copied." if selected.strip() else "Full log copied."
        if _copy_to_clipboard(text):
            self.notify(label, timeout=3)
        else:
            self.notify(
                "No clipboard tool found (xclip / xsel / wl-copy).",
                severity="warning",
                timeout=4,
            )


class JobApp(App[bool]):
    """Full-screen TUI for a single neuropress cloud job.

    Layout::

        ┌──────────────────────────────────────────────────────┐
        │ job-name             Phase   0:42          [Config]  │
        ├──────────────────────────────────────────────────────┤
        │ $ mkdir -p /workspace                                │
        │ $ uv sync --extra llm                                │
        │ ... streaming output ...                             │
        └──────────────────────────────────────────────────────┘
         c Config   y Copy mode   q Quit

    The log body is a :class:`~textual.widgets.RichLog` that renders ANSI
    colours.  Press ``y`` to open a copy-mode overlay where you can
    mouse-select specific lines and copy them to the clipboard.

    Args:
        job_name: Human-readable job name shown in the header.
        config: Key-value pairs shown in the configuration modal.
        run_fn: Called in a background thread; returns ``True`` on success.
            The function should use the runner's console/phase helpers which
            are already wired to the TUI before ``run_fn`` is called.
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #header {
        height: 3;
        background: $primary-darken-2;
        padding: 0 2;
        layout: horizontal;
    }

    #header-label {
        width: 1fr;
        content-align: left middle;
    }

    #header-elapsed {
        width: 7;
        content-align: right middle;
        color: $text-muted;
    }

    #log {
        height: 1fr;
    }
    """

    BINDINGS: ClassVar = [
        Binding("c", "show_config", "Config"),
        Binding("y", "copy_log", "Copy log"),
        Binding("q", "request_quit", "Quit"),
    ]

    def __init__(
        self,
        job_name: str,
        config: dict[str, Any],
        run_fn: Callable[[], bool],
    ) -> None:
        super().__init__()
        self._job_name = job_name
        self._config = config
        self._run_fn = run_fn
        self._phase: str = ""
        self._start: float = 0.0
        self._result: bool = False
        self._log_buffer: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the main layout."""
        with Horizontal(id="header"):
            yield Static(
                f"[bold]{escape(self._job_name)}[/bold]",
                id="header-label",
                markup=True,
            )
            yield Static("0:00", id="header-elapsed")
        yield RichLog(id="log", markup=False, highlight=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        """Start the timer and background job worker."""
        self._start = time.monotonic()
        self.set_interval(1.0, self._tick)
        self._job_worker()

    # ----------------------------------------------------------------- public

    def write(self, content: str | Text) -> None:
        """Append *content* to the scrollable log (thread-safe).

        Safe to call from any thread, including Rich background refresh threads.
        Drops the message silently if the DOM is not yet mounted.

        Appends the plain-text form to :attr:`_log_buffer` so that
        :class:`CopyModeModal` (opened with ``y``) always receives a
        current snapshot of the log.

        Args:
            content: A plain string or a :class:`rich.text.Text` object with
                pre-decoded ANSI styles.
        """
        plain = content.plain if isinstance(content, Text) else str(content)
        self._log_buffer.append(plain)

        def _do() -> None:
            try:
                self.query_one("#log", RichLog).write(content)
            except Exception:  # NoMatches before mount, or widget gone
                pass

        try:
            self.call_from_thread(_do)
        except Exception:  # app not yet running
            pass

    def set_phase(self, phase: str) -> None:
        """Update the phase indicator in the header (thread-safe).

        Args:
            phase: Short label for the current run phase (e.g. ``"Upload"``).
        """

        def _update() -> None:
            self._phase = phase
            self._refresh_header_label()

        self.call_from_thread(_update)

    # --------------------------------------------------------------- internals

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._start
        mins, secs = divmod(int(elapsed), 60)
        self.query_one("#header-elapsed", Static).update(f"{mins}:{secs:02d}")

    def _refresh_header_label(self) -> None:
        label = f"[bold]{escape(self._job_name)}[/bold]"
        if self._phase:
            label += f"  [dim]{escape(self._phase)}[/dim]"
        self.query_one("#header-label", Static).update(label)

    @work(thread=True)
    def _job_worker(self) -> None:
        try:
            self._result = self._run_fn()
        except Exception as exc:
            self.write(f"[ERROR] {exc}")
            self._result = False
        finally:
            self.call_from_thread(self._on_done)

    def _on_done(self) -> None:
        if self._result:
            self._phase = "[green]Done ✓[/green]"
        else:
            self._phase = "[red]Failed ✗[/red]"
        label = f"[bold]● {escape(self._job_name)}[/bold]  {self._phase}"
        self.query_one("#header-label", Static).update(label)

    def action_show_config(self) -> None:
        """Open the configuration modal."""
        self.push_screen(ConfigModal(self._config))

    def action_copy_log(self) -> None:
        """Open copy-mode: a selectable plain-text overlay of the current log.

        Pushes :class:`CopyModeModal` with a snapshot of the accumulated log.
        Inside the modal the user can mouse-select any range of lines and
        press ``y`` to copy the selection (or the whole log if nothing is
        selected).  Press ``Esc`` to return to the live coloured view.
        """
        self.push_screen(CopyModeModal("\n".join(self._log_buffer)))

    def action_request_quit(self) -> None:
        """Exit the TUI, returning the job result."""
        self.exit(self._result)
