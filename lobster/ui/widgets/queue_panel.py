"""
Queue panel showing download and publication queue counts.

Design: Compact header-only display with inline counts.
Detailed status breakdown is handled by QueueStatusBar widget.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive

from rich.text import Text


class QueuePanel(Vertical):
    """
    Compact queue panel showing queue names with active entry counts.

    Layout:
    ┌─ Queues ─────────────────────┐
    │ Downloads (3)                │
    │ Publications (5)             │
    └──────────────────────────────┘

    Detailed status breakdown is provided by QueueStatusBar below.
    """

    DEFAULT_CSS = """
    QueuePanel {
        height: auto;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    QueuePanel > Static.panel-header {
        text-style: bold;
        color: #CC2C18;
        height: 1;
    }

    QueuePanel > Static.queue-line {
        height: 1;
        color: $text 70%;
    }
    """

    # Reactive counts for automatic UI updates
    download_count: reactive[int] = reactive(0)
    publication_count: reactive[int] = reactive(0)

    def __init__(self, client=None, **kwargs) -> None:
        """
        Initialize queue panel.

        Args:
            client: AgentClient with data_manager and publication_queue
        """
        super().__init__(**kwargs)
        self.client = client

    def compose(self) -> ComposeResult:
        yield Static("Queues", classes="panel-header")
        yield Static("Downloads", id="download-line", classes="queue-line")
        yield Static("Publications", id="publication-line", classes="queue-line")

    def on_mount(self) -> None:
        """Start refresh timer."""
        self._refresh_queues()
        self.set_interval(2.0, self._refresh_queues)

    def _refresh_queues(self) -> None:
        """Refresh queue counts and update display."""
        if not self.client:
            return

        try:
            # Download queue count
            download_count = 0
            if hasattr(self.client, "data_manager"):
                queue = self.client.data_manager.download_queue
                entries = queue.list_entries()
                # Count active entries (pending or in_progress)
                download_count = sum(
                    1 for e in entries if e.status in ["pending", "in_progress"]
                )
            self.download_count = download_count
            self._update_line("download-line", "Downloads", download_count)

            # Publication queue count
            publication_count = 0
            if hasattr(self.client, "publication_queue") and self.client.publication_queue:
                entries = self.client.publication_queue.list_entries()
                # Count active entries (not completed or failed)
                publication_count = sum(
                    1 for e in entries if e.status not in ["completed", "failed"]
                )
            self.publication_count = publication_count
            self._update_line("publication-line", "Publications", publication_count)

        except Exception:
            pass  # Queues not available

    def _update_line(self, widget_id: str, label: str, count: int) -> None:
        """
        Update a queue line with label and count.

        Args:
            widget_id: ID of Static widget to update
            label: Queue label (e.g., "Downloads")
            count: Number of active entries
        """
        text = Text()
        text.append(f"{label} ", style="")

        if count > 0:
            text.append(f"({count})", style="bold #CC2C18")
        else:
            text.append("(0)", style="dim")

        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(text)
        except Exception:
            pass  # Widget not ready

    def get_counts(self) -> dict:
        """
        Get current queue counts.

        Public API for external consumers (CLI, tests, other UIs).

        Returns:
            Dict with 'download' and 'publication' counts
        """
        return {
            "download": self.download_count,
            "publication": self.publication_count,
        }
