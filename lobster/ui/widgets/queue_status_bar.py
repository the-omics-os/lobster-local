"""
Queue status bar showing status breakdown for download and publication queues.

Design: Compact status visualization using existing queue.get_statistics() API.
Reusable pattern - any future UI can consume the same statistics API.

Status display configuration imported from schema files (single source of truth).
"""

from typing import Dict, Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from rich.text import Text

# Import status enums with display properties from their single source of truth
from lobster.core.schemas.download_queue import (
    DownloadStatus,
    DOWNLOAD_STATUS_DISPLAY,
)
from lobster.core.schemas.publication_queue import (
    PublicationStatus,
    PUBLICATION_STATUS_DISPLAY,
)


class QueueStatusBar(Vertical):
    """
    Compact status breakdown for download and publication queues.

    Consumes existing queue.get_statistics() API for data - no duplicate logic.
    Updates every 2 seconds alongside QueuePanel.

    Layout (compact inline badges):
    +-Status-----------------------+
    | Downloads: o2 *1 v15 x1      |
    | Pubs: o5 *2 ->3 v20 x2       |
    +------------------------------+

    Public API:
        get_combined_statistics() - Returns stats for both queues
            (useful for CLI, tests, alternative UIs)
    """

    DEFAULT_CSS = """
    QueueStatusBar {
        height: auto;
        max-height: 5;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    QueueStatusBar > Static.panel-header {
        text-style: bold;
        color: #CC2C18;
        height: 1;
    }

    QueueStatusBar > Static.status-line {
        height: 1;
    }
    """

    def __init__(self, client=None, **kwargs) -> None:
        """
        Initialize queue status bar.

        Args:
            client: AgentClient with data_manager and publication_queue
        """
        super().__init__(**kwargs)
        self.client = client

    def compose(self) -> ComposeResult:
        yield Static("Status", classes="panel-header")
        yield Static("", id="download-status", classes="status-line")
        yield Static("", id="pub-status", classes="status-line")

    def on_mount(self) -> None:
        """Start refresh timer - aligned with QueuePanel's 2s interval."""
        self._refresh_status()
        self.set_interval(2.0, self._refresh_status)

    def _refresh_status(self) -> None:
        """Refresh status display using existing get_statistics() API."""
        if not self.client:
            return

        try:
            # Download queue statistics - uses existing API
            download_stats = self._get_download_statistics()
            self._update_status_line(
                "download-status",
                "Downloads:",
                download_stats.get("by_status", {}),
                DOWNLOAD_STATUS_DISPLAY,
            )

            # Publication queue statistics - uses existing API
            pub_stats = self._get_publication_statistics()
            self._update_status_line(
                "pub-status",
                "Pubs:",
                pub_stats.get("by_status", {}),
                PUBLICATION_STATUS_DISPLAY,
            )

        except Exception:
            pass  # Queues may not be initialized yet

    def _get_download_statistics(self) -> Dict[str, Any]:
        """
        Get download queue statistics via existing API.

        Returns:
            Statistics dict from DownloadQueue.get_statistics()
        """
        if not hasattr(self.client, "data_manager"):
            return {}

        queue = self.client.data_manager.download_queue
        if queue is None:
            return {}

        return queue.get_statistics()

    def _get_publication_statistics(self) -> Dict[str, Any]:
        """
        Get publication queue statistics via existing API.

        Returns:
            Statistics dict from PublicationQueue.get_statistics()
        """
        if not hasattr(self.client, "publication_queue"):
            return {}

        queue = self.client.publication_queue
        if queue is None:
            return {}

        return queue.get_statistics()

    def _update_status_line(
        self,
        widget_id: str,
        label: str,
        status_counts: Dict[str, int],
        display_config: Dict[str, tuple],
    ) -> None:
        """
        Update a status line widget with formatted counts.

        Args:
            widget_id: ID of Static widget to update
            label: Label prefix (e.g., "Downloads:")
            status_counts: Dict mapping status name to count
            display_config: Dict mapping status -> (icon, style) from schema
        """
        text = Text()
        text.append(f"{label} ", style="italic dim")

        # Show non-zero counts with icons from schema display config
        has_entries = False
        for status, count in status_counts.items():
            if count > 0:
                has_entries = True
                icon, style = display_config.get(status, ("?", "dim"))
                text.append(f"{icon}{count} ", style=style)

        # Show "empty" if all zeros
        if not has_entries:
            text.append("empty", style="dim italic")

        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(text)
        except Exception:
            pass  # Widget not ready

    def get_combined_statistics(self) -> Dict[str, Any]:
        """
        Get combined statistics for both queues.

        Public API for external consumers (CLI, other UIs, tests).

        Returns:
            Combined statistics dict with 'download' and 'publication' keys
        """
        return {
            "download": self._get_download_statistics(),
            "publication": self._get_publication_statistics(),
        }
