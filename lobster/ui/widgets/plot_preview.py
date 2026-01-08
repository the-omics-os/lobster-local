"""Plot preview widget showing list of generated plots.

Simple list widget for dashboard sidebar. Use /plots command for detailed view.
"""

from typing import Dict, Any, List
from pathlib import Path

from textual.widgets import Static, ListView, ListItem, Label
from textual.reactive import reactive
from textual.containers import Vertical
from textual import on


class PlotPreview(Vertical):
    """
    Compact plot list for dashboard sidebar.

    Shows plot names only. Use /plots for detailed table view.
    Double-click or press Enter to open plot in browser.
    """

    DEFAULT_CSS = """
    PlotPreview {
        height: auto;
        max-height: 10;
    }

    PlotPreview ListView {
        height: auto;
        max-height: 8;
    }

    PlotPreview ListItem {
        height: 1;
        padding: 0 1;
    }

    PlotPreview ListItem:hover {
        background: $primary 20%;
    }
    """

    plot_count = reactive(0)

    def __init__(self, client=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = client
        self._plots: List[Dict[str, Any]] = []

    def compose(self):
        """Render plot list."""
        yield Static("PLOTS [dim](Enter to open)[/dim]", classes="header")
        yield ListView(id="plot-list")

    def on_mount(self) -> None:
        """Initialize plot list."""
        self.refresh_plots()

    def refresh_plots(self) -> None:
        """Refresh the list of plots."""
        plot_list = self.query_one("#plot-list", ListView)
        plot_list.clear()
        self._plots = []

        if not self.client:
            plot_list.append(ListItem(Label("[dim]No session[/dim]")))
            self.plot_count = 0
            return

        # Get plots from data manager
        if not hasattr(self.client, "data_manager") or not self.client.data_manager.has_data():
            plot_list.append(ListItem(Label("[dim]No plots yet[/dim]")))
            self.plot_count = 0
            return

        plots, _, _ = self.client.data_manager.plot_manager.get_latest_plots(8)

        if not plots:
            plot_list.append(ListItem(Label("[dim]No plots yet[/dim]")))
            self.plot_count = 0
            return

        self._plots = plots

        # Add compact plot entries
        for i, plot in enumerate(plots):
            title = plot.get("original_title", plot.get("name", "Untitled"))
            # Truncate long titles
            display_title = title[:20] + "..." if len(title) > 23 else title
            plot_list.append(
                ListItem(Label(f"● {display_title}"), id=f"plot-{i}")
            )

        self.plot_count = len(plots)

    def _open_plot(self, idx: int) -> None:
        """Open plot at given index in browser."""
        if 0 <= idx < len(self._plots):
            plot = self._plots[idx]
            file_path = plot.get("file_path")

            # Fallback: construct path from workspace if not stored
            if not file_path and self.client and hasattr(self.client, "data_manager"):
                plot_id = plot.get("id", "")
                plot_title = plot.get("original_title", plot.get("title", ""))
                if plot_id:
                    # Reconstruct the filename using same logic as save_plots_to_workspace
                    if len(plot_title) > 80:
                        plot_title = f"{plot_title[:38]}...{plot_title[-38:]}"
                    safe_title = "".join(
                        c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                    ).rstrip().replace(" ", "_")
                    filename_base = f"{plot_id}_{safe_title}" if safe_title else plot_id
                    plots_dir = self.client.data_manager.workspace_path / "plots"
                    file_path = str(plots_dir / f"{filename_base}.html")

            if file_path and Path(file_path).exists():
                import webbrowser
                webbrowser.open(f"file://{file_path}")
                self.notify("Opened in browser", timeout=2)
            else:
                self.notify("Plot not saved yet. Use /save first.", severity="warning")

    @on(ListView.Selected, "#plot-list")
    def on_plot_selected(self, event: ListView.Selected) -> None:
        """Handle Enter key on plot - open in browser."""
        item_id = event.item.id
        if item_id and item_id.startswith("plot-"):
            try:
                idx = int(item_id.replace("plot-", ""))
                self._open_plot(idx)
            except (ValueError, IndexError):
                pass


    def watch_plot_count(self, count: int) -> None:
        """Update display when plot count changes."""
        pass  # Header is static, count shown via list length
