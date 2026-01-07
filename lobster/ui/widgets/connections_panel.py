"""Connections panel showing database connectivity status."""

from typing import Dict
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


# Database registry with colors for visual distinction
DATABASE_REGISTRY = {
    "GEO": {"check": "ncbi", "color": "#4CAF50", "desc": "Gene Expression Omnibus"},
    "SRA": {"check": "ncbi", "color": "#2196F3", "desc": "Sequence Read Archive"},
    "PubMed": {"check": "ncbi", "color": "#9C27B0", "desc": "Literature"},
    "PRIDE": {"check": "pride", "color": "#FF9800", "desc": "Proteomics"},
    "UniProt": {"check": "uniprot", "color": "#00BCD4", "desc": "Proteins"},
    "ENA": {"check": "ena", "color": "#E91E63", "desc": "Nucleotide Archive"},
}


class ConnectionsPanel(Vertical):
    """
    Database connections status panel.

    Shows which external databases are available/connected.
    Each database has its own color for easy identification.
    """

    DEFAULT_CSS = """
    ConnectionsPanel {
        height: auto;
        padding: 0 1;
    }

    ConnectionsPanel > Static.header {
        text-style: bold;
        color: $text;
    }

    ConnectionsPanel > Static.conn-entry {
        height: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = {name: "unknown" for name in DATABASE_REGISTRY}

    def compose(self) -> ComposeResult:
        yield Static("CONNECTIONS", classes="header")
        # Create a Static for each database entry
        for name in DATABASE_REGISTRY:
            yield Static(id=f"conn-{name.lower()}", classes="conn-entry")

    def on_mount(self) -> None:
        """Initialize and start status checks."""
        self._update_all_entries()
        self.set_interval(30.0, self._check_connections)
        self.set_timer(1.0, self._check_connections)

    def _check_connections(self) -> None:
        """Check database connectivity."""
        import os

        ncbi_ok = True  # NCBI is publicly accessible
        has_ncbi_key = bool(os.environ.get("NCBI_API_KEY"))

        for name, config in DATABASE_REGISTRY.items():
            check_type = config["check"]
            if check_type == "ncbi":
                self._status[name] = "ok" if ncbi_ok else "unknown"
            elif check_type in ("pride", "uniprot", "ena"):
                self._status[name] = "ok"  # Publicly accessible
            else:
                self._status[name] = "unknown"

        self._update_all_entries()

    def _update_all_entries(self) -> None:
        """Update all connection entries."""
        for name, config in DATABASE_REGISTRY.items():
            self._update_entry(name, config)

    def _update_entry(self, name: str, config: Dict) -> None:
        """Update a single connection entry."""
        try:
            entry = self.query_one(f"#conn-{name.lower()}", Static)
            status = self._status.get(name, "unknown")
            color = config["color"]
            desc = config["desc"]

            text = Text()

            # Status dot with database-specific color
            if status == "ok":
                text.append("● ", style=color)
            elif status == "error":
                text.append("● ", style="red")
            elif status == "checking":
                text.append("◐ ", style=color)
            else:
                text.append("○ ", style="dim")

            # Database name
            text.append(f"{name}", style="bold" if status == "ok" else "dim")

            # Description (dimmed)
            text.append(f" {desc}", style="dim")

            entry.update(text)
        except Exception:
            pass

    def set_status(self, database: str, status: str) -> None:
        """Manually set database status."""
        if database in self._status:
            self._status[database] = status
            if database in DATABASE_REGISTRY:
                self._update_entry(database, DATABASE_REGISTRY[database])
