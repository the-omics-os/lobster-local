"""Adapters panel showing data format support status."""

from typing import Dict, Set
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from rich.text import Text


# Data format adapter registry
ADAPTER_REGISTRY = {
    # Transcriptomics
    "H5AD": {"type": "transcriptomics", "icon": "📦", "desc": "AnnData format"},
    "10X MTX": {"type": "transcriptomics", "icon": "🔢", "desc": "10X Genomics matrix"},
    "Kallisto": {"type": "transcriptomics", "icon": "🐟", "desc": "Kallisto abundance"},
    "Salmon": {"type": "transcriptomics", "icon": "🐠", "desc": "Salmon quant.sf"},
    # Proteomics
    "MaxQuant": {"type": "proteomics", "icon": "🔬", "desc": "MaxQuant DDA"},
    "Spectronaut": {"type": "proteomics", "icon": "📊", "desc": "Spectronaut DIA"},
    "Olink": {"type": "proteomics", "icon": "🧪", "desc": "Olink NPX"},
    # Generic
    "CSV": {"type": "generic", "icon": "📄", "desc": "CSV/TSV files"},
}


class AdaptersPanel(Vertical):
    """
    Data format adapters status panel.

    Shows which data formats are supported:
    ● Available - adapter loaded
    ○ Unavailable - missing dependencies
    """

    DEFAULT_CSS = """
    AdaptersPanel {
        height: auto;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    AdaptersPanel > Static.header {
        text-style: bold;
        color: #CC2C18;
        margin-bottom: 1;
    }

    AdaptersPanel > Static.adapters-grid {
        height: auto;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._adapter_status: Dict[str, str] = {}
        self._check_adapters()

    def compose(self) -> ComposeResult:
        yield Static("ADAPTERS", classes="header")
        yield Static(id="adapters-grid", classes="adapters-grid")

    def on_mount(self) -> None:
        """Initialize display."""
        self._update_display()

    def _check_adapters(self) -> None:
        """Check which adapters are available based on imports."""
        # Check transcriptomics adapters
        try:
            import anndata
            self._adapter_status["H5AD"] = "ok"
        except ImportError:
            self._adapter_status["H5AD"] = "missing"

        try:
            import scanpy
            self._adapter_status["10X MTX"] = "ok"
        except ImportError:
            self._adapter_status["10X MTX"] = "missing"

        # Kallisto/Salmon use pandas - always available
        self._adapter_status["Kallisto"] = "ok"
        self._adapter_status["Salmon"] = "ok"

        # Proteomics adapters
        try:
            import pandas
            self._adapter_status["MaxQuant"] = "ok"
            self._adapter_status["Spectronaut"] = "ok"
            self._adapter_status["Olink"] = "ok"
        except ImportError:
            self._adapter_status["MaxQuant"] = "missing"
            self._adapter_status["Spectronaut"] = "missing"
            self._adapter_status["Olink"] = "missing"

        # CSV is always available
        self._adapter_status["CSV"] = "ok"

    def _update_display(self) -> None:
        """Update the adapters grid display."""
        try:
            grid = self.query_one("#adapters-grid", Static)

            text = Text()

            # Group by type
            by_type: Dict[str, list] = {"transcriptomics": [], "proteomics": [], "generic": []}
            for name, config in ADAPTER_REGISTRY.items():
                by_type[config["type"]].append(name)

            # Display transcriptomics row
            text.append(" Trans: ", style="dim")
            for name in by_type["transcriptomics"]:
                status = self._adapter_status.get(name, "unknown")
                dot, style = self._get_indicator(status)
                text.append(f"{dot}", style=style)
            text.append("\n")

            # Display proteomics row
            text.append(" Prot:  ", style="dim")
            for name in by_type["proteomics"]:
                status = self._adapter_status.get(name, "unknown")
                dot, style = self._get_indicator(status)
                text.append(f"{dot}", style=style)
            text.append("\n")

            # Display generic row
            text.append(" Other: ", style="dim")
            for name in by_type["generic"]:
                status = self._adapter_status.get(name, "unknown")
                dot, style = self._get_indicator(status)
                text.append(f"{dot}", style=style)

            grid.update(text)
        except Exception:
            pass

    def _get_indicator(self, status: str) -> tuple[str, str]:
        """Get status indicator dot and style."""
        indicators = {
            "ok": ("●", "white"),
            "missing": ("○", "dim"),
            "error": ("●", "red"),
            "unknown": ("○", "dim"),
        }
        return indicators.get(status, ("○", "dim"))

    def get_available_adapters(self) -> Set[str]:
        """Return set of available adapter names."""
        return {name for name, status in self._adapter_status.items() if status == "ok"}
