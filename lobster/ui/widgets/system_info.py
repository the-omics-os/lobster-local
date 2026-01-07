"""System info panel showing CPU, memory, GPU status."""

import os
import platform
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive

from rich.text import Text


def _get_memory_info() -> tuple[float, float]:
    """Get memory usage (used_gb, total_gb)."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3)
    except ImportError:
        return 0.0, 0.0


def _get_cpu_percent() -> float:
    """Get CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return 0.0


def _get_cpu_count() -> int:
    """Get number of CPU cores."""
    return os.cpu_count() or 0


def _get_gpu_info() -> Optional[str]:
    """Get GPU info if available."""
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.processor() == "arm":
        return "Apple Silicon (MPS)"

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return f"{name} ({mem:.0f}GB)"
    except ImportError:
        pass

    # Check for MPS (Apple Silicon via torch)
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
    except ImportError:
        pass

    return None


class SystemInfoPanel(Vertical):
    """
    System resource monitor panel.

    Displays:
    - CPU usage and core count
    - Memory usage
    - GPU availability
    - Python version
    """

    DEFAULT_CSS = """
    SystemInfoPanel {
        height: auto;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    SystemInfoPanel > Static {
        height: 1;
        padding: 0;
        margin: 0;
    }

    SystemInfoPanel .header {
        text-style: bold;
        color: #CC2C18;
        margin-bottom: 1;
    }

    SystemInfoPanel .dim {
        color: $text 60%;
    }
    """

    # Reactive values for live updates
    cpu_percent: reactive[float] = reactive(0.0)
    mem_used: reactive[float] = reactive(0.0)
    mem_total: reactive[float] = reactive(0.0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._gpu_info: Optional[str] = None
        self._cpu_count: int = 0

    def compose(self) -> ComposeResult:
        yield Static("System", classes="header")
        yield Static(id="cpu-line")
        yield Static(id="mem-line")
        yield Static(id="gpu-line")

    def on_mount(self) -> None:
        """Initialize and start refresh timer."""
        self._cpu_count = _get_cpu_count()
        self._gpu_info = _get_gpu_info()

        # Initial update
        self._refresh_stats()

        # Update every 2 seconds
        self.set_interval(2.0, self._refresh_stats)

    def _refresh_stats(self) -> None:
        """Refresh CPU and memory stats."""
        self.cpu_percent = _get_cpu_percent()
        self.mem_used, self.mem_total = _get_memory_info()
        self._update_display()

    def _update_display(self) -> None:
        """Update the display labels."""
        try:
            # CPU line
            cpu_text = Text()
            cpu_text.append("CPU  ", style="dim")
            cpu_text.append(f"{self.cpu_percent:4.0f}%", style="bold" if self.cpu_percent > 50 else "")
            cpu_text.append(f"  ({self._cpu_count} cores)", style="dim")
            self.query_one("#cpu-line", Static).update(cpu_text)

            # Memory line
            mem_text = Text()
            mem_text.append("MEM  ", style="dim")
            mem_pct = (self.mem_used / self.mem_total * 100) if self.mem_total > 0 else 0
            mem_text.append(f"{mem_pct:4.0f}%", style="bold" if mem_pct > 70 else "")
            mem_text.append(f"  ({self.mem_used:.1f}/{self.mem_total:.1f} GB)", style="dim")
            self.query_one("#mem-line", Static).update(mem_text)

            # GPU line
            gpu_text = Text()
            gpu_text.append("GPU  ", style="dim")
            if self._gpu_info:
                gpu_text.append(self._gpu_info, style="white")
            else:
                gpu_text.append("none", style="dim italic")
            self.query_one("#gpu-line", Static).update(gpu_text)

        except Exception:
            pass  # Widget not ready
