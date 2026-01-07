"""Status bar widget showing system status (queues, model, provider)."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.reactive import reactive

from rich.text import Text
from rich.console import RenderableType

import re
from typing import Any

DEFAULT_ACCENT = "#CC2C18"
COLOR_RE = re.compile(r"Color\((\d+),\s*(\d+),\s*(\d+)\)")


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse a hex color ("#rrggbb" or "#rgb") into an RGB tuple."""
    value = color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _tint_hex_color(color: Any, amount: float = 0.15) -> str:
    """Return a lighter hex color by mixing with white."""
    hex_color = _normalize_color(color)
    r, g, b = _hex_to_rgb(hex_color)
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _normalize_color(color: Any) -> str:
    """Coerce various color representations to a hex string."""
    if color is None:
        return DEFAULT_ACCENT

    # Color objects stringify as "Color(r, g, b)"
    value = str(color).strip()

    if value.startswith("#"):
        return value

    match = COLOR_RE.match(value)
    if match:
        r, g, b = (int(match.group(i)) for i in range(1, 4))
        return f"#{r:02x}{g:02x}{b:02x}"

    return DEFAULT_ACCENT


def get_friendly_model_name(model_id: str, provider: str) -> str:
    """
    Convert long model IDs to friendly display names.

    Args:
        model_id: Full model identifier (e.g., "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        provider: Provider name ("bedrock", "anthropic", "ollama")

    Returns:
        Friendly name (e.g., "sonnet 4.5", "gptoss:20b")
    """
    # Bedrock model ID mappings
    if provider == "bedrock":
        if "claude-sonnet-4-5" in model_id or "claude-4-5-sonnet" in model_id:
            return "sonnet 4.5"
        elif "claude-haiku-4-5" in model_id or "claude-4-5-haiku" in model_id:
            return "haiku 4.5"
        elif "claude-sonnet-4" in model_id or "claude-4-sonnet" in model_id:
            return "sonnet 4.0"
        elif "claude-opus" in model_id:
            return "opus 4.5"

    # Anthropic Direct
    elif provider == "anthropic":
        if "claude-4-5-sonnet" in model_id or "claude-sonnet-4-5" in model_id:
            return "sonnet 4.5"
        elif "claude-4-5-haiku" in model_id or "claude-haiku-4-5" in model_id:
            return "haiku 4.5"
        elif "claude-4-sonnet" in model_id or "claude-sonnet-4" in model_id:
            return "sonnet 4.0"
        elif "claude-opus" in model_id:
            return "opus 4.5"

    # Ollama - use model name as-is (already friendly)
    elif provider == "ollama":
        return model_id  # e.g., "gptoss:20b", "llama3:8b-instruct"

    # Fallback: truncate long IDs
    if len(model_id) > 25:
        return model_id[:25] + "..."

    return model_id


class StatusSegment(Static):
    """A single segment of the status bar that renders Rich Text."""

    DEFAULT_CSS = """
    StatusSegment {
        width: 1fr;
        height: 1;
        content-align: center middle;
        text-style: bold;
    }

    StatusSegment.left {
        content-align: left middle;
    }

    StatusSegment.center {
        content-align: center middle;
    }

    StatusSegment.right {
        content-align: right middle;
    }
    """

    def __init__(
        self,
        content: RenderableType = "",
        *,
        align: str = "center",
        id: str | None = None,
    ) -> None:
        super().__init__(content, id=id)
        self._align = align

    def on_mount(self) -> None:
        """Apply alignment class on mount."""
        self.add_class(self._align)


class StatusBar(Horizontal):
    """
    Persistent status indicators (Elia-inspired).

    Displays (compact single line):
    - Subscription tier (if not free)
    - Provider + model
    - Download queue count
    - Publication queue count (premium)
    - Background worker count
    - Agent status

    All attributes are reactive (auto-update UI).
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        min-height: 1;
        dock: top;
        padding: 0 1;
        background: transparent;
    }

    StatusBar > StatusSegment {
        color: $text;
    }
    """

    # Reactive attributes (changes trigger re-render)
    subscription_tier: reactive[str] = reactive("free")
    provider_name: reactive[str] = reactive("unknown")
    model_name: reactive[str] = reactive("unknown")
    download_count: reactive[int] = reactive(0)
    publication_count: reactive[int] = reactive(0)
    agent_status: reactive[str] = reactive("idle")
    worker_count: reactive[int] = reactive(0)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mounted = False

    def compose(self) -> ComposeResult:
        """Create the three status segments."""
        yield StatusSegment(id="status-left", align="left")
        yield StatusSegment(id="status-center", align="center")
        yield StatusSegment(id="status-right", align="right")

    def on_mount(self) -> None:
        """Initialize after widget tree is ready."""
        self._mounted = True
        self._refresh_all_segments()

    def _refresh_all_segments(self) -> None:
        """Update all three segments with current values."""
        if not self._mounted:
            return

        try:
            left = self.query_one("#status-left", StatusSegment)
            center = self.query_one("#status-center", StatusSegment)
            right = self.query_one("#status-right", StatusSegment)

            left.update(self._render_left())
            center.update(self._render_center())
            right.update(self._render_right())
        except Exception:
            # Widget tree not ready yet - ignore
            pass

    def _render_left(self) -> Text:
        """Render left segment: tier + provider/model."""
        text = Text()

        # Subscription tier (if not free)
        if self.subscription_tier and self.subscription_tier.lower() != "free":
            text.append(f" {self.subscription_tier.upper()} ", style="bold reverse")
            text.append(" ")

        # Provider/model
        if self.provider_name != "unknown" or self.model_name != "unknown":
            provider = self.provider_name if self.provider_name != "unknown" else "?"
            model = self.model_name if self.model_name != "unknown" else "?"
            text.append(f"{provider}/", style="dim")
            text.append(f"{model}", style="bold")

        return text if text.plain.strip() else Text(" ready", style="dim italic")

    def _render_center(self) -> Text:
        """Render center segment: queue counts."""
        text = Text()
        parts = []

        if self.download_count > 0:
            parts.append(("downloads", str(self.download_count), "#CC2C18"))

        if self.publication_count > 0:
            parts.append(("papers", str(self.publication_count), "white"))

        if self.worker_count > 0:
            parts.append(("workers", str(self.worker_count), "#CC2C18"))

        if not parts:
            return Text(" - ", style="dim")

        for i, (label, count, color) in enumerate(parts):
            if i > 0:
                text.append(" | ", style="dim")
            text.append(f"{count} ", style=f"bold {color}")
            text.append(label, style="dim")

        return text

    def _render_right(self) -> Text:
        """Render right segment: agent status."""
        status = self.agent_status or "idle"

        status_styles = {
            "idle": ("dim", "idle"),
            "processing": ("bold #CC2C18", "processing..."),
            "thinking": ("italic #E84D3A", "thinking..."),
            "error": ("bold red", "error"),
        }

        style, display = status_styles.get(status, ("", status))

        text = Text()
        text.append("agent: ", style="dim")
        text.append(display, style=style)

        return text

    # Watchers - all delegate to refresh
    def watch_subscription_tier(self, _: str) -> None:
        self._refresh_all_segments()

    def watch_provider_name(self, _: str) -> None:
        self._refresh_all_segments()

    def watch_model_name(self, _: str) -> None:
        self._refresh_all_segments()

    def watch_download_count(self, _: int) -> None:
        self._refresh_all_segments()

    def watch_publication_count(self, _: int) -> None:
        self._refresh_all_segments()

    def watch_agent_status(self, _: str) -> None:
        self._refresh_all_segments()

    def watch_worker_count(self, _: int) -> None:
        self._refresh_all_segments()
