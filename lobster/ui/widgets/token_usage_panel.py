"""Token usage panel showing session costs and token consumption."""

from typing import Dict, Any, Optional
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class TokenUsagePanel(Vertical):
    """
    Session token usage and cost tracking panel.

    Shows:
    - Total tokens used (input/output)
    - Estimated cost in USD
    - Per-agent breakdown (compact)

    Updates in real-time as agents process queries.
    """

    DEFAULT_CSS = """
    TokenUsagePanel {
        height: auto;
        padding: 0 1;
        border: round $primary 30%;
    }

    TokenUsagePanel > Static.header {
        text-style: bold;
        color: $text;
    }

    TokenUsagePanel > Static.subheader {
        color: $text 60%;
        margin-bottom: 1;
    }

    TokenUsagePanel > Static.usage-grid {
        height: auto;
    }
    """

    # Reactive properties for live updates
    total_tokens: reactive[int] = reactive(0)
    total_cost: reactive[float] = reactive(0.0)
    input_tokens: reactive[int] = reactive(0)
    output_tokens: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agent_usage: Dict[str, Dict[str, Any]] = {}
        self._mounted = False

    def compose(self) -> ComposeResult:
        yield Static("TOKENS", classes="header")
        yield Static("", id="token-cost", classes="subheader")
        yield Static(id="usage-grid", classes="usage-grid")

    def on_mount(self) -> None:
        """Initialize display."""
        self._mounted = True
        self._update_display()

    def _update_display(self) -> None:
        """Update the token usage display."""
        if not self._mounted:
            return

        try:
            # Update cost header
            cost_widget = self.query_one("#token-cost", Static)
            cost_str = f"${self.total_cost:.4f}" if self.total_cost > 0 else "$0.00"
            cost_widget.update(f"Session: {cost_str}")

            # Update usage grid
            grid = self.query_one("#usage-grid", Static)
            text = Text()

            # Token counts with compact display
            text.append(" In:  ", style="dim")
            text.append(f"{self._format_tokens(self.input_tokens)}\n", style="cyan")
            text.append(" Out: ", style="dim")
            text.append(f"{self._format_tokens(self.output_tokens)}\n", style="green")
            text.append(" Tot: ", style="dim")
            text.append(f"{self._format_tokens(self.total_tokens)}\n", style="bold")

            # Per-agent mini breakdown (top 3)
            if self._agent_usage:
                text.append("\n", style="")
                sorted_agents = sorted(
                    self._agent_usage.items(),
                    key=lambda x: x[1].get("total_tokens", 0),
                    reverse=True
                )[:3]

                for agent, usage in sorted_agents:
                    agent_display = agent.replace("_", " ")[:10]
                    tokens = usage.get("total_tokens", 0)
                    text.append(f" {agent_display}: ", style="dim")
                    text.append(f"{self._format_tokens(tokens)}\n", style="")

            grid.update(text)
        except Exception:
            pass

    def _format_tokens(self, tokens: int) -> str:
        """Format token count for display."""
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        else:
            return str(tokens)

    def update_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cost_usd: float,
        agent_usage: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Update token usage from TokenTrackingCallback.

        Args:
            input_tokens: Total input tokens for session
            output_tokens: Total output tokens for session
            total_tokens: Total tokens for session
            cost_usd: Estimated cost in USD
            agent_usage: Per-agent usage breakdown
        """
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.total_cost = cost_usd

        if agent_usage:
            self._agent_usage = agent_usage

        self._update_display()

    def update_from_callback(self, usage_summary: Dict[str, Any]) -> None:
        """
        Update from TokenTrackingCallback.get_usage_summary().

        Args:
            usage_summary: Dict from TokenTrackingCallback.get_usage_summary()
        """
        self.update_usage(
            input_tokens=usage_summary.get("total_input_tokens", 0),
            output_tokens=usage_summary.get("total_output_tokens", 0),
            total_tokens=usage_summary.get("total_tokens", 0),
            cost_usd=usage_summary.get("total_cost_usd", 0.0),
            agent_usage=usage_summary.get("by_agent", {}),
        )

    def watch_total_tokens(self, _: int) -> None:
        """React to token changes."""
        self._update_display()

    def watch_total_cost(self, _: float) -> None:
        """React to cost changes."""
        self._update_display()

    def reset(self) -> None:
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self._agent_usage = {}
        self._update_display()
