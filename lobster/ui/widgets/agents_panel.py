"""Agents panel showing AI agent readiness and status."""

from typing import Dict, Optional, Set
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class AgentsPanel(Vertical):
    """
    AI agents status panel.

    Shows which agents are available, their status:
    - ● Idle (green) - ready to work
    - ▶ Active (yellow/pulsing) - currently processing
    - ○ Offline (dim) - not available in current tier
    """

    DEFAULT_CSS = """
    AgentsPanel {
        height: auto;
        padding: 0 1;
        border: round #CC2C18 30%;
    }

    AgentsPanel > Static.header {
        text-style: bold;
        color: #CC2C18;
    }

    AgentsPanel > Static.subheader {
        color: $text 60%;
        margin-bottom: 1;
    }

    AgentsPanel > Static.agents-list {
        height: auto;
    }
    """

    # Track which agents are active
    active_agents: reactive[Set[str]] = reactive(set())

    def __init__(self, client=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client = client
        self._agent_status: Dict[str, str] = {}
        self._available_agents: Set[str] = set()
        self._pulse_state = False

    def compose(self) -> ComposeResult:
        yield Static("AGENTS", classes="header")
        yield Static("", id="agents-count", classes="subheader")
        yield Static(id="agents-list", classes="agents-list")

    def on_mount(self) -> None:
        """Initialize agent list."""
        self._load_agents()
        self._update_display()
        # Pulse animation for active agents
        self.set_interval(0.5, self._pulse_active)

    def _load_agents(self) -> None:
        """Load available agents from registry (supervisor-accessible only)."""
        try:
            from lobster.config.agent_registry import AGENT_REGISTRY, _ensure_plugins_loaded
            from lobster.core.license_manager import get_current_tier
            from lobster.config.subscription_tiers import is_agent_available

            # Ensure plugins are loaded before accessing AGENT_REGISTRY
            _ensure_plugins_loaded()

            tier = get_current_tier()

            # First pass: collect all child agents (not supervisor-accessible)
            child_agents = set()
            for config in AGENT_REGISTRY.values():
                if config.child_agents:
                    child_agents.update(config.child_agents)

            for name, config in AGENT_REGISTRY.items():
                # Skip agents that are explicitly not supervisor-accessible
                if config.supervisor_accessible is False:
                    continue
                # Skip child agents (unless explicitly supervisor_accessible=True)
                if name in child_agents and config.supervisor_accessible is not True:
                    continue

                if is_agent_available(name, tier):
                    self._available_agents.add(name)
                    self._agent_status[name] = "idle"
                else:
                    self._agent_status[name] = "offline"

        except Exception:
            # Fallback - show basic agents
            basic_agents = [
                "supervisor", "research_agent", "data_expert",
                "transcriptomics_expert", "proteomics_expert"
            ]
            for name in basic_agents:
                self._available_agents.add(name)
                self._agent_status[name] = "idle"

    def _pulse_active(self) -> None:
        """Toggle pulse state for active agents."""
        self._pulse_state = not self._pulse_state
        if self.active_agents:
            self._update_display()

    def _update_display(self) -> None:
        """Update the agents list display."""
        try:
            # Update count
            online = len(self._available_agents)
            total = len(self._agent_status)
            count_widget = self.query_one("#agents-count", Static)
            count_widget.update(f"({online}/{total} online)")

            # Update list
            list_widget = self.query_one("#agents-list", Static)
            text = Text()

            # Sort: active first, then idle, then offline
            def sort_key(item):
                name, status = item
                if name in self.active_agents:
                    return (0, name)
                elif status == "idle":
                    return (1, name)
                else:
                    return (2, name)

            sorted_agents = sorted(self._agent_status.items(), key=sort_key)

            for name, status in sorted_agents[:8]:  # Show max 8
                # Determine display
                display_name = self._format_agent_name(name)

                if name in self.active_agents:
                    # Active/processing - pulse between ▶ and ▸
                    dot = "▶" if self._pulse_state else "▸"
                    text.append(f" {dot} ", style="bold #CC2C18")
                    text.append(f"{display_name}", style="bold")
                    text.append(" (active)\n", style="#CC2C18")
                elif status == "idle":
                    text.append(" ● ", style="white")
                    text.append(f"{display_name}\n", style="")
                else:
                    text.append(" ○ ", style="dim")
                    text.append(f"{display_name}\n", style="dim")

            if len(self._agent_status) > 8:
                text.append(f" ... +{len(self._agent_status) - 8} more\n", style="dim")

            list_widget.update(text)
        except Exception:
            pass

    def _format_agent_name(self, name: str) -> str:
        """Format agent name for display."""
        # Remove _agent, _expert suffixes and clean up
        display = name.replace("_agent", "").replace("_expert", "")
        return display[:12]  # Truncate for panel width

    def set_agent_active(self, agent_name: str) -> None:
        """Mark an agent as active/processing."""
        new_set = set(self.active_agents)
        new_set.add(agent_name)
        self.active_agents = new_set

    def set_agent_idle(self, agent_name: str) -> None:
        """Mark an agent as idle."""
        new_set = set(self.active_agents)
        new_set.discard(agent_name)
        self.active_agents = new_set

    def set_all_idle(self) -> None:
        """Mark all agents as idle (call when query completes)."""
        self.active_agents = set()

    def watch_active_agents(self, _: Set[str]) -> None:
        """React to active agents changes."""
        self._update_display()
