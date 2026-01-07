"""Activity log panel showing real-time agent events and tool usage."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.reactive import reactive
from rich.text import Text


class ActivityType(Enum):
    """Types of activity events."""
    AGENT_START = "agent_start"
    AGENT_THINKING = "thinking"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    HANDOFF = "handoff"
    COMPLETE = "complete"


@dataclass
class ActivityEvent:
    """A single activity event."""
    type: ActivityType
    content: str
    agent: str = ""
    timestamp: datetime = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ActivityLogPanel(VerticalScroll):
    """
    Live activity log showing agent events and tool usage.

    Features:
    - Scrollable log of recent events
    - Color-coded by event type
    - Compact format for cockpit display
    - Auto-scroll to latest

    This provides the visual equivalent of `lobster chat --verbose`.
    """

    DEFAULT_CSS = """
    ActivityLogPanel {
        height: auto;
        max-height: 12;
        padding: 0 1;
        border: round $primary 30%;
    }

    ActivityLogPanel > Static.header {
        text-style: bold;
        color: $text;
        dock: top;
    }

    ActivityLogPanel > Static.log-entry {
        height: auto;
    }
    """

    # Maximum events to keep in memory
    MAX_EVENTS = 50

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: List[ActivityEvent] = []
        self._mounted = False

    def compose(self) -> ComposeResult:
        yield Static("ACTIVITY", classes="header")

    def on_mount(self) -> None:
        """Initialize display."""
        self._mounted = True
        self._add_system_event("Ready")

    def _add_system_event(self, message: str) -> None:
        """Add a system message to the log."""
        event = ActivityEvent(
            type=ActivityType.COMPLETE,
            content=message,
            agent="system",
        )
        self._append_event(event)

    def _append_event(self, event: ActivityEvent) -> None:
        """Append an event and update display."""
        self._events.append(event)

        # Trim old events
        if len(self._events) > self.MAX_EVENTS:
            self._events = self._events[-self.MAX_EVENTS:]

        # Create and mount the log entry
        if self._mounted:
            entry_widget = self._create_entry_widget(event)
            self.mount(entry_widget)
            self.scroll_end(animate=False)

    def _create_entry_widget(self, event: ActivityEvent) -> Static:
        """Create a Static widget for an event."""
        text = Text()

        # Timestamp (compact: just time)
        time_str = event.timestamp.strftime("%H:%M:%S")
        text.append(f"{time_str} ", style="dim")

        # Event-specific formatting
        if event.type == ActivityType.AGENT_START:
            agent_display = event.agent.replace("_", " ")[:12]
            text.append("▶ ", style="cyan")
            text.append(f"{agent_display}", style="bold cyan")

        elif event.type == ActivityType.AGENT_THINKING:
            text.append("💭 ", style="dim")
            content = event.content[:40] + "..." if len(event.content) > 40 else event.content
            text.append(content, style="dim")

        elif event.type == ActivityType.TOOL_START:
            text.append("🔧 ", style="yellow")
            text.append(event.content[:30], style="yellow")

        elif event.type == ActivityType.TOOL_COMPLETE:
            duration_str = ""
            if event.duration_ms:
                if event.duration_ms < 1000:
                    duration_str = f" [{event.duration_ms:.0f}ms]"
                else:
                    duration_str = f" [{event.duration_ms/1000:.1f}s]"
            text.append("✓ ", style="green")
            text.append(f"{event.content[:25]}{duration_str}", style="green")

        elif event.type == ActivityType.TOOL_ERROR:
            text.append("✗ ", style="red")
            text.append(event.content[:35], style="red")

        elif event.type == ActivityType.HANDOFF:
            text.append("🔄 ", style="cyan")
            text.append(event.content[:35], style="cyan")

        elif event.type == ActivityType.COMPLETE:
            text.append("● ", style="green")
            text.append(event.content[:35], style="")

        return Static(text, classes="log-entry")

    # Public API for receiving events

    def log_agent_start(self, agent_name: str) -> None:
        """Log agent starting."""
        self._append_event(ActivityEvent(
            type=ActivityType.AGENT_START,
            content=f"{agent_name} started",
            agent=agent_name,
        ))

    def log_thinking(self, agent_name: str, thought: str) -> None:
        """Log agent thinking/reasoning."""
        self._append_event(ActivityEvent(
            type=ActivityType.AGENT_THINKING,
            content=thought,
            agent=agent_name,
        ))

    def log_tool_start(self, tool_name: str) -> None:
        """Log tool starting."""
        self._append_event(ActivityEvent(
            type=ActivityType.TOOL_START,
            content=tool_name,
        ))

    def log_tool_complete(self, tool_name: str, duration_ms: Optional[float] = None) -> None:
        """Log tool completion."""
        self._append_event(ActivityEvent(
            type=ActivityType.TOOL_COMPLETE,
            content=tool_name,
            duration_ms=duration_ms,
        ))

    def log_tool_error(self, tool_name: str, error: str) -> None:
        """Log tool error."""
        self._append_event(ActivityEvent(
            type=ActivityType.TOOL_ERROR,
            content=f"{tool_name}: {error[:20]}",
        ))

    def log_handoff(self, from_agent: str, to_agent: str, task_description: str = "") -> None:
        """Log agent handoff with optional task description."""
        from_display = from_agent.replace("_", " ")[:10]
        to_display = to_agent.replace("_", " ")[:10]

        # Include task description if provided
        content = f"{from_display} → {to_display}"
        if task_description:
            # Truncate task description to reasonable length for display
            task_preview = task_description[:80] + "..." if len(task_description) > 80 else task_description
            content += f"\n  Task: {task_preview}"

        self._append_event(ActivityEvent(
            type=ActivityType.HANDOFF,
            content=content,
        ))

    def log_complete(self, message: str = "Complete") -> None:
        """Log completion."""
        self._append_event(ActivityEvent(
            type=ActivityType.COMPLETE,
            content=message,
        ))

    def clear_log(self) -> None:
        """Clear all log entries."""
        self._events = []
        # Remove all log-entry widgets
        for child in self.query(".log-entry"):
            child.remove()
        self._add_system_event("Log cleared")
