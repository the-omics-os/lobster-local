"""
Modern Terminal Callback Handler for Multi-Agent System.
Provides clean, informative display of agent reasoning and execution flow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.tree import Tree

from lobster.config.agent_registry import get_all_agent_names
from lobster.utils.error_handlers import ErrorGuidance, get_error_registry


class EventType(Enum):
    """Types of events in the agent system."""

    AGENT_START = "agent_start"
    AGENT_THINKING = "agent_thinking"
    AGENT_ACTION = "agent_action"
    AGENT_COMPLETE = "agent_complete"
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    HANDOFF = "handoff"
    SYSTEM = "system"


@dataclass
class AgentEvent:
    """Represents an event in the agent execution."""

    type: EventType
    agent_name: str
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None


class TerminalCallbackHandler(BaseCallbackHandler):
    """
    Modern Callback handler with Rich terminal display.
    Shows agent reasoning, tool usage, and execution flow in a clean format.
    """

    def __init__(
        self,
        console: Optional[Console] = None,
        verbose: bool = True,
        show_reasoning: bool = True,
        show_tools: bool = True,
        max_length: int = None,
        use_panels: bool = True,
    ):
        """
        Initialize the terminal callback handler.

        Args:
            console: Rich console instance (creates new if None)
            verbose: Show detailed output
            show_reasoning: Display agent reasoning/thinking
            show_tools: Display tool usage
            max_length: Maximum length for content display
            use_panels: Use Rich panels for output
        """
        self.console = console or Console()
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.max_length = max_length
        self.use_panels = use_panels

        # State tracking
        self.current_agent: Optional[str] = None
        self.agent_stack: List[str] = []
        self.events: List[AgentEvent] = []
        self.start_times: Dict[str, datetime] = {}

        # Display components
        self.progress: Optional[Progress] = None
        self.current_task = None

    def _format_agent_name(self, name: str) -> str:
        """Format agent name for display."""
        if not name or name == "unknown":
            return "System"
        return name.replace("_", " ").title()

    def _truncate_content(self, content: str) -> str:
        """Truncate content if too long."""
        if not content:
            return ""
        content = str(content).strip()
        if self.max_length is not None and len(content) > self.max_length:
            return content[: self.max_length] + "..."
        return content

    def _format_duration(self, seconds: float) -> str:
        """Format duration for display."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            return f"{seconds/60:.1f}m"

    def _display_agent_event(self, event: AgentEvent):
        """Display an agent event with Rich formatting."""
        agent_display = self._format_agent_name(event.agent_name)

        if event.type == EventType.AGENT_START:
            if self.use_panels:
                panel = Panel(
                    "[cyan]Starting analysis...[/cyan]",
                    title=f"🤖 {agent_display}",
                    border_style="cyan",
                    box=box.ROUNDED,
                )
                self.console.print(panel)
            else:
                self.console.print(f"\n[cyan]🤖 {agent_display}[/cyan]")

        elif event.type == EventType.AGENT_THINKING and self.show_reasoning:
            if event.content:
                content = self._truncate_content(event.content)
                self.console.print(f"[dim]   💭 {content}[/dim]")

        elif event.type == EventType.AGENT_ACTION:
            if event.content:
                self.console.print(f"[yellow]   ➤ {event.content}[/yellow]")

        elif event.type == EventType.AGENT_COMPLETE:
            duration_str = ""
            if event.duration:
                duration_str = f" [{self._format_duration(event.duration)}]"
            self.console.print(
                f"[green]   ✓ {agent_display} complete{duration_str}[/green]"
            )

        elif event.type == EventType.HANDOFF:
            from_agent = event.metadata.get("from", "Unknown")
            to_agent = event.metadata.get("to", "Unknown")
            self.console.print(
                f"[cyan]🔄 Handoff: {self._format_agent_name(from_agent)} "
                f"→ {self._format_agent_name(to_agent)}[/cyan]"
            )
            if event.content:
                self.console.print(
                    f"[dim]   Task: {self._truncate_content(event.content)}[/dim]"
                )

    def _display_tool_event(self, event: AgentEvent):
        """Display a tool event with Rich formatting."""
        if not self.show_tools:
            return

        tool_name = event.metadata.get("tool_name", "Unknown Tool")

        if event.type == EventType.TOOL_START:
            self.console.print(f"[yellow]   🔧 Using {tool_name}[/yellow]")
            if self.verbose and event.content:
                self.console.print(
                    f"[dim]      Input: {self._truncate_content(event.content)}[/dim]"
                )

        elif event.type == EventType.TOOL_COMPLETE:
            duration_str = ""
            if event.duration:
                duration_str = f" [{self._format_duration(event.duration)}]"
            self.console.print(
                f"[green]      ✓ {tool_name} complete{duration_str}[/green]"
            )
            if self.verbose and event.content:
                self.console.print(
                    f"[dim]      Result: {self._truncate_content(event.content)}[/dim]"
                )

        elif event.type == EventType.TOOL_ERROR:
            self.console.print(
                f"[red]      ✗ {tool_name} failed: {event.content}[/red]"
            )

    # LangChain Callback Methods

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when an LLM starts."""
        # Handle None serialized
        if serialized is None:
            serialized = {}
        agent_name = kwargs.get("name") or serialized.get("name", "unknown")

        # Track start time
        self.start_times[agent_name] = datetime.now()

        # Create and display event
        event = AgentEvent(type=EventType.AGENT_START, agent_name=agent_name)
        self.events.append(event)
        self._display_agent_event(event)

        # Update current agent
        if agent_name != self.current_agent:
            self.current_agent = agent_name
            self.agent_stack.append(agent_name)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when an LLM finishes."""
        if not self.current_agent:
            return

        # Calculate duration
        duration = None
        if self.current_agent in self.start_times:
            duration = (
                datetime.now() - self.start_times[self.current_agent]
            ).total_seconds()
            del self.start_times[self.current_agent]

        # Extract response content
        content = ""
        if response.generations and response.generations[0]:
            content = response.generations[0][0].text

        # Show reasoning if enabled
        if self.show_reasoning and content:
            event = AgentEvent(
                type=EventType.AGENT_THINKING,
                agent_name=self.current_agent,
                content=content,
            )
            self.events.append(event)
            self._display_agent_event(event)

        # Show completion
        event = AgentEvent(
            type=EventType.AGENT_COMPLETE,
            agent_name=self.current_agent,
            duration=duration,
        )
        self.events.append(event)
        self._display_agent_event(event)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Called when an LLM errors - now with modular error handling."""
        # Get error guidance from registry
        registry = get_error_registry()
        guidance = registry.handle_error(error)

        # Format and display based on guidance
        self._display_error_guidance(guidance)

    def _display_error_guidance(self, guidance: ErrorGuidance):
        """
        Display error guidance with rich formatting.

        Args:
            guidance: ErrorGuidance object with structured error information
        """
        # Choose color based on severity
        severity_colors = {
            "warning": "yellow",
            "error": "red",
            "critical": "bold red on white",
        }
        border_color = severity_colors.get(guidance.severity, "red")

        # Build content
        content_lines = [f"[bold white]{guidance.description}[/bold white]", ""]

        # Add solutions
        if guidance.solutions:
            content_lines.append("[bold]Solutions:[/bold]")
            for i, solution in enumerate(guidance.solutions, 1):
                content_lines.append(f"  {i}. {solution}")
            content_lines.append("")

        # Add retry info
        if guidance.can_retry:
            retry_msg = "You can retry this operation"
            if guidance.retry_delay:
                retry_msg += f" (recommended wait: {guidance.retry_delay}s)"
            content_lines.append(f"[dim]{retry_msg}[/dim]")

        # Add documentation link
        if guidance.documentation_url:
            content_lines.append(
                f"[dim]📚 Documentation: {guidance.documentation_url}[/dim]"
            )

        # Add support contact
        if guidance.support_email:
            content_lines.append(f"[dim]📧 Support: {guidance.support_email}[/dim]")

        content = "\n".join(content_lines)

        # Display as panel
        panel = Panel(
            content,
            title=guidance.title,
            border_style=border_color,
            box=box.ROUNDED,
            expand=False,
            padding=(1, 2),
        )
        self.console.print(panel)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts."""
        # Handle None serialized
        if serialized is None:
            serialized = {}
        tool_name = serialized.get("name", "unknown_tool")

        # Track start time
        self.start_times[f"tool_{tool_name}"] = datetime.now()

        event = AgentEvent(
            type=EventType.TOOL_START,
            agent_name=self.current_agent or "system",
            content=input_str,
            metadata={"tool_name": tool_name},
        )
        self.events.append(event)
        self._display_tool_event(event)

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        # Find the tool name from the last tool start event
        tool_name = "unknown_tool"
        for event in reversed(self.events):
            if event.type == EventType.TOOL_START:
                tool_name = event.metadata.get("tool_name", "unknown_tool")
                break

        # Calculate duration
        duration = None
        tool_key = f"tool_{tool_name}"
        if tool_key in self.start_times:
            duration = (datetime.now() - self.start_times[tool_key]).total_seconds()
            del self.start_times[tool_key]

        event = AgentEvent(
            type=EventType.TOOL_COMPLETE,
            agent_name=self.current_agent or "system",
            content=output,
            metadata={"tool_name": tool_name},
            duration=duration,
        )
        self.events.append(event)
        self._display_tool_event(event)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Called when a tool errors."""
        event = AgentEvent(
            type=EventType.TOOL_ERROR,
            agent_name=self.current_agent or "system",
            content=str(error),
            metadata={"tool_name": "unknown_tool"},
        )
        self.events.append(event)
        self._display_tool_event(event)

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when an agent takes an action."""
        event = AgentEvent(
            type=EventType.AGENT_ACTION,
            agent_name=self.current_agent or "system",
            content=f"Action: {action.tool} with input: {self._truncate_content(str(action.tool_input))}",
        )
        self.events.append(event)
        self._display_agent_event(event)

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when an agent finishes."""
        if self.agent_stack:
            self.agent_stack.pop()
            self.current_agent = self.agent_stack[-1] if self.agent_stack else None

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Called when a chain starts."""
        # Handle None values
        if serialized is None:
            serialized = {}
        if inputs is None:
            inputs = {}

        chain_name = serialized.get("name", "")

        # Detect agent transitions using the agent registry
        agent_names = get_all_agent_names()
        for agent_name in agent_names:
            if agent_name in chain_name.lower():
                if agent_name != self.current_agent:
                    # This is a handoff
                    event = AgentEvent(
                        type=EventType.HANDOFF,
                        agent_name="system",
                        content=inputs.get("task", ""),
                        metadata={
                            "from": self.current_agent or "system",
                            "to": agent_name,
                        },
                    )
                    self.events.append(event)
                    self._display_agent_event(event)
                break

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes."""
        pass

    # Custom methods for graph events

    def on_graph_step(self, node_name: str, state: Dict[str, Any]):
        """Called when the graph executes a step."""
        formatted_name = self._format_agent_name(node_name)

        # Create a tree view of the state
        if self.verbose and state:
            tree = Tree(f"[bold cyan]{formatted_name} State[/bold cyan]")
            for key, value in state.items():
                if key != "messages":  # Skip messages as they're usually too long
                    value_str = self._truncate_content(str(value))
                    tree.add(f"[yellow]{key}[/yellow]: {value_str}")

            if self.use_panels:
                self.console.print(Panel(tree, border_style="dim"))
            else:
                self.console.print(tree)

    def display_summary(self):
        """Display a summary of the execution."""
        if not self.events:
            return

        # Create summary table
        table = Table(title="Execution Summary", box=box.ROUNDED)
        table.add_column("Agent", style="cyan")
        table.add_column("Actions", style="yellow")
        table.add_column("Tools Used", style="magenta")
        table.add_column("Total Time", style="green")

        # Aggregate data
        agent_stats = {}
        for event in self.events:
            agent = event.agent_name
            if agent not in agent_stats:
                agent_stats[agent] = {"actions": 0, "tools": set(), "duration": 0}

            if event.type == EventType.AGENT_ACTION:
                agent_stats[agent]["actions"] += 1
            elif event.type == EventType.TOOL_START:
                tool_name = event.metadata.get("tool_name", "unknown")
                agent_stats[agent]["tools"].add(tool_name)

            if event.duration:
                agent_stats[agent]["duration"] += event.duration

        # Add rows to table
        for agent, stats in agent_stats.items():
            table.add_row(
                self._format_agent_name(agent),
                str(stats["actions"]),
                ", ".join(stats["tools"]) if stats["tools"] else "None",
                (
                    self._format_duration(stats["duration"])
                    if stats["duration"] > 0
                    else "N/A"
                ),
            )

        self.console.print("\n")
        self.console.print(table)

    def reset(self):
        """Reset the callback handler state."""
        self.current_agent = None
        self.agent_stack = []
        self.events = []
        self.start_times = {}
        self.current_task = None


class StreamingTerminalCallback(TerminalCallbackHandler):
    """
    Streaming version that updates display in real-time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live_display: Optional[Live] = None
        self.layout = Layout()

    def start_streaming(self):
        """Start live streaming display."""
        self.live_display = Live(
            self.layout, console=self.console, refresh_per_second=4, screen=False
        )
        self.live_display.start()

    def stop_streaming(self):
        """Stop live streaming display."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None

    def update_display(self, content: str):
        """Update the live display with new content."""
        if self.live_display:
            panel = Panel(
                content,
                title=f"🤖 {self._format_agent_name(self.current_agent)}",
                border_style="cyan",
            )
            self.layout.update(panel)


# Integration with the new agent_cli SimpleCallback
class SimpleTerminalCallback(TerminalCallbackHandler):
    """
    Simplified callback specifically for the agent_cli integration.
    Inherits from TerminalCallbackHandler but provides simpler interface.
    """

    def __init__(self, console: Console, show_reasoning: bool = True):
        super().__init__(
            console=console,
            verbose=False,  # Less verbose for CLI
            show_reasoning=show_reasoning,
            show_tools=True,
            use_panels=False,  # Simpler output for CLI
        )

    def on_agent_start(self, agent_name: str):
        """Simplified agent start notification."""
        event = AgentEvent(type=EventType.AGENT_START, agent_name=agent_name)
        self.events.append(event)
        formatted = self._format_agent_name(agent_name)
        self.console.print(f"\n[cyan]🤖 {formatted} is thinking...[/cyan]")

    def on_agent_reasoning(self, reasoning: str):
        """Display agent reasoning."""
        if self.show_reasoning and reasoning:
            self.console.print(f"[dim]   💭 {self._truncate_content(reasoning)}[/dim]")

    def on_tool_use(self, tool_name: str, tool_input: Any):
        """Display tool usage."""
        self.console.print(f"[yellow]   🔧 Using {tool_name}[/yellow]")
        if self.verbose:
            self.console.print(
                f"[dim]      {self._truncate_content(str(tool_input))}[/dim]"
            )

    def on_agent_complete(self, agent_name: str):
        """Simplified agent completion."""
        formatted = self._format_agent_name(agent_name)
        self.console.print(f"[green]   ✓ {formatted} completed[/green]")
