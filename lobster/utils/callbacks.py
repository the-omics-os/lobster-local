"""
Modern Terminal Callback Handler for Multi-Agent System.
Provides clean, informative display of agent reasoning and execution flow.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
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
        minimal: bool = False,
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
            minimal: Use minimal oh-my-zsh style output (◀ Agent Name)
        """
        self.console = console or Console()
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.max_length = max_length
        self.use_panels = use_panels
        self.minimal = minimal

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

        # Minimal mode: oh-my-zsh style (◀ Agent Name)
        if self.minimal:
            if event.type == EventType.AGENT_START:
                self.console.print(f"[dim]◀ {agent_display}[/dim]")
            elif event.type == EventType.AGENT_THINKING and self.show_reasoning:
                # In minimal mode, only show agent indicator once (on start)
                pass
            elif event.type == EventType.HANDOFF:
                to_agent = event.metadata.get("to", "Unknown")
                self.console.print(f"[dim]◀ {self._format_agent_name(to_agent)}[/dim]")
            # Skip other events in minimal mode
            return

        # Standard verbose mode (existing behavior)
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
            # Minimal oh-my-zsh style tool indicator
            self.console.print(f"[dim]  → {tool_name}[/dim]")
            if self.verbose and event.content:
                self.console.print(
                    f"[dim]      Input: {self._truncate_content(event.content)}[/dim]"
                )

        elif event.type == EventType.TOOL_COMPLETE:
            # Silent completion - only show start
            pass
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
        """
        Called when an LLM starts (completion models like GPT-3 davinci).

        NOTE: This only tracks run timing for the current agent.
        Agent switching is handled ONLY by _handle_handoff via handoff_to_* tools.
        We do NOT display events or change current_agent based on LLM model names.
        """
        # Only track start time for current agent if set
        if self.current_agent:
            self.start_times[self.current_agent] = datetime.now()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Called when an LLM finishes.

        NOTE: We only show reasoning content, NOT AGENT_COMPLETE events.
        Agent completion is implicit when the next handoff occurs or query ends.
        This matches the Textual callback behavior.
        """
        if not self.current_agent:
            return

        # Calculate duration (for timing tracking only)
        if self.current_agent in self.start_times:
            del self.start_times[self.current_agent]

        # Extract response content for reasoning display
        content = ""
        if response.generations and response.generations[0]:
            content = response.generations[0][0].text

        # Show reasoning/thinking if enabled (the actual agent thoughts)
        if self.show_reasoning and content:
            event = AgentEvent(
                type=EventType.AGENT_THINKING,
                agent_name=self.current_agent,
                content=content,
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

    def _extract_task_description(
        self,
        input_str: str,
        inputs: Dict[str, Any] | None = None
    ) -> str:
        """
        Extract task_description from tool input.

        Uses structured `inputs` dict (preferred) with fallback to `input_str`
        for simple string-only tools. Never parses input_str as JSON since
        it's a Python repr string, not JSON.

        Args:
            input_str: String representation of tool input (display only)
            inputs: Structured tool input dict (with injected args filtered out)

        Returns:
            Extracted task description or empty string if not found

        Examples:
            # Case 1: Simple string tool (graph.py delegation)
            input_str = "Download GSE109564"
            inputs = None
            → Returns: "Download GSE109564"

            # Case 2: Multiple params (handoff_tool.py)
            input_str = "{'task_description': 'Download GSE109564', ...}"
            inputs = {'task_description': 'Download GSE109564'}
            → Returns: "Download GSE109564"
        """
        # Strategy 1: Use structured inputs dict (preferred)
        if inputs and isinstance(inputs, dict):
            # Look for explicit 'task_description' field
            if "task_description" in inputs:
                return inputs["task_description"]

            # For tools with single parameter but different name
            # (e.g., 'task', 'description', 'query')
            if len(inputs) == 1:
                return next(iter(inputs.values()), "")

        # Strategy 2: Fallback for simple string-only tools
        # (When tool has signature: def tool(task_description: str))
        if not inputs:
            # input_str IS the raw parameter value (no parsing needed)
            return input_str if input_str else ""

        # Strategy 3: If we have inputs but no recognizable field, return empty
        return ""

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        inputs: Dict[str, Any] | None = None,
        **kwargs
    ) -> None:
        """Called when a tool starts."""
        # Handle None serialized
        if serialized is None:
            serialized = {}
        tool_name = serialized.get("name", "unknown_tool")

        # Track start time
        self.start_times[f"tool_{tool_name}"] = datetime.now()

        # PRIMARY handoff detection via tool names (matches Textual pattern)
        # LangGraph uses: handoff_to_<agent_name> and transfer_back_to_supervisor
        is_handoff = False
        if tool_name.startswith("handoff_to_"):
            is_handoff = True
            target_agent = tool_name.replace("handoff_to_", "")
            # Extract task description from tool input
            task_description = self._extract_task_description(input_str, inputs)
            self._handle_handoff(target_agent, task_description)
        elif tool_name.startswith("transfer_back_to_"):
            is_handoff = True
            task_description = self._extract_task_description(input_str, inputs)
            self._handle_handoff("supervisor", task_description)

        # Only show non-handoff tools in regular tool event
        if not is_handoff:
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

        # Skip logging handoff tool completions (already handled in on_tool_start)
        if tool_name.startswith("handoff_to_") or tool_name.startswith("transfer_back_to_"):
            return

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

    def _handle_handoff(self, target_agent: str, task_description: str = "") -> None:
        """Handle agent handoff detected from tool call (matches Textual pattern).

        Uses hierarchy validation to correct the from_agent during parallel tool calls.
        When multiple agents are called in parallel by the supervisor, the current_agent
        state gets corrupted. Instead of complex run_id tracking, we validate the
        handoff against the known agent hierarchy and correct impossible transitions.

        Args:
            target_agent: The agent being handed off to
            task_description: Description of the task being delegated
        """
        if target_agent == self.current_agent:
            return  # No change

        # Get the supposed from_agent from current state (may be wrong during parallel calls)
        supposed_from_agent = self.current_agent or "system"

        # Validate and correct using agent hierarchy
        # If the handoff is impossible (e.g., data_expert → transcriptomics_expert),
        # correct it to supervisor (the only agent that can reach anywhere)
        from lobster.config.agent_registry import is_valid_handoff

        if is_valid_handoff(supposed_from_agent, target_agent):
            from_agent = supposed_from_agent
        else:
            # Impossible handoff detected - must be supervisor calling in parallel
            from_agent = "supervisor"

        # Create and display handoff event with task description
        event = AgentEvent(
            type=EventType.HANDOFF,
            agent_name="system",
            content=task_description,  # Include task description
            metadata={
                "from": from_agent,
                "to": target_agent,
            },
        )
        self.events.append(event)
        self._display_agent_event(event)

        # Update current agent tracking
        self.current_agent = target_agent
        if target_agent not in self.agent_stack:
            self.agent_stack.append(target_agent)

        # Track start time for new agent
        self.start_times[target_agent] = datetime.now()

        # Show agent start event
        start_event = AgentEvent(
            type=EventType.AGENT_START,
            agent_name=target_agent,
        )
        self.events.append(start_event)
        self._display_agent_event(start_event)

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
        """
        Called when a chain starts - BACKUP method for agent detection.

        PRIMARY detection happens in on_tool_start via handoff_to_* tools.
        This provides backup detection using the agent registry for cases
        where chains are named after agents.

        NOTE: LangGraph often sends empty chain names for LLM chains.
        We filter these out to avoid noise.
        """
        # Handle None values
        if serialized is None:
            serialized = {}
        if inputs is None:
            inputs = {}

        chain_name = serialized.get("name", "")

        # Skip empty chain names (often LLM chains in LangGraph)
        if not chain_name:
            return

        # Filter out known LLM model class names to avoid confusion
        llm_model_names = [
            "chatbedrock", "chatbedrockconverse", "chatanthropic",
            "chatollama", "llm", "chat"
        ]
        chain_name_lower = chain_name.lower()
        if any(model in chain_name_lower for model in llm_model_names):
            return

        # Detect agent transitions using the agent registry (BACKUP only)
        agent_names = get_all_agent_names()
        for agent_name in agent_names:
            if agent_name in chain_name_lower:
                if agent_name != self.current_agent:
                    self._handle_handoff(agent_name)
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

    def __init__(self, console: Console, show_reasoning: bool = True, minimal: bool = False):
        super().__init__(
            console=console,
            verbose=False,  # Less verbose for CLI
            show_reasoning=show_reasoning,
            show_tools=True,
            use_panels=False,  # Simpler output for CLI
            minimal=minimal,  # oh-my-zsh style output
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
        # Minimal oh-my-zsh style tool indicator
        self.console.print(f"[dim]  → {tool_name}[/dim]")
        if self.verbose:
            self.console.print(
                f"[dim]      {self._truncate_content(str(tool_input))}[/dim]"
            )

    def on_agent_complete(self, agent_name: str):
        """Simplified agent completion."""
        # Silent - agent indicators shown at response time only
        pass


@dataclass
class TokenInvocation:
    """Represents a single LLM invocation with token usage."""

    timestamp: str
    agent: str
    model: str
    tool: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class TokenTrackingCallback(BaseCallbackHandler):
    """
    Callback handler for tracking token usage and costs across all LLM invocations.

    This callback automatically extracts token usage from LLMResult objects,
    calculates costs based on model pricing, and aggregates data per agent/session.
    Works with both AWS Bedrock and Anthropic Direct API providers.

    Robust Agent Detection:
    - Multi-source detection: run_name, tags, metadata, registry validation
    - Filters out model class names (ChatBedrock, ChatAnthropic, etc)
    - Maintains agent state across invocations
    - Tracks run_id hierarchy for nested agent calls
    """

    # Known LLM model class names (case-insensitive) to filter out
    MODEL_CLASS_NAMES = frozenset([
        "chatbedrock",
        "chatbedrockconverse",
        "chatanthropic",
        "chatollama",
        "llm",
        "chat",
        "chatmodel",
        "completionmodel",
        "bedrockllm",
    ])

    def __init__(
        self,
        session_id: str,
        pricing_config: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize token tracking callback.

        Args:
            session_id: Unique session identifier
            pricing_config: Model pricing configuration (input/output per million tokens)
        """
        self.session_id = session_id
        self.pricing_config = pricing_config or {}

        # Aggregated totals
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0

        # Per-agent aggregation
        self.by_agent: Dict[str, Dict[str, Any]] = {}

        # Detailed invocation log
        self.invocations: List[TokenInvocation] = []

        # Track current context
        self.current_agent: Optional[str] = None
        self.current_tool: Optional[str] = None

        # Run ID tracking for hierarchy (prevents stale state)
        self.run_to_agent: Dict[str, str] = {}  # run_id -> agent_name
        self.current_run_id: Optional[str] = None

        # Cache valid agent names from registry (lazy-loaded)
        self._valid_agents: Optional[set] = None

    # =================================================================
    # AGENT NAME DETECTION (Core Logic)
    # =================================================================

    def _get_valid_agents(self) -> set:
        """Lazy-load and cache valid agent names from registry."""
        if self._valid_agents is None:
            self._valid_agents = set(get_all_agent_names())
            # Add common system agents not in registry
            self._valid_agents.update(["supervisor", "system"])
        return self._valid_agents

    def _is_model_class(self, name: str) -> bool:
        """Check if name is a known LLM model class (not an agent)."""
        if not name:
            return False
        return name.lower() in self.MODEL_CLASS_NAMES

    def _is_valid_agent(self, name: str) -> bool:
        """
        Validate that name is an actual agent, not a model class.

        Returns:
            True if name is in AGENT_REGISTRY or known system agent
            False if name is a model class or invalid
        """
        if not name or self._is_model_class(name):
            return False

        # Check against agent registry
        valid_agents = self._get_valid_agents()
        return name in valid_agents

    def _extract_agent_name(
        self,
        serialized: Dict[str, Any],
        kwargs: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract agent name from multiple sources with validation.

        Priority order:
        1. run_name from config (explicit, most reliable)
        2. tags[0] from config (secondary indicator)
        3. metadata["agent_name"] (if present)
        4. kwargs["name"] (if validated)
        5. serialized["name"] (last resort, rarely useful)

        Args:
            serialized: LangChain serialized model info
            kwargs: Callback kwargs (contains run_name, tags, metadata)

        Returns:
            Valid agent name or None if not found
        """
        # Priority 1: Explicit run_name (set in graph.py:68)
        agent_name = kwargs.get("run_name")
        if agent_name and self._is_valid_agent(agent_name):
            return agent_name

        # Priority 2: Tags (first tag is often agent name)
        tags = kwargs.get("tags", [])
        if tags and isinstance(tags, list) and len(tags) > 0:
            candidate = tags[0]
            if self._is_valid_agent(candidate):
                return candidate

        # Priority 3: Metadata (explicit agent field)
        metadata = kwargs.get("metadata", {})
        if isinstance(metadata, dict):
            # Try multiple metadata keys
            for key in ["agent_name", "agent", "run_name"]:
                agent_name = metadata.get(key)
                if agent_name and self._is_valid_agent(agent_name):
                    return agent_name

        # Priority 4: kwargs["name"] (validate it's not a model class)
        agent_name = kwargs.get("name")
        if agent_name and self._is_valid_agent(agent_name):
            return agent_name

        # Priority 5: serialized["name"] (very unreliable, usually model class)
        if serialized:
            agent_name = serialized.get("name")
            if agent_name and self._is_valid_agent(agent_name):
                return agent_name

        # No valid agent found
        return None

    def _update_current_agent(
        self,
        detected_agent: Optional[str],
        run_id: Optional[str] = None
    ) -> None:
        """
        Update current_agent with fallback logic.

        Strategy:
        - If detected agent is valid, use it
        - If run_id is known, use mapped agent
        - Otherwise, keep current_agent unchanged (maintain state)

        Args:
            detected_agent: Agent name from detection
            run_id: Current run ID (for hierarchy tracking)
        """
        # Update run_id mapping if we have both
        if run_id and detected_agent:
            self.run_to_agent[str(run_id)] = detected_agent
            self.current_run_id = str(run_id)

        # Fallback chain
        if detected_agent:
            self.current_agent = detected_agent
        elif run_id and str(run_id) in self.run_to_agent:
            # Use run_id to recover agent from hierarchy
            self.current_agent = self.run_to_agent[str(run_id)]
        # else: keep current_agent unchanged (don't set to "unknown")

    # =================================================================
    # CALLBACK METHODS (LangChain Integration)
    # =================================================================

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """
        Called when a traditional completion LLM starts.

        NOTE: Chat models (Claude, GPT-4) use on_chat_model_start instead.
        This method is for legacy completion models (GPT-3 davinci, etc).
        """
        run_id = kwargs.get("run_id")
        detected_agent = self._extract_agent_name(serialized, kwargs)
        self._update_current_agent(detected_agent, run_id)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Called when a chat model starts (Claude, GPT-4, Bedrock models).

        CRITICAL: This is the PRIMARY callback for Claude/Anthropic models.
        on_llm_start is NOT called for chat models.

        This method:
        1. Detects agent name from multiple sources
        2. Validates against AGENT_REGISTRY
        3. Tracks run_id hierarchy
        4. Maintains current_agent state
        """
        # Merge explicit args into kwargs for unified detection
        kwargs_merged = {
            **kwargs,
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "tags": tags or [],
            "metadata": metadata or {},
        }

        # DEBUG: Log what we receive (only in debug mode)
        import os
        if os.getenv("LOBSTER_LOG_LEVEL") == "DEBUG":
            from lobster.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.debug(f"on_chat_model_start: tags={tags}, metadata={metadata}, run_name={kwargs.get('run_name')}")

        detected_agent = self._extract_agent_name(serialized, kwargs_merged)
        self._update_current_agent(detected_agent, run_id)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """
        Track current tool context.

        IMPORTANT: This is where agent handoffs are detected in the main
        callback handlers (via handoff_to_* tool names). We don't track
        agent switches here to avoid duplication, but we do track the tool
        for attribution in invocation logs.
        """
        if serialized is None:
            serialized = {}
        self.current_tool = serialized.get("name", "unknown_tool")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Clear tool context when tool completes."""
        self.current_tool = None

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """
        Extract token usage from LLMResult and update tracking.

        Handles multiple provider formats:
        - Anthropic Direct API: llm_output["usage"]["input_tokens"]
        - AWS Bedrock: llm_output["usage"] or response_metadata
        - Standard LangChain: llm_output["token_usage"]["prompt_tokens"]
        """
        # Extract token usage from response
        usage = self._extract_token_usage(response)
        if not usage:
            return  # No token data available, skip silently

        # Extract model name
        model = self._extract_model_name(response)

        # Calculate cost
        cost = self._calculate_cost(
            model, usage["input_tokens"], usage["output_tokens"]
        )

        # Create invocation record
        invocation = TokenInvocation(
            timestamp=datetime.now().isoformat(),
            agent=self.current_agent or "unknown",
            model=model,
            tool=self.current_tool,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cost_usd=cost,
        )
        self.invocations.append(invocation)

        # Update session totals
        self.total_input_tokens += usage["input_tokens"]
        self.total_output_tokens += usage["output_tokens"]
        self.total_tokens += usage["total_tokens"]
        self.total_cost_usd += cost

        # Update per-agent aggregation
        agent_name = self.current_agent or "unknown"
        if agent_name not in self.by_agent:
            self.by_agent[agent_name] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "invocation_count": 0,
            }

        self.by_agent[agent_name]["input_tokens"] += usage["input_tokens"]
        self.by_agent[agent_name]["output_tokens"] += usage["output_tokens"]
        self.by_agent[agent_name]["total_tokens"] += usage["total_tokens"]
        self.by_agent[agent_name]["cost_usd"] += cost
        self.by_agent[agent_name]["invocation_count"] += 1

    def _extract_token_usage(self, response: LLMResult) -> Optional[Dict[str, int]]:
        """
        Extract token usage from LLMResult, handling different provider formats.

        Args:
            response: LLMResult object from LangChain

        Returns:
            Dict with input_tokens, output_tokens, total_tokens or None if not found
        """
        # Try newest LangChain format: usage_metadata in generations (LangChain 0.3+)
        # This is the primary location for newer Bedrock/Anthropic integrations
        if response.generations and len(response.generations) > 0:
            first_generation_list = response.generations[0]
            if first_generation_list and len(first_generation_list) > 0:
                generation = first_generation_list[0]

                # Check for usage_metadata on the generation's message
                if hasattr(generation, "message") and hasattr(
                    generation.message, "usage_metadata"
                ):
                    usage_metadata = generation.message.usage_metadata
                    if isinstance(usage_metadata, dict):
                        if (
                            "input_tokens" in usage_metadata
                            and "output_tokens" in usage_metadata
                        ):
                            return {
                                "input_tokens": usage_metadata.get("input_tokens", 0),
                                "output_tokens": usage_metadata.get("output_tokens", 0),
                                "total_tokens": usage_metadata.get(
                                    "total_tokens",
                                    usage_metadata.get("input_tokens", 0)
                                    + usage_metadata.get("output_tokens", 0),
                                ),
                            }

        llm_output = response.llm_output or {}

        # Try Anthropic format (input_tokens, output_tokens)
        if "usage" in llm_output:
            usage = llm_output["usage"]
            if "input_tokens" in usage and "output_tokens" in usage:
                return {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                }

        # Try standard LangChain format (prompt_tokens, completion_tokens)
        if "token_usage" in llm_output:
            usage = llm_output["token_usage"]
            return {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # Try response metadata (Bedrock alternative)
        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "usage" in metadata:
                usage = metadata["usage"]
                if "input_tokens" in usage and "output_tokens" in usage:
                    return {
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0),
                    }

        # No token data found
        return None

    def _extract_model_name(self, response: LLMResult) -> str:
        """Extract model name from LLMResult."""
        # Try newest LangChain format: model in generation message response_metadata
        if response.generations and len(response.generations) > 0:
            first_generation_list = response.generations[0]
            if first_generation_list and len(first_generation_list) > 0:
                generation = first_generation_list[0]

                # Check for model in message's response_metadata (LangChain 0.3+)
                if hasattr(generation, "message") and hasattr(
                    generation.message, "response_metadata"
                ):
                    metadata = generation.message.response_metadata
                    if isinstance(metadata, dict):
                        if "model_name" in metadata:
                            return metadata["model_name"]
                        if "model_id" in metadata:
                            return metadata["model_id"]

        llm_output = response.llm_output or {}

        # Try multiple common fields in llm_output
        if "model_name" in llm_output:
            return llm_output["model_name"]
        if "model_id" in llm_output:
            return llm_output["model_id"]
        if "model" in llm_output:
            return llm_output["model"]

        # Try response metadata at response level
        if hasattr(response, "response_metadata") and response.response_metadata:
            metadata = response.response_metadata
            if "model_name" in metadata:
                return metadata["model_name"]
            if "model_id" in metadata:
                return metadata["model_id"]

        return "unknown"

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """
        Calculate cost based on model pricing.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        if model not in self.pricing_config:
            return 0.0

        pricing = self.pricing_config[model]
        input_cost = (input_tokens / 1_000_000) * pricing.get("input_per_million", 0.0)
        output_cost = (output_tokens / 1_000_000) * pricing.get(
            "output_per_million", 0.0
        )
        return input_cost + output_cost

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage summary.

        Returns:
            Dict with session totals, per-agent breakdown, and invocation log
        """
        return {
            "session_id": self.session_id,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "by_agent": {
                agent: {
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_tokens": stats["total_tokens"],
                    "cost_usd": round(stats["cost_usd"], 4),
                    "invocation_count": stats["invocation_count"],
                }
                for agent, stats in self.by_agent.items()
            },
            "invocations": [
                {
                    "timestamp": inv.timestamp,
                    "agent": inv.agent,
                    "model": inv.model,
                    "tool": inv.tool,
                    "input_tokens": inv.input_tokens,
                    "output_tokens": inv.output_tokens,
                    "total_tokens": inv.total_tokens,
                    "cost_usd": round(inv.cost_usd, 4),
                }
                for inv in self.invocations
            ],
        }

    def get_latest_cost(self) -> Dict[str, Any]:
        """
        Get cost info from the most recent invocation.

        Returns:
            Dict with latest cost and session total
        """
        latest_cost = 0.0
        if self.invocations:
            latest_cost = self.invocations[-1].cost_usd

        return {
            "latest_cost_usd": round(latest_cost, 4),
            "session_total_usd": round(self.total_cost_usd, 4),
            "total_tokens": self.total_tokens,
        }

    def get_minimal_summary(self) -> str:
        """
        Get minimal one-line token usage summary with compact agent abbreviations.

        Returns:
            Compact string with total and per-agent breakdown
            Format: "💰 10.4k tokens ($0.03) | S(2.3k) | R(6.6k) | D(1.5k)"

        Agent abbreviations:
            S = supervisor
            R = research_agent
            D = data_expert_agent
            T = transcriptomics_expert
            V = visualization_expert_agent
            M = machine_learning_expert_agent
            P = proteomics_expert
            A = annotation_expert
            E = de_analysis_expert
            U = unknown
        """
        if not self.invocations:
            return ""

        def format_tokens(count: int) -> str:
            """Format token count with k suffix for readability."""
            if count >= 1000:
                return f"{count / 1000:.1f}k"
            return str(count)

        def agent_abbreviation(agent_name: str) -> str:
            """Get single-letter abbreviation for agent."""
            mapping = {
                "supervisor": "S",
                "research_agent": "R",
                "data_expert_agent": "D",
                "transcriptomics_expert": "T",
                "visualization_expert_agent": "V",
                "machine_learning_expert_agent": "M",
                "proteomics_expert": "P",
                "annotation_expert": "A",
                "de_analysis_expert": "E",
                "metadata_assistant": "Meta",
                "protein_structure_visualization_expert_agent": "Prot",
                "custom_feature_agent": "Custom",
                "unknown": "U",
            }
            return mapping.get(agent_name, agent_name[0].upper())

        # Build per-agent breakdown with abbreviations
        agent_parts = []
        for agent_name, stats in sorted(self.by_agent.items()):
            abbrev = agent_abbreviation(agent_name)
            tokens_str = format_tokens(stats["total_tokens"])
            agent_parts.append(f"{abbrev}({tokens_str})")

        # Combine into single line
        total_str = format_tokens(self.total_tokens)
        cost_str = f"${self.total_cost_usd:.4f}".rstrip("0").rstrip(".")
        if cost_str == "$0.":
            cost_str = "$0.00"

        breakdown = " | ".join(agent_parts) if agent_parts else "U(0)"
        return f"💰 {total_str} tokens ({cost_str}) | {breakdown}"

    def save_to_workspace(self, workspace_path: Path):
        """
        Save token usage data to workspace.

        Args:
            workspace_path: Path to session workspace directory
        """
        import json

        # Ensure workspace directory exists
        workspace_path.mkdir(parents=True, exist_ok=True)

        usage_file = workspace_path / "token_usage.json"
        with open(usage_file, "w") as f:
            json.dump(self.get_usage_summary(), f, indent=2)

    def reset(self):
        """Reset all tracking data."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.by_agent = {}
        self.invocations = []
        self.current_agent = None
        self.current_tool = None
        # Reset new tracking state
        self.run_to_agent = {}
        self.current_run_id = None
        self._valid_agents = None  # Re-fetch on next use
