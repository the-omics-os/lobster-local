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

        # PRIMARY handoff detection via tool names (matches Textual pattern)
        # LangGraph uses: handoff_to_<agent_name> and transfer_back_to_supervisor
        is_handoff = False
        if tool_name.startswith("handoff_to_"):
            is_handoff = True
            target_agent = tool_name.replace("handoff_to_", "")
            self._handle_handoff(target_agent)
        elif tool_name.startswith("transfer_back_to_"):
            is_handoff = True
            self._handle_handoff("supervisor")

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

    def _handle_handoff(self, target_agent: str) -> None:
        """Handle agent handoff detected from tool call (matches Textual pattern)."""
        if target_agent == self.current_agent:
            return  # No change

        from_agent = self.current_agent or "system"

        # Create and display handoff event
        event = AgentEvent(
            type=EventType.HANDOFF,
            agent_name="system",
            content="",
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
            "chatopenai", "chatollama", "llm", "chat"
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
    """

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

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Track current agent context when LLM starts."""
        if serialized is None:
            serialized = {}
        self.current_agent = kwargs.get("name") or serialized.get("name", "unknown")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Track current tool context."""
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
