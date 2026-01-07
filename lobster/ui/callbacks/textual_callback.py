"""
Textual-aware callback handler for bridging LangChain events to UI.

This callback handler directly updates UI widgets for reliable activity display.
Uses call_from_thread for thread-safe widget updates (no message bubbling needed).

CRITICAL: Claude/Anthropic models use chat model callbacks, not LLM callbacks.
- on_llm_start → Traditional completion models (GPT-3 davinci)
- on_chat_model_start → Chat models (Claude, GPT-4, etc.)

We implement BOTH to ensure we capture all agent activity.

Agent Detection Strategy:
- on_tool_start detects handoffs via tool names (handoff_to_*, transfer_back_to_*)
- on_chain_start provides backup detection using agent registry
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from lobster.config.agent_registry import get_all_agent_names
from lobster.utils.callbacks import EventType


class TextualCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that directly updates Textual UI widgets.

    This handler tracks:
    - Agent starts/completions
    - Tool starts/completions
    - Agent handoffs
    - Token usage

    IMPORTANT: Pass direct widget references instead of querying for them.
    This avoids race conditions where widgets aren't mounted yet when
    callbacks fire from background threads.

    Usage:
        activity_log = screen.query_one(ActivityLogPanel)
        agents_panel = screen.query_one(AgentsPanel)
        handler = TextualCallbackHandler(
            app=app,
            activity_log=activity_log,
            agents_panel=agents_panel,
        )
        client = AgentClient(custom_callbacks=[handler])
    """

    def __init__(
        self,
        app=None,
        activity_log=None,
        agents_panel=None,
        token_panel=None,
        show_reasoning: bool = False,
        show_tools: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the Textual callback handler.

        Args:
            app: Textual App instance for call_from_thread
            activity_log: Direct reference to ActivityLogPanel widget
            agents_panel: Direct reference to AgentsPanel widget
            token_panel: Direct reference to TokenUsagePanel widget
            show_reasoning: Whether to show agent thinking events
            show_tools: Whether to show tool usage events
            debug: Enable debug notifications to see all callback events
        """
        self.app = app
        self.activity_log = activity_log
        self.agents_panel = agents_panel
        self.token_panel = token_panel
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.debug = debug

        # State tracking
        self.current_agent: Optional[str] = None
        self.agent_stack: List[str] = []
        self.start_times: Dict[str, datetime] = {}
        self.current_tool: Optional[str] = None

        # Run ID tracking for agent hierarchy
        self.run_to_agent: Dict[str, str] = {}
        self.current_run_id: Optional[str] = None

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def _debug_notify(self, message: str) -> None:
        """Show debug notification if debug mode is enabled."""
        if self.debug and self.app:
            try:
                self.app.call_from_thread(
                    self.app.notify,
                    message[:150],
                    timeout=2,
                )
            except Exception:
                pass

    def _update_activity_log(self, method: str, **kwargs) -> None:
        """
        Update ActivityLogPanel widget using direct reference.

        Uses call_from_thread for thread-safe UI updates.
        """
        if not self.activity_log or not self.app:
            return

        def _do_update():
            try:
                if method == "agent_start":
                    self.activity_log.log_agent_start(kwargs.get("agent_name", ""))
                elif method == "thinking":
                    self.activity_log.log_thinking(
                        kwargs.get("agent_name", ""),
                        kwargs.get("thought", ""),
                    )
                elif method == "tool_start":
                    self.activity_log.log_tool_start(kwargs.get("tool_name", ""))
                elif method == "tool_complete":
                    self.activity_log.log_tool_complete(
                        kwargs.get("tool_name", ""),
                        kwargs.get("duration_ms"),
                    )
                elif method == "tool_error":
                    self.activity_log.log_tool_error(
                        kwargs.get("tool_name", ""),
                        kwargs.get("error", ""),
                    )
                elif method == "handoff":
                    self.activity_log.log_handoff(
                        kwargs.get("from_agent", ""),
                        kwargs.get("to_agent", ""),
                        kwargs.get("task_description", ""),
                    )
                elif method == "complete":
                    self.activity_log.log_complete(kwargs.get("message", "Complete"))
            except Exception as e:
                if self.debug:
                    self._debug_notify(f"Activity error: {str(e)[:60]}")

        try:
            self.app.call_from_thread(_do_update)
        except Exception:
            pass

    def _update_agents_panel(self, agent_name: str, active: bool) -> None:
        """Update AgentsPanel widget using direct reference."""
        if not self.agents_panel or not self.app:
            return

        def _do_update():
            try:
                if active:
                    self.agents_panel.set_agent_active(agent_name)
                else:
                    self.agents_panel.set_agent_idle(agent_name)
            except Exception:
                pass

        try:
            self.app.call_from_thread(_do_update)
        except Exception:
            pass

    def _update_token_panel(
        self,
        input_tokens: int,
        output_tokens: int,
        agent_name: str = "",
    ) -> None:
        """Update TokenUsagePanel widget using direct reference."""
        if not self.token_panel or not self.app:
            return

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens = self.total_input_tokens + self.total_output_tokens

        # Calculate cost using MODEL_PRICING
        try:
            from lobster.config.settings import calculate_token_cost

            self.total_cost = calculate_token_cost(
                input_tokens=self.total_input_tokens,
                output_tokens=self.total_output_tokens,
            )
        except Exception:
            self.total_cost = 0.0

        def _do_update():
            try:
                self.token_panel.update_usage(
                    input_tokens=self.total_input_tokens,
                    output_tokens=self.total_output_tokens,
                    total_tokens=self.total_tokens,
                    cost_usd=self.total_cost,
                )
            except Exception:
                pass

        try:
            self.app.call_from_thread(_do_update)
        except Exception:
            pass

    # LangChain Callback Methods

    def _handle_model_start(
        self,
        serialized: Dict[str, Any],
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ) -> None:
        """
        Common handler for both LLM and chat model starts.

        NOTE: This method only tracks run_ids for hierarchy.
        Agent detection happens via tool names (handoff_to_*).
        """
        if serialized is None:
            serialized = {}

        # Track run_id -> current_agent mapping
        if run_id and self.current_agent:
            run_id_str = str(run_id)
            self.run_to_agent[run_id_str] = self.current_agent
            self.current_run_id = run_id_str

        # Track start time for current agent if set
        if self.current_agent:
            self.start_times[self.current_agent] = datetime.now()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when a completion LLM starts (GPT-3 davinci, etc)."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")
        self._handle_model_start(serialized, run_id, parent_run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        Called when a chat model starts (Claude, GPT-4, etc).

        CRITICAL: This is what Claude/Anthropic models fire, NOT on_llm_start.
        """
        self._handle_model_start(serialized, run_id, parent_run_id, **kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when an LLM finishes."""
        if not self.current_agent:
            return

        # Calculate duration
        duration_ms = None
        if self.current_agent in self.start_times:
            delta = datetime.now() - self.start_times[self.current_agent]
            duration_ms = delta.total_seconds() * 1000
            del self.start_times[self.current_agent]

        # Extract token usage if available
        self._extract_and_update_tokens(response)

        # Show reasoning if enabled
        if self.show_reasoning and response.generations:
            if response.generations[0]:
                content = response.generations[0][0].text
                if content:
                    self._update_activity_log(
                        "thinking",
                        agent_name=self.current_agent,
                        thought=content[:200],
                    )

        # Mark completion (optional - could be noisy)
        # self._update_activity_log("complete", message=f"{self.current_agent} done")

    def _extract_and_update_tokens(self, response: LLMResult) -> None:
        """Extract token usage from LLMResult and update UI."""
        # Try newest LangChain format
        if response.generations and len(response.generations) > 0:
            first_gen_list = response.generations[0]
            if first_gen_list and len(first_gen_list) > 0:
                generation = first_gen_list[0]
                if hasattr(generation, "message") and hasattr(
                    generation.message, "usage_metadata"
                ):
                    usage = generation.message.usage_metadata
                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        if input_tokens or output_tokens:
                            self._update_token_panel(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                agent_name=self.current_agent or "",
                            )
                            return

        # Try llm_output format
        llm_output = response.llm_output or {}
        if "usage" in llm_output:
            usage = llm_output["usage"]
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            if input_tokens or output_tokens:
                self._update_token_panel(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    agent_name=self.current_agent or "",
                )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Called when an LLM errors."""
        self._update_activity_log(
            "tool_error",
            tool_name="LLM",
            error=str(error)[:100],
        )

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
        if serialized is None:
            serialized = {}
        tool_name = serialized.get("name", "unknown_tool")

        # Track current tool and start time
        self.current_tool = tool_name
        self.start_times[f"tool_{tool_name}"] = datetime.now()

        # Detect handoff tools - PRIMARY handoff detection method
        # LangGraph uses: handoff_to_<agent_name> and transfer_back_to_supervisor
        is_handoff = False
        if tool_name.startswith("handoff_to_"):
            is_handoff = True
            target_agent = tool_name.replace("handoff_to_", "")
            # Extract task description from tool input
            task_description = self._extract_task_description(input_str, inputs)
            if self.debug:
                self._debug_notify(f"Handoff: {self.current_agent or 'system'} → {target_agent}")
            self._handle_handoff(target_agent, task_description)
        elif tool_name.startswith("transfer_back_to_"):
            is_handoff = True
            task_description = self._extract_task_description(input_str, inputs)
            if self.debug:
                self._debug_notify(f"Return: {self.current_agent or 'system'} → supervisor")
            self._handle_handoff("supervisor", task_description)

        # Show non-handoff tools in activity log
        if not is_handoff and self.show_tools:
            self._update_activity_log("tool_start", tool_name=tool_name)

    def _handle_handoff(self, target_agent: str, task_description: str = "") -> None:
        """Handle agent handoff detected from tool call.

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

        # Update activity log with handoff (including task description)
        self._update_activity_log(
            "handoff",
            from_agent=from_agent,
            to_agent=target_agent,
            task_description=task_description,
        )

        # Update agents panel (mark old agent idle, new agent active)
        if self.current_agent:
            self._update_agents_panel(self.current_agent, active=False)
        self._update_agents_panel(target_agent, active=True)

        # Update current agent tracking
        self.current_agent = target_agent
        if target_agent not in self.agent_stack:
            self.agent_stack.append(target_agent)

        # Track start time for new agent
        self.start_times[target_agent] = datetime.now()

        # Log agent start in activity
        self._update_activity_log("agent_start", agent_name=target_agent)

    def on_tool_end(self, output: Any, **kwargs) -> None:
        """Called when a tool finishes."""
        if not self.show_tools:
            return

        tool_name = self.current_tool or "unknown_tool"

        # Skip logging handoff tool completions (they're already handled)
        if tool_name.startswith("handoff_to_") or tool_name.startswith("transfer_back_to_"):
            self.current_tool = None
            return

        # Calculate duration
        duration_ms = None
        tool_key = f"tool_{tool_name}"
        if tool_key in self.start_times:
            delta = datetime.now() - self.start_times[tool_key]
            duration_ms = delta.total_seconds() * 1000
            del self.start_times[tool_key]

        self._update_activity_log(
            "tool_complete",
            tool_name=tool_name,
            duration_ms=duration_ms,
        )

        # Clear current tool
        self.current_tool = None

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Called when a tool errors."""
        tool_name = self.current_tool or kwargs.get("name", "unknown_tool")
        self._update_activity_log(
            "tool_error",
            tool_name=tool_name,
            error=str(error)[:100],
        )

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Called when an agent takes an action."""
        # This is usually redundant with on_tool_start, so we skip it
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Called when an agent finishes."""
        if self.agent_stack:
            finished_agent = self.agent_stack.pop()
            self._update_agents_panel(finished_agent, active=False)
            self.current_agent = self.agent_stack[-1] if self.agent_stack else None

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """
        Called when a chain starts - BACKUP method for agent detection.

        Primary detection happens in on_tool_start via handoff_to_* tools.
        This provides backup detection using agent registry.
        """
        if serialized is None:
            serialized = {}
        if inputs is None:
            inputs = {}

        chain_name = serialized.get("name", "")

        # Only process if chain_name is non-empty (LangGraph often sends empty)
        if not chain_name:
            return

        # Detect agent transitions using the agent registry
        agent_names = get_all_agent_names()

        for agent_name in agent_names:
            if agent_name in chain_name.lower():
                if agent_name != self.current_agent:
                    self._handle_handoff(agent_name)
                break

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes."""
        pass

    def mark_all_agents_idle(self) -> None:
        """Mark all agents as idle (call when query completes)."""
        if not self.agents_panel or not self.app:
            return

        def _do_update():
            try:
                self.agents_panel.set_all_idle()
            except Exception:
                pass

        try:
            self.app.call_from_thread(_do_update)
        except Exception:
            pass

        # Reset tracking state
        self.current_agent = None
        self.agent_stack = []

    def reset(self) -> None:
        """Reset all tracking state."""
        self.current_agent = None
        self.agent_stack = []
        self.start_times = {}
        self.current_tool = None
        self.run_to_agent = {}
        self.current_run_id = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
