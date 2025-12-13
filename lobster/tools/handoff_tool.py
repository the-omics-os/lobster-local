"""Custom handoff tool for LangGraph multi-agent coordination.

This module extends LangGraph's standard handoff mechanism with task-specific
context passing via the `task_description` parameter.

Key implementation note:
- Single handoffs use `goto=agent_name` (string) for sequential execution
- Parallel handoffs use `goto=[Send(...)]` for fan-out execution
- See langgraph_supervisor/handoff.py for reference implementation
"""

import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send

from lobster.agents.langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION


def _remove_non_handoff_tool_calls(
    last_ai_message: AIMessage, handoff_tool_call_id: str
) -> AIMessage:
    """Remove tool calls that are not meant for the target agent.

    When the supervisor calls multiple agents/tools in parallel,
    we need to remove tool calls not meant for this specific agent
    to ensure valid message history.

    Args:
        last_ai_message: The AIMessage containing tool calls
        handoff_tool_call_id: The ID of the current handoff tool call

    Returns:
        AIMessage with only the relevant tool call
    """
    content = last_ai_message.content

    # Handle multiple content blocks (Anthropic format)
    if isinstance(content, list) and len(content) > 1 and isinstance(content[0], dict):
        content = [
            content_block
            for content_block in content
            if (
                content_block.get("type") == "tool_use"
                and content_block.get("id") == handoff_tool_call_id
            )
            or content_block.get("type") != "tool_use"
        ]

    return AIMessage(
        content=content,
        tool_calls=[
            tool_call
            for tool_call in last_ai_message.tool_calls
            if tool_call["id"] == handoff_tool_call_id
        ],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )


def create_custom_handoff_tool(
    *, agent_name: str, name: str | None, description: str | None
) -> BaseTool:
    """Create a custom handoff tool with task description support.

    This tool extends the standard LangGraph handoff pattern by allowing
    the supervisor to provide a task-specific context message to the target agent.

    Key difference from standard handoff:
    - Accepts `task_description` parameter for context passing
    - Properly detects single vs parallel handoffs
    - Uses correct goto patterns for each scenario

    Args:
        agent_name: Name of the agent to handoff to
        name: Tool name (e.g., "handoff_to_research_agent")
        description: Tool description for LLM

    Returns:
        BaseTool: Configured handoff tool
    """

    @tool(name, description=description)
    def handoff_to_agent(
        # The LLM populates this with context for the next agent
        task_description: Annotated[
            str,
            "Detailed description of what the next agent should do, including all of the relevant context. It must be in task format starting with: 'I am the <your role>, Your task is to ...'",
        ],
        # Injected state from the calling agent
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        # Create the tool message with handoff metadata
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name},
        )

        # Get the last AI message to check for parallel handoffs
        messages = state.get("messages", [])
        last_ai_message = messages[-1] if messages else None

        # Create task description message for the target agent
        task_description_message = {"role": "ai", "content": task_description}

        # Detect parallel vs single handoff based on number of tool calls
        # See: langgraph_supervisor/handoff.py lines 104-129 for reference
        is_parallel_handoff = (
            isinstance(last_ai_message, AIMessage)
            and hasattr(last_ai_message, "tool_calls")
            and len(last_ai_message.tool_calls) > 1
        )

        if is_parallel_handoff:
            # PARALLEL HANDOFF: Multiple tool calls in the same message
            # Use Send to enable ToolNode to combine multiple parallel handoffs
            handoff_messages = messages[:-1]  # Remove last AI message
            handoff_messages.extend(
                [
                    _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                    tool_message,
                ]
            )

            agent_input = {
                **state,
                "messages": handoff_messages + [task_description_message],
            }

            return Command(
                goto=[Send(agent_name, agent_input)],  # List for parallel execution
                graph=Command.PARENT,
                # Note: update is not used for parallel handoffs (state is in Send)
            )
        else:
            # SINGLE HANDOFF: Direct routing to one agent
            # Use goto=agent_name (string) for sequential execution
            handoff_messages = messages + [tool_message, task_description_message]

            return Command(
                goto=agent_name,  # Direct string for single handoff
                graph=Command.PARENT,
                update={
                    **state,
                    "messages": handoff_messages,
                    "active_agent": agent_name,
                },
            )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent
