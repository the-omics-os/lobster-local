from typing import TypeGuard, cast, Annotated

from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.messages import ToolMessage
from langgraph.types import Command, Send
from langgraph.prebuilt import InjectedState
from lobster.agents.langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION

def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:

    @tool(name, description=description)
    def handoff_to_agent(
        # you can add additional tool call arguments for the LLM to populate
        # for example, you can ask the ***LLM to populate a task description*** for the next agent
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context. It must be in task format starting with: 'Your task is to ...'"],
        # you can inject the state of the agent that is calling the tool
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name}
        )
        # messages = cast(AIMessage, state["messages"][-1])
        task_description_message = {
            'role': 'ai',
            'content': task_description
        }
        agent_input = {
            **state,
            "messages": [task_description_message]
        }

        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "active_agent": agent_name,
                # optionally pass the task description to the next agent
                # NOTE: individual agents would need to have `task_description` in their state schema
                # and would need to implement logic for how to consume it
            },
        )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent