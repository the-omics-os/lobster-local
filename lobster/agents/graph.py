"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using Tool Calling pattern: supervisor invokes sub-agents as tools.
This is simpler and more appropriate for centralized orchestration where users
only interact with the supervisor.

See: https://docs.langchain.com/oss/langchain/multi-agent#tool-calling
"""

import inspect
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from lobster.agents.state import OverallState
from lobster.agents.supervisor import create_supervisor_prompt
from lobster.config.agent_registry import get_worker_agents, import_agent_factory
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.config.supervisor_config import SupervisorConfig
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.todo_tools import create_todo_tools
from lobster.tools.workspace_tool import (
    create_delete_from_workspace_tool,
    create_get_content_from_workspace_tool,
    create_list_modalities_tool,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _create_agent_tool(agent_name: str, agent, tool_name: str, description: str):
    """Create a tool that invokes a sub-agent (Tool Calling pattern).

    This follows the LangChain Tool Calling pattern where sub-agents are
    invoked as tools and return their results directly. The supervisor
    maintains centralized control - sub-agents never interact with users.

    See: https://docs.langchain.com/oss/langchain/multi-agent#tool-calling

    Args:
        agent_name: Internal name of the agent (for logging)
        agent: The compiled agent (Pregel) to invoke
        tool_name: Name for the tool (e.g., "handoff_to_research_agent")
        description: Description of when to use this tool
    """

    @tool(tool_name, description=description)
    def invoke_agent(task_description: str) -> str:
        """Invoke a sub-agent with a task description.

        Args:
            task_description: Detailed description of what the agent should do,
                including all relevant context. Should be in task format starting
                with 'Your task is to ...'
        """
        logger.debug(f"Invoking {agent_name} with task: {task_description[:100]}...")

        # Pass explicit agent name in config for proper callback attribution
        # This ensures token tracking and logging correctly attributes sub-agent activity
        config = {
            "run_name": agent_name,  # LangChain uses run_name for agent identification
            "tags": [agent_name],    # Additional tag for tracking
        }

        # Invoke the sub-agent with the task as a user message
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task_description}]},
            config=config
        )

        # Extract the final message content
        final_msg = result.get("messages", [])[-1] if result.get("messages") else None
        if final_msg is None:
            return f"Agent {agent_name} returned no response."

        content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        logger.debug(f"Agent {agent_name} completed. Response length: {len(content)}")

        return content

    return invoke_agent


def _create_delegation_tool(agent_name: str, agent, description: str):
    """Create a delegation tool for parent-child agent relationships.

    This is a simplified version for hierarchical delegation within agent families
    (e.g., transcriptomics_expert -> de_analysis_expert).
    """
    return _create_agent_tool(
        agent_name=agent_name,
        agent=agent,
        tool_name=f"handoff_to_{agent_name}",
        description=f"Delegate task to {agent_name}. {description}",
    )


def create_bioinformatics_graph(
    data_manager: DataManagerV2,
    checkpointer: InMemorySaver = None,
    store: InMemoryStore = None,
    callback_handler=None,
    manual_model_params: dict = None,
    supervisor_config: Optional[SupervisorConfig] = None,
    subscription_tier: str = None,
    agent_filter: callable = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    workspace_path: Optional[Path] = None,
):
    """Create the bioinformatics multi-agent graph using langgraph_supervisor.

    Args:
        data_manager: DataManagerV2 instance for data operations
        checkpointer: Optional memory saver for conversation persistence
        store: Optional in-memory store for shared state
        callback_handler: Optional callback for streaming responses
        manual_model_params: Optional override for supervisor model parameters
        supervisor_config: Optional supervisor configuration
        subscription_tier: Subscription tier for feature gating (free/premium/enterprise).
            If None, will be auto-detected from license. Controls which agents are
            available and which handoffs are allowed.
        agent_filter: Optional callable(agent_name, agent_config) -> bool to filter
            which agents are included in the graph. Used for tier-based restrictions.
        provider_override: Optional explicit provider name (e.g., "bedrock", "anthropic", "ollama")
        model_override: Optional explicit model name (e.g., "llama3:70b-instruct", "claude-4-sonnet")

    Note: When invoking this graph, set the recursion_limit in the config to prevent
    hitting the default limit of 25. Example:
        config = {"recursion_limit": 100, ...}
        graph.invoke(input, config)
    """
    # Auto-detect subscription tier if not provided
    if subscription_tier is None:
        try:
            from lobster.core.license_manager import get_current_tier

            subscription_tier = get_current_tier()
        except ImportError:
            subscription_tier = "free"
    logger.debug(f"Creating graph with subscription tier: {subscription_tier}")

    # Create tier-based agent filter if not provided
    if agent_filter is None:
        from lobster.config.subscription_tiers import is_agent_available

        agent_filter = lambda name, config: is_agent_available(name, subscription_tier)
    logger.debug("Creating bioinformatics multi-agent graph")

    # Get model configuration for the supervisor
    settings = get_settings()

    # ensure this for later
    if manual_model_params:
        # Use provided manual model parameters if available
        model_params = manual_model_params
    else:
        model_params = settings.get_agent_llm_params("supervisor")

    supervisor_model = create_llm("supervisor", model_params, provider_override=provider_override, model_override=model_override, workspace_path=workspace_path)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    # callback_handler may be a single callback, a list of callbacks, or None
    if callback_handler and hasattr(supervisor_model, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        supervisor_model = supervisor_model.with_config(callbacks=callbacks)

    # ==========================================================================
    # Phase 1: Create all sub-agents
    # ==========================================================================
    created_agents = {}
    agent_tools = []  # Tools for supervisor to invoke sub-agents

    worker_agents = get_worker_agents()

    # Apply agent filter for tier-based restrictions
    filtered_worker_agents = {}
    filtered_out_agents = []
    for agent_name, agent_config in worker_agents.items():
        if agent_filter(agent_name, agent_config):
            filtered_worker_agents[agent_name] = agent_config
        else:
            filtered_out_agents.append(agent_name)
    if filtered_out_agents:
        logger.info(
            f"Tier '{subscription_tier}' excludes agents: {filtered_out_agents}"
        )
    worker_agents = filtered_worker_agents

    # Pre-compute child agents for supervisor_accessible inference
    # Agents that appear in ANY parent's child_agents are NOT supervisor-accessible by default
    child_agent_names = set()
    for agent_config in worker_agents.values():
        if agent_config.child_agents:
            child_agent_names.update(agent_config.child_agents)
    if child_agent_names:
        logger.debug(
            f"Child agents (not supervisor-accessible by default): {child_agent_names}"
        )

    # Create all agents first (without delegation tools)
    for agent_name, agent_config in worker_agents.items():
        factory_function = import_agent_factory(agent_config.factory_function)

        # Build kwargs for agent factory
        factory_kwargs = {
            "data_manager": data_manager,
            "callback_handler": callback_handler,
            "agent_name": agent_config.name,
        }

        # Pass optional parameters to factories that support them
        sig = inspect.signature(factory_function)
        if "subscription_tier" in sig.parameters:
            factory_kwargs["subscription_tier"] = subscription_tier
        if "provider_override" in sig.parameters:
            factory_kwargs["provider_override"] = provider_override
        if "model_override" in sig.parameters:
            factory_kwargs["model_override"] = model_override
        if "workspace_path" in sig.parameters:
            factory_kwargs["workspace_path"] = workspace_path

        agent = factory_function(**factory_kwargs)
        created_agents[agent_name] = agent
        logger.debug(f"Created agent: {agent_config.display_name} ({agent_config.name})")

    # ==========================================================================
    # Phase 2: Re-create parent agents WITH delegation tools (for child agents)
    # ==========================================================================
    for agent_name, agent_config in worker_agents.items():
        if agent_config.child_agents:
            delegation_tools = []
            for child_name in agent_config.child_agents:
                if child_name in created_agents:
                    child_agent = created_agents[child_name]
                    child_config = worker_agents.get(child_name)
                    if child_config:
                        delegation_tool = _create_delegation_tool(
                            child_name, child_agent, child_config.description
                        )
                        delegation_tools.append(delegation_tool)

            # Re-create parent with delegation tools
            factory_function = import_agent_factory(agent_config.factory_function)
            factory_kwargs = {
                "data_manager": data_manager,
                "callback_handler": callback_handler,
                "agent_name": agent_config.name,
                "delegation_tools": delegation_tools,
            }
            sig = inspect.signature(factory_function)
            if "subscription_tier" in sig.parameters:
                factory_kwargs["subscription_tier"] = subscription_tier
            if "provider_override" in sig.parameters:
                factory_kwargs["provider_override"] = provider_override
            if "model_override" in sig.parameters:
                factory_kwargs["model_override"] = model_override
            if "workspace_path" in sig.parameters:
                factory_kwargs["workspace_path"] = workspace_path

            created_agents[agent_name] = factory_function(**factory_kwargs)
            logger.debug(
                f"Re-created {agent_name} with {len(delegation_tools)} delegation tool(s)"
            )

    # ==========================================================================
    # Phase 3: Create agent tools for supervisor (Tool Calling pattern)
    # ==========================================================================
    # Only create tools for supervisor-accessible agents (not child agents)
    supervisor_accessible_names = []

    for agent_name, agent_config in worker_agents.items():
        if not agent_config.handoff_tool_name or not agent_config.handoff_tool_description:
            continue

        # Determine supervisor accessibility
        if agent_config.supervisor_accessible is None:
            is_supervisor_accessible = agent_name not in child_agent_names
        else:
            is_supervisor_accessible = agent_config.supervisor_accessible

        if is_supervisor_accessible:
            agent_tool = _create_agent_tool(
                agent_name=agent_config.name,
                agent=created_agents[agent_name],
                tool_name=agent_config.handoff_tool_name,
                description=agent_config.handoff_tool_description,
            )
            agent_tools.append(agent_tool)
            supervisor_accessible_names.append(agent_config.name)
            logger.debug(f"Created supervisor tool: {agent_config.handoff_tool_name}")

    # ==========================================================================
    # Phase 4: Create shared tools and supervisor
    # ==========================================================================
    # Create shared tools with data_manager access
    list_available_modalities = create_list_modalities_tool(data_manager)
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    delete_from_workspace = create_delete_from_workspace_tool(data_manager)

    logger.debug(f"Supervisor-accessible agents: {supervisor_accessible_names}")
    logger.debug(
        f"Total agents created: {len(created_agents)}, "
        f"Supervisor tools: {len(agent_tools)}"
    )

    # Create supervisor prompt with active agents list
    system_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=supervisor_config,
        active_agents=supervisor_accessible_names,
    )

    # Create todo tools for planning
    write_todos, read_todos = create_todo_tools()

    # Combine all tools for the supervisor
    all_supervisor_tools = (
        agent_tools  # Tools to invoke sub-agents
        + [
            list_available_modalities,
            get_content_from_workspace,
            delete_from_workspace,
            write_todos,  # Planning tools
            read_todos,
        ]
    )

    # ==========================================================================
    # Create supervisor using simple Tool Calling pattern
    # ==========================================================================
    # This is much simpler than the Handoffs pattern:
    # - Supervisor is a ReAct agent with tools that invoke sub-agents
    # - No graph-based routing, no Command/Send complexity
    # - Sub-agents are invoked directly and return results
    # - User only ever interacts with supervisor
    #
    # See: https://docs.langchain.com/oss/langchain/multi-agent#tool-calling

    # Create the supervisor as a ReAct agent
    supervisor_agent = create_react_agent(
        model=supervisor_model,
        tools=all_supervisor_tools,
        prompt=system_prompt,
        state_schema=OverallState,
    )

    # Wrap in a StateGraph with explicit "supervisor" node name
    # This ensures events are keyed by "supervisor" (backward compatible with client)
    workflow = StateGraph(OverallState)
    workflow.add_node("supervisor", supervisor_agent)

    # Add all agents as nodes for visualization
    # Execution uses tool calling, but visualization shows full architecture
    for agent_name, agent_instance in created_agents.items():
        workflow.add_node(agent_name, agent_instance)

    # Add edges for visualization
    workflow.add_edge(START, "supervisor")

    # Supervisor can handoff to any supervisor-accessible agent
    for agent_name in supervisor_accessible_names:
        workflow.add_edge("supervisor", agent_name)
        workflow.add_edge(agent_name, "supervisor")

    # Child agents can be delegated to by parents
    for agent_name, agent_config in worker_agents.items():
        if agent_config.child_agents:
            for child_name in agent_config.child_agents:
                if child_name in created_agents:
                    workflow.add_edge(agent_name, child_name)
                    workflow.add_edge(child_name, agent_name)

    workflow.add_edge("supervisor", END)

    # Compile with checkpointer and store
    graph = workflow.compile(checkpointer=checkpointer, store=store)

    logger.debug("Bioinformatics multi-agent graph created successfully (Tool Calling pattern)")
    return graph
