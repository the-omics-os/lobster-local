"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using langgraph_supervisor package for hierarchical multi-agent coordination.
"""

import inspect
from typing import Dict, Optional

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from lobster.agents.langgraph_supervisor import (
    create_forward_message_tool,
    create_supervisor,
)
from lobster.agents.state import OverallState
from lobster.agents.supervisor import create_supervisor_prompt
from lobster.config.agent_registry import get_worker_agents, import_agent_factory
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.config.supervisor_config import SupervisorConfig
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.handoff_tool import create_custom_handoff_tool
from lobster.tools.workspace_tool import (
    create_delete_from_workspace_tool,
    create_get_content_from_workspace_tool,
    create_list_modalities_tool,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _create_delegation_tool(agent_name: str, agent, description: str):
    """Create a tool that delegates to a sub-agent.

    This follows the official LangGraph supervisor pattern where
    sub-agents are wrapped as tools rather than passed as agents.
    """

    @tool
    def delegate(request: str) -> str:
        """Delegate task to a sub-agent."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        # Return only the final message content
        final_msg = result["messages"][-1]
        return final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    # Set proper name and docstring
    delegate.__name__ = f"handoff_to_{agent_name}"
    delegate.__doc__ = f"Delegate task to {agent_name}. {description}"
    return delegate


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

    supervisor_model = create_llm("supervisor", model_params, provider_override=provider_override)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    # callback_handler may be a single callback, a list of callbacks, or None
    if callback_handler and hasattr(supervisor_model, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        supervisor_model = supervisor_model.with_config(callbacks=callbacks)

    # Phase 1: Create all agents (no ordering needed for tool-wrapping)
    agents = []
    handoff_tools = []
    created_agents = {}
    supervisor_accessible_agents = (
        []
    )  # Track which agents should be passed to supervisor

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

    for agent_name, agent_config in worker_agents.items():
        factory_function = import_agent_factory(agent_config.factory_function)

        # Build kwargs for agent factory
        factory_kwargs = {
            "data_manager": data_manager,
            "callback_handler": callback_handler,
            "agent_name": agent_config.name,
        }

        # Pass optional parameters to factories that support them
        # (determined by inspecting function signature)
        sig = inspect.signature(factory_function)
        if "subscription_tier" in sig.parameters:
            factory_kwargs["subscription_tier"] = subscription_tier
        if "provider_override" in sig.parameters:
            factory_kwargs["provider_override"] = provider_override

        # Create agent WITHOUT delegation tools first
        agent = factory_function(**factory_kwargs)
        created_agents[agent_name] = agent

        # Create handoff tool if configured AND supervisor-accessible
        if agent_config.handoff_tool_name and agent_config.handoff_tool_description:
            # Determine supervisor accessibility (inference or explicit override)
            if agent_config.supervisor_accessible is None:
                # Infer: child agents are NOT supervisor-accessible by default
                is_supervisor_accessible = agent_name not in child_agent_names
            else:
                # Explicit override from registry
                is_supervisor_accessible = agent_config.supervisor_accessible

            if is_supervisor_accessible:
                handoff_tool = create_custom_handoff_tool(
                    agent_name=agent_config.name,
                    name=agent_config.handoff_tool_name,
                    description=agent_config.handoff_tool_description,
                )
                handoff_tools.append(handoff_tool)
                supervisor_accessible_agents.append(
                    agent
                )  # Only add to supervisor list if accessible
                logger.debug(
                    f"Created supervisor handoff tool for: {agent_config.display_name}"
                )
            else:
                logger.debug(
                    f"Skipped supervisor handoff for child agent: {agent_config.display_name}"
                )
        else:
            # Only warn if the agent SHOULD be supervisor-accessible but lacks handoff config
            # Sub-agents (supervisor_accessible=False or inferred as child) don't need handoff tools
            is_child_agent = agent_name in child_agent_names
            is_explicitly_not_accessible = agent_config.supervisor_accessible is False
            if not is_child_agent and not is_explicitly_not_accessible:
                logger.warning(
                    f"Agent {agent_config.display_name} has no handoff tool configured"
                )
            else:
                logger.debug(
                    f"Sub-agent {agent_config.display_name} (no handoff tool needed)"
                )

        logger.debug(
            f"Created agent: {agent_config.display_name} ({agent_config.name})"
        )

    # Phase 2: Re-create parent agents WITH delegation tools
    for agent_name, agent_config in worker_agents.items():
        if agent_config.child_agents:
            # Create delegation tools for this parent agent
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

            # Re-create the parent agent WITH delegation tools
            factory_function = import_agent_factory(agent_config.factory_function)

            # Build kwargs including optional parameters if supported
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

            new_agent = factory_function(**factory_kwargs)

            # Replace in our tracking
            old_agent = created_agents[agent_name]
            created_agents[agent_name] = new_agent

            # Also update in supervisor_accessible_agents if present
            if old_agent in supervisor_accessible_agents:
                idx = supervisor_accessible_agents.index(old_agent)
                supervisor_accessible_agents[idx] = new_agent

            logger.debug(
                f"Re-created {agent_name} with {len(delegation_tools)} delegation tool(s)"
            )

    # Create shared tools with data_manager access
    list_available_modalities = create_list_modalities_tool(data_manager)
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    delete_from_workspace = create_delete_from_workspace_tool(data_manager)

    # Get list of supervisor-accessible agents (for prompt generation)
    active_agent_names = [agent.name for agent in supervisor_accessible_agents]
    logger.debug(f"Supervisor-accessible agents: {active_agent_names}")
    logger.debug(
        f"Total agents created: {len(created_agents)}, Supervisor-accessible: {len(supervisor_accessible_agents)}"
    )

    # Create supervisor prompt with configuration and active agents
    system_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=supervisor_config,
        active_agents=active_agent_names,
    )

    # add forwarding tool for supervisor. This is useful when the supervisor determines that the worker's response is sufficient and doesn't require further processing or summarization by the supervisor itself.
    # create_forward_message_tool("supervisor")

    # UPDATED CONFIGURATION - Changed output_mode
    # Only pass supervisor-accessible agents (those with handoff tools)
    workflow = create_supervisor(
        agents=supervisor_accessible_agents,  # Only agents the supervisor can directly access
        model=supervisor_model,
        prompt=system_prompt,
        supervisor_name="supervisor",
        state_schema=OverallState,
        add_handoff_messages=True,
        add_handoff_back_messages=True,
        include_agent_name="inline",
        # Change from "full_history" to "messages" or "last_message"
        # output_mode="full_history",  # This ensures the actual messages are returned
        output_mode="last_message",  # This ensures the actual messages are returned
        tools=handoff_tools
        + [
            list_available_modalities,
            get_content_from_workspace,
            delete_from_workspace,
        ],
        # + [forwarding_tool],  # Supervisor-only tools (handoff tools are auto-created)
    )

    # Compile the graph with the provided checkpointer
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store,
        # debug=True  # Enable debug mode for better visibility
    )

    logger.debug("Bioinformatics multi-agent graph created successfully")
    return graph
