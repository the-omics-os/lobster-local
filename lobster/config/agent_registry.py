"""
Centralized Agent Registry for the Lobster system.

This module defines all agents used in the system with their configurations,
making it easy to add new agents without modifying multiple files.
"""

from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass

from langchain_core.tools import BaseTool


@dataclass
class AgentRegistryConfig:
    """Configuration for an agent in the system."""
    name: str
    display_name: str
    description: str
    factory_function: str  # Module path to the factory function
    handoff_tool_name: Optional[str] = None
    handoff_tool_description: Optional[str] = None


# Central registry of all agents in the system
AGENT_REGISTRY: Dict[str, AgentRegistryConfig] = {
    'data_expert_agent': AgentRegistryConfig(
        name='data_expert_agent',
        display_name='Data Expert',
        description='Handles data fetching and download tasks',
        factory_function='lobster.agents.data_expert.data_expert',
        handoff_tool_name='handoff_to_data_expert',
        handoff_tool_description='Assign data fetching/download tasks to the data expert'
    ),
    'singlecell_expert_agent': AgentRegistryConfig(
        name='singlecell_expert_agent',
        display_name='Single-Cell Expert',
        description='Handles single-cell RNA-seq analysis tasks',
        factory_function='lobster.agents.singlecell_expert.singlecell_expert',
        handoff_tool_name='handoff_to_singlecell_expert',
        handoff_tool_description='Assign single-cell RNA-seq analysis tasks to the single-cell expert'
    ),
    'bulk_rnaseq_expert_agent': AgentRegistryConfig(
        name='bulk_rnaseq_expert_agent',
        display_name='Bulk RNA-seq Expert',
        description='Handles bulk RNA-seq analysis tasks',
        factory_function='lobster.agents.bulk_rnaseq_expert.bulk_rnaseq_expert',
        handoff_tool_name='handoff_to_bulk_rnaseq_expert',
        handoff_tool_description='Assign bulk RNA-seq analysis tasks to the bulk RNA-seq expert'
    ),
    'research_agent': AgentRegistryConfig(
        name='research_agent',
        display_name='Research Agent',
        description='Handles literature discovery and dataset identification tasks',
        factory_function='lobster.agents.research_agent.research_agent',
        handoff_tool_name='handoff_to_research_agent',
        handoff_tool_description='Assign literature search and dataset discovery tasks to the research agent'
    ),
    'method_expert_agent': AgentRegistryConfig(
        name='method_expert_agent',
        display_name='Method Expert',
        description='Handles computational method extraction and parameter analysis from publications',
        factory_function='lobster.agents.method_expert.method_expert',
        handoff_tool_name='handoff_to_method_expert',
        handoff_tool_description='Assign computational parameter extraction tasks to the method expert'
    ),
    'ms_proteomics_expert_agent': AgentRegistryConfig(
        name='ms_proteomics_expert_agent',
        display_name='MS Proteomics Expert',
        description='Handles mass spectrometry proteomics data analysis including DDA/DIA workflows with database search artifact removal',
        factory_function='lobster.agents.ms_proteomics_expert.ms_proteomics_expert',
        handoff_tool_name='handoff_to_ms_proteomics_expert',
        handoff_tool_description='Assign mass spectrometry proteomics analysis tasks to the MS proteomics expert'
    ),
    'affinity_proteomics_expert_agent': AgentRegistryConfig(
        name='affinity_proteomics_expert_agent',
        display_name='Affinity Proteomics Expert',
        description='Handles affinity proteomics data analysis including Olink and targeted protein panels with antibody validation',
        factory_function='lobster.agents.affinity_proteomics_expert.affinity_proteomics_expert',
        handoff_tool_name='handoff_to_affinity_proteomics_expert',
        handoff_tool_description='Assign affinity proteomics and targeted panel analysis tasks to the affinity proteomics expert'
    ),
    'machine_learning_expert_agent': AgentRegistryConfig(
        name='machine_learning_expert_agent',
        display_name='ML Expert',
        description='Handles Machine Learning related tasks like transforming the data in the desired format for downstream tasks',
        factory_function='lobster.agents.machine_learning_expert.machine_learning_expert',
        handoff_tool_name='handoff_to_machine_learning_expert',
        handoff_tool_description='Assign literature/method tasks to the machine learning expert'
    ),
}


# Additional agent names that might appear in chains but aren't worker agents
# SYSTEM_AGENTS = ['supervisor', 'transcriptomics_expert', 'method_agent', 'clarify_with_user']


def get_all_agent_names() -> list[str]:
    """Get all agent names including system agents."""
    return list(AGENT_REGISTRY.keys())


def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    """Get only the worker agents (excluding system agents)."""
    return AGENT_REGISTRY.copy()


def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    """Get registry configuration for a specific agent."""
    return AGENT_REGISTRY.get(agent_name)


def import_agent_factory(factory_path: str) -> Callable:
    """Dynamically import an agent factory function."""
    module_path, function_name = factory_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)


def create_expert_handoff_tools(available_agents: List[str]) -> Dict[str, BaseTool]:
    """
    Automatically create handoff tools between compatible experts
    based on available agents and handoff patterns.

    Args:
        available_agents: List of available agent names

    Returns:
        Dictionary of handoff tools keyed by tool name
    """
    from lobster.tools.expert_handoff_patterns import EXPERT_HANDOFF_PATTERNS
    from lobster.tools.enhanced_handoff_tool import create_expert_handoff_tool

    handoff_tools = {}

    # Normalize agent names by removing '_agent' suffix if present
    normalized_agents = {}
    for agent_name in available_agents:
        if agent_name.endswith('_agent'):
            normalized_name = agent_name[:-6]  # Remove '_agent'
        else:
            normalized_name = agent_name
        normalized_agents[normalized_name] = agent_name

    for pattern_name, pattern in EXPERT_HANDOFF_PATTERNS.items():
        from_expert_norm = pattern.from_expert
        to_expert_norm = pattern.to_expert

        # Check if both experts are available
        if from_expert_norm in normalized_agents and to_expert_norm in normalized_agents:
            from_agent_name = normalized_agents[from_expert_norm]
            to_agent_name = normalized_agents[to_expert_norm]

            # Create handoff tools for each task type
            for task_type in pattern.task_types:
                tool_name = f"handoff_{from_expert_norm}_to_{to_expert_norm}_{task_type}"

                # Create the enhanced handoff tool
                handoff_tool = create_expert_handoff_tool(
                    from_expert=from_expert_norm,
                    to_expert=to_expert_norm,
                    task_type=task_type,
                    context_schema=pattern.context_schema,
                    return_to_sender=(pattern.return_flow == "sender"),
                    custom_description=f"Hand off {task_type} task from {pattern.from_expert} to {pattern.to_expert}"
                )

                handoff_tools[tool_name] = handoff_tool

    return handoff_tools


def get_handoff_tools_for_agent(agent_name: str, available_agents: List[str]) -> List[BaseTool]:
    """
    Get all handoff tools relevant to a specific agent.

    Args:
        agent_name: Name of the agent to get handoff tools for
        available_agents: List of all available agent names

    Returns:
        List of handoff tools that this agent can use
    """
    from lobster.tools.expert_handoff_patterns import get_handoff_patterns_for_expert

    # Normalize agent name
    if agent_name.endswith('_agent'):
        normalized_name = agent_name[:-6]
    else:
        normalized_name = agent_name

    # Get all handoff tools
    all_handoff_tools = create_expert_handoff_tools(available_agents)

    # Filter tools relevant to this agent (outgoing handoffs)
    relevant_tools = []
    for tool_name, tool in all_handoff_tools.items():
        if tool_name.startswith(f"handoff_{normalized_name}_to_"):
            relevant_tools.append(tool)

    return relevant_tools


def get_available_handoff_destinations(from_agent: str) -> List[str]:
    """
    Get list of agents that the given agent can hand off to.

    Args:
        from_agent: Source agent name

    Returns:
        List of destination agent names
    """
    from lobster.tools.expert_handoff_patterns import get_handoff_patterns_for_expert

    # Normalize agent name
    if from_agent.endswith('_agent'):
        normalized_name = from_agent[:-6]
    else:
        normalized_name = from_agent

    patterns = get_handoff_patterns_for_expert(normalized_name, direction="from")
    destinations = [pattern.to_expert for pattern in patterns]

    # Convert back to agent names
    destination_agents = []
    for dest in destinations:
        # Check if this destination exists in the registry
        for agent_name, config in AGENT_REGISTRY.items():
            if agent_name.startswith(dest) or config.display_name.lower().replace(' ', '_') == dest:
                destination_agents.append(agent_name)
                break

    return destination_agents


def validate_handoff_compatibility(from_agent: str, to_agent: str, task_type: str) -> bool:
    """
    Validate if a handoff between two agents is supported.

    Args:
        from_agent: Source agent name
        to_agent: Target agent name
        task_type: Task type to validate

    Returns:
        True if handoff is supported, False otherwise
    """
    from lobster.tools.expert_handoff_patterns import validate_handoff_pattern

    # Normalize agent names
    def normalize_agent_name(name):
        if name.endswith('_agent'):
            return name[:-6]
        return name

    from_normalized = normalize_agent_name(from_agent)
    to_normalized = normalize_agent_name(to_agent)

    return validate_handoff_pattern(from_normalized, to_normalized, task_type)


def get_handoff_registry_summary() -> Dict[str, Any]:
    """
    Get a summary of the handoff registry for debugging and monitoring.

    Returns:
        Dictionary with handoff registry information
    """
    from lobster.tools.expert_handoff_patterns import list_all_handoff_patterns

    patterns = list_all_handoff_patterns()
    available_agents = get_all_agent_names()

    summary = {
        "total_patterns": len(patterns),
        "available_agents": len(available_agents),
        "agents": available_agents,
        "patterns_by_priority": {},
        "handoff_matrix": {}
    }

    # Group patterns by priority
    for pattern_name, pattern in patterns.items():
        priority = pattern.priority
        if priority not in summary["patterns_by_priority"]:
            summary["patterns_by_priority"][priority] = []
        summary["patterns_by_priority"][priority].append({
            "name": pattern_name,
            "from": pattern.from_expert,
            "to": pattern.to_expert,
            "task_types": pattern.task_types,
            "description": pattern.description
        })

    # Create handoff matrix
    expert_names = set()
    for pattern in patterns.values():
        expert_names.add(pattern.from_expert)
        expert_names.add(pattern.to_expert)

    for from_expert in expert_names:
        summary["handoff_matrix"][from_expert] = {}
        for to_expert in expert_names:
            if from_expert != to_expert:
                # Check if handoff exists
                handoff_exists = any(
                    p.from_expert == from_expert and p.to_expert == to_expert
                    for p in patterns.values()
                )
                summary["handoff_matrix"][from_expert][to_expert] = handoff_exists

    return summary
