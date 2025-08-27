"""
Centralized Agent Registry for the Lobster system.

This module defines all agents used in the system with their configurations,
making it easy to add new agents without modifying multiple files.
"""

from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass


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
    # 'machine_learning_expert_agent': AgentConfig(
    #     name='machine_learning_expert_agent',
    #     display_name='ML Expert',
    #     description='Handles Machine Learning related tasks like transforming the data in the desired format for downstream tasks',
    #     factory_function='lobster.agents.machine_learning_expert.machine_learning_expert',
    #     handoff_tool_name='handoff_to_machine_learning_expert',
    #     handoff_tool_description='Assign literature/method tasks to the machine learning expert'
    # ),
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
