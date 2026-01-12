"""
Centralized Agent Registry for the Lobster system.

This module defines all agents used in the system with their configurations,
making it easy to add new agents without modifying multiple files.
"""

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Thread-safe lazy plugin discovery
_plugins_loaded = False
_plugins_lock = threading.Lock()


@dataclass
class AgentRegistryConfig:
    """Configuration for an agent in the system.

    Attributes:
        supervisor_accessible: Controls whether supervisor can directly handoff to this agent.
            - None (default): Inferred from child_agents relationships. If this agent
              appears in ANY parent's child_agents list, it's NOT supervisor-accessible.
            - True: Explicitly allow supervisor access (override inference).
            - False: Explicitly deny supervisor access (override inference).
    """

    name: str
    display_name: str
    description: str
    factory_function: str  # Module path to the factory function
    handoff_tool_name: Optional[str] = None
    handoff_tool_description: Optional[str] = None
    child_agents: Optional[List[str]] = (
        None  # List of agent names this agent can delegate to
    )
    supervisor_accessible: Optional[bool] = None  # None=infer, True/False=override


# Central registry of all agents in the system
AGENT_REGISTRY: Dict[str, AgentRegistryConfig] = {
    "data_expert_agent": AgentRegistryConfig(
        name="data_expert_agent",
        display_name="Data Expert",
        description="Executes queue-based downloads (ZERO online access), manages modalities with CRUD operations, loads local files via adapter system, retry mechanism with strategy overrides, and workspace orchestration",
        factory_function="lobster.agents.data_expert.data_expert.data_expert",
        handoff_tool_name="handoff_to_data_expert_agent",
        handoff_tool_description="Assign LOCAL data operations: execute downloads from validated queue entries, load local files via adapters, manage modalities (list/inspect/remove/validate), retry failed downloads. DO NOT delegate online operations (metadata/URL extraction) - those go to research_agent",
        child_agents=["metadata_assistant"],
    ),
    "research_agent": AgentRegistryConfig(
        name="research_agent",
        display_name="Research Agent",
        description="Handles literature discovery, dataset identification, PDF extraction with AUTOMATIC PMID/DOI resolution, computational method analysis, and parameter extraction from publications and queuing datasets for download.",
        factory_function="lobster.agents.research.research_agent.research_agent",
        handoff_tool_name="handoff_to_research_agent",
        handoff_tool_description="Assign literature search, dataset discovery, method analysis, parameter extraction, and download queue creation to the research agent",
        child_agents=["metadata_assistant"],
    ),
    # === NEW: Unified Transcriptomics Expert (parent agent) ===
    "transcriptomics_expert": AgentRegistryConfig(
        name="transcriptomics_expert",
        display_name="Transcriptomics Expert",
        description="Unified expert for single-cell AND bulk RNA-seq analysis. Handles QC, clustering, and orchestrates annotation and DE analysis via specialized sub-agents.",
        factory_function="lobster.agents.transcriptomics.transcriptomics_expert.transcriptomics_expert",
        handoff_tool_name="handoff_to_transcriptomics_expert",
        handoff_tool_description="Assign ALL transcriptomics analysis tasks (single-cell OR bulk RNA-seq): QC, clustering, cell type annotation, differential expression, pseudobulk, pathway enrichment/functional analysis (GO/KEGG/Reactome gene set enrichment)",
        child_agents=["annotation_expert", "de_analysis_expert"],
    ),
    # === NEW: Annotation Expert (sub-agent, not supervisor-accessible) ===
    "annotation_expert": AgentRegistryConfig(
        name="annotation_expert",
        display_name="Annotation Expert",
        description="Cell type annotation sub-agent: automatic annotation, manual cluster labeling, debris detection, annotation templates",
        factory_function="lobster.agents.transcriptomics.annotation_expert.annotation_expert",
        handoff_tool_name=None,  # Not directly accessible
        handoff_tool_description=None,
        supervisor_accessible=False,  # Only via transcriptomics_expert
    ),
    # === NEW: DE Analysis Expert (sub-agent, not supervisor-accessible) ===
    "de_analysis_expert": AgentRegistryConfig(
        name="de_analysis_expert",
        display_name="DE Analysis Expert",
        description="Differential expression sub-agent: pseudobulk, pyDESeq2, formula-based DE, pathway enrichment",
        factory_function="lobster.agents.transcriptomics.de_analysis_expert.de_analysis_expert",
        handoff_tool_name=None,  # Not directly accessible
        handoff_tool_description=None,
        supervisor_accessible=False,  # Only via transcriptomics_expert
    ),
    # =========================================================================
    # VISUALIZATION AGENTS
    # =========================================================================
    # visualization_expert_agent: FREE tier (synced to lobster-local)
    #
    # Premium agents are loaded via plugin_loader (component_registry) from:
    # - lobster-premium: Shared premium features (future)
    # - lobster-custom-*: Customer-specific packages
    #
    # Premium visualization agents (NOT in this registry):
    # - protein_structure_visualization_expert_agent
    "visualization_expert_agent": AgentRegistryConfig(
        name="visualization_expert_agent",
        display_name="Visualization Expert",
        description="Creates publication-quality visualizations through supervisor-mediated workflows",
        factory_function="lobster.agents.visualization_expert.visualization_expert",
        handoff_tool_name="handoff_to_visualization_expert_agent",
        handoff_tool_description="Delegate visualization tasks to the visualization expert agent",
    ),
}

def _ensure_plugins_loaded() -> None:
    """
    Thread-safe lazy plugin discovery.

    Uses double-checked locking to ensure plugins are loaded exactly once
    across all threads. This defers the ~900ms plugin discovery cost until
    the first access to AGENT_REGISTRY.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    with _plugins_lock:
        if _plugins_loaded:  # Double-check after acquiring lock
            return
        _merge_plugin_agents()
        _plugins_loaded = True


def get_all_agent_names() -> list[str]:
    """Get all agent names including system agents."""
    _ensure_plugins_loaded()
    return list(AGENT_REGISTRY.keys())


def get_worker_agents() -> Dict[str, AgentRegistryConfig]:
    """Get only the worker agents (excluding system agents)."""
    _ensure_plugins_loaded()
    return AGENT_REGISTRY.copy()


def get_agent_registry_config(agent_name: str) -> Optional[AgentRegistryConfig]:
    """Get registry configuration for a specific agent."""
    _ensure_plugins_loaded()
    return AGENT_REGISTRY.get(agent_name)


def get_valid_handoffs() -> Dict[str, set]:
    """
    Build a map of valid agent handoffs from the registry.

    Returns a dict mapping agent_name -> set of agents it can hand off to.
    Used to validate/correct handoff display during parallel tool calls.

    Example:
        {
            "supervisor": {"data_expert_agent", "research_agent", ...},
            "data_expert_agent": {"metadata_assistant"},
            "research_agent": {"metadata_assistant"},
            "transcriptomics_expert": {"annotation_expert", "de_analysis_expert"},
        }
    """
    _ensure_plugins_loaded()

    valid_handoffs: Dict[str, set] = {}

    # Build supervisor's valid targets (all supervisor-accessible agents)
    supervisor_targets = set()
    for name, config in AGENT_REGISTRY.items():
        # Check explicit supervisor_accessible flag first
        if config.supervisor_accessible is True:
            supervisor_targets.add(name)
        elif config.supervisor_accessible is False:
            continue  # Explicitly not accessible
        else:
            # Infer: accessible if NOT a child of any other agent
            is_child = False
            for other_config in AGENT_REGISTRY.values():
                if other_config.child_agents and name in other_config.child_agents:
                    is_child = True
                    break
            if not is_child:
                supervisor_targets.add(name)

    valid_handoffs["supervisor"] = supervisor_targets

    # Build each agent's valid targets from child_agents
    for name, config in AGENT_REGISTRY.items():
        if config.child_agents:
            valid_handoffs[name] = set(config.child_agents)
        else:
            valid_handoffs[name] = set()

    return valid_handoffs


def is_valid_handoff(from_agent: str, to_agent: str) -> bool:
    """
    Check if a handoff from one agent to another is valid.

    Args:
        from_agent: The agent initiating the handoff
        to_agent: The agent being handed off to

    Returns:
        True if the handoff is valid according to the agent hierarchy
    """
    valid_handoffs = get_valid_handoffs()
    return to_agent in valid_handoffs.get(from_agent, set())


def import_agent_factory(factory_path: str) -> Callable:
    """Dynamically import an agent factory function."""
    module_path, function_name = factory_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[function_name])
    return getattr(module, function_name)


# =============================================================================
# PLUGIN DISCOVERY AND REGISTRY MERGING
# =============================================================================
# Discover and merge agents from premium/custom packages at module load time.
# This allows lobster-premium and lobster-custom-* packages to register
# additional agents that become available based on subscription tier.


def _merge_plugin_agents() -> None:
    """
    Discover and merge plugin agents into the registry.

    Now delegates to the unified ComponentRegistry which discovers both
    services and agents via entry points (lobster.services, lobster.agents).

    Custom agents from lobster-premium and lobster-custom-* packages are
    discovered automatically and merged with core agents.
    """
    try:
        from lobster.core.component_registry import component_registry

        # Trigger component discovery (services + custom agents)
        component_registry.load_components()

        # Merge custom agents into AGENT_REGISTRY for backward compatibility
        custom_agents = component_registry.list_custom_agents()
        if custom_agents:
            AGENT_REGISTRY.update(custom_agents)
            # Log at debug level to avoid noise during imports
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Merged {len(custom_agents)} custom agents into registry")
    except ImportError:
        # component_registry not available (shouldn't happen in normal installs)
        pass
    except Exception as e:
        # Don't let component discovery failures break the core system
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Component registry loading failed: {e}")


# Plugin merging is now lazy-loaded via _ensure_plugins_loaded()
# This defers ~900ms of plugin discovery until first registry access
