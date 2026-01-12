"""
State definitions for the research agent.

This module defines the state class for the research agent
which handles literature discovery and dataset identification.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["ResearchAgentState"]


class ResearchAgentState(AgentState):
    """
    State for the research agent.

    Handles literature discovery, dataset identification, metadata validation,
    publication queue management, and workspace caching for scientific content.
    """

    next: str

    # Research-specific context
    task_description: str  # Description of the current research task

    # Literature discovery
    search_history: List[Dict[str, Any]]  # History of searches performed
    publications_found: Dict[str, Any]  # Publications discovered (PMID/DOI -> metadata)
    literature_cache: Dict[str, Any]  # Cached publication content (abstracts, methods)

    # Dataset discovery
    datasets_discovered: Dict[
        str, Any
    ]  # Datasets found (GSE/SRA/PRIDE accessions -> metadata)
    dataset_validation_results: Dict[str, Any]  # Validation results per dataset
    related_entries: Dict[str, Any]  # Related publications/datasets/samples

    # Queue management
    download_queue_entries: List[str]  # Created download queue entry IDs
    publication_queue_entries: List[str]  # Created publication queue entry IDs
    queue_status_tracking: Dict[str, Any]  # Status tracking for queue entries

    # Content extraction
    methods_extracted: Dict[str, Any]  # Extracted methods sections (PMID -> methods)
    metadata_extracted: Dict[str, Any]  # Extracted metadata (identifier -> metadata)
    identifiers_found: Dict[str, Any]  # Dataset identifiers found in publications

    # Workspace management
    cached_content: List[str]  # List of workspace-cached content identifiers
    workspace_locations: Dict[str, Any]  # Mapping of content -> workspace locations

    # Standard expert fields (for consistency)
    file_paths: List[str]  # Paths to cached files, reports, exports
    methodology_parameters: Dict[str, Any]  # Research methodology parameters
    data_context: str  # Research context and focus area
    intermediate_outputs: Dict[
        str, Any
    ]  # Partial research results before handoff to supervisor
