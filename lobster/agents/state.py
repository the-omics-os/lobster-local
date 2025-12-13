"""
State definitions for the bioinformatics multi-agent system.

Following the LangGraph 0.2.x multi-agent template pattern, but expanded to
capture routing metadata, agent-specific working memory, and intermediate outputs.
"""

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.prebuilt.chat_agent_executor import AgentState


# =============================================================================
# Todo List State (v3.5+)
# =============================================================================


class TodoItem(TypedDict):
    """Individual todo item for planning multi-step tasks.

    Used by supervisor to decompose complex requests into trackable subtasks.
    Follows the DeepAgents/LangChain TodoListMiddleware pattern.

    Attributes:
        content: Task description in imperative form (e.g., "Download GSE12345")
        status: Current state - "pending", "in_progress", or "completed"
        activeForm: Present continuous form for UI display (e.g., "Downloading GSE12345")
    """

    content: str
    status: str  # "pending" | "in_progress" | "completed"
    activeForm: str


def _todo_reducer(
    left: Optional[List[TodoItem]], right: Optional[List[TodoItem]]
) -> List[TodoItem]:
    """Reducer for todos field - replaces entire list on update.

    This is a replace reducer (not append) because:
    1. Todos represent the CURRENT plan state, not history
    2. Agent sends complete updated list on each write_todos call
    3. Matches DeepAgents/LangChain TodoListMiddleware behavior

    Args:
        left: Previous todos list (or None)
        right: New todos list from tool update (or None)

    Returns:
        The new todos list (right), or previous (left), or empty list
    """
    if right is not None:
        return right
    return left if left is not None else []


# =============================================================================
# Core Agent States
# =============================================================================


class OverallState(AgentState):
    """
    Supervisor state for coordinating agent workflows.

    Note: Individual agents are subgraphs with their own state schemas.
    The supervisor doesn't need to track agent-specific state fields.
    """

    # Meta routing information
    last_active_agent: str = ""
    conversation_id: str = ""

    # Optional: Task context for handoffs
    current_task: str = ""
    task_context: Dict[str, Any] = {}

    # Todo list for planning multi-step tasks (v3.5+)
    # Updated via write_todos tool using Command pattern
    todos: Annotated[List[TodoItem], _todo_reducer] = []


# class SupervisorState(AgentState):
#     """
#     State for the supervisor agent.
#     """
#     next: str  # The next node to route to (agent name or END)
#     last_active_agent: str

#     # Bioinformatics-specific context the supervisor might maintain
#     analysis_results: Dict[str, Any]     # Combined results from multiple experts
#     methodology_parameters: Dict[str, Any]
#     data_context: str                    # High-level description of the data in use

#     # Control information
#     delegation_history: List[str]        # Track which agents have been called
#     pending_tasks: List[str]


# class TranscriptomicsExpertState(AgentState):
#     """
#     State for the transcriptomics expert agent.
#     """
#     next: str

#     # Transcriptomics-specific context
#     analysis_results: Dict[str, Any]     # Gene expression, DEG lists, etc.
#     file_paths: List[str]            # Paths to input/output files
#     methodology_parameters: Dict[str, Any]
#     data_context: str                    # Type/source of transcriptomics data (RNA-seq, microarray)
#     quality_control_metrics: Dict[str, Any]
#     intermediate_outputs: Dict[str, Any] # For partial computations before returning to supervisor


class SingleCellExpertState(AgentState):
    """
    State for the single-cell RNA-seq expert agent.
    """

    next: str

    # Single-cell specific context
    task_description: str  # Description of the current task
    analysis_results: Dict[str, Any]  # Single-cell analysis results, clustering, etc.
    clustering_parameters: Dict[
        str, Any
    ]  # Leiden resolution, batch correction settings
    cell_type_annotations: Dict[str, Any]  # Cell type assignment results
    quality_control_metrics: Dict[str, Any]  # QC metrics specific to single-cell
    doublet_detection_results: Dict[str, Any]  # Doublet detection outcomes
    marker_genes: Dict[str, Any]  # Marker genes per cluster
    file_paths: List[str]  # Paths to input/output files
    methodology_parameters: Dict[str, Any]
    data_context: str  # Single-cell data context
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial computations before returning to supervisor


class BulkRNASeqExpertState(AgentState):
    """
    State for the bulk RNA-seq expert agent.
    """

    next: str

    # Bulk RNA-seq specific context
    task_description: str  # Description of the current task
    analysis_results: Dict[str, Any]  # Bulk RNA-seq analysis results, DE genes, etc.
    differential_expression_results: Dict[str, Any]  # DE analysis outcomes
    pathway_enrichment_results: Dict[str, Any]  # Pathway analysis results
    experimental_design: Dict[str, Any]  # Sample grouping and experimental setup
    quality_control_metrics: Dict[str, Any]  # QC metrics specific to bulk RNA-seq
    statistical_parameters: Dict[str, Any]  # Statistical method parameters
    file_paths: List[str]  # Paths to input/output files
    methodology_parameters: Dict[str, Any]
    data_context: str  # Bulk RNA-seq data context
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial computations before returning to supervisor


class DataExpertState(AgentState):
    """
    State for the data expert agent.
    """

    next: str

    # Data management specific context
    task_description: str  # Description of the current task
    available_datasets: Dict[str, Any]  # Catalog of available datasets
    current_dataset_id: str  # Currently loaded dataset identifier
    dataset_metadata: Dict[str, Any]  # Metadata for current dataset
    download_history: List[Dict[str, Any]]  # History of downloaded datasets
    data_sources: List[str]  # Available data sources (GEO, custom, etc.)

    # Standard expert fields (for consistency)
    file_paths: List[str]  # Paths to downloaded files, loaded datasets
    methodology_parameters: Dict[str, Any]  # Download and loading parameters
    data_context: str  # Data management context
    intermediate_outputs: Dict[
        str, Any
    ]  # Partial download/load results before returning to supervisor


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


class MethodState(AgentState):
    """
    State for the method expert agent.
    """

    next: str

    # Methodology-specific context
    task_description: str  # Description of the current task
    methods_information: Dict[
        str, Any
    ]  # Details about computational/experimental methods
    data_context: str
    evaluation_metrics: Dict[str, Any]  # Accuracy, runtime, reproducibility metrics
    recommendations: List[str]  # Suggested methods or pipelines
    references: List[str]

    # Standard expert fields (for consistency)
    file_paths: List[str]  # Paths to method documentation, protocols, exports
    intermediate_outputs: Dict[
        str, Any
    ]  # Partial method analysis before handoff to supervisor


class MachineLearningExpertState(AgentState):
    """
    State for the machine learning expert agent.
    """

    next: str

    # Machine learning specific context
    task_description: str  # Description of the current task
    ml_ready_modalities: Dict[str, Any]  # Assessment of modalities ready for ML
    feature_engineering_results: Dict[str, Any]  # Feature preparation outcomes
    data_splits: Dict[str, Any]  # Train/test/validation split information
    exported_datasets: Dict[str, Any]  # Framework export results and paths
    ml_metadata: Dict[str, Any]  # ML-specific metadata and preprocessing info
    framework_exports: List[str]  # List of export formats and paths
    file_paths: List[str]  # Paths to ML-ready files
    methodology_parameters: Dict[str, Any]  # ML method parameters used
    data_context: str  # ML data context and characteristics
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial ML computations before returning to supervisor


class VisualizationExpertState(AgentState):
    """
    State for the visualization expert agent.
    """

    next: str

    # Visualization specific context
    task_description: str  # Description of the current task
    current_request: Dict[str, Any]  # Current visualization request details
    last_plot_id: str  # Last created plot ID for tracking
    visualization_history: List[Dict[str, Any]]  # History of created visualizations
    plot_metadata: Dict[str, Any]  # Metadata for current plot session
    active_modalities: List[str]  # Modalities currently being visualized
    visualization_parameters: Dict[str, Any]  # Current visualization parameters
    plot_queue: List[Dict[str, Any]]  # Queue of pending visualization tasks
    file_paths: List[str]  # Paths to saved visualization files
    methodology_parameters: Dict[str, Any]  # Visualization method parameters
    data_context: str  # Visualization data context
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial visualization work before returning to supervisor


class CustomFeatureAgentState(AgentState):
    """
    State for the custom feature creation agent.

    This agent uses Claude Code SDK to generate new agents, services,
    providers, tools, tests, and documentation following Lobster patterns.
    """

    next: str

    # Feature creation specific context
    task_description: str  # Description of the feature creation task
    feature_name: str  # Name of the feature being created
    feature_type: str  # Type: agent, service, provider, agent_with_service
    requirements: str  # Detailed feature requirements
    research_findings: Dict[str, Any]  # Internet research results (Linkup)
    created_files: List[str]  # List of files created during feature generation
    validation_errors: List[str]  # Validation errors encountered
    sdk_output: str  # Output from Claude Code SDK
    integration_instructions: str  # Instructions for manual integration steps
    test_results: Dict[str, Any]  # Results from automated testing
    file_paths: List[str]  # Paths to created files
    methodology_parameters: Dict[str, Any]  # Feature creation parameters
    data_context: str  # Context about the feature being created
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial feature creation work before returning to supervisor


class ProteinStructureVisualizationExpertState(AgentState):
    """
    State for the protein structure visualization expert agent.

    This agent specializes in fetching protein structures from PDB,
    creating ChimeraX visualizations, performing structural analysis,
    and linking structures to omics data.
    """

    next: str

    # Protein structure specific context
    task_description: str  # Description of the current task
    structure_data: Dict[str, Any]  # Current protein structure data
    pdb_ids: List[str]  # List of PDB IDs being worked with
    visualization_settings: Dict[str, Any]  # ChimeraX visualization parameters
    analysis_results: Dict[
        str, Any
    ]  # Structure analysis results (RMSD, secondary structure, geometry)
    comparison_results: Dict[str, Any]  # RMSD comparison results between structures
    metadata: Dict[str, Any]  # PDB metadata (organism, resolution, experiment method)
    file_paths: List[str]  # Paths to structure files and visualizations
    methodology_parameters: Dict[str, Any]  # Analysis parameters and settings
    data_context: str  # Structural biology context
    intermediate_outputs: Dict[
        str, Any
    ]  # For partial structure analysis work before returning to supervisor
