"""
State definitions for the bioinformatics multi-agent system.

Following the LangGraph 0.2.x multi-agent template pattern, but expanded to 
capture routing metadata, agent-specific working memory, and intermediate outputs.
"""

from typing import Dict, Any, List
from langgraph.prebuilt.chat_agent_executor import AgentState



class OverallState(AgentState):
    """
    Master state that holds sub-states for each agent and conversation-level memory.
    """
    # Meta routing information
    last_active_agent: str
    conversation_id: str

    # Sub-states for each agent
    data_expert_state: "DataExpertState"
    singlecell_expert_state: "SingleCellExpertState"
    bulk_rnaseq_expert_state: "BulkRNASeqExpertState"
    method_state: "MethodState"
    machine_learning_expert_state: "MachineLearningExpertState"
    


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
    analysis_results: Dict[str, Any]     # Single-cell analysis results, clustering, etc.
    clustering_parameters: Dict[str, Any] # Leiden resolution, batch correction settings
    cell_type_annotations: Dict[str, Any] # Cell type assignment results
    quality_control_metrics: Dict[str, Any] # QC metrics specific to single-cell
    doublet_detection_results: Dict[str, Any] # Doublet detection outcomes
    marker_genes: Dict[str, Any]        # Marker genes per cluster
    file_paths: List[str]               # Paths to input/output files
    methodology_parameters: Dict[str, Any]
    data_context: str                   # Single-cell data context
    intermediate_outputs: Dict[str, Any] # For partial computations before returning to supervisor


class BulkRNASeqExpertState(AgentState):
    """
    State for the bulk RNA-seq expert agent.
    """
    next: str

    # Bulk RNA-seq specific context
    analysis_results: Dict[str, Any]     # Bulk RNA-seq analysis results, DE genes, etc.
    differential_expression_results: Dict[str, Any] # DE analysis outcomes
    pathway_enrichment_results: Dict[str, Any] # Pathway analysis results
    experimental_design: Dict[str, Any]  # Sample grouping and experimental setup
    quality_control_metrics: Dict[str, Any] # QC metrics specific to bulk RNA-seq
    statistical_parameters: Dict[str, Any] # Statistical method parameters
    file_paths: List[str]               # Paths to input/output files
    methodology_parameters: Dict[str, Any]
    data_context: str                   # Bulk RNA-seq data context
    intermediate_outputs: Dict[str, Any] # For partial computations before returning to supervisor


class DataExpertState(AgentState):
    """
    State for the data expert agent.
    """
    next: str

    # Data management specific context
    available_datasets: Dict[str, Any]   # Catalog of available datasets
    current_dataset_id: str              # Currently loaded dataset identifier
    dataset_metadata: Dict[str, Any]     # Metadata for current dataset
    download_history: List[Dict[str, Any]]  # History of downloaded datasets
    data_sources: List[str]              # Available data sources (GEO, custom, etc.)


class MethodState(AgentState):
    """
    State for the method expert agent.
    """
    next: str

    # Methodology-specific context
    methods_information: Dict[str, Any]  # Details about computational/experimental methods
    data_context: str
    evaluation_metrics: Dict[str, Any]   # Accuracy, runtime, reproducibility metrics
    recommendations: List[str]           # Suggested methods or pipelines
    references: List[str]


class MachineLearningExpertState(AgentState):
    """
    State for the machine learning expert agent.
    """
    next: str

    # Machine learning specific context
    ml_ready_modalities: Dict[str, Any]  # Assessment of modalities ready for ML
    feature_engineering_results: Dict[str, Any]  # Feature preparation outcomes
    data_splits: Dict[str, Any]         # Train/test/validation split information
    exported_datasets: Dict[str, Any]   # Framework export results and paths
    ml_metadata: Dict[str, Any]         # ML-specific metadata and preprocessing info
    framework_exports: List[str]        # List of export formats and paths
    file_paths: List[str]               # Paths to ML-ready files
    methodology_parameters: Dict[str, Any] # ML method parameters used
    data_context: str                   # ML data context and characteristics
    intermediate_outputs: Dict[str, Any] # For partial ML computations before returning to supervisor
