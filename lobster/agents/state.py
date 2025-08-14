"""
State definitions for the bioinformatics multi-agent system.

Following the LangGraph 0.2.x multi-agent template pattern, but expanded to 
capture routing metadata, agent-specific working memory, and intermediate outputs.
"""

from typing import Dict, Any, List
from langgraph.graph import MessagesState
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
    transcriptomics_expert_state: "TranscriptomicsExpertState"
    method_state: "MethodState"
    


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


class TranscriptomicsExpertState(AgentState):
    """
    State for the transcriptomics expert agent.
    """
    next: str

    # Transcriptomics-specific context
    analysis_results: Dict[str, Any]     # Gene expression, DEG lists, etc.
    file_paths: List[str]            # Paths to input/output files
    methodology_parameters: Dict[str, Any] 
    data_context: str                    # Type/source of transcriptomics data (RNA-seq, microarray)
    quality_control_metrics: Dict[str, Any]
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
