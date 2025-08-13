"""
State definitions for the bioinformatics multi-agent system.

Following the LangGraph 0.2.x multi-agent template pattern, but expanded to 
capture routing metadata, agent-specific working memory, and intermediate outputs.
"""

from typing import Dict, Any, List
from langgraph.graph import MessagesState


class OverallState(MessagesState):
    """
    Master state that holds sub-states for each agent and conversation-level memory.
    """
    # Meta routing information
    last_active_agent: str
    conversation_id: str
    task_history: List[str]

    # Sub-states for each agent
    supervisor_state: "SupervisorState"
    transcriptomics_expert_state: "TranscriptomicsExpertState"
    method_state: "MethodState"
    human_state: "HumanState"


class SupervisorState(MessagesState):
    """
    State for the supervisor agent.
    """
    next: str  # The next node to route to (agent name or END)
    last_active_agent: str

    # Bioinformatics-specific context the supervisor might maintain
    analysis_results: Dict[str, Any]     # Combined results from multiple experts
    methodology_parameters: Dict[str, Any]
    data_context: str                    # High-level description of the data in use

    # Control information
    delegation_history: List[str]        # Track which agents have been called
    pending_tasks: List[str]             


class TranscriptomicsExpertState(MessagesState):
    """
    State for the transcriptomics expert agent.
    """
    next: str

    # Transcriptomics-specific context
    analysis_results: Dict[str, Any]     # Gene expression, DEG lists, etc.
    methodology_parameters: Dict[str, Any] 
    data_context: str                    # Type/source of transcriptomics data (RNA-seq, microarray)
    quality_control_metrics: Dict[str, Any]
    intermediate_outputs: Dict[str, Any] # For partial computations before returning to supervisor


class MethodState(MessagesState):
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


class HumanState(MessagesState):
    """
    State for the human user interaction agent.
    """
    next: str

    # Track what’s been asked and answered
    user_goals: List[str]
    clarifying_questions: List[str]
    confirmed_requirements: Dict[str, Any]