"""
State definitions for the transcriptomics multi-agent system.

This module defines state classes for the transcriptomics expert parent agent
and its specialized sub-agents (annotation and differential expression).

Following the LangGraph 0.2.x multi-agent template pattern, with fields
tailored to transcriptomics analysis workflows.
"""

from typing import Any, Dict, List, Optional

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = [
    "TranscriptomicsExpertState",
    "AnnotationExpertState",
    "DEAnalysisExpertState",
]


class TranscriptomicsExpertState(AgentState):
    """
    State for the parent transcriptomics expert agent.

    This agent coordinates between single-cell and bulk RNA-seq workflows,
    managing QC, preprocessing, clustering, and routing to specialized
    sub-agents for annotation and differential expression analysis.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    data_type: str = ""  # "single_cell" or "bulk" (auto-detected from modality)

    # QC state
    quality_metrics: Dict[str, Any] = {}
    preprocessing_applied: bool = False

    # Clustering state (single-cell only)
    clustering_parameters: Dict[str, Any] = {}
    cluster_quality_scores: Dict[str, float] = {}
    marker_genes: Dict[str, List[str]] = {}

    # Cross-cutting
    file_paths: List[str] = []
    intermediate_outputs: Dict[str, Any] = {}


class AnnotationExpertState(AgentState):
    """
    State for the annotation sub-agent.

    This agent handles cell type annotation for single-cell data,
    including manual and automated annotation workflows, confidence
    scoring, and debris/doublet cluster identification.
    """

    next: str = ""

    # Task context
    task_description: str = ""

    # Annotation state
    cell_type_annotations: Dict[str, str] = {}  # cluster_id -> cell_type
    annotation_confidence: Dict[str, float] = {}  # cluster_id -> confidence_score
    debris_clusters: List[str] = []  # cluster IDs marked as debris/doublets
    annotation_template: Optional[str] = (
        None  # template name if using guided annotation
    )
    pending_annotations: List[str] = []  # cluster IDs awaiting annotation

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}


class DEAnalysisExpertState(AgentState):
    """
    State for the differential expression analysis sub-agent.

    This agent handles DE analysis for both single-cell (pseudobulk)
    and bulk RNA-seq data, including experimental design specification,
    formula-based analysis, and pathway enrichment.
    """

    next: str = ""

    # Task context
    task_description: str = ""

    # DE analysis state
    experimental_design: Dict[str, Any] = {}  # sample grouping, covariates, contrasts
    formula: Optional[str] = None  # pyDESeq2/DESeq2 formula string
    de_results: Dict[str, Any] = {}  # contrast_name -> DE results
    iteration_history: List[Dict[str, Any]] = []  # history of DE iterations
    pathway_results: Dict[str, Any] = {}  # enrichment analysis results

    # Cross-cutting
    intermediate_outputs: Dict[str, Any] = {}
