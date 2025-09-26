"""
Expert Handoff Patterns for standardized expert-to-expert collaborations.

This module defines common handoff patterns and provides automatic tool registration
based on available agents in the system.

Coming soon
"""

from typing import Dict, List, Any, Type, Optional
from dataclasses import dataclass

from .enhanced_handoff_tool import SCVI_CONTEXT_SCHEMA, PSEUDOBULK_CONTEXT_SCHEMA, DATA_LOADING_SCHEMA, METHOD_CONTEXT_SCHEMA


@dataclass
class HandoffPattern:
    """Configuration for a handoff pattern between experts."""
    from_expert: str
    to_expert: str
    task_types: List[str]
    context_schema: Dict[str, Type]
    return_flow: str  # "sender" or "supervisor"
    description: str
    priority: int = 0  # Higher priority patterns are preferred


# Define standardized handoff patterns for common expert collaborations
EXPERT_HANDOFF_PATTERNS = {
    "singlecell_to_ml": HandoffPattern(
        from_expert="singlecell_expert",
        to_expert="machine_learning_expert",
        task_types=["scvi_training", "deep_learning_embedding", "neural_network_analysis"],
        context_schema=SCVI_CONTEXT_SCHEMA,
        return_flow="sender",
        description="Single Cell Expert to Machine Learning Expert for deep learning tasks",
        priority=10
    ),

    "singlecell_to_bulk": HandoffPattern(
        from_expert="singlecell_expert",
        to_expert="bulk_rnaseq_expert",
        task_types=["pseudobulk_analysis", "differential_expression", "bulk_conversion"],
        context_schema=PSEUDOBULK_CONTEXT_SCHEMA,
        return_flow="sender",
        description="Single Cell Expert to Bulk RNA-seq Expert for pseudobulk analysis",
        priority=8
    ),

    "data_to_singlecell": HandoffPattern(
        from_expert="data_expert",
        to_expert="singlecell_expert",
        task_types=["singlecell_analysis", "cell_type_annotation", "trajectory_analysis"],
        context_schema={
            "modality_name": str,
            "analysis_type": str,
            "quality_control": Optional[bool],
            "clustering_resolution": Optional[float]
        },
        return_flow="supervisor",
        description="Data Expert to Single Cell Expert for specialized analysis",
        priority=9
    ),

    "data_to_bulk": HandoffPattern(
        from_expert="data_expert",
        to_expert="bulk_rnaseq_expert",
        task_types=["bulk_analysis", "deseq2_analysis", "pathway_analysis"],
        context_schema={
            "modality_name": str,
            "design_formula": str,
            "contrast": Optional[str],
            "analysis_type": str
        },
        return_flow="supervisor",
        description="Data Expert to Bulk RNA-seq Expert for bulk analysis",
        priority=9
    ),

    "data_to_research": HandoffPattern(
        from_expert="data_expert",
        to_expert="research_agent",
        task_types=["dataset_search", "metadata_extraction", "geo_download"],
        context_schema=DATA_LOADING_SCHEMA,
        return_flow="sender",
        description="Data Expert to Research Agent for dataset discovery",
        priority=7
    ),

    "research_to_method": HandoffPattern(
        from_expert="research_agent",
        to_expert="method_expert",
        task_types=["parameter_extraction", "method_recommendation", "protocol_analysis"],
        context_schema=METHOD_CONTEXT_SCHEMA,
        return_flow="sender",
        description="Research Agent to Method Expert for parameter extraction from publications",
        priority=6
    ),

    "research_to_data": HandoffPattern(
        from_expert="research_agent",
        to_expert="data_expert",
        task_types=["data_loading", "dataset_validation", "metadata_integration"],
        context_schema={
            "dataset_id": str,
            "data_source": str,
            "file_paths": List[str],
            "metadata": Optional[Dict[str, Any]]
        },
        return_flow="sender",
        description="Research Agent to Data Expert for dataset loading after discovery",
        priority=8
    ),

    "method_to_singlecell": HandoffPattern(
        from_expert="method_expert",
        to_expert="singlecell_expert",
        task_types=["apply_method", "parameter_optimization", "protocol_execution"],
        context_schema={
            "method_name": str,
            "parameters": Dict[str, Any],
            "modality_name": str,
            "reference_paper": Optional[str]
        },
        return_flow="sender",
        description="Method Expert to Single Cell Expert for applying extracted methods",
        priority=7
    ),

    "method_to_bulk": HandoffPattern(
        from_expert="method_expert",
        to_expert="bulk_rnaseq_expert",
        task_types=["apply_method", "parameter_optimization", "statistical_analysis"],
        context_schema={
            "method_name": str,
            "parameters": Dict[str, Any],
            "modality_name": str,
            "design_matrix": Optional[Dict[str, Any]]
        },
        return_flow="sender",
        description="Method Expert to Bulk RNA-seq Expert for applying statistical methods",
        priority=7
    ),

    "bulk_to_singlecell": HandoffPattern(
        from_expert="bulk_rnaseq_expert",
        to_expert="singlecell_expert",
        task_types=["signature_analysis", "cell_type_deconvolution", "bulk_to_sc_mapping"],
        context_schema={
            "bulk_modality": str,
            "singlecell_modality": str,
            "signature_genes": Optional[List[str]],
            "method": str
        },
        return_flow="sender",
        description="Bulk RNA-seq Expert to Single Cell Expert for signature-based analysis",
        priority=6
    ),

    # Proteomics patterns
    "data_to_ms_proteomics": HandoffPattern(
        from_expert="data_expert",
        to_expert="ms_proteomics_expert",
        task_types=["ms_analysis", "protein_quantification", "proteomics_qc"],
        context_schema={
            "modality_name": str,
            "data_type": str,  # "DDA", "DIA", "TMT", etc.
            "missing_value_strategy": Optional[str],
            "normalization_method": Optional[str]
        },
        return_flow="supervisor",
        description="Data Expert to MS Proteomics Expert for mass spectrometry analysis",
        priority=9
    ),

    "data_to_affinity_proteomics": HandoffPattern(
        from_expert="data_expert",
        to_expert="affinity_proteomics_expert",
        task_types=["affinity_analysis", "olink_analysis", "antibody_validation"],
        context_schema={
            "modality_name": str,
            "panel_type": str,  # "Olink", "Antibody Array", etc.
            "cv_threshold": Optional[float],
            "quality_filters": Optional[Dict[str, Any]]
        },
        return_flow="supervisor",
        description="Data Expert to Affinity Proteomics Expert for targeted protein analysis",
        priority=9
    ),

    "ms_proteomics_to_research": HandoffPattern(
        from_expert="ms_proteomics_expert",
        to_expert="research_agent",
        task_types=["protein_annotation", "pathway_enrichment", "literature_search"],
        context_schema={
            "protein_list": List[str],
            "organism": Optional[str],
            "enrichment_database": Optional[str],
            "context": str
        },
        return_flow="sender",
        description="MS Proteomics Expert to Research Agent for protein functional analysis",
        priority=6
    ),

    "affinity_proteomics_to_research": HandoffPattern(
        from_expert="affinity_proteomics_expert",
        to_expert="research_agent",
        task_types=["biomarker_validation", "clinical_annotation", "panel_optimization"],
        context_schema={
            "biomarkers": List[str],
            "clinical_context": str,
            "validation_studies": Optional[List[str]],
            "panel_name": Optional[str]
        },
        return_flow="sender",
        description="Affinity Proteomics Expert to Research Agent for biomarker validation",
        priority=6
    )
}


def get_handoff_patterns_for_expert(expert_name: str, direction: str = "from") -> List[HandoffPattern]:
    """
    Get all handoff patterns for a specific expert.

    Args:
        expert_name: Name of the expert agent
        direction: "from" for outgoing handoffs, "to" for incoming handoffs

    Returns:
        List of matching handoff patterns
    """
    patterns = []
    for pattern_name, pattern in EXPERT_HANDOFF_PATTERNS.items():
        if direction == "from" and pattern.from_expert == expert_name:
            patterns.append(pattern)
        elif direction == "to" and pattern.to_expert == expert_name:
            patterns.append(pattern)

    # Sort by priority (higher priority first)
    return sorted(patterns, key=lambda p: p.priority, reverse=True)


def get_handoff_pattern(from_expert: str, to_expert: str, task_type: Optional[str] = None) -> Optional[HandoffPattern]:
    """
    Get a specific handoff pattern between two experts.

    Args:
        from_expert: Source expert name
        to_expert: Target expert name
        task_type: Optional specific task type to match

    Returns:
        Matching handoff pattern or None
    """
    matching_patterns = []

    for pattern_name, pattern in EXPERT_HANDOFF_PATTERNS.items():
        if pattern.from_expert == from_expert and pattern.to_expert == to_expert:
            if task_type is None or task_type in pattern.task_types:
                matching_patterns.append(pattern)

    if not matching_patterns:
        return None

    # Return highest priority pattern
    return max(matching_patterns, key=lambda p: p.priority)


def get_available_task_types(from_expert: str, to_expert: str) -> List[str]:
    """
    Get available task types for handoff between two experts.

    Args:
        from_expert: Source expert name
        to_expert: Target expert name

    Returns:
        List of available task types
    """
    pattern = get_handoff_pattern(from_expert, to_expert)
    if pattern:
        return pattern.task_types.copy()
    return []


def list_all_handoff_patterns() -> Dict[str, HandoffPattern]:
    """Get all available handoff patterns."""
    return EXPERT_HANDOFF_PATTERNS.copy()


def validate_handoff_pattern(from_expert: str, to_expert: str, task_type: str) -> bool:
    """
    Validate if a handoff pattern is supported.

    Args:
        from_expert: Source expert name
        to_expert: Target expert name
        task_type: Task type to validate

    Returns:
        True if pattern is supported, False otherwise
    """
    pattern = get_handoff_pattern(from_expert, to_expert, task_type)
    return pattern is not None


def get_context_schema_for_handoff(from_expert: str, to_expert: str, task_type: str) -> Optional[Dict[str, Type]]:
    """
    Get the context schema for a specific handoff.

    Args:
        from_expert: Source expert name
        to_expert: Target expert name
        task_type: Task type

    Returns:
        Context schema dictionary or None
    """
    pattern = get_handoff_pattern(from_expert, to_expert, task_type)
    if pattern:
        return pattern.context_schema.copy()
    return None


def should_return_to_sender(from_expert: str, to_expert: str, task_type: str) -> bool:
    """
    Determine if handoff should return to sender or supervisor.

    Args:
        from_expert: Source expert name
        to_expert: Target expert name
        task_type: Task type

    Returns:
        True if should return to sender, False if should return to supervisor
    """
    pattern = get_handoff_pattern(from_expert, to_expert, task_type)
    if pattern:
        return pattern.return_flow == "sender"
    return True  # Default to returning to sender


def get_handoff_description(from_expert: str, to_expert: str, task_type: str) -> Optional[str]:
    """
    Get description for a specific handoff pattern.

    Args:
        from_expert: Source expert name
        to_expert: Target expert name
        task_type: Task type

    Returns:
        Description string or None
    """
    pattern = get_handoff_pattern(from_expert, to_expert, task_type)
    if pattern:
        return pattern.description
    return None