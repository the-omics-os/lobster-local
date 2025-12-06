"""
State definitions for the unified proteomics agent.

This module defines the state class for the proteomics expert agent
which handles both mass spectrometry and affinity proteomics platforms.

Following the LangGraph 0.2.x multi-agent template pattern, with fields
tailored to proteomics analysis workflows.
"""

from typing import Any, Dict, List

from langgraph.prebuilt.chat_agent_executor import AgentState

__all__ = ["ProteomicsExpertState"]


class ProteomicsExpertState(AgentState):
    """
    State for the unified proteomics expert agent.

    This agent handles both mass spectrometry (DDA/DIA) and affinity-based
    (Olink, SomaScan) proteomics analysis, auto-detecting platform type
    and applying appropriate defaults.
    """

    next: str = ""

    # Task context
    task_description: str = ""
    platform_type: str = ""  # "mass_spec" or "affinity" (auto-detected from data)

    # QC state
    quality_metrics: Dict[str, Any] = {}
    preprocessing_applied: bool = False

    # Platform-specific state
    platform_defaults: Dict[str, Any] = {}  # defaults applied based on detection
    missing_value_info: Dict[str, Any] = {}  # MNAR vs MAR patterns
    normalization_info: Dict[str, Any] = {}

    # Analysis state
    differential_results: Dict[str, Any] = {}
    clustering_results: Dict[str, Any] = {}

    # Cross-cutting
    file_paths: List[str] = []
    intermediate_outputs: Dict[str, Any] = {}
