# Transcriptomics Agent Module
# Unified agent for single-cell and bulk RNA-seq analysis

from lobster.agents.transcriptomics.annotation_expert import annotation_expert
from lobster.agents.transcriptomics.config import detect_data_type, get_qc_defaults
from lobster.agents.transcriptomics.de_analysis_expert import de_analysis_expert
from lobster.agents.transcriptomics.prompts import (
    create_annotation_expert_prompt,
    create_de_analysis_expert_prompt,
    create_transcriptomics_expert_prompt,
)
from lobster.agents.transcriptomics.shared_tools import create_shared_tools
from lobster.agents.transcriptomics.state import (
    AnnotationExpertState,
    DEAnalysisExpertState,
    TranscriptomicsExpertState,
)
from lobster.agents.transcriptomics.transcriptomics_expert import transcriptomics_expert

__all__ = [
    # Main agents
    "transcriptomics_expert",
    "annotation_expert",
    "de_analysis_expert",
    # Configuration
    "detect_data_type",
    "get_qc_defaults",
    # Prompts
    "create_transcriptomics_expert_prompt",
    "create_annotation_expert_prompt",
    "create_de_analysis_expert_prompt",
    # Shared tools
    "create_shared_tools",
    # State classes
    "TranscriptomicsExpertState",
    "AnnotationExpertState",
    "DEAnalysisExpertState",
]
