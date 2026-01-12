"""
Data Expert Assistant for LLM-based operations.

This lightweight assistant handles ONLY LLM orchestration for the data_expert agent.
Pure logic has been moved to services (ModalityDetectionService, StrategyRecommendationService).

Responsibilities:
- LLM-based modality classification (detect_modality)
- LLM-based strategy extraction (extract_strategy_config)
- Formatting utilities for LLM prompts
- Data sanitization for Pydantic models

Non-Responsibilities (moved to services):
- Rule-based platform detection → ModalityDetectionService
- Strategy recommendation logic → StrategyRecommendationService
- Configuration management → config.py
"""

import ast
import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# PYDANTIC MODELS (from old data_expert_assistant.py)
# =============================================================================


class StrategyConfig(BaseModel):
    """Configuration for GEO dataset download strategy (LLM-extracted)."""

    summary_file_name: str = Field(
        default="",
        description="name of the summary or overview file (has 'GSE' in the name). Example: 'GSE131907_Lung_Cancer_Feature_Summary'",
    )
    summary_file_type: str = Field(
        default="", description="Filetype of the Summary File. Example: 'xlsx'"
    )
    processed_matrix_name: str = Field(
        default="",
        description="name of the TARGET processed (log2, tpm, normalized, etc) matrix file (preference is non R objects). Example: 'GSE131907_Lung_Cancer_normalized_log2TPM_matrix'",
    )
    processed_matrix_filetype: str = Field(
        default="",
        description="filetype of the TARGET processed file (preference is non R objects). Example: 'txt'",
    )
    raw_UMI_like_matrix_name: str = Field(
        default="",
        description="name of the raw TARGET UMI-like (UMI, raw, tar) matrix file (preference is non R objects). Example: 'GSE131907_Lung_Cancer_raw_UMI_matrix'",
    )
    raw_UMI_like_matrix_filetype: str = Field(
        default="",
        description="filetype of the TARGET raw UMI-like file (preference is non R objects). Example: 'txt'",
    )
    cell_annotation_name: str = Field(
        default="",
        description="Filename of the file which likely is linked to the cell annotation. Example:'GSE131907_Lung_Cancer_cell_annotation'",
    )
    cell_annotation_filetype: str = Field(
        default="", description="Filetype of cell annotation file. Example: 'txt'"
    )
    raw_data_available: bool = Field(
        default=False,
        description="If based on the metadata raw data is available for the samples. Sometimes it stated in the study description",
    )


class ModalityDetectionResult(BaseModel):
    """Result of LLM-based modality detection from GEO metadata."""

    modality: str = Field(
        description=(
            "Detected sequencing modality. Valid values: "
            "bulk_rna, scrna_10x, scrna_smartseq, scatac_10x, "
            "multiome_gex_atac, cite_seq, spatial_visium, perturb_seq, "
            "microarray, unknown"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for modality classification (0.0-1.0)",
    )
    is_supported: bool = Field(
        description="Whether Lobster currently supports this modality"
    )
    compatibility_reason: str = Field(
        description="Human-readable explanation of compatibility decision"
    )
    detected_signals: List[str] = Field(
        default_factory=list,
        description="Key signals used for classification (file names, keywords)",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="User guidance for unsupported modalities",
    )


# =============================================================================
# ASSISTANT CLASS (LLM-only operations)
# =============================================================================


class DataExpertAssistant:
    """
    Lightweight LLM orchestrator for data expert agent.

    This assistant handles ONLY LLM-based operations:
    - Modality classification via LLM
    - Strategy extraction via LLM
    - Data formatting for LLM consumption
    - Response parsing and sanitization

    All rule-based logic has been moved to services.
    """

    def __init__(self):
        """Initialize the Data Expert Assistant."""
        self.settings = get_settings()
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of LLM using provider-agnostic factory."""
        if self._llm is None:
            from lobster.core.workspace import resolve_workspace

            llm_params = self.settings.get_agent_llm_params("data_expert_assistant")
            workspace_path = resolve_workspace(explicit_path=None, create=False)
            self._llm = create_llm(
                "data_expert_assistant", llm_params, workspace_path=workspace_path
            )
        return self._llm

    def _sanitize_null_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize various null value representations to ensure consistency.

        Args:
            data: Dictionary that may contain various null representations

        Returns:
            Dictionary with null values properly sanitized
        """
        sanitized = {}

        for key, value in data.items():
            # Handle various null representations
            if value is None:
                # Actual Python None
                if key == "raw_data_available":
                    sanitized[key] = False
                else:
                    sanitized[key] = "NA"
            elif isinstance(value, str):
                # Check if string looks like a list representation
                if value.strip().startswith("[") and value.strip().endswith("]"):
                    try:
                        # Parse string-encoded list using ast.literal_eval for safety
                        sanitized[key] = ast.literal_eval(value.strip())
                    except Exception as e:
                        logger.warning(
                            f"Failed to parse list string for key '{key}': {e}"
                        )
                        # Fallback if parsing fails
                        if key in ["detected_signals", "suggestions"]:
                            sanitized[key] = []
                        else:
                            sanitized[key] = value.strip()
                else:
                    # Handle string representations of null
                    null_strings = [
                        "null",
                        "None",
                        "NULL",
                        "none",
                        "nil",
                        "NIL",
                        "n/a",
                        "N/A",
                        "na",
                        "NA",
                    ]
                    if value.strip() in null_strings or value.strip() == "":
                        if key == "raw_data_available":
                            sanitized[key] = False
                        else:
                            sanitized[key] = "NA"
                    else:
                        sanitized[key] = value.strip()
            elif isinstance(value, list):
                # Preserve lists as-is (for detected_signals, suggestions)
                sanitized[key] = value
            elif isinstance(value, bool):
                # Boolean values are fine as-is
                sanitized[key] = value
            elif isinstance(value, (int, float)):
                # Preserve numeric types as-is (for confidence score)
                sanitized[key] = value
            else:
                # Convert any other type to string, but check for null-like values
                str_value = str(value).strip()
                null_strings = [
                    "null",
                    "None",
                    "NULL",
                    "none",
                    "nil",
                    "NIL",
                    "n/a",
                    "N/A",
                    "na",
                    "NA",
                ]
                if str_value in null_strings or str_value == "":
                    if key == "raw_data_available":
                        sanitized[key] = False
                    else:
                        sanitized[key] = "NA"
                else:
                    sanitized[key] = str_value

        return sanitized

    def _format_supplementary_files_for_llm(
        self,
        metadata: Dict[str, Any],
        max_series_files: int = 15,
        max_sample_examples: int = 5,
    ) -> str:
        """
        Format supplementary files for LLM consumption.

        Used by both detect_modality and extract_strategy_config to ensure
        consistent file formatting.

        Args:
            metadata: GEO dataset metadata dictionary
            max_series_files: Maximum series-level files to show
            max_sample_examples: Maximum sample files to show as examples

        Returns:
            Formatted string for LLM prompt
        """
        series_files = metadata.get("supplementary_file", [])
        samples = metadata.get("samples", {})

        # Collect all supplementary files (series-level + sample-level)
        file_list = series_files if isinstance(series_files, list) else []

        # Add sample-level supplementary files
        sample_file_count = 0
        sample_file_examples = []
        for gsm_id, sample_meta in list(samples.items())[:10]:  # Check first 10 samples
            sample_files = sample_meta.get("supplementary_file", [])
            if isinstance(sample_files, list) and sample_files:
                sample_file_count += len(sample_files)
                if len(sample_file_examples) < max_sample_examples:
                    sample_file_examples.extend(sample_files[:2])  # 2 files per sample

        # Format file display for LLM
        file_display_parts = []

        # Series-level files
        if file_list:
            series_sample = file_list[:max_series_files]
            series_display = "\n".join([f"  - {f}" for f in series_sample])
            if len(file_list) > max_series_files:
                series_display += (
                    f"\n  ... and {len(file_list) - max_series_files} more series-level files"
                )
            file_display_parts.append(
                f"Series-level files ({len(file_list)} total):\n{series_display}"
            )

        # Sample-level files (examples)
        if sample_file_examples:
            sample_display = "\n".join([f"  - {f}" for f in sample_file_examples[:10]])
            if sample_file_count > len(sample_file_examples):
                sample_display += f"\n  ... and {sample_file_count - len(sample_file_examples)} more sample-level files across {len(samples)} samples"
            file_display_parts.append(
                f"Sample-level files ({sample_file_count} total across {len(samples)} samples, showing examples):\n{sample_display}"
            )

        file_display = (
            "\n\n".join(file_display_parts)
            if file_display_parts
            else "No supplementary files listed"
        )

        return file_display

    def detect_modality(
        self, metadata: Dict[str, Any], geo_id: str
    ) -> Optional[ModalityDetectionResult]:
        """
        Detect sequencing modality from GEO metadata using LLM analysis.

        NOTE: This method uses LLM for classification. For rule-based platform
        detection from file patterns, use ModalityDetectionService instead.

        Args:
            metadata: GEO dataset metadata dictionary
            geo_id: GEO series identifier (e.g., "GSE156793")

        Returns:
            ModalityDetectionResult if successful, None if LLM analysis fails
        """
        try:
            # Import service for pre-filtering
            from lobster.services.data_management.modality_detection_service import (
                ModalityDetectionService,
            )

            detection_service = ModalityDetectionService()

            # PRE-FILTER: Check for multi-omics keywords before LLM call
            pre_filtered_modality = detection_service.pre_filter_multiomics_keywords(
                metadata
            )
            if pre_filtered_modality:
                logger.info(
                    f"Pre-filter detected {pre_filtered_modality} for {geo_id}, skipping LLM"
                )
                return ModalityDetectionResult(
                    modality=pre_filtered_modality,
                    confidence=0.95,  # High confidence from keyword matching
                    is_supported=False,  # All multi-omics are unsupported
                    compatibility_reason=(
                        f"Dataset mentions both RNA and additional modalities ({pre_filtered_modality}). "
                        "Multi-omics data may be in controlled-access repositories (dbGaP) and not visible in GEO files."
                    ),
                    detected_signals=[
                        "Keyword pattern detected in metadata (pre-filter)",
                        f"Title/Summary/Design mentions multi-omics modality: {pre_filtered_modality}",
                    ],
                    suggestions=[
                        f"Wait for Lobster v2.6+ with {pre_filtered_modality} support",
                        "Check if controlled-access data (dbGaP) contains additional modalities",
                        "Manually download only the RNA component if available in GEO",
                    ],
                )

            # Extract key metadata fields
            title = metadata.get("title", "N/A")
            summary = metadata.get("summary", "N/A")
            overall_design = metadata.get("overall_design", "N/A")
            platforms = metadata.get("platforms", {})

            # Use shared utility for file formatting
            file_display = self._format_supplementary_files_for_llm(metadata)

            # Format platform information
            platform_display = ""
            for gpl_id, platform_data in platforms.items():
                platform_title = platform_data.get("title", "Unknown")
                platform_display += f"  - {gpl_id}: {platform_title}\n"

            # Prepare metadata context for LLM
            metadata_context = f"""
Dataset: {geo_id}

Title: {title}

Summary: {summary}

Overall Design: {overall_design}

Platforms:
{platform_display if platform_display else 'No platform information'}

Supplementary Files:
{file_display}
"""

            # Get the schema from ModalityDetectionResult
            modality_schema = ModalityDetectionResult.model_json_schema()

            # Create system prompt for LLM
            system_prompt = f"""You are a bioinformatics expert analyzing GEO dataset metadata to classify sequencing modality.

The ModalityDetectionResult schema is:
{json.dumps(modality_schema, indent=2)}

**CLASSIFICATION DECISION TREE**:

STEP 1: Text Keyword Matching (HIGHEST PRIORITY)
- "chromatin accessibility" + "gene expression" → multiome_gex_atac
- "CITE-seq" OR "antibody" + "RNA" → cite_seq
- "spatial" + ("transcriptomics" OR "Visium") → spatial_visium
- ONLY "chromatin accessibility" (no RNA keywords) → scatac_10x

STEP 2: File Pattern Analysis (ONLY if Step 1 found no multi-omics keywords)
- 10X files (mtx, barcodes, features) → scrna_10x
- Smart-seq files (per-sample) → scrna_smartseq
- Bulk count matrices → bulk_rna

**Supported Modalities**: bulk_rna, scrna_10x, scrna_smartseq
**Unsupported Modalities**: multiome_gex_atac, cite_seq, spatial_visium, scatac_10x, perturb_seq

**CRITICAL INSTRUCTIONS**:
1. Return ONLY valid JSON. Do not include markdown code blocks.
2. DO NOT use null, None - use empty strings "" for unknown fields.
3. Set confidence based on signal clarity (0.0-1.0)
4. For unsupported modalities, provide 3-5 specific suggestions.
5. Include key signals in detected_signals field.

Return only a valid JSON object that matches the ModalityDetectionResult schema."""

            # Create the prompt
            prompt = f"""Analyze this GEO dataset and classify its sequencing modality:

{metadata_context}

Return only a valid JSON object conforming to the ModalityDetectionResult schema."""

            # Invoke the LLM
            logger.info(f"Invoking LLM for modality detection: {geo_id}")
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            # Extract the JSON from the response
            response_text = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content)
            )

            logger.debug(f"LLM response for {geo_id}: {response_text[:200]}...")

            # Parse the JSON response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                modality_dict = json.loads(json_match.group())
            else:
                modality_dict = json.loads(response_text)

            # Sanitize null values
            sanitized_dict = self._sanitize_null_values(modality_dict)

            # Ensure lists are properly initialized
            if (
                "detected_signals" not in sanitized_dict
                or sanitized_dict["detected_signals"] == "NA"
                or isinstance(sanitized_dict.get("detected_signals"), str)
            ):
                sanitized_dict["detected_signals"] = []
            if (
                "suggestions" not in sanitized_dict
                or sanitized_dict["suggestions"] == "NA"
                or isinstance(sanitized_dict.get("suggestions"), str)
            ):
                sanitized_dict["suggestions"] = []

            # Create ModalityDetectionResult object
            modality_result = ModalityDetectionResult(**sanitized_dict)

            logger.info(
                f"Successfully detected modality for {geo_id}: "
                f"{modality_result.modality} (confidence: {modality_result.confidence:.2f}, "
                f"supported: {modality_result.is_supported})"
            )

            return modality_result

        except Exception as e:
            logger.warning(
                f"Failed to detect modality using LLM for {geo_id}: {e}",
                exc_info=True,
            )
            return None

    def extract_strategy_config(
        self, metadata: Dict[str, Any], geo_id: str
    ) -> Optional[StrategyConfig]:
        """
        Extract strategy configuration from GEO metadata using LLM.

        Args:
            metadata: GEO dataset metadata dictionary
            geo_id: GEO accession number

        Returns:
            StrategyConfig object if successful, None otherwise
        """
        try:
            # Extract key metadata fields
            title = metadata.get("title", "N/A")
            summary = metadata.get("summary", "N/A")
            overall_design = metadata.get("overall_design", "N/A")

            # Use shared utility for file formatting
            file_display = self._format_supplementary_files_for_llm(metadata)

            # Prepare the context for LLM
            metadata_context = f"""
Title: {title}

Summary: {summary}

Overall Design: {overall_design}

Supplementary Files:
{file_display}
"""

            # Get the schema from StrategyConfig class
            strategy_schema = StrategyConfig.model_json_schema()

            # Create system prompt using the schema
            system_prompt = f"""You are a bioinformatics expert analyzing GEO dataset metadata. Your task is to extract file information and populate a StrategyConfig object.

The StrategyConfig schema is:
{json.dumps(strategy_schema, indent=2)}

Important notes:
1. Extract filename WITHOUT extension (e.g., "GSE131907_summary" not "GSE131907_summary.xlsx")
2. Extract file extension separately (e.g., "xlsx", "txt", "csv")
3. Prefer non-R objects (.txt, .csv) over .rds files
4. Check for raw data availability in study description
5. Use empty string "" for unknown fields, false for boolean
6. DO NOT use null, None

**RAW.tar File Handling**:
- Files like "GSE*_RAW.tar" contain bundled per-sample data
- Set raw_data_available: true when RAW.tar present
- Files inside tarball not directly listed

Return only valid JSON matching the StrategyConfig schema."""

            # Create the prompt
            prompt = f"""Extract file information from this GEO dataset:

{metadata_context}

Return only a valid JSON object that conforms to the StrategyConfig schema."""

            # Invoke the LLM
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )

            # Extract the JSON from the response
            response_text = (
                response.content[0].text
                if hasattr(response.content[0], "text")
                else str(response.content)
            )

            # Parse the JSON response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                strategy_dict = json.loads(json_match.group())
            else:
                strategy_dict = json.loads(response_text)

            # Sanitize null values
            sanitized_dict = self._sanitize_null_values(strategy_dict)

            # Create StrategyConfig object
            strategy_config = StrategyConfig(**sanitized_dict)

            logger.info(f"Successfully extracted strategy config for {geo_id}")
            return strategy_config

        except Exception as e:
            logger.warning(
                f"Failed to extract strategy config using LLM for {geo_id}: {e}"
            )
            return None

    def _format_file_display(self, filename: str, filetype: str) -> str:
        """
        Format filename and filetype for display, handling NA values.

        Args:
            filename: The filename (may be "NA")
            filetype: The filetype (may be "NA")

        Returns:
            Formatted string for display
        """
        if not filename or filename in ["NA", "Not found", ""]:
            return "Not found"

        if not filetype or filetype in ["NA", "Not found", ""]:
            return filename

        return f"{filename}.{filetype}"

    def format_strategy_section(self, strategy_config: StrategyConfig) -> str:
        """
        Format the strategy configuration into a readable section.

        Args:
            strategy_config: StrategyConfig object

        Returns:
            Formatted string section
        """
        config_dict = strategy_config.model_dump()

        # Format each file type with proper null handling
        summary_file = self._format_file_display(
            config_dict.get("summary_file_name", ""),
            config_dict.get("summary_file_type", ""),
        )

        processed_matrix = self._format_file_display(
            config_dict.get("processed_matrix_name", ""),
            config_dict.get("processed_matrix_filetype", ""),
        )

        raw_matrix = self._format_file_display(
            config_dict.get("raw_UMI_like_matrix_name", ""),
            config_dict.get("raw_UMI_like_matrix_filetype", ""),
        )

        cell_annotations = self._format_file_display(
            config_dict.get("cell_annotation_name", ""),
            config_dict.get("cell_annotation_filetype", ""),
        )

        strategy_section = f"""

📁 **Extracted File Strategy Configuration:**
- **Summary File:** {summary_file}
- **Processed Matrix:** {processed_matrix}
- **Raw/UMI Matrix:** {raw_matrix}
- **Cell Annotations:** {cell_annotations}
- **Raw Data Available:** {'Yes' if config_dict.get('raw_data_available', False) else 'No (see study description for details)'}
"""

        return strategy_section
