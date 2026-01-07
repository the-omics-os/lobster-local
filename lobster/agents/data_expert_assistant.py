"""
Data Expert Assistant for LLM-based operations.

This module handles all LLM-based strategy extraction and decision making
for the Data Expert Agent, keeping the main agent file clean and focused
on data operations.
"""

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class StrategyConfig(BaseModel):
    """Configuration for GEO dataset download strategy."""

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


class DataExpertAssistant:
    """Assistant class for handling LLM-based operations for Data Expert."""

    def __init__(self):
        """Initialize the Data Expert Assistant."""
        self.settings = get_settings()
        self._llm = None

    def _sanitize_null_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize various null value representations to ensure consistency.

        Args:
            data: Dictionary that may contain various null representations

        Returns:
            Dictionary with null values properly sanitized
        """
        import ast

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
                        # Broader exception catching for any parsing failure
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

    @property
    def llm(self):
        """Lazy initialization of LLM using provider-agnostic factory."""
        if self._llm is None:
            llm_params = self.settings.get_agent_llm_params("data_expert_assistant")
            self._llm = create_llm("data_expert_assistant", llm_params)
        return self._llm

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
                series_display += f"\n  ... and {len(file_list) - max_series_files} more series-level files"
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

    def _pre_filter_multiomics_keywords(
        self, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Pre-filter metadata for multi-omics keyword patterns.

        This catches edge cases where multi-omics data (ATAC, protein) is in
        controlled-access repositories (dbGaP) and not visible in GEO files.

        Returns:
            Modality string if keyword pattern detected, None otherwise
        """
        # Combine all text fields for searching
        text = " ".join(
            [
                str(metadata.get("title", "")),
                str(metadata.get("summary", "")),
                str(metadata.get("overall_design", "")),
            ]
        ).lower()

        # Check for multiome keywords
        has_chromatin = "chromatin accessibility" in text or "atac" in text
        has_rna = any(
            kw in text
            for kw in ["gene expression", "rna-seq", "rna seq", "transcriptome"]
        )
        if has_chromatin and has_rna:
            logger.info(
                "[PRE-FILTER] Detected multiome keywords: chromatin_accessibility + RNA"
            )
            return "multiome_gex_atac"

        # Check for CITE-seq keywords
        if "cite-seq" in text or ("antibody" in text and has_rna):
            logger.info("[PRE-FILTER] Detected CITE-seq keywords")
            return "cite_seq"

        # Check for spatial keywords
        if "spatial transcriptomics" in text or "visium" in text:
            logger.info("[PRE-FILTER] Detected spatial keywords")
            return "spatial_visium"

        # Check for Perturb-seq keywords
        if "perturb-seq" in text or ("crispr" in text and has_rna):
            logger.info("[PRE-FILTER] Detected Perturb-seq keywords")
            return "perturb_seq"

        # Check for scATAC (chromatin only, no RNA)
        if has_chromatin and not has_rna:
            logger.info("[PRE-FILTER] Detected scATAC keywords")
            return "scatac_10x"

        return None

    def detect_modality(
        self, metadata: Dict[str, Any], geo_id: str
    ) -> Optional[ModalityDetectionResult]:
        """
        Detect sequencing modality from GEO metadata using LLM analysis.

        This method uses LLM natural language understanding to classify
        datasets into specific modalities (bulk RNA, 10X single-cell, multiome, etc.)
        based on study descriptions and file signatures.

        Args:
            metadata: GEO dataset metadata dictionary with keys:
                - title: Study title
                - summary: Study summary/abstract
                - overall_design: Experimental design description
                - supplementary_file: List of supplementary file names
                - platforms: Platform information (GPL IDs)
            geo_id: GEO series identifier (e.g., "GSE156793")

        Returns:
            ModalityDetectionResult if successful, None if LLM analysis fails

        Examples:
            >>> assistant = DataExpertAssistant()
            >>> result = assistant.detect_modality(metadata, "GSE156793")
            >>> if not result.is_supported:
            ...     print(f"Unsupported modality: {result.modality}")
            ...     for suggestion in result.suggestions:
            ...         print(f"  - {suggestion}")
        """
        try:
            # PRE-FILTER: Check for multi-omics keywords before LLM call
            # This catches edge cases where multi-omics data is in dbGaP (not visible in GEO)
            pre_filtered_modality = self._pre_filter_multiomics_keywords(metadata)
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
            metadata.get("samples", {})

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

            # Create system prompt with STRICT JSON-only instruction
            system_prompt = f"""You are a bioinformatics expert analyzing GEO dataset metadata to classify sequencing modality.

The ModalityDetectionResult schema is:
{json.dumps(modality_schema, indent=2)}

**MANDATORY PRE-CHECK** (MUST perform BEFORE file analysis):

🔴 STOP! Before analyzing file patterns, you MUST scan the Title + Summary + Overall Design for these EXACT PHRASES:
1. "chromatin accessibility" → If found with "gene expression" OR "RNA" → MUST classify as multiome_gex_atac
2. "CITE-seq" OR "antibody capture" → MUST classify as cite_seq
3. "spatial transcriptomics" OR "Visium" → MUST classify as spatial_visium

These keyword combinations OVERRIDE any file patterns you see. Even if files only show RNA data, the keywords take precedence because multi-omics data may be in controlled-access repositories (dbGaP).

**CLASSIFICATION DECISION TREE**:

STEP 1: Text Keyword Matching (HIGHEST PRIORITY)
- "chromatin accessibility" + "gene expression" → multiome_gex_atac (EVEN IF NO ATAC FILES)
- "CITE-seq" OR "antibody" + "RNA" → cite_seq
- "spatial" + ("transcriptomics" OR "Visium") → spatial_visium
- ONLY "chromatin accessibility" (no RNA keywords) → scatac_10x

STEP 2: File Pattern Analysis (ONLY if Step 1 found no multi-omics keywords)
- 10X files (mtx, barcodes, features) → scrna_10x
- Smart-seq files (per-sample) → scrna_smartseq
- Bulk count matrices → bulk_rna

**Modality Classification Guidelines**:

1. **bulk_rna**: Standard bulk RNA-seq
   - Keywords: "bulk RNA-seq", "total RNA", "polyA-selected"
   - Sample count: Typically <100 samples
   - File patterns: *_counts.txt, *_matrix.txt

2. **scrna_10x**: 10X Chromium single-cell RNA-seq
   - Keywords: "10X", "Chromium", "single-cell", "scRNA-seq"
   - File patterns: matrix.mtx, barcodes.tsv, features.tsv, *_filtered_feature_bc_matrix*

3. **scrna_smartseq**: Smart-seq2/Smart-seq3 full-length single-cell
   - Keywords: "Smart-seq", "full-length", "single-cell"
   - File patterns: Per-sample FASTQ files, *_RSEM*, *_kallisto*

4. **multiome_gex_atac**: 10X Multiome (joint GEX + ATAC)
   - **CRITICAL DETECTION RULE**: If summary/title/design mentions ANY of these keyword pairs → classify as multiome:
     * "gene expression" + "chromatin accessibility"
     * "gene expression" + "ATAC"
     * "RNA" + "chromatin accessibility"
     * "transcriptome" + "chromatin"
     * Explicit mention of "multiome" or "joint profiling"
   - **IMPORTANT**: ATAC data may be in dbGaP (controlled access), not visible in GEO files
   - File patterns (if present): fragments.tsv.gz, atac_*, peaks.bed, *_gex_*, *_atac_*
   - **Prioritize keywords over file patterns** - absence of ATAC files does NOT rule out multiome!

5. **cite_seq**: CITE-seq (RNA + surface proteins)
   - Keywords: "CITE-seq", "antibody", "ADT", "protein"
   - File patterns: features.tsv with "Antibody Capture" rows

6. **spatial_visium**: 10X Visium spatial transcriptomics
   - Keywords: "Visium", "spatial", "spatial transcriptomics"
   - File patterns: spatial/ directory, tissue_positions.csv

7. **scatac_10x**: 10X Chromium single-cell ATAC-seq (chromatin accessibility ONLY)
   - Keywords: "scATAC-seq", "chromatin accessibility", "ATAC"
   - File patterns: fragments.tsv.gz, peaks.bed, singlecell.csv
   - **CRITICAL**: If you see ONLY ATAC files (no RNA) → this is scATAC

8. **perturb_seq**: Perturb-seq (RNA + perturbations)
   - Keywords: "Perturb-seq", "CRISPR", "perturbation", "guide RNA"

**Supported Modalities** (set is_supported=True):
- bulk_rna
- scrna_10x
- scrna_smartseq

**Unsupported Modalities** (set is_supported=False):
- multiome_gex_atac (planned v2.6)
- cite_seq (planned v2.6)
- spatial_visium (planned v2.7)
- scatac_10x (planned v2.7)
- perturb_seq (planned v2.8)
- microarray (should not reach this point - handled by GPL registry)

**CRITICAL INSTRUCTIONS**:
1. Return ONLY valid JSON. Do not include markdown code blocks, explanations, or comments.
2. DO NOT use null, None, or any null-like values - use empty strings "" for unknown string fields.
3. Set confidence based on clarity of signals (0.0-1.0):
   - 0.9-1.0: Strong signals (explicit keywords + matching files)
   - 0.7-0.9: Good signals (keywords OR files match)
   - 0.5-0.7: Weak signals (ambiguous description)
   - <0.5: Very uncertain (use "unknown" modality)
4. For unsupported modalities, provide 3-5 specific suggestions in the suggestions field.
5. Include key signals in detected_signals field (file names, keywords found).

Return only a valid JSON object that matches the ModalityDetectionResult schema."""

            # Log the metadata context for debugging
            logger.info(
                f"[MODALITY DEBUG] Metadata context for {geo_id}:\n{metadata_context[:1000]}..."
            )

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

            # Parse the JSON response (with regex fallback for markdown-wrapped JSON)
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                modality_dict = json.loads(json_match.group())
            else:
                modality_dict = json.loads(response_text)

            # Sanitize null values before creating ModalityDetectionResult
            logger.info(
                f"[MODALITY DEBUG] Raw LLM response for {geo_id}: {response_text[:500]}"
            )
            logger.info(
                f"[MODALITY DEBUG] Before sanitization - detected_signals type: {type(modality_dict.get('detected_signals'))}, value: {str(modality_dict.get('detected_signals'))[:200]}"
            )
            sanitized_dict = self._sanitize_null_values(modality_dict)
            logger.info(
                f"[MODALITY DEBUG] After sanitization - detected_signals type: {type(sanitized_dict.get('detected_signals'))}, value: {str(sanitized_dict.get('detected_signals'))[:200]}"
            )

            # Ensure lists are properly initialized (handle remaining string cases)
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
1. For each file type, extract the filename WITHOUT the extension (e.g., "GSE131907_Lung_Cancer_Feature_Summary" not "GSE131907_Lung_Cancer_Feature_Summary.xlsx")
2. Extract the file extension separately (e.g., "xlsx", "txt", "csv")
3. Prefer non-R objects (.txt, .csv) over .rds files when multiple options exist
4. Check the summary and overall design text for mentions of raw data availability
5. If a field cannot be determined from the metadata, use an empty string "" for string fields and false for boolean fields
6. DO NOT use null, None, or any null-like values - use empty strings for unknown string fields

7. **CRITICAL - RAW.tar File Handling:**
   - Files named like "GSE*_RAW.tar" are bundled archives containing per-sample data files
   - These typically contain matrix files, barcodes, features for 10X/scRNA-seq datasets
   - When you see a RAW.tar file:
     * Set raw_data_available: true
     * The actual data files are INSIDE the tarball and not directly listed in metadata
     * For 10X datasets, expect files like: matrix.mtx, barcodes.tsv, features.tsv
     * Extract the GSE ID from the filename (e.g., GSE248556_RAW.tar → GSE248556)
   - If ONLY a RAW.tar exists with no other specific filenames:
     * Use the GSE ID as the base for matrix names (e.g., "GSE248556" for raw_UMI_like_matrix_name)
     * Leave filetype as empty string since files are bundled
     * Still mark raw_data_available: true

Return only a valid JSON object that matches the StrategyConfig schema."""

            # Create the prompt
            prompt = f"""Given this GEO dataset metadata, extract the file information into a StrategyConfig:

{metadata_context}

Return only a valid JSON object that conforms to the StrategyConfig schema provided in the system prompt.
"""

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
            # Try to extract JSON from the response in case it's wrapped in markdown or other text
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if json_match:
                strategy_dict = json.loads(json_match.group())
            else:
                strategy_dict = json.loads(response_text)

            # Sanitize null values before creating StrategyConfig
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
        # Handle various null-like values
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

    def _has_valid_file(self, filename: str) -> bool:
        """
        Check if a filename represents a valid file (not null/NA/empty).

        Args:
            filename: The filename to check

        Returns:
            True if filename is valid, False otherwise
        """
        if not filename:
            return False

        null_values = [
            "NA",
            "N/A",
            "na",
            "n/a",
            "null",
            "None",
            "NULL",
            "none",
            "",
            " ",
        ]
        return filename.strip() not in null_values

    def analyze_download_strategy(
        self, strategy_config: StrategyConfig, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze the strategy configuration to provide download recommendations.

        Args:
            strategy_config: Extracted strategy configuration
            metadata: Original metadata

        Returns:
            Dictionary with analysis results and recommendations
        """
        analysis = {
            "has_processed_matrix": self._has_valid_file(
                strategy_config.processed_matrix_name
            ),
            "has_raw_matrix": self._has_valid_file(
                strategy_config.raw_UMI_like_matrix_name
            ),
            "has_cell_annotations": self._has_valid_file(
                strategy_config.cell_annotation_name
            ),
            "has_summary": self._has_valid_file(strategy_config.summary_file_name),
            "raw_data_available": strategy_config.raw_data_available,
            "recommendations": [],
        }

        # Generate recommendations based on available files
        if analysis["has_processed_matrix"]:
            analysis["recommendations"].append(
                "MATRIX_FIRST: Use processed matrix for quick analysis (may include preprocessing decisions)"
            )

        if analysis["has_raw_matrix"] or analysis["raw_data_available"]:
            analysis["recommendations"].append(
                "SAMPLES_FIRST: Start from raw data for full control over preprocessing"
            )

        # Check for H5AD files in supplementary files
        supp_files = metadata.get("supplementary_file", [])
        has_h5ad = any(".h5ad" in str(f).lower() for f in supp_files)

        if has_h5ad:
            analysis["has_h5ad"] = True
            analysis["recommendations"].insert(
                0,
                "H5_FIRST: Use H5AD file for pre-processed single-cell data with metadata",
            )
        else:
            analysis["has_h5ad"] = False

        # Default recommendation if no clear strategy
        if not analysis["recommendations"]:
            analysis["recommendations"].append(
                "AUTO: Let the system determine the best approach based on available files"
            )

        return analysis
