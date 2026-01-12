"""
Modality Detection Service for platform-type detection from file patterns.

This service provides stateless, configuration-driven platform detection
extracted from data_expert_assistant.py. It uses file pattern matching
and metadata analysis (no LLM calls) following the 3-tuple pattern.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from lobster.agents.data_expert.config import (
    PLATFORM_SIGNATURES,
    PLATFORM_STRATEGY_CONFIGS,
    get_platform_signature,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

logger = logging.getLogger(__name__)


class ModalityDetectionService:
    """
    Stateless service for platform-type detection from file patterns.

    This service analyzes file patterns and metadata to detect the platform type
    (10x, h5ad, kallisto, salmon, etc.) using configuration-driven logic.
    Does NOT use LLM - purely rule-based detection.
    """

    def __init__(self):
        """Initialize stateless service."""
        logger.debug(f"Initializing {self.__class__.__name__}")

    def detect_platform_from_files(
        self,
        file_paths: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, List[str], AnalysisStep]:
        """
        Detect platform type from file patterns using configuration-driven logic.

        This is a pure, deterministic function that doesn't require LLM.
        It uses PLATFORM_SIGNATURES from config.py for pattern matching.

        Args:
            file_paths: List of file names/paths to analyze
            metadata: Optional metadata dictionary for additional context

        Returns:
            Tuple containing:
            - platform_type: Detected platform (e.g., "10x", "h5ad", "kallisto")
            - confidence: Confidence score (0.0-1.0)
            - warnings: List of detection warnings
            - ir: AnalysisStep for provenance tracking

        Examples:
            >>> service = ModalityDetectionService()
            >>> platform, conf, warns, ir = service.detect_platform_from_files(
            ...     ["matrix.mtx", "barcodes.tsv", "features.tsv"]
            ... )
            >>> print(platform)
            '10x'
        """
        logger.info(f"Detecting platform from {len(file_paths)} files")

        # Initialize detection scores
        detection_scores: Dict[str, float] = {
            platform: 0.0 for platform in PLATFORM_SIGNATURES.keys()
        }
        detected_signals: List[str] = []
        warnings: List[str] = []

        # Convert file paths to lowercase for case-insensitive matching
        file_names_lower = [fp.lower() for fp in file_paths]

        # Score each platform based on file patterns
        for platform_type, signature in PLATFORM_SIGNATURES.items():
            required_patterns = signature["required_patterns"]
            optional_patterns = signature["optional_patterns"]
            exclusion_patterns = signature["exclusion_patterns"]
            confidence_boost = signature["confidence_boost"]

            score = 0.0

            # Check required patterns (must ALL be present)
            required_matches = sum(
                1
                for pattern in required_patterns
                if any(pattern.lower() in fn for fn in file_names_lower)
            )

            if required_patterns and required_matches == len(required_patterns):
                score += confidence_boost * 2.0  # Strong signal
                detected_signals.append(
                    f"{platform_type}: All required files present ({required_patterns})"
                )
            elif required_patterns and required_matches > 0:
                score += confidence_boost * 0.5  # Partial match
                detected_signals.append(
                    f"{platform_type}: Partial required files ({required_matches}/{len(required_patterns)})"
                )

            # Check optional patterns (nice to have)
            optional_matches = sum(
                1
                for pattern in optional_patterns
                if any(pattern.lower() in fn for fn in file_names_lower)
            )

            if optional_matches > 0:
                score += optional_matches * 0.5 * confidence_boost
                detected_signals.append(
                    f"{platform_type}: {optional_matches} optional files matched"
                )

            # Check exclusion patterns (if present, reduce score)
            exclusion_matches = sum(
                1
                for pattern in exclusion_patterns
                if any(pattern.lower() in fn for fn in file_names_lower)
            )

            if exclusion_matches > 0:
                score -= exclusion_matches * 2.0  # Strong penalty
                detected_signals.append(
                    f"{platform_type}: Excluded due to {exclusion_matches} exclusion patterns"
                )

            detection_scores[platform_type] = max(0.0, score)  # Never negative

        # Find platform with highest score
        if not detection_scores or all(s == 0.0 for s in detection_scores.values()):
            # No clear match - default to CSV
            platform_type = "csv"
            confidence = 0.3
            warnings.append(
                "No strong platform signature detected. Defaulting to generic CSV format."
            )
            warnings.append(
                f"Available files: {file_paths[:5]}" + (
                    f"... and {len(file_paths) - 5} more" if len(file_paths) > 5 else ""
                )
            )
        else:
            # Get platform with max score
            platform_type = max(detection_scores, key=detection_scores.get)
            max_score = detection_scores[platform_type]

            # Normalize confidence to 0.0-1.0 range
            # Max possible score is ~6.0 (required + optional + boost)
            confidence = min(1.0, max_score / 6.0)

            # Add warnings for low confidence
            if confidence < 0.6:
                warnings.append(
                    f"Low confidence detection ({confidence:.2f}). "
                    f"Platform: {platform_type}. Consider manual verification."
                )

            # Check for competing platforms
            sorted_scores = sorted(
                detection_scores.items(), key=lambda x: x[1], reverse=True
            )
            if len(sorted_scores) > 1 and sorted_scores[1][1] > 0:
                second_platform, second_score = sorted_scores[1]
                if second_score / max_score > 0.7:  # Within 70% of winner
                    warnings.append(
                        f"Competing platform detected: {second_platform} "
                        f"(score: {second_score:.2f} vs {max_score:.2f})"
                    )

        logger.info(
            f"Platform detection complete: {platform_type} "
            f"(confidence: {confidence:.2f}, signals: {len(detected_signals)})"
        )

        # Create IR for provenance tracking
        ir = self._create_ir(
            file_paths=file_paths,
            platform_type=platform_type,
            confidence=confidence,
            detected_signals=detected_signals,
        )

        return platform_type, confidence, warnings, ir

    def pre_filter_multiomics_keywords(
        self, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Pre-filter metadata for multi-omics keyword patterns.

        This catches edge cases where multi-omics data (ATAC, protein) is in
        controlled-access repositories (dbGaP) and not visible in GEO files.

        Args:
            metadata: GEO dataset metadata with title, summary, overall_design

        Returns:
            Modality string if keyword pattern detected (e.g., "multiome_gex_atac"),
            None otherwise
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

    def _create_ir(
        self,
        file_paths: List[str],
        platform_type: str,
        confidence: float,
        detected_signals: List[str],
    ) -> AnalysisStep:
        """
        Create Intermediate Representation for notebook export.

        Args:
            file_paths: List of file paths analyzed
            platform_type: Detected platform type
            confidence: Detection confidence score
            detected_signals: List of detection signals

        Returns:
            AnalysisStep with code template and parameter schema
        """
        # Parameter schema for Papermill injection
        parameter_schema = {
            "file_paths": ParameterSpec(
                param_type="list[str]",
                papermill_injectable=True,
                default_value=file_paths,
                required=True,
                validation_rule="non_empty_list",
                description="List of file paths to analyze for platform detection",
            ),
            "platform_type": ParameterSpec(
                param_type="str",
                papermill_injectable=False,  # Output, not input
                default_value=platform_type,
                required=False,
                validation_rule=None,
                description="Detected platform type",
            ),
            "confidence": ParameterSpec(
                param_type="float",
                papermill_injectable=False,  # Output, not input
                default_value=confidence,
                required=False,
                validation_rule="between_0_and_1",
                description="Detection confidence score",
            ),
        }

        # Jinja2 code template (ONLY STANDARD LIBRARIES)
        code_template = """# Platform Detection from File Patterns
# Configuration-driven platform detection using file pattern matching

from typing import Dict, List

# Parameters
file_paths = {{ file_paths }}

# Platform signatures (simplified for notebook)
PLATFORM_SIGNATURES = {
    "10x": {"required": ["matrix.mtx", "barcodes.tsv"], "optional": ["features.tsv"]},
    "h5ad": {"required": [".h5ad"], "optional": []},
    "kallisto": {"required": ["abundance.tsv"], "optional": ["abundance.h5"]},
}

# Detection logic
detection_scores = {}
for platform, signature in PLATFORM_SIGNATURES.items():
    score = 0.0
    file_names_lower = [fp.lower() for fp in file_paths]

    # Check required patterns
    required_matches = sum(
        1 for pattern in signature["required"]
        if any(pattern in fn for fn in file_names_lower)
    )
    if required_matches == len(signature["required"]):
        score += 2.0

    detection_scores[platform] = score

# Get platform with highest score
platform_type = max(detection_scores, key=detection_scores.get) if detection_scores else "csv"
confidence = min(1.0, detection_scores.get(platform_type, 0.0) / 4.0)

print(f"Platform detected: {platform_type} (confidence: {confidence:.2f})")
"""

        return AnalysisStep(
            operation="lobster.services.data_management.modality_detection_service.detect_platform_from_files",
            tool_name="detect_platform_from_files",
            description=f"""## Platform Detection

Analyzes file patterns to detect platform type using configuration-driven logic.

**Input Files**: {len(file_paths)} files analyzed
**Detected Platform**: {platform_type}
**Confidence**: {confidence:.2f}
**Signals**: {len(detected_signals)} detection signals

**Detection Signals**:
{chr(10).join(f"- {signal}" for signal in detected_signals[:5])}
""",
            library="lobster",
            code_template=code_template,
            imports=[
                "from typing import Dict, List",
            ],
            parameters={
                "file_paths": file_paths,
                "platform_type": platform_type,
                "confidence": confidence,
            },
            parameter_schema=parameter_schema,
            input_entities=["file_paths"],
            output_entities=["platform_type", "confidence"],
            execution_context={
                "method": "detect_platform_from_files",
                "detected_signals_count": len(detected_signals),
            },
            validates_on_export=True,
            exportable=True,
        )
