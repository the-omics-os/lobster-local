"""
Dynamic Pipeline Strategy Engine for GEO Service.

This module provides a flexible, rule-based system for determining
the optimal processing pipeline based on available data files and
their characteristics extracted via LLM.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PipelineType(Enum):
    """Enumeration of pipeline processing strategies."""

    MATRIX_FIRST = auto()  # Prioritize processed matrix files
    RAW_FIRST = auto()  # Prioritize raw UMI/count matrices
    SUPPLEMENTARY_FIRST = auto()  # Start with supplementary files
    SAMPLES_FIRST = auto()  # Download individual samples
    H5_FIRST = auto()  # Prioritize H5/H5AD files
    ARCHIVE_FIRST = auto()  # Extract from archives first
    FALLBACK = auto()  # Use fallback mechanisms


class DataAvailability(Enum):
    """Data availability levels."""

    COMPLETE = auto()  # All preferred files available
    PARTIAL = auto()  # Some files available
    MINIMAL = auto()  # Only basic files available
    NONE = auto()  # No direct files, need alternatives


@dataclass
class PipelineContext:
    """Context object containing all information for pipeline selection."""

    geo_id: str
    strategy_config: Dict[str, Any]
    metadata: Dict[str, Any]
    available_files: List[str] = field(default_factory=list)
    data_type: str = "single_cell_rna_seq"

    def has_file(self, file_type: str) -> bool:
        """Check if a specific file type is available."""
        return bool(self.strategy_config.get(f"{file_type}_name"))

    def get_file_info(self, file_type: str) -> Tuple[str, str]:
        """Get file name and type for a specific file category."""
        name = self.strategy_config.get(f"{file_type}_name", "")
        filetype = self.strategy_config.get(f"{file_type}_filetype", "")
        return name, filetype

    @property
    def data_availability(self) -> DataAvailability:
        """Assess overall data availability."""
        has_processed = self.has_file("processed_matrix")
        has_raw = self.has_file("raw_UMI_like_matrix")
        has_summary = self.has_file("summary_file")
        has_annotations = self.has_file("cell_annotation")
        raw_available = self.strategy_config.get("raw_data_available", False)

        if has_processed and has_annotations:
            return DataAvailability.COMPLETE
        elif has_processed or has_raw:
            return DataAvailability.PARTIAL
        elif raw_available or has_summary:
            return DataAvailability.MINIMAL
        else:
            return DataAvailability.NONE


class PipelineRule(ABC):
    """Abstract base class for pipeline selection rules."""

    @abstractmethod
    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        """Evaluate if this rule applies to the given context."""
        pass

    @abstractmethod
    def get_priority(self) -> int:
        """Get rule priority (higher = evaluated first)."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of the rule."""
        pass


class ProcessedMatrixRule(PipelineRule):
    """Rule for when processed matrix files are available."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        name, filetype = context.get_file_info("processed_matrix")
        if name and filetype in ["txt", "csv", "tsv", "h5", "h5ad"]:
            logger.info(f"ProcessedMatrixRule matched: {name}.{filetype}")
            return PipelineType.MATRIX_FIRST
        return None

    def get_priority(self) -> int:
        return 100  # High priority

    def get_description(self) -> str:
        return "Use processed matrix when available (preferred for pre-normalized data)"


class RawMatrixRule(PipelineRule):
    """Rule for when raw UMI/count matrices are available."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        name, filetype = context.get_file_info("raw_UMI_like_matrix")
        if name and filetype in ["txt", "csv", "tsv", "mtx", "h5"]:
            logger.info(f"RawMatrixRule matched: {name}.{filetype}")
            return PipelineType.RAW_FIRST
        return None

    def get_priority(self) -> int:
        return 90

    def get_description(self) -> str:
        return "Use raw UMI/count matrix for fresh processing"


class H5FormatRule(PipelineRule):
    """Rule for H5/H5AD format files."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        # Check both processed and raw for H5 formats
        processed_name, processed_type = context.get_file_info("processed_matrix")
        raw_name, raw_type = context.get_file_info("raw_UMI_like_matrix")

        if processed_type in ["h5", "h5ad"] or raw_type in ["h5", "h5ad"]:
            logger.info("H5FormatRule matched: H5/H5AD format detected")
            return PipelineType.H5_FIRST
        return None

    def get_priority(self) -> int:
        return 95  # Very high priority for efficient formats

    def get_description(self) -> str:
        return "Prioritize H5/H5AD formats for efficient loading"


class NoDirectFilesRule(PipelineRule):
    """Rule for when no direct matrix files are available."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        if context.data_availability == DataAvailability.NONE:
            logger.info("NoDirectFilesRule matched: No direct files available")
            return PipelineType.SUPPLEMENTARY_FIRST
        return None

    def get_priority(self) -> int:
        return 50

    def get_description(self) -> str:
        return "Fall back to supplementary files when no direct matrices available"


class SingleCellWithRawDataRule(PipelineRule):
    """Rule for single-cell data with raw data availability."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        is_single_cell = "single_cell" in context.data_type
        raw_available = context.strategy_config.get("raw_data_available", False)

        if (
            is_single_cell
            and raw_available
            and context.data_availability == DataAvailability.MINIMAL
        ):
            logger.info("SingleCellWithRawDataRule matched: Single-cell with raw data")
            return PipelineType.SAMPLES_FIRST
        return None

    def get_priority(self) -> int:
        return 80

    def get_description(self) -> str:
        return "Download individual samples for single-cell with raw data"


class SupplementaryFilesRule(PipelineRule):
    """Rule for datasets that primarily use supplementary files."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        # Check if supplementary files are mentioned in metadata
        suppl_files = context.metadata.get("supplementary_file", [])
        has_suppl = bool(suppl_files)
        no_matrices = not context.has_file("processed_matrix") and not context.has_file(
            "raw_UMI_like_matrix"
        )

        if has_suppl and no_matrices:
            logger.info("SupplementaryFilesRule matched: Supplementary files available")
            return PipelineType.SUPPLEMENTARY_FIRST
        return None

    def get_priority(self) -> int:
        return 70

    def get_description(self) -> str:
        return "Use supplementary files when no standard matrices available"


class DefaultFallbackRule(PipelineRule):
    """Default fallback rule when no other rules match."""

    def evaluate(self, context: PipelineContext) -> Optional[PipelineType]:
        logger.info("DefaultFallbackRule matched: Using fallback pipeline")
        return PipelineType.FALLBACK

    def get_priority(self) -> int:
        return 0  # Lowest priority - always matches

    def get_description(self) -> str:
        return "Default fallback when no specific rules match"


class PipelineStrategyEngine:
    """
    Main engine for determining processing pipeline strategy.

    This engine evaluates rules in priority order and selects
    the most appropriate pipeline for the given context.
    """

    def __init__(self):
        """Initialize the strategy engine with default rules."""
        self.rules: List[PipelineRule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register the default set of rules."""
        default_rules = [
            ProcessedMatrixRule(),
            RawMatrixRule(),
            H5FormatRule(),
            SingleCellWithRawDataRule(),
            NoDirectFilesRule(),
            SupplementaryFilesRule(),
            DefaultFallbackRule(),
        ]

        for rule in default_rules:
            self.register_rule(rule)

    def register_rule(self, rule: PipelineRule):
        """Register a new rule with the engine."""
        self.rules.append(rule)
        # Sort by priority (highest first)
        self.rules.sort(key=lambda r: r.get_priority(), reverse=True)
        logger.debug(
            f"Registered rule: {rule.__class__.__name__} (priority: {rule.get_priority()})"
        )

    def determine_pipeline(self, context: PipelineContext) -> Tuple[PipelineType, str]:
        """
        Determine the best pipeline for the given context.

        Args:
            context: Pipeline context with all necessary information

        Returns:
            Tuple of (PipelineType, description)
        """
        logger.info(f"Evaluating pipeline strategy for {context.geo_id}")
        logger.debug(f"Data availability: {context.data_availability.name}")

        for rule in self.rules:
            pipeline_type = rule.evaluate(context)
            if pipeline_type is not None:
                description = rule.get_description()
                logger.info(
                    f"Selected pipeline: {pipeline_type.name} via {rule.__class__.__name__}"
                )
                return pipeline_type, description

        # This should never happen due to DefaultFallbackRule
        logger.error("No pipeline rule matched - this should not happen!")
        return PipelineType.FALLBACK, "Emergency fallback"

    def get_pipeline_functions(
        self, pipeline_type: "PipelineType | str", geo_service_instance: Any
    ) -> List[Callable]:
        """
        Map pipeline type to actual processing functions.

        Args:
            pipeline_type: The selected pipeline type (PipelineType enum or string name)
            geo_service_instance: Instance of GEOService with the processing methods

        Returns:
            List of processing functions to execute in order
        """
        # Handle string pipeline_type by converting to enum
        if isinstance(pipeline_type, str):
            try:
                # Convert string to PipelineType enum
                pipeline_type = PipelineType[pipeline_type.upper()]
                logger.debug(f"Converted string '{pipeline_type}' to PipelineType enum")
            except KeyError:
                logger.warning(
                    f"Invalid pipeline type string: '{pipeline_type}'. Using FALLBACK pipeline."
                )
                pipeline_type = PipelineType.FALLBACK

        pipeline_map = {
            PipelineType.MATRIX_FIRST: [
                geo_service_instance._try_processed_matrix_first,
                geo_service_instance._try_geoparse_download,
                geo_service_instance._try_supplementary_fallback,
            ],
            PipelineType.RAW_FIRST: [
                geo_service_instance._try_raw_matrix_first,
                geo_service_instance._try_geoparse_download,
                geo_service_instance._try_supplementary_fallback,
            ],
            PipelineType.H5_FIRST: [
                geo_service_instance._try_h5_format_first,
                geo_service_instance._try_geoparse_download,
                geo_service_instance._try_supplementary_fallback,
            ],
            PipelineType.SUPPLEMENTARY_FIRST: [
                geo_service_instance._try_supplementary_first,
                geo_service_instance._try_geoparse_download,
            ],
            PipelineType.SAMPLES_FIRST: [
                geo_service_instance._try_geoparse_download,  # This includes sample downloading
                geo_service_instance._try_supplementary_fallback,
            ],
            PipelineType.ARCHIVE_FIRST: [
                geo_service_instance._try_archive_extraction_first,
                geo_service_instance._try_geoparse_download,
                geo_service_instance._try_supplementary_fallback,
            ],
            PipelineType.FALLBACK: [
                geo_service_instance._try_geoparse_download,
                geo_service_instance._try_supplementary_fallback,
                geo_service_instance._try_emergency_fallback,
            ],
        }

        return pipeline_map.get(pipeline_type, pipeline_map[PipelineType.FALLBACK])

    def explain_strategy(self, context: PipelineContext) -> str:
        """
        Generate a human-readable explanation of the strategy selection.

        Args:
            context: Pipeline context

        Returns:
            Explanation string
        """
        pipeline_type, description = self.determine_pipeline(context)

        explanation = f"""
Pipeline Strategy Analysis for {context.geo_id}:
================================================

Data Availability: {context.data_availability.name}
Data Type: {context.data_type}

Available Files:
- Processed Matrix: {context.get_file_info('processed_matrix')}
- Raw Matrix: {context.get_file_info('raw_UMI_like_matrix')}
- Cell Annotations: {context.get_file_info('cell_annotation')}
- Summary: {context.get_file_info('summary_file')}
- Raw Data Available: {context.strategy_config.get('raw_data_available', False)}

Selected Pipeline: {pipeline_type.name}
Reason: {description}

This pipeline will attempt processing in the following order:
1. Primary approach based on available files
2. Standard GEOparse download as backup
3. Supplementary files as final fallback
"""
        return explanation


def create_pipeline_context(
    geo_id: str,
    strategy_config: Dict[str, Any],
    metadata: Dict[str, Any],
    data_type: str = "single_cell_rna_seq",
) -> PipelineContext:
    """
    Factory function to create a pipeline context.

    Args:
        geo_id: GEO accession ID
        strategy_config: Extracted strategy configuration
        metadata: GEO metadata
        data_type: Type of data

    Returns:
        Configured PipelineContext
    """
    # Extract available files from metadata
    available_files = []
    if "supplementary_file" in metadata:
        suppl_files = metadata["supplementary_file"]
        if isinstance(suppl_files, list):
            available_files.extend(suppl_files)
        else:
            available_files.append(suppl_files)

    return PipelineContext(
        geo_id=geo_id,
        strategy_config=strategy_config,
        metadata=metadata,
        available_files=available_files,
        data_type=data_type,
    )
