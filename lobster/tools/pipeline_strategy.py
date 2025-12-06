"""
Dynamic Pipeline Strategy Engine for GEO Service.

DEPRECATED: This module has been moved to lobster.services.data_access.geo.strategy.
Please update your imports. This alias will be removed in a future version.

This module provides a flexible, rule-based system for determining
the optimal processing pipeline based on available data files and
their characteristics extracted via LLM.
"""

import warnings

warnings.warn(
    "lobster.tools.pipeline_strategy is deprecated. "
    "Use lobster.services.data_access.geo.strategy instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from new location for backward compatibility
from lobster.services.data_access.geo.strategy import (
    DataAvailability,
    DefaultFallbackRule,
    H5FormatRule,
    NoDirectFilesRule,
    PipelineContext,
    PipelineRule,
    PipelineStrategyEngine,
    PipelineType,
    ProcessedMatrixRule,
    RawMatrixRule,
    SingleCellWithRawDataRule,
    SupplementaryFilesRule,
    create_pipeline_context,
)

__all__ = [
    "PipelineType",
    "DataAvailability",
    "PipelineContext",
    "PipelineRule",
    "ProcessedMatrixRule",
    "RawMatrixRule",
    "H5FormatRule",
    "SupplementaryFilesRule",
    "NoDirectFilesRule",
    "SingleCellWithRawDataRule",
    "DefaultFallbackRule",
    "PipelineStrategyEngine",
    "create_pipeline_context",
]
