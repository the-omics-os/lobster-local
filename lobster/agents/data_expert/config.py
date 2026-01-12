"""
Configuration for Data Expert agent: download strategies, platform detection, and validation.

This module contains all configuration data extracted from the data_expert and
data_expert_assistant modules for better maintainability and testability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "DownloadStrategyConfig",
    "PLATFORM_STRATEGY_CONFIGS",
    "PLATFORM_SIGNATURES",
    "VALIDATION_THRESHOLDS",
    "FILE_SIZE_LIMITS",
    "ADAPTER_REGISTRY",
    "QUEUE_STATUS_MAPPING",
    "SUPPORTED_DOWNLOAD_STRATEGIES",
    "get_strategy_config",
    "get_platform_signature",
]


@dataclass
class DownloadStrategyConfig:
    """
    Configuration for platform-specific download strategies.

    Replaces the old StrategyConfig from data_expert_assistant.py with
    a more comprehensive, configuration-driven approach.
    """

    # Required fields (no defaults)
    platform_type: str
    display_name: str
    description: str
    default_strategy: str
    supported_strategies: List[str]
    prefer_processed: bool
    file_patterns: List[str]
    required_files: List[str]
    validation_level: str  # "strict", "moderate", "permissive"

    # Optional fields (with defaults)
    optional_files: List[str] = field(default_factory=list)
    validation_thresholds: Dict[str, Any] = field(default_factory=dict)
    platform_specific: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# PLATFORM STRATEGY CONFIGURATIONS
# =============================================================================

PLATFORM_STRATEGY_CONFIGS: Dict[str, DownloadStrategyConfig] = {
    "10x": DownloadStrategyConfig(
        platform_type="10x",
        display_name="10x Genomics Single-Cell",
        description="Single-cell RNA-seq data from 10x Genomics platform",
        default_strategy="MATRIX_FIRST",
        supported_strategies=["MATRIX_FIRST", "H5_FIRST", "AUTO"],
        prefer_processed=True,
        file_patterns=["matrix.mtx", "barcodes.tsv", "features.tsv", "genes.tsv"],
        required_files=["matrix.mtx", "barcodes.tsv"],
        optional_files=["features.tsv", "genes.tsv"],
        validation_level="strict",
        validation_thresholds={
            "min_genes_per_cell": 200,
            "min_cells_per_gene": 3,
            "max_missing_per_sample": 0.3,
        },
        platform_specific={
            "supports_filtered": True,
            "supports_raw": True,
            "typical_format": "mtx",
        },
    ),
    "h5ad": DownloadStrategyConfig(
        platform_type="h5ad",
        display_name="H5AD (AnnData)",
        description="Pre-processed AnnData in H5AD format",
        default_strategy="H5_FIRST",
        supported_strategies=["H5_FIRST", "AUTO"],
        prefer_processed=True,
        file_patterns=[".h5ad", ".h5"],
        required_files=[".h5ad"],
        optional_files=[],
        validation_level="moderate",
        validation_thresholds={
            "min_genes_per_cell": 200,
            "min_cells_per_gene": 3,
        },
        platform_specific={
            "supports_layers": True,
            "supports_obsm": True,
            "typical_format": "h5ad",
        },
    ),
    "kallisto": DownloadStrategyConfig(
        platform_type="kallisto",
        display_name="Kallisto (Bulk RNA-seq)",
        description="Kallisto pseudo-alignment output for bulk RNA-seq",
        default_strategy="SUPPLEMENTARY_FILES",
        supported_strategies=["SUPPLEMENTARY_FILES", "AUTO"],
        prefer_processed=True,
        file_patterns=["abundance.tsv", "abundance.h5"],
        required_files=["abundance.tsv"],
        optional_files=["abundance.h5", "run_info.json"],
        validation_level="moderate",
        validation_thresholds={
            "min_transcripts": 10000,
            "min_samples": 2,
        },
        platform_specific={
            "quantification_tool": "kallisto",
            "typical_format": "tsv",
        },
    ),
    "salmon": DownloadStrategyConfig(
        platform_type="salmon",
        display_name="Salmon (Bulk RNA-seq)",
        description="Salmon quasi-mapping output for bulk RNA-seq",
        default_strategy="SUPPLEMENTARY_FILES",
        supported_strategies=["SUPPLEMENTARY_FILES", "AUTO"],
        prefer_processed=True,
        file_patterns=["quant.sf", "quant.genes.sf"],
        required_files=["quant.sf"],
        optional_files=["quant.genes.sf", "aux_info"],
        validation_level="moderate",
        validation_thresholds={
            "min_transcripts": 10000,
            "min_samples": 2,
        },
        platform_specific={
            "quantification_tool": "salmon",
            "typical_format": "sf",
        },
    ),
    "csv": DownloadStrategyConfig(
        platform_type="csv",
        display_name="CSV/TSV Matrix",
        description="Generic delimited text matrix files",
        default_strategy="SUPPLEMENTARY_FILES",
        supported_strategies=["SUPPLEMENTARY_FILES", "AUTO"],
        prefer_processed=True,
        file_patterns=[".csv", ".tsv", ".txt"],
        required_files=[],
        optional_files=[],
        validation_level="permissive",
        validation_thresholds={},
        platform_specific={
            "typical_format": "csv",
            "supports_metadata": True,
        },
    ),
    "proteomics": DownloadStrategyConfig(
        platform_type="proteomics",
        display_name="Proteomics (MS/Affinity)",
        description="Mass spectrometry or affinity proteomics data",
        default_strategy="SUPPLEMENTARY_FILES",
        supported_strategies=["SUPPLEMENTARY_FILES", "AUTO"],
        prefer_processed=True,
        file_patterns=[
            "proteinGroups.txt",  # MaxQuant
            "NPX.csv",  # Olink
            ".xlsx",  # Generic
        ],
        required_files=[],
        optional_files=[],
        validation_level="permissive",
        validation_thresholds={
            "max_missing_per_sample": 0.7,
            "max_missing_per_protein": 0.8,
        },
        platform_specific={
            "supports_missing_values": True,
            "typical_format": "txt",
        },
    ),
}


# =============================================================================
# PLATFORM DETECTION SIGNATURES
# =============================================================================

PLATFORM_SIGNATURES = {
    "10x": {
        "required_patterns": ["matrix.mtx", "barcodes.tsv"],
        "optional_patterns": ["features.tsv", "genes.tsv"],
        "exclusion_patterns": [".h5ad"],
        "confidence_boost": 2.0,  # High confidence if required files present
    },
    "h5ad": {
        "required_patterns": [".h5ad"],
        "optional_patterns": [],
        "exclusion_patterns": ["matrix.mtx"],
        "confidence_boost": 3.0,  # Very high confidence for .h5ad files
    },
    "kallisto": {
        "required_patterns": ["abundance.tsv"],
        "optional_patterns": ["abundance.h5", "run_info.json"],
        "exclusion_patterns": [],
        "confidence_boost": 1.5,
    },
    "salmon": {
        "required_patterns": ["quant.sf"],
        "optional_patterns": ["quant.genes.sf"],
        "exclusion_patterns": [],
        "confidence_boost": 1.5,
    },
    "proteomics": {
        "required_patterns": [],
        "optional_patterns": ["proteinGroups.txt", "NPX.csv", ".xlsx"],
        "exclusion_patterns": [],
        "confidence_boost": 1.0,
    },
    "csv": {
        "required_patterns": [],
        "optional_patterns": [".csv", ".tsv", ".txt"],
        "exclusion_patterns": [],
        "confidence_boost": 0.5,  # Low confidence - generic format
    },
}


# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

VALIDATION_THRESHOLDS = {
    # Single-cell RNA-seq
    "min_genes_per_cell": 200,
    "max_genes_per_cell": 10000,
    "min_cells_per_gene": 3,
    "max_mt_percent": 20.0,

    # Bulk RNA-seq
    "min_transcripts": 10000,
    "min_samples": 2,

    # Proteomics
    "max_missing_per_sample": 0.7,
    "max_missing_per_protein": 0.8,
    "min_proteins": 100,

    # General
    "min_features": 50,
    "min_observations": 10,
}


# =============================================================================
# FILE SIZE LIMITS
# =============================================================================

FILE_SIZE_LIMITS = {
    "max_csv_size_mb": 500,
    "max_h5ad_size_mb": 5000,
    "max_xlsx_size_mb": 200,
    "warn_threshold_mb": 1000,
    "max_download_size_mb": 10000,
}


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

ADAPTER_REGISTRY = {
    "10x": "TenXAdapter",
    "h5ad": "H5ADAdapter",
    "csv": "CSVAdapter",
    "tsv": "CSVAdapter",
    "txt": "CSVAdapter",
    "kallisto": "KallistoAdapter",
    "salmon": "SalmonAdapter",
    "proteomics": "ProteomicsAdapter",
    "olink": "OlinkAdapter",
    "maxquant": "MaxQuantAdapter",
}


# =============================================================================
# QUEUE STATUS MAPPING
# =============================================================================

QUEUE_STATUS_MAPPING = {
    "pending": "PENDING",
    "in_progress": "IN_PROGRESS",
    "completed": "COMPLETED",
    "failed": "FAILED",
    "validation_failed": "VALIDATION_FAILED",
    "cancelled": "CANCELLED",
}


# =============================================================================
# DOWNLOAD STRATEGIES
# =============================================================================

SUPPORTED_DOWNLOAD_STRATEGIES = [
    "AUTO",
    "H5_FIRST",
    "MATRIX_FIRST",
    "SUPPLEMENTARY_FILES",
    "SAMPLES_FIRST",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_strategy_config(platform_type: str) -> DownloadStrategyConfig:
    """
    Get download strategy configuration for a platform type.

    Args:
        platform_type: Platform identifier (e.g., "10x", "h5ad", "kallisto")

    Returns:
        DownloadStrategyConfig for the specified platform

    Raises:
        ValueError: If platform_type is not recognized
    """
    if platform_type not in PLATFORM_STRATEGY_CONFIGS:
        available = list(PLATFORM_STRATEGY_CONFIGS.keys())
        raise ValueError(
            f"Unknown platform type: '{platform_type}'. "
            f"Available platforms: {available}"
        )
    return PLATFORM_STRATEGY_CONFIGS[platform_type]


def get_platform_signature(platform_type: str) -> Dict[str, Any]:
    """
    Get detection signature for a platform type.

    Args:
        platform_type: Platform identifier

    Returns:
        Dictionary with required/optional/exclusion patterns

    Raises:
        ValueError: If platform_type is not recognized
    """
    if platform_type not in PLATFORM_SIGNATURES:
        available = list(PLATFORM_SIGNATURES.keys())
        raise ValueError(
            f"Unknown platform type: '{platform_type}'. "
            f"Available platforms: {available}"
        )
    return PLATFORM_SIGNATURES[platform_type]


def get_validation_thresholds(platform_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get validation thresholds for a platform type.

    Args:
        platform_type: Optional platform identifier. If None, returns all thresholds.

    Returns:
        Dictionary of validation thresholds
    """
    if platform_type is None:
        return VALIDATION_THRESHOLDS.copy()

    config = get_strategy_config(platform_type)
    # Merge platform-specific thresholds with global defaults
    thresholds = VALIDATION_THRESHOLDS.copy()
    thresholds.update(config.validation_thresholds)
    return thresholds
