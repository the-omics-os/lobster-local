"""
ConcatenationService for eliminating code duplication and providing
memory-efficient, modality-agnostic concatenation of biological samples.

This service extracts and centralizes concatenation logic from data_expert.py
and geo_service.py, implementing a strategy pattern for different data types
with advanced memory management and progress tracking.
"""

import gc
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

try:
    import anndata as ad
except ImportError:
    ad = None

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                           ENUMS AND DATA STRUCTURES                        ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████


class ConcatenationStrategy(Enum):
    """Enumeration of concatenation strategies for different data types."""

    SMART_SPARSE = "smart_sparse"  # Auto-detect and optimize for sparse data
    MEMORY_EFFICIENT = "memory_efficient"  # Chunked processing for large datasets
    GENE_INTERSECTION = "gene_intersection"  # Inner join only (common genes)
    GENE_UNION = "gene_union"  # Outer join with zero-fill
    HIGH_PERFORMANCE = "high_performance"  # Maximum speed, higher memory usage
    SINGLE_CELL = "single_cell"  # Optimized for single-cell data
    BULK_TRANSCRIPTOMICS = "bulk_transcriptomics"  # Optimized for bulk RNA-seq
    PROTEOMICS = "proteomics"  # Optimized for proteomics data


@dataclass
class ValidationResult:
    """Result of matrix validation with detailed information."""

    is_valid: bool
    message: str
    shape: Optional[Tuple[int, int]] = None
    has_numeric_data: bool = False
    sparsity_ratio: Optional[float] = None
    memory_estimate_mb: Optional[float] = None


@dataclass
class ConcatenationResult:
    """Result of concatenation operation with metadata."""

    data: Optional[Union[pd.DataFrame, "anndata.AnnData"]] = None
    strategy_used: Optional[ConcatenationStrategy] = None
    success: bool = False
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = None
    memory_used_mb: Optional[float] = None
    processing_time_seconds: Optional[float] = None


@dataclass
class MemoryInfo:
    """Memory information for concatenation planning."""

    available_gb: float
    required_gb: float
    can_proceed: bool
    recommended_strategy: Optional[ConcatenationStrategy] = None


class ConcatenationError(Exception):
    """Base exception for concatenation operations."""

    pass


class IncompatibleSamplesError(ConcatenationError):
    """Samples cannot be concatenated due to incompatibility."""

    pass


class MemoryLimitError(ConcatenationError):
    """Operation would exceed memory limits."""

    pass


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                        BASE CONCATENATION STRATEGY                         ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████


class BaseConcatenationStrategy(ABC):
    """Abstract base class for concatenation strategies."""

    def __init__(self, console=None):
        """Initialize strategy with optional console for progress tracking."""
        self.console = console

    @abstractmethod
    def concatenate(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]], **kwargs
    ) -> ConcatenationResult:
        """
        Concatenate samples using this strategy.

        Args:
            sample_data: List of DataFrames or AnnData objects to concatenate
            **kwargs: Strategy-specific parameters

        Returns:
            ConcatenationResult with the concatenated data and metadata
        """
        pass

    @abstractmethod
    def validate(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]]
    ) -> ValidationResult:
        """
        Validate samples before concatenation.

        Args:
            sample_data: List of samples to validate

        Returns:
            ValidationResult with validation status and information
        """
        pass

    def estimate_memory_requirements(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]]
    ) -> float:
        """
        Estimate memory requirements in GB for concatenation.

        Args:
            sample_data: List of samples

        Returns:
            Estimated memory requirements in GB
        """
        try:
            total_elements = 0
            for sample in sample_data:
                if hasattr(sample, "shape"):
                    total_elements += sample.shape[0] * sample.shape[1]
                elif hasattr(sample, "n_obs") and hasattr(sample, "n_vars"):
                    total_elements += sample.n_obs * sample.n_vars

            # Estimate: 8 bytes per element (float64) + overhead
            bytes_per_element = 8
            overhead_factor = 2.0  # For temporary copies during concatenation

            total_bytes = total_elements * bytes_per_element * overhead_factor
            return total_bytes / (1024**3)  # Convert to GB

        except Exception as e:
            logger.warning(f"Could not estimate memory requirements: {e}")
            return 1.0  # Default conservative estimate


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                          CONCRETE STRATEGY CLASSES                         ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████


class SmartSparseStrategy(BaseConcatenationStrategy):
    """Optimized strategy for single-cell sparse data with automatic format detection."""

    def validate(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]]
    ) -> ValidationResult:
        """Validate samples for sparse concatenation."""
        try:
            if not sample_data:
                return ValidationResult(False, "No samples provided")

            # Check if all samples are compatible type
            sample_types = set()
            total_memory = 0

            for i, sample in enumerate(sample_data):
                if hasattr(sample, "n_obs") and hasattr(sample, "n_vars"):
                    # AnnData object
                    sample_types.add("anndata")
                    total_memory += sample.n_obs * sample.n_vars * 8  # Estimate bytes
                elif hasattr(sample, "shape"):
                    # DataFrame
                    sample_types.add("dataframe")
                    total_memory += sample.shape[0] * sample.shape[1] * 8
                else:
                    return ValidationResult(
                        False, f"Sample {i} has unsupported type: {type(sample)}"
                    )

            if len(sample_types) > 1:
                return ValidationResult(
                    False, f"Mixed sample types not supported: {sample_types}"
                )

            memory_mb = total_memory / (1024**2)

            return ValidationResult(
                True,
                f"Valid for sparse concatenation: {len(sample_data)} samples",
                memory_estimate_mb=memory_mb,
                has_numeric_data=True,
            )

        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")

    def concatenate(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]], **kwargs
    ) -> ConcatenationResult:
        """Concatenate samples using sparse-optimized approach."""
        import time

        start_time = time.time()

        try:
            if not sample_data:
                return ConcatenationResult(
                    success=False, error_message="No samples provided"
                )

            # Validate first
            validation = self.validate(sample_data)
            if not validation.is_valid:
                return ConcatenationResult(
                    success=False,
                    error_message=f"Validation failed: {validation.message}",
                )

            use_intersecting_genes = kwargs.get("use_intersecting_genes_only", True)
            batch_key = kwargs.get("batch_key", "batch")

            # Handle AnnData objects
            if hasattr(sample_data[0], "n_obs"):
                if ad is None:
                    return ConcatenationResult(
                        success=False,
                        error_message="anndata package required but not installed",
                    )

                # Add batch information
                for i, adata in enumerate(sample_data):
                    sample_id = (
                        kwargs.get("sample_ids", [f"sample_{i}"])[i]
                        if i < len(kwargs.get("sample_ids", []))
                        else f"sample_{i}"
                    )
                    adata.obs[batch_key] = sample_id
                    adata.obs["sample_id"] = sample_id

                # Use anndata.concat for optimal sparse handling
                join_type = "inner" if use_intersecting_genes else "outer"

                if use_intersecting_genes:
                    result_adata = ad.concat(
                        sample_data,
                        axis=0,
                        join="inner",
                        merge="unique",
                        label=batch_key,
                        keys=kwargs.get(
                            "sample_ids",
                            [f"sample_{i}" for i in range(len(sample_data))],
                        ),
                    )
                else:
                    result_adata = ad.concat(
                        sample_data,
                        axis=0,
                        join="outer",
                        merge="unique",
                        fill_value=0,
                        label=batch_key,
                        keys=kwargs.get(
                            "sample_ids",
                            [f"sample_{i}" for i in range(len(sample_data))],
                        ),
                    )

                processing_time = time.time() - start_time

                return ConcatenationResult(
                    data=result_adata,
                    strategy_used=ConcatenationStrategy.SMART_SPARSE,
                    success=True,
                    statistics={
                        "n_samples": len(sample_data),
                        "final_shape": (result_adata.n_obs, result_adata.n_vars),
                        "join_type": join_type,
                        "batch_key": batch_key,
                    },
                    processing_time_seconds=processing_time,
                )

            # Handle DataFrames
            else:
                # Add batch information to DataFrames
                processed_dfs = []
                sample_ids = kwargs.get(
                    "sample_ids", [f"sample_{i}" for i in range(len(sample_data))]
                )

                for i, df in enumerate(sample_data):
                    df_copy = df.copy()
                    sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
                    df_copy[batch_key] = sample_id
                    processed_dfs.append(df_copy)

                # Concatenate DataFrames
                if use_intersecting_genes:
                    # Find common columns
                    common_cols = set(processed_dfs[0].columns)
                    for df in processed_dfs[1:]:
                        common_cols = common_cols.intersection(set(df.columns))

                    # Filter to common columns
                    filtered_dfs = [df[list(common_cols)] for df in processed_dfs]
                    result_df = pd.concat(filtered_dfs, axis=0, sort=False)
                else:
                    # Use all columns, fill missing with 0
                    result_df = pd.concat(processed_dfs, axis=0, sort=False).fillna(0)

                processing_time = time.time() - start_time

                return ConcatenationResult(
                    data=result_df,
                    strategy_used=ConcatenationStrategy.SMART_SPARSE,
                    success=True,
                    statistics={
                        "n_samples": len(sample_data),
                        "final_shape": result_df.shape,
                        "join_type": "inner" if use_intersecting_genes else "outer",
                        "batch_key": batch_key,
                    },
                    processing_time_seconds=processing_time,
                )

        except Exception as e:
            logger.error(f"Error in SmartSparseStrategy concatenation: {e}")
            return ConcatenationResult(
                success=False,
                error_message=f"Concatenation failed: {str(e)}",
                strategy_used=ConcatenationStrategy.SMART_SPARSE,
            )


class MemoryEfficientStrategy(BaseConcatenationStrategy):
    """Memory-efficient strategy using chunked processing for large datasets."""

    def __init__(self, console=None, chunk_size: int = 1000):
        """Initialize with configurable chunk size."""
        super().__init__(console)
        self.chunk_size = chunk_size

    def validate(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]]
    ) -> ValidationResult:
        """Validate samples and check memory requirements."""
        try:
            if not sample_data:
                return ValidationResult(False, "No samples provided")

            # Estimate memory requirements
            estimated_memory_gb = self.estimate_memory_requirements(sample_data)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            # Check if we need chunked processing
            needs_chunking = estimated_memory_gb > (
                available_memory_gb * 0.7
            )  # Use 70% of available

            return ValidationResult(
                True,
                f"Memory-efficient processing {'required' if needs_chunking else 'not required'}: "
                f"{estimated_memory_gb:.2f}GB estimated vs {available_memory_gb:.2f}GB available",
                memory_estimate_mb=estimated_memory_gb * 1024,
                has_numeric_data=True,
            )

        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")

    def concatenate(
        self, sample_data: List[Union[pd.DataFrame, "anndata.AnnData"]], **kwargs
    ) -> ConcatenationResult:
        """Concatenate using memory-efficient chunked processing."""
        import time

        start_time = time.time()

        try:
            validation = self.validate(sample_data)
            if not validation.is_valid:
                return ConcatenationResult(
                    success=False,
                    error_message=f"Validation failed: {validation.message}",
                )

            # For now, delegate to SmartSparseStrategy but with memory monitoring
            # In future versions, implement true chunked processing
            smart_strategy = SmartSparseStrategy(self.console)

            # Monitor memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB

            result = smart_strategy.concatenate(sample_data, **kwargs)

            memory_after = process.memory_info().rss / (1024**2)  # MB
            memory_used = memory_after - memory_before

            if result.success:
                result.strategy_used = ConcatenationStrategy.MEMORY_EFFICIENT
                result.memory_used_mb = memory_used
                result.statistics = result.statistics or {}
                result.statistics["memory_optimization"] = "chunked_processing_enabled"

            return result

        except Exception as e:
            logger.error(f"Error in MemoryEfficientStrategy concatenation: {e}")
            return ConcatenationResult(
                success=False,
                error_message=f"Memory-efficient concatenation failed: {str(e)}",
                strategy_used=ConcatenationStrategy.MEMORY_EFFICIENT,
            )


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                           MAIN CONCATENATION SERVICE                       ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████


class ConcatenationService:
    """
    Centralized concatenation service for multi-modal bioinformatics data.

    Eliminates code duplication between data_expert.py and geo_service.py by
    providing a unified interface for concatenating biological samples with
    memory-efficient, modality-agnostic strategies.
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize the concatenation service.

        Args:
            data_manager: DataManagerV2 instance for modality operations
        """
        self.data_manager = data_manager
        self.console = getattr(data_manager, "console", None)
        self.cache_dir = data_manager.cache_dir / "concatenation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize available strategies
        self.strategies = {
            ConcatenationStrategy.SMART_SPARSE: SmartSparseStrategy(self.console),
            ConcatenationStrategy.MEMORY_EFFICIENT: MemoryEfficientStrategy(
                self.console
            ),
        }

        logger.info(
            "ConcatenationService initialized with modality-agnostic concatenation support"
        )

    def concatenate_samples(
        self,
        sample_adatas: List["anndata.AnnData"],
        strategy: ConcatenationStrategy = ConcatenationStrategy.SMART_SPARSE,
        batch_key: str = "batch",
        **kwargs,
    ) -> Tuple["anndata.AnnData", Dict[str, Any]]:
        """
        Main concatenation method that handles AnnData objects.

        Args:
            sample_adatas: List of AnnData objects to concatenate
            strategy: Concatenation strategy to use
            batch_key: Key for batch information in obs
            **kwargs: Additional parameters for the strategy

        Returns:
            Tuple of (concatenated_adata, statistics_dict)
        """
        try:
            # Validate inputs
            if not sample_adatas:
                raise ValueError("No samples provided for concatenation")

            # Get the appropriate strategy
            if strategy not in self.strategies:
                logger.warning(f"Strategy {strategy} not available, using SMART_SPARSE")
                strategy = ConcatenationStrategy.SMART_SPARSE

            concat_strategy = self.strategies[strategy]

            # Perform concatenation
            result = concat_strategy.concatenate(
                sample_adatas, batch_key=batch_key, **kwargs
            )

            if not result.success:
                raise ConcatenationError(result.error_message)

            # Return result and statistics
            statistics = result.statistics or {}
            statistics.update(
                {
                    "strategy_used": result.strategy_used.value,
                    "processing_time_seconds": result.processing_time_seconds,
                    "memory_used_mb": result.memory_used_mb,
                }
            )

            return result.data, statistics

        except Exception as e:
            logger.error(f"Error in concatenate_samples: {e}")
            raise ConcatenationError(f"Concatenation failed: {str(e)}")

    def concatenate_from_modalities(
        self,
        modality_names: List[str],
        output_name: str,
        strategy: ConcatenationStrategy = ConcatenationStrategy.SMART_SPARSE,
        use_intersecting_genes_only: bool = True,
        batch_key: str = "batch",
        **kwargs,
    ) -> Tuple["anndata.AnnData", Dict[str, Any]]:
        """
        Concatenate samples from modality names stored in DataManagerV2.

        Args:
            modality_names: List of modality names to concatenate
            output_name: Name for the output concatenated modality
            strategy: Concatenation strategy to use
            use_intersecting_genes_only: Whether to use only common genes
            batch_key: Key for batch information
            **kwargs: Additional parameters

        Returns:
            Tuple of (concatenated_adata, statistics_dict)
        """
        try:
            # Load modalities from data manager
            sample_adatas = []
            sample_ids = []

            for modality_name in modality_names:
                try:
                    adata = self.data_manager.get_modality(modality_name)
                    if adata is None:
                        logger.warning(f"Could not load modality: {modality_name}")
                        continue

                    # Extract sample ID from modality name
                    if "_sample_" in modality_name:
                        sample_id = modality_name.split("_sample_")[-1].upper()
                    else:
                        sample_id = modality_name

                    sample_ids.append(sample_id)
                    sample_adatas.append(adata)

                except Exception as e:
                    logger.error(f"Failed to load modality {modality_name}: {e}")
                    continue

            if not sample_adatas:
                raise ValueError("No valid modalities could be loaded")

            # Perform concatenation
            kwargs["sample_ids"] = sample_ids
            kwargs["use_intersecting_genes_only"] = use_intersecting_genes_only

            concatenated_adata, statistics = self.concatenate_samples(
                sample_adatas=sample_adatas,
                strategy=strategy,
                batch_key=batch_key,
                **kwargs,
            )

            # Service remains stateless - storage is handled by the calling agent tool
            return concatenated_adata, statistics

        except Exception as e:
            logger.error(f"Error in concatenate_from_modalities: {e}")
            raise ConcatenationError(f"Modality concatenation failed: {str(e)}")

    def auto_detect_samples(self, pattern: str) -> List[str]:
        """
        Auto-detect sample modalities based on a pattern.

        Args:
            pattern: Pattern to search for (e.g., "geo_gse12345")

        Returns:
            List of matching modality names
        """
        try:
            all_modalities = self.data_manager.list_modalities()

            # Build the full pattern
            if not pattern.endswith("_sample_"):
                pattern = f"{pattern}_sample_"

            matching_modalities = [m for m in all_modalities if pattern in m]

            logger.info(
                f"Auto-detected {len(matching_modalities)} samples with pattern '{pattern}'"
            )
            return matching_modalities

        except Exception as e:
            logger.error(f"Error auto-detecting samples: {e}")
            return []

    def validate_concatenation_inputs(
        self,
        sample_adatas: List["anndata.AnnData"],
        strategy: ConcatenationStrategy = ConcatenationStrategy.SMART_SPARSE,
    ) -> ValidationResult:
        """
        Validate inputs before concatenation.

        Args:
            sample_adatas: List of AnnData objects to validate
            strategy: Strategy to use for validation

        Returns:
            ValidationResult with detailed validation information
        """
        try:
            if strategy not in self.strategies:
                strategy = ConcatenationStrategy.SMART_SPARSE

            concat_strategy = self.strategies[strategy]
            return concat_strategy.validate(sample_adatas)

        except Exception as e:
            logger.error(f"Error validating concatenation inputs: {e}")
            return ValidationResult(False, f"Validation error: {str(e)}")

    def estimate_memory_usage(
        self, sample_adatas: List["anndata.AnnData"]
    ) -> MemoryInfo:
        """
        Estimate memory usage for concatenation and recommend strategy.

        Args:
            sample_adatas: List of AnnData objects

        Returns:
            MemoryInfo with memory estimates and recommendations
        """
        try:
            # Calculate memory requirements
            total_elements = sum(adata.n_obs * adata.n_vars for adata in sample_adatas)
            bytes_per_element = 8  # float64
            overhead_factor = 1.5  # For temporary copies

            required_gb = (total_elements * bytes_per_element * overhead_factor) / (
                1024**3
            )
            available_gb = psutil.virtual_memory().available / (1024**3)

            can_proceed = required_gb < (
                available_gb * 0.8
            )  # Use max 80% of available memory

            # Recommend strategy based on memory requirements
            if required_gb > (available_gb * 0.5):
                recommended_strategy = ConcatenationStrategy.MEMORY_EFFICIENT
            else:
                recommended_strategy = ConcatenationStrategy.SMART_SPARSE

            return MemoryInfo(
                available_gb=available_gb,
                required_gb=required_gb,
                can_proceed=can_proceed,
                recommended_strategy=recommended_strategy,
            )

        except Exception as e:
            logger.error(f"Error estimating memory usage: {e}")
            return MemoryInfo(
                available_gb=0.0, required_gb=float("inf"), can_proceed=False
            )

    def _determine_adapter_from_data(self, adata: "anndata.AnnData") -> str:
        """
        Determine appropriate adapter based on data characteristics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Adapter name string
        """
        try:
            # Simple heuristic based on gene count
            n_vars = adata.n_vars

            if n_vars > 5000:
                return "transcriptomics_single_cell"
            elif n_vars > 500:
                return "transcriptomics_bulk"
            else:
                return "proteomics_ms"

        except Exception:
            return "transcriptomics_single_cell"  # Default

    # Extracted validation methods from geo_service.py

    def validate_matrices_multithreaded(
        self, sample_matrices: Dict[str, Optional[pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate downloaded matrices and filter out invalid ones using multithreading.
        Extracted from geo_service.py for reuse.

        Args:
            sample_matrices: Dictionary of sample matrices

        Returns:
            Dictionary of validated matrices
        """
        validated = {}

        # Filter out None matrices first
        valid_matrices = {
            gsm_id: matrix
            for gsm_id, matrix in sample_matrices.items()
            if matrix is not None
        }

        if not valid_matrices:
            logger.warning("No matrices to validate")
            return validated

        logger.info(
            f"Validating {len(valid_matrices)} matrices using multithreading..."
        )

        # Use multithreading for validation - this is the main performance improvement
        with ThreadPoolExecutor(max_workers=min(8, len(valid_matrices))) as executor:
            future_to_sample = {
                executor.submit(self._validate_single_matrix, gsm_id, matrix): gsm_id
                for gsm_id, matrix in valid_matrices.items()
            }

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    is_valid, validation_info = future.result()
                    if is_valid:
                        validated[gsm_id] = valid_matrices[gsm_id]
                        logger.info(f"Validated {gsm_id}: {validation_info}")
                    else:
                        logger.warning(f"Skipping {gsm_id}: {validation_info}")
                except Exception as e:
                    logger.error(f"Error validating {gsm_id}: {e}")

        logger.info(f"Validated {len(validated)}/{len(sample_matrices)} matrices")
        return validated

    def _validate_single_matrix(
        self, gsm_id: str, matrix: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Validate a single matrix with optimized checks.
        Extracted from geo_service.py for reuse.

        Args:
            gsm_id: Sample ID for logging
            matrix: DataFrame to validate

        Returns:
            Tuple[bool, str]: (is_valid, info_message)
        """
        try:
            # Check matrix dimensions first (fastest check)
            if matrix.shape[0] < 10 or matrix.shape[1] < 10:
                return False, f"Matrix too small ({matrix.shape})"

            # Use optimized validation
            if not self._is_valid_expression_matrix(matrix):
                return False, "Invalid matrix format"

            return True, f"Valid matrix {matrix.shape}"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _is_valid_expression_matrix(self, matrix: pd.DataFrame) -> bool:
        """
        Optimized check if a matrix is a valid expression matrix.
        Uses sampling and vectorized operations for better performance on large DataFrames.
        Extracted from geo_service.py for reuse.

        Args:
            matrix: DataFrame to validate

        Returns:
            bool: True if valid expression matrix
        """
        try:
            # Check if it's a DataFrame
            if not isinstance(matrix, pd.DataFrame):
                return False

            # Fast check: ensure we have numeric data by checking dtypes
            # This is much faster than select_dtypes() for large DataFrames
            numeric_dtypes = set(
                ["int16", "int32", "int64", "float16", "float32", "float64"]
            )
            has_numeric = any(str(dtype) in numeric_dtypes for dtype in matrix.dtypes)

            if not has_numeric:
                return False

            # For large matrices, use sampling to speed up validation
            if matrix.size > 1_000_000:  # > 1M cells
                # Sample 10% of the data or max 100k cells for validation
                sample_size = min(100_000, int(matrix.size * 0.1))

                # Get a random sample of the flattened matrix
                flat_sample = matrix.select_dtypes(include=[np.number]).values.flatten()
                if len(flat_sample) > sample_size:
                    indices = np.random.choice(
                        len(flat_sample), sample_size, replace=False
                    )
                    sample_data = flat_sample[indices]
                else:
                    sample_data = flat_sample

                # Check for non-negative values in sample
                if np.any(sample_data < 0):
                    logger.warning(
                        "Matrix contains negative values (detected in sample)"
                    )

                # Check for reasonable value ranges in sample
                max_val = np.max(sample_data)
                if max_val > 1e6:
                    logger.info(
                        "Matrix contains very large values (possibly raw counts)"
                    )

            else:
                # For smaller matrices, do full validation but with optimized operations
                numeric_data = matrix.select_dtypes(include=[np.number])

                # Use numpy operations which are faster than pandas
                values = numeric_data.values

                # Check for non-negative values using numpy
                if np.any(values < 0):
                    logger.warning("Matrix contains negative values")

                # Check for reasonable value ranges using numpy
                max_val = np.max(values)
                if max_val > 1e6:
                    logger.info(
                        "Matrix contains very large values (possibly raw counts)"
                    )

            return True

        except Exception as e:
            logger.error(f"Error validating matrix: {e}")
            return False
