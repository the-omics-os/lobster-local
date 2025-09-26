# Creating Services - Lobster AI Service Development Guide

## 🎯 Overview

This guide covers how to create stateless analysis services in the Lobster AI system. Services handle the core computational work for bioinformatics analyses, while agents orchestrate their usage. Services follow a strict stateless design pattern that promotes reusability and testability.

## 🏗️ Service Architecture

### Service Responsibilities
- **Computational Logic**: Implement bioinformatics algorithms and analyses
- **Data Processing**: Transform AnnData objects following scientific standards
- **Statistical Analysis**: Provide rigorous statistical methods and metrics
- **Error Handling**: Robust error handling with specific exceptions
- **Progress Reporting**: Optional progress callbacks for long-running operations

### Service Design Principles
- **Stateless**: No instance state between method calls
- **Pure Functions**: Deterministic outputs for given inputs
- **AnnData-Centric**: Work with AnnData objects as primary data structure
- **Return Tuples**: Always return `(processed_adata, statistics_dict)`
- **Scientific Rigor**: Follow established bioinformatics best practices

## 📋 Service Pattern Template

### Basic Service Structure
```python
# lobster/tools/your_service.py
"""
Your analysis service for specific bioinformatics workflow.

This service provides stateless methods for [describe specific analysis type].
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
import anndata
import numpy as np
import pandas as pd

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class YourAnalysisError(Exception):
    """Base exception for your analysis operations."""
    pass


class YourService:
    """
    Stateless service for your specific analysis workflow.

    This class provides methods to perform [specific analysis] on biological data
    following established scientific protocols and best practices.
    """

    def __init__(self):
        """
        Initialize the service.

        This service is stateless and doesn't require external dependencies.
        """
        logger.debug("Initializing stateless YourService")
        self.progress_callback = None
        self.current_progress = 0
        self.total_steps = 0
        logger.debug("YourService initialized successfully")

    def set_progress_callback(self, callback: Callable[[int, str], None]) -> None:
        """
        Set a callback function to report progress.

        Args:
            callback: Function accepting (progress_percent: int, message: str)
        """
        self.progress_callback = callback
        logger.info("Progress callback set for YourService")

    def _update_progress(self, step_name: str) -> None:
        """Update progress and call callback if set."""
        self.current_progress += 1
        if self.progress_callback is not None:
            progress_percent = int((self.current_progress / self.total_steps) * 100)
            self.progress_callback(progress_percent, step_name)
            logger.debug(f"Progress: {progress_percent}% - {step_name}")

    def main_analysis_method(
        self,
        adata: anndata.AnnData,
        parameter1: float = 1.0,
        parameter2: str = "default",
        parameter3: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Main analysis method following the service pattern.

        Args:
            adata: Input AnnData object to analyze
            parameter1: Numerical parameter with scientific meaning
            parameter2: Categorical parameter
            parameter3: Optional list parameter
            progress_callback: Optional progress reporting function

        Returns:
            Tuple of (processed_adata, statistics_dict)

        Raises:
            YourAnalysisError: If analysis fails
            ValueError: If parameters are invalid
        """
        try:
            # Set up progress tracking
            if progress_callback:
                self.set_progress_callback(progress_callback)

            self.total_steps = 4  # Adjust based on actual steps
            self.current_progress = 0

            # Validate inputs
            self._validate_inputs(adata, parameter1, parameter2, parameter3)
            self._update_progress("Input validation")

            # Create working copy
            adata_result = adata.copy()

            # Step 1: Preprocessing
            adata_result = self._preprocess_data(adata_result, parameter1)
            self._update_progress("Data preprocessing")

            # Step 2: Core analysis
            analysis_results = self._perform_core_analysis(
                adata_result, parameter2, parameter3
            )
            self._update_progress("Core analysis")

            # Step 3: Post-processing and statistics
            statistics = self._calculate_statistics(adata_result, analysis_results)
            self._update_progress("Statistical analysis")

            # Step 4: Store results in AnnData
            self._store_results(adata_result, analysis_results)
            self._update_progress("Storing results")

            logger.info(f"Analysis completed successfully with {len(statistics)} metrics")
            return adata_result, statistics

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            if isinstance(e, (YourAnalysisError, ValueError)):
                raise
            else:
                raise YourAnalysisError(f"Unexpected error during analysis: {str(e)}")

    def _validate_inputs(
        self,
        adata: anndata.AnnData,
        parameter1: float,
        parameter2: str,
        parameter3: Optional[List[str]]
    ) -> None:
        """Validate input parameters and data."""
        if adata.n_obs == 0:
            raise ValueError("Input data is empty (no observations)")

        if adata.n_vars == 0:
            raise ValueError("Input data has no features")

        if parameter1 <= 0:
            raise ValueError("Parameter1 must be positive")

        if parameter2 not in ["option1", "option2", "default"]:
            raise ValueError(f"Invalid parameter2: {parameter2}")

        if parameter3 is not None and len(parameter3) == 0:
            raise ValueError("Parameter3 cannot be empty list")

        logger.debug("Input validation passed")

    def _preprocess_data(
        self,
        adata: anndata.AnnData,
        parameter1: float
    ) -> anndata.AnnData:
        """Perform data preprocessing steps."""
        # Implement preprocessing logic
        # Example: normalization, filtering, etc.

        logger.debug(f"Preprocessing with parameter1={parameter1}")

        # Store preprocessing parameters
        adata.uns['preprocessing_params'] = {
            'parameter1': parameter1,
            'method': 'your_preprocessing_method'
        }

        return adata

    def _perform_core_analysis(
        self,
        adata: anndata.AnnData,
        parameter2: str,
        parameter3: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Perform the core analysis computation."""
        results = {}

        # Implement your core analysis logic here
        logger.debug(f"Core analysis with parameter2={parameter2}")

        # Example analysis results
        results['analysis_method'] = parameter2
        results['features_analyzed'] = parameter3 or []
        results['n_observations'] = adata.n_obs
        results['n_features'] = adata.n_vars

        return results

    def _calculate_statistics(
        self,
        adata: anndata.AnnData,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate analysis statistics and metrics."""
        statistics = {
            'n_observations': adata.n_obs,
            'n_features': adata.n_vars,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'analysis_method': analysis_results.get('analysis_method', 'unknown')
        }

        # Add scientific metrics
        if 'X' in adata.layers:
            statistics['mean_expression'] = np.mean(adata.layers['X'])
            statistics['std_expression'] = np.std(adata.layers['X'])

        # Add analysis-specific metrics
        statistics.update(self._compute_domain_specific_metrics(adata, analysis_results))

        logger.debug(f"Calculated {len(statistics)} statistical metrics")
        return statistics

    def _compute_domain_specific_metrics(
        self,
        adata: anndata.AnnData,
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute metrics specific to your analysis domain."""
        metrics = {}

        # Implement domain-specific statistical calculations
        # Examples: clustering metrics, differential expression stats, etc.

        return metrics

    def _store_results(
        self,
        adata: anndata.AnnData,
        analysis_results: Dict[str, Any]
    ) -> None:
        """Store analysis results in appropriate AnnData slots."""

        # Store in .obs (cell-level annotations)
        # adata.obs['new_annotation'] = analysis_results['cell_annotations']

        # Store in .var (feature-level annotations)
        # adata.var['new_feature_info'] = analysis_results['feature_info']

        # Store in .obsm (embeddings/matrices)
        # adata.obsm['X_your_embedding'] = analysis_results['embedding']

        # Store in .uns (unstructured metadata)
        adata.uns['your_analysis'] = {
            'method': analysis_results['analysis_method'],
            'parameters': analysis_results.get('parameters', {}),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        logger.debug("Results stored in AnnData object")

    def auxiliary_method(
        self,
        adata: anndata.AnnData,
        specific_parameter: str
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Auxiliary analysis method for specific sub-tasks.

        Following the same pattern as main analysis method.
        """
        try:
            # Validate inputs
            if specific_parameter not in ["valid_option1", "valid_option2"]:
                raise ValueError(f"Invalid specific_parameter: {specific_parameter}")

            # Create working copy
            adata_result = adata.copy()

            # Perform specific analysis
            # ... implementation ...

            # Calculate statistics
            statistics = {
                'method': 'auxiliary_analysis',
                'parameter': specific_parameter,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            return adata_result, statistics

        except Exception as e:
            logger.error(f"Auxiliary analysis failed: {str(e)}")
            raise YourAnalysisError(f"Auxiliary analysis error: {str(e)}")
```

## 🔬 Scientific Service Patterns

### Single-Cell RNA-seq Service Example
```python
class SingleCellQualityService:
    """Service for single-cell RNA-seq quality control."""

    def assess_quality(
        self,
        adata: anndata.AnnData,
        min_genes: int = 200,
        max_mt_pct: float = 20.0,
        max_ribo_pct: float = 50.0
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Assess single-cell data quality with standard metrics."""

        adata_result = adata.copy()

        # Calculate QC metrics
        adata_result.var['mt'] = adata_result.var_names.str.startswith('MT-')
        adata_result.var['ribo'] = adata_result.var_names.str.startswith(('RPS', 'RPL'))

        sc.pp.calculate_qc_metrics(
            adata_result,
            percent_top=None,
            log1p=False,
            inplace=True
        )

        # Add mitochondrial and ribosomal percentages
        adata_result.obs['pct_counts_mt'] = (
            adata_result.obs['total_counts_mt'] / adata_result.obs['total_counts'] * 100
        )
        adata_result.obs['pct_counts_ribo'] = (
            adata_result.obs['total_counts_ribo'] / adata_result.obs['total_counts'] * 100
        )

        # Calculate statistics
        statistics = {
            'n_cells': adata_result.n_obs,
            'n_genes': adata_result.n_vars,
            'median_genes_per_cell': np.median(adata_result.obs['n_genes_by_counts']),
            'median_counts_per_cell': np.median(adata_result.obs['total_counts']),
            'mean_mt_pct': np.mean(adata_result.obs['pct_counts_mt']),
            'mean_ribo_pct': np.mean(adata_result.obs['pct_counts_ribo']),
        }

        return adata_result, statistics
```

### Proteomics Service Example
```python
class ProteomicsPreprocessingService:
    """Service for proteomics data preprocessing."""

    def normalize_intensities(
        self,
        adata: anndata.AnnData,
        method: str = "log2",
        handle_missing: str = "remove"
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """Normalize protein intensity data."""

        adata_result = adata.copy()

        # Handle missing values
        if handle_missing == "remove":
            # Remove proteins with >50% missing values
            missing_pct = (adata_result.X == 0).sum(axis=0) / adata_result.n_obs
            keep_proteins = missing_pct < 0.5
            adata_result = adata_result[:, keep_proteins]

        # Apply normalization
        if method == "log2":
            adata_result.X = np.log2(adata_result.X + 1)
            adata_result.layers['log2_normalized'] = adata_result.X
        elif method == "quantile":
            # Implement quantile normalization
            pass

        # Calculate statistics
        statistics = {
            'normalization_method': method,
            'proteins_before': adata.n_vars,
            'proteins_after': adata_result.n_vars,
            'missing_value_handling': handle_missing,
            'mean_intensity': np.mean(adata_result.X[adata_result.X > 0])
        }

        return adata_result, statistics
```

## 🔧 Integration with DataManagerV2

Services are called by agent tools that handle DataManagerV2 integration:

```python
# In agent tool
@tool
def perform_quality_assessment(modality_name: str, **params) -> str:
    """Agent tool that uses the service."""

    # 1. Get data from DataManagerV2
    adata = data_manager.get_modality(modality_name)

    # 2. Call stateless service
    service = QualityService()
    result_adata, statistics = service.assess_quality(adata, **params)

    # 3. Store result in DataManagerV2
    result_modality = f"{modality_name}_quality_assessed"
    data_manager.modalities[result_modality] = result_adata

    # 4. Log for provenance
    data_manager.log_tool_usage("quality_assessment", params, statistics)

    # 5. Return formatted response
    return format_quality_response(statistics, result_modality)
```

## 🧪 Testing Services

### Unit Test Template
```python
# tests/unit/tools/test_your_service.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from lobster.tools.your_service import YourService, YourAnalysisError
from tests.mock_data.generators import generate_adata


class TestYourService:

    @pytest.fixture
    def service(self):
        return YourService()

    @pytest.fixture
    def sample_adata(self):
        return generate_adata(n_obs=100, n_vars=50)

    def test_service_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.progress_callback is None
        assert service.current_progress == 0

    def test_progress_callback(self, service):
        """Test progress callback functionality."""
        callback_calls = []

        def mock_callback(progress, message):
            callback_calls.append((progress, message))

        service.set_progress_callback(mock_callback)
        assert service.progress_callback is not None

    def test_main_analysis_success(self, service, sample_adata):
        """Test successful analysis execution."""
        result_adata, statistics = service.main_analysis_method(
            sample_adata,
            parameter1=1.5,
            parameter2="option1"
        )

        # Validate results
        assert result_adata is not None
        assert isinstance(statistics, dict)
        assert 'n_observations' in statistics
        assert statistics['n_observations'] == sample_adata.n_obs

    def test_input_validation(self, service, sample_adata):
        """Test input validation."""

        # Test invalid parameter1
        with pytest.raises(ValueError, match="Parameter1 must be positive"):
            service.main_analysis_method(sample_adata, parameter1=-1.0)

        # Test invalid parameter2
        with pytest.raises(ValueError, match="Invalid parameter2"):
            service.main_analysis_method(sample_adata, parameter2="invalid")

    def test_empty_data_handling(self, service):
        """Test handling of empty data."""
        empty_adata = generate_adata(n_obs=0, n_vars=10)

        with pytest.raises(ValueError, match="Input data is empty"):
            service.main_analysis_method(empty_adata)

    def test_analysis_error_handling(self, service, sample_adata, monkeypatch):
        """Test error handling during analysis."""

        # Mock a method to raise an exception
        def mock_preprocess(*args, **kwargs):
            raise RuntimeError("Preprocessing failed")

        monkeypatch.setattr(service, '_preprocess_data', mock_preprocess)

        with pytest.raises(YourAnalysisError, match="Unexpected error"):
            service.main_analysis_method(sample_adata)

    def test_statistics_calculation(self, service, sample_adata):
        """Test statistical calculations are correct."""
        result_adata, statistics = service.main_analysis_method(sample_adata)

        # Verify statistical accuracy
        assert statistics['n_observations'] == result_adata.n_obs
        assert statistics['n_features'] == result_adata.n_vars
        assert 'analysis_timestamp' in statistics
```

### Integration Test Template
```python
# tests/integration/test_service_integration.py
def test_service_with_real_data(real_dataset_fixture):
    """Test service with realistic biological data."""

    service = YourService()
    adata = real_dataset_fixture  # Use real data fixture

    result_adata, statistics = service.main_analysis_method(adata)

    # Validate biological relevance
    assert result_adata.n_obs > 0
    assert 'your_analysis' in result_adata.uns
    assert statistics['mean_expression'] > 0

def test_service_performance(large_dataset_fixture):
    """Test service performance with large datasets."""
    import time

    service = YourService()
    start_time = time.time()

    result_adata, statistics = service.main_analysis_method(large_dataset_fixture)

    duration = time.time() - start_time
    assert duration < 60  # Should complete within 60 seconds
    assert result_adata.n_obs == large_dataset_fixture.n_obs
```

## 📊 Best Practices

### 1. Performance Optimization
```python
# Use efficient NumPy operations
def efficient_computation(data):
    # Good: vectorized operations
    result = np.mean(data, axis=1)

    # Avoid: loops when possible
    # result = [np.mean(row) for row in data]

# Memory management for large datasets
def memory_efficient_analysis(adata):
    # Process in chunks if memory-constrained
    if adata.n_obs > 100000:
        # Implement chunked processing
        pass
    else:
        # Standard processing
        pass
```

### 2. Scientific Rigor
```python
# Proper statistical methods
def calculate_pvalues(data, method='benjamini-hochberg'):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    # Calculate p-values
    pvalues = stats.ttest_1samp(data, 0).pvalue

    # Multiple testing correction
    rejected, pvals_corrected, _, _ = multipletests(
        pvalues, method=method
    )

    return pvals_corrected

# Quality control checks
def validate_scientific_parameters(min_cells=3, min_genes=200):
    if min_cells < 1:
        raise ValueError("Minimum cells must be at least 1")
    if min_genes < 50:
        logger.warning("Very low gene threshold may affect analysis quality")
```

### 3. Error Handling
```python
class ServiceError(Exception):
    """Base service exception."""
    pass

class DataValidationError(ServiceError):
    """Data validation failed."""
    pass

class ComputationError(ServiceError):
    """Computation failed."""
    pass

def robust_computation(data):
    try:
        result = complex_algorithm(data)
    except np.linalg.LinAlgError as e:
        raise ComputationError(f"Linear algebra error: {e}")
    except ValueError as e:
        raise DataValidationError(f"Invalid data: {e}")
    except Exception as e:
        raise ServiceError(f"Unexpected error: {e}")

    return result
```

### 4. Documentation
```python
def well_documented_method(
    adata: anndata.AnnData,
    threshold: float = 0.05,
    method: str = "wilcoxon"
) -> Tuple[anndata.AnnData, Dict[str, Any]]:
    """
    Perform differential expression analysis.

    This method implements [specific algorithm] following the methodology
    described in [reference]. The analysis identifies differentially expressed
    genes between conditions using [statistical test].

    Args:
        adata: AnnData object with expression data and group annotations
        threshold: Significance threshold for adjusted p-values (default: 0.05)
        method: Statistical test method ('wilcoxon', 'ttest', 'deseq2')

    Returns:
        Tuple containing:
            - AnnData object with differential expression results in .var
            - Dictionary with summary statistics:
                * 'n_significant': Number of significant genes
                * 'method_used': Statistical method applied
                * 'threshold_used': Significance threshold
                * 'total_genes_tested': Total number of genes tested

    Raises:
        ValueError: If method is not supported or data lacks required annotations
        ComputationError: If statistical computation fails

    Example:
        >>> service = DifferentialExpressionService()
        >>> result_adata, stats = service.find_markers(adata, threshold=0.01)
        >>> print(f"Found {stats['n_significant']} significant genes")

    Notes:
        - Requires 'group' column in adata.obs for group comparisons
        - Results stored in adata.var['pvals'] and adata.var['pvals_adj']
        - For single-cell data, considers only genes expressed in >3 cells per group
    """
```

## 🎯 Service Categories

### 1. Data Processing Services
- **PreprocessingService**: Filtering, normalization, batch correction
- **QualityService**: Quality control metrics and filtering recommendations
- **ConcatenationService**: Sample merging and batch handling

### 2. Analysis Services
- **ClusteringService**: Cell clustering and dimensionality reduction
- **DifferentialExpressionService**: Statistical comparison between groups
- **EnrichmentService**: Pathway and gene set enrichment analysis

### 3. Visualization Services
- **VisualizationService**: Interactive plotting with Plotly
- **NetworkService**: Network analysis and visualization
- **ReportService**: Automated report generation

### 4. Proteomics-Specific Services
- **ProteomicsPreprocessingService**: Intensity normalization and missing value handling
- **ProteomicsQualityService**: CV analysis and batch detection
- **ProteomicsDifferentialService**: Linear models with empirical Bayes

This comprehensive guide provides everything needed to create robust, scientifically rigorous services that integrate seamlessly with the Lobster AI platform while maintaining high performance and reliability standards.