"""
Unit tests for ConcatenationService.

These tests verify that the ConcatenationService correctly eliminates code duplication
while maintaining all existing functionality from data_expert.py and geo_service.py.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict

# Import the ConcatenationService and related classes
from lobster.tools.concatenation_service import (
    ConcatenationService,
    ConcatenationStrategy,
    ConcatenationResult,
    ValidationResult,
    MemoryInfo,
    SmartSparseStrategy,
    MemoryEfficientStrategy,
    ConcatenationError,
    IncompatibleSamplesError,
    MemoryLimitError
)


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 instance for testing."""
    mock_dm = Mock()
    mock_dm.cache_dir = Path("/tmp/test_cache")
    mock_dm.console = Mock()
    mock_dm.list_modalities.return_value = []
    mock_dm.get_modality.return_value = None
    mock_dm.load_modality.return_value = Mock()
    mock_dm.save_modality.return_value = "test_path.h5ad"
    return mock_dm


@pytest.fixture
def sample_anndata_objects():
    """Create sample AnnData objects for testing concatenation."""
    try:
        import anndata as ad
    except ImportError:
        pytest.skip("anndata not available")
    
    # Create sample data
    sample1_data = pd.DataFrame(
        np.random.rand(100, 50),
        index=[f"cell_{i}_sample1" for i in range(100)],
        columns=[f"gene_{i}" for i in range(50)]
    )
    
    sample2_data = pd.DataFrame(
        np.random.rand(80, 50),
        index=[f"cell_{i}_sample2" for i in range(80)],
        columns=[f"gene_{i}" for i in range(50)]
    )
    
    # Create AnnData objects
    adata1 = ad.AnnData(X=sample1_data.values)
    adata1.obs_names = sample1_data.index
    adata1.var_names = sample1_data.columns
    adata1.obs['sample'] = 'sample1'
    
    adata2 = ad.AnnData(X=sample2_data.values)
    adata2.obs_names = sample2_data.index
    adata2.var_names = sample2_data.columns
    adata2.obs['sample'] = 'sample2'
    
    return [adata1, adata2]


@pytest.fixture
def sample_dataframes():
    """Create sample DataFrame objects for testing concatenation."""
    df1 = pd.DataFrame(
        np.random.rand(100, 50),
        index=[f"cell_{i}_sample1" for i in range(100)],
        columns=[f"gene_{i}" for i in range(50)]
    )
    
    df2 = pd.DataFrame(
        np.random.rand(80, 45),  # Different number of genes
        index=[f"cell_{i}_sample2" for i in range(80)],
        columns=[f"gene_{i}" for i in range(45)]
    )
    
    return [df1, df2]


class TestConcatenationService:
    """Test the main ConcatenationService class."""
    
    def test_initialization(self, mock_data_manager):
        """Test service initialization."""
        service = ConcatenationService(mock_data_manager)
        
        assert service.data_manager == mock_data_manager
        assert service.console == mock_data_manager.console
        assert service.cache_dir == mock_data_manager.cache_dir / "concatenation"
        assert ConcatenationStrategy.SMART_SPARSE in service.strategies
        assert ConcatenationStrategy.MEMORY_EFFICIENT in service.strategies
    
    def test_auto_detect_samples(self, mock_data_manager):
        """Test automatic sample detection functionality."""
        # Mock available modalities
        mock_data_manager.list_modalities.return_value = [
            "geo_gse12345_sample_gsm001",
            "geo_gse12345_sample_gsm002",
            "geo_gse12345_sample_gsm003",
            "other_modality"
        ]
        
        service = ConcatenationService(mock_data_manager)
        samples = service.auto_detect_samples("geo_gse12345")
        
        expected = [
            "geo_gse12345_sample_gsm001",
            "geo_gse12345_sample_gsm002", 
            "geo_gse12345_sample_gsm003"
        ]
        
        assert samples == expected
        assert "other_modality" not in samples
    
    def test_auto_detect_samples_no_matches(self, mock_data_manager):
        """Test auto-detection when no samples match the pattern."""
        mock_data_manager.list_modalities.return_value = ["unrelated_modality"]
        
        service = ConcatenationService(mock_data_manager)
        samples = service.auto_detect_samples("geo_gse99999")
        
        assert samples == []
    
    def test_concatenate_samples_success(self, mock_data_manager, sample_anndata_objects):
        """Test successful concatenation of AnnData objects."""
        service = ConcatenationService(mock_data_manager)
        
        # Mock strategy execution
        with patch.object(service.strategies[ConcatenationStrategy.SMART_SPARSE], 'concatenate') as mock_concat:
            mock_result = ConcatenationResult(
                data=sample_anndata_objects[0],  # Return first sample as result
                strategy_used=ConcatenationStrategy.SMART_SPARSE,
                success=True,
                statistics={
                    'n_samples': 2,
                    'final_shape': (180, 50),
                    'join_type': 'inner'
                },
                processing_time_seconds=1.5,
                memory_used_mb=10.5
            )
            mock_concat.return_value = mock_result
            
            result_adata, statistics = service.concatenate_samples(
                sample_anndata_objects,
                strategy=ConcatenationStrategy.SMART_SPARSE
            )
            
            # Compare AnnData object attributes instead of direct comparison
            assert result_adata.n_obs == sample_anndata_objects[0].n_obs
            assert result_adata.n_vars == sample_anndata_objects[0].n_vars
            assert statistics['strategy_used'] == 'smart_sparse'
            assert statistics['processing_time_seconds'] == 1.5
            assert statistics['memory_used_mb'] == 10.5
    
    def test_concatenate_samples_failure(self, mock_data_manager, sample_anndata_objects):
        """Test concatenation failure handling."""
        service = ConcatenationService(mock_data_manager)
        
        # Mock strategy failure
        with patch.object(service.strategies[ConcatenationStrategy.SMART_SPARSE], 'concatenate') as mock_concat:
            mock_result = ConcatenationResult(
                success=False,
                error_message="Test error"
            )
            mock_concat.return_value = mock_result
            
            with pytest.raises(ConcatenationError, match="Test error"):
                service.concatenate_samples(sample_anndata_objects)
    
    def test_concatenate_from_modalities_success(self, mock_data_manager, sample_anndata_objects):
        """Test concatenation from modality names."""
        # Mock modality loading
        mock_data_manager.get_modality.side_effect = sample_anndata_objects
        mock_data_manager.list_modalities.return_value = []  # Output modality doesn't exist
        
        service = ConcatenationService(mock_data_manager)
        
        # Mock concatenation success
        with patch.object(service, 'concatenate_samples') as mock_concat_samples:
            mock_concat_samples.return_value = (
                sample_anndata_objects[0],
                {'n_samples': 2, 'strategy_used': 'smart_sparse'}
            )
            
            result_adata, statistics = service.concatenate_from_modalities(
                modality_names=["test_sample1", "test_sample2"],
                output_name="test_output",
                use_intersecting_genes_only=True
            )
            
            # Compare AnnData object attributes instead of direct comparison
            assert result_adata.n_obs == sample_anndata_objects[0].n_obs
            assert result_adata.n_vars == sample_anndata_objects[0].n_vars
            assert statistics['n_samples'] == 2
            
            # Verify modalities were loaded
            assert mock_data_manager.get_modality.call_count == 2
    
    def test_concatenate_from_modalities_missing_modalities(self, mock_data_manager):
        """Test handling of missing modalities."""
        mock_data_manager.get_modality.return_value = None
        
        service = ConcatenationService(mock_data_manager)
        
        with pytest.raises(ConcatenationError, match="No valid modalities could be loaded"):
            service.concatenate_from_modalities(
                modality_names=["missing_modality"],
                output_name="test_output"
            )
    
    def test_validate_concatenation_inputs(self, mock_data_manager, sample_anndata_objects):
        """Test input validation."""
        service = ConcatenationService(mock_data_manager)
        
        # Mock validation success
        with patch.object(service.strategies[ConcatenationStrategy.SMART_SPARSE], 'validate') as mock_validate:
            mock_result = ValidationResult(
                is_valid=True,
                message="Valid samples",
                has_numeric_data=True,
                memory_estimate_mb=50.0
            )
            mock_validate.return_value = mock_result
            
            result = service.validate_concatenation_inputs(sample_anndata_objects)
            
            assert result.is_valid is True
            assert result.message == "Valid samples"
            assert result.has_numeric_data is True
    
    def test_estimate_memory_usage(self, mock_data_manager, sample_anndata_objects):
        """Test memory usage estimation."""
        service = ConcatenationService(mock_data_manager)
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Mock 8GB available memory
            mock_memory.return_value.available = 8 * 1024**3
            
            memory_info = service.estimate_memory_usage(sample_anndata_objects)
            
            assert memory_info.available_gb == 8.0
            assert memory_info.required_gb > 0
            assert memory_info.can_proceed is not None
            assert memory_info.recommended_strategy is not None


class TestSmartSparseStrategy:
    """Test the SmartSparseStrategy class."""
    
    def test_validate_success(self, sample_anndata_objects):
        """Test successful validation of AnnData objects."""
        strategy = SmartSparseStrategy()
        
        result = strategy.validate(sample_anndata_objects)
        
        assert result.is_valid is True
        assert "Valid for sparse concatenation" in result.message
        assert result.has_numeric_data is True
        assert result.memory_estimate_mb is not None
    
    def test_validate_mixed_types(self, sample_anndata_objects, sample_dataframes):
        """Test validation failure with mixed data types."""
        strategy = SmartSparseStrategy()
        
        mixed_data = [sample_anndata_objects[0], sample_dataframes[0]]
        result = strategy.validate(mixed_data)
        
        assert result.is_valid is False
        assert "Mixed sample types not supported" in result.message
    
    def test_validate_empty_input(self):
        """Test validation with empty input."""
        strategy = SmartSparseStrategy()
        
        result = strategy.validate([])
        
        assert result.is_valid is False
        assert "No samples provided" in result.message
    
    @pytest.mark.skipif(not pytest.importorskip("anndata", reason="anndata not available"), reason="anndata required")
    def test_concatenate_anndata_success(self, sample_anndata_objects):
        """Test successful concatenation of AnnData objects."""
        strategy = SmartSparseStrategy()
        
        result = strategy.concatenate(
            sample_anndata_objects,
            use_intersecting_genes_only=True,
            sample_ids=['sample1', 'sample2'],
            batch_key='batch'
        )
        
        assert result.success is True
        assert result.strategy_used == ConcatenationStrategy.SMART_SPARSE
        assert result.data is not None
        assert result.statistics['n_samples'] == 2
        assert result.statistics['join_type'] == 'inner'
    
    def test_concatenate_dataframes_success(self, sample_dataframes):
        """Test successful concatenation of DataFrame objects."""
        strategy = SmartSparseStrategy()
        
        result = strategy.concatenate(
            sample_dataframes,
            use_intersecting_genes_only=False,  # Use union to handle different gene sets
            sample_ids=['sample1', 'sample2'],
            batch_key='batch'
        )
        
        assert result.success is True
        assert result.strategy_used == ConcatenationStrategy.SMART_SPARSE
        assert result.data is not None
        assert result.statistics['n_samples'] == 2
        assert result.statistics['join_type'] == 'outer'
    
    def test_concatenate_empty_input(self):
        """Test concatenation with empty input."""
        strategy = SmartSparseStrategy()
        
        result = strategy.concatenate([])
        
        assert result.success is False
        assert "No samples provided" in result.error_message


class TestMemoryEfficientStrategy:
    """Test the MemoryEfficientStrategy class."""
    
    def test_initialization(self):
        """Test strategy initialization with custom chunk size."""
        strategy = MemoryEfficientStrategy(chunk_size=500)
        
        assert strategy.chunk_size == 500
    
    def test_validate_memory_requirements(self, sample_anndata_objects):
        """Test memory requirements validation."""
        strategy = MemoryEfficientStrategy()
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Mock 4GB available memory
            mock_memory.return_value.available = 4 * 1024**3
            
            result = strategy.validate(sample_anndata_objects)
            
            assert result.is_valid is True
            assert "Memory-efficient processing" in result.message
            assert result.memory_estimate_mb is not None
    
    def test_concatenate_with_memory_monitoring(self, sample_anndata_objects):
        """Test concatenation with memory usage monitoring."""
        strategy = MemoryEfficientStrategy()
        
        with patch('psutil.Process') as mock_process:
            mock_memory_info = Mock()
            mock_memory_info.rss = 1024 * 1024 * 100  # 100MB initial
            mock_process.return_value.memory_info.side_effect = [
                mock_memory_info,  # Before
                Mock(rss=1024 * 1024 * 150)  # After (50MB increase)
            ]
            
            result = strategy.concatenate(
                sample_anndata_objects,
                use_intersecting_genes_only=True
            )
            
            assert result.success is True
            assert result.strategy_used == ConcatenationStrategy.MEMORY_EFFICIENT
            assert result.memory_used_mb == 50.0
            assert 'memory_optimization' in result.statistics


class TestValidationFunctions:
    """Test validation functions extracted from geo_service.py."""
    
    def test_validate_matrices_multithreaded(self, mock_data_manager):
        """Test multithreaded matrix validation."""
        service = ConcatenationService(mock_data_manager)
        
        # Create test matrices
        valid_matrix = pd.DataFrame(
            np.random.rand(100, 50),
            columns=[f"gene_{i}" for i in range(50)]
        )
        
        invalid_matrix = pd.DataFrame(
            np.random.rand(5, 5),  # Too small
            columns=[f"gene_{i}" for i in range(5)]
        )
        
        sample_matrices = {
            'valid_sample': valid_matrix,
            'invalid_sample': invalid_matrix,
            'none_sample': None
        }
        
        validated = service.validate_matrices_multithreaded(sample_matrices)
        
        # Should only include the valid matrix
        assert len(validated) == 1
        assert 'valid_sample' in validated
        assert 'invalid_sample' not in validated
        assert 'none_sample' not in validated
    
    def test_validate_single_matrix_valid(self, mock_data_manager):
        """Test validation of a single valid matrix."""
        service = ConcatenationService(mock_data_manager)
        
        valid_matrix = pd.DataFrame(
            np.random.rand(100, 50),
            columns=[f"gene_{i}" for i in range(50)]
        )
        
        is_valid, message = service._validate_single_matrix('test_sample', valid_matrix)
        
        assert is_valid is True
        assert "Valid matrix" in message
    
    def test_validate_single_matrix_invalid_size(self, mock_data_manager):
        """Test validation of matrix that's too small."""
        service = ConcatenationService(mock_data_manager)
        
        small_matrix = pd.DataFrame(np.random.rand(5, 5))
        
        is_valid, message = service._validate_single_matrix('test_sample', small_matrix)
        
        assert is_valid is False
        assert "Matrix too small" in message
    
    def test_is_valid_expression_matrix_numeric(self, mock_data_manager):
        """Test expression matrix validation with numeric data."""
        service = ConcatenationService(mock_data_manager)
        
        # Valid numeric matrix
        valid_matrix = pd.DataFrame(
            np.random.rand(100, 50),
            columns=[f"gene_{i}" for i in range(50)]
        )
        
        assert service._is_valid_expression_matrix(valid_matrix) is True
    
    def test_is_valid_expression_matrix_non_numeric(self, mock_data_manager):
        """Test expression matrix validation with non-numeric data."""
        service = ConcatenationService(mock_data_manager)
        
        # Non-numeric matrix
        text_matrix = pd.DataFrame([
            ['a', 'b', 'c'],
            ['d', 'e', 'f']
        ])
        
        assert service._is_valid_expression_matrix(text_matrix) is False
    
    def test_is_valid_expression_matrix_large_sampling(self, mock_data_manager):
        """Test expression matrix validation with large matrix (sampling)."""
        service = ConcatenationService(mock_data_manager)
        
        # Create large matrix (> 1M elements)
        large_matrix = pd.DataFrame(
            np.random.rand(2000, 600),  # 1.2M elements
            columns=[f"gene_{i}" for i in range(600)]
        )
        
        with patch('numpy.random.choice') as mock_choice:
            mock_choice.return_value = np.arange(100000)  # Sample indices
            
            result = service._is_valid_expression_matrix(large_matrix)
            
            assert result is True
            # Verify sampling was used
            mock_choice.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_concatenation_error_inheritance(self):
        """Test custom exception hierarchy."""
        assert issubclass(IncompatibleSamplesError, ConcatenationError)
        assert issubclass(MemoryLimitError, ConcatenationError)
        assert issubclass(ConcatenationError, Exception)
    
    def test_concatenation_error_with_invalid_strategy(self, mock_data_manager, sample_anndata_objects):
        """Test handling of invalid concatenation strategy."""
        service = ConcatenationService(mock_data_manager)
        
        # Use a strategy not in the service's strategies dict
        invalid_strategy = ConcatenationStrategy.PROTEOMICS  # Not implemented
        
        # Mock the strategy execution to verify fallback to SMART_SPARSE
        with patch.object(service.strategies[ConcatenationStrategy.SMART_SPARSE], 'concatenate') as mock_concat:
            mock_result = ConcatenationResult(
                data=sample_anndata_objects[0],
                strategy_used=ConcatenationStrategy.SMART_SPARSE,
                success=True,
                statistics={'n_samples': 2}
            )
            mock_concat.return_value = mock_result
            
            # This should log a warning and fallback to SMART_SPARSE
            result_adata, statistics = service.concatenate_samples(
                sample_anndata_objects,
                strategy=invalid_strategy
            )
            
            # Verify it fell back to SMART_SPARSE
            assert statistics['strategy_used'] == 'smart_sparse'
            # Verify the SMART_SPARSE strategy was called
            mock_concat.assert_called_once()
    
    def test_memory_estimation_error_handling(self, mock_data_manager):
        """Test memory estimation with error conditions."""
        service = ConcatenationService(mock_data_manager)
        
        # Create mock data that will cause estimation errors
        problematic_data = [Mock(spec=['unknown_method'])]  # No shape or n_obs/n_vars
        
        memory_info = service.estimate_memory_usage(problematic_data)
        
        # Should return safe defaults
        assert memory_info.available_gb == 0.0
        assert memory_info.required_gb == float('inf')
        assert memory_info.can_proceed is False


class TestIntegration:
    """Integration tests that verify the service works with DataManagerV2."""
    
    def test_determine_adapter_from_data(self, mock_data_manager):
        """Test adapter determination logic."""
        service = ConcatenationService(mock_data_manager)
        
        # Mock AnnData object with different gene counts
        mock_adata_sc = Mock()
        mock_adata_sc.n_vars = 10000  # High gene count -> single-cell
        
        mock_adata_bulk = Mock()
        mock_adata_bulk.n_vars = 1000  # Medium gene count -> bulk
        
        mock_adata_proteomics = Mock()
        mock_adata_proteomics.n_vars = 100  # Low gene count -> proteomics
        
        assert service._determine_adapter_from_data(mock_adata_sc) == "transcriptomics_single_cell"
        assert service._determine_adapter_from_data(mock_adata_bulk) == "transcriptomics_bulk"
        assert service._determine_adapter_from_data(mock_adata_proteomics) == "proteomics_ms"
    
    @pytest.mark.skipif(not pytest.importorskip("anndata", reason="anndata not available"), reason="anndata required")
    def test_end_to_end_concatenation_workflow(self, mock_data_manager, sample_anndata_objects):
        """Test complete concatenation workflow from start to finish."""
        # Setup mock data manager
        mock_data_manager.get_modality.side_effect = sample_anndata_objects
        mock_data_manager.list_modalities.return_value = ["sample1", "sample2"]
        
        service = ConcatenationService(mock_data_manager)
        
        # Test auto-detection
        samples = service.auto_detect_samples("geo_gse12345")
        assert len(samples) == 0  # No matching pattern
        
        # Test validation
        validation_result = service.validate_concatenation_inputs(sample_anndata_objects)
        assert validation_result.is_valid is True
        
        # Test memory estimation
        memory_info = service.estimate_memory_usage(sample_anndata_objects)
        assert memory_info.required_gb > 0
        
        # Test concatenation
        result_adata, statistics = service.concatenate_samples(sample_anndata_objects)
        assert result_adata is not None
        assert statistics['strategy_used'] == 'smart_sparse'


if __name__ == '__main__':
    pytest.main([__file__])
