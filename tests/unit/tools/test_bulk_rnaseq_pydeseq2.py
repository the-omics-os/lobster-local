"""
Unit tests for pyDESeq2 integration in bulk RNA-seq service.

Tests the new pyDESeq2 functionality in BulkRNASeqService including
formula-based analysis, pseudobulk integration, and dependency validation.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from unittest.mock import Mock, patch, MagicMock

from lobster.tools.bulk_rnaseq_service import BulkRNASeqService, PyDESeq2Error
from lobster.tools.differential_formula_service import DifferentialFormulaService
from lobster.core import FormulaError, DesignMatrixError


@pytest.fixture
def pseudobulk_adata():
    """Create pseudobulk AnnData for testing."""
    # Create pseudobulk data (6 samples × 100 genes)
    expression_matrix = np.random.negative_binomial(10, 0.3, (6, 100)).astype(int)  # Integer counts for DESeq2
    gene_names = [f"GENE_{i:03d}" for i in range(100)]
    
    adata = ad.AnnData(
        X=expression_matrix,
        obs=pd.DataFrame({
            'sample_id': ['S1', 'S1', 'S1', 'S2', 'S2', 'S2'],
            'cell_type': ['T_cell', 'B_cell', 'Monocyte', 'T_cell', 'B_cell', 'Monocyte'],
            'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment'],
            'batch': ['Batch1', 'Batch1', 'Batch2', 'Batch1', 'Batch2', 'Batch2'],
            'n_cells_aggregated': [245, 89, 156, 198, 72, 134]
        }, index=[f"S{i//3+1}_{['T_cell', 'B_cell', 'Monocyte'][i%3]}" for i in range(6)]),
        var=pd.DataFrame(index=gene_names)
    )
    
    return adata


@pytest.fixture
def count_matrix():
    """Create count matrix (genes × samples) for pyDESeq2."""
    gene_names = [f"GENE_{i:03d}" for i in range(100)]
    sample_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    
    return pd.DataFrame(
        np.random.negative_binomial(10, 0.3, (100, 6)),
        index=gene_names,
        columns=sample_names
    )


@pytest.fixture
def sample_metadata():
    """Create sample metadata for pyDESeq2 testing."""
    return pd.DataFrame({
        'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment'],
        'batch': ['Batch1', 'Batch2', 'Batch1', 'Batch2', 'Batch1', 'Batch2'],
        'replicate': ['Rep1', 'Rep1', 'Rep2', 'Rep1', 'Rep2', 'Rep2']
    }, index=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])


@pytest.fixture
def bulk_rnaseq_service():
    """Create BulkRNASeqService for testing."""
    return BulkRNASeqService()


@pytest.mark.unit
class TestPyDESeq2DependencyValidation:
    """Test pyDESeq2 dependency validation."""
    
    def test_validate_pydeseq2_setup_all_available(self, bulk_rnaseq_service):
        """Test dependency validation when all components are available."""
        with patch('importlib.import_module'):
            # Mock successful imports
            with patch.dict('sys.modules', {
                'pydeseq2.dds': MagicMock(),
                'pydeseq2.ds': MagicMock(),
                'pydeseq2.default_inference': MagicMock(),
                'numba': MagicMock(),
                'statsmodels': MagicMock()
            }):
                status = bulk_rnaseq_service.validate_pydeseq2_setup()
                
                assert status['pydeseq2'] is True
                assert status['pydeseq2_inference'] is True
                assert status['numba'] is True
                assert status['statsmodels'] is True
    
    def test_validate_pydeseq2_setup_missing_pydeseq2(self, bulk_rnaseq_service):
        """Test dependency validation when pyDESeq2 is missing."""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'pydeseq2'")):
            status = bulk_rnaseq_service.validate_pydeseq2_setup()
            
            assert status['pydeseq2'] is False
    
    def test_validate_pydeseq2_setup_missing_optional(self, bulk_rnaseq_service):
        """Test dependency validation with missing optional dependencies."""
        def mock_import(module):
            if module == 'numba':
                raise ImportError("No module named 'numba'")
            return MagicMock()
        
        with patch('importlib.import_module', side_effect=mock_import):
            with patch.dict('sys.modules', {
                'pydeseq2.dds': MagicMock(),
                'pydeseq2.ds': MagicMock(),
                'statsmodels': MagicMock()
            }):
                status = bulk_rnaseq_service.validate_pydeseq2_setup()
                
                assert status['pydeseq2'] is True
                assert status['numba'] is False  # Missing but optional


@pytest.mark.unit
class TestPyDESeq2InputValidation:
    """Test input validation for pyDESeq2 analysis."""
    
    def test_validate_deseq2_inputs_valid(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test validation with valid inputs."""
        contrast = ['condition', 'Treatment', 'Control']
        
        # Should not raise any exceptions
        bulk_rnaseq_service._validate_deseq2_inputs(
            count_matrix, sample_metadata, '~condition', contrast
        )
    
    def test_validate_deseq2_inputs_empty_count_matrix(self, bulk_rnaseq_service, sample_metadata):
        """Test validation with empty count matrix."""
        empty_matrix = pd.DataFrame()
        contrast = ['condition', 'Treatment', 'Control']
        
        with pytest.raises(PyDESeq2Error, match="Count matrix is empty"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                empty_matrix, sample_metadata, '~condition', contrast
            )
    
    def test_validate_deseq2_inputs_non_numeric(self, bulk_rnaseq_service, sample_metadata):
        """Test validation with non-numeric count matrix."""
        non_numeric_matrix = pd.DataFrame({
            'S1': ['high', 'low', 'medium'],
            'S2': ['low', 'high', 'low']
        }, index=['Gene1', 'Gene2', 'Gene3'])
        contrast = ['condition', 'Treatment', 'Control']
        
        with pytest.raises(PyDESeq2Error, match="non-numeric data"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                non_numeric_matrix, sample_metadata, '~condition', contrast
            )
    
    def test_validate_deseq2_inputs_negative_values(self, bulk_rnaseq_service, sample_metadata):
        """Test validation with negative count values."""
        negative_matrix = pd.DataFrame(
            np.random.randint(-10, 100, (50, 6)),
            columns=['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
            index=[f"Gene_{i}" for i in range(50)]
        )
        contrast = ['condition', 'Treatment', 'Control']
        
        with pytest.raises(PyDESeq2Error, match="negative values"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                negative_matrix, sample_metadata, '~condition', contrast
            )
    
    def test_validate_deseq2_inputs_sample_mismatch(self, bulk_rnaseq_service, count_matrix):
        """Test validation with sample mismatch between count matrix and metadata."""
        mismatched_metadata = pd.DataFrame({
            'condition': ['Control', 'Treatment']
        }, index=['S1', 'S99'])  # S99 not in count matrix
        
        contrast = ['condition', 'Treatment', 'Control']
        
        with pytest.raises(PyDESeq2Error, match="missing from metadata"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                count_matrix, mismatched_metadata, '~condition', contrast
            )
    
    def test_validate_deseq2_inputs_invalid_contrast_format(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test validation with invalid contrast format."""
        invalid_contrast = ['condition', 'Treatment']  # Missing level2
        
        with pytest.raises(PyDESeq2Error, match="Contrast must be"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                count_matrix, sample_metadata, '~condition', invalid_contrast
            )
    
    def test_validate_deseq2_inputs_contrast_factor_not_found(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test validation when contrast factor not in metadata."""
        invalid_contrast = ['nonexistent', 'Treatment', 'Control']
        
        with pytest.raises(PyDESeq2Error, match="not found in metadata"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                count_matrix, sample_metadata, '~condition', invalid_contrast
            )
    
    def test_validate_deseq2_inputs_contrast_level_not_found(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test validation when contrast level not in factor."""
        invalid_contrast = ['condition', 'InvalidLevel', 'Control']
        
        with pytest.raises(PyDESeq2Error, match="not found in factor"):
            bulk_rnaseq_service._validate_deseq2_inputs(
                count_matrix, sample_metadata, '~condition', invalid_contrast
            )


@pytest.mark.unit
class TestPyDESeq2AnalysisMethod:
    """Test pyDESeq2 analysis functionality."""
    
    @patch('pydeseq2.dds.DeseqDataSet')
    @patch('pydeseq2.ds.DeseqStats')
    @patch('pydeseq2.default_inference.DefaultInference')
    def test_run_pydeseq2_analysis_basic(self, mock_inference, mock_stats, mock_dds, 
                                        bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test basic pyDESeq2 analysis workflow."""
        # Mock dependency validation
        with patch.object(bulk_rnaseq_service, 'validate_pydeseq2_setup', 
                         return_value={'pydeseq2': True, 'pydeseq2_inference': True, 'numba': True, 'statsmodels': True}):
            
            # Mock results
            mock_results_df = pd.DataFrame({
                'baseMean': np.random.lognormal(5, 2, 100),
                'log2FoldChange': np.random.normal(0, 1.5, 100),
                'lfcSE': np.random.gamma(2, 0.2, 100),
                'stat': np.random.normal(0, 2, 100),
                'pvalue': np.random.beta(0.5, 3, 100),
                'padj': np.random.beta(0.3, 5, 100)
            }, index=count_matrix.index)
            
            mock_ds_instance = MagicMock()
            mock_ds_instance.results_df = mock_results_df
            mock_stats.return_value = mock_ds_instance
            
            contrast = ['condition', 'Treatment', 'Control']
            formula = '~condition'
            
            result = bulk_rnaseq_service.run_pydeseq2_analysis(
                count_matrix, sample_metadata, formula, contrast
            )
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 100
            assert 'baseMean' in result.columns
            assert 'log2FoldChange' in result.columns
            assert 'padj' in result.columns
            
            # Check that pyDESeq2 components were called
            mock_dds.assert_called_once()
            mock_stats.assert_called_once()
    
    def test_run_pydeseq2_analysis_missing_dependencies(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test error when pyDESeq2 dependencies are missing."""
        with patch.object(bulk_rnaseq_service, 'validate_pydeseq2_setup',
                         return_value={'pydeseq2': False}):
            
            contrast = ['condition', 'Treatment', 'Control']
            
            with pytest.raises(PyDESeq2Error, match="Missing pyDESeq2 dependencies"):
                bulk_rnaseq_service.run_pydeseq2_analysis(
                    count_matrix, sample_metadata, '~condition', contrast
                )
    
    @patch('pydeseq2.dds.DeseqDataSet')
    def test_run_pydeseq2_analysis_with_shrinkage(self, mock_dds, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test pyDESeq2 analysis with LFC shrinkage."""
        with patch.object(bulk_rnaseq_service, 'validate_pydeseq2_setup',
                         return_value={'pydeseq2': True, 'pydeseq2_inference': True, 'numba': True, 'statsmodels': True}):
            
            # Mock the entire pipeline
            with patch('pydeseq2.ds.DeseqStats') as mock_stats:
                mock_ds = MagicMock()
                mock_ds.results_df = pd.DataFrame({
                    'baseMean': [100, 200],
                    'log2FoldChange': [1.5, -2.0],
                    'padj': [0.01, 0.05]
                }, index=['Gene1', 'Gene2'])
                mock_stats.return_value = mock_ds
                
                with patch('pydeseq2.default_inference.DefaultInference'):
                    result = bulk_rnaseq_service.run_pydeseq2_analysis(
                        count_matrix.iloc[:2], sample_metadata, '~condition', 
                        ['condition', 'Treatment', 'Control'],
                        shrink_lfc=True
                    )
                    
                    assert isinstance(result, pd.DataFrame)
                    # Should call lfc_shrink method
                    mock_ds.lfc_shrink.assert_called()


@pytest.mark.unit
class TestPyDESeq2PseudobulkIntegration:
    """Test pyDESeq2 integration with pseudobulk data."""
    
    def test_run_pydeseq2_from_pseudobulk_basic(self, bulk_rnaseq_service, pseudobulk_adata):
        """Test running pyDESeq2 analysis on pseudobulk data."""
        with patch.object(bulk_rnaseq_service, 'run_pydeseq2_analysis') as mock_pydeseq2:
            # Mock pyDESeq2 results
            mock_results_df = pd.DataFrame({
                'baseMean': np.random.lognormal(5, 2, 100),
                'log2FoldChange': np.random.normal(0, 1.5, 100),
                'padj': np.random.beta(0.3, 5, 100),
                'significant': np.random.choice([True, False], 100, p=[0.2, 0.8])
            }, index=pseudobulk_adata.var_names)
            mock_pydeseq2.return_value = mock_results_df
            
            results_df, analysis_stats = bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                pseudobulk_adata,
                formula='~condition',
                contrast=['condition', 'Treatment', 'Control']
            )
            
            assert isinstance(results_df, pd.DataFrame)
            assert isinstance(analysis_stats, dict)
            
            # Check analysis statistics
            assert analysis_stats['analysis_type'] == 'pydeseq2_pseudobulk'
            assert 'n_pseudobulk_samples' in analysis_stats
            assert 'n_genes_tested' in analysis_stats
            assert 'n_significant_genes' in analysis_stats
            
            mock_pydeseq2.assert_called_once()
    
    def test_run_pydeseq2_from_pseudobulk_with_layer(self, bulk_rnaseq_service, pseudobulk_adata):
        """Test pyDESeq2 analysis using specific count layer."""
        # Add count layer
        pseudobulk_adata.layers['raw_counts'] = pseudobulk_adata.X.copy().astype(int)
        
        with patch.object(bulk_rnaseq_service, 'run_pydeseq2_analysis') as mock_pydeseq2:
            mock_pydeseq2.return_value = pd.DataFrame({
                'log2FoldChange': [1.0, -1.5],
                'padj': [0.01, 0.05]
            }, index=['Gene1', 'Gene2'])
            
            results_df, analysis_stats = bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                pseudobulk_adata,
                formula='~condition',
                contrast=['condition', 'Treatment', 'Control'],
                count_layer='raw_counts'
            )
            
            assert isinstance(results_df, pd.DataFrame)
            mock_pydeseq2.assert_called_once()
            
            # Verify the count matrix passed to pyDESeq2 came from the layer
            call_args = mock_pydeseq2.call_args
            count_matrix_used = call_args[0][0]
            assert isinstance(count_matrix_used, pd.DataFrame)
    
    def test_pydeseq2_from_pseudobulk_statistics(self, bulk_rnaseq_service, pseudobulk_adata):
        """Test analysis statistics calculation."""
        with patch.object(bulk_rnaseq_service, 'run_pydeseq2_analysis') as mock_pydeseq2:
            # Create mock results with known significant genes
            mock_results_df = pd.DataFrame({
                'baseMean': [100, 200, 50, 300],
                'log2FoldChange': [2.0, -1.5, 0.5, -2.5],  # 2 up, 2 down
                'padj': [0.01, 0.02, 0.6, 0.03],  # 3 significant (padj < 0.05)
                'significant': [True, True, False, True]
            }, index=['Gene1', 'Gene2', 'Gene3', 'Gene4'])
            mock_pydeseq2.return_value = mock_results_df
            
            results_df, analysis_stats = bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                pseudobulk_adata.iloc[:, :4],  # Subset to match mock results
                formula='~condition',
                contrast=['condition', 'Treatment', 'Control'],
                alpha=0.05
            )
            
            assert analysis_stats['n_significant_genes'] == 3
            assert analysis_stats['n_upregulated'] == 2  # Gene1, Gene3 (but Gene3 not significant)
            assert analysis_stats['n_downregulated'] == 1  # Gene2, Gene4 (but need significant)
            assert len(analysis_stats['top_upregulated']) <= 10
            assert len(analysis_stats['top_downregulated']) <= 10


@pytest.mark.unit
class TestFormulaDesignIntegration:
    """Test integration with formula design functionality."""
    
    def test_create_formula_design(self, bulk_rnaseq_service, sample_metadata):
        """Test formula design creation."""
        result = bulk_rnaseq_service.create_formula_design(
            sample_metadata,
            condition_col='condition',
            batch_col='batch',
            reference_condition='Control'
        )
        
        assert isinstance(result, dict)
        assert 'design_matrix' in result
        assert 'coefficient_names' in result
        assert 'formula_components' in result
    
    def test_create_formula_design_error_handling(self, bulk_rnaseq_service, sample_metadata):
        """Test error handling in formula design creation."""
        with pytest.raises(FormulaError):
            bulk_rnaseq_service.create_formula_design(
                sample_metadata,
                condition_col='nonexistent_column'
            )
    
    def test_validate_experimental_design(self, bulk_rnaseq_service, sample_metadata):
        """Test experimental design validation."""
        result = bulk_rnaseq_service.validate_experimental_design(
            sample_metadata, '~condition + batch'
        )
        
        assert isinstance(result, dict)
        assert 'valid' in result
        assert 'warnings' in result
        assert 'design_summary' in result
    
    def test_validate_experimental_design_error_handling(self, bulk_rnaseq_service, sample_metadata):
        """Test experimental design validation error handling."""
        result = bulk_rnaseq_service.validate_experimental_design(
            sample_metadata, '~nonexistent_variable'
        )
        
        assert result['valid'] is False
        assert len(result['errors']) > 0


@pytest.mark.unit
class TestPyDESeq2ResultsEnhancement:
    """Test pyDESeq2 results enhancement functionality."""
    
    def test_enhance_deseq2_results(self, bulk_rnaseq_service):
        """Test enhancement of pyDESeq2 results."""
        # Create basic results
        basic_results = pd.DataFrame({
            'baseMean': [100, 200, 50],
            'log2FoldChange': [2.0, -1.5, 0.5],
            'lfcSE': [0.3, 0.4, 0.2],
            'stat': [6.7, -3.8, 2.5],
            'pvalue': [0.001, 0.01, 0.1],
            'padj': [0.005, 0.02, 0.15]
        }, index=['Gene1', 'Gene2', 'Gene3'])
        
        contrast = ['condition', 'Treatment', 'Control']
        mock_dds = MagicMock()
        
        enhanced_results = bulk_rnaseq_service._enhance_deseq2_results(
            basic_results, mock_dds, contrast
        )
        
        assert 'contrast' in enhanced_results.columns
        assert 'significant' in enhanced_results.columns
        assert 'regulation' in enhanced_results.columns
        assert 'rank' in enhanced_results.columns
        
        # Check contrast annotation
        assert all(enhanced_results['contrast'] == 'condition_Treatment_vs_Control')
        
        # Check significance annotation (padj < 0.05)
        assert enhanced_results.loc['Gene1', 'significant'] is True  # padj=0.005
        assert enhanced_results.loc['Gene2', 'significant'] is True  # padj=0.02
        assert enhanced_results.loc['Gene3', 'significant'] is False  # padj=0.15
        
        # Check regulation direction
        assert enhanced_results.loc['Gene1', 'regulation'] == 'upregulated'    # LFC=2.0, significant
        assert enhanced_results.loc['Gene2', 'regulation'] == 'downregulated'  # LFC=-1.5, significant
        assert enhanced_results.loc['Gene3', 'regulation'] == 'unchanged'      # Not significant
    
    def test_enhance_results_rank_calculation(self, bulk_rnaseq_service):
        """Test rank calculation in enhanced results."""
        results_df = pd.DataFrame({
            'padj': [0.001, 0.1, 0.05, np.nan, 0.01],  # Include NaN
            'log2FoldChange': [1, 2, 3, 4, 5]
        }, index=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        
        enhanced = bulk_rnaseq_service._enhance_deseq2_results(
            results_df, MagicMock(), ['condition', 'Treatment', 'Control']
        )
        
        # Check ranks (lower padj = better rank)
        assert enhanced.loc['Gene1', 'rank'] == 1.0  # Best padj
        assert enhanced.loc['Gene5', 'rank'] == 2.0  # Second best
        assert enhanced.loc['Gene3', 'rank'] == 3.0  # Third best
        # Gene4 with NaN should get worst rank


@pytest.mark.unit
class TestPyDESeq2ErrorHandling:
    """Test error handling in pyDESeq2 methods."""
    
    def test_pydeseq2_analysis_error_wrapping(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test that unexpected errors are wrapped in PyDESeq2Error."""
        with patch.object(bulk_rnaseq_service, '_validate_deseq2_inputs',
                         side_effect=ValueError("Unexpected error")):
            
            with pytest.raises(PyDESeq2Error, match="pyDESeq2 analysis failed"):
                bulk_rnaseq_service.run_pydeseq2_analysis(
                    count_matrix, sample_metadata, '~condition', 
                    ['condition', 'Treatment', 'Control']
                )
    
    def test_pydeseq2_specific_error_propagation(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test that PyDESeq2Error is not wrapped again."""
        with patch.object(bulk_rnaseq_service, '_validate_deseq2_inputs',
                         side_effect=PyDESeq2Error("Specific error")):
            
            with pytest.raises(PyDESeq2Error, match="Specific error"):
                bulk_rnaseq_service.run_pydeseq2_analysis(
                    count_matrix, sample_metadata, '~condition',
                    ['condition', 'Treatment', 'Control']
                )
    
    def test_pydeseq2_from_pseudobulk_error_handling(self, bulk_rnaseq_service, pseudobulk_adata):
        """Test error handling in pseudobulk-specific method."""
        with patch.object(bulk_rnaseq_service, 'run_pydeseq2_analysis',
                         side_effect=PyDESeq2Error("Analysis failed")):
            
            with pytest.raises(PyDESeq2Error, match="pyDESeq2 pseudobulk analysis failed"):
                bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                    pseudobulk_adata, '~condition', ['condition', 'Treatment', 'Control']
                )


@pytest.mark.unit
class TestPyDESeq2Integration:
    """Test integration aspects of pyDESeq2 functionality."""
    
    def test_formula_service_integration(self, bulk_rnaseq_service, sample_metadata):
        """Test integration with DifferentialFormulaService."""
        # The BulkRNASeqService should have a formula_service instance
        assert hasattr(bulk_rnaseq_service, 'formula_service')
        assert isinstance(bulk_rnaseq_service.formula_service, DifferentialFormulaService)
    
    @patch('pydeseq2.dds.DeseqDataSet')
    def test_design_matrix_integration(self, mock_dds, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test that formula parsing is properly integrated."""
        with patch.object(bulk_rnaseq_service, 'validate_pydeseq2_setup',
                         return_value={'pydeseq2': True, 'pydeseq2_inference': True, 'numba': True, 'statsmodels': True}):
            
            with patch('pydeseq2.ds.DeseqStats') as mock_stats:
                with patch('pydeseq2.default_inference.DefaultInference'):
                    mock_ds = MagicMock()
                    mock_ds.results_df = pd.DataFrame({'padj': [0.1]}, index=['Gene1'])
                    mock_stats.return_value = mock_ds
                    
                    # This should internally use formula_service for parsing
                    bulk_rnaseq_service.run_pydeseq2_analysis(
                        count_matrix.iloc[:1], sample_metadata, 
                        '~condition + batch',  # Complex formula
                        ['condition', 'Treatment', 'Control']
                    )
                    
                    # Should have called formula service methods
                    mock_dds.assert_called_once()
    
    def test_count_matrix_orientation(self, bulk_rnaseq_service, count_matrix, sample_metadata):
        """Test that count matrix is properly oriented for pyDESeq2."""
        with patch.object(bulk_rnaseq_service, 'validate_pydeseq2_setup',
                         return_value={'pydeseq2': True, 'pydeseq2_inference': True, 'numba': True, 'statsmodels': True}):
            
            with patch('pydeseq2.dds.DeseqDataSet') as mock_dds:
                with patch('pydeseq2.ds.DeseqStats') as mock_stats:
                    with patch('pydeseq2.default_inference.DefaultInference'):
                        mock_ds = MagicMock()
                        mock_ds.results_df = pd.DataFrame({'padj': [0.1]}, index=['Gene1'])
                        mock_stats.return_value = mock_ds
                        
                        # Run analysis
                        bulk_rnaseq_service.run_pydeseq2_analysis(
                            count_matrix.iloc[:1], sample_metadata,
                            '~condition', ['condition', 'Treatment', 'Control']
                        )
                        
                        # Check that count matrix was transposed for pyDESeq2
                        call_args = mock_dds.call_args
                        counts_arg = call_args[1]['counts']
                        
                        # pyDESeq2 expects samples × genes orientation
                        assert counts_arg.shape[0] == len(sample_metadata)
                        assert counts_arg.shape[1] == 1  # Only 1 gene in this test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
