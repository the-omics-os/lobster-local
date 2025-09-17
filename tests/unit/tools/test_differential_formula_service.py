"""
Unit tests for differential expression formula service.

Tests the DifferentialFormulaService class functionality including R-style
formula parsing, design matrix construction, and contrast specification.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from lobster.tools.differential_formula_service import DifferentialFormulaService
from lobster.core import FormulaError, DesignMatrixError


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pd.DataFrame({
        'condition': ['Control', 'Control', 'Control', 'Treatment', 'Treatment', 'Treatment'],
        'batch': ['Batch1', 'Batch1', 'Batch2', 'Batch1', 'Batch2', 'Batch2'],
        'sex': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'age': [25, 30, 35, 28, 32, 29],
        'sample_id': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    }, index=['Sample1', 'Sample2', 'Sample3', 'Sample4', 'Sample5', 'Sample6'])


@pytest.fixture
def formula_service():
    """Create DifferentialFormulaService for testing."""
    return DifferentialFormulaService()


@pytest.mark.unit
class TestDifferentialFormulaServiceInit:
    """Test DifferentialFormulaService initialization."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = DifferentialFormulaService()
        
        assert hasattr(service, 'parse_formula')
        assert hasattr(service, 'construct_design_matrix')
        assert hasattr(service, 'label_encoders')
        assert isinstance(service.label_encoders, dict)


@pytest.mark.unit
class TestFormulaParsingBasic:
    """Test basic formula parsing functionality."""
    
    def test_parse_simple_formula(self, formula_service, sample_metadata):
        """Test parsing simple formula."""
        result = formula_service.parse_formula('~condition', sample_metadata)
        
        assert isinstance(result, dict)
        assert result['formula_string'] == '~condition'
        assert result['response_variable'] is None
        assert len(result['predictor_terms']) == 1
        assert result['predictor_terms'][0]['term'] == 'condition'
        assert result['predictor_terms'][0]['type'] == 'main_effect'
    
    def test_parse_formula_with_covariates(self, formula_service, sample_metadata):
        """Test parsing formula with multiple covariates."""
        result = formula_service.parse_formula('~condition + batch + sex', sample_metadata)
        
        assert len(result['predictor_terms']) == 3
        terms = [term['term'] for term in result['predictor_terms']]
        assert 'condition' in terms
        assert 'batch' in terms
        assert 'sex' in terms
    
    def test_parse_formula_with_interaction(self, formula_service, sample_metadata):
        """Test parsing formula with interaction terms."""
        result = formula_service.parse_formula('~condition + batch + condition*batch', sample_metadata)
        
        assert len(result['predictor_terms']) == 3
        
        # Find interaction term
        interaction_term = [t for t in result['predictor_terms'] if t['type'] == 'interaction'][0]
        assert interaction_term['variables'] == ['condition', 'batch']
        assert interaction_term['order'] == 2
    
    def test_parse_formula_continuous_variable(self, formula_service, sample_metadata):
        """Test parsing formula with continuous variables."""
        result = formula_service.parse_formula('~condition + age', sample_metadata)
        
        var_info = result['variable_info']
        assert var_info['condition']['type'] == 'categorical'
        assert var_info['age']['type'] == 'continuous'
        assert var_info['condition']['levels'] == ['Control', 'Treatment']
        assert var_info['age']['levels'] is None
    
    def test_parse_formula_with_reference_levels(self, formula_service, sample_metadata):
        """Test parsing formula with custom reference levels."""
        reference_levels = {'condition': 'Treatment'}
        
        result = formula_service.parse_formula(
            '~condition', sample_metadata, reference_levels=reference_levels
        )
        
        var_info = result['variable_info']['condition']
        assert var_info['reference_level'] == 'Treatment'
        assert var_info['levels'][0] == 'Treatment'  # Reference should be first


@pytest.mark.unit
class TestFormulaParsingValidation:
    """Test formula parsing validation and error handling."""
    
    def test_parse_empty_formula_error(self, formula_service, sample_metadata):
        """Test error on empty formula."""
        with pytest.raises(FormulaError, match="Empty formula"):
            formula_service.parse_formula('~', sample_metadata)
    
    def test_parse_formula_missing_variable(self, formula_service, sample_metadata):
        """Test error when formula variable not in metadata."""
        with pytest.raises(FormulaError, match="Variables not found"):
            formula_service.parse_formula('~nonexistent_variable', sample_metadata)
    
    def test_parse_formula_missing_reference_level(self, formula_service, sample_metadata):
        """Test warning when reference level not found."""
        reference_levels = {'condition': 'NonexistentLevel'}
        
        # Should not raise error but may warn
        result = formula_service.parse_formula(
            '~condition', sample_metadata, reference_levels=reference_levels
        )
        
        # Should fall back to default reference level
        assert result['variable_info']['condition']['reference_level'] == 'Control'  # Alphabetically first
    
    def test_clean_formula_normalization(self, formula_service):
        """Test formula cleaning and normalization."""
        # Test various input formats
        assert formula_service._clean_formula('condition + batch') == '~condition + batch'
        assert formula_service._clean_formula('  ~condition   +   batch  ') == '~condition + batch'
        assert formula_service._clean_formula('~condition') == '~condition'
    
    def test_split_formula_variations(self, formula_service):
        """Test formula splitting with various formats."""
        # Formula with response variable
        response, predictors = formula_service._split_formula('response ~ condition + batch')
        assert response == 'response'
        assert predictors == 'condition + batch'
        
        # Formula without response variable
        response, predictors = formula_service._split_formula('~condition + batch')
        assert response is None
        assert predictors == 'condition + batch'
    
    def test_parse_terms_complex(self, formula_service):
        """Test parsing complex term structures."""
        terms = formula_service._parse_terms('condition + batch + age + condition*batch')
        
        assert len(terms) == 4
        
        # Check main effects
        main_effects = [t for t in terms if t['type'] == 'main_effect']
        assert len(main_effects) == 3
        
        # Check interaction
        interactions = [t for t in terms if t['type'] == 'interaction']
        assert len(interactions) == 1
        assert interactions[0]['variables'] == ['condition', 'batch']


@pytest.mark.unit
class TestDesignMatrixConstruction:
    """Test design matrix construction functionality."""
    
    def test_construct_simple_design_matrix(self, formula_service, sample_metadata):
        """Test construction of simple design matrix."""
        formula_components = formula_service.parse_formula('~condition', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        assert isinstance(result, dict)
        assert 'design_matrix' in result
        assert 'coefficient_names' in result
        
        design_matrix = result['design_matrix']
        coef_names = result['coefficient_names']
        
        # Should have intercept + treatment effect
        assert design_matrix.shape == (6, 2)  # 6 samples, 2 coefficients
        assert '(Intercept)' in coef_names
        assert 'condition[T.Treatment]' in coef_names
        
        # Check intercept column is all 1s
        assert all(design_matrix[:, 0] == 1.0)
    
    def test_construct_design_matrix_with_batch(self, formula_service, sample_metadata):
        """Test design matrix with batch covariate."""
        formula_components = formula_service.parse_formula('~condition + batch', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        design_matrix = result['design_matrix']
        coef_names = result['coefficient_names']
        
        # Should have intercept + condition + batch effects
        expected_cols = 4  # (Intercept), condition[T.Treatment], batch[T.Batch2]
        assert design_matrix.shape == (6, expected_cols)
        
        # Check coefficient names
        assert '(Intercept)' in coef_names
        assert 'condition[T.Treatment]' in coef_names
        assert 'batch[T.Batch2]' in coef_names
    
    def test_construct_design_matrix_continuous(self, formula_service, sample_metadata):
        """Test design matrix with continuous variables."""
        formula_components = formula_service.parse_formula('~condition + age', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        design_matrix = result['design_matrix']
        coef_names = result['coefficient_names']
        
        # Should include age as continuous variable
        assert 'age' in coef_names
        assert design_matrix.shape[1] == 3  # (Intercept), condition[T.Treatment], age
        
        # Age column should be the actual age values
        age_col_idx = coef_names.index('age')
        np.testing.assert_array_equal(design_matrix[:, age_col_idx], sample_metadata['age'].values)
    
    def test_construct_design_matrix_interaction(self, formula_service, sample_metadata):
        """Test design matrix with interaction terms."""
        formula_components = formula_service.parse_formula('~condition + batch + condition*batch', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        design_matrix = result['design_matrix']
        coef_names = result['coefficient_names']
        
        # Should include interaction term
        interaction_terms = [name for name in coef_names if ':' in name]
        assert len(interaction_terms) > 0
    
    def test_design_matrix_validation_nan_values(self, formula_service):
        """Test design matrix validation with NaN values."""
        design_matrix = np.array([[1, 2, np.nan], [1, 3, 4]])
        
        with pytest.raises(DesignMatrixError, match="Design matrix contains NaN values"):
            formula_service._validate_design_matrix(design_matrix, ['Intercept', 'var1', 'var2'])
    
    def test_design_matrix_validation_infinite_values(self, formula_service):
        """Test design matrix validation with infinite values."""
        design_matrix = np.array([[1, 2, 3], [1, np.inf, 4]])
        
        with pytest.raises(DesignMatrixError, match="Design matrix contains infinite values"):
            formula_service._validate_design_matrix(design_matrix, ['Intercept', 'var1', 'var2'])
    
    def test_design_matrix_rank_deficiency_warning(self, formula_service):
        """Test warning for rank deficient design matrix."""
        # Create rank deficient matrix (collinear columns)
        design_matrix = np.array([[1, 2, 4], [1, 3, 6], [1, 4, 8]])  # Third column = 2 * second column
        
        with patch.object(formula_service.logger, 'warning') as mock_warning:
            formula_service._validate_design_matrix(design_matrix, ['Intercept', 'var1', 'var2'])
            mock_warning.assert_called()


@pytest.mark.unit
class TestContrastSpecification:
    """Test contrast specification and validation."""
    
    def test_construct_contrast_basic(self, formula_service, sample_metadata):
        """Test basic contrast construction."""
        formula_components = formula_service.parse_formula('~condition', sample_metadata)
        design_result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        contrast = ['condition', 'Treatment', 'Control']
        contrast_vector, contrast_name = formula_service._construct_contrast(
            contrast, design_result['coefficient_names'], formula_components
        )
        
        assert isinstance(contrast_vector, np.ndarray)
        assert contrast_name == 'condition_Treatment_vs_Control'
        assert len(contrast_vector) == len(design_result['coefficient_names'])
        
        # Treatment vs Control contrast
        coef_names = design_result['coefficient_names']
        treatment_idx = coef_names.index('condition[T.Treatment]')
        assert contrast_vector[treatment_idx] == 1.0
    
    def test_construct_contrast_with_design_matrix(self, formula_service, sample_metadata):
        """Test contrast construction integrated with design matrix."""
        formula_components = formula_service.parse_formula('~condition + batch', sample_metadata)
        contrast = ['condition', 'Treatment', 'Control']
        
        result = formula_service.construct_design_matrix(
            formula_components, sample_metadata, contrast=contrast
        )
        
        assert 'contrast_vector' in result
        assert 'contrast_name' in result
        assert result['contrast_name'] == 'condition_Treatment_vs_Control'
        assert result['contrast_vector'] is not None
    
    def test_construct_contrast_invalid_factor(self, formula_service, sample_metadata):
        """Test error with invalid contrast factor."""
        formula_components = formula_service.parse_formula('~condition', sample_metadata)
        design_result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        invalid_contrast = ['nonexistent', 'Treatment', 'Control']
        
        with pytest.raises(FormulaError, match="Factor 'nonexistent' not found"):
            formula_service._construct_contrast(
                invalid_contrast, design_result['coefficient_names'], formula_components
            )
    
    def test_construct_contrast_invalid_levels(self, formula_service, sample_metadata):
        """Test error with invalid contrast levels."""
        formula_components = formula_service.parse_formula('~condition', sample_metadata)
        design_result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        invalid_contrast = ['condition', 'InvalidLevel', 'Control']
        
        with pytest.raises(FormulaError, match="Level 'InvalidLevel' not found"):
            formula_service._construct_contrast(
                invalid_contrast, design_result['coefficient_names'], formula_components
            )
    
    def test_construct_contrast_wrong_format(self, formula_service):
        """Test error with wrong contrast format."""
        with pytest.raises(FormulaError, match="Contrast must be"):
            formula_service._construct_contrast(
                ['condition', 'Treatment'],  # Missing second level
                ['Intercept', 'condition[T.Treatment]'],
                {'variable_info': {'condition': {'type': 'categorical'}}}
            )
    
    def test_construct_contrast_continuous_variable(self, formula_service, sample_metadata):
        """Test error when trying to contrast continuous variable."""
        formula_components = formula_service.parse_formula('~age', sample_metadata)
        design_result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        invalid_contrast = ['age', '25', '30']  # Age is continuous
        
        with pytest.raises(FormulaError, match="must be categorical for contrasts"):
            formula_service._construct_contrast(
                invalid_contrast, design_result['coefficient_names'], formula_components
            )


@pytest.mark.unit
class TestDesignMatrixProperties:
    """Test design matrix mathematical properties."""
    
    def test_design_matrix_rank(self, formula_service, sample_metadata):
        """Test design matrix rank calculation."""
        formula_components = formula_service.parse_formula('~condition + batch', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        assert 'rank' in result
        assert result['rank'] > 0
        assert result['rank'] <= result['n_coefficients']
    
    def test_design_matrix_full_rank(self, formula_service, sample_metadata):
        """Test full rank design matrix."""
        # Simple design should be full rank
        formula_components = formula_service.parse_formula('~condition', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        assert result['rank'] == result['n_coefficients']
    
    def test_design_matrix_dimensions(self, formula_service, sample_metadata):
        """Test design matrix dimensions are correct."""
        formula_components = formula_service.parse_formula('~condition + batch + sex', sample_metadata)
        result = formula_service.construct_design_matrix(formula_components, sample_metadata)
        
        design_matrix = result['design_matrix']
        
        # Should have correct dimensions
        assert design_matrix.shape[0] == len(sample_metadata)  # One row per sample
        assert design_matrix.shape[1] == len(result['coefficient_names'])  # One col per coefficient
    
    def test_estimate_design_rank(self, formula_service):
        """Test design rank estimation."""
        variable_info = {
            'condition': {'type': 'categorical', 'n_levels': 2},  # +1 coefficient
            'batch': {'type': 'categorical', 'n_levels': 2},      # +1 coefficient  
            'age': {'type': 'continuous', 'n_levels': None}       # +1 coefficient
        }
        
        estimated_rank = formula_service._estimate_design_rank(variable_info)
        
        # 1 (intercept) + 1 (condition) + 1 (batch) + 1 (age) = 4
        assert estimated_rank == 4


@pytest.mark.unit
class TestSimpleDesignCreation:
    """Test simple design creation convenience method."""
    
    def test_create_simple_design_condition_only(self, formula_service, sample_metadata):
        """Test simple design with condition only."""
        result = formula_service.create_simple_design(
            sample_metadata, condition_col='condition'
        )
        
        assert 'design_matrix' in result
        assert 'coefficient_names' in result
        
        # Should be equivalent to ~condition
        coef_names = result['coefficient_names']
        assert '(Intercept)' in coef_names
        assert 'condition[T.Treatment]' in coef_names
    
    def test_create_simple_design_with_batch(self, formula_service, sample_metadata):
        """Test simple design with batch covariate."""
        result = formula_service.create_simple_design(
            sample_metadata, condition_col='condition', batch_col='batch'
        )
        
        coef_names = result['coefficient_names']
        assert 'condition[T.Treatment]' in coef_names
        assert 'batch[T.Batch2]' in coef_names
    
    def test_create_simple_design_reference_condition(self, formula_service, sample_metadata):
        """Test simple design with custom reference condition."""
        result = formula_service.create_simple_design(
            sample_metadata, condition_col='condition', reference_condition='Treatment'
        )
        
        # Control should now be the contrast (non-reference)
        coef_names = result['coefficient_names']
        assert 'condition[T.Control]' in coef_names


@pytest.mark.unit
class TestExperimentalDesignValidation:
    """Test experimental design validation."""
    
    def test_validate_experimental_design_valid(self, formula_service, sample_metadata):
        """Test validation of valid experimental design."""
        result = formula_service.validate_experimental_design(
            sample_metadata, '~condition + batch', min_replicates=2
        )
        
        assert result['valid'] is True
        assert isinstance(result['warnings'], list)
        assert isinstance(result['errors'], list)
        assert 'design_summary' in result
    
    def test_validate_experimental_design_small_sample(self, formula_service):
        """Test validation with small sample size."""
        small_metadata = pd.DataFrame({
            'condition': ['Control', 'Treatment'],
            'batch': ['Batch1', 'Batch1']
        })
        
        result = formula_service.validate_experimental_design(
            small_metadata, '~condition', min_replicates=2
        )
        
        assert 'Small sample size' in str(result['warnings'])
    
    def test_validate_experimental_design_insufficient_replicates(self, formula_service):
        """Test validation with insufficient replicates."""
        unbalanced_metadata = pd.DataFrame({
            'condition': ['Control', 'Control', 'Treatment', 'Treatment', 'Treatment'],
            'batch': ['A', 'B', 'A', 'B', 'A']  # Batch B has only 2 samples
        }, index=[f"S{i}" for i in range(5)])
        
        result = formula_service.validate_experimental_design(
            unbalanced_metadata, '~condition + batch', min_replicates=3
        )
        
        assert len(result['warnings']) > 0
        warning_str = ' '.join(result['warnings'])
        assert 'replicates' in warning_str.lower()
    
    def test_validate_experimental_design_missing_values(self, formula_service):
        """Test validation with missing values."""
        metadata_with_na = pd.DataFrame({
            'condition': ['Control', 'Treatment', None, 'Control'],  # Missing value
            'batch': ['A', 'A', 'B', 'B']
        })
        
        result = formula_service.validate_experimental_design(
            metadata_with_na, '~condition', min_replicates=1
        )
        
        warning_str = ' '.join(result['warnings'])
        assert 'Missing values' in warning_str
    
    def test_validate_experimental_design_error_handling(self, formula_service, sample_metadata):
        """Test error handling in design validation."""
        # Invalid formula should be caught gracefully
        result = formula_service.validate_experimental_design(
            sample_metadata, '~nonexistent_variable', min_replicates=2
        )
        
        assert result['valid'] is False
        assert len(result['errors']) > 0


@pytest.mark.unit
class TestFormulaServiceHelperMethods:
    """Test helper methods in formula service."""
    
    def test_validate_variables(self, formula_service, sample_metadata):
        """Test variable validation against metadata."""
        terms = [
            {'variables': ['condition', 'batch']},
            {'variables': ['sex']}
        ]
        
        # Should not raise error with valid variables
        formula_service._validate_variables(terms, sample_metadata)
        
        # Should raise error with invalid variables
        invalid_terms = [{'variables': ['nonexistent']}]
        with pytest.raises(FormulaError, match="Variables not found"):
            formula_service._validate_variables(invalid_terms, sample_metadata)
    
    def test_analyze_variables(self, formula_service, sample_metadata):
        """Test variable analysis and type detection."""
        terms = [
            {'variables': ['condition', 'age', 'batch']}
        ]
        
        result = formula_service._analyze_variables(terms, sample_metadata)
        
        assert 'condition' in result
        assert 'age' in result
        assert 'batch' in result
        
        assert result['condition']['type'] == 'categorical'
        assert result['age']['type'] == 'continuous'
        assert result['batch']['type'] == 'categorical'
        
        # Check levels for categorical variables
        assert set(result['condition']['levels']) == {'Control', 'Treatment'}
        assert result['age']['levels'] is None
    
    def test_get_variable_columns_categorical(self, formula_service, sample_metadata):
        """Test getting variable columns for categorical variables."""
        var_info = {
            'type': 'categorical',
            'levels': ['Control', 'Treatment']
        }
        
        columns = formula_service._get_variable_columns('condition', var_info, sample_metadata)
        
        assert 'condition[Control]' in columns
        assert 'condition[Treatment]' in columns
        assert len(columns) == 2
    
    def test_get_variable_columns_continuous(self, formula_service, sample_metadata):
        """Test getting variable columns for continuous variables."""
        var_info = {
            'type': 'continuous',
            'levels': None
        }
        
        columns = formula_service._get_variable_columns('age', var_info, sample_metadata)
        
        assert 'age' in columns
        assert len(columns) == 1
        np.testing.assert_array_equal(columns['age'], sample_metadata['age'].values)


@pytest.mark.unit
class TestFormulaServiceErrorHandling:
    """Test error handling in formula service."""
    
    def test_parse_formula_error_wrapping(self, formula_service, sample_metadata):
        """Test that unexpected errors are wrapped in FormulaError."""
        with patch.object(formula_service, '_split_formula', side_effect=ValueError("Unexpected")):
            with pytest.raises(FormulaError, match="Failed to parse formula"):
                formula_service.parse_formula('~condition', sample_metadata)
    
    def test_construct_design_matrix_error_wrapping(self, formula_service, sample_metadata):
        """Test that construction errors are wrapped in DesignMatrixError."""
        formula_components = formula_service.parse_formula('~condition', sample_metadata)
        
        with patch.object(formula_service, '_add_main_effect', side_effect=ValueError("Unexpected")):
            with pytest.raises(DesignMatrixError, match="Failed to construct design matrix"):
                formula_service.construct_design_matrix(formula_components, sample_metadata)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
