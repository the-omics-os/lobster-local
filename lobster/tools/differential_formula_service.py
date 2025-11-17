"""
Differential expression formula service for statistical design.

This module provides the DifferentialFormulaService that handles parsing
R-style formulas and constructing design matrices for differential expression
analysis, supporting complex experimental designs with covariates.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lobster.core import DesignMatrixError, FormulaError
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class DifferentialFormulaService:
    """
    Service for formula-based differential expression design.

    This service parses R-style formulas and constructs design matrices
    for differential expression analysis, handling complex experimental
    designs with multiple factors and covariates.
    """

    def __init__(self):
        """Initialize the formula service."""
        self.logger = logger
        self.label_encoders = {}  # Store encoders for categorical variables

    def parse_formula(
        self,
        formula: str,
        metadata: pd.DataFrame,
        reference_levels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Parse R-style formula into components and validate against metadata.

        Args:
            formula: R-style formula string (e.g., "~condition + batch")
            metadata: Sample metadata DataFrame
            reference_levels: Optional reference levels for categorical variables

        Returns:
            Dict[str, Any]: Parsed formula components

        Raises:
            FormulaError: If formula is invalid or variables not found
        """
        self.logger.info(f"Parsing formula: {formula}")

        try:
            # Clean and validate formula
            formula = self._clean_formula(formula)

            # Parse formula components
            response_var, predictor_vars = self._split_formula(formula)

            # Parse predictor terms
            terms = self._parse_terms(predictor_vars)

            # Validate variables against metadata
            self._validate_variables(terms, metadata)

            # Analyze variable types
            variable_info = self._analyze_variables(terms, metadata, reference_levels)

            # Create formula components dictionary
            formula_components = {
                "formula_string": formula,
                "response_variable": response_var,
                "predictor_terms": terms,
                "variable_info": variable_info,
                "reference_levels": reference_levels or {},
                "n_samples": len(metadata),
                "design_rank": self._estimate_design_rank(variable_info),
            }

            self.logger.info(
                f"Formula parsed successfully: {len(terms)} terms, rank ≈ {formula_components['design_rank']}"
            )

            return formula_components

        except Exception as e:
            if isinstance(e, FormulaError):
                raise
            else:
                raise FormulaError(f"Failed to parse formula '{formula}': {e}")

    def construct_design_matrix(
        self,
        formula_components: Dict[str, Any],
        metadata: pd.DataFrame,
        contrast: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Construct design matrix from parsed formula components.

        Args:
            formula_components: Parsed formula from parse_formula()
            metadata: Sample metadata DataFrame
            contrast: Optional contrast specification [factor, level1, level2]

        Returns:
            Dict[str, Any]: Design matrix and related information

        Raises:
            DesignMatrixError: If construction fails
        """
        self.logger.info("Constructing design matrix...")

        try:
            # Initialize design matrix with intercept
            design_df = pd.DataFrame(index=metadata.index)
            design_df["(Intercept)"] = 1.0

            # Add terms to design matrix
            for term in formula_components["predictor_terms"]:
                if term["type"] == "main_effect":
                    design_df = self._add_main_effect(
                        design_df, term, metadata, formula_components
                    )
                elif term["type"] == "interaction":
                    design_df = self._add_interaction(
                        design_df, term, metadata, formula_components
                    )

            # Convert to numpy array
            design_matrix = design_df.values.astype(np.float64)

            # Check design matrix properties
            self._validate_design_matrix(design_matrix, design_df.columns)

            # Create coefficient names
            coef_names = list(design_df.columns)

            # Construct contrast vector if specified
            contrast_vector = None
            contrast_name = None
            if contrast:
                contrast_vector, contrast_name = self._construct_contrast(
                    contrast, coef_names, formula_components
                )

            # Create result dictionary
            result = {
                "design_matrix": design_matrix,
                "design_df": design_df,
                "coefficient_names": coef_names,
                "n_coefficients": design_matrix.shape[1],
                "rank": np.linalg.matrix_rank(design_matrix),
                "contrast_vector": contrast_vector,
                "contrast_name": contrast_name,
                "formula_components": formula_components,
            }

            self.logger.info(
                f"Design matrix constructed: {design_matrix.shape[0]} samples × "
                f"{design_matrix.shape[1]} coefficients (rank: {result['rank']})"
            )

            return result

        except Exception as e:
            if isinstance(e, DesignMatrixError):
                raise
            else:
                raise DesignMatrixError(f"Failed to construct design matrix: {e}")

    def _clean_formula(self, formula: str) -> str:
        """Clean and normalize formula string."""

        # Remove extra whitespace
        formula = re.sub(r"\s+", " ", formula.strip())

        # Ensure it starts with ~
        if not formula.startswith("~"):
            formula = "~" + formula

        # Basic validation
        if formula == "~":
            raise FormulaError("Empty formula")

        return formula

    def _split_formula(self, formula: str) -> Tuple[Optional[str], str]:
        """Split formula into response and predictor parts."""

        parts = formula.split("~")

        if len(parts) == 2:
            response = parts[0].strip() if parts[0].strip() else None
            predictors = parts[1].strip()
        elif len(parts) == 1:
            response = None
            predictors = parts[0].replace("~", "").strip()
        else:
            raise FormulaError(f"Invalid formula format: {formula}")

        if not predictors:
            raise FormulaError("No predictor variables specified")

        return response, predictors

    def _parse_terms(self, predictor_string: str) -> List[Dict[str, Any]]:
        """Parse predictor terms from formula string."""

        terms = []

        # Split by + operator (handling spaces)
        term_strings = [t.strip() for t in predictor_string.split("+")]

        for term_str in term_strings:
            if not term_str:
                continue

            # Check for interaction terms (containing *)
            if "*" in term_str:
                # Interaction term
                variables = [v.strip() for v in term_str.split("*")]
                terms.append(
                    {
                        "term": term_str,
                        "type": "interaction",
                        "variables": variables,
                        "order": len(variables),
                    }
                )
            else:
                # Main effect term
                terms.append(
                    {
                        "term": term_str,
                        "type": "main_effect",
                        "variables": [term_str],
                        "order": 1,
                    }
                )

        return terms

    def _validate_variables(
        self, terms: List[Dict[str, Any]], metadata: pd.DataFrame
    ) -> None:
        """Validate that all variables exist in metadata."""

        all_variables = set()
        for term in terms:
            all_variables.update(term["variables"])

        missing_vars = []
        for var in all_variables:
            if var not in metadata.columns:
                missing_vars.append(var)

        if missing_vars:
            raise FormulaError(
                f"Variables not found in metadata: {missing_vars}. "
                f"Available variables: {list(metadata.columns)}"
            )

    def _analyze_variables(
        self,
        terms: List[Dict[str, Any]],
        metadata: pd.DataFrame,
        reference_levels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze variable types and properties."""

        variable_info = {}
        all_variables = set()

        for term in terms:
            all_variables.update(term["variables"])

        for var in all_variables:
            series = metadata[var]

            # Determine variable type
            if pd.api.types.is_numeric_dtype(series):
                var_type = "continuous"
                levels = None
                n_levels = None
            else:
                var_type = "categorical"
                levels = list(series.unique())
                n_levels = len(levels)

                # Sort levels consistently
                if reference_levels and var in reference_levels:
                    ref_level = reference_levels[var]
                    if ref_level in levels:
                        levels = [ref_level] + [
                            level for level in sorted(levels) if level != ref_level
                        ]
                    else:
                        self.logger.warning(
                            f"Reference level '{ref_level}' not found for variable '{var}'"
                        )
                        levels = sorted(levels)
                else:
                    levels = sorted(levels)

            variable_info[var] = {
                "type": var_type,
                "levels": levels,
                "n_levels": n_levels,
                "n_missing": series.isna().sum(),
                "reference_level": levels[0] if levels else None,
            }

        return variable_info

    def _estimate_design_rank(self, variable_info: Dict[str, Dict[str, Any]]) -> int:
        """Estimate rank of design matrix."""

        rank = 1  # Intercept

        for var, info in variable_info.items():
            if info["type"] == "continuous":
                rank += 1
            elif info["type"] == "categorical":
                # Number of levels - 1 (reference level)
                rank += max(0, info["n_levels"] - 1)

        return rank

    def _add_main_effect(
        self,
        design_df: pd.DataFrame,
        term: Dict[str, Any],
        metadata: pd.DataFrame,
        formula_components: Dict[str, Any],
    ) -> pd.DataFrame:
        """Add main effect term to design matrix."""

        var = term["variables"][0]
        var_info = formula_components["variable_info"][var]

        if var_info["type"] == "continuous":
            # Continuous variable - add as-is
            design_df[var] = metadata[var].values

        elif var_info["type"] == "categorical":
            # Categorical variable - create dummy variables
            levels = var_info["levels"]
            levels[0]

            # Create dummy variables for all levels except reference
            for level in levels[1:]:
                col_name = f"{var}[T.{level}]"
                design_df[col_name] = (metadata[var] == level).astype(float)

        return design_df

    def _add_interaction(
        self,
        design_df: pd.DataFrame,
        term: Dict[str, Any],
        metadata: pd.DataFrame,
        formula_components: Dict[str, Any],
    ) -> pd.DataFrame:
        """Add interaction term to design matrix."""

        variables = term["variables"]

        if len(variables) == 2:
            # Two-way interaction
            var1, var2 = variables
            var1_info = formula_components["variable_info"][var1]
            var2_info = formula_components["variable_info"][var2]

            # Get individual variable columns (without intercept effects)
            var1_cols = self._get_variable_columns(var1, var1_info, metadata)
            var2_cols = self._get_variable_columns(var2, var2_info, metadata)

            # Create interaction columns
            for col1_name, col1_values in var1_cols.items():
                for col2_name, col2_values in var2_cols.items():
                    interaction_name = f"{col1_name}:{col2_name}"
                    design_df[interaction_name] = col1_values * col2_values

        else:
            # Higher-order interactions (not commonly used)
            raise DesignMatrixError(
                f"Higher-order interactions (>{len(variables)}) not yet supported"
            )

        return design_df

    def _get_variable_columns(
        self, var: str, var_info: Dict[str, Any], metadata: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Get columns for a variable (for interaction terms)."""

        columns = {}

        if var_info["type"] == "continuous":
            columns[var] = metadata[var].values

        elif var_info["type"] == "categorical":
            levels = var_info["levels"]

            # Include all levels for interactions (not just non-reference)
            for level in levels:
                col_name = f"{var}[{level}]"
                columns[col_name] = (metadata[var] == level).astype(float)

        return columns

    def _validate_design_matrix(
        self, design_matrix: np.ndarray, column_names: List[str]
    ) -> None:
        """Validate design matrix properties."""

        # Check for NaN or infinite values
        if np.any(np.isnan(design_matrix)):
            raise DesignMatrixError("Design matrix contains NaN values")

        if np.any(np.isinf(design_matrix)):
            raise DesignMatrixError("Design matrix contains infinite values")

        # Check rank
        rank = np.linalg.matrix_rank(design_matrix)
        n_cols = design_matrix.shape[1]

        if rank < n_cols:
            self.logger.warning(
                f"Design matrix is rank deficient: rank {rank} < {n_cols} columns. "
                "This may indicate collinear variables."
            )

        # Check for constant columns (other than intercept)
        for i, col_name in enumerate(column_names):
            if col_name != "(Intercept)":
                col_values = design_matrix[:, i]
                if np.all(col_values == col_values[0]):
                    self.logger.warning(f"Column '{col_name}' is constant")

    def _construct_contrast(
        self,
        contrast: List[str],
        coef_names: List[str],
        formula_components: Dict[str, Any],
    ) -> Tuple[np.ndarray, str]:
        """
        Construct contrast vector for hypothesis testing.

        Args:
            contrast: [factor, level1, level2] specification
            coef_names: List of coefficient names
            formula_components: Parsed formula components

        Returns:
            Tuple of contrast vector and contrast name
        """
        if len(contrast) != 3:
            raise FormulaError("Contrast must be [factor, level1, level2]")

        factor, level1, level2 = contrast

        # Check if factor exists
        if factor not in formula_components["variable_info"]:
            raise FormulaError(f"Factor '{factor}' not found in formula")

        var_info = formula_components["variable_info"][factor]

        if var_info["type"] != "categorical":
            raise FormulaError(f"Factor '{factor}' must be categorical for contrasts")

        # Check if levels exist
        if level1 not in var_info["levels"]:
            raise FormulaError(f"Level '{level1}' not found in factor '{factor}'")

        if level2 not in var_info["levels"]:
            raise FormulaError(f"Level '{level2}' not found in factor '{factor}'")

        # Construct contrast vector
        contrast_vector = np.zeros(len(coef_names))

        # Find coefficient names for these levels
        ref_level = var_info["reference_level"]

        # Handle reference level encoding
        if level1 == ref_level:
            coef1_name = None  # Reference level has coefficient 0
        else:
            coef1_name = f"{factor}[T.{level1}]"

        if level2 == ref_level:
            coef2_name = None
        else:
            coef2_name = f"{factor}[T.{level2}]"

        # Set contrast coefficients
        if coef1_name and coef1_name in coef_names:
            contrast_vector[coef_names.index(coef1_name)] = 1.0

        if coef2_name and coef2_name in coef_names:
            contrast_vector[coef_names.index(coef2_name)] = -1.0

        contrast_name = f"{factor}_{level1}_vs_{level2}"

        return contrast_vector, contrast_name

    def create_simple_design(
        self,
        metadata: pd.DataFrame,
        condition_col: str,
        batch_col: Optional[str] = None,
        reference_condition: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create simple design for common use cases.

        Args:
            metadata: Sample metadata
            condition_col: Main condition column
            batch_col: Optional batch column
            reference_condition: Reference level for condition

        Returns:
            Dict[str, Any]: Design matrix information
        """
        # Build formula
        formula = f"~{condition_col}"
        if batch_col:
            formula += f" + {batch_col}"

        # Set reference levels
        reference_levels = {}
        if reference_condition:
            reference_levels[condition_col] = reference_condition

        # Parse and construct
        formula_components = self.parse_formula(formula, metadata, reference_levels)
        design_result = self.construct_design_matrix(formula_components, metadata)

        return design_result

    def validate_experimental_design(
        self, metadata: pd.DataFrame, formula: str, min_replicates: int = 2
    ) -> Dict[str, Any]:
        """
        Validate experimental design for statistical power.

        Args:
            metadata: Sample metadata
            formula: Formula string
            min_replicates: Minimum replicates per condition

        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            formula_components = self.parse_formula(formula, metadata)

            validation_results = {
                "valid": True,
                "warnings": [],
                "errors": [],
                "design_summary": {},
            }

            # Check sample size
            n_samples = len(metadata)
            if n_samples < 6:
                validation_results["warnings"].append(
                    f"Small sample size: {n_samples} samples"
                )

            # Check balance for categorical variables
            for var, info in formula_components["variable_info"].items():
                if info["type"] == "categorical":
                    counts = metadata[var].value_counts()
                    min_count = counts.min()

                    if min_count < min_replicates:
                        validation_results["warnings"].append(
                            f"Variable '{var}' has levels with <{min_replicates} replicates: {dict(counts)}"
                        )

                    validation_results["design_summary"][var] = dict(counts)

            # Check for missing values
            missing_vars = []
            for var in formula_components["variable_info"]:
                if metadata[var].isna().any():
                    missing_vars.append(var)

            if missing_vars:
                validation_results["warnings"].append(
                    f"Missing values in: {missing_vars}"
                )

            return validation_results

        except Exception as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
                "design_summary": {},
            }

    def suggest_formulas(
        self, metadata: pd.DataFrame, analysis_goal: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest appropriate formulas based on metadata structure.

        Analyzes available columns, their types, and distributions to suggest
        statistically appropriate formulas for differential expression analysis.

        Args:
            metadata: Sample metadata DataFrame
            analysis_goal: Optional analysis goal description

        Returns:
            List[Dict[str, Any]]: List of formula suggestions with details
        """
        self.logger.info("Analyzing metadata to suggest appropriate formulas")

        try:
            n_samples = len(metadata)
            suggestions = []

            # Analyze variables
            variable_info = {}
            for col in metadata.columns:
                if col.startswith("_") or col in ["n_cells", "total_counts"]:
                    continue

                series = metadata[col]
                if pd.api.types.is_numeric_dtype(series):
                    var_type = "continuous"
                    unique_vals = len(series.unique())
                    missing = series.isna().sum()
                else:
                    var_type = "categorical"
                    unique_vals = len(series.unique())
                    missing = series.isna().sum()

                # Only consider useful variables
                if missing < n_samples / 2 and unique_vals > 1:
                    variable_info[col] = {
                        "type": var_type,
                        "unique_values": unique_vals,
                        "missing_count": missing,
                        "levels": (
                            list(series.unique()) if var_type == "categorical" else None
                        ),
                    }

            # Categorize variables
            categorical_vars = [
                col
                for col, info in variable_info.items()
                if info["type"] == "categorical"
                and info["unique_values"] < n_samples / 2
            ]
            continuous_vars = [
                col
                for col, info in variable_info.items()
                if info["type"] == "continuous"
            ]

            # Identify main condition (binary categorical)
            main_conditions = [
                col
                for col in categorical_vars
                if variable_info[col]["unique_values"] == 2
            ]

            # Identify batch variables
            batch_vars = []
            for col in categorical_vars:
                if col.lower() in ["batch", "sample", "donor", "patient", "subject"]:
                    batch_vars.append(col)
                elif (
                    variable_info[col]["unique_values"] > 2
                    and variable_info[col]["unique_values"] <= 6
                ):
                    batch_vars.append(col)

            if main_conditions:
                main_condition = main_conditions[0]

                # Simple formula
                suggestions.append(
                    {
                        "formula": f"~{main_condition}",
                        "complexity": "simple",
                        "description": f"Direct comparison between {main_condition} groups",
                        "pros": ["Maximum statistical power", "Clear interpretation"],
                        "cons": ["Ignores confounders", "May have batch effects"],
                        "recommended_for": "Initial exploratory analysis",
                        "min_samples_needed": 6,
                        "variables_used": [main_condition],
                    }
                )

                # Batch-corrected formula
                if batch_vars:
                    primary_batch = batch_vars[0]
                    suggestions.append(
                        {
                            "formula": f"~{main_condition} + {primary_batch}",
                            "complexity": "batch_corrected",
                            "description": f"Compare {main_condition} accounting for {primary_batch} effects",
                            "pros": ["Controls batch effects", "More robust estimates"],
                            "cons": ["Reduced power", "Requires balanced design"],
                            "recommended_for": "Multi-batch experiments",
                            "min_samples_needed": 8,
                            "variables_used": [main_condition, primary_batch],
                        }
                    )

                # Multi-factor formula
                if len(batch_vars) > 1 or continuous_vars:
                    covariates = batch_vars[:2] + continuous_vars[:1]
                    all_vars = [main_condition] + covariates
                    suggestions.append(
                        {
                            "formula": f'~{" + ".join(all_vars)}',
                            "complexity": "multifactor",
                            "description": f"Comprehensive model with {len(covariates)} covariates",
                            "pros": [
                                "Controls multiple confounders",
                                "Publication ready",
                            ],
                            "cons": [
                                "Requires larger sample size",
                                "Complex interpretation",
                            ],
                            "recommended_for": "Final analysis with adequate samples",
                            "min_samples_needed": max(12, len(all_vars) * 3),
                            "variables_used": all_vars,
                        }
                    )

                # Interaction formula if justified
                if batch_vars and n_samples >= 16:
                    interaction_vars = [main_condition, batch_vars[0]]
                    suggestions.append(
                        {
                            "formula": f"~{main_condition}*{batch_vars[0]}",
                            "complexity": "interaction",
                            "description": f"Interaction between {main_condition} and {batch_vars[0]}",
                            "pros": [
                                "Captures interaction effects",
                                "Flexible modeling",
                            ],
                            "cons": [
                                "High sample requirement",
                                "Complex interpretation",
                            ],
                            "recommended_for": "When condition effects vary by batch",
                            "min_samples_needed": 16,
                            "variables_used": interaction_vars,
                        }
                    )

            self.logger.info(f"Generated {len(suggestions)} formula suggestions")
            return suggestions

        except Exception as e:
            self.logger.error(f"Error suggesting formulas: {e}")
            return []

    def preview_design_matrix(
        self, formula: str, metadata: pd.DataFrame, max_rows: int = 5
    ) -> str:
        """
        Generate human-readable preview of design matrix.

        Shows first few rows of design matrix with column explanations,
        useful for agent to show users what their formula produces.

        Args:
            formula: R-style formula
            metadata: Sample metadata
            max_rows: Maximum rows to show in preview

        Returns:
            str: Formatted design matrix preview
        """
        try:
            # Parse formula and construct design matrix
            formula_components = self.parse_formula(formula, metadata)
            design_result = self.construct_design_matrix(formula_components, metadata)

            design_df = design_result["design_df"]

            # Create preview text
            preview = f"Design Matrix Preview ({design_df.shape[0]} × {design_df.shape[1]}):\n\n"

            # Show first few rows
            preview_df = design_df.head(max_rows)
            preview += preview_df.to_string(
                max_cols=8, float_format=lambda x: f"{x:.2f}"
            )

            if len(design_df) > max_rows:
                preview += f"\n... and {len(design_df) - max_rows} more rows"

            preview += "\n\nColumn Explanations:\n"

            # Explain each column
            for col in design_df.columns:
                if col == "(Intercept)":
                    preview += f"• {col}: Baseline/intercept term\n"
                elif "[T." in col:
                    # Categorical variable level
                    var_name = col.split("[T.")[0]
                    level_name = col.split("[T.")[1].rstrip("]")
                    preview += (
                        f"• {col}: Effect of {var_name}={level_name} vs reference\n"
                    )
                elif ":" in col:
                    # Interaction term
                    vars_in_interaction = col.split(":")
                    preview += f"• {col}: Interaction between {' and '.join(vars_in_interaction)}\n"
                else:
                    # Continuous variable
                    preview += f"• {col}: Continuous variable effect\n"

            # Add design properties
            rank = np.linalg.matrix_rank(design_result["design_matrix"])
            n_cols = design_result["design_matrix"].shape[1]

            preview += "\nDesign Properties:\n"
            preview += f"• Matrix rank: {rank}/{n_cols} ({'full rank' if rank == n_cols else 'rank deficient'})\n"
            preview += f"• Degrees of freedom: {len(metadata) - n_cols}\n"

            return preview

        except Exception as e:
            return f"Error creating design matrix preview: {str(e)}"

    def estimate_statistical_power(
        self, design_matrix: np.ndarray, effect_size: float = 0.5, alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Quick power estimation for the experimental design.

        Provides rough power estimate to help users understand if their
        design has sufficient samples for detecting effects.

        Args:
            design_matrix: Design matrix from construct_design_matrix
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level

        Returns:
            Dict[str, float]: Power estimates and recommendations
        """
        try:
            n_samples = design_matrix.shape[0]
            n_params = design_matrix.shape[1]
            df_residual = n_samples - n_params

            # Simple power approximation based on sample size and effect size
            # This is a rough estimate, not exact power calculation

            # Standard effect size interpretations
            if effect_size >= 0.8:
                effect_category = "large"
            elif effect_size >= 0.5:
                effect_category = "medium"
            elif effect_size >= 0.2:
                effect_category = "small"
            else:
                effect_category = "very small"

            # Rough power estimates based on sample size and effect size
            # These are approximations for guidance only
            if df_residual < 4:
                estimated_power = 0.1
                power_category = "very low"
            elif df_residual < 10:
                if effect_size >= 0.8:
                    estimated_power = 0.6
                elif effect_size >= 0.5:
                    estimated_power = 0.4
                else:
                    estimated_power = 0.2
                power_category = "low"
            elif df_residual < 20:
                if effect_size >= 0.8:
                    estimated_power = 0.85
                elif effect_size >= 0.5:
                    estimated_power = 0.65
                else:
                    estimated_power = 0.35
                power_category = "moderate"
            else:
                if effect_size >= 0.8:
                    estimated_power = 0.95
                elif effect_size >= 0.5:
                    estimated_power = 0.80
                else:
                    estimated_power = 0.50
                power_category = "good" if estimated_power >= 0.8 else "moderate"

            # Recommendations
            recommendations = []
            if estimated_power < 0.5:
                recommendations.append("Consider increasing sample size")
                if n_params > 5:
                    recommendations.append("Consider simplifying the model")
            elif estimated_power < 0.8:
                recommendations.append(
                    "Power is moderate - consider more samples for robust results"
                )
            else:
                recommendations.append(
                    "Good power for detecting medium to large effects"
                )

            if df_residual < 6:
                recommendations.append(
                    "Very few degrees of freedom - results may be unreliable"
                )

            return {
                "estimated_power": estimated_power,
                "power_category": power_category,
                "effect_size": effect_size,
                "effect_category": effect_category,
                "sample_size": n_samples,
                "parameters": n_params,
                "df_residual": df_residual,
                "alpha": alpha,
                "recommendations": recommendations,
            }

        except Exception as e:
            self.logger.error(f"Error estimating statistical power: {e}")
            return {
                "estimated_power": 0.0,
                "power_category": "unknown",
                "error": str(e),
                "recommendations": ["Unable to estimate power - check design matrix"],
            }
