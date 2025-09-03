"""
Proteomics schema definitions for mass spectrometry and affinity proteomics data.

This module defines the expected structure and metadata for proteomics
datasets including MS-based and affinity-based proteomics with appropriate
validation rules and peptide-to-protein mapping support.
"""

from typing import Any, Dict, List, Optional

from .validation import FlexibleValidator


class ProteomicsSchema:
    """
    Schema definitions for proteomics data modalities.
    
    This class provides schema definitions for mass spectrometry and
    affinity proteomics data with appropriate metadata requirements
    and validation rules.
    """

    @staticmethod
    def get_mass_spectrometry_schema() -> Dict[str, Any]:
        """
        Get schema for mass spectrometry proteomics data.

        Returns:
            Dict[str, Any]: Mass spectrometry proteomics schema definition
        """
        return {
            "modality": "mass_spectrometry_proteomics",
            "description": "Mass spectrometry proteomics data schema",
            
            "obs": {
                "required": [],  # Made flexible - no strictly required columns
                "optional": [
                    "sample_id",         # Unique sample identifier
                    "condition",         # Experimental condition
                    "treatment",         # Treatment information
                    "batch",             # MS run batch
                    "replicate",         # Biological replicate
                    "instrument",        # MS instrument used
                    "acquisition_method", # DDA, DIA, SRM, etc.
                    "tissue",            # Tissue type
                    "organism",          # Organism
                    "n_proteins",        # Number of proteins detected
                    "total_intensity",   # Total protein intensity
                    "missing_values",    # Number of missing values
                    "pct_missing",       # Percentage of missing values
                    "median_intensity",  # Median protein intensity
                    "median_cv",         # Median coefficient of variation
                ],
                "types": {
                    "sample_id": "string",
                    "condition": "categorical",
                    "treatment": "categorical",
                    "batch": "string",
                    "replicate": "string",
                    "instrument": "categorical",
                    "acquisition_method": "categorical",
                    "tissue": "categorical",
                    "organism": "string",
                    "n_proteins": "numeric",
                    "total_intensity": "numeric",
                    "missing_values": "numeric",
                    "pct_missing": "numeric",
                    "median_intensity": "numeric",
                    "median_cv": "numeric"
                }
            },
            
            "var": {
                "required": [],  # Made flexible
                "optional": [
                    "protein_id",        # Primary protein identifier
                    "uniprot_id",        # UniProt accession
                    "gene_symbol",       # Gene symbol
                    "gene_name",         # Full gene name
                    "protein_name",      # Full protein name
                    "organism",          # Species
                    "sequence_length",   # Protein sequence length
                    "molecular_weight",  # Molecular weight (Da)
                    "n_peptides",        # Number of peptides identified
                    "n_unique_peptides", # Number of unique peptides
                    "sequence_coverage", # Percentage sequence coverage
                    "n_samples",         # Number of samples with detection
                    "mean_intensity",    # Mean intensity across samples
                    "median_intensity",  # Median intensity across samples
                    "total_intensity",   # Total intensity across samples
                    "cv",                # Coefficient of variation
                    "missing_values",    # Number of missing values
                    "pct_missing",       # Percentage missing across samples
                    "is_contaminant",    # Contaminant protein flag
                    "is_reverse",        # Reverse database hit flag
                    "protein_group",     # Protein group identifier
                    "leading_protein",   # Leading protein in group
                ],
                "types": {
                    "protein_id": "string",
                    "uniprot_id": "string",
                    "gene_symbol": "string",
                    "gene_name": "string",
                    "protein_name": "string",
                    "organism": "string",
                    "sequence_length": "numeric",
                    "molecular_weight": "numeric",
                    "n_peptides": "numeric",
                    "n_unique_peptides": "numeric",
                    "sequence_coverage": "numeric",
                    "n_samples": "numeric",
                    "mean_intensity": "numeric",
                    "median_intensity": "numeric",
                    "total_intensity": "numeric",
                    "cv": "numeric",
                    "missing_values": "numeric",
                    "pct_missing": "numeric",
                    "is_contaminant": "boolean",
                    "is_reverse": "boolean",
                    "protein_group": "categorical",
                    "leading_protein": "boolean"
                }
            },
            
            "layers": {
                "required": [],  # X matrix contains main intensities
                "optional": [
                    "raw_intensity",     # Raw intensities
                    "normalized",        # Normalized intensities
                    "log2_intensity",    # Log2-transformed intensities
                    "imputed",           # Imputed values
                    "median_normalized", # Median normalized
                    "quantile_normalized", # Quantile normalized
                    "vsn_normalized",    # Variance stabilizing normalization
                ]
            },
            
            "obsm": {
                "required": [],
                "optional": [
                    "X_pca",             # PCA coordinates
                    "X_tsne",            # t-SNE embedding
                    "X_umap"             # UMAP embedding
                ]
            },
            
            "uns": {
                "required": [],
                "optional": [
                    "ms_params",         # MS acquisition parameters
                    "search_params",     # Database search parameters
                    "normalization",     # Normalization parameters
                    "imputation",        # Imputation parameters
                    "peptide_to_protein", # Peptide mapping information
                    "raw_data_uri",      # Links to raw mzML/mzTab files
                    "differential_expression", # DE analysis results
                    "pathway_analysis",  # Pathway enrichment results
                    "provenance"         # Provenance tracking
                ]
            }
        }

    @staticmethod
    def get_affinity_proteomics_schema() -> Dict[str, Any]:
        """
        Get schema for affinity-based proteomics data (e.g., antibody arrays).

        Returns:
            Dict[str, Any]: Affinity proteomics schema definition
        """
        return {
            "modality": "affinity_proteomics",
            "description": "Affinity-based proteomics data schema",
            
            "obs": {
                "required": [],
                "optional": [
                    "sample_id",         # Unique sample identifier
                    "condition",         # Experimental condition
                    "treatment",         # Treatment information
                    "batch",             # Array batch
                    "replicate",         # Biological replicate
                    "array_type",        # Type of protein array
                    "tissue",            # Tissue type
                    "organism",          # Organism
                    "n_proteins",        # Number of proteins detected
                    "total_signal",      # Total signal intensity
                    "median_signal",     # Median signal intensity
                    "background_signal", # Background signal level
                ],
                "types": {
                    "sample_id": "string",
                    "condition": "categorical",
                    "treatment": "categorical",
                    "batch": "string",
                    "replicate": "string",
                    "array_type": "categorical",
                    "tissue": "categorical",
                    "organism": "string",
                    "n_proteins": "numeric",
                    "total_signal": "numeric",
                    "median_signal": "numeric",
                    "background_signal": "numeric"
                }
            },
            
            "var": {
                "required": [],
                "optional": [
                    "protein_id",        # Primary protein identifier
                    "uniprot_id",        # UniProt accession
                    "gene_symbol",       # Gene symbol
                    "protein_name",      # Full protein name
                    "antibody_id",       # Antibody identifier
                    "antibody_clone",    # Antibody clone information
                    "organism",          # Species
                    "mean_signal",       # Mean signal across samples
                    "median_signal",     # Median signal across samples
                    "cv",                # Coefficient of variation
                    "n_samples",         # Number of samples with detection
                ],
                "types": {
                    "protein_id": "string",
                    "uniprot_id": "string",
                    "gene_symbol": "string",
                    "protein_name": "string",
                    "antibody_id": "string",
                    "antibody_clone": "string",
                    "organism": "string",
                    "mean_signal": "numeric",
                    "median_signal": "numeric",
                    "cv": "numeric",
                    "n_samples": "numeric"
                }
            },
            
            "layers": {
                "required": [],
                "optional": [
                    "raw_signal",        # Raw signal intensities
                    "normalized",        # Normalized signals
                    "background_corrected", # Background corrected
                    "log2_signal",       # Log2-transformed signals
                ]
            },
            
            "obsm": {
                "required": [],
                "optional": [
                    "X_pca",             # PCA coordinates
                    "X_tsne",            # t-SNE embedding
                    "X_umap"             # UMAP embedding
                ]
            },
            
            "uns": {
                "required": [],
                "optional": [
                    "array_params",      # Array parameters
                    "normalization",     # Normalization parameters
                    "antibody_info",     # Antibody information
                    "differential_expression", # DE analysis results
                    "pathway_analysis",  # Pathway enrichment results
                    "provenance"         # Provenance tracking
                ]
            }
        }

    @staticmethod
    def create_validator(
        schema_type: str = "mass_spectrometry",
        strict: bool = False,
        ignore_warnings: Optional[List[str]] = None
    ) -> FlexibleValidator:
        """
        Create a validator for proteomics data.

        Args:
            schema_type: Type of schema ('mass_spectrometry' or 'affinity')
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator

        Raises:
            ValueError: If schema_type is not recognized
        """
        if schema_type == "mass_spectrometry":
            schema = ProteomicsSchema.get_mass_spectrometry_schema()
        elif schema_type == "affinity":
            schema = ProteomicsSchema.get_affinity_proteomics_schema()
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        ignore_set = set(ignore_warnings) if ignore_warnings else set()
        
        # Add default ignored warnings for proteomics
        ignore_set.update([
            "Unexpected obs columns",
            "Unexpected var columns",
            "missing values",
            "Very sparse data"  # Proteomics often has many missing values
        ])

        validator = FlexibleValidator(
            schema=schema,
            name=f"ProteomicsValidator_{schema_type}",
            ignore_warnings=ignore_set
        )

        # Add proteomics-specific validation rules
        validator.add_custom_rule("check_protein_ids", _validate_protein_ids)
        validator.add_custom_rule("check_missing_values", _validate_missing_values)
        validator.add_custom_rule("check_intensity_data", _validate_intensity_data)
        
        if schema_type == "mass_spectrometry":
            validator.add_custom_rule("check_ms_metrics", _validate_ms_metrics)
        elif schema_type == "affinity":
            validator.add_custom_rule("check_affinity_metrics", _validate_affinity_metrics)

        return validator

    @staticmethod
    def get_recommended_qc_thresholds(schema_type: str = "mass_spectrometry") -> Dict[str, Any]:
        """
        Get recommended quality control thresholds.

        Args:
            schema_type: Type of schema ('mass_spectrometry' or 'affinity')

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        if schema_type == "mass_spectrometry":
            return {
                "min_proteins_per_sample": 100,
                "max_missing_per_sample": 0.7,  # 70% missing values threshold
                "min_peptides_per_protein": 1,
                "min_unique_peptides_per_protein": 1,
                "min_sequence_coverage": 5.0,   # 5% minimum coverage
                "max_cv_threshold": 50.0,       # 50% CV threshold
                "min_samples_per_protein": 2,
                "contaminant_threshold": 0.1    # 10% contaminants max
            }
        elif schema_type == "affinity":
            return {
                "min_proteins_per_sample": 50,
                "max_missing_per_sample": 0.5,  # 50% missing values threshold
                "min_samples_per_protein": 3,
                "max_cv_threshold": 30.0,       # 30% CV threshold
                "min_signal_to_background": 2.0 # 2x background signal
            }
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

    @staticmethod
    def get_peptide_to_protein_mapping_schema() -> Dict[str, Any]:
        """
        Get schema for peptide-to-protein mapping data.

        Returns:
            Dict[str, Any]: Peptide mapping schema
        """
        return {
            "peptide_sequence": "string",     # Peptide amino acid sequence
            "modified_sequence": "string",    # Modified sequence with modifications
            "protein_id": "string",           # Protein identifier
            "uniprot_id": "string",          # UniProt accession
            "gene_symbol": "string",         # Gene symbol
            "start_position": "numeric",     # Start position in protein
            "end_position": "numeric",       # End position in protein
            "is_unique": "boolean",          # Unique to single protein
            "n_proteins": "numeric",         # Number of proteins containing peptide
            "charge": "numeric",             # Precursor charge state
            "mass": "numeric",               # Peptide mass
            "rt": "numeric",                 # Retention time
            "score": "numeric",              # Identification score
            "q_value": "numeric"             # Q-value (FDR)
        }


def _validate_protein_ids(adata) -> 'ValidationResult':
    """Validate protein identifier format and uniqueness."""
    from lobster.core.interfaces.validator import ValidationResult
    
    result = ValidationResult()
    
    # Check for protein IDs in var
    protein_id_cols = ['protein_id', 'uniprot_id', 'gene_symbol']
    
    for col in protein_id_cols:
        if col in adata.var.columns:
            ids = adata.var[col]
            
            # Check for duplicates
            duplicates = ids.duplicated().sum()
            if duplicates > 0:
                result.add_warning(f"{duplicates} duplicate {col} found")
            
            # Check for missing IDs
            missing = ids.isna().sum()
            if missing > 0:
                result.add_warning(f"{missing} missing {col}")
    
    return result


def _validate_missing_values(adata) -> 'ValidationResult':
    """Validate missing values in proteomics data."""
    from lobster.core.interfaces.validator import ValidationResult
    import numpy as np
    
    result = ValidationResult()
    
    # Check for missing values in X matrix
    if hasattr(adata.X, 'isnan'):
        nan_count = np.isnan(adata.X).sum()
        total_values = adata.X.size
        nan_percentage = (nan_count / total_values) * 100
        
        if nan_percentage > 50:
            result.add_warning(f"High proportion of missing values: {nan_percentage:.1f}%")
        else:
            result.add_info(f"Missing values: {nan_percentage:.1f}%")
    
    # Check for samples with too many missing values
    if hasattr(adata.X, 'isnan'):
        sample_missing = np.isnan(adata.X).sum(axis=1) / adata.n_vars
        high_missing_samples = (sample_missing > 0.7).sum()
        if high_missing_samples > 0:
            result.add_warning(f"{high_missing_samples} samples with >70% missing values")
    
    # Check for proteins with too many missing values
    if hasattr(adata.X, 'isnan'):
        protein_missing = np.isnan(adata.X).sum(axis=0) / adata.n_obs
        high_missing_proteins = (protein_missing > 0.8).sum()
        if high_missing_proteins > 0:
            result.add_warning(f"{high_missing_proteins} proteins with >80% missing values")
    
    return result


def _validate_intensity_data(adata) -> 'ValidationResult':
    """Validate intensity data characteristics."""
    from lobster.core.interfaces.validator import ValidationResult
    import numpy as np
    
    result = ValidationResult()
    
    # Check for negative values (unusual in proteomics)
    if hasattr(adata.X, 'min'):
        min_val = np.nanmin(adata.X)
        if min_val < 0:
            result.add_warning(f"Negative values found in intensity matrix (min: {min_val})")
    
    # Check for very large values (potential outliers)
    if hasattr(adata.X, 'max'):
        max_val = np.nanmax(adata.X)
        if max_val > 1e9:  # Very large intensity values
            result.add_warning(f"Very large intensity values found (max: {max_val:.2e})")
    
    # Check dynamic range
    if hasattr(adata.X, 'min') and hasattr(adata.X, 'max'):
        min_val = np.nanmin(adata.X[adata.X > 0])  # Exclude zeros/NaN
        max_val = np.nanmax(adata.X)
        if min_val > 0 and max_val > 0:
            dynamic_range = np.log10(max_val / min_val)
            if dynamic_range > 6:  # >6 orders of magnitude
                result.add_info(f"Large dynamic range: {dynamic_range:.1f} orders of magnitude")
    
    return result


def _validate_ms_metrics(adata) -> 'ValidationResult':
    """Validate mass spectrometry specific metrics."""
    from lobster.core.interfaces.validator import ValidationResult
    
    result = ValidationResult()
    
    # Check peptide counts if available
    if 'n_peptides' in adata.var.columns:
        n_peptides = adata.var['n_peptides']
        single_peptide = (n_peptides == 1).sum()
        if single_peptide > 0:
            result.add_warning(f"{single_peptide} proteins identified with only 1 peptide")
    
    # Check sequence coverage if available
    if 'sequence_coverage' in adata.var.columns:
        coverage = adata.var['sequence_coverage']
        low_coverage = (coverage < 10).sum()
        if low_coverage > 0:
            result.add_warning(f"{low_coverage} proteins with <10% sequence coverage")
    
    # Check for contaminants
    if 'is_contaminant' in adata.var.columns:
        contaminants = adata.var['is_contaminant'].sum()
        contaminant_pct = (contaminants / len(adata.var)) * 100
        if contaminant_pct > 10:
            result.add_warning(f"High proportion of contaminants: {contaminant_pct:.1f}%")
    
    return result


def _validate_affinity_metrics(adata) -> 'ValidationResult':
    """Validate affinity proteomics specific metrics."""
    from lobster.core.interfaces.validator import ValidationResult
    
    result = ValidationResult()
    
    # Check signal-to-background ratio if available
    if 'background_signal' in adata.obs.columns:
        bg_signals = adata.obs['background_signal']
        high_background = (bg_signals > 1000).sum()  # Arbitrary threshold
        if high_background > 0:
            result.add_warning(f"{high_background} samples with high background signal")
    
    # Check antibody information
    if 'antibody_id' in adata.var.columns:
        antibody_ids = adata.var['antibody_id']
        missing_antibody = antibody_ids.isna().sum()
        if missing_antibody > 0:
            result.add_warning(f"{missing_antibody} proteins without antibody information")
    
    return result
