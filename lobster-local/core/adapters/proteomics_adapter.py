"""
Proteomics data adapter with schema enforcement and peptide mapping support.

This module provides the ProteomicsAdapter that handles loading,
validation, and preprocessing of proteomics data including mass spectrometry
and affinity-based proteomics with appropriate schema enforcement.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd

from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.proteomics import ProteomicsSchema

logger = logging.getLogger(__name__)


class ProteomicsAdapter(BaseAdapter):
    """
    Adapter for proteomics data with schema enforcement.
    
    This adapter handles loading and validation of mass spectrometry and
    affinity proteomics data with appropriate schema validation and
    peptide-to-protein mapping support.
    """

    def __init__(
        self, 
        data_type: str = "mass_spectrometry",
        strict_validation: bool = False,
        handle_missing_values: str = "keep"
    ):
        """
        Initialize the proteomics adapter.

        Args:
            data_type: Type of data ('mass_spectrometry' or 'affinity')
            strict_validation: Whether to use strict validation
            handle_missing_values: How to handle missing values ('keep', 'fill_zero', 'drop')
        """
        super().__init__(name="ProteomicsAdapter")
        
        if data_type not in ["mass_spectrometry", "affinity"]:
            raise ValueError(f"Unknown data_type: {data_type}. Must be 'mass_spectrometry' or 'affinity'")
        
        self.data_type = data_type
        self.strict_validation = strict_validation
        self.handle_missing_values = handle_missing_values
        
        # Create validator for this data type
        self.validator = ProteomicsSchema.create_validator(
            schema_type=data_type,
            strict=strict_validation
        )
        
        # Get QC thresholds
        self.qc_thresholds = ProteomicsSchema.get_recommended_qc_thresholds(data_type)

    def from_source(
        self, 
        source: Union[str, Path, pd.DataFrame], 
        **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with proteomics schema.

        Args:
            source: Data source (file path, DataFrame, or AnnData)
            **kwargs: Additional parameters:
                - transpose: Whether to transpose matrix (default: False for proteomics)
                - protein_id_col: Column name for protein identifiers
                - sample_metadata: Additional sample metadata DataFrame
                - protein_metadata: Additional protein metadata DataFrame
                - intensity_columns: List of columns containing intensity data
                - missing_value_indicators: List of values to treat as missing

        Returns:
            anndata.AnnData: Loaded and validated data

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If source file doesn't exist
        """
        self._log_operation("loading", source=str(source), data_type=self.data_type)
        
        try:
            # Handle different source types
            if isinstance(source, anndata.AnnData):
                adata = source.copy()
            elif isinstance(source, pd.DataFrame):
                adata = self._create_anndata_from_dataframe(source, **kwargs)
            elif isinstance(source, (str, Path)):
                adata = self._load_from_file(source, **kwargs)
            else:
                raise TypeError(f"Unsupported source type: {type(source)}")

            # Handle missing values according to strategy
            adata = self._handle_missing_values(adata)
            
            # Add basic metadata
            adata = self._add_basic_metadata(adata, source)
            
            # Apply proteomics-specific preprocessing
            adata = self.preprocess_data(adata, **kwargs)
            
            # Add provenance information
            adata = self.add_provenance(
                adata,
                source_info={
                    "source": str(source),
                    "data_type": self.data_type,
                    "source_type": type(source).__name__
                },
                processing_params=kwargs
            )
            
            self.logger.info(f"Loaded proteomics data: {adata.n_obs} obs × {adata.n_vars} vars")
            return adata
            
        except Exception as e:
            self.logger.error(f"Failed to load proteomics data from {source}: {e}")
            raise

    def _load_from_file(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load data from file with format detection."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        format_type = self.detect_format(path)
        
        if format_type == 'h5ad':
            return self._load_h5ad_data(path)
        elif format_type in ['csv', 'tsv', 'txt']:
            return self._load_csv_proteomics_data(path, **kwargs)
        elif format_type in ['xlsx', 'xls']:
            return self._load_excel_proteomics_data(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format for proteomics: {format_type}")

    def _load_csv_proteomics_data(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load proteomics data from CSV/TSV with proper handling."""
        # Extract parameters
        transpose = kwargs.get('transpose', False)  # Samples as rows by default for proteomics
        protein_id_col = kwargs.get('protein_id_col', None)
        intensity_columns = kwargs.get('intensity_columns', None)
        missing_value_indicators = kwargs.get('missing_value_indicators', ['', 'NA', 'NaN', 'NULL'])
        
        # Load the data
        df = self._load_csv_data(
            path, 
            index_col=0 if protein_id_col is None else protein_id_col,
            na_values=missing_value_indicators,
            **{k: v for k, v in kwargs.items() if k not in [
                'transpose', 'protein_id_col', 'intensity_columns', 'missing_value_indicators'
            ]}
        )
        
        # Handle intensity columns selection
        if intensity_columns is not None:
            # Use only specified intensity columns
            metadata_cols = [col for col in df.columns if col not in intensity_columns]
            intensity_df = df[intensity_columns]
            metadata_df = df[metadata_cols] if metadata_cols else None
        else:
            # Try to auto-detect intensity vs metadata columns
            intensity_df, metadata_df = self._separate_intensity_metadata_columns(df)
        
        # Create protein metadata from non-intensity columns
        var_metadata = None
        if metadata_df is not None and len(metadata_df.columns) > 0:
            var_metadata = metadata_df.copy()
            
            # Standardize common proteomics metadata column names
            var_metadata = self._standardize_proteomics_metadata(var_metadata)
        
        # Create AnnData
        adata = self._create_anndata_from_dataframe(
            intensity_df,
            var_metadata=var_metadata,
            transpose=transpose
        )
        
        return adata

    def _load_excel_proteomics_data(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load proteomics data from Excel file."""
        transpose = kwargs.get('transpose', False)
        sheet_name = kwargs.get('sheet_name', 0)
        
        df = self._load_excel_data(path, sheet_name=sheet_name, index_col=0)
        
        # Separate intensity and metadata columns
        intensity_df, metadata_df = self._separate_intensity_metadata_columns(df)
        
        return self._create_anndata_from_dataframe(
            intensity_df, 
            var_metadata=metadata_df,
            transpose=transpose
        )

    def _separate_intensity_metadata_columns(
        self, 
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Separate intensity data from protein metadata columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            tuple: (intensity_df, metadata_df)
        """
        # Common proteomics metadata column patterns
        metadata_patterns = [
            'protein', 'gene', 'uniprot', 'accession', 'description', 'name',
            'organism', 'species', 'length', 'weight', 'mw', 'peptide', 'coverage',
            'sequence', 'fasta', 'contaminant', 'reverse', 'group', 'leading'
        ]
        
        # Identify metadata columns
        metadata_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in metadata_patterns):
                metadata_cols.append(col)
            elif df[col].dtype == 'object' and not self._is_numeric_string_column(df[col]):
                # Non-numeric string columns are likely metadata
                metadata_cols.append(col)
        
        # Intensity columns are the remaining ones
        intensity_cols = [col for col in df.columns if col not in metadata_cols]
        
        if not intensity_cols:
            raise ValueError("No intensity columns detected in the data")
        
        intensity_df = df[intensity_cols].copy()
        metadata_df = df[metadata_cols].copy() if metadata_cols else None
        
        # Convert intensity columns to numeric, handling missing values
        for col in intensity_cols:
            intensity_df[col] = pd.to_numeric(intensity_df[col], errors='coerce')
        
        return intensity_df, metadata_df

    def _is_numeric_string_column(self, series: pd.Series) -> bool:
        """Check if a string column contains numeric values."""
        if series.dtype != 'object':
            return False
        
        # Try to convert a sample to numeric
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except (ValueError, TypeError):
            return False

    def _standardize_proteomics_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize proteomics metadata column names."""
        
        # Common column name mappings
        column_mappings = {
            # Protein identifiers
            'accession': 'uniprot_id',
            'acc': 'uniprot_id', 
            'protein_accession': 'uniprot_id',
            'uniprot_acc': 'uniprot_id',
            'protein_id': 'protein_id',
            'protein': 'protein_id',
            'id': 'protein_id',
            
            # Gene information
            'gene': 'gene_symbol',
            'gene_name': 'gene_symbol',
            'gene_symbol': 'gene_symbol',
            'symbol': 'gene_symbol',
            
            # Protein names and descriptions
            'description': 'protein_name',
            'protein_description': 'protein_name',
            'protein_name': 'protein_name',
            'name': 'protein_name',
            
            # Organism
            'species': 'organism',
            'organism': 'organism',
            
            # Molecular properties
            'molecular_weight': 'molecular_weight',
            'mw': 'molecular_weight',
            'weight': 'molecular_weight',
            'length': 'sequence_length',
            'sequence_length': 'sequence_length',
            
            # MS-specific
            'peptides': 'n_peptides',
            'unique_peptides': 'n_unique_peptides',
            'coverage': 'sequence_coverage',
            'sequence_coverage': 'sequence_coverage',
            
            # Quality flags
            'contaminant': 'is_contaminant',
            'reverse': 'is_reverse',
            'protein_group': 'protein_group'
        }
        
        # Apply mappings
        renamed_df = metadata_df.copy()
        for old_name, new_name in column_mappings.items():
            matching_cols = [col for col in renamed_df.columns 
                           if old_name.lower() in str(col).lower()]
            if matching_cols:
                # Use the first matching column
                renamed_df = renamed_df.rename(columns={matching_cols[0]: new_name})
        
        return renamed_df

    def _handle_missing_values(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Handle missing values according to the specified strategy."""
        
        if self.handle_missing_values == "keep":
            # Keep missing values as NaN
            pass
        elif self.handle_missing_values == "fill_zero":
            # Replace NaN with zeros
            if hasattr(adata.X, 'isnan'):
                adata.X = np.nan_to_num(adata.X, nan=0.0)
        elif self.handle_missing_values == "drop":
            # Remove observations/variables with too many missing values
            adata = self._drop_high_missing_features(adata)
        
        return adata

    def _drop_high_missing_features(
        self, 
        adata: anndata.AnnData,
        max_missing_obs: float = 0.8,
        max_missing_vars: float = 0.9
    ) -> anndata.AnnData:
        """Drop observations and variables with high missing value rates."""
        
        if not hasattr(adata.X, 'isnan'):
            return adata
        
        original_shape = adata.shape
        
        # Calculate missing rates
        obs_missing_rate = np.isnan(adata.X).sum(axis=1) / adata.n_vars
        vars_missing_rate = np.isnan(adata.X).sum(axis=0) / adata.n_obs
        
        # Filter observations
        obs_keep = obs_missing_rate <= max_missing_obs
        if obs_keep.sum() < adata.n_obs:
            adata = adata[obs_keep, :].copy()
            self.logger.info(f"Removed {(~obs_keep).sum()} observations with >{max_missing_obs*100}% missing values")
        
        # Filter variables
        vars_keep = vars_missing_rate <= max_missing_vars
        if vars_keep.sum() < adata.n_vars:
            adata = adata[:, vars_keep].copy()
            self.logger.info(f"Removed {(~vars_keep).sum()} proteins with >{max_missing_vars*100}% missing values")
        
        self.logger.info(f"Shape after filtering: {adata.shape} (was {original_shape})")
        
        return adata

    def validate(
        self, 
        adata: anndata.AnnData, 
        strict: bool = None
    ) -> ValidationResult:
        """
        Validate AnnData against proteomics schema.

        Args:
            adata: AnnData object to validate
            strict: Override default strict setting

        Returns:
            ValidationResult: Validation results
        """
        if strict is None:
            strict = self.strict_validation
        
        # Use the configured validator
        result = self.validator.validate(adata, strict=strict)
        
        # Add basic structural validation
        basic_result = self._validate_basic_structure(adata)
        result = result.merge(basic_result)
        
        return result

    def get_schema(self) -> Dict[str, Any]:
        """
        Return the expected schema for this modality.

        Returns:
            Dict[str, Any]: Schema definition
        """
        if self.data_type == "mass_spectrometry":
            return ProteomicsSchema.get_mass_spectrometry_schema()
        else:
            return ProteomicsSchema.get_affinity_proteomics_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ['csv', 'tsv', 'txt', 'xlsx', 'xls', 'h5ad']

    def preprocess_data(
        self, 
        adata: anndata.AnnData, 
        **kwargs
    ) -> anndata.AnnData:
        """
        Apply proteomics-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Apply base preprocessing
        adata = super().preprocess_data(adata, **kwargs)
        
        # Add proteomics-specific metadata
        adata = self._add_proteomics_metadata(adata)
        
        # Flag contaminants and reverse hits if not already done
        adata = self._flag_contaminants_and_reverse(adata)
        
        return adata

    def _add_proteomics_metadata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Add proteomics-specific metadata."""
        
        # Calculate per-sample metrics
        if 'n_proteins' not in adata.obs.columns:
            # Count detected proteins (non-NaN values)
            if hasattr(adata.X, 'isnan'):
                adata.obs['n_proteins'] = (~np.isnan(adata.X)).sum(axis=1)
            else:
                adata.obs['n_proteins'] = (adata.X > 0).sum(axis=1)
        
        if 'total_intensity' not in adata.obs.columns:
            # Sum of all intensities (excluding NaN)
            adata.obs['total_intensity'] = np.nansum(adata.X, axis=1)
        
        if 'median_intensity' not in adata.obs.columns:
            # Median intensity per sample
            adata.obs['median_intensity'] = np.nanmedian(adata.X, axis=1)
        
        if 'pct_missing' not in adata.obs.columns and hasattr(adata.X, 'isnan'):
            # Percentage of missing values per sample
            adata.obs['pct_missing'] = np.isnan(adata.X).sum(axis=1) / adata.n_vars * 100
        
        # Calculate per-protein metrics
        if 'n_samples' not in adata.var.columns:
            # Count samples with detection
            if hasattr(adata.X, 'isnan'):
                adata.var['n_samples'] = (~np.isnan(adata.X)).sum(axis=0)
            else:
                adata.var['n_samples'] = (adata.X > 0).sum(axis=0)
        
        if 'mean_intensity' not in adata.var.columns:
            # Mean intensity per protein
            adata.var['mean_intensity'] = np.nanmean(adata.X, axis=0)
        
        if 'median_intensity' not in adata.var.columns:
            # Median intensity per protein
            adata.var['median_intensity'] = np.nanmedian(adata.X, axis=0)
        
        if 'pct_missing' not in adata.var.columns and hasattr(adata.X, 'isnan'):
            # Percentage of missing values per protein
            adata.var['pct_missing'] = np.isnan(adata.X).sum(axis=0) / adata.n_obs * 100
        
        # Calculate coefficient of variation
        if 'cv' not in adata.var.columns:
            means = np.nanmean(adata.X, axis=0)
            stds = np.nanstd(adata.X, axis=0)
            adata.var['cv'] = (stds / means * 100).fillna(0)  # CV as percentage
        
        return adata

    def _flag_contaminants_and_reverse(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Flag contaminant proteins and reverse database hits."""
        
        # Check if we have protein identifiers for pattern matching
        protein_ids = None
        if 'protein_id' in adata.var.columns:
            protein_ids = adata.var['protein_id']
        elif 'uniprot_id' in adata.var.columns:
            protein_ids = adata.var['uniprot_id']
        elif adata.var_names is not None:
            protein_ids = adata.var_names
        
        if protein_ids is not None:
            # Flag contaminants (common patterns)
            if 'is_contaminant' not in adata.var.columns:
                contaminant_patterns = ['CON_', 'CONT_', 'contaminant', 'KERATIN', 'TRYP_']
                is_contaminant = protein_ids.str.contains('|'.join(contaminant_patterns), case=False, na=False)
                adata.var['is_contaminant'] = is_contaminant
            
            # Flag reverse database hits
            if 'is_reverse' not in adata.var.columns:
                reverse_patterns = ['REV_', 'REVERSE_', 'rev_']
                is_reverse = protein_ids.str.contains('|'.join(reverse_patterns), case=False, na=False)
                adata.var['is_reverse'] = is_reverse
        
        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate proteomics-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        metrics = super().get_quality_metrics(adata)
        
        # Add proteomics-specific metrics
        if hasattr(adata.X, 'isnan'):
            total_values = adata.X.size
            missing_values = np.isnan(adata.X).sum()
            metrics['missing_value_percentage'] = float((missing_values / total_values) * 100)
        
        if 'n_proteins' in adata.obs.columns:
            metrics['mean_proteins_per_sample'] = float(adata.obs['n_proteins'].mean())
            metrics['median_proteins_per_sample'] = float(adata.obs['n_proteins'].median())
        
        if 'n_samples' in adata.var.columns:
            metrics['mean_samples_per_protein'] = float(adata.var['n_samples'].mean())
        
        if 'cv' in adata.var.columns:
            metrics['median_cv'] = float(adata.var['cv'].median())
            metrics['high_cv_proteins'] = int((adata.var['cv'] > 50).sum())
        
        # Contaminant and reverse metrics
        if 'is_contaminant' in adata.var.columns:
            metrics['contaminant_proteins'] = int(adata.var['is_contaminant'].sum())
            metrics['contaminant_percentage'] = float((adata.var['is_contaminant'].sum() / len(adata.var)) * 100)
        
        if 'is_reverse' in adata.var.columns:
            metrics['reverse_hits'] = int(adata.var['is_reverse'].sum())
        
        return metrics

    def add_peptide_mapping(
        self, 
        adata: anndata.AnnData,
        peptide_data: Union[pd.DataFrame, str, Path]
    ) -> anndata.AnnData:
        """
        Add peptide-to-protein mapping information to AnnData.

        Args:
            adata: AnnData object to annotate
            peptide_data: Peptide mapping data (DataFrame or file path)

        Returns:
            anndata.AnnData: AnnData with peptide mapping information
        """
        try:
            # Load peptide data if it's a file
            if isinstance(peptide_data, (str, Path)):
                peptide_df = pd.read_csv(peptide_data)
            else:
                peptide_df = peptide_data.copy()
            
            # Validate peptide mapping schema
            peptide_schema = ProteomicsSchema.get_peptide_to_protein_mapping_schema()
            
            # Group peptides by protein
            if 'protein_id' in peptide_df.columns:
                peptide_groups = peptide_df.groupby('protein_id').agg({
                    'peptide_sequence': ['count', 'nunique', list],
                    'is_unique': lambda x: x.sum() if 'is_unique' in peptide_df.columns else None,
                    'sequence_coverage': 'first' if 'sequence_coverage' in peptide_df.columns else None
                }).reset_index()
                
                # Flatten column names
                peptide_groups.columns = ['protein_id', 'n_peptides', 'n_unique_peptides', 'peptide_list', 'n_unique_total', 'sequence_coverage']
                
                # Add peptide information to var metadata
                for idx, row in peptide_groups.iterrows():
                    protein_id = row['protein_id']
                    
                    # Find matching protein in adata
                    if 'protein_id' in adata.var.columns:
                        mask = adata.var['protein_id'] == protein_id
                    elif protein_id in adata.var_names:
                        mask = adata.var_names == protein_id
                    else:
                        continue
                    
                    if mask.any():
                        # Update peptide counts
                        if 'n_peptides' not in adata.var.columns:
                            adata.var['n_peptides'] = 0
                        if 'n_unique_peptides' not in adata.var.columns:
                            adata.var['n_unique_peptides'] = 0
                        
                        adata.var.loc[mask, 'n_peptides'] = row['n_peptides']
                        adata.var.loc[mask, 'n_unique_peptides'] = row['n_unique_peptides']
                        
                        if row['sequence_coverage'] is not None:
                            if 'sequence_coverage' not in adata.var.columns:
                                adata.var['sequence_coverage'] = np.nan
                            adata.var.loc[mask, 'sequence_coverage'] = row['sequence_coverage']
            
            # Store full peptide mapping in uns
            adata.uns['peptide_to_protein'] = {
                'mapping_data': peptide_df.to_dict('records'),
                'n_peptides': len(peptide_df),
                'n_proteins': peptide_df['protein_id'].nunique() if 'protein_id' in peptide_df.columns else 0,
                'schema': peptide_schema
            }
            
            self.logger.info(f"Added peptide mapping: {len(peptide_df)} peptides for {adata.uns['peptide_to_protein']['n_proteins']} proteins")
            
            return adata
            
        except Exception as e:
            self.logger.error(f"Failed to add peptide mapping: {e}")
            raise

    def detect_data_type(self, adata: anndata.AnnData) -> str:
        """
        Detect whether data is mass spectrometry or affinity proteomics.

        Args:
            adata: AnnData object to analyze

        Returns:
            str: Detected data type
        """
        # Heuristics for detecting proteomics data type
        
        # Check for MS-specific metadata
        ms_indicators = ['n_peptides', 'sequence_coverage', 'is_contaminant', 'protein_group']
        ms_score = sum(1 for indicator in ms_indicators if indicator in adata.var.columns)
        
        # Check for affinity-specific metadata
        affinity_indicators = ['antibody_id', 'antibody_clone', 'array_type']
        affinity_score = sum(1 for indicator in affinity_indicators if indicator in adata.var.columns)
        
        # Check data characteristics
        n_vars = adata.n_vars
        missing_rate = 0
        
        if hasattr(adata.X, 'isnan'):
            missing_rate = np.isnan(adata.X).sum() / adata.X.size
        
        # MS data typically has:
        # - More proteins (hundreds to thousands)
        # - Higher missing value rate
        # - MS-specific metadata
        if ms_score > affinity_score and n_vars > 200:
            return "mass_spectrometry"
        
        # Affinity data typically has:
        # - Fewer proteins (tens to hundreds) 
        # - Lower missing value rate
        # - Affinity-specific metadata
        elif affinity_score > ms_score or (n_vars < 200 and missing_rate < 0.3):
            return "affinity"
        
        # If unclear, return the current setting
        return self.data_type
