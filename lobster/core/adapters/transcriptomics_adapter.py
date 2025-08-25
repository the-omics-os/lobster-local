"""
Transcriptomics data adapter with schema enforcement.

This module provides the TranscriptomicsAdapter that handles loading,
validation, and preprocessing of single-cell and bulk RNA-seq data
with appropriate schema enforcement.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

from lobster.core.adapters.base import BaseAdapter
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.transcriptomics import TranscriptomicsSchema

logger = logging.getLogger(__name__)


class TranscriptomicsAdapter(BaseAdapter):
    """
    Adapter for transcriptomics data with schema enforcement.
    
    This adapter handles loading and validation of single-cell and bulk
    RNA-seq data with appropriate schema validation and quality control.
    """

    def __init__(
        self, 
        data_type: str = "single_cell",
        strict_validation: bool = False
    ):
        """
        Initialize the transcriptomics adapter.

        Args:
            data_type: Type of data ('single_cell' or 'bulk')
            strict_validation: Whether to use strict validation
        """
        super().__init__(name="TranscriptomicsAdapter")
        
        if data_type not in ["single_cell", "bulk"]:
            raise ValueError(f"Unknown data_type: {data_type}. Must be 'single_cell' or 'bulk'")
        
        self.data_type = data_type
        self.strict_validation = strict_validation
        
        # Create validator for this data type
        self.validator = TranscriptomicsSchema.create_validator(
            schema_type=data_type,
            strict=strict_validation
        )
        
        # Get QC thresholds
        self.qc_thresholds = TranscriptomicsSchema.get_recommended_qc_thresholds(data_type)

    def from_source(
        self, 
        source: Union[str, Path, pd.DataFrame], 
        **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with transcriptomics schema.

        Args:
            source: Data source (file path, DataFrame, or AnnData)
            **kwargs: Additional parameters:
                - transpose: Whether to transpose matrix (default: True for files)
                - gene_symbols_col: Column name for gene symbols
                - sample_metadata: Additional sample metadata DataFrame
                - gene_metadata: Additional gene metadata DataFrame
                - var_names: Column to use as variable names
                - first_column_names: Use first column as obs names
                - Additional metadata fields will be stored in uns

        Returns:
            anndata.AnnData: Loaded and validated data

        Raises:
            ValueError: If source data is invalid
            FileNotFoundError: If source file doesn't exist
        """
        self._log_operation("loading", source=str(source), data_type=self.data_type)
        
        try:
            # Extract metadata fields that should be stored in uns
            metadata_fields = {}
            dataframe_params = {}
            
            # Known parameters for _create_anndata_from_dataframe
            valid_dataframe_params = {'transpose', 'obs_metadata', 'var_metadata'}
            
            # Known parameters for file loading
            file_params = {'transpose', 'gene_symbols_col', 'sample_metadata', 
                          'gene_metadata', 'var_names', 'first_column_names'}
            
            # Separate metadata from processing parameters
            for key, value in kwargs.items():
                if key in valid_dataframe_params:
                    dataframe_params[key] = value
                elif key not in file_params:
                    # Store as metadata
                    metadata_fields[key] = value
            
            # Handle different source types
            if isinstance(source, anndata.AnnData):
                adata = source.copy()
            elif isinstance(source, pd.DataFrame):
                adata = self._create_anndata_from_dataframe(
                    df = source, 
                    obs_metadata=dataframe_params.get('obs_metadata'),
                    var_metadata=dataframe_params.get('var_metadata')
                    )
            elif isinstance(source, (str, Path)):
                adata = self._load_from_file(source, **kwargs)
            else:
                raise TypeError(f"Unsupported source type: {type(source)}")
            
            # Store metadata fields in uns
            if metadata_fields:
                for key, value in metadata_fields.items():
                    adata.uns[key] = value
            
            # Apply transcriptomics-specific preprocessing
            adata = self.preprocess_data(adata, **kwargs)
            
            # # Add provenance information
            # adata = self.add_provenance(
            #     adata,
            #     source_info={
            #         # "source": str(source),
            #         "data_type": self.data_type,
            #         "source_type": type(source).__name__
            #     },
            #     processing_params=kwargs
            # )
            
            self.logger.info(f"Loaded transcriptomics data: {adata.n_obs} obs × {adata.n_vars} vars")
            return adata
            
        except Exception as e:
            self.logger.error(f"Failed to load transcriptomics data from {source}: {e}")
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
            return self._load_csv_transcriptomics_data(path, **kwargs)
        elif format_type in ['xlsx', 'xls']:
            return self._load_excel_transcriptomics_data(path, **kwargs)
        elif format_type == 'h5':
            return self._load_h5_transcriptomics_data(path, **kwargs)
        elif format_type == 'mtx':
            return self._load_mtx_data(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")

    def _load_csv_transcriptomics_data(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load transcriptomics data from CSV/TSV with proper handling."""
        # Extract parameters
        transpose = kwargs.get('transpose', True)  # Genes as rows by default
        first_column_names = kwargs.get('first_column_names', True)
        gene_symbols_col = kwargs.get('gene_symbols_col', None)
        
        # Filter kwargs for _load_csv_data to avoid passing metadata fields
        csv_params = {k: v for k, v in kwargs.items() 
                     if k not in ['transpose', 'first_column_names', 'gene_symbols_col', 
                                  'dataset_id', 'dataset_type', 'source_metadata', 
                                  'processing_date', 'download_source', 'processing_method']}
        
        # Load the data
        df = self._load_csv_data(
            path, 
            index_col=0 if first_column_names else None,
            **csv_params
        )
        
        # Handle gene symbols if specified
        var_metadata = None
        if gene_symbols_col and gene_symbols_col in df.columns:
            var_metadata = df[[gene_symbols_col]].copy()
            var_metadata = var_metadata.rename(columns={gene_symbols_col: 'gene_symbol'})
            df = df.drop(columns=[gene_symbols_col])
        
        # Create AnnData
        adata = self._create_anndata_from_dataframe(
            df,
            var_metadata=var_metadata,
            transpose=transpose
        )
        
        return adata

    def _load_excel_transcriptomics_data(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load transcriptomics data from Excel file."""
        transpose = kwargs.get('transpose', True)
        sheet_name = kwargs.get('sheet_name', 0)
        
        # Filter kwargs to avoid passing unexpected parameters
        excel_params = {k: v for k, v in kwargs.items() 
                       if k in ['sheet_name'] and k not in ['transpose']}
        
        df = self._load_excel_data(path, sheet_name=sheet_name, index_col=0, **excel_params)
        
        return self._create_anndata_from_dataframe(df, transpose=transpose)

    def _load_h5_transcriptomics_data(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load transcriptomics data from HDF5 file."""
        try:
            # Try to load as scanpy H5 format first
            adata = sc.read_h5ad(path)
            return adata
        except:
            # Fallback to pandas HDF5
            key = kwargs.get('key', 'expression_data')
            df = pd.read_hdf(path, key=key)
            transpose = kwargs.get('transpose', True)
            return self._create_anndata_from_dataframe(df, transpose=transpose)

    def _load_mtx_data(
        self, 
        path: Union[str, Path], 
        **kwargs
    ) -> anndata.AnnData:
        """Load data from Matrix Market format (10X Genomics style)."""
        path = Path(path)
        
        # For MTX format, expect directory with matrix.mtx, features.tsv, barcodes.tsv
        if path.is_file() and path.suffix == '.mtx':
            # Single MTX file - need additional files
            base_path = path.parent
            matrix_path = path
            features_path = base_path / 'features.tsv.gz'
            barcodes_path = base_path / 'barcodes.tsv.gz'
            
            # Try alternative names
            if not features_path.exists():
                features_path = base_path / 'genes.tsv'
                if not features_path.exists():
                    features_path = base_path / 'features.tsv'
            
            if not barcodes_path.exists():
                barcodes_path = base_path / 'barcodes.tsv'
        
        elif path.is_dir():
            # Directory containing 10X files
            base_path = path
            
            # Find matrix file
            matrix_candidates = ['matrix.mtx.gz', 'matrix.mtx']
            matrix_path = None
            for candidate in matrix_candidates:
                candidate_path = base_path / candidate
                if candidate_path.exists():
                    matrix_path = candidate_path
                    break
            
            if matrix_path is None:
                raise FileNotFoundError(f"No matrix file found in {base_path}")
            
            # Find features file
            features_candidates = ['features.tsv.gz', 'genes.tsv.gz', 'features.tsv', 'genes.tsv']
            features_path = None
            for candidate in features_candidates:
                candidate_path = base_path / candidate
                if candidate_path.exists():
                    features_path = candidate_path
                    break
            
            # Find barcodes file
            barcodes_candidates = ['barcodes.tsv.gz', 'barcodes.tsv']
            barcodes_path = None
            for candidate in barcodes_candidates:
                candidate_path = base_path / candidate
                if candidate_path.exists():
                    barcodes_path = candidate_path
                    break
        
        else:
            raise ValueError(f"Invalid MTX path: {path}")
        
        # Load using scanpy
        try:
            adata = sc.read_10x_mtx(
                base_path if path.is_dir() else path.parent,
                var_names='gene_symbols' if features_path and features_path.exists() else 'gene_ids',
                cache=True
            )
            
            # Make variable names unique
            adata.var_names_unique()
            
            return adata
            
        except Exception as e:
            raise ValueError(f"Failed to load 10X MTX data: {e}")

    def validate(
        self, 
        adata: anndata.AnnData, 
        strict: bool = None
    ) -> ValidationResult:
        """
        Validate AnnData against transcriptomics schema.

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
        if self.data_type == "single_cell":
            return TranscriptomicsSchema.get_single_cell_schema()
        else:
            return TranscriptomicsSchema.get_bulk_rna_seq_schema()

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions
        """
        return ['csv', 'tsv', 'txt', 'xlsx', 'xls', 'h5ad', 'h5', 'mtx']

    def preprocess_data(
        self, 
        adata: anndata.AnnData, 
        **kwargs
    ) -> anndata.AnnData:
        """
        Apply transcriptomics-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Apply base preprocessing
        adata = super().preprocess_data(adata, **kwargs)
        
        # Add transcriptomics-specific metadata
        adata = self._add_transcriptomics_metadata(adata)
        
        # Detect and flag special gene sets
        adata = self._flag_special_genes(adata)
        
        return adata

    def _add_transcriptomics_metadata(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Add transcriptomics-specific metadata."""
        
        # Calculate per-cell metrics for single-cell data
        if self.data_type == "single_cell":
            # Add cell metrics if not present
            if 'n_genes_by_counts' not in adata.obs.columns:
                adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
            
            if 'total_counts' not in adata.obs.columns:
                adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        
        # Calculate per-gene metrics
        if 'n_cells_by_counts' not in adata.var.columns:
            adata.var['n_cells_by_counts'] = (adata.X > 0).sum(axis=0)
        
        if 'mean_counts' not in adata.var.columns:
            adata.var['mean_counts'] = np.array(adata.X.mean(axis=0)).flatten()
        
        if 'total_counts' not in adata.var.columns:
            adata.var['total_counts'] = np.array(adata.X.sum(axis=0)).flatten()
        
        return adata

    def _flag_special_genes(self, adata: anndata.AnnData) -> anndata.AnnData:
        """Flag mitochondrial and ribosomal genes."""
        
        # Get gene symbols for pattern matching
        gene_symbols = None
        if 'gene_symbol' in adata.var.columns:
            gene_symbols = adata.var['gene_symbol']
        elif adata.var_names is not None:
            gene_symbols = adata.var_names
        
        if gene_symbols is not None:
            # Flag mitochondrial genes
            if 'mt' not in adata.var.columns:
                mt_genes = gene_symbols.str.startswith(('MT-', 'mt-', 'Mt-'), na=False)
                adata.var['mt'] = mt_genes
            
            # Flag ribosomal genes
            if 'ribo' not in adata.var.columns:
                ribo_genes = gene_symbols.str.startswith(('RPL', 'RPS', 'rpl', 'rps'), na=False)
                adata.var['ribo'] = ribo_genes
            
            # Calculate mitochondrial and ribosomal percentages for single-cell
            if self.data_type == "single_cell":
                if 'mt' in adata.var.columns and 'pct_counts_mt' not in adata.obs.columns:
                    adata.obs['pct_counts_mt'] = (
                        adata[:, adata.var['mt']].X.sum(axis=1) / 
                        adata.X.sum(axis=1) * 100
                    ).A1 if hasattr(adata.X, 'A1') else (
                        adata[:, adata.var['mt']].X.sum(axis=1) / 
                        adata.X.sum(axis=1) * 100
                    )
                
                if 'ribo' in adata.var.columns and 'pct_counts_ribo' not in adata.obs.columns:
                    adata.obs['pct_counts_ribo'] = (
                        adata[:, adata.var['ribo']].X.sum(axis=1) / 
                        adata.X.sum(axis=1) * 100
                    ).A1 if hasattr(adata.X, 'A1') else (
                        adata[:, adata.var['ribo']].X.sum(axis=1) / 
                        adata.X.sum(axis=1) * 100
                    )
        
        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate transcriptomics-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        metrics = super().get_quality_metrics(adata)
        
        # Add transcriptomics-specific metrics
        if self.data_type == "single_cell":
            if 'pct_counts_mt' in adata.obs.columns:
                metrics['mean_pct_mt'] = float(adata.obs['pct_counts_mt'].mean())
                metrics['high_mt_cells'] = int((adata.obs['pct_counts_mt'] > 20).sum())
            
            if 'n_genes_by_counts' in adata.obs.columns:
                metrics['mean_genes_per_cell'] = float(adata.obs['n_genes_by_counts'].mean())
                metrics['low_gene_cells'] = int((adata.obs['n_genes_by_counts'] < 200).sum())
        
        # Gene-level metrics
        if 'mt' in adata.var.columns:
            metrics['mt_genes'] = int(adata.var['mt'].sum())
        
        if 'ribo' in adata.var.columns:
            metrics['ribo_genes'] = int(adata.var['ribo'].sum())
        
        if 'n_cells_by_counts' in adata.var.columns:
            metrics['mean_cells_per_gene'] = float(adata.var['n_cells_by_counts'].mean())
        
        return metrics

    def detect_data_type(self, adata: anndata.AnnData) -> str:
        """
        Detect whether data is single-cell or bulk RNA-seq.

        Args:
            adata: AnnData object to analyze

        Returns:
            str: Detected data type
        """
        # Heuristics for detecting data type
        n_obs = adata.n_obs
        n_vars = adata.n_vars
        
        # Calculate sparsity
        if hasattr(adata.X, 'nnz'):
            sparsity = 1.0 - (adata.X.nnz / adata.X.size)
        else:
            sparsity = 1.0 - (adata.X != 0).sum() / adata.X.size
        
        # Single-cell typically has:
        # - Many observations (cells)
        # - High sparsity
        # - Lower counts per cell
        if n_obs > 100 and sparsity > 0.8:
            return "single_cell"
        
        # Bulk typically has:
        # - Fewer observations (samples)
        # - Lower sparsity
        # - Higher counts per sample
        elif n_obs < 100 and sparsity < 0.5:
            return "bulk"
        
        # If unclear, return the current setting
        return self.data_type
