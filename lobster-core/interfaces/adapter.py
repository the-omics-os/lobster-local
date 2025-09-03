"""
Modality adapter interface definitions.

This module defines the abstract interface for modality-specific data adapters,
enabling support for different biological data modalities (transcriptomics,
proteomics, etc.) with consistent schema enforcement and validation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import pandas as pd


class IModalityAdapter(ABC):
    """
    Abstract interface for modality-specific data adapters.
    
    This interface defines the contract for converting raw data from various
    sources into standardized AnnData objects with modality-specific schemas.
    Each adapter handles the specific requirements and conventions of its
    biological data modality.
    """

    @abstractmethod
    def from_source(
        self, 
        source: Union[str, Path, pd.DataFrame], 
        **kwargs
    ) -> anndata.AnnData:
        """
        Convert source data to AnnData with appropriate schema.

        Args:
            source: Data source (file path, DataFrame, or other format)
            **kwargs: Modality-specific conversion parameters

        Returns:
            anndata.AnnData: Standardized data object with proper schema

        Raises:
            ValueError: If source data is invalid or cannot be converted
            FileNotFoundError: If source file doesn't exist
            TypeError: If source format is not supported
        """
        pass

    @abstractmethod
    def validate(
        self, 
        adata: anndata.AnnData, 
        strict: bool = False
    ) -> "ValidationResult":
        """
        Validate AnnData against modality schema.

        Args:
            adata: AnnData object to validate
            strict: If True, treat warnings as errors

        Returns:
            ValidationResult: Validation results with errors/warnings

        Raises:
            ValueError: If strict=True and validation fails
        """
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Return the expected schema for this modality.

        Returns:
            Dict[str, Any]: Schema definition containing:
                - required_obs: Required observation (cell/sample) metadata
                - required_var: Required variable (gene/protein) metadata
                - optional_obs: Optional observation metadata
                - optional_var: Optional variable metadata
                - layers: Expected data layers
                - obsm: Expected multi-dimensional observations
                - uns: Expected unstructured metadata
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.

        Returns:
            List[str]: List of supported file extensions or format names
        """
        pass

    def get_modality_name(self) -> str:
        """
        Get the name of this modality.

        Returns:
            str: Modality name (e.g., 'transcriptomics', 'proteomics')
        """
        return self.__class__.__name__.lower().replace('adapter', '')

    def detect_format(self, source: Union[str, Path]) -> Optional[str]:
        """
        Detect the format of a source file.

        Args:
            source: Path to the source file

        Returns:
            Optional[str]: Detected format name, None if unknown
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            extension = path.suffix.lower()
            
            format_mapping = {
                '.csv': 'csv',
                '.tsv': 'tsv',
                '.txt': 'txt',
                '.h5ad': 'h5ad',
                '.h5': 'h5',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.mtx': 'mtx',
                '.h5mu': 'h5mu'
            }
            
            return format_mapping.get(extension)
        
        return None

    def preprocess_data(
        self, 
        adata: anndata.AnnData, 
        **kwargs
    ) -> anndata.AnnData:
        """
        Apply modality-specific preprocessing steps.

        Args:
            adata: Input AnnData object
            **kwargs: Preprocessing parameters

        Returns:
            anndata.AnnData: Preprocessed data object
        """
        # Default implementation - subclasses should override for specific preprocessing
        return adata

    def get_quality_metrics(self, adata: anndata.AnnData) -> Dict[str, Any]:
        """
        Calculate modality-specific quality metrics.

        Args:
            adata: AnnData object to analyze

        Returns:
            Dict[str, Any]: Quality metrics dictionary
        """
        # Default implementation with basic metrics
        return {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "sparsity": 1.0 - (adata.X != 0).sum() / adata.X.size if hasattr(adata.X, 'size') else 0.0,
            "memory_usage": adata.X.nbytes if hasattr(adata.X, 'nbytes') else 0
        }

    def standardize_metadata(
        self, 
        adata: anndata.AnnData,
        metadata_mapping: Optional[Dict[str, str]] = None
    ) -> anndata.AnnData:
        """
        Standardize metadata column names according to schema.

        Args:
            adata: AnnData object to standardize
            metadata_mapping: Optional mapping of current -> standard names

        Returns:
            anndata.AnnData: AnnData with standardized metadata
        """
        if metadata_mapping:
            # Rename observation metadata
            for old_name, new_name in metadata_mapping.items():
                if old_name in adata.obs.columns:
                    adata.obs = adata.obs.rename(columns={old_name: new_name})
                
                if old_name in adata.var.columns:
                    adata.var = adata.var.rename(columns={old_name: new_name})
        
        return adata

    def add_provenance(
        self,
        adata: anndata.AnnData,
        source_info: Dict[str, Any],
        processing_params: Optional[Dict[str, Any]] = None
    ) -> anndata.AnnData:
        """
        Add provenance information to AnnData object.

        Args:
            adata: AnnData object to annotate
            source_info: Information about data source
            processing_params: Parameters used in processing

        Returns:
            anndata.AnnData: AnnData with provenance information
        """
        import datetime
        
        provenance = {
            "adapter": self.__class__.__name__,
            "modality": self.get_modality_name(),
            "source": source_info,
            "processing_params": processing_params or {},
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0"  # Could be extracted from package version
        }
        
        # Add to uns (unstructured metadata)
        if "provenance" not in adata.uns:
            adata.uns["provenance"] = []
        
        adata.uns["provenance"].append(provenance)
        
        return adata
