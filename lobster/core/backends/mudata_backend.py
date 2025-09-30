"""
MuData backend implementation for multi-modal data storage.

This module provides the MuDataBackend for storing multi-modal
biological data using the MuData format, enabling integrated
analysis of transcriptomics, proteomics, and other modalities.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np

from lobster.core.backends.base import BaseBackend

# Import MuData with fallback
import mudata
MUDATA_AVAILABLE = True


logger = logging.getLogger(__name__)


class MuDataBackend(BaseBackend):
    """
    Backend for MuData multi-modal storage.
    
    This backend handles multi-modal biological data stored in MuData format,
    enabling integrated analysis of multiple data modalities like
    transcriptomics and proteomics.
    """

    def __init__(
        self, 
        base_path: Optional[Union[str, Path]] = None,
        compression: str = "gzip",
        compression_opts: Optional[int] = None
    ):
        """
        Initialize the MuData backend.

        Args:
            base_path: Optional base path for all operations
            compression: Compression method for MuData files
            compression_opts: Compression level (1-9 for gzip)

        Raises:
            ImportError: If MuData is not available
        """
        if not MUDATA_AVAILABLE:
            raise ImportError(
                "MuData is not available. Please install it with: pip install mudata"
            )
        
        super().__init__(base_path=base_path)
        self.compression = compression
        self.compression_opts = compression_opts or 6

    def load(self, path: Union[str, Path], **kwargs) -> mudata.MuData:
        """
        Load MuData from file.

        Args:
            path: Path to MuData file (local path or future S3 URI)
            **kwargs: Additional loading parameters:
                - backed: Load in backed mode (default: False)

        Returns:
            mudata.MuData: Loaded MuData object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        resolved_path = self._resolve_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"MuData file not found: {resolved_path}")
        
        try:
            # Extract loading parameters
            backed = kwargs.get('backed', False)
            
            if backed:
                # Load in backed mode for large files
                mdata = mudata.read_h5mu(resolved_path, backed='r')
            else:
                # Load fully into memory
                mdata = mudata.read_h5mu(resolved_path)
            
            self._log_operation(
                "load", 
                resolved_path, 
                backed=backed,
                n_modalities=len(mdata.mod),
                size_mb=resolved_path.stat().st_size / 1024**2
            )
            
            return mdata
            
        except Exception as e:
            raise ValueError(f"Failed to load MuData file {resolved_path}: {e}")

    def save(
        self, 
        mdata: mudata.MuData, 
        path: Union[str, Path], 
        **kwargs
    ) -> None:
        """
        Save MuData to file.

        Args:
            mdata: MuData object to save (can also accept AnnData for single modality)
            path: Destination path
            **kwargs: Additional saving parameters:
                - compression: Override default compression
                - compression_opts: Override compression level

        Raises:
            ValueError: If the data cannot be serialized
            TypeError: If data type is not supported
        """
        resolved_path = self._resolve_path(path)
        
        # Handle AnnData input by converting to MuData
        if isinstance(mdata, anndata.AnnData):
            # Create single-modality MuData
            modality_name = kwargs.get('modality_name', 'rna')  # Default to 'rna'
            mdata = mudata.MuData({modality_name: mdata})
            self.logger.info(f"Converted AnnData to MuData with modality '{modality_name}'")
        
        if not isinstance(mdata, mudata.MuData):
            raise TypeError(f"Expected MuData or AnnData object, got {type(mdata)}")
        
        # Ensure parent directory exists
        self._ensure_directory(resolved_path)
        
        # Create backup if file exists
        if resolved_path.exists():
            backup_path = self.create_backup(resolved_path)
            if backup_path:
                self.logger.info(f"Created backup: {backup_path}")
        
        try:
            # Extract saving parameters
            compression = kwargs.get('compression', self.compression)
            compression_opts = kwargs.get('compression_opts', self.compression_opts)
            
            # Save to MuData format
            mudata.write_h5mu(
                resolved_path,
                mdata,
                compression=compression,
                compression_opts=compression_opts
            )
            
            self._log_operation(
                "save",
                resolved_path,
                compression=compression,
                compression_opts=compression_opts,
                n_modalities=len(mdata.mod),
                modalities=list(mdata.mod.keys()),
                size_mb=resolved_path.stat().st_size / 1024**2
            )
            
        except Exception as e:
            # Remove failed file if it was created
            if resolved_path.exists():
                try:
                    resolved_path.unlink()
                except:
                    pass
            raise ValueError(f"Failed to save MuData file {resolved_path}: {e}")

    def supports_format(self, format_name: str) -> bool:
        """
        Check if the backend supports a specific file format.

        Args:
            format_name: Format to check

        Returns:
            bool: True if format is supported
        """
        return format_name.lower() in ['h5mu', 'mudata']

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about the storage backend.

        Returns:
            Dict[str, Any]: Storage backend information
        """
        info = super().get_storage_info()
        info.update({
            "supported_formats": ["h5mu"],
            "compression": self.compression,
            "compression_opts": self.compression_opts,
            "mudata_available": MUDATA_AVAILABLE,
            "mudata_version": mudata.__version__ if MUDATA_AVAILABLE else None,
            "multi_modal": True,
            "backed_mode_support": True
        })
        return info

    def add_modality(
        self,
        mdata: mudata.MuData,
        modality_name: str,
        adata: anndata.AnnData
    ) -> mudata.MuData:
        """
        Add a new modality to existing MuData object.

        Args:
            mdata: Existing MuData object
            modality_name: Name for the new modality
            adata: AnnData object to add

        Returns:
            mudata.MuData: Updated MuData object

        Raises:
            ValueError: If modality already exists
        """
        if modality_name in mdata.mod:
            raise ValueError(f"Modality '{modality_name}' already exists")
        
        # Add the new modality
        mdata.mod[modality_name] = adata
        
        # Update global observations if needed
        self._update_global_obs(mdata)
        
        self.logger.info(f"Added modality '{modality_name}' with shape {adata.shape}")
        
        return mdata

    def remove_modality(
        self,
        mdata: mudata.MuData,
        modality_name: str
    ) -> mudata.MuData:
        """
        Remove a modality from MuData object.

        Args:
            mdata: MuData object
            modality_name: Name of modality to remove

        Returns:
            mudata.MuData: Updated MuData object

        Raises:
            ValueError: If modality doesn't exist
        """
        if modality_name not in mdata.mod:
            raise ValueError(f"Modality '{modality_name}' does not exist")
        
        # Remove the modality
        del mdata.mod[modality_name]
        
        self.logger.info(f"Removed modality '{modality_name}'")
        
        return mdata

    def get_modality(
        self,
        mdata: mudata.MuData,
        modality_name: str
    ) -> anndata.AnnData:
        """
        Extract a specific modality from MuData object.

        Args:
            mdata: MuData object
            modality_name: Name of modality to extract

        Returns:
            anndata.AnnData: The requested modality

        Raises:
            ValueError: If modality doesn't exist
        """
        if modality_name not in mdata.mod:
            raise ValueError(f"Modality '{modality_name}' does not exist")
        
        return mdata.mod[modality_name]

    def list_modalities(self, mdata: mudata.MuData) -> List[str]:
        """
        List all modalities in MuData object.

        Args:
            mdata: MuData object

        Returns:
            List[str]: List of modality names
        """
        return list(mdata.mod.keys())

    def get_modality_info(self, mdata: mudata.MuData) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all modalities.

        Args:
            mdata: MuData object

        Returns:
            Dict[str, Dict[str, Any]]: Information about each modality
        """
        info = {}
        
        for mod_name, mod_data in mdata.mod.items():
            info[mod_name] = {
                "shape": mod_data.shape,
                "n_obs": mod_data.n_obs,
                "n_vars": mod_data.n_vars,
                "obs_columns": list(mod_data.obs.columns),
                "var_columns": list(mod_data.var.columns),
                "layers": list(mod_data.layers.keys()) if mod_data.layers else [],
                "obsm": list(mod_data.obsm.keys()) if mod_data.obsm else [],
                "varm": list(mod_data.varm.keys()) if mod_data.varm else [],
                "uns": list(mod_data.uns.keys()) if mod_data.uns else []
            }
        
        return info

    def create_mudata_from_dict(
        self,
        modality_dict: Dict[str, anndata.AnnData],
        global_obs: Optional[Dict[str, Any]] = None
    ) -> mudata.MuData:
        """
        Create MuData object from dictionary of modalities.

        Args:
            modality_dict: Dictionary mapping modality names to AnnData objects
            global_obs: Optional global observation metadata

        Returns:
            mudata.MuData: Created MuData object
        """
        # Create MuData object
        mdata = mudata.MuData(modality_dict)
        
        # Add global observation metadata if provided
        if global_obs:
            for key, value in global_obs.items():
                mdata.obs[key] = value
        
        # Update global observations
        self._update_global_obs(mdata)
        
        self.logger.info(f"Created MuData with {len(modality_dict)} modalities: {list(modality_dict.keys())}")
        
        return mdata

    def _update_global_obs(self, mdata: mudata.MuData) -> None:
        """
        Update global observation metadata from individual modalities.

        Args:
            mdata: MuData object to update
        """
        if not mdata.mod:
            return
        
        # Get observation names from the first modality
        first_mod = next(iter(mdata.mod.values()))
        obs_names = first_mod.obs_names
        
        # Check that all modalities have the same observations
        for mod_name, mod_data in mdata.mod.items():
            if not obs_names.equals(mod_data.obs_names):
                self.logger.warning(f"Modality '{mod_name}' has different observation names")
        
        # Update global obs if empty
        if mdata.obs.empty:
            mdata.obs = first_mod.obs.copy()
        
        # Add modality-specific summary statistics to global obs
        for mod_name, mod_data in mdata.mod.items():
            # Add basic metrics only if observation names match
            if hasattr(mod_data.X, 'sum') and mdata.obs.index.equals(mod_data.obs.index):
                mdata.obs[f'{mod_name}_total_counts'] = np.array(mod_data.X.sum(axis=1)).flatten()
                mdata.obs[f'{mod_name}_n_features'] = (mod_data.X > 0).sum(axis=1)
            elif hasattr(mod_data.X, 'sum'):
                # If observation names don't match, skip adding these metrics to avoid index mismatch
                self.logger.warning(f"Skipping summary statistics for modality '{mod_name}' due to observation name mismatch")

    def validate_file_integrity(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate MuData file integrity and structure.

        Args:
            path: Path to MuData file

        Returns:
            Dict[str, Any]: Validation results
        """
        resolved_path = self._resolve_path(path)
        
        validation = {
            "valid": False,
            "readable": False,
            "n_modalities": 0,
            "modalities": [],
            "global_shape": None,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if file exists and is readable
            if not resolved_path.exists():
                validation["errors"].append("File does not exist")
                return validation
            
            # Try to load the file
            mdata = self.load(resolved_path, backed=True)
            validation["readable"] = True
            
            # Check basic structure
            validation["n_modalities"] = len(mdata.mod)
            validation["modalities"] = list(mdata.mod.keys())
            validation["global_shape"] = mdata.shape
            
            # Validate each modality
            for mod_name, mod_data in mdata.mod.items():
                if mod_data.n_obs == 0:
                    validation["warnings"].append(f"Modality '{mod_name}' has no observations")
                if mod_data.n_vars == 0:
                    validation["warnings"].append(f"Modality '{mod_name}' has no variables")
            
            # Check for empty MuData
            if len(mdata.mod) == 0:
                validation["errors"].append("No modalities found in MuData object")
            
            validation["valid"] = len(validation["errors"]) == 0
            
        except Exception as e:
            validation["errors"].append(f"Failed to read file: {e}")
        
        return validation

    def convert_to_mudata(
        self,
        data: Union[anndata.AnnData, Dict[str, anndata.AnnData]],
        modality_name: str = "rna"
    ) -> mudata.MuData:
        """
        Convert AnnData or dictionary of AnnData to MuData.

        Args:
            data: AnnData object or dictionary of AnnData objects
            modality_name: Name for single AnnData modality

        Returns:
            mudata.MuData: Converted MuData object
        """
        if isinstance(data, anndata.AnnData):
            # Single modality
            return mudata.MuData({modality_name: data})
        elif isinstance(data, dict):
            # Multiple modalities
            return self.create_mudata_from_dict(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def merge_mudata_objects(
        self,
        mudata_list: List[mudata.MuData],
        merge_strategy: str = "outer"
    ) -> mudata.MuData:
        """
        Merge multiple MuData objects.

        Args:
            mudata_list: List of MuData objects to merge
            merge_strategy: How to merge ('outer', 'inner')

        Returns:
            mudata.MuData: Merged MuData object

        Raises:
            ValueError: If merge strategy is invalid or objects can't be merged
        """
        if not mudata_list:
            raise ValueError("No MuData objects provided")
        
        if len(mudata_list) == 1:
            return mudata_list[0].copy()
        
        # This is a simplified merge - in practice, you'd want more sophisticated logic
        # For now, just merge the first two and extend if needed
        
        if merge_strategy not in ["outer", "inner"]:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Start with first object
        merged = mudata_list[0].copy()
        
        # Add modalities from other objects
        for i, other_mdata in enumerate(mudata_list[1:], 1):
            for mod_name, mod_data in other_mdata.mod.items():
                # Create unique modality name if it already exists
                unique_mod_name = mod_name
                counter = 1
                while unique_mod_name in merged.mod:
                    unique_mod_name = f"{mod_name}_{counter}"
                    counter += 1
                
                merged.mod[unique_mod_name] = mod_data.copy()
                
                if unique_mod_name != mod_name:
                    self.logger.warning(f"Renamed modality '{mod_name}' to '{unique_mod_name}' to avoid conflicts")
        
        self.logger.info(f"Merged {len(mudata_list)} MuData objects into one with {len(merged.mod)} modalities")
        
        return merged
