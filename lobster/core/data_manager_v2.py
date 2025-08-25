"""
DataManagerV2: Modular orchestration layer for multi-omics data management.

This module provides the new DataManagerV2 class that orchestrates
modality adapters, storage backends, and validation to enable
flexible multi-omics data analysis with complete provenance tracking.
"""

import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anndata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.backend import IDataBackend
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.provenance import ProvenanceTracker

# Import available backends and adapters
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter

# Import MuData backend if available
try:
    from lobster.core.backends.mudata_backend import MuDataBackend
    MUDATA_BACKEND_AVAILABLE = True
except ImportError:
    MUDATA_BACKEND_AVAILABLE = False

# Import MuData if available
try:
    import mudata
    MUDATA_AVAILABLE = True
except ImportError:
    MUDATA_AVAILABLE = False
    mudata = None

logger = logging.getLogger(__name__)


class DataManagerV2:
    """
    Modular data manager for multi-omics analysis.
    
    This class orchestrates modality adapters, storage backends,
    and validation to provide a unified interface for managing
    multi-modal biological data with complete provenance tracking.
    """

    def __init__(
        self,
        default_backend: str = "h5ad",
        workspace_path: Optional[Union[str, Path]] = None,
        enable_provenance: bool = True,
        console=None
    ):
        """
        Initialize DataManagerV2.

        Args:
            default_backend: Default storage backend to use
            workspace_path: Optional workspace directory for data storage
            enable_provenance: Whether to enable provenance tracking
            console: Optional Rich console instance for progress tracking
        """
        self.default_backend = default_backend
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd() / ".lobster_workspace"
        self.enable_provenance = enable_provenance
        self.console = console  # Store console for progress tracking in tools
        
        # Core storage
        self.backends: Dict[str, IDataBackend] = {}
        self.adapters: Dict[str, IModalityAdapter] = {}
        self.modalities: Dict[str, anndata.AnnData] = {}
        
        # Metadata storage for GEO datasets and other sources temporary until data is loaded
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        
        # Legacy compatibility attributes for tool usage tracking
        self.tool_usage_history: List[Dict[str, Any]] = []
        self.processing_log: List[str] = []
        
        # Plot management
        self.latest_plots: List[Dict[str, Any]] = []  # Store plots with metadata
        self.plot_counter: int = 0  # Counter for generating unique IDs
        self.max_plots_history: int = 50  # Maximum number of plots to keep in history
        
        # Current dataset management (for legacy compatibility)
        self.current_dataset: Optional[str] = None  # Name of current active modality
        self.current_data: Optional[pd.DataFrame] = None  # Legacy compatibility
        self.current_metadata: Dict[str, Any] = {}  # Legacy metadata
        self.adata: Optional[anndata.AnnData] = None  # Legacy AnnData reference
        
        # Provenance tracking
        self.provenance = ProvenanceTracker() if enable_provenance else None
        
        # Setup workspace
        self._setup_workspace()
        
        # Register default backends and adapters
        self._register_default_backends()
        self._register_default_adapters()
        
        logger.info(f"Initialized DataManagerV2 with workspace: {self.workspace_path}")

    def _setup_workspace(self) -> None:
        """Set up workspace directories."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.workspace_path / "data"
        self.exports_dir = self.workspace_path / "exports" 
        self.cache_dir = self.workspace_path / "cache"
        
        for directory in [self.data_dir, self.exports_dir, self.cache_dir]:
            directory.mkdir(exist_ok=True)

    def _register_default_backends(self) -> None:
        """Register default storage backends."""
        # H5AD backend
        self.register_backend(
            "h5ad",
            H5ADBackend(base_path=self.data_dir)
        )
        
        # MuData backend if available
        if MUDATA_BACKEND_AVAILABLE:
            self.register_backend(
                "mudata",
                MuDataBackend(base_path=self.data_dir)
            )

    def _register_default_adapters(self) -> None:
        """Register default modality adapters."""
        # Transcriptomics adapters
        self.register_adapter(
            "transcriptomics_single_cell",
            TranscriptomicsAdapter(data_type="single_cell", strict_validation=False)
        )
        
        self.register_adapter(
            "transcriptomics_bulk",
            TranscriptomicsAdapter(data_type="bulk", strict_validation=False)
        )
        
        # Proteomics adapters
        self.register_adapter(
            "proteomics_ms", 
            ProteomicsAdapter(data_type="mass_spectrometry", strict_validation=False)
        )
        
        self.register_adapter(
            "proteomics_affinity",
            ProteomicsAdapter(data_type="affinity", strict_validation=False)
        )

    def register_backend(self, name: str, backend: IDataBackend) -> None:
        """
        Register a storage backend.

        Args:
            name: Name for the backend
            backend: Backend implementation

        Raises:
            ValueError: If backend name already exists
        """
        if name in self.backends:
            raise ValueError(f"Backend '{name}' already registered")
        
        self.backends[name] = backend
        logger.info(f"Registered backend: {name} ({backend.__class__.__name__})")

    def register_adapter(self, name: str, adapter: IModalityAdapter) -> None:
        """
        Register a modality adapter.

        Args:
            name: Name for the adapter
            adapter: Adapter implementation

        Raises:
            ValueError: If adapter name already exists
        """
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already registered")
        
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name} ({adapter.__class__.__name__})")

    def load_modality(
        self,
        name: str,
        source: Union[str, Path, pd.DataFrame, anndata.AnnData],
        adapter: str,
        validate: bool = True,
        **kwargs
    ) -> anndata.AnnData:
        """
        Load data for a specific modality.

        Args:
            name: Name for the modality
            source: Data source (file path, DataFrame, or AnnData)
            adapter: Name of adapter to use
            validate: Whether to validate the loaded data
            **kwargs: Additional parameters passed to adapter

        Returns:
            anndata.AnnData: Loaded and validated data

        Raises:
            ValueError: If adapter is not registered or validation fails
        """
        if adapter not in self.adapters:
            raise ValueError(f"Adapter '{adapter}' not registered")
        
        adapter_instance = self.adapters[adapter]
        
        # Load data using adapter
        adata = adapter_instance.from_source(source, **kwargs)
        
        # Validate data if requested
        if validate:
            validation_result = adapter_instance.validate(adata, strict=False)
            
            if validation_result.has_errors:
                error_msg = f"Validation failed for modality '{name}': {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if validation_result.has_warnings:
                logger.warning(f"Validation warnings for modality '{name}': {validation_result.warnings}")
        
        # Store modality
        self.modalities[name] = adata
        
        # Log provenance
        if self.provenance:
            entity_id = self.provenance.create_entity(
                entity_type="modality_data",
                metadata={
                    "modality_name": name,
                    "adapter": adapter,
                    "shape": adata.shape
                }
            )
            
            # self.provenance.log_data_loading(
            #     source_path=source,
            #     output_entity_id=entity_id,
            #     adapter_name=adapter,
            #     parameters=kwargs
            # )
            
            # Add provenance to AnnData
            adata = self.provenance.add_to_anndata(adata)
        
        logger.info(f"Loaded modality '{name}': {adata.shape} using adapter '{adapter}'")
        return adata

    def save_modality(
        self,
        name: str,
        path: Union[str, Path],
        backend: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save a modality using specified backend.

        Args:
            name: Name of modality to save
            path: Destination path
            backend: Backend to use (default: default_backend)
            **kwargs: Additional parameters passed to backend

        Returns:
            str: Path where data was saved

        Raises:
            ValueError: If modality or backend not found
        """
        if name not in self.modalities:
            raise ValueError(f"Modality '{name}' not loaded")
        
        backend_name = backend or self.default_backend
        if backend_name not in self.backends:
            raise ValueError(f"Backend '{backend_name}' not registered")
        
        backend_instance = self.backends[backend_name]
        adata = self.modalities[name]
        
        # Resolve path
        if not Path(path).is_absolute():
            path = self.data_dir / path
        
        # Save data
        backend_instance.save(adata, path, **kwargs)
        
        # Log provenance
        if self.provenance:
            entity_id = self.provenance.create_entity(
                entity_type="modality_data",
                uri=path,  # Use in-memory representation as source
                metadata={
                    "modality_name": name,
                    "shape": adata.shape
                }
            )
        
        logger.info(f"Saved modality '{name}' to {path} using backend '{backend_name}'")
        return str(path)

    def get_modality(self, name: str) -> anndata.AnnData:
        """
        Get a specific modality.

        Args:
            name: Name of modality

        Returns:
            anndata.AnnData: The requested modality

        Raises:
            ValueError: If modality not found
        """
        if name not in self.modalities:
            raise ValueError(f"Modality '{name}' not found")
        
        return self.modalities[name]

    def list_modalities(self) -> List[str]:
        """
        List all loaded modalities.

        Returns:
            List[str]: List of modality names
        """
        return list(self.modalities.keys())

    def remove_modality(self, name: str) -> None:
        """
        Remove a modality from memory.

        Args:
            name: Name of modality to remove

        Raises:
            ValueError: If modality not found
        """
        if name not in self.modalities:
            raise ValueError(f"Modality '{name}' not found")
        
        del self.modalities[name]
        logger.info(f"Removed modality '{name}'")

    def to_mudata(self, modalities: Optional[List[str]] = None) -> Any:
        """
        Convert modalities to MuData object.

        Args:
            modalities: List of modality names to include (default: all)

        Returns:
            mudata.MuData: MuData object containing specified modalities

        Raises:
            ImportError: If MuData is not available
            ValueError: If no modalities are loaded
        """
        if not MUDATA_AVAILABLE:
            raise ImportError("MuData is not available. Please install it with: pip install mudata")
        
        if not self.modalities:
            raise ValueError("No modalities loaded")
        
        # Use specified modalities or all available
        modality_names = modalities or list(self.modalities.keys())
        
        # Check that all requested modalities exist
        missing = [name for name in modality_names if name not in self.modalities]
        if missing:
            raise ValueError(f"Modalities not found: {missing}")
        
        # Create modality dictionary
        modality_dict = {name: self.modalities[name] for name in modality_names}
        
        # Create MuData object
        mdata = mudata.MuData(modality_dict)
        
        # Add provenance if enabled
        if self.provenance:
            mdata = self.provenance.add_to_anndata(mdata)
        
        logger.info(f"Created MuData with {len(modality_dict)} modalities: {list(modality_dict.keys())}")
        return mdata

    def save_mudata(
        self,
        path: Union[str, Path],
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Save modalities as MuData file.

        Args:
            path: Destination path
            modalities: List of modality names to include (default: all)
            **kwargs: Additional parameters passed to MuData backend

        Returns:
            str: Path where data was saved

        Raises:
            ValueError: If MuData backend not available
        """
        if "mudata" not in self.backends:
            raise ValueError("MuData backend not available")
        
        # Create MuData object
        mdata = self.to_mudata(modalities=modalities)
        
        # Resolve path
        if not Path(path).is_absolute():
            path = self.data_dir / path
        
        # Save using MuData backend
        mudata_backend = self.backends["mudata"]
        mudata_backend.save(mdata, path, **kwargs)
        
        logger.info(f"Saved MuData to {path}")
        return str(path)

    def get_quality_metrics(self, modality: Optional[str] = None) -> Dict[str, Any]:
        """
        Get quality metrics for modalities.

        Args:
            modality: Specific modality name (default: all modalities)

        Returns:
            Dict[str, Any]: Quality metrics
        """
        if modality:
            if modality not in self.modalities:
                raise ValueError(f"Modality '{modality}' not found")
            
            # Get adapter for this modality (find by checking registered adapters)
            adapter_instance = None
            for adapter_name, adapter in self.adapters.items():
                if modality in adapter_name or adapter.get_modality_name() in modality.lower():
                    adapter_instance = adapter
                    break
            
            if adapter_instance:
                return adapter_instance.get_quality_metrics(self.modalities[modality])
            else:
                # Use basic metrics if no specific adapter found
                from lobster.core.adapters.base import BaseAdapter
                base_adapter = BaseAdapter()
                return base_adapter.get_quality_metrics(self.modalities[modality])
        
        else:
            # Return metrics for all modalities
            all_metrics = {}
            for mod_name in self.modalities:
                all_metrics[mod_name] = self.get_quality_metrics(mod_name)
            return all_metrics

    def get_workspace_status(self) -> Dict[str, Any]:
        """
        Get comprehensive workspace status.

        Returns:
            Dict[str, Any]: Workspace status information
        """
        status = {
            "workspace_path": str(self.workspace_path),
            "modalities_loaded": len(self.modalities),
            "modality_names": list(self.modalities.keys()),
            "registered_backends": list(self.backends.keys()),
            "registered_adapters": list(self.adapters.keys()),
            "default_backend": self.default_backend,
            "provenance_enabled": self.enable_provenance,
            "mudata_available": MUDATA_AVAILABLE,
            "directories": {
                "data": str(self.data_dir),
                "exports": str(self.exports_dir),
                "cache": str(self.cache_dir)
            }
        }
        
        # Add modality details
        if self.modalities:
            status["modality_details"] = {}
            for name, adata in self.modalities.items():
                status["modality_details"][name] = {
                    "shape": adata.shape,
                    "obs_columns": list(adata.obs.columns)[:10],  # First 10 columns
                    "var_columns": list(adata.var.columns)[:10],  # First 10 columns
                    "layers": list(adata.layers.keys()) if adata.layers else [],
                    "obsm": list(adata.obsm.keys()) if adata.obsm else []
                }
        
        # Add provenance info
        if self.provenance:
            status["provenance"] = {
                "n_activities": len(self.provenance.activities),
                "n_entities": len(self.provenance.entities),
                "n_agents": len(self.provenance.agents)
            }
        
        return status

    def validate_modalities(self, strict: bool = False) -> Dict[str, ValidationResult]:
        """
        Validate all loaded modalities.

        Args:
            strict: Whether to use strict validation

        Returns:
            Dict[str, ValidationResult]: Validation results for each modality
        """
        results = {}
        
        for name, adata in self.modalities.items():
            # Find appropriate adapter for validation
            adapter_instance = None
            for adapter_name, adapter in self.adapters.items():
                if name in adapter_name or adapter.get_modality_name() in name.lower():
                    adapter_instance = adapter
                    break
            
            if adapter_instance:
                results[name] = adapter_instance.validate(adata, strict=strict)
            else:
                # Use basic validation if no specific adapter found
                from lobster.core.adapters.base import BaseAdapter
                base_adapter = BaseAdapter()
                results[name] = base_adapter._validate_basic_structure(adata)
        
        return results

    def export_provenance(self, path: Union[str, Path]) -> str:
        """
        Export provenance information to file.

        Args:
            path: Export file path

        Returns:
            str: Path to exported file

        Raises:
            ValueError: If provenance tracking is disabled
        """
        if not self.provenance:
            raise ValueError("Provenance tracking is disabled")
        
        # Resolve path
        if not Path(path).is_absolute():
            path = self.exports_dir / path
        
        # Export provenance
        provenance_data = self.provenance.to_dict()
        
        import json
        with open(path, 'w') as f:
            json.dump(provenance_data, f, indent=2, default=str)
        
        logger.info(f"Exported provenance to {path}")
        return str(path)

    def clear_workspace(self, confirm: bool = False) -> None:
        """
        Clear all modalities and optionally workspace files.

        Args:
            confirm: Must be True to actually clear workspace

        Raises:
            ValueError: If confirm is not True
        """
        if not confirm:
            raise ValueError("Must set confirm=True to clear workspace")
        
        # Clear modalities from memory
        self.modalities.clear()
        
        # Reset provenance
        if self.provenance:
            self.provenance = ProvenanceTracker()
        
        logger.info("Cleared workspace")

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about registered backends.

        Returns:
            Dict[str, Any]: Backend information
        """
        info = {}
        for name, backend in self.backends.items():
            info[name] = backend.get_storage_info()
        return info

    def get_adapter_info(self) -> Dict[str, Any]:
        """
        Get information about registered adapters.

        Returns:
            Dict[str, Any]: Adapter information
        """
        info = {}
        for name, adapter in self.adapters.items():
            info[name] = {
                "modality_name": adapter.get_modality_name(),
                "supported_formats": adapter.get_supported_formats(),
                "schema": adapter.get_schema()
            }
        return info

    # Legacy compatibility methods for tools
    def log_tool_usage(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any], 
        description: str = None
    ) -> None:
        """
        Log tool usage for reproducibility tracking.
        
        Args:
            tool_name: Name of the tool used
            parameters: Parameters used with the tool
            description: Optional description of what was done
        """
        import datetime
        
        self.tool_usage_history.append({
            "tool": tool_name,
            "parameters": parameters,
            "description": description,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        logger.info(f"Tool usage logged: {tool_name}")

    def save_processed_data(
        self,
        processing_step: str,
        data_source: str = None,
        dataset_id: str = None,
        processing_params: Dict[str, Any] = None
    ) -> str:
        """
        Save processed data with professional naming and enhanced metadata tracking.
        
        Args:
            processing_step: Current processing step (e.g., 'filtered', 'normalized')
            data_source: Data source (extracted from metadata if not provided)
            dataset_id: Dataset identifier (extracted from metadata if not provided)
            processing_params: Parameters used in the processing step
            
        Returns:
            str: Path to saved file
        """
        if not self.modalities:
            logger.warning("No modalities to save")
            return None

        try:
            from lobster.utils.file_naming import BioinformaticsFileNaming
            import json
            
            # Use the first modality if multiple exist
            modality_name = list(self.modalities.keys())[0]
            adata = self.modalities[modality_name]
            
            # Extract information from metadata or parameters
            data_source = data_source or 'DATA'
            dataset_id = dataset_id or 'unknown'
            
            # Generate professional filename with auto-selected extension
            filename = BioinformaticsFileNaming.generate_filename(
                data_source=data_source,
                dataset_id=dataset_id,
                processing_step=processing_step
            )
            filepath = self.data_dir / filename
            
            # Save using appropriate backend
            saved_path = self.save_modality(modality_name, filepath)
            
            # Create enhanced metadata
            enhanced_metadata = {
                'processing_step': processing_step,
                'data_source': data_source,
                'dataset_id': dataset_id,
                'saved_filename': filename,
                'saved_path': str(filepath),
                'save_timestamp': pd.Timestamp.now().isoformat(),
                'data_shape': list(adata.shape),
                'processing_params': processing_params or {},
                'processing_order': BioinformaticsFileNaming.get_processing_step_order(processing_step),
                'suggested_next_step': BioinformaticsFileNaming.suggest_next_step(processing_step),
                'modality_name': modality_name
            }

            # Save metadata
            metadata_filename = BioinformaticsFileNaming.generate_metadata_filename(filename)
            metadata_path = self.data_dir / metadata_filename
            
            with open(metadata_path, "w") as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)

            # Log the processing step
            self.processing_log.append(
                f"Saved {processing_step} data: {adata.shape[0]} obs × {adata.shape[1]} vars -> {filename}"
            )

            logger.info(f"Processed data saved with professional naming: {filepath}")
            logger.info(f"Next suggested step: {enhanced_metadata['suggested_next_step']}")
            
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return None

    def list_workspace_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all files in the workspace organized by category.
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Files organized by category
        """
        files_by_category = {"data": [], "exports": [], "cache": []}

        # List data files
        for file_path in self.data_dir.iterdir():
            if file_path.is_file():
                files_by_category["data"].append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                })

        # List export files
        for file_path in self.exports_dir.iterdir():
            if file_path.is_file():
                files_by_category["exports"].append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                })

        # List cache files
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                files_by_category["cache"].append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                })

        return files_by_category

    def auto_save_state(self) -> List[str]:
        """
        Automatically save current state including all modalities.
        
        Returns:
            List[str]: List of saved items
        """
        saved_items = []

        # Save all modalities
        for modality_name in self.modalities:
            try:
                save_path = f"{modality_name}_autosave.h5ad"
                saved_path = self.save_modality(modality_name, save_path)
                saved_items.append(f"Modality '{modality_name}': {Path(saved_path).name}")
            except Exception as e:
                logger.error(f"Failed to auto-save modality {modality_name}: {e}")

        # Save processing log and tool usage history
        if self.processing_log or self.tool_usage_history:
            try:
                log_path = self.exports_dir / "processing_log.json"
                log_data = {
                    "processing_log": self.processing_log,
                    "tool_usage_history": self.tool_usage_history,
                    "timestamp": pd.Timestamp.now().isoformat(),
                }
                import json
                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=2, default=str)
                saved_items.append("Processing log")
            except Exception as e:
                logger.error(f"Failed to save processing log: {e}")

        if saved_items:
            logger.info(f"Auto-saved: {', '.join(saved_items)}")

        return saved_items

    def has_data(self) -> bool:
        """
        Check if any modalities are loaded.
        
        Returns:
            bool: True if modalities exist, False otherwise
        """
        return len(self.modalities) > 0

    # ========================================
    # PLOT MANAGEMENT CAPABILITIES
    # ========================================
    
    def add_plot(
        self,
        plot: go.Figure,
        title: str = None,
        source: str = None,
        dataset_info: Dict[str, Any] = None,
        analysis_params: Dict[str, Any] = None,
    ) -> str:
        """
        Add a plot to the collection with comprehensive metadata.

        Args:
            plot: Plotly Figure object
            title: Optional title for the plot
            source: Optional source identifier (e.g., service name)
            dataset_info: Optional information about the dataset used
            analysis_params: Optional parameters used for the analysis

        Returns:
            str: The unique ID assigned to the plot

        Raises:
            ValueError: If plot is not a Plotly Figure
        """
        try:
            if not isinstance(plot, go.Figure):
                raise ValueError("Plot must be a plotly Figure object.")

            # Generate a unique identifier for the plot
            self.plot_counter += 1
            plot_id = f"plot_{self.plot_counter}"

            # Create timestamp
            timestamp = datetime.now().isoformat()
            human_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get current dataset information for context
            current_dataset_info = dataset_info or {}
            if self.current_dataset and not current_dataset_info:
                # Use current modality info
                adata = self.modalities.get(self.current_dataset)
                if adata:
                    current_dataset_info = {
                        "data_shape": adata.shape,
                        "modality_name": self.current_dataset,
                        "n_obs": adata.n_obs,
                        "n_vars": adata.n_vars,
                    }
            elif self.modalities and not current_dataset_info:
                # Use first available modality
                first_mod = list(self.modalities.keys())[0]
                adata = self.modalities[first_mod]
                current_dataset_info = {
                    "data_shape": adata.shape,
                    "modality_name": first_mod,
                    "n_obs": adata.n_obs,
                    "n_vars": adata.n_vars,
                }

            # Create comprehensive title with context
            enhanced_title = title or "Untitled"
            if current_dataset_info and "modality_name" in current_dataset_info:
                modality_name = current_dataset_info["modality_name"]
                enhanced_title = f"{enhanced_title} ({modality_name} - {human_timestamp})"
            elif current_dataset_info and "data_shape" in current_dataset_info:
                shape_info = f"{current_dataset_info['data_shape'][0]}x{current_dataset_info['data_shape'][1]}"
                enhanced_title = f"{enhanced_title} (Data: {shape_info} - {human_timestamp})"
            else:
                enhanced_title = f"{enhanced_title} ({human_timestamp})"

            # Update plot title
            plot.update_layout(title=enhanced_title)

            # Store plot with comprehensive metadata
            plot_entry = {
                "id": plot_id,
                "figure": plot,
                "title": enhanced_title,
                "original_title": title or "Untitled",
                "timestamp": timestamp,
                "human_timestamp": human_timestamp,
                "source": source or "unknown",
                "dataset_info": current_dataset_info,
                "analysis_params": analysis_params or {},
                "created_at": datetime.now(),
                "data_context": {
                    "has_modalities": self.has_data(),
                    "modality_names": list(self.modalities.keys()),
                    "current_dataset": self.current_dataset,
                },
            }

            # Add to the queue
            self.latest_plots.append(plot_entry)

            # Maintain maximum size of plot history
            if len(self.latest_plots) > self.max_plots_history:
                self.latest_plots.pop(0)  # Remove oldest plot

            logger.info(f"Plot added: '{enhanced_title}' with ID {plot_id} from {source}")
            return plot_id

        except Exception as e:
            logger.exception(f"Error in add_plot: {e}")
            return None

    def clear_plots(self) -> None:
        """Clear all stored plots."""
        self.latest_plots = []
        logger.info("All plots cleared")

    def get_plot_by_id(self, plot_id: str) -> Optional[go.Figure]:
        """
        Get a plot by its unique ID.

        Args:
            plot_id: The unique ID of the plot

        Returns:
            Optional[go.Figure]: The plot if found, None otherwise
        """
        for plot_entry in self.latest_plots:
            if plot_entry["id"] == plot_id:
                return plot_entry["figure"]
        return None

    def get_latest_plots(self, n: int = None) -> List[Dict[str, Any]]:
        """
        Get the n most recent plots with their metadata.

        Args:
            n: Number of plots to return (None for all)

        Returns:
            List[Dict[str, Any]]: List of plot entries with metadata
        """
        if n is None:
            return self.latest_plots
        return self.latest_plots[-n:]

    def get_plot_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete plot history with minimal metadata (no figures).

        Returns:
            List[Dict[str, Any]]: List of plot history entries
        """
        return [
            {
                "id": p["id"],
                "title": p["title"],
                "timestamp": p["timestamp"],
                "source": p["source"],
            }
            for p in self.latest_plots
        ]

    def save_plots_to_workspace(self) -> List[str]:
        """Save all current plots to the workspace directory."""
        if not self.latest_plots:
            logger.info("No plots to save")
            return []

        # Create plots directory in workspace
        plots_dir = self.workspace_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        saved_files = []

        for plot_entry in self.latest_plots:
            try:
                plot = plot_entry["figure"]
                plot_id = plot_entry["id"]
                plot_title = plot_entry["title"]

                # Create sanitized filename
                safe_title = "".join(
                    c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                ).rstrip()
                safe_title = safe_title.replace(" ", "_")
                filename_base = f"{plot_id}_{safe_title}" if safe_title else plot_id

                # Save as HTML (interactive)
                html_path = plots_dir / f"{filename_base}.html"
                pio.write_html(plot, html_path)
                saved_files.append(str(html_path))

                # Save as PNG (static)
                png_path = plots_dir / f"{filename_base}.png"
                try:
                    pio.write_image(plot, png_path)
                    saved_files.append(str(png_path))
                except Exception as e:
                    logger.warning(f"Could not save PNG for {plot_id}: {e}")

                logger.info(f"Saved plot {plot_id} to workspace")

            except Exception as e:
                logger.error(f"Failed to save plot {plot_id}: {e}")

        return saved_files

    # ========================================
    # CURRENT DATASET MANAGEMENT (Legacy Support)
    # ========================================

    def set_current_dataset(self, modality_name: str) -> None:
        """
        Set the current active modality for legacy compatibility.
        
        Args:
            modality_name: Name of modality to set as current
            
        Raises:
            ValueError: If modality not found
        """
        if modality_name not in self.modalities:
            raise ValueError(f"Modality '{modality_name}' not found")
        
        self.current_dataset = modality_name
        adata = self.modalities[modality_name]
        
        # Update legacy compatibility attributes
        try:
            # Convert AnnData to DataFrame for legacy compatibility
            if hasattr(adata, 'to_df'):
                self.current_data = adata.to_df().T  # Transpose to match expected format
            else:
                # Manual conversion
                self.current_data = pd.DataFrame(
                    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                    index=adata.obs_names,
                    columns=adata.var_names
                ).T  # Transpose for legacy format
            
            self.adata = adata
            
            # Update current metadata with modality info
            self.current_metadata = {
                "modality_name": modality_name,
                "shape": adata.shape,
                "obs_columns": list(adata.obs.columns),
                "var_columns": list(adata.var.columns),
                "data_type": self._detect_modality_type(modality_name),
            }
            
            # Add any existing uns metadata
            if hasattr(adata, 'uns') and adata.uns:
                self.current_metadata.update({
                    f"uns_{k}": v for k, v in adata.uns.items() 
                    if isinstance(v, (str, int, float, bool))
                })
                
        except Exception as e:
            logger.error(f"Failed to set current dataset: {e}")
            # Fallback - at least set the references
            self.current_data = None
            self.adata = adata
            self.current_metadata = {"modality_name": modality_name, "error": str(e)}
        
        logger.info(f"Set current dataset to modality: {modality_name}")

    def get_current_data(self) -> Optional[pd.DataFrame]:
        """Get current dataset as DataFrame (legacy compatibility)."""
        if self.current_dataset and self.current_dataset in self.modalities:
            # Ensure current_data is up-to-date
            if self.current_data is None:
                self.set_current_dataset(self.current_dataset)
        
        return self.current_data

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of current dataset or all modalities.
        
        Returns:
            Dict[str, Any]: Summary information
        """
        if not self.has_data():
            return {"status": "No modalities loaded"}
        
        if self.current_dataset and self.current_dataset in self.modalities:
            # Summary for current dataset
            adata = self.modalities[self.current_dataset]
            metrics = self.get_quality_metrics(self.current_dataset)
            
            summary = {
                "status": "Modality loaded",
                "modality_name": self.current_dataset,
                "shape": adata.shape,
                "obs_names": list(adata.obs_names[:5]),  # First 5 observations
                "var_names": list(adata.var_names[:5]),  # First 5 variables
                "obs_columns": list(adata.obs.columns[:10]),  # First 10 obs columns
                "var_columns": list(adata.var.columns[:10]),  # First 10 var columns
                "layers": list(adata.layers.keys()) if adata.layers else [],
                "obsm": list(adata.obsm.keys()) if adata.obsm else [],
                "memory_usage": f"{adata.X.nbytes / 1024**2:.1f} MB" if hasattr(adata.X, 'nbytes') else "Unknown",
                "metadata_keys": list(self.current_metadata.keys()),
                "processing_log": self.processing_log[-5:] if self.processing_log else [],
                "quality_metrics": metrics
            }
        else:
            # Summary for all modalities
            modality_summaries = {}
            for mod_name, adata in self.modalities.items():
                modality_summaries[mod_name] = {
                    "shape": adata.shape,
                    "obs_columns": len(adata.obs.columns),
                    "var_columns": len(adata.var.columns),
                    "layers": len(adata.layers) if adata.layers else 0,
                }
            
            summary = {
                "status": f"{len(self.modalities)} modalities loaded",
                "modalities": modality_summaries,
                "total_obs": sum(adata.shape[0] for adata in self.modalities.values()),
                "total_vars": sum(adata.shape[1] for adata in self.modalities.values()),
                "metadata_keys": list(self.current_metadata.keys()),
                "processing_log": self.processing_log[-5:] if self.processing_log else [],
            }
        
        return summary

    def set_data(self, data: pd.DataFrame, metadata: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Legacy compatibility method - converts DataFrame to modality.
        
        Args:
            data: DataFrame containing biological data
            metadata: Optional metadata dictionary
            
        Returns:
            pd.DataFrame: The processed data that was set
        """
        try:
            if data is None or not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame.")

            if data.shape[0] == 0 or data.shape[1] == 0:
                raise ValueError("DataFrame is empty.")

            # Store metadata
            self.current_metadata = metadata or {}

            # Detect appropriate adapter based on data characteristics
            n_obs, n_vars = data.shape
            if n_vars > 5000:  # High gene count suggests single-cell
                adapter_name = "transcriptomics_single_cell"
                modality_name = "legacy_single_cell"
            elif n_vars < 1000:  # Low feature count might be proteomics
                adapter_name = "proteomics_ms"
                modality_name = "legacy_proteomics"
            else:  # Middle range suggests bulk RNA-seq
                adapter_name = "transcriptomics_bulk"
                modality_name = "legacy_bulk"

            # Load as modality
            adata = self.load_modality(
                name=modality_name,
                source=data,
                adapter=adapter_name,
                validate=True,
                **self.current_metadata
            )

            # Set as current dataset
            self.set_current_dataset(modality_name)

            # Log the processing step
            self.processing_log.append(
                f"Legacy data loaded: {data.shape[0]} samples × {data.shape[1]} features"
            )

            return data

        except Exception as e:
            logger.exception(f"Error in set_data: {e}")
            raise

    # ========================================
    # METADATA AND SESSION MANAGEMENT
    # ========================================

    def store_metadata(
        self, 
        dataset_id: str, 
        metadata: Dict[str, Any],
        validation_info: Dict[str, Any] = None
    ) -> None:
        """
        Store metadata for a dataset in the metadata store.
        
        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Metadata dictionary
            validation_info: Optional validation information
        """
        self.metadata_store[dataset_id] = {
            "metadata": metadata,
            "validation": validation_info or {},
            "fetch_timestamp": datetime.now().isoformat(),
            "stored_by": "DataManagerV2"
        }
        logger.info(f"Stored metadata for dataset: {dataset_id}")

    def get_stored_metadata(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored metadata for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Optional[Dict[str, Any]]: Metadata if found, None otherwise
        """
        return self.metadata_store.get(dataset_id)

    def list_stored_datasets(self) -> List[str]:
        """
        List all datasets with stored metadata.
        
        Returns:
            List[str]: List of dataset IDs with metadata
        """
        return list(self.metadata_store.keys())

    def get_technical_summary(self) -> str:
        """
        Generate a technical summary of all analysis and processing.

        Returns:
            str: Formatted technical summary
        """
        summary = "# DataManagerV2 Technical Summary\n\n"

        # Add modality information
        if self.modalities:
            summary += "## Loaded Modalities\n\n"
            for name, adata in self.modalities.items():
                summary += f"### {name}\n"
                summary += f"- Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                summary += f"- Memory usage: {adata.X.nbytes / 1024**2:.2f} MB\n" if hasattr(adata.X, 'nbytes') else ""
                if adata.obs.columns.tolist():
                    summary += f"- Observation metadata: {', '.join(list(adata.obs.columns)[:5])}\n"
                if adata.var.columns.tolist():
                    summary += f"- Variable metadata: {', '.join(list(adata.var.columns)[:5])}\n"
                if adata.layers:
                    summary += f"- Data layers: {', '.join(list(adata.layers.keys()))}\n"
                summary += "\n"

        # Add processing log
        if self.processing_log:
            summary += "## Processing Log\n\n"
            for entry in self.processing_log:
                summary += f"- {entry}\n"
            summary += "\n"

        # Add tool usage history
        if self.tool_usage_history:
            summary += "## Tool Usage History\n\n"
            for i, entry in enumerate(self.tool_usage_history, 1):
                summary += f"### {i}. {entry['tool']} ({entry['timestamp']})\n\n"
                if entry.get("description"):
                    summary += f"{entry['description']}\n\n"
                summary += "**Parameters:**\n\n"
                for param_name, param_value in entry["parameters"].items():
                    # Format parameter value based on its type
                    if isinstance(param_value, (list, tuple)) and len(param_value) > 5:
                        param_str = f"[{', '.join(str(x) for x in param_value[:5])}...] (length: {len(param_value)})"
                    else:
                        param_str = str(param_value)
                    summary += f"- {param_name}: {param_str}\n"
                summary += "\n"

        # Add provenance information
        if self.provenance and self.provenance.activities:
            summary += "## Provenance Information\n\n"
            summary += f"- Activities: {len(self.provenance.activities)}\n"
            summary += f"- Entities: {len(self.provenance.entities)}\n"
            summary += f"- Agents: {len(self.provenance.agents)}\n"
            summary += "\n"

        return summary

    def create_data_package(self, output_dir: str = None) -> str:
        """
        Create a comprehensive data package with modalities, plots, and analysis summary.

        Args:
            output_dir: Directory to save the package (defaults to exports_dir)

        Returns:
            str: Path to the created zip file
        """
        if not self.has_data() and not self.latest_plots:
            raise ValueError("No data or plots to export")

        # Use exports directory if not specified
        if output_dir is None:
            output_dir = str(self.exports_dir)
        else:
            os.makedirs(output_dir, exist_ok=True)

        # Create a timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = Path(output_dir) / f"lobster_analysis_package_{timestamp}.zip"

        # Create a temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save technical summary
            with open(temp_path / "technical_summary.md", "w") as f:
                f.write(self.get_technical_summary())

            # Save all modalities
            if self.modalities:
                modalities_dir = temp_path / "modalities"
                modalities_dir.mkdir()
                
                for name, adata in self.modalities.items():
                    try:
                        # Save as H5AD
                        h5ad_path = modalities_dir / f"{name}.h5ad"
                        adata.write_h5ad(h5ad_path)
                        
                        # Also save as CSV for broader compatibility
                        csv_path = modalities_dir / f"{name}.csv"
                        df = pd.DataFrame(
                            adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                            index=adata.obs_names,
                            columns=adata.var_names
                        )
                        df.to_csv(csv_path)
                        
                        # Save metadata
                        metadata_path = modalities_dir / f"{name}_metadata.json"
                        modality_metadata = {
                            "shape": adata.shape,
                            "obs_columns": list(adata.obs.columns),
                            "var_columns": list(adata.var.columns),
                            "layers": list(adata.layers.keys()) if adata.layers else [],
                            "obsm": list(adata.obsm.keys()) if adata.obsm else [],
                            "uns": list(adata.uns.keys()) if adata.uns else []
                        }
                        with open(metadata_path, "w") as f:
                            json.dump(modality_metadata, f, indent=2, default=str)
                            
                    except Exception as e:
                        logger.error(f"Failed to save modality {name}: {e}")

            # Save plots
            if self.latest_plots:
                plots_dir = temp_path / "plots"
                plots_dir.mkdir()

                # Create an index of all plots
                plots_index = []

                for i, plot_entry in enumerate(self.latest_plots):
                    try:
                        plot = plot_entry["figure"]
                        plot_id = plot_entry["id"]
                        plot_title = plot_entry["title"]

                        # Create sanitized filename
                        safe_title = "".join(
                            c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                        ).rstrip()
                        safe_title = safe_title.replace(" ", "_")
                        filename_base = f"{plot_id}_{safe_title}" if safe_title else plot_id

                        # Save as HTML and PNG
                        pio.write_html(plot, plots_dir / f"{filename_base}.html")
                        try:
                            pio.write_image(plot, plots_dir / f"{filename_base}.png")
                        except Exception as e:
                            logger.warning(f"Could not save PNG for {plot_id}: {e}")

                        # Save plot metadata
                        with open(plots_dir / f"{filename_base}_info.txt", "w") as f:
                            f.write(f"ID: {plot_id}\n")
                            f.write(f"Title: {plot_title}\n")
                            f.write(f"Created: {plot_entry.get('timestamp', 'N/A')}\n")
                            f.write(f"Source: {plot_entry.get('source', 'N/A')}\n")

                        # Add to index
                        plots_index.append({
                            "id": plot_id,
                            "title": plot_title,
                            "filename": filename_base,
                            "timestamp": plot_entry.get("timestamp", "N/A"),
                            "source": plot_entry.get("source", "N/A"),
                        })
                    except Exception as e:
                        logger.error(f"Failed to save plot {plot_id}: {e}")

                # Save plots index
                with open(plots_dir / "index.json", "w") as f:
                    json.dump(plots_index, f, indent=2)

                # Create human-readable plot index
                with open(plots_dir / "README.md", "w") as f:
                    f.write("# Generated Plots\n\n")
                    for idx, plot_info in enumerate(plots_index, 1):
                        f.write(f"## {idx}. {plot_info['title']}\n\n")
                        f.write(f"- ID: {plot_info['id']}\n")
                        f.write(f"- Created: {plot_info['timestamp']}\n")
                        f.write(f"- Source: {plot_info['source']}\n")
                        f.write(f"- Files: [{plot_info['filename']}.html]({plot_info['filename']}.html), [{plot_info['filename']}.png]({plot_info['filename']}.png)\n\n")

            # Save workspace status
            with open(temp_path / "workspace_status.json", "w") as f:
                json.dump(self.get_workspace_status(), f, indent=2, default=str)

            # Save provenance if available
            if self.provenance:
                with open(temp_path / "provenance.json", "w") as f:
                    json.dump(self.provenance.to_dict(), f, indent=2, default=str)

            # Create the zip file
            with zipfile.ZipFile(zip_filename, "w") as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(temp_path)
                        zipf.write(file_path, arcname)

        logger.info(f"Data package created: {zip_filename}")
        return str(zip_filename)

    def _detect_modality_type(self, modality_name: str) -> str:
        """Detect modality type from name and registered adapters."""
        name_lower = modality_name.lower()
        
        if 'transcriptomics' in name_lower or 'rna' in name_lower or 'geo' in name_lower:
            if 'single_cell' in name_lower or 'sc' in name_lower:
                return 'single_cell_rna_seq'
            else:
                return 'bulk_rna_seq'
        elif 'proteomics' in name_lower or 'protein' in name_lower:
            if 'ms' in name_lower or 'mass' in name_lower:
                return 'mass_spectrometry_proteomics'
            else:
                return 'affinity_proteomics'
        
        return 'unknown'

    # ========================================
    # LEGACY COMPATIBILITY METHODS
    # ========================================

    def list_processed_geo_datasets(self) -> List[Dict[str, Any]]:
        """List GEO datasets with stored metadata."""
        geo_datasets = []
        
        for dataset_id, info in self.metadata_store.items():
            if dataset_id.startswith('GSE'):
                metadata = info.get('metadata', {})
                geo_datasets.append({
                    "gse_id": dataset_id,
                    "title": metadata.get('title', 'Unknown'),
                    "n_samples": len(metadata.get('samples', {})),
                    "processing_date": info.get('fetch_timestamp', 'Unknown'),
                    "modality_loaded": dataset_id.lower() in [m.lower() for m in self.modalities.keys()]
                })
        
        # Sort by processing date (most recent first)
        geo_datasets.sort(key=lambda x: x['processing_date'], reverse=True)
        return geo_datasets
