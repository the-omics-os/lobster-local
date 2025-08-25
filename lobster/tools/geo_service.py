"""
Professional GEO data service using GEOparse with modular fallback architecture.

This service provides a unified interface for downloading and processing
data from the Gene Expression Omnibus (GEO) database using a layered approach:
1. Primary: GEOparse library for standard operations
2. Fallback: Specialized downloader and parser for complex cases
3. Integration: Full DataManagerV2 compatibility with comprehensive error handling
"""

import json
import os
import re
import tarfile
import urllib.request
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union, Callable
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

# Import helper modules for fallback functionality
from lobster.tools.geo_downloader import GEODownloadManager
from lobster.tools.geo_parser import GEOParser

logger = get_logger(__name__)


# ========================================
# DATA STRUCTURES AND ENUMS
# ========================================

class GEODataSource(Enum):
    """Enumeration of data source types for GEO downloads."""
    GEOPARSE = "geoparse"
    SOFT_FILE = "soft_file" 
    SUPPLEMENTARY = "supplementary"
    TAR_ARCHIVE = "tar_archive"
    SAMPLE_MATRICES = "sample_matrices"


class GEODataType(Enum):
    """Enumeration of data types for GEO datasets."""
    SINGLE_CELL = "single_cell"
    BULK = "bulk"
    MIXED = "mixed"


@dataclass
class DownloadStrategy:
    """Configuration for GEO download preferences and fallback options."""
    prefer_geoparse: bool = True
    allow_fallback: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300
    prefer_supplementary: bool = False
    force_tar_extraction: bool = False


@dataclass
class GEOResult:
    """Result wrapper containing data, metadata, and processing information."""
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None
    source: GEODataSource = GEODataSource.GEOPARSE
    processing_info: Dict[str, Any] = None
    success: bool = False
    error_message: Optional[str] = None


class GEOServiceError(Exception):
    """Custom exception for GEO service errors."""
    pass


class GEOFallbackError(Exception):
    """Custom exception for fallback mechanism failures."""
    pass


class GEOService:
    """
    Professional service for accessing and processing GEO data using GEOparse and DataManagerV2.

    This class provides a high-level interface for working with GEO data,
    handling the downloading, parsing, and processing of datasets using GEOparse
    and storing them as modalities in the DataManagerV2 system.
    """

    def __init__(
        self, data_manager: DataManagerV2, cache_dir: Optional[str] = None, console=None
    ):
        """
        Initialize the GEO service with modular architecture.

        Args:
            data_manager: DataManagerV2 instance for storing processed data as modalities
            cache_dir: Directory to cache downloaded files
            console: Rich console instance for display (creates new if None)
        """
        if GEOparse is None:
            raise ImportError(
                "GEOparse is required but not installed. Please install with: pip install GEOparse"
            )

        self.data_manager = data_manager
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_manager.cache_dir / "geo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.console = console

        # Initialize helper services for fallback functionality
        self.geo_downloader = GEODownloadManager(
            cache_dir=str(self.cache_dir), 
            console=self.console
        )
        self.geo_parser = GEOParser()
        
        # Default download strategy
        self.download_strategy = DownloadStrategy()
        
        # Processing pipeline registry for different scenarios
        self.processing_pipelines = self._initialize_processing_pipelines()

        logger.info("GEOService initialized with modular architecture: GEOparse + fallback helpers")

    def _initialize_processing_pipelines(self) -> Dict[str, List[Callable]]:
        """
        Initialize processing pipelines for different scenarios.
        
        Returns:
            Dict[str, List[Callable]]: Pipeline functions for each scenario
        """
        return {
            "single_cell": [
                self._try_geoparse_download,
                self._try_supplementary_tar,
                self._try_sample_matrices_fallback,
                self._try_helper_download_fallback
            ],
            "bulk": [
                self._try_geoparse_download,
                self._try_series_matrix,
                self._try_supplementary_files,
                self._try_helper_download_fallback
            ],
            "mixed": [
                self._try_geoparse_download,
                self._try_supplementary_tar,
                self._try_supplementary_files,
                self._try_sample_matrices_fallback,
                self._try_helper_download_fallback
            ]
        }

    # ========================================
    # USED BY DATAEXPERT
    # ========================================

    def download_dataset(self, geo_id: str, modality_type) -> str:
        """
        Download and process a dataset using modular strategy with fallbacks (Scenarios 2 & 3).

        Args:
            geo_id: GEO accession ID

        Returns:
            str: Status message with detailed information
        """
        try:
            logger.info(f"Processing GEO query with modular strategy: {geo_id}")
            
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."

            # Check if metadata already exists (should be fetched first)
            if clean_geo_id not in self.data_manager.metadata_store:
                logger.info(f"Metadata not found, fetching first for {clean_geo_id}")
                metadata_result = self.fetch_metadata_only(clean_geo_id)
                if "Error" in metadata_result:
                    return f"Failed to fetch metadata: {metadata_result}"

            # Check if modality already exists in DataManagerV2
            modality_name = f"geo_{clean_geo_id.lower()}"
            existing_modalities = self.data_manager.list_modalities()
            if modality_name in existing_modalities:
                return f"Dataset {clean_geo_id} already loaded as modality '{modality_name}'. Use data_manager.get_modality('{modality_name}') to access it."

            #==================================================
            #==================================================
            # Use the strategic download approach
            result = self.download_with_strategy(geo_id = clean_geo_id)
            #==================================================
            #==================================================
            
            if not result.success:
                return f"Failed to download {clean_geo_id} using all available methods. Last error: {result.error_message}"
            
            # Store as modality in DataManagerV2
            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "source_metadata": result.metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "download_source": result.source.value,
                "processing_method": result.processing_info.get("method", "unknown")
            }
            enhanced_metadata.update(result.processing_info)

            # Determine appropriate adapter based on data characteristics and metadata
            cached_metadata = self.data_manager.metadata_store[clean_geo_id]['metadata']
            predicted_type = self._determine_data_type_from_metadata(cached_metadata)
            
            n_obs, n_vars = result.data.shape
            if predicted_type == 'single_cell_rna_seq' or (n_obs > 1000 and n_vars > 5000):
                adapter_name = "transcriptomics_single_cell"
            elif predicted_type == 'bulk_rna_seq' or n_obs < 200:
                adapter_name = "transcriptomics_bulk"
            else:
                adapter_name = "transcriptomics_single_cell"  # Default for GEO

            logger.info(f"Using adapter '{adapter_name}' based on predicted type '{predicted_type}' and data shape {result.data.shape}")

            # Load as modality in DataManagerV2
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=result.data,
                adapter=adapter_name,
                validate=True,
                **enhanced_metadata
            )

            # Save to workspace
            save_path = f"{clean_geo_id.lower()}_raw.h5ad"
            saved_file = self.data_manager.save_modality(modality_name, save_path)

            # Log successful download and save
            self.data_manager.log_tool_usage(
                tool_name="download_geo_dataset_strategic",
                parameters={
                    "geo_id": clean_geo_id, 
                    "download_source": result.source.value,
                    "processing_method": result.processing_info.get("method", "unknown")
                },
                description=f"Downloaded GEO dataset {clean_geo_id} using strategic approach ({result.source.value}), saved to {saved_file}",
            )

            # Auto-save current state
            self.data_manager.auto_save_state()

            return f"""Successfully downloaded and loaded GEO dataset {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
🔬 Adapter: {adapter_name} (predicted: {predicted_type.replace('_', ' ').title()})
💾 Saved to: {save_path}
🎯 Source: {result.source.value} ({result.processing_info.get('method', 'unknown')})
⚡ Ready for quality control and downstream analysis!

The dataset is now available as modality '{modality_name}' for other agents to use."""

        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            return f"Error downloading dataset: {str(e)}"
        
    # ========================================
    # MAIN PUBLIC METHODS FOR THE 6 SCENARIOS
    # ========================================        
    def download_with_strategy(
        self, 
        geo_id: str, 
        strategy: Optional[DownloadStrategy] = None,
        data_type: Optional[GEODataType] = None
    ) -> GEOResult:
        """
        Master function implementing layered download approach (Scenarios 1 & 2).
        
        Args:
            geo_id: GEO accession ID
            strategy: Download strategy configuration
            data_type: Predicted data type (auto-detected if None)
            
        Returns:
            GEOResult: Comprehensive result with data and metadata
        """
        clean_geo_id = geo_id.strip().upper()
        strategy = strategy or self.download_strategy
        
        logger.debug(f"Starting strategic download for {clean_geo_id}")
        
        try:
            # Step 1: Ensure metadata exists
            if clean_geo_id not in self.data_manager.metadata_store:
                metadata_summary = self.fetch_metadata_only(clean_geo_id)
                if "Error" in metadata_summary:
                    return GEOResult(
                        success=False,
                        error_message=f"Failed to fetch metadata: {metadata_summary}",
                        source=GEODataSource.GEOPARSE
                    )
            
            # Step 2: Detect data type if not provided
            cached_metadata = self.data_manager.metadata_store[clean_geo_id]['metadata']
            if data_type is None:
                predicted_type = self._determine_data_type_from_metadata(cached_metadata)
                if predicted_type == 'single_cell_rna_seq':
                    data_type = GEODataType.SINGLE_CELL
                elif predicted_type == 'bulk_rna_seq':
                    data_type = GEODataType.BULK
                else:
                    data_type = GEODataType.MIXED
            
            # Step 3: Route to appropriate pipeline
            pipeline_name = data_type.value
            pipeline = self.processing_pipelines.get(pipeline_name, self.processing_pipelines["mixed"])
            
            logger.debug(f"Using {pipeline_name} pipeline with {len(pipeline)} steps")
            
            # Step 4: Execute pipeline with retries
            for i, pipeline_func in enumerate(pipeline):
                logger.debug(f"Executing pipeline step {i + 1}: {pipeline_func.__name__}")
                
                try:
                    result = pipeline_func(clean_geo_id, cached_metadata)
                    if result.success:
                        logger.debug(f"Success via {pipeline_func.__name__}")
                        return result
                    else:
                        logger.warning(f"Step failed: {result.error_message}")
                except Exception as e:
                    logger.warning(f"Pipeline step {pipeline_func.__name__} failed: {e}")
                    continue

            
            return GEOResult(
                success=False,
                error_message=f"All pipeline steps failed after {strategy.max_retries} attempts",
                metadata=cached_metadata,
                source=GEODataSource.GEOPARSE
            )
            
        except Exception as e:
            logger.exception(f"Error in strategic download: {e}")
            return GEOResult(
                success=False,
                error_message=str(e),
                source=GEODataSource.GEOPARSE
            )

    def download_single_cell_sample(self, gsm_id: str) -> str:
        """
        Specialized single-cell sample downloading (Scenario 4).
        
        Args:
            gsm_id: GEO sample ID (e.g., GSM123456)
            
        Returns:
            str: Status message
        """
        try:
            logger.debug(f"Downloading single-cell sample: {gsm_id}")
            
            # Clean sample ID
            clean_gsm_id = gsm_id.strip().upper()
            if not clean_gsm_id.startswith('GSM'):
                return f"Invalid sample ID format: {gsm_id}. Must be a GSM accession (e.g., GSM123456)."
            
            # Try GEOparse first
            try:
                gsm = GEOparse.get_GEO(geo=clean_gsm_id, destdir=str(self.cache_dir))
                
                # Check for 10X format supplementary files
                if hasattr(gsm, "metadata") and "supplementary_file" in gsm.metadata:
                    suppl_files = gsm.metadata["supplementary_file"]
                    if not isinstance(suppl_files, list):
                        suppl_files = [suppl_files]
                    
                    # Look for 10X format files
                    for suppl_url in suppl_files:
                        if any(pattern in suppl_url.lower() for pattern in 
                               ['matrix.mtx', 'barcodes.tsv', 'features.tsv', '.h5']):
                            
                            # Download and parse 10X data
                            matrix = self._download_and_parse_10x_sample(suppl_url, clean_gsm_id)
                            if matrix is not None:
                                return self._store_single_sample_as_modality(clean_gsm_id, matrix, gsm)
                
                # Fallback to expression table
                if hasattr(gsm, "table") and gsm.table is not None:
                    matrix = gsm.table
                    return self._store_single_sample_as_modality(clean_gsm_id, matrix, gsm)
                    
            except Exception as e:
                logger.warning(f"GEOparse failed for {clean_gsm_id}: {e}")
            
            # Fallback to helper downloader
            try:
                return self._download_sample_with_helpers(clean_gsm_id)
            except Exception as e:
                logger.error(f"Helper download failed for {clean_gsm_id}: {e}")
                return f"Failed to download sample {clean_gsm_id}: {str(e)}"
                
        except Exception as e:
            logger.exception(f"Error downloading single-cell sample {gsm_id}: {e}")
            return f"Error downloading single-cell sample {gsm_id}: {str(e)}"

    def download_bulk_dataset(self, geo_id: str, prefer_series_matrix: bool = True) -> str:
        """
        Enhanced bulk data downloading (Scenario 5).
        
        Args:
            geo_id: GEO accession ID
            prefer_series_matrix: Whether to prefer series matrix over supplementary files
            
        Returns:
            str: Status message
        """
        try:
            logger.debug(f"Downloading bulk dataset: {geo_id}")
            
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."
            
            # Ensure metadata exists
            if clean_geo_id not in self.data_manager.metadata_store:
                metadata_summary = self.fetch_metadata_only(clean_geo_id)
                if "Error" in metadata_summary:
                    return f"Failed to fetch metadata: {metadata_summary}"
            
            # Strategy for bulk data
            strategy = DownloadStrategy(
                prefer_geoparse=True,
                prefer_supplementary=not prefer_series_matrix,
                max_retries=2
            )
            
            result = self.download_with_strategy(clean_geo_id, strategy, GEODataType.BULK)
            
            if result.success:
                # Store as bulk RNA-seq modality
                modality_name = f"geo_{clean_geo_id.lower()}_bulk"
                adata = self.data_manager.load_modality(
                    name=modality_name,
                    source=result.data,
                    adapter="transcriptomics_bulk",
                    validate=True,
                    **result.processing_info
                )
                
                # Save to workspace
                save_path = f"{clean_geo_id.lower()}_bulk_raw.h5ad"
                saved_file = self.data_manager.save_modality(modality_name, save_path)
                
                return f"""Successfully downloaded bulk dataset {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} samples × {adata.n_vars} genes)
🔬 Adapter: transcriptomics_bulk  
💾 Saved to: {save_path}
📈 Ready for bulk RNA-seq analysis!"""
            else:
                return f"Failed to download bulk dataset {clean_geo_id}: {result.error_message}"
                
        except Exception as e:
            logger.exception(f"Error downloading bulk dataset {geo_id}: {e}")
            return f"Error downloading bulk dataset {geo_id}: {str(e)}"

    def process_supplementary_tar_files(self, geo_id: str) -> str:
        """
        TAR file processing fallback for single-cell data (Scenario 6).
        
        Args:
            geo_id: GEO accession ID
            
        Returns:
            str: Status message
        """
        try:
            logger.debug(f"Processing supplementary TAR files for: {geo_id}")
            
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."
            
            # Use helper downloader for TAR processing
            soft_file, data_sources = self.geo_downloader.download_geo_data(clean_geo_id)
            
            if not data_sources:
                return f"No TAR files found for {clean_geo_id}"
            
            # Process TAR files using helper parser
            if isinstance(data_sources, dict):
                if 'tar' in data_sources:
                    matrix = self.geo_parser.parse_supplementary_file(data_sources['tar'])
                elif 'tar_dir' in data_sources:
                    # Process directory with multiple files
                    matrix = self._process_tar_directory_with_helpers(data_sources['tar_dir'])
                else:
                    return f"No TAR data found in downloaded files for {clean_geo_id}"
            else:
                # Single file
                matrix = self.geo_parser.parse_supplementary_file(data_sources)
            
            if matrix is None or matrix.empty:
                return f"Failed to parse TAR files for {clean_geo_id}"
            
            # Store as single-cell modality
            modality_name = f"geo_{clean_geo_id.lower()}_tar"
            
            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "data_source": "tar_supplementary",
                "processing_date": pd.Timestamp.now().isoformat(),
                "parser_version": "helper_parser"
            }
            
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=matrix,
                adapter="transcriptomics_single_cell",
                validate=True,
                **enhanced_metadata
            )
            
            # Save to workspace
            save_path = f"{clean_geo_id.lower()}_tar_processed.h5ad"
            saved_file = self.data_manager.save_modality(modality_name, save_path)
            
            self.data_manager.log_tool_usage(
                tool_name="process_supplementary_tar_files",
                parameters={"geo_id": clean_geo_id},
                description=f"Processed TAR supplementary files for {clean_geo_id}"
            )
            
            return f"""Successfully processed TAR supplementary files for {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} cells × {adata.n_vars} genes)
🔬 Source: TAR supplementary files
💾 Saved to: {save_path}
📈 Ready for single-cell analysis!"""
            
        except Exception as e:
            logger.exception(f"Error processing TAR files for {geo_id}: {e}")
            return f"Error processing TAR files for {geo_id}: {str(e)}"

    # ========================================================================================================================
    # ========================================================================================================================
    # ========================================================================================================================
    # ========================================================================================================================
    # PIPELINE STEP FUNCTIONS
    # ========================================================================================================================
    # ========================================================================================================================
    # ========================================================================================================================
    # ========================================================================================================================


    def _try_geoparse_download(self, geo_id: str, metadata: Dict[str, Any], use_intersecting_genes_only: bool = True) -> GEOResult:
        """Pipeline step: Try standard GEOparse download with proper single-cell/bulk handling."""
        try:
            logger.info(f"Trying GEOparse download for {geo_id}")
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))
            
            # Determine data type from metadata
            data_type = self._determine_data_type_from_metadata(metadata)
            is_single_cell = data_type == 'single_cell_rna_seq'
            
            # Try sample matrices
            sample_info = self._get_sample_info(gse)
            if sample_info:
                sample_matrices = self._download_sample_matrices(sample_info, geo_id)
                validated_matrices = self._validate_matrices(sample_matrices)
                
                if validated_matrices:
                    if is_single_cell and len(validated_matrices) > 1:
                        # For single-cell with multiple samples: store individually first
                        stored_samples = self._store_samples_as_anndata(validated_matrices, geo_id, metadata)
                        
                        if stored_samples:
                            # Immediately concatenate for a complete dataset
                            concatenated_result = self._concatenate_stored_samples(
                                geo_id, stored_samples, use_intersecting_genes_only
                            )
                            
                            if concatenated_result is not None:
                                return GEOResult(
                                    data=concatenated_result,
                                    metadata=metadata,
                                    source=GEODataSource.SAMPLE_MATRICES,
                                    processing_info={
                                        "method": "geoparse_samples_concatenated", 
                                        "n_samples": len(validated_matrices),
                                        "stored_sample_ids": stored_samples,
                                        "use_intersecting_genes_only": use_intersecting_genes_only,
                                        "batch_info": {gsm_id: gsm_id for gsm_id in validated_matrices.keys()},
                                        "note": "Single-cell samples concatenated with batch tracking"
                                    },
                                    success=True
                                )
                    else:
                        # For bulk RNA-seq or single sample: concatenate directly
                        combined_matrix = self._concatenate_matrices(
                            validated_matrices, geo_id, use_intersecting_genes_only
                        )
                        
                        if combined_matrix is not None:
                            # Add batch information to the matrix
                            if 'batch' not in combined_matrix.columns:
                                batch_info = []
                                for gsm_id in validated_matrices.keys():
                                    n_cells = len([idx for idx in combined_matrix.index if idx.startswith(gsm_id)])
                                    batch_info.extend([gsm_id] * n_cells)
                                combined_matrix['batch'] = batch_info
                            
                            return GEOResult(
                                data=combined_matrix,
                                metadata=metadata,
                                source=GEODataSource.SAMPLE_MATRICES,
                                processing_info={
                                    "method": "geoparse_samples_direct", 
                                    "n_samples": len(validated_matrices),
                                    "use_intersecting_genes_only": use_intersecting_genes_only,
                                    "data_type": "bulk" if not is_single_cell else "single_cell_single_sample",
                                    "batch_info": {gsm_id: gsm_id for gsm_id in validated_matrices.keys()}
                                },
                                success=True
                            )
                    
            # Try supplementary files as fallback
            data = self._process_supplementary_files(gse, geo_id)
            if data is not None and not data.empty:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.GEOPARSE,
                    processing_info={"method": "geoparse_supplementary", "n_samples": len(gse.gsms) if hasattr(gse, "gsms") else 0},
                    success=True
                )
            
            return GEOResult(success=False, error_message="GEOparse could not find usable data")
            
        except Exception as e:
            logger.warning(f"GEOparse download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_tar(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Try TAR supplementary files."""
        try:
            logger.debug(f"Trying TAR supplementary files for {geo_id}")
            soft_file, data_sources = self.geo_downloader.download_geo_data(geo_id)
            
            if isinstance(data_sources, dict) and 'tar' in data_sources:
                matrix = self.geo_parser.parse_supplementary_file(data_sources['tar'])
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.TAR_ARCHIVE,
                        processing_info={"method": "helper_tar", "source_file": str(data_sources['tar'])},
                        success=True
                    )
            
            return GEOResult(success=False, error_message="No TAR files found or parsable")
            
        except Exception as e:
            logger.warning(f"TAR supplementary download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_series_matrix(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Try series matrix files (for bulk data)."""
        try:
            logger.debug(f"Trying series matrix for {geo_id}")
            # Use GEOparse to get series matrix
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))
            
            # Check if there's a series matrix
            if hasattr(gse, 'table') and gse.table is not None:
                return GEOResult(
                    data=gse.table,
                    metadata=metadata,
                    source=GEODataSource.GEOPARSE,
                    processing_info={"method": "series_matrix"},
                    success=True
                )
            
            return GEOResult(success=False, error_message="No series matrix available")
            
        except Exception as e:
            logger.warning(f"Series matrix download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_files(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Try supplementary files (non-TAR)."""
        try:
            logger.debug(f"Trying supplementary files for {geo_id}")
            soft_file, data_sources = self.geo_downloader.download_geo_data(geo_id)
            
            if isinstance(data_sources, dict) and 'supplementary' in data_sources:
                matrix = self.geo_parser.parse_supplementary_file(data_sources['supplementary'])
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={"method": "helper_supplementary", "source_file": str(data_sources['supplementary'])},
                        success=True
                    )
            
            return GEOResult(success=False, error_message="No supplementary files found or parsable")
            
        except Exception as e:
            logger.warning(f"Supplementary files download failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_sample_matrices_fallback(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Try individual sample matrices as fallback."""
        try:
            logger.debug(f"Trying sample matrices fallback for {geo_id}")
            # This is already handled in _try_geoparse_download, so this is a no-op
            return GEOResult(success=False, error_message="Sample matrices already tried in GEOparse step")
            
        except Exception as e:
            logger.warning(f"Sample matrices fallback failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_helper_download_fallback(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Pipeline step: Final fallback using helper downloader."""
        try:
            logger.debug(f"Trying helper download fallback for {geo_id}")
            soft_file, data_sources = self.geo_downloader.download_geo_data(geo_id)
            
            if not data_sources:
                return GEOResult(success=False, error_message="No data sources found")
            
            # Try all available data sources
            if isinstance(data_sources, dict):
                for source_name, source_path in data_sources.items():
                    try:
                        matrix = self.geo_parser.parse_supplementary_file(source_path)
                        if matrix is not None and not matrix.empty:
                            return GEOResult(
                                data=matrix,
                                metadata=metadata,
                                source=GEODataSource.SOFT_FILE,
                                processing_info={"method": f"helper_fallback_{source_name}", "source_file": str(source_path)},
                                success=True
                            )
                    except Exception as e:
                        logger.warning(f"Failed to parse {source_name}: {e}")
                        continue
            else:
                # Single file
                matrix = self.geo_parser.parse_supplementary_file(data_sources)
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SOFT_FILE,
                        processing_info={"method": "helper_fallback_single", "source_file": str(data_sources)},
                        success=True
                    )
            
            return GEOResult(success=False, error_message="All helper download attempts failed")
            
        except Exception as e:
            logger.warning(f"Helper download fallback failed: {e}")
            return GEOResult(success=False, error_message=str(e))

    # ========================================
    # HELPER INTEGRATION METHODS
    # ========================================

    # def _download_and_parse_10x_sample(self, suppl_url: str, gsm_id: str) -> Optional[pd.DataFrame]:
    #     """Download and parse 10X format single-cell sample."""
    #     try:
    #         # Download using helper downloader for better progress tracking
    #         local_file = self.cache_dir / f"{gsm_id}_10x_data"
    #         if self.geo_downloader.download_file(suppl_url, local_file):
    #             return self.geo_parser.parse_supplementary_file(local_file)
    #         return None
    #     except Exception as e:
    #         logger.error(f"Error downloading 10X sample {gsm_id}: {e}")
    #         return None

    def _store_single_sample_as_modality(self, gsm_id: str, matrix: pd.DataFrame, gsm) -> str:
        """Store single sample as modality in DataManagerV2."""
        try:
            modality_name = f"geo_sample_{gsm_id.lower()}"
            
            # Extract sample metadata
            sample_metadata = {}
            if hasattr(gsm, "metadata"):
                for key, value in gsm.metadata.items():
                    if isinstance(value, list) and len(value) == 1:
                        sample_metadata[key] = value[0]
                    else:
                        sample_metadata[key] = value
            
            enhanced_metadata = {
                "sample_id": gsm_id,
                "dataset_type": "GEO_Sample",
                "sample_metadata": sample_metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "data_source": "single_sample"
            }
            
            # Determine adapter based on data characteristics
            n_obs, n_vars = matrix.shape
            if n_vars > 5000:  # High gene count suggests single-cell
                adapter_name = "transcriptomics_single_cell"
            else:
                adapter_name = "transcriptomics_bulk"
            
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=matrix,
                adapter=adapter_name,
                validate=True,
                **enhanced_metadata
            )
            
            # Save to workspace
            save_path = f"{gsm_id.lower()}_sample.h5ad"
            saved_file = self.data_manager.save_modality(modality_name, save_path)
            
            return f"""Successfully downloaded single-cell sample {gsm_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} cells × {adata.n_vars} genes)
🔬 Adapter: {adapter_name}
💾 Saved to: {save_path}
📈 Ready for single-cell analysis!"""
            
        except Exception as e:
            logger.error(f"Error storing sample {gsm_id}: {e}")
            return f"Error storing sample {gsm_id}: {str(e)}"

    def _download_sample_with_helpers(self, gsm_id: str) -> str:
        """Download sample using helper services."""
        try:
            # Use helper downloader for sample-level downloads
            # This would need to be implemented in the helper downloader
            # For now, return a not implemented message
            return f"Helper-based sample download for {gsm_id} not yet implemented. Please use GEOparse method."
            
        except Exception as e:
            logger.error(f"Error downloading sample with helpers {gsm_id}: {e}")
            return f"Error downloading sample with helpers {gsm_id}: {str(e)}"

    def _process_tar_directory_with_helpers(self, tar_dir: Path) -> Optional[pd.DataFrame]:
        """Process TAR directory using helper parser."""
        try:
            # Look for expression files in the directory
            expression_files = []
            for file_path in tar_dir.rglob("*"):
                if file_path.is_file() and any(
                    ext in file_path.name.lower()
                    for ext in [".txt", ".csv", ".tsv", ".gz", ".h5", ".mtx"]
                ):
                    if file_path.stat().st_size > 10000:  # > 10KB
                        expression_files.append(file_path)
            
            # Sort by size, try largest first
            expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            
            for file_path in expression_files[:5]:  # Try top 5 files
                try:
                    matrix = self.geo_parser.parse_supplementary_file(file_path)
                    if matrix is not None and not matrix.empty:
                        logger.debug(f"Successfully parsed TAR file: {file_path.name}")
                        return matrix
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path.name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing TAR directory: {e}")
            return None

    def fetch_metadata_only(self, geo_id: str) -> str:
        """
        Fetch and validate GEO metadata with fallback mechanisms (Scenario 1).
        
        This function first tries GEOparse, then falls back to helper services
        if needed, storing the metadata in data_manager for user review.
        
        Args:
            geo_id: GEO accession ID (e.g., GSE194247)
            
        Returns:
            str: Formatted metadata summary for user review
        """
        try:
            logger.info(f"Fetching metadata for GEO ID: {geo_id}")
            
            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith('GSE'):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession (e.g., GSE194247)."
            
            # Check if already cached
            if clean_geo_id in self.data_manager.metadata_store:
                summary = self._format_metadata_summary(
                    clean_geo_id,
                    self.data_manager.metadata_store[clean_geo_id]
                )
                return summary
            
            # Primary approach: GEOparse
            metadata = None
            validation_result = None
            data_source = "geoparse"
            
            try:
                logger.debug(f"Downloading SOFT metadata for {clean_geo_id} using GEOparse...")
                gse = GEOparse.get_GEO(geo=clean_geo_id, destdir=str(self.cache_dir))
                metadata = self._extract_metadata(gse)
                logger.debug(f"Successfully extracted metadata using GEOparse for {clean_geo_id}")
                
            except Exception as geoparse_error:
                import time
                logger.warning(f"GEOparse metadata fetch failed: {geoparse_error}")
                time.sleep(5)
                # Fallback: Use helper downloader to get SOFT file
                try:
                    logger.debug(f"Falling back to helper services for metadata: {clean_geo_id}")
                    soft_file, _ = self.geo_downloader.download_geo_data(clean_geo_id)
                    
                    if soft_file and soft_file.exists():
                        # Parse SOFT file using helper parser
                        _, soft_metadata = self.geo_parser.parse_soft_file(soft_file)
                        if soft_metadata:
                            metadata = soft_metadata
                            data_source = "helper_soft_file"
                            logger.debug(f"Successfully extracted metadata using helper parser for {clean_geo_id}")
                        else:
                            raise GEOFallbackError("Helper parser could not extract metadata from SOFT file")
                    else:
                        raise GEOFallbackError("Helper downloader could not download SOFT file")
                        
                except Exception as helper_error:
                    logger.error(f"Helper metadata fetch also failed: {helper_error}")
                    return f"Failed to fetch metadata for {clean_geo_id}. Both GEOparse ({geoparse_error}) and helper services ({helper_error}) failed."
            
            if not metadata:
                return f"No metadata could be extracted for {clean_geo_id}"
            
            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)
            
            # Store metadata in data_manager for future use
            self.data_manager.metadata_store[clean_geo_id] = {
                'metadata': metadata,
                'validation': validation_result,
                'fetch_timestamp': pd.Timestamp.now().isoformat(),
                'data_source': data_source
            }
            
            # Log the metadata fetch operation
            self.data_manager.log_tool_usage(
                tool_name="fetch_geo_metadata_with_fallback",
                parameters={"geo_id": clean_geo_id, "data_source": data_source},
                description=f"Fetched metadata for GEO dataset {clean_geo_id} using {data_source}"
            )
            
            # Format comprehensive metadata summary
            summary = self._format_metadata_summary(clean_geo_id, metadata, validation_result)
            
            logger.debug(f"Successfully fetched and validated metadata for {clean_geo_id} using {data_source}")
            return summary
            
        except Exception as e:
            logger.exception(f"Error fetching metadata for {geo_id}: {e}")
            return f"Error fetching metadata for {geo_id}: {str(e)}"



    def _process_supplementary_files(self, gse, gse_id: str) -> Optional[pd.DataFrame]:
        """
        Process supplementary files (TAR archives, etc.) to extract expression data.

        Args:
            gse: GEOparse GSE object
            gse_id: GEO series ID

        Returns:
            DataFrame: Combined expression matrix from supplementary files or None
        """
        try:
            if not hasattr(gse, "metadata") or "supplementary_file" not in gse.metadata:
                logger.debug(f"No supplementary files found for {gse_id}")
                return None

            suppl_files = gse.metadata["supplementary_file"]
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            logger.debug(f"Found {len(suppl_files)} supplementary files for {gse_id}")

            # Look for TAR files first (most common for expression data)
            tar_files = [f for f in suppl_files if f.lower().endswith(".tar")]

            if tar_files:
                logger.debug(f"Processing TAR file: {tar_files[0]}")
                return self._process_tar_file(tar_files[0], gse_id)

            # Look for other expression data files
            expression_files = [
                f
                for f in suppl_files
                if any(
                    ext in f.lower()
                    for ext in [".txt.gz", ".csv.gz", ".tsv.gz", ".h5", ".h5ad"]
                )
            ]

            if expression_files:
                logger.debug(f"Processing expression file: {expression_files[0]}")
                return self._download_and_parse_file(expression_files[0], gse_id)

            logger.warning(
                f"No suitable expression files found in supplementary files for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing supplementary files: {e}")
            return None

    def _process_tar_file(self, tar_url: str, gse_id: str) -> Optional[pd.DataFrame]:
        """
        Download and process a TAR file containing expression data.

        Args:
            tar_url: URL to TAR file
            gse_id: GEO series ID

        Returns:
            DataFrame: Combined expression matrix or None
        """
        try:
            # Download TAR file
            tar_file_path = self.cache_dir / f"{gse_id}_RAW.tar"

            if not tar_file_path.exists():
                logger.debug(f"Downloading TAR file from: {tar_url}")
                urllib.request.urlretrieve(tar_url, tar_file_path)
                logger.debug(f"Downloaded TAR file: {tar_file_path}")
            else:
                logger.debug(f"Using cached TAR file: {tar_file_path}")

            # Extract TAR file
            extract_dir = self.cache_dir / f"{gse_id}_extracted"
            if not extract_dir.exists():
                logger.info(f"Extracting TAR file to: {extract_dir}")
                extract_dir.mkdir(exist_ok=True)

                with tarfile.open(tar_file_path, "r") as tar:
                    # Security check for path traversal
                    def is_safe_member(member):
                        member_path = Path(member.name)
                        try:
                            target_path = (extract_dir / member_path).resolve()
                            common_path = Path(
                                os.path.commonpath([extract_dir.resolve(), target_path])
                            )
                            return common_path == extract_dir.resolve()
                        except (ValueError, RuntimeError):
                            return False

                    safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                    tar.extractall(path=extract_dir, members=safe_members)

                logger.info(f"Extracted {len(safe_members)} files from TAR")

            # Process nested archives and find expression data
            nested_extract_dir = self.cache_dir / f"{gse_id}_nested_extracted"
            nested_extract_dir.mkdir(exist_ok=True)

            # Extract any nested TAR.GZ files
            nested_archives = list(extract_dir.glob("*.tar.gz"))
            if nested_archives:
                logger.info(f"Found {len(nested_archives)} nested TAR.GZ files")

                all_matrices = []
                for archive_path in nested_archives:
                    try:
                        sample_id = archive_path.stem.split(".")[
                            0
                        ]  # Extract sample ID from filename
                        sample_extract_dir = nested_extract_dir / sample_id
                        sample_extract_dir.mkdir(exist_ok=True)

                        logger.info(f"Extracting nested archive: {archive_path.name}")
                        with tarfile.open(archive_path, "r:gz") as nested_tar:
                            nested_tar.extractall(path=sample_extract_dir)

                        # Try to parse 10X Genomics format data
                        matrix = self._parse_10x_data(sample_extract_dir, sample_id)
                        if matrix is not None:
                            all_matrices.append(matrix)
                            logger.info(
                                f"Successfully parsed 10X data for {sample_id}: {matrix.shape}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract/parse {archive_path.name}: {e}"
                        )
                        continue

                if all_matrices:
                    # Concatenate all sample matrices
                    logger.info(f"Concatenating {len(all_matrices)} 10X matrices")
                    combined_matrix = pd.concat(all_matrices, axis=0, sort=False)
                    logger.info(f"Combined matrix shape: {combined_matrix.shape}")
                    return combined_matrix

            # Fallback: look for regular expression files
            expression_files = []
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file() and any(
                    ext in file_path.name.lower()
                    for ext in [".txt", ".csv", ".tsv", ".gz"]
                ):
                    # Skip small files (likely barcodes or metadata)
                    if file_path.stat().st_size > 100000:  # > 100KB
                        expression_files.append(file_path)

            if expression_files:
                logger.info(f"Found {len(expression_files)} potential expression files")

                # Try to parse the largest file first (likely the main expression matrix)
                expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)

                for file_path in expression_files[:3]:  # Try top 3 largest files
                    try:
                        logger.info(f"Attempting to parse: {file_path.name}")
                        matrix = self._parse_expression_file(file_path)
                        if (
                            matrix is not None
                            and matrix.shape[0] > 0
                            and matrix.shape[1] > 0
                        ):
                            logger.info(
                                f"Successfully parsed expression matrix: {matrix.shape}"
                            )
                            return matrix
                    except Exception as e:
                        logger.warning(f"Failed to parse {file_path.name}: {e}")
                        continue

            logger.warning(
                f"Could not parse any expression files from TAR for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing TAR file: {e}")
            return None

    def _download_and_parse_file(
        self, file_url: str, gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse a single expression file.

        Args:
            file_url: URL to expression file
            gse_id: GEO series ID

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            file_name = file_url.split("/")[-1]
            local_file = self.cache_dir / f"{gse_id}_{file_name}"

            if not local_file.exists():
                logger.info(f"Downloading file: {file_url}")
                urllib.request.urlretrieve(file_url, local_file)

            return self._parse_expression_file(local_file)

        except Exception as e:
            logger.error(f"Error downloading and parsing file: {e}")
            return None

    def _parse_10x_data(
        self, extract_dir: Path, sample_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Parse 10X Genomics format data (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz).

        Args:
            extract_dir: Directory containing 10X format files
            sample_id: Sample identifier for cell naming

        Returns:
            DataFrame: Expression matrix with cells as rows, genes as columns
        """
        try:
            logger.info(f"Parsing 10X data for {sample_id} in {extract_dir}")

            # Look for 10X files in the directory and subdirectories
            matrix_file = None
            barcodes_file = None
            features_file = None

            # Search recursively for 10X files
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file():
                    name_lower = file_path.name.lower()
                    if "matrix.mtx" in name_lower:
                        matrix_file = file_path
                    elif "barcode" in name_lower or "barcodes" in name_lower:
                        barcodes_file = file_path
                    elif "feature" in name_lower or "genes" in name_lower:
                        features_file = file_path

            if not matrix_file:
                logger.warning(f"No matrix.mtx file found for {sample_id}")
                return None

            logger.info(
                f"Found 10X files - Matrix: {matrix_file.name}, Barcodes: {barcodes_file.name if barcodes_file else 'None'}, Features: {features_file.name if features_file else 'None'}"
            )

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
                from scipy.sparse import csr_matrix
            except ImportError:
                logger.error("scipy is required for parsing 10X data but not available")
                return None

            # Read the sparse matrix
            logger.info(f"Reading sparse matrix from {matrix_file}")
            if matrix_file.name.endswith(".gz"):
                import gzip

                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)

            # Convert to dense format and transpose (10X format is genes x cells, we want cells x genes)
            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix

            # Transpose so that cells are rows and genes are columns
            matrix_dense = matrix_dense.T

            logger.info(f"Matrix shape after transpose: {matrix_dense.shape}")

            # Read barcodes (cell identifiers)
            cell_ids = []
            if barcodes_file and barcodes_file.exists():
                try:
                    if barcodes_file.name.endswith(".gz"):
                        with gzip.open(barcodes_file, "rt") as f:
                            cell_ids = [line.strip() for line in f]
                    else:
                        with open(barcodes_file, "r") as f:
                            cell_ids = [line.strip() for line in f]
                    logger.info(f"Read {len(cell_ids)} cell barcodes")
                except Exception as e:
                    logger.warning(f"Error reading barcodes file: {e}")

            # Read features/genes
            gene_ids = []
            gene_names = []
            if features_file and features_file.exists():
                try:
                    if features_file.name.endswith(".gz"):
                        with gzip.open(features_file, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(features_file, "r") as f:
                            lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[1])
                        elif len(parts) == 1:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[0])

                    logger.info(f"Read {len(gene_ids)} gene features")
                except Exception as e:
                    logger.warning(f"Error reading features file: {e}")

            # Create appropriate cell and gene names
            if not cell_ids:
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]
            else:
                # Add sample prefix to cell IDs
                cell_ids = [f"{sample_id}_{cell_id}" for cell_id in cell_ids]

            if not gene_names:
                if gene_ids:
                    gene_names = gene_ids
                else:
                    gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Ensure we have the right number of identifiers
            if len(cell_ids) != matrix_dense.shape[0]:
                logger.warning(
                    f"Cell ID count mismatch: {len(cell_ids)} vs {matrix_dense.shape[0]}"
                )
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]

            if len(gene_names) != matrix_dense.shape[1]:
                logger.warning(
                    f"Gene name count mismatch: {len(gene_names)} vs {matrix_dense.shape[1]}"
                )
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)

            logger.info(
                f"Successfully created 10X DataFrame for {sample_id}: {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error parsing 10X data for {sample_id}: {e}")
            return None

    def _parse_expression_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Parse an expression data file.

        Args:
            file_path: Path to expression file

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Parsing expression file: {file_path}")

            # Check if it's a Matrix Market format file (.mtx or .mtx.gz)
            if "matrix.mtx" in file_path.name.lower():
                logger.info(f"Detected Matrix Market format file: {file_path.name}")
                return self._parse_matrix_market_file(file_path)

            # Handle regular text files
            # Handle compressed files
            if file_path.name.endswith(".gz"):
                opener = lambda: pd.read_csv(
                    file_path,
                    compression="gzip",
                    sep="\t",
                    index_col=0,
                    low_memory=False,
                )
            else:
                opener = lambda: pd.read_csv(
                    file_path, sep="\t", index_col=0, low_memory=False
                )

            # Try tab-separated first
            try:
                df = opener()
                if df.shape[0] > 0 and df.shape[1] > 0:
                    logger.info(f"Successfully parsed as tab-separated: {df.shape}")
                    return df
            except:
                pass

            # Try comma-separated
            try:
                if file_path.name.endswith(".gz"):
                    df = pd.read_csv(
                        file_path,
                        compression="gzip",
                        sep=",",
                        index_col=0,
                        low_memory=False,
                    )
                else:
                    df = pd.read_csv(file_path, sep=",", index_col=0, low_memory=False)

                if df.shape[0] > 0 and df.shape[1] > 0:
                    logger.info(f"Successfully parsed as comma-separated: {df.shape}")
                    return df
            except:
                pass

            logger.warning(f"Could not parse expression file: {file_path}")
            return None

        except Exception as e:
            logger.error(f"Error parsing expression file {file_path}: {e}")
            return None

    def _parse_matrix_market_file(self, matrix_file: Path) -> Optional[pd.DataFrame]:
        """
        Parse a Matrix Market format file (.mtx or .mtx.gz) and look for associated barcodes/features.

        Args:
            matrix_file: Path to Matrix Market format file

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Parsing Matrix Market file: {matrix_file}")

            # Import scipy for sparse matrix handling
            try:
                import scipy.io as sio
                from scipy.sparse import csr_matrix
            except ImportError:
                logger.error(
                    "scipy is required for parsing Matrix Market files but not available"
                )
                return None

            # Read the sparse matrix
            if matrix_file.name.endswith(".gz"):
                import gzip

                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)

            # Convert to dense format and transpose (Matrix Market format is often genes x cells, we want cells x genes)
            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix

            # Transpose so that cells are rows and genes are columns
            matrix_dense = matrix_dense.T

            logger.info(f"Matrix shape after transpose: {matrix_dense.shape}")

            # Look for associated barcodes and features files in the same directory
            matrix_dir = matrix_file.parent
            sample_id = (
                matrix_file.name.replace("_matrix.mtx.gz", "")
                .replace("_matrix.mtx", "")
                .replace("matrix.mtx.gz", "")
                .replace("matrix.mtx", "")
            )
            if not sample_id:
                sample_id = matrix_dir.name

            logger.info(f"Looking for barcodes/features files for sample: {sample_id}")

            # Find barcodes and features files
            barcodes_file = None
            features_file = None

            # Common patterns for barcodes and features files
            barcode_patterns = [
                "barcodes.tsv.gz",
                "barcodes.tsv",
                f"{sample_id}_barcodes.tsv.gz",
                f"{sample_id}_barcodes.tsv",
            ]
            feature_patterns = [
                "features.tsv.gz",
                "features.tsv",
                "genes.tsv.gz",
                "genes.tsv",
                f"{sample_id}_features.tsv.gz",
                f"{sample_id}_features.tsv",
                f"{sample_id}_genes.tsv.gz",
                f"{sample_id}_genes.tsv",
            ]

            # Search in the same directory
            for file_path in matrix_dir.glob("*"):
                if file_path.is_file():
                    name_lower = file_path.name.lower()
                    if any(
                        pattern.lower() in name_lower for pattern in barcode_patterns
                    ):
                        barcodes_file = file_path
                    elif any(
                        pattern.lower() in name_lower for pattern in feature_patterns
                    ):
                        features_file = file_path

            logger.info(
                f"Found files - Barcodes: {barcodes_file.name if barcodes_file else 'None'}, Features: {features_file.name if features_file else 'None'}"
            )

            # Read barcodes (cell identifiers)
            cell_ids = []
            if barcodes_file and barcodes_file.exists():
                try:
                    if barcodes_file.name.endswith(".gz"):
                        import gzip

                        with gzip.open(barcodes_file, "rt") as f:
                            cell_ids = [line.strip() for line in f]
                    else:
                        with open(barcodes_file, "r") as f:
                            cell_ids = [line.strip() for line in f]
                    logger.info(f"Read {len(cell_ids)} cell barcodes")
                except Exception as e:
                    logger.warning(f"Error reading barcodes file: {e}")

            # Read features/genes
            gene_ids = []
            gene_names = []
            if features_file and features_file.exists():
                try:
                    if features_file.name.endswith(".gz"):
                        import gzip

                        with gzip.open(features_file, "rt") as f:
                            lines = f.readlines()
                    else:
                        with open(features_file, "r") as f:
                            lines = f.readlines()

                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[1])
                        elif len(parts) == 1:
                            gene_ids.append(parts[0])
                            gene_names.append(parts[0])

                    logger.info(f"Read {len(gene_ids)} gene features")
                except Exception as e:
                    logger.warning(f"Error reading features file: {e}")

            # Create appropriate cell and gene names
            if not cell_ids:
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]
            else:
                # Add sample prefix to cell IDs
                cell_ids = [f"{sample_id}_{cell_id}" for cell_id in cell_ids]

            if not gene_names:
                if gene_ids:
                    gene_names = gene_ids
                else:
                    gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Ensure we have the right number of identifiers
            if len(cell_ids) != matrix_dense.shape[0]:
                logger.warning(
                    f"Cell ID count mismatch: {len(cell_ids)} vs {matrix_dense.shape[0]}"
                )
                cell_ids = [
                    f"{sample_id}_cell_{i}" for i in range(matrix_dense.shape[0])
                ]

            if len(gene_names) != matrix_dense.shape[1]:
                logger.warning(
                    f"Gene name count mismatch: {len(gene_names)} vs {matrix_dense.shape[1]}"
                )
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            # Create DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)

            logger.info(
                f"Successfully created Matrix Market DataFrame for {sample_id}: {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Error parsing Matrix Market file {matrix_file}: {e}")
            return None

    def _extract_metadata(self, gse) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from GEOparse GSE object.

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Extracted metadata
        """
        try:
            metadata = {}

            # Basic metadata from GEOparse
            if hasattr(gse, "metadata"):
                for key, value in gse.metadata.items():
                    if isinstance(value, list) and len(value) == 1:
                        metadata[key] = value[0]
                    else:
                        metadata[key] = value

            # Platform information
            if hasattr(gse, "gpls"):
                platforms = {}
                for gpl_id, gpl in gse.gpls.items():
                    platforms[gpl_id] = {
                        "title": getattr(gpl, "metadata", {}).get("title", [""])[0]
                        if hasattr(gpl, "metadata")
                        else "",
                        "organism": getattr(gpl, "metadata", {}).get("organism", [""])[
                            0
                        ]
                        if hasattr(gpl, "metadata")
                        else "",
                        "technology": getattr(gpl, "metadata", {}).get(
                            "technology", [""]
                        )[0]
                        if hasattr(gpl, "metadata")
                        else "",
                    }
                metadata["platforms"] = platforms

            # Sample metadata
            if hasattr(gse, "gsms"):
                sample_metadata = {}
                for gsm_id, gsm in gse.gsms.items():
                    sample_meta = {}
                    if hasattr(gsm, "metadata"):
                        for key, value in gsm.metadata.items():
                            if isinstance(value, list) and len(value) == 1:
                                sample_meta[key] = value[0]
                            else:
                                sample_meta[key] = value
                    sample_metadata[gsm_id] = sample_meta
                metadata["samples"] = sample_metadata

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _validate_geo_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate GEO metadata against transcriptomics schema.
        
        Args:
            metadata: Extracted GEO metadata dictionary
            
        Returns:
            Dict containing validation results and schema alignment
        """
        try:
            from lobster.core.schemas.transcriptomics import TranscriptomicsSchema
            
            # Get the single-cell schema (covers most GEO datasets)
            schema = TranscriptomicsSchema.get_single_cell_schema()
            uns_schema = schema.get('uns', {}).get('optional', [])
            
            # Check which metadata fields align with our schema
            schema_aligned = {}
            schema_missing = []
            extra_fields = []
            
            # Check alignment for each field in metadata
            for field in metadata.keys():
                if field in uns_schema:
                    schema_aligned[field] = metadata[field]
                else:
                    extra_fields.append(field)
            
            # Check for schema fields not present in metadata
            for schema_field in uns_schema:
                if schema_field not in metadata:
                    schema_missing.append(schema_field)
            
            # Determine data type based on metadata
            data_type = self._determine_data_type_from_metadata(metadata)
            
            validation_result = {
                'schema_aligned_fields': len(schema_aligned),
                'schema_missing_fields': len(schema_missing),
                'extra_fields_count': len(extra_fields),
                'alignment_percentage': (len(schema_aligned) / len(uns_schema) * 100) if uns_schema else 0.0,
                'aligned_metadata': schema_aligned,
                'missing_fields': schema_missing[:10],  # Limit for display
                'extra_fields': extra_fields[:10],  # Limit for display
                'predicted_data_type': data_type,
                'validation_status': 'PASS' if len(schema_aligned) > len(schema_missing) else 'WARNING'
            }
            
            logger.info(f"Metadata validation: {validation_result['alignment_percentage']:.1f}% schema alignment")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            return {
                'validation_status': 'ERROR',
                'error_message': str(e),
                'schema_aligned_fields': 0,
                'alignment_percentage': 0.0
            }

    def _determine_data_type_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Determine likely data type (single-cell vs bulk) from metadata.
        
        Args:
            metadata: GEO metadata dictionary
            
        Returns:
            str: Predicted data type
        """
        try:
            # Check platform information
            platforms = metadata.get('platforms', {})
            platform_info = str(platforms).lower()
            
            # Check overall design
            overall_design = str(metadata.get('overall_design', '')).lower()
            
            # Check sample characteristics  
            samples = metadata.get('samples', {})
            sample_chars = []
            for sample in samples.values():
                chars = sample.get('characteristics_ch1', [])
                if isinstance(chars, list):
                    sample_chars.extend([str(c).lower() for c in chars])
            
            sample_text = ' '.join(sample_chars)
            
            # Keywords that suggest single-cell
            single_cell_keywords = [
                'single cell', 'single-cell', 'scrnaseq', 'scrna-seq', 
                '10x', 'chromium', 'droplet', 'microwell', 'smart-seq',
                'cell sorting', 'sorted cells', 'individual cells'
            ]
            
            # Keywords that suggest bulk
            bulk_keywords = [
                'bulk', 'tissue', 'whole', 'total rna', 'population'
            ]
            
            combined_text = f"{platform_info} {overall_design} {sample_text}"
            
            # Count keyword matches
            single_cell_score = sum(1 for keyword in single_cell_keywords if keyword in combined_text)
            bulk_score = sum(1 for keyword in bulk_keywords if keyword in combined_text)
            
            # Make prediction
            if single_cell_score > bulk_score:
                return 'single_cell_rna_seq'
            elif bulk_score > single_cell_score:
                return 'bulk_rna_seq'
            else:
                # Default to single-cell for GEO datasets (more common)
                return 'single_cell_rna_seq'
                
        except Exception as e:
            logger.warning(f"Error determining data type: {e}")
            return 'single_cell_rna_seq'  # Default

    def _format_metadata_summary(
        self, 
        geo_id: str, 
        metadata: Dict[str, Any], 
        validation_result: Dict[str, Any] = None
    ) -> str:
        """
        Format comprehensive metadata summary for user review.
        
        Args:
            geo_id: GEO accession ID
            metadata: Extracted metadata dictionary
            validation_result: Validation results
            
        Returns:
            str: Formatted metadata summary
        """
        try:
            # Extract key information
            title = metadata.get('title', 'N/A')
            summary = metadata.get('summary', 'N/A')[:500] + ('...' if len(str(metadata.get('summary', ''))) > 500 else '')
            overall_design = metadata.get('overall_design', 'N/A')[:300] + ('...' if len(str(metadata.get('overall_design', ''))) > 300 else '')
            
            # Sample information
            samples = metadata.get('samples', {})
            sample_count = len(samples)
            
            # Platform information
            platforms = metadata.get('platforms', {})
            platform_info = []
            for platform_id, platform_data in platforms.items():
                platform_info.append(f"{platform_id}: {platform_data.get('title', 'N/A')}")
            
            # Contact information
            contact_name = metadata.get('contact_name', 'N/A')
            contact_institute = metadata.get('contact_institute', 'N/A')
            
            # Publication info
            pubmed_id = metadata.get('pubmed_id', 'Not available')
            
            # Dates
            submission_date = metadata.get('submission_date', 'N/A')
            last_update = metadata.get('last_update_date', 'N/A')
            
            # Sample characteristics preview
            sample_preview = []
            for i, (sample_id, sample_data) in enumerate(samples.items()):
                if i < 3:  # Show first 3 samples
                    chars = sample_data.get('characteristics_ch1', [])
                    if isinstance(chars, list) and chars:
                        sample_preview.append(f"  - {sample_id}: {chars[0]}")
                    else:
                        sample_preview.append(f"  - {sample_id}: {sample_data.get('title', 'No title')}")
            
            if sample_count > 3:
                sample_preview.append(f"  ... and {sample_count - 3} more samples")
            
            # Validation status
            if not validation_result:
                validation_status = None
                alignment_pct = None
                predicted_type = None
            else:
                validation_status = validation_result.get('validation_status', 'UNKNOWN')
                alignment_pct = validation_result.get('alignment_percentage', 0.0)
                predicted_type = validation_result.get('predicted_data_type', 'unknown')
            
            # Format the summary
            summary_text = f"""📊 **GEO Dataset Metadata Summary: {geo_id}**

🔬 **Study Information:**
- **Title:** {title}
- **Summary:** {summary}
- **Design:** {overall_design}
- **Predicted Type:** {predicted_type.replace('_', ' ').title()}

👥 **Research Details:**
- **Contact:** {contact_name} ({contact_institute})
- **PubMed ID:** {pubmed_id}
- **Submission:** {submission_date}
- **Last Update:** {last_update}

🧪 **Platform Information:**
{chr(10).join(platform_info) if platform_info else '- No platform information available'}

🔢 **Sample Information ({sample_count} samples):**
{chr(10).join(sample_preview) if sample_preview else '- No sample information available'}

✅ **Schema Validation:**
- **Status:** {validation_status}
- **Schema Alignment:** {alignment_pct:.1f}% of expected fields present
- **Aligned Fields:** {validation_result.get('schema_aligned_fields', 0)}
- **Missing Fields:** {validation_result.get('schema_missing_fields', 0)}

📋 **Next Steps:**
1. **Review this metadata** to ensure it matches your research needs
2. **Confirm the predicted data type** is correct for your analysis
3. **Proceed to download** the full dataset if satisfied
4. **Use:** `download_geo_dataset('{geo_id}')` to download expression data

💡 **Note:** This metadata has been cached and validated against our transcriptomics schema. 
The actual expression data download will be much faster now that metadata is prepared."""

            return summary_text
            
        except Exception as e:
            logger.error(f"Error formatting metadata summary: {e}")
            return f"Error formatting metadata summary for {geo_id}: {str(e)}"

    def _get_sample_info(self, gse) -> Dict[str, Dict[str, Any]]:
        """
        Get sample information for downloading individual matrices.

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Sample information dictionary
        """
        sample_info = {}

        try:
            if hasattr(gse, "gsms"):
                for gsm_id, gsm in gse.gsms.items():
                    sample_info[gsm_id] = {
                        "title": getattr(gsm, "metadata", {}).get("title", [""])[0]
                        if hasattr(gsm, "metadata")
                        else "",
                        "platform": getattr(gsm, "metadata", {}).get(
                            "platform_id", [""]
                        )[0]
                        if hasattr(gsm, "metadata")
                        else "",
                        "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}",
                        "download_url": f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm_id[:6]}nnn/{gsm_id}/suppl/",
                    }

            logger.info(f"Collected information for {len(sample_info)} samples")
            return sample_info

        except Exception as e:
            logger.error(f"Error getting sample info: {e}"
            )
            return {}

    # def _download_sample_matrices(
    #     self, sample_info: Dict[str, Dict[str, Any]], gse_id: str
    # ) -> Dict[str, Optional[pd.DataFrame]]:
    #     """
    #     Download individual sample expression matrices sequentially.

    #     Args:
    #         sample_info: Dictionary of sample information
    #         gse_id: GEO series ID

    #     Returns:
    #         dict: Dictionary of sample matrices
    #     """
    #     sample_matrices = {}

    #     logger.info(f"Downloading matrices for {len(sample_info)} samples...")

    #     # Sequential download (no threading)
    #     for gsm_id, info in sample_info.items():
    #         try:
    #             matrix = self._download_single_sample(gsm_id, info, gse_id)
    #             sample_matrices[gsm_id] = matrix
    #             if matrix is not None:
    #                 logger.info(
    #                     f"Successfully downloaded matrix for {gsm_id}: {matrix.shape}"
    #                 )
    #             else:
    #                 logger.warning(f"No matrix data found for {gsm_id}")
    #         except Exception as e:
    #             logger.error(f"Error downloading {gsm_id}: {e}")
    #             sample_matrices[gsm_id] = None

    #     return sample_matrices


    def _download_sample_matrices(
        self, sample_info: Dict[str, Dict[str, Any]], gse_id: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Download individual sample expression matrices.

        Args:
            sample_info: Dictionary of sample information
            gse_id: GEO series ID

        Returns:
            dict: Dictionary of sample matrices
        """
        sample_matrices = {}

        logger.info(f"Downloading matrices for {len(sample_info)} samples...")

        # Use threading for parallel downloads
        #==========================================================================================================
        #==========================================================================================================
        #==========================================================================================================
        #==========================================================================================================
        #==========================================================================================================
        #==========================================================================================================
        #==========================================================================================================
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_sample = {
                executor.submit(
                    self._download_single_sample, gsm_id, info, gse_id
                ): gsm_id
                for gsm_id, info in sample_info.items()
            }

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    matrix = future.result()
                    sample_matrices[gsm_id] = matrix
                    if matrix is not None:
                        logger.info(
                            f"Successfully downloaded matrix for {gsm_id}: {matrix.shape}"
                        )
                    else:
                        logger.warning(f"No matrix data found for {gsm_id}")
                except Exception as e:
                    logger.error(f"Error downloading {gsm_id}: {e}")
                    sample_matrices[gsm_id] = None

        return sample_matrices

    def _download_single_sample(
        self, gsm_id: str, info: Dict[str, Any], gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download a single sample matrix with enhanced single-cell support.

        Args:
            gsm_id: GEO sample ID
            sample_info: Sample information
            gse_id: GEO series ID

        Returns:
            DataFrame: Sample expression matrix or None
        """
        try:
            # Try to get sample using GEOparse
            gsm = GEOparse.get_GEO(geo=gsm_id, destdir=str(self.cache_dir))

            # Check for 10X format supplementary files
            if hasattr(gsm, "metadata"):
                suppl_files_mapped = self._extract_supplementary_files_from_metadata(gsm.metadata, gsm_id)
                # suppl_files = gsm.metadata["supplementary_file"]
                df = self._download_and_combine_single_cell_files(suppl_files_mapped, gsm_id)
                return df
                
        except Exception as e:
            print(e)
            
            # Fallback to expression table
            if hasattr(gsm, "table") and gsm.table is not None:
                matrix = gsm.table
                return self._store_single_sample_as_modality(gsm_id, matrix, gsm)

            logger.warning(f"No expression data found for {gsm_id}")
            return None

        except Exception as e:
            logger.error(f"Error downloading sample {gsm_id}: {e}")
            return None

    def _extract_supplementary_files_from_metadata(self, metadata: Dict[str, Any], gsm_id: str) -> Dict[str, str]:
        """
        Extract all supplementary file URLs from sample metadata and classify them.
        
        Args:
            metadata: Sample metadata dictionary
            gsm_id: GEO sample ID for logging
            
        Returns:
            Dict[str, str]: Dictionary mapping file types ('matrix', 'barcodes', 'features') to URLs
        """
        try:
            supplementary_files = {}
            
            # Find all keys that contain 'supplementary'
            suppl_keys = [key for key in metadata.keys() if 'supplementary' in key.lower()]
            logger.info(f"Found {len(suppl_keys)} supplementary file keys for {gsm_id}: {suppl_keys}")
            
            for key in suppl_keys:
                file_urls = metadata[key]
                if isinstance(file_urls, list):
                    file_urls = file_urls
                else:
                    file_urls = [file_urls]
                
                for url in file_urls:
                    if not url or not isinstance(url, str):
                        continue
                        
                    url_lower = url.lower()
                    filename = url.split('/')[-1].lower()
                    
                    # Classify file type based on filename
                    if any(pattern in filename for pattern in ['matrix.mtx', 'matrix.txt', '_matrix']):
                        supplementary_files['matrix'] = url
                        logger.info(f"Classified as matrix: {filename}")
                    elif any(pattern in filename for pattern in ['barcodes.tsv', 'barcodes.txt', '_barcode', '-barcode']):
                        supplementary_files['barcodes'] = url
                        logger.info(f"Classified as barcodes: {filename}")
                    elif any(pattern in filename for pattern in ['features.tsv', 'features.txt', 'genes.tsv', '_feature', '_gene']):
                        supplementary_files['features'] = url
                        logger.info(f"Classified as features: {filename}")
                    elif any(ext in filename for ext in ['.h5', '.h5ad']):
                        supplementary_files['h5_data'] = url
                        logger.info(f"Classified as H5 data: {filename}")
                    else:
                        # Generic expression file
                        if any(ext in filename for ext in ['.txt', '.csv', '.tsv']) and any(keyword in filename for keyword in ['expr', 'count', 'rpkm', 'fpkm', 'tpm']):
                            supplementary_files['expression'] = url
                            logger.info(f"Classified as expression file: {filename}")
            
            logger.info(f"Classified supplementary files for {gsm_id}: {list(supplementary_files.keys())}")
            return supplementary_files
            
        except Exception as e:
            logger.error(f"Error extracting supplementary files from metadata for {gsm_id}: {e}")
            return {}

    def _download_and_combine_single_cell_files(self, supplementary_files_info: Dict[str, str], gsm_id: str) -> Optional[pd.DataFrame]:
        """
        Download and combine single-cell format files (matrix, barcodes, features).
        
        Args:
            supplementary_files_info: Dictionary mapping file types to URLs
            gsm_id: GEO sample ID
            
        Returns:
            DataFrame: Combined single-cell expression matrix or None
        """
        try:
            logger.info(f"Downloading and combining single-cell files for {gsm_id}")
            
            # Check if we have the trio of 10X files (matrix, barcodes, features)
            if all(key in supplementary_files_info for key in ['matrix', 'barcodes', 'features']):
                logger.info(f"Found complete 10X trio for {gsm_id}")
                return self._download_10x_trio(supplementary_files_info, gsm_id)
            
            # Check for H5 format (contains everything in one file)
            elif 'h5_data' in supplementary_files_info:
                logger.info(f"Found H5 data for {gsm_id}")
                return self._download_h5_file(supplementary_files_info['h5_data'], gsm_id)
            
            # Check for single expression matrix file
            elif 'expression' in supplementary_files_info:
                logger.info(f"Found expression file for {gsm_id}")
                return self._download_single_expression_file(supplementary_files_info['expression'], gsm_id)
            
            # Try individual files
            elif 'matrix' in supplementary_files_info:
                logger.info(f"Found matrix file only for {gsm_id}")
                return self._download_single_expression_file(supplementary_files_info['matrix'], gsm_id)
            
            else:
                logger.warning(f"No suitable file combination found for {gsm_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading and combining files for {gsm_id}: {e}")
            return None

    def _download_10x_trio(self, files_info: Dict[str, str], gsm_id: str) -> Optional[pd.DataFrame]:
        """
        Download and combine 10X format trio (matrix, barcodes, features).
        
        Args:
            files_info: Dictionary with 'matrix', 'barcodes', 'features' URLs
            gsm_id: GEO sample ID
            
        Returns:
            DataFrame: Combined 10X expression matrix or None
        """
        try:
            logger.info(f"Processing 10X trio for {gsm_id}")
            
            # Download all three files
            local_files = {}
            for file_type, url in files_info.items():
                if file_type in ['matrix', 'barcodes', 'features']:
                    filename = url.split('/')[-1]
                    local_path = self.cache_dir / f"{gsm_id}_{file_type}_{filename}"
                    
                    if not local_path.exists():
                        logger.info(f"Downloading {file_type} file: {url}")
                        if self.geo_downloader.download_file(url, local_path):
                            local_files[file_type] = local_path
                        else:
                            logger.error(f"Failed to download {file_type} file")
                            return None
                    else:
                        logger.info(f"Using cached {file_type} file: {local_path}")
                        local_files[file_type] = local_path
            
            if len(local_files) != 3:
                logger.error(f"Could not download all three 10X files for {gsm_id}")
                return None
            
            # Parse matrix file
            try:
                import scipy.io as sio
            except ImportError:
                logger.error("scipy is required for parsing 10X matrix files but not available")
                return None
            
            matrix_file = local_files['matrix']
            logger.info(f"Parsing matrix file: {matrix_file}")
            
            # Read the sparse matrix
            if matrix_file.name.endswith(".gz"):
                import gzip
                with gzip.open(matrix_file, "rt") as f:
                    matrix = sio.mmread(f)
            else:
                matrix = sio.mmread(matrix_file)
            
            # Convert to dense format and transpose (10X format is genes x cells, we want cells x genes)
            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix
            
            # Transpose so that cells are rows and genes are columns
            matrix_dense = matrix_dense.T
            logger.debug(f"Matrix shape after transpose: {matrix_dense.shape}")
            
            # Read barcodes
            barcodes_file = local_files['barcodes']
            logger.debug(f"Reading barcodes from: {barcodes_file}")
            cell_ids = []
            
            try:
                if barcodes_file.name.endswith(".gz"):
                    with gzip.open(barcodes_file, "rt") as f:
                        cell_ids = [line.strip() for line in f]
                else:
                    with open(barcodes_file, "r") as f:
                        cell_ids = [line.strip() for line in f]
                logger.info(f"Read {len(cell_ids)} cell barcodes")
            except Exception as e:
                logger.warning(f"Error reading barcodes file: {e}")
                cell_ids = [f"{gsm_id}_cell_{i}" for i in range(matrix_dense.shape[0])]
            
            # Read features
            features_file = local_files['features']
            logger.info(f"Reading features from: {features_file}")
            gene_ids = []
            gene_names = []
            
            try:
                if features_file.name.endswith(".gz"):
                    with gzip.open(features_file, "rt") as f:
                        lines = f.readlines()
                else:
                    with open(features_file, "r") as f:
                        lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        gene_ids.append(parts[0])
                        gene_names.append(parts[1])
                    elif len(parts) == 1:
                        gene_ids.append(parts[0])
                        gene_names.append(parts[0])
                
                logger.info(f"Read {len(gene_ids)} gene features")
            except Exception as e:
                logger.warning(f"Error reading features file: {e}")
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]
            
            # Create DataFrame with proper cell and gene names
            if not cell_ids or len(cell_ids) != matrix_dense.shape[0]:
                cell_ids = [f"{gsm_id}_cell_{i}" for i in range(matrix_dense.shape[0])]
            else:
                cell_ids = [f"{gsm_id}_{cell_id}" for cell_id in cell_ids]
            
            if not gene_names or len(gene_names) != matrix_dense.shape[1]:
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]
            
            # Create the final DataFrame
            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)
            logger.info(f"Successfully created 10X DataFrame for {gsm_id}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing 10X trio for {gsm_id}: {e}")
            return None

    def _download_h5_file(self, url: str, gsm_id: str) -> Optional[pd.DataFrame]:
        """
        Download and parse H5 format single-cell data.
        
        Args:
            url: URL to H5 file
            gsm_id: GEO sample ID
            
        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Processing H5 file for {gsm_id}")
            
            filename = url.split('/')[-1]
            local_path = self.cache_dir / f"{gsm_id}_h5_{filename}"
            
            # Download file
            if not local_path.exists():
                logger.info(f"Downloading H5 file: {url}")
                if not self.geo_downloader.download_file(url, local_path):
                    logger.error(f"Failed to download H5 file")
                    return None
            else:
                logger.info(f"Using cached H5 file: {local_path}")
            
            # Parse using geo_parser
            matrix = self.geo_parser.parse_supplementary_file(local_path)
            if matrix is not None and not matrix.empty:
                # Add sample prefix to cell names
                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                logger.info(f"Successfully parsed H5 file for {gsm_id}: {matrix.shape}")
                return matrix
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing H5 file for {gsm_id}: {e}")
            return None

    def _download_single_expression_file(self, url: str, gsm_id: str) -> Optional[pd.DataFrame]:
        """
        Download and parse a single expression file with enhanced support.
        
        Args:
            url: URL to expression file
            gsm_id: GEO sample ID
            
        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Processing single expression file for {gsm_id}")
            
            filename = url.split('/')[-1]
            local_path = self.cache_dir / f"{gsm_id}_expr_{filename}"
            
            # Download file using helper downloader (supports FTP)
            if not local_path.exists():
                logger.info(f"Downloading expression file: {url}")
                if not self.geo_downloader.download_file(url, local_path):
                    logger.error(f"Failed to download expression file")
                    return None
            else:
                logger.info(f"Using cached expression file: {local_path}")
            
            # Parse using geo_parser for better format support
            matrix = self.geo_parser.parse_supplementary_file(local_path)
            if matrix is not None and not matrix.empty:
                # Add sample prefix to row names if they look like cells
                if matrix.shape[0] > matrix.shape[1]:  # More rows than columns suggests cells x genes
                    matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                else:  # More columns than rows suggests genes x cells, transpose
                    matrix = matrix.T
                    matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                
                logger.info(f"Successfully parsed expression file for {gsm_id}: {matrix.shape}")
                return matrix
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing expression file for {gsm_id}: {e}")
            return None

    def _download_supplementary_file(
        self, url: str, gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Professional download and parse of supplementary files with FTP support.

        Args:
            url: URL to supplementary file (supports HTTP/HTTPS/FTP)
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Parsed matrix or None
        """
        try:
            logger.info(f"Downloading supplementary file for {gsm_id}: {url}")
            
            # Extract filename and create local path
            filename = url.split('/')[-1]
            local_file = self.cache_dir / f"{gsm_id}_suppl_{filename}"

            # Use helper downloader for better FTP/HTTP support and progress tracking
            if not local_file.exists():
                if not self.geo_downloader.download_file(url, local_file, f"Downloading {filename}"):
                    logger.error(f"Failed to download supplementary file: {url}")
                    return None
            else:
                logger.info(f"Using cached supplementary file: {local_file}")

            # Use geo_parser for enhanced format detection and parsing
            matrix = self.geo_parser.parse_supplementary_file(local_file)
            
            if matrix is not None and not matrix.empty:
                # Add sample prefix to row names for proper identification
                if matrix.shape[0] > matrix.shape[1]:  # More rows than columns suggests cells x genes
                    matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                else:  # More columns than rows suggests genes x cells, transpose
                    matrix = matrix.T
                    matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                
                logger.info(f"Successfully parsed supplementary file for {gsm_id}: {matrix.shape}")
                return matrix
            else:
                logger.warning(f"Could not parse supplementary file or file is empty: {local_file}")
                return None

        except Exception as e:
            logger.error(f"Error downloading supplementary file {url} for {gsm_id}: {e}")
            return None

    def _validate_matrices(
        self, sample_matrices: Dict[str, Optional[pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Validate downloaded matrices and filter out invalid ones using multithreading.

        Args:
            sample_matrices: Dictionary of sample matrices

        Returns:
            dict: Dictionary of validated matrices
        """
        validated = {}
        
        # Filter out None matrices first
        valid_matrices = {gsm_id: matrix for gsm_id, matrix in sample_matrices.items() if matrix is not None}
        
        if not valid_matrices:
            logger.warning("No matrices to validate")
            return validated

        logger.info(f"Validating {len(valid_matrices)} matrices using multithreading...")

        # Use multithreading for validation - this is the main performance improvement
        with ThreadPoolExecutor(max_workers=min(8, len(valid_matrices))) as executor:
            future_to_sample = {
                executor.submit(self._validate_single_matrix, gsm_id, matrix): gsm_id
                for gsm_id, matrix in valid_matrices.items()
            }

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    is_valid, validation_info = future.result()
                    if is_valid:
                        validated[gsm_id] = valid_matrices[gsm_id]
                        logger.info(f"Validated {gsm_id}: {validation_info}")
                    else:
                        logger.warning(f"Skipping {gsm_id}: {validation_info}")
                except Exception as e:
                    logger.error(f"Error validating {gsm_id}: {e}")

        logger.info(f"Validated {len(validated)}/{len(sample_matrices)} matrices")
        return validated

    def _validate_single_matrix(self, gsm_id: str, matrix: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate a single matrix with optimized checks.
        
        Args:
            gsm_id: Sample ID for logging
            matrix: DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, info_message)
        """
        try:
            # Check matrix dimensions first (fastest check)
            if matrix.shape[0] < 10 or matrix.shape[1] < 10:
                return False, f"Matrix too small ({matrix.shape})"

            # Use optimized validation
            if not self._is_valid_expression_matrix(matrix):
                return False, "Invalid matrix format"

            return True, f"Valid matrix {matrix.shape}"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _is_valid_expression_matrix(self, matrix: pd.DataFrame) -> bool:
        """
        Optimized check if a matrix is a valid expression matrix.
        Uses sampling and vectorized operations for better performance on large DataFrames.

        Args:
            matrix: DataFrame to validate

        Returns:
            bool: True if valid expression matrix
        """
        try:
            # Check if it's a DataFrame
            if not isinstance(matrix, pd.DataFrame):
                return False

            # Fast check: ensure we have numeric data by checking dtypes
            # This is much faster than select_dtypes() for large DataFrames
            numeric_dtypes = set(['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
            has_numeric = any(str(dtype) in numeric_dtypes for dtype in matrix.dtypes)
            
            if not has_numeric:
                return False

            # For large matrices, use sampling to speed up validation  
            if matrix.size > 1_000_000:  # > 1M cells
                # Sample 10% of the data or max 100k cells for validation
                sample_size = min(100_000, int(matrix.size * 0.1))
                
                # Get a random sample of the flattened matrix
                flat_sample = matrix.select_dtypes(include=[np.number]).values.flatten()
                if len(flat_sample) > sample_size:
                    indices = np.random.choice(len(flat_sample), sample_size, replace=False)
                    sample_data = flat_sample[indices]
                else:
                    sample_data = flat_sample
                
                # Check for non-negative values in sample
                if np.any(sample_data < 0):
                    logger.warning("Matrix contains negative values (detected in sample)")
                
                # Check for reasonable value ranges in sample
                max_val = np.max(sample_data)
                if max_val > 1e6:
                    logger.info("Matrix contains very large values (possibly raw counts)")
                    
            else:
                # For smaller matrices, do full validation but with optimized operations
                numeric_data = matrix.select_dtypes(include=[np.number])
                
                # Use numpy operations which are faster than pandas
                values = numeric_data.values
                
                # Check for non-negative values using numpy
                if np.any(values < 0):
                    logger.warning("Matrix contains negative values")

                # Check for reasonable value ranges using numpy
                max_val = np.max(values)
                if max_val > 1e6:
                    logger.info("Matrix contains very large values (possibly raw counts)")

            return True

        except Exception as e:
            logger.error(f"Error validating matrix: {e}")
            return False

    def _store_samples_as_anndata(
        self, validated_matrices: Dict[str, pd.DataFrame], gse_id: str, metadata: Dict[str, Any]
    ) -> List[str]:
        """
        Store each sample as an individual AnnData object in DataManagerV2.
        
        Args:
            validated_matrices: Dictionary of validated sample matrices
            gse_id: GEO series ID
            metadata: Metadata from GEO
            
        Returns:
            List[str]: List of modality names that were successfully stored
        """
        stored_samples = []
        
        try:
            logger.info(f"Storing {len(validated_matrices)} samples as individual AnnData objects")
            
            for gsm_id, matrix in validated_matrices.items():
                try:
                    # Create unique modality name for this sample
                    modality_name = f"geo_{gse_id.lower()}_sample_{gsm_id.lower()}"
                    
                    # Extract sample-specific metadata
                    sample_metadata = {}
                    if 'samples' in metadata and gsm_id in metadata['samples']:
                        sample_metadata = metadata['samples'][gsm_id]
                    
                    # Create enhanced metadata for this sample
                    enhanced_metadata = {
                        "dataset_id": gse_id,
                        "sample_id": gsm_id,
                        "dataset_type": "GEO_Sample",
                        "parent_dataset": gse_id,
                        "sample_metadata": sample_metadata,
                        "processing_date": pd.Timestamp.now().isoformat(),
                        "data_source": "individual_sample_matrix",
                        "is_preprocessed": False,
                        "needs_concatenation": True
                    }
                    
                    # Determine adapter based on data characteristics
                    n_obs, n_vars = matrix.shape
                    if n_vars > 5000:  # High gene count suggests single-cell
                        adapter_name = "transcriptomics_single_cell"
                    else:
                        adapter_name = "transcriptomics_bulk"
                    
                    # Load as modality in DataManagerV2
                    adata = self.data_manager.load_modality(
                        name=modality_name,
                        source=matrix,
                        adapter=adapter_name,
                        validate=False,  # Skip validation for individual samples
                        **enhanced_metadata
                    )
                    
                    # Save to workspace
                    save_path = f"{gse_id.lower()}_{gsm_id.lower()}_raw.h5ad"
                    # Create directory if needed
                    (self.data_manager.data_dir / gse_id.lower()).mkdir(exist_ok=True)
                    self.data_manager.save_modality(
                        name=modality_name, 
                        path=save_path)
                    
                    stored_samples.append(modality_name)
                    logger.info(f"Stored sample {gsm_id} as modality '{modality_name}' ({adata.shape})")
                    
                except Exception as e:
                    logger.error(f"Failed to store sample {gsm_id}: {e}")
                    continue
            
            logger.info(f"Successfully stored {len(stored_samples)} samples as individual AnnData objects")
            return stored_samples
            
        except Exception as e:
            logger.error(f"Error storing samples as AnnData: {e}")
            return stored_samples

    def _create_placeholder_matrix(
        self, stored_samples: List[str], gse_id: str
    ) -> pd.DataFrame:
        """
        Create a placeholder matrix for immediate compatibility.
        This will contain summary information about the stored samples.
        
        Args:
            stored_samples: List of modality names that were stored
            gse_id: GEO series ID
            
        Returns:
            DataFrame: Placeholder matrix with sample information
        """
        try:
            # Collect information about stored samples
            sample_info_list = []
            total_cells = 0
            all_genes = set()
            
            for modality_name in stored_samples:
                adata = self.data_manager.get_modality(modality_name)
                sample_id = modality_name.split('_')[-1].upper()
                
                sample_info_list.append({
                    'sample_id': sample_id,
                    'modality_name': modality_name,
                    'n_cells': adata.n_obs,
                    'n_genes': adata.n_vars,
                    'stored': True
                })
                
                total_cells += adata.n_obs
                all_genes.update(adata.var_names)
            
            # Create a summary DataFrame
            summary_df = pd.DataFrame(sample_info_list)
            
            # Create a minimal placeholder matrix with metadata
            # This matrix will have one row per sample with summary statistics
            placeholder_matrix = pd.DataFrame(
                index=summary_df['sample_id'],
                columns=['n_cells', 'n_genes', 'modality_name', 'status']
            )
            
            for _, row in summary_df.iterrows():
                placeholder_matrix.loc[row['sample_id']] = [
                    row['n_cells'],
                    row['n_genes'],
                    row['modality_name'],
                    'stored_for_preprocessing'
                ]
            
            # Add metadata as attributes
            placeholder_matrix.attrs = {
                'dataset_id': gse_id,
                'total_samples': len(stored_samples),
                'total_cells': total_cells,
                'total_unique_genes': len(all_genes),
                'stored_modalities': stored_samples,
                'note': 'Individual samples stored as AnnData objects for preprocessing',
                'concatenation_pending': True
            }
            
            logger.info(f"Created placeholder matrix with {len(stored_samples)} samples")
            return placeholder_matrix
            
        except Exception as e:
            logger.error(f"Error creating placeholder matrix: {e}")
            # Return a minimal placeholder
            return pd.DataFrame({'status': ['error']}, index=[gse_id])

    def _concatenate_stored_samples(
        self, geo_id: str, stored_samples: List[str], use_intersecting_genes_only: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Concatenate stored AnnData samples using anndata's concat function.
        
        This function loads individual AnnData objects that were previously stored,
        concatenates them using anndata.concat(), and returns the result as a DataFrame
        for compatibility with the rest of the pipeline.
        
        Args:
            geo_id: GEO series ID
            stored_samples: List of modality names that were stored
            use_intersecting_genes_only: If True, use only common genes across all samples.
                                       If False, use all genes (filling missing with zeros).
            
        Returns:
            DataFrame: Concatenated expression matrix or None if concatenation fails
        """
        try:
            import anndata as ad
            
            logger.info(f"Concatenating {len(stored_samples)} stored samples for {geo_id}")
            
            # Validate input
            if not stored_samples:
                logger.error("No stored samples provided for concatenation")
                return None
            
            # Load all AnnData objects
            adata_list = []
            sample_ids = []
            
            for modality_name in stored_samples:
                try:
                    # Load the AnnData object from data manager
                    adata = self.data_manager.get_modality(modality_name)
                    if adata is None:
                        logger.warning(f"Could not load modality: {modality_name}")
                        continue
                    
                    # Extract sample ID from modality name
                    sample_id = modality_name.split('_')[-1].upper()
                    sample_ids.append(sample_id)
                    
                    # Add batch information to obs
                    adata.obs['batch'] = sample_id
                    adata.obs['sample_id'] = sample_id
                    
                    adata_list.append(adata)
                    logger.debug(f"Loaded {modality_name}: {adata.shape}")
                    
                except Exception as e:
                    logger.error(f"Failed to load modality {modality_name}: {e}")
                    continue
            
            # Check if we have any valid data
            if not adata_list:
                logger.error("No valid AnnData objects could be loaded")
                return None
            
            logger.info(f"Successfully loaded {len(adata_list)} AnnData objects")
            
            # Concatenate using anndata
            if use_intersecting_genes_only:
                # Use intersection (inner join) - only common genes
                logger.info("Concatenating with gene intersection (inner join)")
                combined_adata = ad.concat(
                    adata_list, 
                    axis=0,         # Concatenate along observations (cells)
                    join='inner',   # Use only common genes
                    merge='unique', # Merge unique keys in uns
                    label='batch',
                    keys=sample_ids
                )
            else:
                # Use union (outer join) - all genes
                logger.info("Concatenating with gene union (outer join)")
                combined_adata = ad.concat(
                    adata_list,
                    axis=0,         # Concatenate along observations (cells)
                    join='outer',   # Use all genes
                    merge='unique', # Merge unique keys in uns
                    fill_value=0,   # Fill missing values with 0
                    label='batch',
                    keys=sample_ids
                )
            
            logger.info(f"Concatenation successful: {combined_adata.shape}")
            
            # Convert to DataFrame for compatibility with the rest of the pipeline
            # The expression matrix is stored in X
            if hasattr(combined_adata.X, 'todense'):
                # Handle sparse matrix
                expression_matrix = pd.DataFrame(
                    combined_adata.X.todense(),
                    index=combined_adata.obs_names,
                    columns=combined_adata.var_names
                )
            else:
                # Handle dense matrix
                expression_matrix = pd.DataFrame(
                    combined_adata.X,
                    index=combined_adata.obs_names,
                    columns=combined_adata.var_names
                )
            
            # Add batch information as a column (for compatibility)
            if 'batch' in combined_adata.obs.columns:
                expression_matrix['batch'] = combined_adata.obs['batch'].values
            
            logger.info(f"Converted to DataFrame: {expression_matrix.shape}")
            
            # Log statistics
            n_genes = len(expression_matrix.columns) - (1 if 'batch' in expression_matrix.columns else 0)
            logger.info(f"Final matrix: {len(expression_matrix)} cells × {n_genes} genes")
            logger.info(f"Samples included: {', '.join(sample_ids)}")
            
            return expression_matrix
            
        except ImportError:
            logger.error("anndata package is required but not installed. Install with: pip install anndata")
            return None
        except Exception as e:
            logger.error(f"Error concatenating stored samples: {e}")
            logger.exception("Full traceback:")
            return None

    def _check_workspace_for_processed_data(
        self, gse_id: str
    ) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Check if processed data for a GSE ID already exists in workspace.
        Uses professional naming patterns with backward compatibility.

        Args:
            gse_id: GEO series ID

        Returns:
            Tuple of (DataFrame, metadata) if found, None otherwise
        """
        try:
            from lobster.utils.file_naming import BioinformaticsFileNaming
            
            # First, try professional naming pattern
            data_dir = self.data_manager.data_dir
            existing_file = BioinformaticsFileNaming.find_latest_file(
                directory=data_dir,
                data_source='GEO',
                dataset_id=gse_id,
                processing_step='raw_matrix'
            )
            
            if existing_file and existing_file.exists():
                logger.info(f"Found existing file with professional naming: {existing_file.name}")
                
                # Load the data
                combined_matrix = pd.read_csv(existing_file, index_col=0)
                
                # Load associated metadata
                metadata_file = existing_file.parent / BioinformaticsFileNaming.generate_metadata_filename(existing_file.name)
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                logger.info(f"Successfully loaded existing data: {combined_matrix.shape}")
                return combined_matrix, metadata
            
            # Backward compatibility: check for legacy naming patterns
            data_files = self.data_manager.list_workspace_files()["data"]
            for file_info in data_files:
                if f"{gse_id}_processed" in file_info["name"] and file_info["name"].endswith(".csv"):
                    logger.info(f"Found existing legacy file: {file_info['name']}")

                    # Load the data
                    data_path = Path(file_info["path"])
                    combined_matrix = pd.read_csv(data_path, index_col=0)

                    # Load associated metadata
                    metadata_path = data_path.parent / f"{data_path.stem}_metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                    logger.info(f"Successfully loaded legacy data: {combined_matrix.shape}")
                    return combined_matrix, metadata

            return None

        except Exception as e:
            logger.warning(f"Error checking workspace for existing data: {e}")
            return None


    def _format_download_response(
        self,
        gse_id: str,
        combined_matrix: pd.DataFrame,
        metadata: Dict[str, Any],
        sample_count: int,
        saved_file: str = None,
    ) -> str:
        """
        Format download response message.

        Args:
            gse_id: GEO series ID
            combined_matrix: Combined expression matrix
            metadata: Metadata dictionary
            sample_count: Number of samples processed
            saved_file: Path to saved workspace file

        Returns:
            str: Formatted response message
        """
        study_title = (
            metadata.get("title", ["N/A"])[0]
            if isinstance(metadata.get("title"), list)
            else metadata.get("title", "N/A")
        )
        organism = (
            metadata.get("organism", ["N/A"])[0]
            if isinstance(metadata.get("organism"), list)
            else metadata.get("organism", "N/A")
        )

        workspace_info = ""
        if saved_file and "Error" not in saved_file:
            workspace_info = f"\n💾 Saved to workspace: {Path(saved_file).name}\n⚡ Future loads will be instant (no re-downloading needed)"

        return f"""Successfully downloaded and processed {gse_id} using GEOparse!

📊 Combined expression matrix: {combined_matrix.shape[0]} cells × {combined_matrix.shape[1]} genes
📋 Study: {study_title}
🧬 Organism: {organism}
🔬 Successfully processed {sample_count} samples{workspace_info}
📈 Ready for downstream analysis (quality control, clustering, ML)

The data has been professionally concatenated and is ready for:
- Quality assessment and filtering
- Clustering and cell type annotation
- Machine learning model preparation
- Differential expression analysis"""
