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
from lobster.tools.pipeline_strategy import (
    PipelineStrategyEngine,
    PipelineContext,
    PipelineType,
    create_pipeline_context
)

logger = get_logger(__name__)


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                        DATA STRUCTURES AND ENUMS                          ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

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


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                           MAIN SERVICE CLASS                               ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

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
        
        # Initialize the pipeline strategy engine
        self.pipeline_engine = PipelineStrategyEngine()
        
        # Default download strategy
        self.download_strategy = DownloadStrategy()

        logger.info("GEOService initialized with modular architecture: GEOparse + dynamic pipeline strategy")


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                  MAIN ENTRY POINTS (USED BY DATA_EXPERT)                  ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

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
            
            # Primary approach: GEOparse
            metadata = None
            validation_result = None
            
            try:
                logger.debug(f"Downloading SOFT metadata for {clean_geo_id} using GEOparse...")
                gse = GEOparse.get_GEO(geo=clean_geo_id, destdir=str(self.cache_dir))
                metadata = self._extract_metadata(gse)
                logger.debug(f"Successfully extracted metadata using GEOparse for {clean_geo_id}")
                
            except Exception as geoparse_error:
                logger.error(f"Helper metadata fetch failed:")
                return f"Failed to fetch metadata for {clean_geo_id}. GEOparse ({geoparse_error}) failed."                
            
            if not metadata:
                return f"No metadata could be extracted for {clean_geo_id}"
            
            # Validate metadata against transcriptomics schema
            validation_result = self._validate_geo_metadata(metadata)
            
            return metadata, validation_result
            
        except Exception as e:
            logger.exception(f"Error fetching metadata for {geo_id}: {e}")
            return f"Error fetching metadata for {geo_id}: {str(e)}"

    def download_dataset(self, geo_id: str, adapter: str = None, **kwargs) -> str:
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
            modality_name = f"geo_{clean_geo_id.lower()}_{adapter}"
            existing_modalities = self.data_manager.list_modalities()
            if modality_name in existing_modalities:
                return f"Dataset {clean_geo_id} already loaded as modality '{modality_name}'. Use data_manager.get_modality('{modality_name}') to access it."

            # Use the strategic download approach
            geo_result = self.download_with_strategy(geo_id = clean_geo_id, **kwargs)
            
            if not geo_result.success:
                return f"Failed to download {clean_geo_id} using all available methods. Last error: {geo_result.error_message}"
            
            # Store as modality in DataManagerV2
            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "source_metadata": geo_result.metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "download_source": geo_result.source.value,
                "processing_method": geo_result.processing_info.get("method", "unknown"),
                "data_type": geo_result.processing_info.get("data_type", "unknown")
            }

            # Determine appropriate adapter based on data characteristics and metadata
            if not enhanced_metadata.get('data_type', None):
                cached_metadata = self.data_manager.metadata_store[clean_geo_id]['metadata']
                predicted_type = self._determine_data_type_from_metadata(cached_metadata)
                
            n_obs, n_vars = geo_result.data.shape

            # if no adapter name is given find out from data downloading step
            if not adapter:
                if enhanced_metadata.get('data_type') == 'single_cell_rna_seq':
                    adapter_name = "transcriptomics_single_cell"
                elif enhanced_metadata.get('data_type') == 'bulk_rna_seq':
                    adapter_name = "transcriptomics_bulk"
                else:
                    # Default to single-cell for GEO datasets (more common)
                    adapter_name = 'single_cell_rna_seq'
            else: 
                adapter_name = adapter

            logger.debug(f"Using adapter '{adapter_name}' based on predicted type '{enhanced_metadata.get('data_type', None)}' and data shape {geo_result.data.shape}")

            # Load as modality in DataManagerV2
            adata = self.data_manager.load_modality(
                name=modality_name,
                source=geo_result.data,
                adapter=adapter_name,
                validate=True,
                **enhanced_metadata
            )

            # Save to workspace
            save_path = f"{modality_name}_raw.h5ad"
            saved_file = self.data_manager.save_modality(modality_name, save_path)

            # Log successful download and save
            self.data_manager.log_tool_usage(
                tool_name="download_geo_dataset_strategic",
                parameters={
                    "geo_id": clean_geo_id, 
                    "download_source": geo_result.source.value,
                    "processing_method": geo_result.processing_info.get("method", "unknown")
                },
                description=f"Downloaded GEO dataset {clean_geo_id} using strategic approach ({geo_result.source.value}), saved to {saved_file}",
            )

            # Auto-save current state
            self.data_manager.auto_save_state()

            return f"""Successfully downloaded and loaded GEO dataset {clean_geo_id}!

📊 Modality: '{modality_name}' ({adata.n_obs} obs × {adata.n_vars} vars)
🔬 Adapter: {adapter_name} (predicted: {enhanced_metadata.get('data_type', None)})
💾 Saved to: {save_path}
🎯 Source: {geo_result.source.value} ({geo_result.processing_info.get('method', 'unknown')})
⚡ Ready for quality control and downstream analysis!

The dataset is now available as modality '{modality_name}' for other agents to use."""

        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            return f"Error downloading dataset: {str(e)}"


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                   STRATEGIC DOWNLOAD COORDINATION                          ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

    def download_with_strategy(self, geo_id: str, manual_strategy_override: PipelineType = None) -> GEOResult:
        """
        Master function implementing layered download approach using dynamic pipeline strategy.
        
        Args:
            geo_id: GEO accession ID
            
        Returns:
            GEOResult: Comprehensive result with data and metadata
        """
        clean_geo_id = geo_id.strip().upper()
        
        logger.debug(f"Starting strategic download for {clean_geo_id}")
        
        try:
            # Step 1: Ensure metadata exists
            if clean_geo_id not in self.data_manager.metadata_store:
                metadata_summary = self.fetch_metadata_only(clean_geo_id)
                if "Error" in str(metadata_summary):
                    return GEOResult(
                        success=False,
                        error_message=f"Failed to fetch metadata: {metadata_summary}",
                        source=GEODataSource.GEOPARSE
                    )
            
            # Step 2: Get metadata and strategy config
            stored_metadata_info = self.data_manager.metadata_store[clean_geo_id]
            cached_metadata = stored_metadata_info['metadata']
            strategy_config = stored_metadata_info.get('strategy_config', {})
            
            if not strategy_config:
                # Extract strategy config if not present (backward compatibility)
                logger.warning(f"No strategy config found for {clean_geo_id}, using defaults")
                strategy_config = {
                    'raw_data_available': True,
                    'summary_file_name': '',
                    'processed_matrix_name': '',
                    'raw_UMI_like_matrix_name': '',
                    'cell_annotation_name': ''
                }
            
            # Step 3: IF USER DECIDES WHICH APPROACH TO CHOOSE MANUALLY OVERRIDE THE AUTOMATED APPRAOCH
            if manual_strategy_override:
                pipeline = self.pipeline_engine.get_pipeline_functions(manual_strategy_override, self)
            else:
                pipeline = self._get_processing_pipeline(clean_geo_id, cached_metadata, strategy_config)
            
            logger.debug(f"Using dynamic pipeline with {len(pipeline)} steps")
            
            # Step 4: Execute pipeline with retries
            for i, pipeline_func in enumerate(pipeline):
                logger.debug(f"Executing pipeline step {i + 1}: {pipeline_func.__name__}")
                
                try:
                    #===============================================================
                    result = pipeline_func(clean_geo_id, cached_metadata)
                    #===============================================================
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
                error_message=f"All pipeline steps failed after enough attempts",
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

    def _get_processing_pipeline(
        self, 
        geo_id: str,
        metadata: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> List[Callable]:
        """
        Get the appropriate processing pipeline using the strategy engine.
        
        Args:
            geo_id: GEO accession ID  
            metadata: GEO metadata
            strategy_config: Extracted strategy configuration
            
        Returns:
            List[Callable]: Pipeline functions to execute in order
        """
        # Determine data type
        data_type = self._determine_data_type_from_metadata(metadata)
        
        # Create pipeline context
        context = create_pipeline_context(
            geo_id=geo_id,
            strategy_config=strategy_config,
            metadata=metadata,
            data_type=data_type
        )
        
        # Determine best pipeline type
        pipeline_type, description = self.pipeline_engine.determine_pipeline(context)
        
        logger.info(f"Pipeline selection for {geo_id}: {pipeline_type.name}")
        logger.info(f"Reason: {description}")
        
        # Get the actual processing functions
        pipeline_functions = self.pipeline_engine.get_pipeline_functions(pipeline_type, self)
        
        return pipeline_functions


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                        PIPELINE STEP FUNCTIONS                            ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

    def _try_processed_matrix_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to directly download and use processed matrix files based on LLM strategy config."""
        try:
            logger.info(f"Attempting to use processed matrix for {geo_id}")
            
            # Get strategy config from stored metadata
            stored_metadata = self.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get('strategy_config', {})
            
            matrix_name = strategy_config.get('processed_matrix_name', '')
            matrix_type = strategy_config.get('processed_matrix_filetype', '')
            
            if not matrix_name or not matrix_type:
                return GEOResult(success=False, error_message="No processed matrix information available in strategy config")
            
            # Try to download the specific file from supplementary files
            suppl_files = metadata.get('supplementary_file', [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]
            
            # Find the matching file
            target_file = None
            for file_url in suppl_files:
                if matrix_name in file_url and matrix_type in file_url:
                    target_file = file_url
                    break
            
            if target_file:
                logger.info(f"Found processed matrix file: {target_file}")
                matrix = self._download_and_parse_file(target_file, geo_id)
                
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "processed_matrix_direct",
                            "file": f"{matrix_name}.{matrix_type}",
                            "data_type": self._determine_data_type_from_metadata(metadata)
                        },
                        success=True
                    )
            
            return GEOResult(success=False, error_message=f"Could not download processed matrix: {matrix_name}.{matrix_type}")
            
        except Exception as e:
            logger.error(f"Error in processed matrix pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_raw_matrix_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to directly download and use raw UMI/count matrix files based on LLM strategy config."""
        try:
            logger.info(f"Attempting to use raw matrix for {geo_id}")
            
            # Get strategy config from stored metadata
            stored_metadata = self.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get('strategy_config', {})
            
            matrix_name = strategy_config.get('raw_UMI_like_matrix_name', '')
            matrix_type = strategy_config.get('raw_UMI_like_matrix_filetype', '')
            
            if not matrix_name or not matrix_type:
                return GEOResult(success=False, error_message="No raw matrix information available in strategy config")
            
            # Similar logic to processed matrix
            suppl_files = metadata.get('supplementary_file', [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]
            
            target_file = None
            for file_url in suppl_files:
                if matrix_name in file_url and matrix_type in file_url:
                    target_file = file_url
                    break
            
            if target_file:
                logger.info(f"Found raw matrix file: {target_file}")
                matrix = self._download_and_parse_file(target_file, geo_id)
                
                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "raw_matrix_direct",
                            "file": f"{matrix_name}.{matrix_type}",
                            "data_type": self._determine_data_type_from_metadata(metadata)
                        },
                        success=True
                    )
            
            return GEOResult(success=False, error_message=f"Could not download raw matrix: {matrix_name}.{matrix_type}")
            
        except Exception as e:
            logger.error(f"Error in raw matrix pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_h5_format_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to prioritize H5/H5AD format files for efficient loading."""
        try:
            logger.info(f"Attempting to use H5 format files for {geo_id}")
            
            # Check both processed and raw for H5 formats
            stored_metadata = self.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get('strategy_config', {})
            
            # Check processed matrix first
            processed_name = strategy_config.get('processed_matrix_name', '')
            processed_type = strategy_config.get('processed_matrix_filetype', '')
            
            # Check raw matrix
            raw_name = strategy_config.get('raw_UMI_like_matrix_name', '')
            raw_type = strategy_config.get('raw_UMI_like_matrix_filetype', '')
            
            h5_files = []
            if processed_type in ["h5", "h5ad"]:
                h5_files.append((processed_name, processed_type, "processed"))
            if raw_type in ["h5", "h5ad"]:
                h5_files.append((raw_name, raw_type, "raw"))
            
            if not h5_files:
                return GEOResult(success=False, error_message="No H5 format files found in strategy config")
            
            # Try to download H5 files
            suppl_files = metadata.get('supplementary_file', [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]
            
            for file_name, file_type, file_category in h5_files:
                target_file = None
                for file_url in suppl_files:
                    if file_name in file_url and file_type in file_url:
                        target_file = file_url
                        break
                
                if target_file:
                    logger.info(f"Found H5 {file_category} file: {target_file}")
                    # For H5 files, try parsing with geo_parser directly
                    filename = target_file.split('/')[-1]
                    local_path = self.cache_dir / f"{geo_id}_h5_{filename}"
                    
                    if not local_path.exists():
                        if not self.geo_downloader.download_file(target_file, local_path):
                            continue
                    
                    matrix = self.geo_parser.parse_supplementary_file(local_path)
                    if matrix is not None and not matrix.empty:
                        return GEOResult(
                            data=matrix,
                            metadata=metadata,
                            source=GEODataSource.SUPPLEMENTARY,
                            processing_info={
                                "method": f"h5_format_{file_category}",
                                "file": f"{file_name}.{file_type}",
                                "data_type": self._determine_data_type_from_metadata(metadata)
                            },
                            success=True
                        )
            
            return GEOResult(success=False, error_message="Could not download or parse any H5 format files")
            
        except Exception as e:
            logger.error(f"Error in H5 format pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try supplementary files as primary approach when no direct matrices available."""
        try:
            logger.info(f"Attempting supplementary files first for {geo_id}")
            
            # Get GEO object for supplementary file processing
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))
            
            # Use existing supplementary file processing
            data = self._process_supplementary_files(gse, geo_id)
            if data is not None and not data.empty:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.SUPPLEMENTARY,
                    processing_info={
                        "method": "supplementary_first",
                        "data_type": self._determine_data_type_from_metadata(metadata),
                        "n_samples": len(gse.gsms) if hasattr(gse, "gsms") else 0
                    },
                    success=True
                )
            
            return GEOResult(success=False, error_message="No usable data found in supplementary files")
            
        except Exception as e:
            logger.error(f"Error in supplementary first pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_archive_extraction_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try extracting from archive files (TAR, ZIP) as primary approach."""
        try:
            logger.info(f"Attempting archive extraction first for {geo_id}")
            
            suppl_files = metadata.get('supplementary_file', [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]
            
            # Look for archive files
            archive_files = [f for f in suppl_files if any(ext in f.lower() for ext in ['.tar', '.zip', '.rar'])]
            
            if not archive_files:
                return GEOResult(success=False, error_message="No archive files found")
            
            # Try processing archive files
            for archive_url in archive_files:
                if archive_url.lower().endswith('.tar'):
                    matrix = self._process_tar_file(archive_url, geo_id)
                    if matrix is not None and not matrix.empty:
                        return GEOResult(
                            data=matrix,
                            metadata=metadata,
                            source=GEODataSource.TAR_ARCHIVE,
                            processing_info={
                                "method": "archive_extraction_first",
                                "file": archive_url.split('/')[-1],
                                "data_type": self._determine_data_type_from_metadata(metadata)
                            },
                            success=True
                        )
            
            return GEOResult(success=False, error_message="Could not extract usable data from archive files")
            
        except Exception as e:
            logger.error(f"Error in archive extraction pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_fallback(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Fallback method using supplementary files when primary approaches fail."""
        try:
            logger.info(f"Trying supplementary fallback for {geo_id}")
            
            # This is essentially the same as _try_supplementary_first but with different logging
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))
            
            data = self._process_supplementary_files(gse, geo_id)
            if data is not None and not data.empty:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.SUPPLEMENTARY,
                    processing_info={
                        "method": "supplementary_fallback",
                        "data_type": self._determine_data_type_from_metadata(metadata),
                        "note": "Used as fallback after primary methods failed"
                    },
                    success=True
                )
            
            return GEOResult(success=False, error_message="Supplementary fallback found no usable data")
            
        except Exception as e:
            logger.error(f"Error in supplementary fallback pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_emergency_fallback(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Emergency fallback when all other methods fail."""
        try:
            logger.warning(f"Using emergency fallback for {geo_id} - all other methods failed")
            
            # Try to get any available data using basic GEOparse approach
            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.cache_dir))
            
            # Try to get expression data from any available sample
            if hasattr(gse, 'gsms') and gse.gsms:
                for gsm_id, gsm in list(gse.gsms.items())[:5]:  # Try first 5 samples only
                    try:
                        if hasattr(gsm, 'table') and gsm.table is not None:
                            matrix = gsm.table
                            if matrix.shape[0] > 0 and matrix.shape[1] > 0:
                                # Add sample prefix to avoid conflicts
                                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                                
                                logger.warning(f"Emergency fallback found data from sample {gsm_id}: {matrix.shape}")
                                return GEOResult(
                                    data=matrix,
                                    metadata=metadata,
                                    source=GEODataSource.GEOPARSE,
                                    processing_info={
                                        "method": "emergency_fallback_single_sample",
                                        "sample_id": gsm_id,
                                        "data_type": self._determine_data_type_from_metadata(metadata),
                                        "note": "Emergency fallback - only partial data recovered"
                                    },
                                    success=True
                                )
                    except Exception as e:
                        logger.debug(f"Could not get data from sample {gsm_id}: {e}")
                        continue
            
            return GEOResult(success=False, error_message="Emergency fallback could not recover any data")
            
        except Exception as e:
            logger.error(f"Error in emergency fallback pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

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
                            concatinated_dataset_annDataObject = self._concatenate_stored_samples(
                                geo_id, stored_samples, use_intersecting_genes_only
                            )
                            
                            if concatinated_dataset_annDataObject is not None:
                                return GEOResult(
                                    data=concatinated_dataset_annDataObject,
                                    metadata=metadata,
                                    source=GEODataSource.GEOPARSE,
                                    processing_info={
                                        "method": "geoparse_samples_concatenated", 
                                        "data_type" : data_type,
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


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                   METADATA AND VALIDATION UTILITIES                       ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

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

            # Define which fields should remain as lists vs be joined as strings
            # Fields that need to remain as lists for downstream processing
            LIST_FIELDS = {
                'supplementary_file', 'relation', 'sample_id', 'platform_id', 
                'platform_taxid', 'sample_taxid'
            }
            
            # Fields that should be joined as strings for display/summary
            STRING_FIELDS = {
                'title', 'summary', 'overall_design', 'type', 'contributor',
                'contact_name', 'contact_email', 'contact_phone', 'contact_department',
                'contact_institute', 'contact_address', 'contact_city', 
                'contact_zip/postal_code', 'contact_country', 'geo_accession',
                'status', 'submission_date', 'last_update_date', 'pubmed_id', 'web_link'
            }

            # Basic metadata from GEOparse
            if hasattr(gse, "metadata"):
                for key, value in gse.metadata.items():
                    if isinstance(value, list):
                        # Keep file-related and ID fields as lists for downstream processing
                        if key in LIST_FIELDS:
                            metadata[key] = value
                        # Join descriptive/text fields as strings for summary generation  
                        elif key in STRING_FIELDS:
                            metadata[key] = ', '.join(value) if value else ''
                        else:
                            # For unknown fields, use a conservative approach:
                            # If it looks like a file/ID field, keep as list; otherwise join
                            if any(keyword in key.lower() for keyword in ['file', 'url', 'id', 'accession']):
                                metadata[key] = value
                            else:
                                metadata[key] = ', '.join(value) if value else ''
                    else:
                        metadata[key] = value

            # Platform information - keep as structured dict
            if hasattr(gse, "gpls"):
                platforms = {}
                for gpl_id, gpl in gse.gpls.items():
                    platforms[gpl_id] = {
                        "title": self._safely_extract_metadata_field(gpl, "title"),
                        "organism": self._safely_extract_metadata_field(gpl, "organism"),
                        "technology": self._safely_extract_metadata_field(gpl, "technology"),
                    }
                metadata["platforms"] = platforms

            # Sample metadata - keep as structured dict
            if hasattr(gse, "gsms"):
                sample_metadata = {}
                for gsm_id, gsm in gse.gsms.items():
                    sample_meta = {}
                    if hasattr(gsm, "metadata"):
                        for key, value in gsm.metadata.items():
                            if isinstance(value, list):
                                # For sample-level metadata, preserve lists for characteristics
                                # but join others for display
                                if key in ['characteristics_ch1', 'supplementary_file'] or 'file' in key.lower():
                                    sample_meta[key] = value
                                else:
                                    sample_meta[key] = ', '.join(value) if value else ''
                            else:
                                sample_meta[key] = value
                    sample_metadata[gsm_id] = sample_meta
                metadata["samples"] = sample_metadata

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _safely_extract_metadata_field(self, obj, field_name: str, default: str = "") -> str:
        """
        Safely extract a metadata field from a GEOparse object.
        
        Handles the common pattern of checking for metadata attribute,
        extracting the field, and joining all elements if it's a list.
        
        Args:
            obj: GEOparse object (GSE, GPL, or GSM)
            field_name: Name of the metadata field to extract
            default: Default value if field is not found
            
        Returns:
            str: Extracted metadata value or default
        """
        try:
            if not hasattr(obj, "metadata"):
                return default
            
            field_value = obj.metadata.get(field_name, [default])
            
            # Handle list values by joining all elements
            if isinstance(field_value, list):
                return ', '.join(field_value) if field_value else default
            
            return str(field_value) if field_value else default
            
        except (AttributeError, IndexError, KeyError):
            return default

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
            # Extract key information with safe string conversion
            title = str(metadata.get('title', 'N/A')).strip()
            summary = str(metadata.get('summary', 'N/A')).strip()
            overall_design = str(metadata.get('overall_design', 'N/A')).strip()
            
            # Sample information
            samples = metadata.get('samples', {})
            if not isinstance(samples, dict):
                samples = {}
            sample_count = len(samples)
            
            # Platform information with safe handling
            platforms = metadata.get('platforms', {})
            if not isinstance(platforms, dict):
                platforms = {}
            platform_info = []
            for platform_id, platform_data in platforms.items():
                if isinstance(platform_data, dict):
                    title_info = platform_data.get('title', 'N/A')
                    platform_info.append(f"{platform_id}: {title_info}")
                else:
                    platform_info.append(f"{platform_id}: N/A")
            
            # Contact information with safe string conversion
            contact_name = str(metadata.get('contact_name', 'N/A')).strip()
            contact_institute = str(metadata.get('contact_institute', 'N/A')).strip()
            
            # Publication info
            pubmed_id = str(metadata.get('pubmed_id', 'Not available')).strip()
            
            # Dates with safe string conversion
            submission_date = str(metadata.get('submission_date', 'N/A')).strip()
            last_update = str(metadata.get('last_update_date', 'N/A')).strip()
            
            # Sample characteristics preview with safe handling
            sample_preview = []
            for i, (sample_id, sample_data) in enumerate(samples.items()):
                if i < 3:  # Show first 3 samples
                    if isinstance(sample_data, dict):
                        chars = sample_data.get('characteristics_ch1', [])
                        if isinstance(chars, list) and chars:
                            sample_preview.append(f"  - {sample_id}: {str(chars[0]).strip()}")
                        else:
                            title_info = sample_data.get('title', 'No title')
                            sample_preview.append(f"  - {sample_id}: {str(title_info).strip()}")
                    else:
                        sample_preview.append(f"  - {sample_id}: No title")
            
            if sample_count > 3:
                sample_preview.append(f"  ... and {sample_count - 3} more samples")
            
            # Robust validation status handling with type checking
            validation_status = 'UNKNOWN'
            alignment_pct_raw = 'UNKNOWN'
            alignment_pct_formatted = 'UNKNOWN'
            predicted_type = 'UNKNOWN'
            aligned_fields = 'UNKNOWN'
            missing_fields = 'UNKNOWN'
            
            if validation_result and isinstance(validation_result, dict):
                # Validation status
                validation_status = str(validation_result.get('validation_status', 'UNKNOWN')).strip()
                
                # Alignment percentage with robust type handling
                alignment_raw = validation_result.get('alignment_percentage', None)
                if alignment_raw is not None:
                    try:
                        # Try to convert to float
                        alignment_float = float(alignment_raw)
                        alignment_pct_raw = alignment_float
                        alignment_pct_formatted = f"{alignment_float:.1f}"
                    except (ValueError, TypeError):
                        # If conversion fails, use string representation
                        alignment_pct_raw = str(alignment_raw)
                        alignment_pct_formatted = str(alignment_raw)
                
                # Predicted type
                predicted_type_raw = validation_result.get('predicted_data_type', 'unknown')
                if predicted_type_raw:
                    predicted_type = str(predicted_type_raw).replace('_', ' ').title()
                
                # Schema field counts with robust type handling
                aligned_raw = validation_result.get('schema_aligned_fields', None)
                if aligned_raw is not None:
                    try:
                        aligned_fields = int(aligned_raw)
                    except (ValueError, TypeError):
                        aligned_fields = str(aligned_raw)
                
                missing_raw = validation_result.get('schema_missing_fields', None)
                if missing_raw is not None:
                    try:
                        missing_fields = int(missing_raw)
                    except (ValueError, TypeError):
                        missing_fields = str(missing_raw)
            
            # Format the summary with safe string formatting
            summary_text = f"""📊 **GEO Dataset Metadata Summary: {geo_id}**

🔬 **Study Information:**
- **Title:** {title}
- **Summary:** {summary}
- **Design:** {overall_design}
- **Predicted Type:** {predicted_type}

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
- **Schema Alignment:** {alignment_pct_formatted}% of expected fields present
- **Aligned Fields:** {aligned_fields}
- **Missing Fields:** {missing_fields}

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
            logger.exception("Full traceback for metadata formatting error:")
            return f"Error formatting metadata summary for {geo_id}: {str(e)}"


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                     CORE PROCESSING UTILITIES                             ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

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
                        matrix = self.geo_parser.parse_10x_data(sample_extract_dir, sample_id)
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
                        matrix = self.geo_parser.parse_expression_file(file_path)
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
        Download and parse a single expression file with progress tracking.

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
                # Use geo_downloader for progress tracking and better protocol support
                if not self.geo_downloader.download_file(file_url, local_file):
                    logger.error(f"Failed to download file: {file_url}")
                    return None
                logger.info(f"Successfully downloaded: {local_file}")
            else:
                logger.info(f"Using cached file: {local_file}")

            return self.geo_parser.parse_expression_file(local_file)

        except Exception as e:
            logger.error(f"Error downloading and parsing file: {e}")
            return None

    # Parsing functions have been moved to geo_parser.py for better separation of concerns
    # and reusability across different modalities

    def _concatenate_matrices(
        self, validated_matrices: Dict[str, pd.DataFrame], geo_id: str, use_intersecting_genes_only: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Concatenate multiple sample matrices into a single DataFrame.
        
        Args:
            validated_matrices: Dictionary of sample matrices
            geo_id: GEO series ID
            use_intersecting_genes_only: Whether to use only common genes
            
        Returns:
            Combined DataFrame or None
        """
        try:
            if not validated_matrices:
                logger.warning("No matrices to concatenate")
                return None
            
            logger.info(f"Concatenating {len(validated_matrices)} matrices for {geo_id}")
            
            matrices_list = list(validated_matrices.values())
            
            if use_intersecting_genes_only:
                # Find common genes across all matrices
                common_genes = set(matrices_list[0].columns)
                for matrix in matrices_list[1:]:
                    common_genes = common_genes.intersection(set(matrix.columns))
                
                if not common_genes:
                    logger.error("No common genes found across matrices")
                    return None
                
                # Filter matrices to common genes
                filtered_matrices = [matrix[list(common_genes)] for matrix in matrices_list]
                combined_matrix = pd.concat(filtered_matrices, axis=0, sort=False)
                logger.info(f"Concatenated with {len(common_genes)} common genes")
                
            else:
                # Use all genes, filling missing with zeros
                combined_matrix = pd.concat(matrices_list, axis=0, sort=False).fillna(0)
                logger.info(f"Concatenated with all genes, filled missing with zeros")
            
            logger.info(f"Final combined matrix shape: {combined_matrix.shape}")
            return combined_matrix
            
        except Exception as e:
            logger.error(f"Error concatenating matrices: {e}")
            return None


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                   SAMPLE PROCESSING AND VALIDATION                        ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

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
            logger.error(f"Error getting sample info: {e}")
            return {}

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
                df = self._download_and_combine_single_cell_files(suppl_files_mapped, gsm_id)
                return df
                
        except Exception as e:
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
        Extract and classify supplementary files using robust pattern matching and scoring.
        
        This function uses a professional scoring system with regex patterns to classify
        supplementary files more accurately, handling edge cases and variations in naming
        conventions commonly found in GEO datasets.
        
        Args:
            metadata: Sample metadata dictionary
            gsm_id: GEO sample ID for logging
            
        Returns:
            Dict[str, str]: Dictionary mapping file types to URLs with highest confidence scores
        """
        try:
            # Initialize file type definitions with regex patterns and confidence scores
            file_type_patterns = self._initialize_file_type_patterns()
            
            # Find all supplementary file URLs
            file_urls = self._extract_all_supplementary_urls(metadata, gsm_id)
            
            if not file_urls:
                logger.warning(f"No supplementary files found for {gsm_id}")
                return {}
            
            # Score and classify each file
            classified_files = {}
            file_scores = {}  # Track scores for debugging
            
            for url in file_urls:
                filename = url.split('/')[-1]
                file_classification = self._classify_single_file(filename, url, file_type_patterns)
                
                for file_type, score in file_classification.items():
                    if score > 0:  # Only consider positive scores
                        # Keep track of best score for each file type
                        if file_type not in classified_files or score > file_scores.get(file_type, 0):
                            classified_files[file_type] = url
                            file_scores[file_type] = score
                            logger.debug(f"Updated {file_type}: {filename} (score: {score:.2f})")
            
            # Report final classifications
            logger.info(f"Final classification for {gsm_id}:")
            for file_type, url in classified_files.items():
                filename = url.split('/')[-1]
                score = file_scores[file_type]
                logger.info(f"  {file_type}: {filename} (confidence: {score:.2f})")
            
            # Validate 10X trio completeness if matrix found
            if 'matrix' in classified_files:
                self._validate_10x_trio_completeness(classified_files, gsm_id)
            
            return classified_files
            
        except Exception as e:
            logger.error(f"Error extracting supplementary files from metadata for {gsm_id}: {e}")
            return {}

    def _initialize_file_type_patterns(self) -> Dict[str, Dict[str, Union[List[re.Pattern], float]]]:
        """
        Initialize comprehensive file type classification patterns with confidence scoring.
        
        Returns:
            Dict containing regex patterns and base scores for each file type
        """
        patterns = {
            'matrix': {
                'patterns': [
                    # High confidence patterns (exact matches)
                    re.compile(r'matrix\.mtx(\.gz)?$', re.IGNORECASE),
                    re.compile(r'matrix\.txt(\.gz)?$', re.IGNORECASE),
                    re.compile(r'matrix\.csv(\.gz)?$', re.IGNORECASE),
                    re.compile(r'matrix\.tsv(\.gz)?$', re.IGNORECASE),
                    
                    # Medium confidence patterns (with context)
                    re.compile(r'.*_matrix\.(mtx|txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*-matrix\.(mtx|txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*\.matrix\.(mtx|txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    
                    # Lower confidence patterns (broader matches)
                    re.compile(r'.*matrix.*\.(mtx|txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*(count|expr|expression).*matrix.*', re.IGNORECASE),
                ],
                'base_score': 1.0,
                'boost_keywords': ['count', 'expression', 'sparse', '10x', 'chromium']
            },
            
            'barcodes': {
                'patterns': [
                    # High confidence patterns
                    re.compile(r'barcodes\.tsv(\.gz)?$', re.IGNORECASE),
                    re.compile(r'barcodes\.txt(\.gz)?$', re.IGNORECASE),
                    re.compile(r'barcodes\.csv(\.gz)?$', re.IGNORECASE),
                    
                    # Medium confidence patterns
                    re.compile(r'.*_barcode.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*-barcode.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*\.barcode.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    
                    # Lower confidence patterns
                    re.compile(r'.*barcode.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*(cell|bc).*id.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                ],
                'base_score': 1.0,
                'boost_keywords': ['cell', '10x', 'chromium', 'droplet']
            },
            
            'features': {
                'patterns': [
                    # High confidence patterns  
                    re.compile(r'features\.tsv(\.gz)?$', re.IGNORECASE),
                    re.compile(r'genes\.tsv(\.gz)?$', re.IGNORECASE),
                    re.compile(r'features\.txt(\.gz)?$', re.IGNORECASE),
                    re.compile(r'genes\.txt(\.gz)?$', re.IGNORECASE),
                    
                    # Medium confidence patterns
                    re.compile(r'.*_feature.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*_gene.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*-feature.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*-gene.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    
                    # Lower confidence patterns
                    re.compile(r'.*feature.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*gene.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*annotation.*\.(tsv|txt|csv)(\.gz)?$', re.IGNORECASE),
                ],
                'base_score': 1.0,
                'boost_keywords': ['gene', 'ensembl', 'symbol', 'annotation', '10x']
            },
            
            'h5_data': {
                'patterns': [
                    # High confidence patterns
                    re.compile(r'.*\.h5$', re.IGNORECASE),
                    re.compile(r'.*\.h5ad$', re.IGNORECASE),
                    re.compile(r'.*\.hdf5$', re.IGNORECASE),
                    
                    # Medium confidence patterns
                    re.compile(r'.*_filtered.*\.h5$', re.IGNORECASE),
                    re.compile(r'.*_raw.*\.h5$', re.IGNORECASE),
                    re.compile(r'.*matrix.*\.h5$', re.IGNORECASE),
                ],
                'base_score': 1.0,
                'boost_keywords': ['filtered', 'raw', 'matrix', '10x', 'chromium']
            },
            
            'expression': {
                'patterns': [
                    # High confidence patterns for expression files
                    re.compile(r'.*(expr|expression|count|fpkm|tpm|rpkm).*\.(txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    
                    # Medium confidence patterns
                    re.compile(r'.*_counts?\.(txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*_expr\.(txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*_data\.(txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                    
                    # Lower confidence patterns (generic data files)
                    re.compile(r'.*\.(txt|csv|tsv)(\.gz)?$', re.IGNORECASE),
                ],
                'base_score': 0.5,  # Lower base score for generic expression
                'boost_keywords': ['normalized', 'filtered', 'processed', 'log']
            },
            
            'archive': {
                'patterns': [
                    re.compile(r'.*\.tar(\.gz)?$', re.IGNORECASE),
                    re.compile(r'.*\.zip$', re.IGNORECASE),
                    re.compile(r'.*\.rar$', re.IGNORECASE),
                ],
                'base_score': 0.8,
                'boost_keywords': ['raw', 'supplementary', 'all', 'complete']
            }
        }
        
        return patterns

    def _extract_all_supplementary_urls(self, metadata: Dict[str, Any], gsm_id: str) -> List[str]:
        """
        Extract all supplementary file URLs from metadata using flexible key detection.
        
        Args:
            metadata: Sample metadata dictionary
            gsm_id: GEO sample ID for logging
            
        Returns:
            List[str]: All supplementary file URLs found
        """
        file_urls = []
        
        # Look for various supplementary file keys (case-insensitive)
        supplementary_key_patterns = [
            re.compile(r'.*supplement.*file.*', re.IGNORECASE),
            re.compile(r'.*suppl.*file.*', re.IGNORECASE),
            re.compile(r'.*additional.*file.*', re.IGNORECASE),
            re.compile(r'.*raw.*file.*', re.IGNORECASE),
            re.compile(r'.*data.*file.*', re.IGNORECASE),
        ]
        
        # Find matching keys
        matching_keys = []
        for key in metadata.keys():
            for pattern in supplementary_key_patterns:
                if pattern.match(key):
                    matching_keys.append(key)
                    break
        
        if not matching_keys:
            # Fallback to any key containing 'supplementary'
            matching_keys = [key for key in metadata.keys() if 'supplement' in key.lower()]
        
        logger.debug(f"Found supplementary keys for {gsm_id}: {matching_keys}")
        
        # Extract URLs from matching keys
        for key in matching_keys:
            urls = metadata[key]
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list):
                continue
                
            for url in urls:
                if url and isinstance(url, str) and ('http' in url or 'ftp' in url):
                    file_urls.append(url)
        
        logger.info(f"Extracted {len(file_urls)} supplementary file URLs for {gsm_id}")
        return file_urls

    def _classify_single_file(self, filename: str, url: str, patterns: Dict) -> Dict[str, float]:
        """
        Classify a single file using pattern matching and scoring.
        
        Args:
            filename: Name of the file
            url: Full URL of the file
            patterns: File type patterns dictionary
            
        Returns:
            Dict[str, float]: Scores for each file type
        """
        scores = {}
        filename_lower = filename.lower()
        url_lower = url.lower()
        
        for file_type, type_config in patterns.items():
            type_patterns = type_config['patterns']
            base_score = type_config['base_score']
            boost_keywords = type_config.get('boost_keywords', [])
            
            max_pattern_score = 0.0
            
            # Check each pattern for this file type
            for i, pattern in enumerate(type_patterns):
                if pattern.search(filename):
                    # Higher index patterns have lower confidence
                    pattern_confidence = 1.0 - (i * 0.1)  # Decreasing confidence
                    max_pattern_score = max(max_pattern_score, pattern_confidence)
            
            if max_pattern_score > 0:
                # Apply base score
                total_score = base_score * max_pattern_score
                
                # Apply keyword boosts
                keyword_boost = 0.0
                for keyword in boost_keywords:
                    if keyword in filename_lower or keyword in url_lower:
                        keyword_boost += 0.1
                
                total_score += keyword_boost
                scores[file_type] = min(total_score, 2.0)  # Cap at 2.0
        
        return scores

    def _validate_10x_trio_completeness(self, classified_files: Dict[str, str], gsm_id: str) -> None:
        """
        Validate and report on 10X Genomics file trio completeness.
        
        Args:
            classified_files: Dictionary of classified file types and URLs
            gsm_id: GEO sample ID for logging
        """
        required_10x_files = {'matrix', 'barcodes', 'features'}
        found_10x_files = set(classified_files.keys()) & required_10x_files
        missing_10x_files = required_10x_files - found_10x_files
        
        if len(found_10x_files) == 3:
            logger.info(f"Complete 10X trio found for {gsm_id} ✓")
        elif len(found_10x_files) >= 1:
            logger.warning(f"Incomplete 10X trio for {gsm_id}. Found: {list(found_10x_files)}, Missing: {list(missing_10x_files)}")
            
            # Suggest alternatives
            if 'h5_data' in classified_files:
                logger.info(f"H5 format available as alternative for {gsm_id}")
            elif 'expression' in classified_files:
                logger.info(f"Generic expression file available as fallback for {gsm_id}")
        else:
            logger.info(f"No 10X format files detected for {gsm_id}")

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
            
            return combined_adata
            
        except ImportError:
            logger.error("anndata package is required but not installed. Install with: pip install anndata")
            return None
        except Exception as e:
            logger.error(f"Error concatenating stored samples: {e}")
            logger.exception("Full traceback:")
            return None


# ████████████████████████████████████████████████████████████████████████████████
# ██                                                                            ██
# ██                        LEGACY FUNCTIONS - REVIEW NEEDED                   ██
# ██                                                                            ██
# ████████████████████████████████████████████████████████████████████████████████

    def _store_single_sample_as_modality(self, gsm_id: str, matrix: pd.DataFrame, gsm) -> str:  #FIXME
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

    def _download_supplementary_file(  #FIXME
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

    def _create_placeholder_matrix(  #FIXME
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

    def _check_workspace_for_processed_data(  #FIXME
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

    def _format_download_response(  #FIXME
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
