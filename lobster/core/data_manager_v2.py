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
import threading
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union
from unittest.mock import Mock

import numpy as np
import pandas as pd

from lobster.core.adapters.base import BaseAdapter
from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.core.analysis_ir import create_data_loading_ir, create_data_saving_ir

# Import available backends and adapters
from lobster.core.backends.h5ad_backend import H5ADBackend
from lobster.core.interfaces.adapter import IModalityAdapter
from lobster.core.interfaces.backend import IDataBackend
from lobster.core.interfaces.validator import ValidationResult
from lobster.core.provenance import ProvenanceTracker
from lobster.core.queue_storage import atomic_write_json, queue_file_lock
from lobster.core.utils.h5ad_utils import validate_for_h5ad
from lobster.core.workspace import resolve_workspace

# Import for IR support (TYPE_CHECKING to avoid circular import)
if TYPE_CHECKING:
    from anndata import AnnData
    from plotly.graph_objects import Figure

    from lobster.core.analysis_ir import AnalysisStep
else:
    AnnData = Any
    Figure = Any

# Try to import ProteomicsAdapter - may not be available in public distribution
try:
    from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter

    PROTEOMICS_AVAILABLE = True
except ImportError:
    PROTEOMICS_AVAILABLE = False
    ProteomicsAdapter = None

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

_anndata_module = None
_nbformat_module = None
_plotly_go = None
_plotly_io = None


def _ensure_anndata():
    """Lazily import anndata with a helpful error message."""

    global _anndata_module
    if _anndata_module is None:
        try:
            import anndata as _anndata_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "anndata is required for DataManagerV2 operations. "
                "Install it with `pip install anndata`."
            ) from exc
    return _anndata_module


def _is_anndata_instance(obj: Any) -> bool:
    try:
        module = _ensure_anndata()
    except ImportError:
        return False
    return isinstance(obj, module.AnnData)


def _ensure_nbformat():
    """Lazily import nbformat only when needed."""

    global _nbformat_module
    if _nbformat_module is None:
        try:
            import nbformat as _nbformat_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "nbformat is required for notebook export features. "
                "Install it with `pip install nbformat`."
            ) from exc
    return _nbformat_module


def _ensure_plotly():
    """Lazily import Plotly graph objects and IO helpers."""

    global _plotly_go, _plotly_io
    if _plotly_go is None or _plotly_io is None:
        try:
            import plotly.graph_objects as go_module  # type: ignore
            import plotly.io as pio_module  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Plotly is required for visualization features. "
                "Install it with `pip install plotly`."
            ) from exc
        _plotly_go = go_module
        _plotly_io = pio_module
    return _plotly_go, _plotly_io


class MetadataEntry(TypedDict, total=False):
    """
    TypedDict for GEO metadata store entries.

    Enforces consistent structure across all metadata storage locations
    to prevent KeyError bugs when retrieving metadata.

    Structure:
        metadata (Dict[str, Any]): The actual GEO metadata from GEOparse
        validation (Dict[str, Any]): Validation information (optional)
        fetch_timestamp (str): ISO format timestamp when metadata was fetched
        strategy_config (Dict[str, Any]): Download strategy configuration (optional)
        stored_by (str): Component that stored the metadata
        modality_detection (Dict[str, Any]): Modality detection results (optional)
        concatenation_decision (Dict[str, Any]): Concatenation strategy (optional)
    """

    metadata: Dict[str, Any]
    validation: Dict[str, Any]
    fetch_timestamp: str
    strategy_config: Dict[str, Any]
    stored_by: str
    modality_detection: Dict[str, Any]
    concatenation_decision: Dict[str, Any]


class SuppressKaleidoLogging:
    """Context manager to temporarily suppress verbose Kaleido logging during image generation."""

    def __init__(self):
        self.kaleido_logger = logging.getLogger("kaleido")
        self.original_level = None

    def __enter__(self):
        self.original_level = self.kaleido_logger.level
        self.kaleido_logger.setLevel(logging.WARNING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            self.kaleido_logger.setLevel(self.original_level)


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
        console=None,
        auto_scan: bool = True,
    ):
        """
        Initialize DataManagerV2.

        Args:
            default_backend: Default storage backend to use
            workspace_path: Optional workspace directory for data storage.
                Resolution order: explicit path > LOBSTER_WORKSPACE env var > cwd/.lobster_workspace
            enable_provenance: Whether to enable provenance tracking
            console: Optional Rich console instance for progress tracking
            auto_scan: Whether to automatically scan workspace for available datasets
        """
        self.default_backend = default_backend
        self.workspace_path = resolve_workspace(
            explicit_path=workspace_path, create=True
        )
        self.enable_provenance = enable_provenance
        self.console = console  # Store console for progress tracking in tools

        # Core storage
        self.backends: Dict[str, IDataBackend] = {}
        self.adapters: Dict[str, IModalityAdapter] = {}
        self.modalities: Dict[str, "AnnData"] = {}

        # Workspace restoration attributes
        self.session_file = self.workspace_path / ".session.json"
        self._available_datasets: Dict[str, Dict] = {}
        self._auto_scan_on_access = auto_scan
        self.session_data: Optional[Dict] = None
        self.session_id = str(datetime.now().timestamp())  # Unique session ID

        # Cache workspace scan results with 30-second TTL (Time To Live)
        self._available_datasets_cache: Optional[Dict[str, Dict]] = None
        self._scan_timestamp: float = 0  # Timestamp of last scan
        self._scan_ttl: int = 30  # Cache TTL in seconds (configurable)
        self._timing_enabled = False
        self._latest_timings: Dict[str, float] = {}
        self.profile_timings_enabled = False

        # Metadata storage for GEO datasets and other sources temporary until data is loaded
        self.metadata_store: Dict[str, Dict[str, Any]] = {}

        # Download queue for dataset downloads (research_agent → data_expert handoff)
        # Lazily initialized queue directories
        self._queues_dir = self.workspace_path / ".lobster" / "queues"
        self._download_queue = None
        self._publication_queue = None
        self._publication_queue_unavailable = False

        # Processing log for user-facing messages
        self.processing_log: List[str] = []

        # Plot management
        self.latest_plots: List[Dict[str, Any]] = []  # Store plots with metadata
        self.plot_counter: int = 0  # Counter for generating unique IDs
        self.max_plots_history: int = 50  # Maximum number of plots to keep in history

        # Safety mechanisms to prevent infinite loops and concurrent saves
        self._save_lock = threading.Lock()  # Prevent concurrent saves
        self._save_in_progress = False  # Track if save is currently running
        self._last_save_time = 0  # Track last save timestamp for rate limiting
        self._min_save_interval = 2.0  # Minimum seconds between saves

        # Multi-process safe session file access
        self._session_lock = threading.Lock()
        self._session_lock_path = self.session_file.with_suffix(".lock")

        # Visualization state management for Visualization Expert Agent
        self.visualization_state = {
            "history": [],  # List of created visualizations
            "settings": {  # User preferences
                "default_width": 800,
                "default_height": 600,
                "color_scheme": "Set1",
                "save_by_default": True,
                "export_formats": ["html", "png"],
            },
            "plot_registry": {},  # UUID -> plot metadata mapping
        }

        # Current dataset management (for legacy compatibility)
        self.current_dataset: Optional[str] = None  # Name of current active modality
        self.current_data: Optional[pd.DataFrame] = None  # Legacy compatibility
        self.current_metadata: Dict[str, Any] = {}  # Legacy metadata
        self.adata: Optional["AnnData"] = None  # Legacy AnnData reference

        # Provenance tracking
        self.provenance = ProvenanceTracker() if enable_provenance else None

        # Notebook-based pipeline support (lazy initialization to avoid circular imports)
        self._notebook_exporter = None
        self._notebook_executor = None
        self._enable_notebooks = enable_provenance

        # Setup workspace
        self._setup_workspace()

        # Scan workspace for available datasets
        # Workspace scanning is deferred until first access via get_available_datasets()

        # Load session metadata if exists
        if self.session_file.exists():
            self._load_session_metadata()

        # Register default backends and adapters
        self._register_default_backends()
        self._register_default_adapters()

        # BUG009 FIX: Auto-load existing modalities from workspace (session persistence)
        if auto_scan:
            self._auto_load_modalities()

        logger.debug(f"Initialized DataManagerV2 with workspace: {self.workspace_path}")

    def _get_queues_dir(self) -> Path:
        """Ensure the queues directory exists before use."""

        self._queues_dir.mkdir(parents=True, exist_ok=True)
        return self._queues_dir

    @property
    def download_queue(self):  # type: ignore[override]
        if self._download_queue is None:
            from lobster.core.download_queue import DownloadQueue

            queue_file = self._get_queues_dir() / "download_queue.jsonl"
            self._download_queue = DownloadQueue(queue_file=queue_file)
        return self._download_queue

    @property
    def publication_queue(self):  # type: ignore[override]
        if self._publication_queue_unavailable:
            return None
        if self._publication_queue is None:
            try:
                from lobster.core.publication_queue import PublicationQueue
            except ImportError:
                logger.debug(
                    "Publication queue feature not available (premium feature)"
                )
                self._publication_queue_unavailable = True
                return None

            queue_file = self._get_queues_dir() / "publication_queue.jsonl"
            self._publication_queue = PublicationQueue(queue_file=queue_file)
        return self._publication_queue

    @property
    def notebook_exporter(self):
        """Lazy initialize notebook exporter."""
        if self._notebook_exporter is None and self._enable_notebooks:
            from lobster.core.notebook_exporter import NotebookExporter

            self._notebook_exporter = NotebookExporter(self.provenance, self)
        return self._notebook_exporter

    @property
    def notebook_executor(self):
        """Lazy initialize notebook executor."""
        if self._notebook_executor is None:
            from lobster.core.notebook_executor import NotebookExecutor

            self._notebook_executor = NotebookExecutor(self)
        return self._notebook_executor

    def _setup_workspace(self) -> None:
        """Set up workspace directories."""
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.data_dir = self.workspace_path / "data"
        self.exports_dir = self.workspace_path / "exports"
        self.cache_dir = self.workspace_path / "cache"
        self.literature_cache_dir = self.workspace_path / "literature_cache"

        for directory in [
            self.data_dir,
            self.exports_dir,
            self.cache_dir,
            self.literature_cache_dir,
        ]:
            directory.mkdir(exist_ok=True)

        # Create literature cache subdirectories
        (self.literature_cache_dir / "publications").mkdir(exist_ok=True)
        (self.literature_cache_dir / "parsed_docs").mkdir(exist_ok=True)

    def _register_default_backends(self) -> None:
        """Register default storage backends."""
        # H5AD backend
        self.register_backend("h5ad", H5ADBackend(base_path=self.data_dir))

        # MuData backend if available
        if MUDATA_BACKEND_AVAILABLE:
            self.register_backend("mudata", MuDataBackend(base_path=self.data_dir))

    def _register_default_adapters(self) -> None:
        """Register default modality adapters."""
        # Transcriptomics adapters
        self.register_adapter(
            "transcriptomics_single_cell",
            TranscriptomicsAdapter(data_type="single_cell", strict_validation=False),
        )

        self.register_adapter(
            "transcriptomics_bulk",
            TranscriptomicsAdapter(data_type="bulk", strict_validation=False),
        )

        # Proteomics adapters (only if available)
        if PROTEOMICS_AVAILABLE:
            self.register_adapter(
                "proteomics_ms",
                ProteomicsAdapter(
                    data_type="mass_spectrometry", strict_validation=False
                ),
            )

            self.register_adapter(
                "proteomics_affinity",
                ProteomicsAdapter(data_type="affinity", strict_validation=False),
            )

    def _auto_load_modalities(self) -> None:
        """
        Auto-load existing H5AD modalities from workspace data directory.

        BUG009 FIX: Enables session persistence by loading previously saved modalities
        when DataManagerV2 is initialized. This allows users to continue multi-step
        workflows across separate lobster query/chat sessions.

        Behavior:
        - Scans workspace/data/ directory for .h5ad files
        - Loads each file into self.modalities dict (key = filename without extension)
        - Skips files >2GB (memory safety - user must explicitly load large files)
        - Silently skips corrupted/incompatible files (logs warning)
        - Performance: Lazy loading on __init__ (happens once per session)

        Example:
            Session 1: lobster query "Analyze GSE123, create filtered modality"
                       → Saves: workspace/data/gse123_filtered.h5ad
            Session 2: lobster query "Cluster the filtered GSE123 data"
                       → Auto-loads: gse123_filtered from workspace
                       → Analysis continues seamlessly
        """
        if not self.data_dir.exists():
            return

        h5ad_files = list(self.data_dir.glob("*.h5ad"))
        if not h5ad_files:
            return

        logger.info(f"Auto-loading {len(h5ad_files)} existing modalities from workspace...")

        loaded_count = 0
        skipped_large = 0

        for h5ad_file in h5ad_files:
            try:
                # Safety: Skip very large files (>2GB) - user must explicitly load
                file_size_gb = h5ad_file.stat().st_size / (1024 ** 3)
                if file_size_gb > 2.0:
                    skipped_large += 1
                    logger.debug(
                        f"Skipped large file {h5ad_file.name} ({file_size_gb:.1f} GB > 2GB threshold). "
                        f"Use load_modality tool to explicitly load if needed."
                    )
                    continue

                modality_name = h5ad_file.stem  # Filename without .h5ad extension

                # Skip if modality already exists (prevent overwriting)
                if modality_name in self.modalities:
                    logger.debug(f"Modality '{modality_name}' already loaded, skipping")
                    continue

                # Load AnnData object
                import anndata
                adata = anndata.read_h5ad(h5ad_file)
                self.modalities[modality_name] = adata

                loaded_count += 1
                logger.debug(
                    f"Auto-loaded '{modality_name}': {adata.n_obs} obs × {adata.n_vars} vars"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to auto-load {h5ad_file.name}: {e}. "
                    f"File may be corrupted or require manual loading."
                )
                continue

        if loaded_count > 0:
            logger.info(f"✅ Auto-loaded {loaded_count} modalities from workspace")
        if skipped_large > 0:
            logger.info(
                f"Skipped {skipped_large} large files (>2GB). Use load_modality tool to load explicitly."
            )

    @property
    def available_datasets(self) -> Dict[str, Dict]:  # type: ignore[override]
        """Return cached dataset metadata, performing a scan if needed."""

        if (
            self._auto_scan_on_access
            and not self._available_datasets
            and self._available_datasets_cache is None
        ):
            try:
                self.get_available_datasets(force_refresh=True)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(f"Deferred workspace scan failed: {exc}")
        return self._available_datasets

    @available_datasets.setter
    def available_datasets(self, value: Dict[str, Dict]) -> None:
        self._available_datasets = value

    @contextmanager
    def _measure_step(self, name: str):
        if not self._timing_enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._latest_timings[name] = elapsed

    def enable_timing(self, enabled: bool = True) -> None:
        self._timing_enabled = enabled
        self.profile_timings_enabled = enabled
        if not enabled:
            self._latest_timings = {}

    def get_latest_timings(self, clear: bool = False) -> Dict[str, float]:
        timings = dict(self._latest_timings)
        if clear:
            self._latest_timings = {}
        return timings

    def register_backend(
        self, name: str, backend: IDataBackend, overwrite: bool = False
    ) -> None:
        """
        Register a storage backend.

        Args:
            name: Name for the backend
            backend: Backend implementation
            overwrite: If True, allow overwriting existing backend

        Raises:
            ValueError: If backend name already exists and overwrite=False
        """
        if name in self.backends:
            if overwrite:
                logger.warning(f"Overwriting existing backend: {name}")
            else:
                raise ValueError(
                    f"Backend '{name}' already registered. Use overwrite=True to replace."
                )

        self.backends[name] = backend
        logger.debug(f"Registered backend: {name} ({backend.__class__.__name__})")

    def register_adapter(
        self, name: str, adapter: IModalityAdapter, overwrite: bool = False
    ) -> None:
        """
        Register a modality adapter.

        Args:
            name: Name for the adapter
            adapter: Adapter implementation
            overwrite: If True, allow overwriting existing adapter

        Raises:
            ValueError: If adapter name already exists and overwrite=False
        """
        if name in self.adapters:
            if overwrite:
                logger.warning(f"Overwriting existing adapter: {name}")
            else:
                raise ValueError(
                    f"Adapter '{name}' already registered. Use overwrite=True to replace."
                )

        self.adapters[name] = adapter
        logger.debug(f"Registered adapter: {name} ({adapter.__class__.__name__})")

    def load_modality(
        self,
        name: str,
        source: Union[str, Path, pd.DataFrame, "AnnData"],
        adapter: str,
        validate: bool = True,
        **kwargs,
    ) -> "AnnData":
        """
        Load data for a specific modality.

        Args:
            name: Name for the modality
            source: Data source (file path, DataFrame, or AnnData)
            adapter: Name of adapter to use
            validate: Whether to validate the loaded data
            **kwargs: Additional parameters passed to adapter

        Returns:
            AnnData: Loaded and validated data

        Raises:
            ValueError: If adapter is not registered or validation fails
        """
        if adapter not in self.adapters:
            raise ValueError(f"Adapter '{adapter}' not registered")

        adapter_instance = self.adapters[adapter]

        # Load data using adapter
        if not _is_anndata_instance(source):
            adata = adapter_instance.from_source(source, **kwargs)
        else:
            adata = source

        # Validate data if requested
        if validate:
            validation_result = adapter_instance.validate(adata, strict=False)

            if validation_result.has_errors:
                error_msg = f"Validation failed for modality '{name}': {validation_result.errors}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if validation_result.has_warnings:
                logger.warning(
                    f"Validation warnings for modality '{name}': {validation_result.warnings}"
                )

        # Store modality
        self.modalities[name] = adata

        # Log provenance
        if self.provenance:
            self.provenance.create_entity(
                entity_type="modality_data",
                metadata={
                    "modality_name": name,
                    "adapter": adapter,
                    "shape": adata.shape,
                },
            )

            # Create IR for data loading operation
            source_path = (
                str(source) if not _is_anndata_instance(source) else "AnnData object"
            )
            loading_ir = create_data_loading_ir(
                input_param_name="input_data",
                description=f"Load {name} data from {source_path}",
            )

            # Log data loading operation with IR
            self.log_tool_usage(
                tool_name="load_dataset",
                parameters={
                    "name": name,
                    "source": source_path,
                    "adapter": adapter,
                    **kwargs,
                },
                description=f"Loaded modality '{name}' using {adapter} adapter",
                ir=loading_ir,
            )

            # Add provenance to AnnData
            adata = self.provenance.add_to_anndata(adata)

        logger.info(
            f"Loaded modality '{name}': {adata.shape} using adapter '{adapter}'"
        )
        return adata

    def save_modality(
        self, name: str, path: Union[str, Path], backend: Optional[str] = None, **kwargs
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
        with self._measure_step("dm:save_modality"):
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

            # Validate data for H5AD serialization (optional pre-save check)
            if backend_name in ["h5ad", "H5ADBackend"] and hasattr(adata, "uns"):
                validation_issues = validate_for_h5ad(adata.uns, path="adata.uns")
                if validation_issues:
                    logger.warning(
                        f"Pre-save validation found {len(validation_issues)} potential "
                        f"serialization issues in modality '{name}'. These will be "
                        f"sanitized automatically during save:\n"
                        + "\n".join(f"  - {issue}" for issue in validation_issues[:5])
                    )
                    if len(validation_issues) > 5:
                        logger.debug(
                            f"  ... and {len(validation_issues) - 5} more issues. "
                            f"Set logging to DEBUG for full list."
                        )

            # Save data
            backend_instance.save(adata, path, **kwargs)

            # Log provenance
            if self.provenance:
                self.provenance.create_entity(
                    entity_type="modality_data",
                    uri=path,
                    metadata={"modality_name": name, "shape": adata.shape},
                )

                # Create IR for data saving operation
                saving_ir = create_data_saving_ir(
                    output_prefix_param="output_prefix",
                    filename_suffix=Path(path).stem,
                    description=f"Save {name} data to {path}",
                    compression=kwargs.get("compression", "gzip"),
                )

                # Log data saving operation with IR
                self.log_tool_usage(
                    tool_name="save_dataset",
                    parameters={
                        "name": name,
                        "path": str(path),
                        "backend": backend_name,
                        **kwargs,
                    },
                    description=(
                        f"Saved modality '{name}' to {path} using {backend_name} backend"
                    ),
                    ir=saving_ir,
                )

            logger.info(
                f"Saved modality '{name}' to {path} using backend '{backend_name}'"
            )
            return str(path)

    def get_modality(self, name: str) -> "AnnData":
        """
        Get a specific modality.

        Args:
            name: Name of modality

        Returns:
            AnnData: The requested modality

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
            raise ImportError(
                "MuData is not available. Please install it with: pip install mudata"
            )

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

        logger.info(
            f"Created MuData with {len(modality_dict)} modalities: {list(modality_dict.keys())}"
        )
        return mdata

    def save_mudata(
        self, path: Union[str, Path], modalities: Optional[List[str]] = None, **kwargs
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

    def _match_modality_to_adapter(self, modality_name: str) -> Optional[str]:
        """
        Smart matching of modality names to adapter names.

        This method uses a hierarchical matching strategy to identify the correct
        adapter for a given modality name, handling various naming patterns.

        Args:
            modality_name: Name of the modality to match

        Returns:
            Optional[str]: Matching adapter name, or None if no match found
        """
        modality_lower = modality_name.lower()

        # Define keyword mappings for each adapter type
        adapter_keywords = {
            "transcriptomics_single_cell": {
                "primary": ["single_cell", "single-cell", "sc_", "scrna", "10x"],
                "secondary": ["transcriptom", "rna", "gene_expression"],
                "exclude": ["bulk"],
            },
            "transcriptomics_bulk": {
                "primary": ["bulk"],
                "secondary": ["transcriptom", "rna", "gene_expression"],
                "exclude": ["single_cell", "single-cell", "sc_", "scrna"],
            },
            "proteomics_ms": {
                "primary": ["ms", "mass_spec", "mass-spec", "mass_spectrometry"],
                "secondary": ["proteomic", "protein"],
                "exclude": ["affinity", "antibody"],
            },
            "proteomics_affinity": {
                "primary": ["affinity", "antibody", "immunoassay", "elisa", "western"],
                "secondary": ["proteomic", "protein"],
                "exclude": ["ms", "mass_spec"],
            },
        }

        # Score each adapter based on keyword matches
        adapter_scores = {}

        for adapter_name, keywords in adapter_keywords.items():
            score = 0

            # Check exclusion keywords first - if any match, skip this adapter
            if any(
                exclude in modality_lower for exclude in keywords.get("exclude", [])
            ):
                continue

            # Primary keywords have higher weight
            for primary in keywords.get("primary", []):
                if primary in modality_lower:
                    score += 10

            # Secondary keywords have lower weight
            for secondary in keywords.get("secondary", []):
                if secondary in modality_lower:
                    score += 5

            # Store score if > 0
            if score > 0:
                adapter_scores[adapter_name] = score

        # Return adapter with highest score, or None if no matches
        if adapter_scores:
            return max(adapter_scores.items(), key=lambda x: x[1])[0]

        # Fallback: try to infer from modality name patterns
        if "geo" in modality_lower or "gse" in modality_lower:
            # GEO datasets are usually transcriptomics
            if any(keyword in modality_lower for keyword in ["single", "sc", "10x"]):
                return "transcriptomics_single_cell"
            else:
                return "transcriptomics_bulk"

        return None

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

            # Use smart matching to find appropriate adapter
            matched_adapter_name = self._match_modality_to_adapter(modality)

            if matched_adapter_name and matched_adapter_name in self.adapters:
                adapter_instance = self.adapters[matched_adapter_name]
                logger.debug(
                    f"Matched modality '{modality}' to adapter '{matched_adapter_name}'"
                )
                return adapter_instance.get_quality_metrics(self.modalities[modality])
            else:
                # Use basic metrics if no specific adapter found
                logger.warning(
                    f"No specific adapter found for modality '{modality}', using basic metrics"
                )
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
                "cache": str(self.cache_dir),
            },
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
                    "obsm": list(adata.obsm.keys()) if adata.obsm else [],
                }

        # Add provenance info
        if self.provenance:
            status["provenance"] = {
                "n_activities": len(self.provenance.activities),
                "n_entities": len(self.provenance.entities),
                "n_agents": len(self.provenance.agents),
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
            # Use smart matching to find appropriate adapter
            matched_adapter_name = self._match_modality_to_adapter(name)

            if matched_adapter_name and matched_adapter_name in self.adapters:
                adapter_instance = self.adapters[matched_adapter_name]
                logger.debug(
                    f"Validating modality '{name}' with adapter '{matched_adapter_name}'"
                )
                results[name] = adapter_instance.validate(adata, strict=strict)
            else:
                # Use basic validation if no specific adapter found
                logger.warning(
                    f"No specific adapter found for modality '{name}', using basic validation"
                )
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

        with open(path, "w") as f:
            json.dump(provenance_data, f, indent=2, default=str)

        logger.info(f"Exported provenance to {path}")
        return str(path)

    def save_workspace(self, workspace_name: str) -> Path:
        """
        Save all current modalities to a named workspace directory.

        This method creates a dedicated directory for the workspace and saves
        all loaded modalities as individual .h5ad files. It also saves session
        metadata including the list of active modalities for later restoration.

        Args:
            workspace_name: Name for the workspace (will create a subdirectory)

        Returns:
            Path: Path to the workspace directory containing saved modalities

        Example:
            >>> dm.save_workspace("my_analysis_20250116")
            PosixPath('/path/.lobster_workspace/my_analysis_20250116')

        Note:
            - Uses thread-safe locking to prevent concurrent saves
            - Saves all modalities as {modality_name}.h5ad files
            - Creates session metadata for restoration via restore_session()
        """
        # Create workspace directory
        workspace_dir = self.workspace_path / workspace_name
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Save each modality
        saved_modalities = []
        for modality_name in self.modalities:
            try:
                save_path = workspace_dir / f"{modality_name}.h5ad"
                self.save_modality(modality_name, str(save_path))
                saved_modalities.append(modality_name)
                logger.debug(f"Saved modality '{modality_name}' to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save modality '{modality_name}': {e}")

        # Save session metadata
        session_metadata = {
            "workspace_name": workspace_name,
            "active_modalities": saved_modalities,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "n_modalities": len(saved_modalities),
        }

        metadata_path = workspace_dir / ".session.json"
        with open(metadata_path, "w") as f:
            json.dump(session_metadata, f, indent=2)

        logger.info(
            f"Saved workspace '{workspace_name}' with {len(saved_modalities)} modalities to {workspace_dir}"
        )
        return workspace_dir

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

    def clear(self) -> None:
        """
        Clear all modalities from memory without requiring confirmation.

        This is a convenience alias for clear_workspace() that doesn't require
        the confirm=True parameter. Use this for programmatic clearing in tests
        and scripts where confirmation prompts are not needed.

        Note:
            This method clears modalities and resets provenance but does not
            delete files from disk.

        Example:
            >>> dm.clear()  # Simple, no confirmation needed
        """
        # Clear modalities from memory
        self.modalities.clear()

        # Reset provenance
        if self.provenance:
            self.provenance = ProvenanceTracker()

        logger.info("Cleared workspace (via clear() alias)")

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
                "schema": adapter.get_schema(),
            }
        return info

    # Tool usage tracking via provenance
    def log_tool_usage(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        description: str = None,
        ir: Optional["AnalysisStep"] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Log tool usage for reproducibility tracking via W3C-PROV provenance system.

        Args:
            tool_name: Name of the tool used
            parameters: Parameters used with the tool
            description: Optional description of what was done
            ir: Optional AnalysisStep Intermediate Representation for notebook export

        Returns:
            Optional[Dict[str, Any]]: The created activity record, or None if no provenance

        Notes:
            The `ir` parameter enables automatic Jupyter notebook generation.
            Services should emit AnalysisStep objects alongside their results,
            and agents should pass these IR objects to this method for storage
            in the provenance record.
        """
        if self.provenance:
            activity_id = self.provenance.create_activity(
                activity_type=tool_name,
                agent="data_manager",
                parameters=parameters,
                description=description or f"{tool_name} operation",
                ir=ir,
            )

            # Find and return the activity dict
            for activity in self.provenance.activities:
                if activity["id"] == activity_id:
                    if ir is not None:
                        logger.info(
                            f"Tool usage logged with IR: {tool_name} (operation: {ir.operation})"
                        )
                    else:
                        logger.info(f"Tool usage logged: {tool_name}")
                    return activity

        return None

    def save_processed_data(
        self,
        processing_step: str,
        data_source: str = None,
        dataset_id: str = None,
        processing_params: Dict[str, Any] = None,
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
            import json

            from lobster.utils.file_naming import BioinformaticsFileNaming

            # Use the first modality if multiple exist
            modality_name = list(self.modalities.keys())[0]
            adata = self.modalities[modality_name]

            # Extract information from metadata or parameters
            data_source = data_source or "DATA"
            dataset_id = dataset_id or "unknown"

            # Generate professional filename with auto-selected extension
            filename = BioinformaticsFileNaming.generate_filename(
                data_source=data_source,
                dataset_id=dataset_id,
                processing_step=processing_step,
            )
            filepath = self.data_dir / filename

            # Save using appropriate backend
            self.save_modality(modality_name, filepath)

            # Create enhanced metadata
            enhanced_metadata = {
                "processing_step": processing_step,
                "data_source": data_source,
                "dataset_id": dataset_id,
                "saved_filename": filename,
                "saved_path": str(filepath),
                "save_timestamp": pd.Timestamp.now().isoformat(),
                "data_shape": list(adata.shape),
                "processing_params": processing_params or {},
                "processing_order": BioinformaticsFileNaming.get_processing_step_order(
                    processing_step
                ),
                "suggested_next_step": BioinformaticsFileNaming.suggest_next_step(
                    processing_step
                ),
                "modality_name": modality_name,
            }

            # Save metadata
            metadata_filename = BioinformaticsFileNaming.generate_metadata_filename(
                filename
            )
            metadata_path = self.data_dir / metadata_filename

            with open(metadata_path, "w") as f:
                json.dump(enhanced_metadata, f, indent=2, default=str)

            # Log the processing step
            self.processing_log.append(
                f"Saved {processing_step} data: {adata.shape[0]} obs × {adata.shape[1]} vars -> {filename}"
            )

            logger.info(f"Processed data saved with professional naming: {filepath}")
            logger.info(
                f"Next suggested step: {enhanced_metadata['suggested_next_step']}"
            )

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
                files_by_category["data"].append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

        # List export files
        for file_path in self.exports_dir.iterdir():
            if file_path.is_file():
                files_by_category["exports"].append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

        # List cache files
        for file_path in self.cache_dir.iterdir():
            if file_path.is_file():
                files_by_category["cache"].append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

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
                saved_items.append(
                    f"Modality '{modality_name}': {Path(saved_path).name}"
                )
            except Exception as e:
                logger.error(f"Failed to auto-save modality {modality_name}: {e}")

        # Save processing log and provenance
        if self.processing_log or (self.provenance and self.provenance.activities):
            try:
                log_path = self.exports_dir / "processing_log.json"
                log_data = {
                    "processing_log": self.processing_log,
                    "provenance_summary": {
                        "n_activities": (
                            len(self.provenance.activities) if self.provenance else 0
                        ),
                        "activities": [
                            {
                                "type": act.get("type"),
                                "timestamp": act.get("timestamp"),
                                "parameters": act.get("parameters", {}),
                            }
                            for act in (
                                self.provenance.activities if self.provenance else []
                            )
                        ],
                    },
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
        plot: "Figure",
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
            # DIAGNOSTIC: Track add_plot calls (should NOT happen during /plot command)
            import traceback

            call_stack = "".join(traceback.format_stack()[-3:-1])  # Get caller info
            logger.info(
                f"🔍 DIAGNOSTIC: add_plot() called - creating plot_{self.plot_counter + 1}"
            )
            logger.debug(f"🔍 DIAGNOSTIC: add_plot called from:\n{call_stack.strip()}")

            go_module, _ = _ensure_plotly()
            if not isinstance(plot, go_module.Figure):
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
                # Only append modality name if it's not already in the title (prevents duplication)
                if modality_name not in enhanced_title:
                    enhanced_title = (
                        f"{enhanced_title} ({modality_name} - {human_timestamp})"
                    )
                else:
                    # Modality name already present, just add timestamp
                    enhanced_title = f"{enhanced_title} ({human_timestamp})"
            elif current_dataset_info and "data_shape" in current_dataset_info:
                shape_info = f"{current_dataset_info['data_shape'][0]}x{current_dataset_info['data_shape'][1]}"
                enhanced_title = (
                    f"{enhanced_title} (Data: {shape_info} - {human_timestamp})"
                )
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

            # DIAGNOSTIC: Show current state after adding plot
            plot_ids = [p.get("id", "unknown") for p in self.latest_plots]
            logger.info(
                f"Plot added: '{enhanced_title}' with ID {plot_id} from {source}"
            )
            logger.info(
                f"🔍 DIAGNOSTIC: add_plot() completed. Current plots: {plot_ids}"
            )
            return plot_id

        except ValueError:
            # Re-raise ValueError as documented in docstring
            raise
        except Exception as e:
            logger.exception(f"Error in add_plot: {e}")
            return None

    def clear_plots(self) -> None:
        """Clear all stored plots."""
        self.latest_plots = []
        logger.info("All plots cleared")

    def get_plot_by_id(self, plot_id: str) -> Optional["Figure"]:
        """
        Get a plot by its unique ID.

        Args:
            plot_id: The unique ID of the plot

        Returns:
            Optional[Figure]: The plot if found, None otherwise
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
        with self._measure_step("dm:save_plots_to_workspace"):
            current_time = time.time()

            if self._save_in_progress:
                logger.warning(
                    "🔒 SAFETY: save_plots_to_workspace already in progress, skipping"
                )
                return []

            if current_time - self._last_save_time < self._min_save_interval:
                logger.warning(
                    f"🔒 SAFETY: Rate limited - last save was {current_time - self._last_save_time:.1f}s ago (min: {self._min_save_interval}s)"
                )
                return []

            if not self._save_lock.acquire(blocking=False):
                logger.warning(
                    "🔒 SAFETY: Could not acquire save lock, another save in progress"
                )
                return []

            try:
                self._save_in_progress = True
                self._last_save_time = current_time

                if not self.latest_plots:
                    logger.info("No plots to save")
                    return []

                plots_dir = self.workspace_path / "plots"
                plots_dir.mkdir(exist_ok=True)

                saved_files = []
                for plot_entry in self.latest_plots:
                    try:
                        plot = plot_entry["figure"]
                        plot_id = plot_entry["id"]
                        plot_title = plot_entry.get(
                            "original_title", plot_entry["title"]
                        )

                        if len(plot_title) > 80:
                            available_chars = 80 - 3
                            start_length = (available_chars + 1) // 2
                            end_length = available_chars // 2
                            plot_title = f"{plot_title[:start_length]}...{plot_title[-end_length:]}"

                        safe_title = "".join(
                            c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                        ).rstrip()
                        safe_title = safe_title.replace(" ", "_")
                        filename_base = (
                            f"{plot_id}_{safe_title}" if safe_title else plot_id
                        )

                        html_path = plots_dir / f"{filename_base}.html"
                        _, pio_module = _ensure_plotly()
                        pio_module.write_html(plot, html_path)
                        saved_files.append(str(html_path))

                        # BUG010 FIX: Skip PNG export for large datasets (Kaleido limitation)
                        # Kaleido hangs indefinitely when exporting PNG with custom hover data for >50K points
                        dataset_info = plot_entry.get("dataset_info", {})
                        n_cells = dataset_info.get("n_cells", 0)
                        skip_png = n_cells > 50000

                        if not skip_png:
                            png_path = plots_dir / f"{filename_base}.png"
                            try:
                                with SuppressKaleidoLogging():
                                    _, pio_module = _ensure_plotly()
                                    pio_module.write_image(plot, png_path)
                                saved_files.append(str(png_path))
                            except Exception as e:
                                logger.warning(f"Could not save PNG for {plot_id}: {e}")
                        else:
                            logger.info(
                                f"Skipped PNG export for {plot_id} ({n_cells:,} cells > 50K threshold). "
                                f"Large datasets cause Kaleido to hang. HTML version available."
                            )

                        logger.info(f"Saved plot {plot_id} to workspace")

                    except Exception as e:
                        logger.error(
                            f"Failed to save plot {plot_entry.get('id', 'unknown')}: {e}"
                        )

                plot_ids = [p.get("id", "unknown") for p in self.latest_plots]
                logger.info(
                    f"🔍 DIAGNOSTIC: save_plots_to_workspace() completed. Current plots: {plot_ids}"
                )

                return saved_files

            finally:
                self._save_in_progress = False
                self._save_lock.release()
                logger.debug("🔒 SAFETY: Released save lock and reset progress flag")

    # ========================================
    # VISUALIZATION STATE MANAGEMENT (Visualization Expert Agent)
    # ========================================

    def add_visualization_record(self, plot_id: str, metadata: Dict[str, Any]) -> None:
        """
        Track visualization creation in the visualization state.

        Args:
            plot_id: Unique identifier for the plot
            metadata: Metadata about the visualization
        """
        try:
            # Add to history
            self.visualization_state["history"].append(
                {
                    "plot_id": plot_id,
                    "timestamp": pd.Timestamp.now(),
                    "metadata": metadata,
                }
            )

            # Add to registry
            self.visualization_state["plot_registry"][plot_id] = metadata

            logger.debug(f"Added visualization record for plot {plot_id}")

        except Exception as e:
            logger.error(f"Failed to add visualization record: {e}")

    def get_visualization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent visualization history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List[Dict[str, Any]]: List of recent visualizations
        """
        try:
            return self.visualization_state["history"][-limit:]
        except Exception as e:
            logger.error(f"Failed to get visualization history: {e}")
            return []

    def get_visualization_settings(self) -> Dict[str, Any]:
        """
        Get current visualization settings.

        Returns:
            Dict[str, Any]: Current visualization settings
        """
        return self.visualization_state["settings"].copy()

    def update_visualization_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update visualization settings.

        Args:
            settings: Settings to update
        """
        try:
            self.visualization_state["settings"].update(settings)
            logger.debug(f"Updated visualization settings: {settings}")
        except Exception as e:
            logger.error(f"Failed to update visualization settings: {e}")

    def get_plot_by_uuid(self, plot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get plot metadata by UUID from visualization registry.

        Args:
            plot_id: UUID of the plot

        Returns:
            Optional[Dict[str, Any]]: Plot metadata if found, None otherwise
        """
        return self.visualization_state["plot_registry"].get(plot_id)

    def clear_visualization_history(self) -> None:
        """Clear visualization history and registry."""
        self.visualization_state["history"] = []
        self.visualization_state["plot_registry"] = {}
        logger.info("Cleared visualization history and registry")

    # ========================================
    # CURRENT DATASET MANAGEMENT (Legacy Support)
    # ========================================

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of current dataset or all modalities.
        Handles various data types robustly (h5ad, sparse matrices, etc.).

        Returns:
            Dict[str, Any]: Summary information
        """
        if not self.has_data():
            return {"status": "No modalities loaded"}

        if self.current_dataset and self.current_dataset in self.modalities:
            # Summary for current dataset
            adata = self.modalities[self.current_dataset]

            try:
                metrics = self.get_quality_metrics(self.current_dataset)
            except Exception as e:
                logger.warning(
                    f"Could not get quality metrics for {self.current_dataset}: {e}"
                )
                metrics = {"error": str(e)}

            # Safely get memory usage for different matrix types
            memory_usage = self._get_safe_memory_usage(adata.X)

            # Safely get data type information
            data_type_info = self._get_safe_data_type_info(adata.X)

            summary = {
                "status": "Modality loaded",
                "modality_name": self.current_dataset,
                "shape": self._get_safe_shape(adata),
                "data_type": data_type_info,
                "sample_names": self._get_safe_obs_names(
                    adata
                ),  # CLI expects 'sample_names'
                "columns": self._get_safe_obs_columns(
                    adata
                ),  # CLI expects 'columns' for obs columns
                "memory_usage": memory_usage,
                "is_sparse": self._is_sparse_matrix(adata.X),
                "metadata_keys": (
                    list(self.current_metadata.keys()) if self.current_metadata else []
                ),
                "processing_log": (
                    self.processing_log[-5:] if self.processing_log else []
                ),
                # Additional fields for advanced usage
                "obs_names": self._get_safe_obs_names(adata),
                "var_names": self._get_safe_var_names(adata),
                "obs_columns": self._get_safe_obs_columns(adata),
                "var_columns": self._get_safe_var_columns(adata),
                "layers": self._get_safe_layers(adata),
                "obsm": self._get_safe_obsm(adata),
                "uns": self._get_safe_uns(adata),
                "quality_metrics": metrics,
            }
        else:
            # Summary for all modalities
            modality_summaries = {}
            for mod_name, adata in self.modalities.items():
                try:
                    modality_summaries[mod_name] = {
                        "shape": self._get_safe_shape(adata),
                        "data_type": self._get_safe_data_type_info(adata.X),
                        "obs_columns": (
                            len(adata.obs.columns)
                            if hasattr(adata, "obs") and adata.obs is not None
                            else 0
                        ),
                        "var_columns": (
                            len(adata.var.columns)
                            if hasattr(adata, "var") and adata.var is not None
                            else 0
                        ),
                        "layers": (
                            len(adata.layers)
                            if hasattr(adata, "layers") and adata.layers
                            else 0
                        ),
                        "memory_usage": self._get_safe_memory_usage(adata.X),
                        "is_sparse": self._is_sparse_matrix(adata.X),
                    }
                except Exception as e:
                    logger.warning(
                        f"Error getting summary for modality {mod_name}: {e}"
                    )
                    modality_summaries[mod_name] = {
                        "error": str(e),
                        "shape": (0, 0),
                        "data_type": "unknown",
                    }

            # Safely calculate totals
            total_obs, total_vars = 0, 0
            for adata in self.modalities.values():
                try:
                    shape = self._get_safe_shape(adata)
                    total_obs += shape[0]
                    total_vars += shape[1]
                except Exception:
                    continue

            summary = {
                "status": f"{len(self.modalities)} modalities loaded",
                "modalities": modality_summaries,
                "total_obs": total_obs,
                "total_vars": total_vars,
                "metadata_keys": (
                    list(self.current_metadata.keys()) if self.current_metadata else []
                ),
                "processing_log": (
                    self.processing_log[-5:] if self.processing_log else []
                ),
            }

        return summary

    def _get_safe_memory_usage(self, X) -> str:
        """Safely get memory usage for different matrix types."""
        try:
            if X is None:
                return "N/A (No data matrix)"

            # Handle sparse matrices
            if hasattr(X, "nnz") and hasattr(X, "data"):  # Likely a sparse matrix
                # For sparse matrices, calculate memory from data + indices + indptr
                if hasattr(X, "data") and hasattr(X.data, "nbytes"):
                    data_bytes = (
                        int(X.data.nbytes) if not isinstance(X.data.nbytes, Mock) else 0
                    )
                    if hasattr(X, "indices") and hasattr(X.indices, "nbytes"):
                        indices_bytes = (
                            int(X.indices.nbytes)
                            if not isinstance(X.indices.nbytes, Mock)
                            else 0
                        )
                        data_bytes += indices_bytes
                    if hasattr(X, "indptr") and hasattr(X.indptr, "nbytes"):
                        indptr_bytes = (
                            int(X.indptr.nbytes)
                            if not isinstance(X.indptr.nbytes, Mock)
                            else 0
                        )
                        data_bytes += indptr_bytes
                    return f"{data_bytes / 1024**2:.2f} MB (sparse)"
                else:
                    return f"~{X.nnz * 12 / 1024**2:.2f} MB (sparse, estimated)"

            # Handle dense matrices
            elif hasattr(X, "nbytes"):
                return f"{X.nbytes / 1024**2:.2f} MB (dense)"

            # Handle arrays with size and dtype
            elif hasattr(X, "size") and hasattr(X, "dtype"):
                bytes_estimate = X.size * X.dtype.itemsize
                return f"{bytes_estimate / 1024**2:.2f} MB (estimated)"

            # Fallback for other types
            else:
                return "Unknown (unsupported matrix type)"

        except Exception as e:
            return f"Error calculating memory: {str(e)}"

    def _get_safe_data_type_info(self, X) -> str:
        """Safely get data type information."""
        try:
            if X is None:
                return "None"

            # Check if it's a sparse matrix
            if hasattr(X, "nnz"):
                matrix_type = type(X).__name__
                if hasattr(X, "dtype"):
                    return f"{matrix_type}[{X.dtype}]"
                else:
                    return f"{matrix_type}[unknown dtype]"

            # Dense array/matrix
            elif hasattr(X, "dtype"):
                return f"{type(X).__name__}[{X.dtype}]"

            # Other types
            else:
                return str(type(X).__name__)

        except Exception:
            return "unknown"

    def _get_safe_shape(self, adata) -> tuple:
        """Safely get shape from AnnData object."""
        try:
            if hasattr(adata, "shape"):
                return adata.shape
            elif hasattr(adata, "n_obs") and hasattr(adata, "n_vars"):
                return (adata.n_obs, adata.n_vars)
            elif (
                hasattr(adata, "X")
                and adata.X is not None
                and hasattr(adata.X, "shape")
            ):
                return adata.X.shape
            else:
                return (0, 0)
        except Exception:
            return (0, 0)

    def _get_safe_obs_names(self, adata) -> list:
        """Safely get observation names."""
        try:
            if hasattr(adata, "obs_names") and adata.obs_names is not None:
                return list(adata.obs_names[:5])
            elif (
                hasattr(adata, "obs")
                and adata.obs is not None
                and hasattr(adata.obs, "index")
            ):
                return list(adata.obs.index[:5])
            else:
                return []
        except Exception:
            return []

    def _get_safe_var_names(self, adata) -> list:
        """Safely get variable names."""
        try:
            if hasattr(adata, "var_names") and adata.var_names is not None:
                return list(adata.var_names[:5])
            elif (
                hasattr(adata, "var")
                and adata.var is not None
                and hasattr(adata.var, "index")
            ):
                return list(adata.var.index[:5])
            else:
                return []
        except Exception:
            return []

    def _get_safe_obs_columns(self, adata) -> list:
        """Safely get observation columns."""
        try:
            if (
                hasattr(adata, "obs")
                and adata.obs is not None
                and hasattr(adata.obs, "columns")
            ):
                return list(adata.obs.columns[:10])
            else:
                return []
        except Exception:
            return []

    def _get_safe_var_columns(self, adata) -> list:
        """Safely get variable columns."""
        try:
            if (
                hasattr(adata, "var")
                and adata.var is not None
                and hasattr(adata.var, "columns")
            ):
                return list(adata.var.columns[:10])
            else:
                return []
        except Exception:
            return []

    def _get_safe_layers(self, adata) -> list:
        """Safely get layer names."""
        try:
            if hasattr(adata, "layers") and adata.layers is not None:
                return list(adata.layers.keys())
            else:
                return []
        except Exception:
            return []

    def _get_safe_obsm(self, adata) -> list:
        """Safely get obsm keys."""
        try:
            if hasattr(adata, "obsm") and adata.obsm is not None:
                return list(adata.obsm.keys())
            else:
                return []
        except Exception:
            return []

    def _get_safe_uns(self, adata) -> list:
        """Safely get uns keys."""
        try:
            if hasattr(adata, "uns") and adata.uns is not None:
                return list(adata.uns.keys())
            else:
                return []
        except Exception:
            return []

    def _is_sparse_matrix(self, X) -> bool:
        """Check if matrix is sparse."""
        try:
            return hasattr(X, "nnz") or "sparse" in str(type(X)).lower()
        except Exception:
            return False

    def set_current_dataset(self, modality_name: str) -> None:
        """
        Set the current active dataset/modality.

        Args:
            modality_name: Name of the modality to set as current

        Raises:
            ValueError: If modality not found
        """
        if modality_name not in self.modalities:
            raise ValueError(f"Modality '{modality_name}' not found")

        self.current_dataset = modality_name
        self.adata = self.modalities[modality_name]

        # Update session file to reflect current dataset access
        self._update_session_file("accessed")

        logger.info(f"Set current dataset to: {modality_name}")

    def set_data(
        self, data: pd.DataFrame, metadata: Dict[str, Any] = None
    ) -> pd.DataFrame:
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
            self.load_modality(
                name=modality_name,
                source=data,
                adapter=adapter_name,
                validate=True,
                **self.current_metadata,
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
        validation_info: Dict[str, Any] = None,
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
            "stored_by": "DataManagerV2",
        }
        logger.info(f"Stored metadata for dataset: {dataset_id}")

    def _store_geo_metadata(
        self, geo_id: str, metadata: Dict[str, Any], stored_by: str, **kwargs
    ) -> MetadataEntry:
        """
        Store GEO metadata with enforced consistent structure.

        This helper method ensures all GEO metadata is stored with the same
        nested structure, preventing KeyError bugs when retrieving metadata.

        Args:
            geo_id: GEO dataset identifier (e.g., 'GSE12345')
            metadata: Raw GEO metadata dictionary from GEOparse
            stored_by: Component storing the metadata (for tracking)
            **kwargs: Optional additional fields:
                - validation (Dict): Validation information
                - strategy_config (Dict): Download strategy configuration
                - modality_detection (Dict): Modality detection results
                - concatenation_decision (Dict): Concatenation strategy

        Returns:
            MetadataEntry: The stored metadata entry

        Example:
            >>> entry = data_manager._store_geo_metadata(
            ...     geo_id="GSE12345",
            ...     metadata=geoparse_metadata,
            ...     stored_by="_check_platform_compatibility",
            ...     modality_detection={"modality": "single_cell", "confidence": 0.95}
            ... )
        """
        # Create entry with required fields
        entry: MetadataEntry = {
            "metadata": metadata.copy(),
            "fetch_timestamp": kwargs.get(
                "fetch_timestamp", datetime.now().isoformat()
            ),
            "stored_by": stored_by,
        }

        # Add optional fields if provided
        if "validation" in kwargs:
            entry["validation"] = kwargs["validation"]

        if "strategy_config" in kwargs:
            entry["strategy_config"] = kwargs["strategy_config"]

        if "modality_detection" in kwargs:
            entry["modality_detection"] = kwargs["modality_detection"]

        if "concatenation_decision" in kwargs:
            entry["concatenation_decision"] = kwargs["concatenation_decision"]

        # Store with nested structure
        self.metadata_store[geo_id] = entry

        logger.debug(
            f"Stored GEO metadata for {geo_id} with nested structure "
            f"(stored_by: {stored_by}, keys: {list(entry.keys())})"
        )

        return entry

    def _get_geo_metadata(self, geo_id: str) -> Optional[MetadataEntry]:
        """
        Safely retrieve GEO metadata with structure validation.

        This helper method retrieves GEO metadata from the store and validates
        that it has the expected nested structure.

        Args:
            geo_id: GEO dataset identifier

        Returns:
            Optional[MetadataEntry]: Metadata entry if found and valid, None otherwise

        Example:
            >>> entry = data_manager._get_geo_metadata("GSE12345")
            >>> if entry:
            ...     raw_metadata = entry["metadata"]
            ...     strategy = entry.get("strategy_config", {})
        """
        if geo_id not in self.metadata_store:
            logger.debug(f"No metadata found for {geo_id}")
            return None

        stored_entry = self.metadata_store[geo_id]

        # Validate structure
        if not isinstance(stored_entry, dict):
            logger.warning(
                f"Metadata for {geo_id} is not a dictionary: {type(stored_entry)}"
            )
            return None

        if "metadata" not in stored_entry:
            logger.warning(
                f"Metadata for {geo_id} missing 'metadata' key. "
                f"Found keys: {list(stored_entry.keys())}. "
                f"This indicates a structure mismatch bug."
            )
            return None

        return stored_entry

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

    # ========================================
    # MACHINE LEARNING INTEGRATION METHODS
    # ========================================

    def check_ml_readiness(self, modality: str = None) -> Dict[str, Any]:
        """
        Check if modalities are ready for machine learning workflows.

        Args:
            modality: Specific modality to check (default: all modalities)

        Returns:
            Dict[str, Any]: ML readiness assessment
        """
        if modality:
            if modality not in self.modalities:
                raise ValueError(f"Modality '{modality}' not found")
            modalities_to_check = [modality]
        else:
            modalities_to_check = list(self.modalities.keys())

        if not modalities_to_check:
            return {"status": "error", "message": "No modalities loaded"}

        readiness_results = {}

        for mod_name in modalities_to_check:
            adata = self.modalities[mod_name]

            # Basic data structure checks
            checks = {
                "has_expression_data": adata.X is not None,
                "sufficient_samples": adata.n_obs >= 10,
                "sufficient_features": adata.n_vars >= 50,
                "no_missing_values": not np.any(
                    pd.isna(
                        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                    )
                ),
                "has_metadata": len(adata.obs.columns) > 0,
                "numeric_data": True,  # Assume true for now, could add numeric check
            }

            # Advanced checks based on modality type
            modality_type = self._detect_modality_type(mod_name)

            if modality_type in ["single_cell_rna_seq", "bulk_rna_seq"]:
                # Transcriptomics-specific checks
                checks.update(
                    {
                        "gene_symbols_available": "gene_symbols" in adata.var.columns
                        or adata.var.index.dtype == "object",
                        "count_data": (
                            np.all(adata.X >= 0)
                            if hasattr(adata.X, "__iter__")
                            else True
                        ),
                        "reasonable_gene_count": 500 <= adata.n_vars <= 50000,
                    }
                )
            elif "proteomics" in modality_type:
                # Proteomics-specific checks
                checks.update(
                    {
                        "protein_identifiers": len(adata.var.index) > 0,
                        "reasonable_protein_count": 10 <= adata.n_vars <= 10000,
                        "positive_values": (
                            np.all(adata.X >= 0)
                            if hasattr(adata.X, "__iter__")
                            else True
                        ),
                    }
                )

            # Calculate overall readiness score
            passed_checks = sum(checks.values())
            total_checks = len(checks)
            readiness_score = passed_checks / total_checks

            # Determine readiness level
            if readiness_score >= 0.9:
                readiness_level = "excellent"
            elif readiness_score >= 0.75:
                readiness_level = "good"
            elif readiness_score >= 0.5:
                readiness_level = "fair"
            else:
                readiness_level = "poor"

            # Compile results
            readiness_results[mod_name] = {
                "modality_type": modality_type,
                "shape": adata.shape,
                "readiness_score": readiness_score,
                "readiness_level": readiness_level,
                "checks": checks,
                "recommendations": self._generate_ml_recommendations(
                    checks, modality_type
                ),
            }

        # Overall assessment
        if len(readiness_results) == 1:
            return readiness_results[list(readiness_results.keys())[0]]
        else:
            return {
                "status": "success",
                "modalities": readiness_results,
                "overall_readiness": np.mean(
                    [r["readiness_score"] for r in readiness_results.values()]
                ),
            }

    def _generate_ml_recommendations(
        self, checks: Dict[str, bool], modality_type: str
    ) -> List[str]:
        """Generate ML-specific recommendations based on failed checks."""
        recommendations = []

        if not checks.get("sufficient_samples", True):
            recommendations.append(
                "Consider data augmentation or collecting more samples (minimum 10 recommended)"
            )

        if not checks.get("sufficient_features", True):
            recommendations.append(
                "Feature count is low - consider feature selection strategies"
            )

        if not checks.get("no_missing_values", True):
            recommendations.append(
                "Handle missing values through imputation or removal"
            )

        if not checks.get("has_metadata", True):
            recommendations.append("Add sample metadata for supervised learning tasks")

        if modality_type in ["single_cell_rna_seq", "bulk_rna_seq"]:
            if not checks.get("reasonable_gene_count", True):
                recommendations.append(
                    "Gene count outside typical range - verify data quality"
                )
            if not checks.get("count_data", True):
                recommendations.append(
                    "Negative values detected - ensure proper preprocessing for count data"
                )

        if len(recommendations) == 0:
            recommendations.append("Data appears ML-ready!")

        return recommendations

    def prepare_ml_features(
        self,
        modality: str,
        feature_selection: str = "variance",
        n_features: int = 2000,
        normalization: str = "log1p",
        scaling: str = "standard",
    ) -> Dict[str, Any]:
        """
        Prepare ML-ready feature matrices from biological data.

        Args:
            modality: Name of modality to process
            feature_selection: Method for feature selection ('variance', 'correlation', 'chi2', 'mutual_info')
            n_features: Number of features to select
            normalization: Normalization method ('log1p', 'cpm', 'none')
            scaling: Scaling method ('standard', 'minmax', 'robust', 'none')

        Returns:
            Dict[str, Any]: Processed feature information and metadata
        """
        if modality not in self.modalities:
            raise ValueError(f"Modality '{modality}' not found")

        adata = self.modalities[modality].copy()
        processing_steps = []

        # Import sklearn components (always needed)
        try:
            from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
        except ImportError:
            raise ImportError(
                "scikit-learn is required for ML feature preparation. Install with: pip install scikit-learn"
            )

        # Import scanpy separately
        try:
            import scanpy as sc

            scanpy_available = True
        except ImportError:
            scanpy_available = False
            logger.warning("Scanpy not available - using basic processing")

        # Step 1: Normalization
        if normalization == "log1p":
            if scanpy_available:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            else:
                # Basic log1p normalization
                X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                X = np.log1p(X / np.sum(X, axis=1, keepdims=True) * 1e4)
                adata.X = X
            processing_steps.append(f"Applied {normalization} normalization")
        elif normalization == "cpm":
            if scanpy_available:
                sc.pp.normalize_total(adata, target_sum=1e6)
            else:
                X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
                X = X / np.sum(X, axis=1, keepdims=True) * 1e6
                adata.X = X
            processing_steps.append("Applied CPM normalization")

        # Step 2: Feature Selection
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        if feature_selection == "variance" and scanpy_available:
            # Use scanpy's highly variable genes
            sc.pp.highly_variable_genes(
                adata, n_top_genes=min(n_features, adata.n_vars)
            )
            selected_features = adata.var["highly_variable"]
            selected_indices = np.where(selected_features)[0]
        elif feature_selection == "variance":
            # Basic variance-based selection
            variances = np.var(X, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]
        else:
            # Fallback to top variance features
            variances = np.var(X, axis=0)
            selected_indices = np.argsort(variances)[-n_features:]

        # Apply feature selection
        X_selected = X[:, selected_indices]
        selected_feature_names = adata.var_names[selected_indices]
        processing_steps.append(
            f"Selected {len(selected_indices)} features using {feature_selection}"
        )

        # Step 3: Scaling
        if scaling == "standard":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            processing_steps.append("Applied standard scaling")
        elif scaling == "minmax":
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_selected)
            processing_steps.append("Applied min-max scaling")
        elif scaling == "robust":
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            processing_steps.append("Applied robust scaling")
        else:
            X_scaled = X_selected
            scaler = None

        # Create feature matrix DataFrame
        feature_matrix = pd.DataFrame(
            X_scaled, index=adata.obs_names, columns=selected_feature_names
        )

        # Store processed data in a new modality
        processed_modality_name = f"{modality}_ml_features"

        # Create new AnnData with processed features
        anndata_module = _ensure_anndata()
        adata_processed = anndata_module.AnnData(
            X=X_scaled,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=selected_feature_names),
        )

        # Add processing metadata
        adata_processed.uns["ml_processing"] = {
            "source_modality": modality,
            "feature_selection": feature_selection,
            "n_features_selected": len(selected_indices),
            "normalization": normalization,
            "scaling": scaling,
            "processing_steps": processing_steps,
            "selected_indices": selected_indices.tolist(),
            "original_feature_names": list(adata.var_names),
            "timestamp": datetime.now().isoformat(),
        }

        # Store processed modality
        self.modalities[processed_modality_name] = adata_processed

        # Log processing
        self.processing_log.append(
            f"ML feature preparation: {modality} -> {processed_modality_name} "
            f"({adata.shape[0]} samples × {len(selected_indices)} features)"
        )

        return {
            "processed_modality": processed_modality_name,
            "feature_matrix": feature_matrix,
            "original_shape": adata.shape,
            "processed_shape": adata_processed.shape,
            "selected_features": list(selected_feature_names),
            "processing_steps": processing_steps,
            "scaler": scaler,
            "processing_metadata": adata_processed.uns["ml_processing"],
        }

    def create_ml_splits(
        self,
        modality: str,
        target_column: str = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Create stratified train/validation/test splits for ML workflows.

        Args:
            modality: Name of modality to split
            target_column: Column name in obs for stratification (if None, uses random splits)
            test_size: Proportion of data for test set
            validation_size: Proportion of remaining data for validation set
            stratify: Whether to stratify splits based on target_column
            random_state: Random seed for reproducibility

        Returns:
            Dict[str, Any]: Split information and indices
        """
        if modality not in self.modalities:
            raise ValueError(f"Modality '{modality}' not found")

        adata = self.modalities[modality]
        n_samples = adata.n_obs

        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise ImportError(
                "scikit-learn required for ML splits. Install with: pip install scikit-learn"
            )

        # Prepare indices and target
        sample_indices = np.arange(n_samples)

        if target_column and target_column in adata.obs.columns:
            target = adata.obs[target_column].values
            if stratify:
                # Check if stratification is possible
                unique_classes, counts = np.unique(target, return_counts=True)
                min_class_count = np.min(counts)
                min_required = max(2, int(1 / min(test_size, validation_size)))

                if min_class_count < min_required:
                    logger.warning(
                        f"Insufficient samples in some classes for stratification. "
                        f"Minimum class has {min_class_count} samples, need {min_required}"
                    )
                    stratify = False
                    target = None
        else:
            target = None
            stratify = False

        # Create train/temp split
        if stratify and target is not None:
            train_idx, temp_idx = train_test_split(
                sample_indices,
                test_size=test_size + validation_size,
                stratify=target,
                random_state=random_state,
            )
            temp_target = target[temp_idx]
        else:
            train_idx, temp_idx = train_test_split(
                sample_indices,
                test_size=test_size + validation_size,
                random_state=random_state,
            )
            temp_target = None

        # Create validation/test split from temp
        if len(temp_idx) > 1 and validation_size > 0:
            val_test_ratio = test_size / (test_size + validation_size)

            if stratify and temp_target is not None:
                val_idx, test_idx = train_test_split(
                    temp_idx,
                    test_size=val_test_ratio,
                    stratify=temp_target,
                    random_state=random_state,
                )
            else:
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=val_test_ratio, random_state=random_state
                )
        else:
            test_idx = temp_idx
            val_idx = np.array([])

        # Create split metadata
        splits = {
            "train": {
                "indices": train_idx.tolist(),
                "size": len(train_idx),
                "proportion": len(train_idx) / n_samples,
            },
            "validation": (
                {
                    "indices": val_idx.tolist(),
                    "size": len(val_idx),
                    "proportion": len(val_idx) / n_samples,
                }
                if len(val_idx) > 0
                else None
            ),
            "test": {
                "indices": test_idx.tolist(),
                "size": len(test_idx),
                "proportion": len(test_idx) / n_samples,
            },
        }

        # Add target distribution information
        if target is not None:
            for split_name, split_info in splits.items():
                if split_info is not None:
                    split_indices = split_info["indices"]
                    split_target = target[split_indices]
                    unique, counts = np.unique(split_target, return_counts=True)
                    split_info["target_distribution"] = dict(zip(unique, counts))

        # Store splits in modality metadata
        split_metadata = {
            "modality": modality,
            "target_column": target_column,
            "stratified": stratify,
            "test_size": test_size,
            "validation_size": validation_size,
            "random_state": random_state,
            "n_samples": n_samples,
            "splits": splits,
            "timestamp": datetime.now().isoformat(),
        }

        # Add to AnnData uns
        adata.uns["ml_splits"] = split_metadata

        # Log split creation
        self.processing_log.append(
            f"ML splits created for {modality}: "
            f"train({len(train_idx)}) / val({len(val_idx)}) / test({len(test_idx)})"
        )

        return split_metadata

    def export_for_ml_framework(
        self,
        modality: str,
        framework: str = "sklearn",
        split: str = None,
        target_column: str = None,
        output_dir: str = None,
    ) -> Dict[str, Any]:
        """
        Export data in formats suitable for ML frameworks.

        Args:
            modality: Name of modality to export
            framework: Target framework ('sklearn', 'pytorch', 'tensorflow', 'xgboost')
            split: Specific split to export ('train', 'validation', 'test', or None for all)
            target_column: Target column for supervised learning
            output_dir: Directory to save exports (defaults to exports_dir)

        Returns:
            Dict[str, Any]: Export information and file paths
        """
        if modality not in self.modalities:
            raise ValueError(f"Modality '{modality}' not found")

        adata = self.modalities[modality]

        # Set up output directory
        if output_dir is None:
            output_dir = self.exports_dir / "ml_exports"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get data matrix
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

        # Get target data if specified
        y = None
        if target_column and target_column in adata.obs.columns:
            y = adata.obs[target_column].values

        # Get splits if available
        splits_info = adata.uns.get("ml_splits", {})
        splits = splits_info.get("splits", {})

        # Prepare data for export
        export_info = {
            "modality": modality,
            "framework": framework,
            "shape": X.shape,
            "has_target": y is not None,
            "target_column": target_column,
            "export_timestamp": datetime.now().isoformat(),
            "files": {},
        }

        # Framework-specific export logic
        if framework == "sklearn":
            # Export as NumPy arrays and pickle files
            if split is None:
                # Export full dataset
                np.save(output_dir / f"{modality}_X.npy", X)
                export_info["files"]["features"] = str(output_dir / f"{modality}_X.npy")

                if y is not None:
                    np.save(output_dir / f"{modality}_y.npy", y)
                    export_info["files"]["target"] = str(
                        output_dir / f"{modality}_y.npy"
                    )
            else:
                # Export specific split
                if split in splits:
                    indices = splits[split]["indices"]
                    X_split = X[indices]
                    np.save(output_dir / f"{modality}_{split}_X.npy", X_split)
                    export_info["files"][f"{split}_features"] = str(
                        output_dir / f"{modality}_{split}_X.npy"
                    )

                    if y is not None:
                        y_split = y[indices]
                        np.save(output_dir / f"{modality}_{split}_y.npy", y_split)
                        export_info["files"][f"{split}_target"] = str(
                            output_dir / f"{modality}_{split}_y.npy"
                        )

        elif framework == "pytorch":
            # Export as PyTorch tensors
            try:
                import torch

                if split is None:
                    X_tensor = torch.FloatTensor(X)
                    torch.save(X_tensor, output_dir / f"{modality}_X.pt")
                    export_info["files"]["features"] = str(
                        output_dir / f"{modality}_X.pt"
                    )

                    if y is not None:
                        y_tensor = (
                            torch.LongTensor(y)
                            if y.dtype.kind in ["i", "u"]
                            else torch.FloatTensor(y)
                        )
                        torch.save(y_tensor, output_dir / f"{modality}_y.pt")
                        export_info["files"]["target"] = str(
                            output_dir / f"{modality}_y.pt"
                        )
                else:
                    if split in splits:
                        indices = splits[split]["indices"]
                        X_split = X[indices]
                        X_tensor = torch.FloatTensor(X_split)
                        torch.save(X_tensor, output_dir / f"{modality}_{split}_X.pt")
                        export_info["files"][f"{split}_features"] = str(
                            output_dir / f"{modality}_{split}_X.pt"
                        )

                        if y is not None:
                            y_split = y[indices]
                            y_tensor = (
                                torch.LongTensor(y_split)
                                if y_split.dtype.kind in ["i", "u"]
                                else torch.FloatTensor(y_split)
                            )
                            torch.save(
                                y_tensor, output_dir / f"{modality}_{split}_y.pt"
                            )
                            export_info["files"][f"{split}_target"] = str(
                                output_dir / f"{modality}_{split}_y.pt"
                            )

            except ImportError:
                logger.warning("PyTorch not available, falling back to NumPy export")
                framework = "sklearn"  # Fallback to sklearn format

        elif framework == "tensorflow":
            # Export as TensorFlow SavedModel or .npy for now
            logger.warning(
                "TensorFlow export not fully implemented, using NumPy format"
            )
            framework = "sklearn"  # Fallback

        # Also export metadata and feature names
        metadata = {
            "feature_names": list(adata.var_names),
            "sample_names": list(adata.obs_names),
            "shape": X.shape,
            "modality_metadata": dict(adata.obs.dtypes.astype(str)),
            "processing_info": adata.uns.get("ml_processing", {}),
            "splits_info": splits_info,
        }

        metadata_path = output_dir / f"{modality}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        export_info["files"]["metadata"] = str(metadata_path)

        # Log export
        self.processing_log.append(
            f"ML export completed: {modality} -> {framework} format "
            f"({len(export_info['files'])} files created)"
        )

        return export_info

    def get_ml_summary(self, modality: str = None) -> Dict[str, Any]:
        """
        Get comprehensive ML workflow summary for modalities.

        Args:
            modality: Specific modality to summarize (default: all modalities)

        Returns:
            Dict[str, Any]: ML workflow summary
        """
        if modality:
            if modality not in self.modalities:
                raise ValueError(f"Modality '{modality}' not found")
            modalities_to_summarize = [modality]
        else:
            modalities_to_summarize = list(self.modalities.keys())

        if not modalities_to_summarize:
            return {"status": "error", "message": "No modalities loaded"}

        ml_summaries = {}

        for mod_name in modalities_to_summarize:
            adata = self.modalities[mod_name]

            # Basic information
            summary = {
                "modality_name": mod_name,
                "modality_type": self._detect_modality_type(mod_name),
                "shape": adata.shape,
                "data_type": (
                    str(adata.X.dtype) if hasattr(adata.X, "dtype") else "unknown"
                ),
            }

            # ML readiness check
            readiness = self.check_ml_readiness(mod_name)
            summary["ml_readiness"] = {
                "score": readiness.get("readiness_score", 0),
                "level": readiness.get("readiness_level", "unknown"),
                "recommendations": readiness.get("recommendations", []),
            }

            # Feature processing information
            if "ml_processing" in adata.uns:
                processing_info = adata.uns["ml_processing"]
                summary["feature_processing"] = {
                    "processed": True,
                    "source_modality": processing_info.get("source_modality"),
                    "n_features_selected": processing_info.get("n_features_selected"),
                    "processing_steps": processing_info.get("processing_steps", []),
                }
            else:
                summary["feature_processing"] = {"processed": False}

            # Splits information
            if "ml_splits" in adata.uns:
                splits_info = adata.uns["ml_splits"]
                splits = splits_info.get("splits", {})
                summary["splits"] = {
                    "created": True,
                    "stratified": splits_info.get("stratified", False),
                    "target_column": splits_info.get("target_column"),
                    "train_size": splits.get("train", {}).get("size", 0),
                    "validation_size": (
                        splits.get("validation", {}).get("size", 0)
                        if splits.get("validation")
                        else 0
                    ),
                    "test_size": splits.get("test", {}).get("size", 0),
                }
            else:
                summary["splits"] = {"created": False}

            # Available metadata for supervised learning
            categorical_columns = []
            numerical_columns = []

            for col in adata.obs.columns:
                if adata.obs[col].dtype.kind in ["O", "S"]:  # Object or string
                    categorical_columns.append(col)
                elif adata.obs[col].dtype.kind in ["i", "u", "f"]:  # Integer or float
                    numerical_columns.append(col)

            summary["metadata"] = {
                "categorical_columns": categorical_columns,
                "numerical_columns": numerical_columns,
                "total_metadata_columns": len(adata.obs.columns),
            }

            ml_summaries[mod_name] = summary

        # Overall summary if multiple modalities
        if len(ml_summaries) == 1:
            return ml_summaries[list(ml_summaries.keys())[0]]
        else:
            return {
                "status": "success",
                "n_modalities": len(ml_summaries),
                "modalities": ml_summaries,
                "overall_ml_readiness": np.mean(
                    [s["ml_readiness"]["score"] for s in ml_summaries.values()]
                ),
            }

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
                summary += (
                    f"- Memory usage: {adata.X.nbytes / 1024**2:.2f} MB\n"
                    if hasattr(adata.X, "nbytes")
                    else ""
                )
                if adata.obs.columns.tolist():
                    summary += f"- Observation metadata: {', '.join(list(adata.obs.columns)[:5])}\n"
                if adata.var.columns.tolist():
                    summary += f"- Variable metadata: {', '.join(list(adata.var.columns)[:5])}\n"
                if adata.layers:
                    summary += (
                        f"- Data layers: {', '.join(list(adata.layers.keys()))}\n"
                    )
                summary += "\n"

        # Add processing log
        if self.processing_log:
            summary += "## Processing Log\n\n"
            for entry in self.processing_log:
                summary += f"- {entry}\n"
            summary += "\n"

        # Add provenance information with detailed activities
        if self.provenance and self.provenance.activities:
            summary += "## Provenance-Tracked Activities\n\n"
            summary += f"**Summary**: {len(self.provenance.activities)} activities, "
            summary += f"{len(self.provenance.entities)} entities, "
            summary += f"{len(self.provenance.agents)} agents\n\n"

            for i, activity in enumerate(self.provenance.activities, 1):
                activity_type = activity.get("type", "unknown")
                timestamp = activity.get("timestamp", "N/A")
                summary += f"### {i}. {activity_type} ({timestamp})\n\n"

                if activity.get("description"):
                    summary += f"{activity['description']}\n\n"

                if activity.get("parameters"):
                    summary += "**Parameters:**\n\n"
                    for param_name, param_value in activity["parameters"].items():
                        # Format parameter value based on its type
                        if (
                            isinstance(param_value, (list, tuple))
                            and len(param_value) > 5
                        ):
                            param_str = f"[{', '.join(str(x) for x in param_value[:5])}...] (length: {len(param_value)})"
                        else:
                            param_str = str(param_value)
                        summary += f"- {param_name}: {param_str}\n"
                    summary += "\n"

        return summary

    def create_data_package(
        self,
        output_dir: str = None,
        progress_callback=None,
        include_png: bool = True,
        compression: str = "gzip",
        include_provenance: bool = True,
    ) -> str:
        """
        Create a comprehensive data package with modalities, plots, and analysis summary.

        Args:
            output_dir: Directory to save the package (defaults to exports_dir)
            progress_callback: Optional callback function to report progress (receives message string)
            include_png: Whether to generate PNG versions of plots (default: True)
            compression: H5AD compression method ('gzip', 'lzf', None) (default: 'gzip')
            include_provenance: Whether to include provenance information (default: True)

        Returns:
            str: Path to the created zip file
        """
        if not self.has_data() and not self.latest_plots:
            raise ValueError("No data or plots to export")

        if progress_callback:
            progress_callback("Preparing export package...")

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

            if progress_callback:
                progress_callback("Creating technical summary...")

            # Save technical summary
            with open(temp_path / "technical_summary.md", "w") as f:
                f.write(self.get_technical_summary())

            # Save all modalities
            if self.modalities:
                if progress_callback:
                    progress_callback(
                        f"Exporting {len(self.modalities)} data modalities..."
                    )

                modalities_dir = temp_path / "modalities"
                modalities_dir.mkdir()

                for i, (name, adata) in enumerate(self.modalities.items(), 1):
                    if progress_callback:
                        progress_callback(
                            f"Saving modality {i}/{len(self.modalities)}: {name}"
                        )
                    try:
                        # Save as H5AD with configurable compression (professional standard for bioinformatics)
                        h5ad_path = modalities_dir / f"{name}.h5ad"
                        adata.write_h5ad(h5ad_path, compression=compression)

                        # Save metadata (optimized to avoid unnecessary list conversions)
                        metadata_path = modalities_dir / f"{name}_metadata.json"
                        modality_metadata = {
                            "shape": adata.shape,
                            "n_obs": adata.n_obs,
                            "n_vars": adata.n_vars,
                            "obs_columns": adata.obs.columns.tolist(),
                            "var_columns": adata.var.columns.tolist(),
                            "layers": list(adata.layers.keys()) if adata.layers else [],
                            "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
                            "uns_keys": list(adata.uns.keys()) if adata.uns else [],
                            "is_sparse": hasattr(adata.X, "toarray"),
                            "data_type": (
                                str(adata.X.dtype)
                                if hasattr(adata.X, "dtype")
                                else "unknown"
                            ),
                        }
                        with open(metadata_path, "w") as f:
                            json.dump(modality_metadata, f, indent=2, default=str)

                    except Exception as e:
                        logger.error(f"Failed to save modality {name}: {e}")
                        if progress_callback:
                            progress_callback(f"Error saving {name}: {str(e)[:50]}...")

            # Save plots
            if self.latest_plots:
                if progress_callback:
                    progress_callback(
                        f"Exporting {len(self.latest_plots)} visualizations..."
                    )

                plots_dir = temp_path / "plots"
                plots_dir.mkdir()

                # Create an index of all plots
                plots_index = []

                for i, plot_entry in enumerate(self.latest_plots):
                    if progress_callback:
                        plot_title = plot_entry.get("title", f"Plot {i+1}")
                        progress_callback(
                            f"Saving plot {i+1}/{len(self.latest_plots)}: {plot_title}"
                        )
                    try:
                        plot = plot_entry["figure"]
                        plot_id = plot_entry["id"]
                        plot_title = plot_entry.get(
                            "original_title", plot_entry["title"]
                        )

                        # Create sanitized filename
                        safe_title = "".join(
                            c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                        ).rstrip()
                        safe_title = safe_title.replace(" ", "_")
                        filename_base = (
                            f"{plot_id}_{safe_title}" if safe_title else plot_id
                        )

                        # Save as HTML (fast and reliable - primary format)
                        _, pio_module = _ensure_plotly()
                        pio_module.write_html(plot, plots_dir / f"{filename_base}.html")

                        # Save as PNG (configurable, non-blocking)
                        if include_png:
                            try:
                                # Use kaleido engine for better reliability
                                with SuppressKaleidoLogging():
                                    _, pio_module = _ensure_plotly()
                                    pio_module.write_image(
                                        plot,
                                        plots_dir / f"{filename_base}.png",
                                        engine="kaleido",
                                        width=1200,
                                        height=800,
                                    )
                            except Exception as e:
                                # PNG generation failed - continue without it (don't block export)
                                logger.debug(
                                    f"Skipped PNG for {plot_id}: {type(e).__name__}: {e}"
                                )
                                # Note: HTML version is still available and fully interactive

                        # Save plot metadata
                        with open(plots_dir / f"{filename_base}_info.txt", "w") as f:
                            f.write(f"ID: {plot_id}\n")
                            f.write(f"Title: {plot_title}\n")
                            f.write(f"Created: {plot_entry.get('timestamp', 'N/A')}\n")
                            f.write(f"Source: {plot_entry.get('source', 'N/A')}\n")

                        # Add to index
                        plots_index.append(
                            {
                                "id": plot_id,
                                "title": plot_title,
                                "filename": filename_base,
                                "timestamp": plot_entry.get("timestamp", "N/A"),
                                "source": plot_entry.get("source", "N/A"),
                            }
                        )
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
                        f.write(
                            f"- Files: [{plot_info['filename']}.html]({plot_info['filename']}.html), [{plot_info['filename']}.png]({plot_info['filename']}.png)\n\n"
                        )

            if progress_callback:
                progress_callback("Saving workspace metadata...")

            # Save workspace status
            with open(temp_path / "workspace_status.json", "w") as f:
                json.dump(self.get_workspace_status(), f, indent=2, default=str)

            # Save provenance if available and requested
            if include_provenance and self.provenance:
                with open(temp_path / "provenance.json", "w") as f:
                    json.dump(self.provenance.to_dict(), f, indent=2, default=str)

            # Create professional README for the export package
            readme_content = f"""# Lobster Analysis Export Package

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents

### Data Files (H5AD Format)
- **Format**: H5AD (HDF5-based AnnData format)
- **Compression**: {compression or 'none'}
- **Recommendation**: Open with scanpy, cellxgene, or other single-cell analysis tools

### Visualizations
- **HTML Files**: Interactive Plotly visualizations (recommended)
- **PNG Files**: Static images for publications {'(included)' if include_png else '(not generated)'}

### Metadata
- **technical_summary.md**: Analysis overview and methods
- **workspace_status.json**: System configuration and data provenance
{'- **provenance.json**: Detailed analysis history (W3C-PROV compliant)' if include_provenance else ''}

## Usage

### Loading Data in Python
```python
import scanpy as sc
adata = sc.read_h5ad('modalities/your_dataset.h5ad')
```

### Loading Data in R
```r
library(anndata)
adata <- read_h5ad('modalities/your_dataset.h5ad')
```

### Viewing Plots
Open HTML files in any web browser for interactive exploration.

## Citation
Generated with Lobster AI - Multi-Agent Bioinformatics Analysis Platform
https://github.com/OmicsOS/lobster
"""

            with open(temp_path / "README.md", "w") as f:
                f.write(readme_content)

            if progress_callback:
                progress_callback("Creating final package...")

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

        if (
            "transcriptomics" in name_lower
            or "rna" in name_lower
            or "geo" in name_lower
        ):
            if "single_cell" in name_lower or "sc" in name_lower:
                return "single_cell_rna_seq"
            else:
                return "bulk_rna_seq"
        elif "proteomics" in name_lower or "protein" in name_lower:
            if "ms" in name_lower or "mass" in name_lower:
                return "mass_spectrometry_proteomics"
            else:
                return "affinity_proteomics"

        return "unknown"

    def _scan_workspace(self) -> Dict[str, Dict]:
        """
        Scan workspace for available datasets without loading them.

        BUG FIX #2: Modified to return dict instead of modifying self.available_datasets directly,
        enabling proper caching in get_available_datasets() method.

        Returns:
            Dict mapping dataset names to their metadata
        """
        with self._measure_step("dm:scan_workspace"):
            datasets = {}
            data_dir = self.workspace_path / "data"

            if not data_dir.exists():
                return datasets

            for h5ad_file in data_dir.glob("*.h5ad"):
                try:
                    # Use h5py for efficient metadata extraction
                    import h5py

                    with h5py.File(h5ad_file, "r") as f:
                        # Extract basic metadata - handle both dense and sparse matrices per AnnData spec
                        if "X" in f:
                            if isinstance(f["X"], h5py.Dataset):
                                # Dense matrix - has shape attribute
                                shape = f["X"].shape
                            elif isinstance(f["X"], h5py.Group):
                                # Sparse matrix (CSR/CSC) - shape is in attributes per AnnData spec
                                shape = (
                                    tuple(f["X"].attrs["shape"])
                                    if "shape" in f["X"].attrs
                                    else (0, 0)
                                )
                            else:
                                shape = (0, 0)
                        else:
                            shape = (0, 0)

                        stat = h5ad_file.stat()
                        datasets[h5ad_file.stem] = {
                            "path": str(h5ad_file),
                            "size_mb": stat.st_size / 1e6,
                            "shape": shape,
                            "modified": datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            "type": "h5ad",
                        }
                except Exception as e:
                    logger.warning(f"Could not scan {h5ad_file}: {e}")

            # Update self.available_datasets for backward compatibility
            self.available_datasets = datasets
            return datasets

    def get_available_datasets(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Get available datasets with intelligent TTL-based caching.

        BUG FIX #2: Implement workspace scan caching to prevent repeated expensive I/O.
        Without caching, each /workspace operation triggers a full scan (~850ms on 50 datasets).
        With caching, subsequent accesses within TTL window are <1ms (99.9% improvement).

        Args:
            force_refresh: If True, bypass cache and rescan filesystem

        Returns:
            Dict mapping dataset names to their metadata

        Performance:
            - Cache miss (first call): ~850ms (filesystem scan)
            - Cache hit (within 30s): <1ms (in-memory access)
            - Expected improvement: 75-80% for typical workflows
        """
        with self._measure_step("dm:get_available_datasets"):
            current_time = time.time()
            cache_age = current_time - self._scan_timestamp

            # Check if cache is valid (not expired and not force refresh)
            cache_valid = (
                not force_refresh
                and self._available_datasets_cache is not None
                and cache_age < self._scan_ttl
            )

            if cache_valid:
                logger.debug(
                    f"Workspace cache hit (age: {cache_age:.1f}s, TTL: {self._scan_ttl}s)"
                )
                return self._available_datasets_cache

            # Cache miss or expired - perform scan
            logger.debug(
                f"Workspace cache {'forced refresh' if force_refresh else 'miss'}  "
                f"(age: {cache_age:.1f}s, TTL: {self._scan_ttl}s)"
            )
            self._available_datasets_cache = self._scan_workspace()
            self._scan_timestamp = current_time

            return self._available_datasets_cache

    def invalidate_scan_cache(self) -> None:
        """
        Force refresh on next workspace scan access.

        BUG FIX #2: Explicitly invalidate cache after operations that modify workspace
        (e.g., loading new data, deleting datasets). This ensures the cache stays consistent
        with actual filesystem state.

        Usage:
            - After saving new datasets
            - After deleting datasets
            - When user explicitly requests refresh (e.g., /workspace list --refresh)
        """
        self._available_datasets_cache = None
        self._scan_timestamp = 0
        logger.debug(
            "Workspace scan cache invalidated - next access will trigger fresh scan"
        )

    def _load_session_metadata(self) -> None:
        """Load session metadata from file (multi-process safe)."""
        try:
            with queue_file_lock(self._session_lock, self._session_lock_path):
                with open(self.session_file, "r") as f:
                    self.session_data = json.load(f)
            logger.debug(f"Loaded session metadata from {self.session_file}")
        except Exception as e:
            logger.warning(f"Could not load session metadata: {e}")
            self.session_data = None

    def _update_session_file(self, action: str = "update") -> None:
        """Update session file with current state (multi-process safe)."""
        try:
            # Get Lobster version if available
            try:
                from importlib.metadata import version

                lobster_version = version("lobster")
            except Exception:
                lobster_version = "unknown"

            session_data = {
                "session_id": self.session_id,
                "created_at": (
                    self.session_data.get("created_at", datetime.now().isoformat())
                    if self.session_data
                    else datetime.now().isoformat()
                ),
                "last_modified": datetime.now().isoformat(),
                "lobster_version": lobster_version,
                "active_modalities": {},
                "workspace_stats": {
                    "total_datasets": len(self.available_datasets),
                    "total_loaded": len(self.modalities),
                    "total_size_mb": sum(
                        d["size_mb"] for d in self.available_datasets.values()
                    ),
                },
            }

            # Add loaded modalities info
            for name, adata in self.modalities.items():
                if name in self.available_datasets:
                    session_data["active_modalities"][name] = {
                        **self.available_datasets[name],
                        "last_accessed": datetime.now().isoformat(),
                    }
                else:
                    # For modalities not in available_datasets (newly created)
                    session_data["active_modalities"][name] = {
                        "shape": adata.shape,
                        "last_accessed": datetime.now().isoformat(),
                        "type": "in_memory",
                    }

            # Add command history from provenance if available
            if self.provenance and self.provenance.activities:
                # Keep last 50 activities
                recent_activities = self.provenance.activities[-50:]
                session_data["command_history"] = [
                    {"timestamp": act.get("timestamp"), "command": act.get("type")}
                    for act in recent_activities
                ]

            # Write atomically with inter-process lock
            with queue_file_lock(self._session_lock, self._session_lock_path):
                atomic_write_json(self.session_file, session_data)

            logger.debug(f"Updated session file: {self.session_file}")
        except Exception as e:
            logger.warning(f"Could not update session file: {e}")

    def load_dataset(self, name: str, force_reload: bool = False) -> bool:
        """Load a specific dataset from disk.

        Args:
            name: Name of the dataset to load
            force_reload: Whether to reload even if already in memory

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        with self._measure_step("dm:load_dataset"):
            if name in self.modalities and not force_reload:
                return True  # Already loaded

            if name not in self.available_datasets:
                logger.error(f"Dataset '{name}' not found in workspace")
                return False

            try:
                path = Path(self.available_datasets[name]["path"])
                self.modalities[name] = _ensure_anndata().read_h5ad(path)

                # Update session file
                self._update_session_file("loaded")

                # Log operation
                self.log_tool_usage(
                    tool_name="load_dataset",
                    parameters={"name": name},
                    description=(
                        f"Loaded dataset {name} ("
                        f"{self.available_datasets[name]['size_mb']:.1f} MB)"
                    ),
                )

                logger.info(f"Loaded dataset '{name}' from workspace")
                return True
            except Exception as e:
                logger.error(f"Failed to load dataset {name}: {e}")
                return False

    def restore_session(
        self, pattern: str = "recent", max_size_mb: float = 1000
    ) -> Dict[str, Any]:
        """Restore datasets based on pattern with memory limits.

        Args:
            pattern: Restoration pattern ('recent', 'all', or glob pattern)
            max_size_mb: Maximum total size of datasets to load

        Returns:
            Dict[str, Any]: Restoration results
        """
        restored = []
        skipped = []
        total_size = 0

        if pattern == "recent":
            # Load datasets from last session
            if self.session_data and "active_modalities" in self.session_data:
                for name in self.session_data["active_modalities"]:
                    if name in self.available_datasets:
                        size_mb = self.available_datasets[name]["size_mb"]
                        if total_size + size_mb <= max_size_mb:
                            if self.load_dataset(name):
                                restored.append(name)
                                total_size += size_mb
                        else:
                            skipped.append((name, "size_limit"))

        elif pattern == "all":
            # Load all available datasets
            for name in sorted(self.available_datasets.keys()):
                size_mb = self.available_datasets[name]["size_mb"]
                if total_size + size_mb <= max_size_mb:
                    if self.load_dataset(name):
                        restored.append(name)
                        total_size += size_mb
                else:
                    skipped.append((name, "size_limit"))

        else:
            # Pattern matching (glob-style)
            import fnmatch

            for name in self.available_datasets:
                if fnmatch.fnmatch(name, pattern):
                    size_mb = self.available_datasets[name]["size_mb"]
                    if total_size + size_mb <= max_size_mb:
                        if self.load_dataset(name):
                            restored.append(name)
                            total_size += size_mb
                    else:
                        skipped.append((name, "size_limit"))

        return {
            "restored": restored,
            "skipped": skipped,
            "total_size_mb": total_size,
            "pattern": pattern,
        }

    # ========================================
    # NOTEBOOK PIPELINE SUPPORT
    # ========================================

    def export_notebook(
        self,
        name: str,
        description: str = "",
        filter_strategy: str = "successful",
    ) -> Path:
        """
        Export current session as Jupyter notebook.

        Args:
            name: Notebook filename (no extension)
            description: Human-readable description
            filter_strategy: "successful" | "all" | "manual"

        Returns:
            Path to generated .ipynb file

        Raises:
            ValueError: If provenance tracking disabled or no activities

        Example:
            >>> path = dm.export_notebook(
            ...     name="standard_qc_workflow",
            ...     description="Quality control and clustering for 10X data"
            ... )
            >>> print(f"Exported: {path}")
            Exported: /Users/kevin/.lobster/notebooks/standard_qc_workflow.ipynb
        """
        if not self.provenance:
            raise ValueError("Provenance tracking disabled - cannot export notebook")

        if not self.notebook_exporter:
            raise ValueError("Notebook exporter not initialized")

        if not self.provenance.activities:
            raise ValueError("No activities recorded - nothing to export")

        path = self.notebook_exporter.export(name, description, filter_strategy)

        logger.info(f"Notebook exported: {path}")
        logger.info("Next steps:")
        logger.info(f"  1. Review: jupyter notebook {path}")
        logger.info(f"  2. Commit: git add {path} && git commit -m 'Add {name}'")
        logger.info(f"  3. Run: /pipeline run {path.name} <modality>")

        return path

    def run_notebook(
        self,
        notebook_path: Union[str, Path],
        input_modality: str,
        parameters: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute saved notebook with new data.

        Args:
            notebook_path: Path to .ipynb file or filename in .lobster/notebooks/
            input_modality: Name of modality to use as input
            parameters: Optional parameter overrides
            dry_run: If True, validate but don't execute

        Returns:
            Execution result with status, output_notebook, etc.

        Example:
            >>> # Dry run first
            >>> result = dm.run_notebook(
            ...     "standard_qc_workflow.ipynb",
            ...     "my_new_dataset",
            ...     dry_run=True
            ... )
            >>> print(result['validation'])

            >>> # Actually run
            >>> result = dm.run_notebook(
            ...     "standard_qc_workflow.ipynb",
            ...     "my_new_dataset"
            ... )
            >>> print(f"Output: {result['output_notebook']}")
        """
        # Resolve notebook path
        nb_path = Path(notebook_path)
        if not nb_path.exists():
            # Try .lobster/notebooks/
            nb_path = Path.home() / ".lobster" / "notebooks" / notebook_path

        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook not found: {notebook_path}")

        # Get input data
        if input_modality not in self.modalities:
            raise ValueError(f"Modality '{input_modality}' not loaded")

        # Save input modality as H5AD for notebook
        import tempfile

        temp_dir = Path(tempfile.mkdtemp())
        temp_input = temp_dir / "input.h5ad"
        self.save_modality(input_modality, temp_input)

        # Execute
        if dry_run:
            result = self.notebook_executor.dry_run(nb_path, temp_input)
        else:
            result = self.notebook_executor.execute(nb_path, temp_input, parameters)

        return result

    def list_notebooks(self) -> List[Dict[str, Any]]:
        """
        List available notebooks in .lobster/notebooks/

        Returns:
            List of notebook metadata
        """
        notebooks_dir = Path.home() / ".lobster" / "notebooks"
        if not notebooks_dir.exists():
            return []

        notebooks = []
        for nb_file in notebooks_dir.glob("*.ipynb"):
            # Read metadata
            try:
                with open(nb_file) as f:
                    nbformat_module = _ensure_nbformat()
                    nb = nbformat_module.read(f, as_version=4)

                metadata = nb.metadata.get("lobster", {})

                notebooks.append(
                    {
                        "filename": nb_file.name,
                        "path": str(nb_file),
                        "name": nb_file.stem,
                        "created_by": metadata.get("created_by", "unknown"),
                        "created_at": metadata.get("created_at", ""),
                        "lobster_version": metadata.get("lobster_version", ""),
                        "n_steps": len([c for c in nb.cells if c.cell_type == "code"]),
                        "size_kb": nb_file.stat().st_size / 1024,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not read notebook {nb_file}: {e}")

        return notebooks

    # ==================== PUBLICATION CACHING (Phase 2) ====================

    def cache_publication_content(
        self,
        identifier: str,
        content: Dict[str, Any],
        format: str = "markdown",
    ) -> Path:
        """
        Cache publication content for future retrieval.

        This method delegates to DoclingService for actual caching while providing
        a unified interface through DataManager. All publication caching should go
        through this method to ensure proper provenance tracking.

        Args:
            identifier: Publication identifier (PMID, DOI, or URL)
            content: Extraction result dictionary containing:
                - 'markdown': str - Extracted content as markdown
                - 'source': str - Source URL
                - 'parser': str - Parser used (docling/pypdf2)
                - 'methods_text': Optional[str] - Methods section text
                - 'software_detected': Optional[List[str]] - Detected software
            format: Cache format ('markdown' for human-readable, 'json' for structured)

        Returns:
            Path to cached file

        Example:
            >>> content = {
            ...     'markdown': '# Methods\\nWe used Seurat...',
            ...     'source': 'https://pmc.../PMC123.pdf',
            ...     'parser': 'docling',
            ...     'methods_text': 'Analysis with Seurat',
            ...     'software_detected': ['seurat']
            ... }
            >>> path = dm.cache_publication_content('PMID:12345678', content)
            >>> print(f"Cached to: {path}")

        Note:
            This is part of Phase 2 refactoring to consolidate all caching
            through DataManager (architectural requirement).
        """
        import hashlib

        publications_dir = self.literature_cache_dir / "publications"
        publications_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize identifier for filename
        safe_identifier = (
            identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        )

        if format == "markdown":
            # Store as human-readable markdown
            cache_file = publications_dir / f"{safe_identifier}.md"

            # Format markdown document
            markdown_content = f"""# Publication: {identifier}

**Extraction Date**: {content.get('timestamp', 'N/A')}
**Parser**: {content.get('parser', 'unknown')}

---

## Methods Section

{content.get('methods_text', 'No methods section extracted')}

---

## Software Tools Detected

{chr(10).join(f'`{tool}`' for tool in content.get('software_detected', []))}

---

## Extraction Metadata

- **Source**: {content.get('source', 'N/A')}
- **Parser**: {content.get('parser', 'N/A')}
- **Fallback Used**: {content.get('fallback_used', False)}
- **Timestamp**: {content.get('timestamp', 'N/A')}
"""

            cache_file.write_text(markdown_content, encoding="utf-8")
            logger.info(f"Cached publication as markdown: {cache_file.name}")

        elif format == "json":
            # Use MD5-based key for JSON (DoclingService format)
            source = content.get("source", identifier)
            cache_key = hashlib.md5(source.encode()).hexdigest()
            cache_file = publications_dir / f"{cache_key}.json"

            import json

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

            logger.info(f"Cached publication as JSON: {cache_file.name}")

        else:
            raise ValueError(f"Unsupported cache format: {format}")

        # Log to provenance
        if self.provenance:
            self.log_tool_usage(
                tool_name="cache_publication_content",
                parameters={"identifier": identifier, "format": format},
                description=f"Cached publication content for {identifier}",
            )

        return cache_file

    def get_cached_publication(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached publication by identifier.

        This method checks for cached publication content in the literature cache
        directory. It supports both markdown and JSON formats.

        Args:
            identifier: Publication identifier (PMID, DOI, or URL)

        Returns:
            Dictionary with extraction results or None if not found

        Example:
            >>> content = dm.get_cached_publication('PMID:12345678')
            >>> if content:
            ...     print(f"Found cached content: {content['methods_text'][:100]}...")
            ... else:
            ...     print("No cache found, need to fetch")

        Note:
            This is part of Phase 2 refactoring. Delegates to existing
            cache infrastructure while providing unified DataManager interface.
        """
        import hashlib
        import json

        publications_dir = self.literature_cache_dir / "publications"

        # Sanitize identifier for filename
        safe_identifier = (
            identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        )

        # Try markdown format first
        markdown_file = publications_dir / f"{safe_identifier}.md"
        if markdown_file.exists():
            logger.info(f"Cache hit (markdown): {markdown_file.name}")
            markdown_content = markdown_file.read_text(encoding="utf-8")

            # Parse markdown to extract structured data
            result = {
                "identifier": identifier,
                "format": "markdown",
                "markdown": markdown_content,
                "cache_file": str(markdown_file),
                "cache_hit": True,
            }

            # Extract methods section if present
            if "## Methods Section" in markdown_content:
                methods_start = markdown_content.find("## Methods Section") + len(
                    "## Methods Section"
                )
                methods_end = markdown_content.find("---", methods_start)
                if methods_end > methods_start:
                    result["methods_text"] = markdown_content[
                        methods_start:methods_end
                    ].strip()

            # Extract software tools if present
            if "## Software Tools Detected" in markdown_content:
                software_start = markdown_content.find(
                    "## Software Tools Detected"
                ) + len("## Software Tools Detected")
                software_end = markdown_content.find("---", software_start)
                if software_end > software_start:
                    software_section = markdown_content[
                        software_start:software_end
                    ].strip()
                    # Extract tools from markdown code blocks
                    import re

                    tools = re.findall(r"`([^`]+)`", software_section)
                    result["software_detected"] = tools

            return result

        # Try JSON format (MD5-based key)
        cache_key = hashlib.md5(identifier.encode()).hexdigest()
        json_file = publications_dir / f"{cache_key}.json"
        if json_file.exists():
            logger.info(f"Cache hit (JSON): {json_file.name}")
            with open(json_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            result["cache_hit"] = True
            result["cache_file"] = str(json_file)
            return result

        # No cache found
        logger.debug(f"Cache miss for identifier: {identifier}")
        return None

    def list_session_publications(self) -> List[Dict[str, Any]]:
        """
        List publications extracted in the current session.

        This method uses provenance tracking to identify publications that were
        extracted during the current session, then checks cache status for each.

        Returns:
            List of publication summaries:
            [{
                "identifier": str,
                "tool_name": str,
                "timestamp": str,
                "cache_status": "markdown" | "json" | "both" | "none",
                "methods_length": int,
                "source": str
            }]

        Examples:
            >>> publications = dm.list_session_publications()
            >>> for pub in publications:
            ...     print(f"{pub['identifier']}: {pub['cache_status']}")

        Note:
            This is part of Phase 3 migration. Session features will be
            fully refactored in Phase 4.
        """
        publications = []

        # Find all publication extraction operations in tool usage history
        extraction_tools = [
            "extract_pdf_content",
            "extract_methods_from_paper",
            "extract_methods_section",
            "extract_paper_methods",  # Phase 3 tool
        ]

        for entry in self.tool_usage_history:
            tool_name = entry.get("tool_name", "")
            if tool_name in extraction_tools:
                params = entry.get("parameters", {})
                description = entry.get("description", "")

                # Try to extract identifier from parameters
                identifier = (
                    params.get("url_or_pmid")
                    or params.get("source")
                    or params.get("url")
                    or "unknown"
                )

                # Truncate long URLs for display
                if len(identifier) > 80:
                    identifier = identifier[:77] + "..."

                # Check cache status
                cache_status = self._check_publication_cache_status(identifier)

                # Estimate methods length from description
                methods_length = 0
                if "characters" in description:
                    try:
                        methods_length = int(description.split()[0].replace(",", ""))
                    except (ValueError, IndexError):
                        pass

                publications.append(
                    {
                        "identifier": identifier,
                        "tool_name": tool_name,
                        "timestamp": entry.get("timestamp", "unknown"),
                        "cache_status": cache_status,
                        "methods_length": methods_length,
                        "source": params.get("parser", "unknown"),
                    }
                )

        logger.info(f"Found {len(publications)} publications in current session")
        return publications

    def _check_publication_cache_status(self, identifier: str) -> str:
        """
        Check cache status for a publication identifier.

        Args:
            identifier: Publication identifier

        Returns:
            Cache status: "markdown", "json", "both", or "none"
        """
        import hashlib

        has_markdown = False
        has_json = False

        # Check markdown cache
        safe_identifier = (
            identifier.replace(":", "_").replace("/", "_").replace("\\", "_")
        )
        publications_dir = self.literature_cache_dir / "publications"
        md_file = publications_dir / f"{safe_identifier}.md"

        if md_file.exists():
            has_markdown = True

        # Check JSON cache (MD5-based key)
        cache_key = hashlib.md5(identifier.encode()).hexdigest()
        json_file = publications_dir / f"{cache_key}.json"

        if json_file.exists():
            has_json = True

        # Return combined status
        if has_markdown and has_json:
            return "both"
        elif has_markdown:
            return "markdown"
        elif has_json:
            return "json"
        else:
            return "none"
