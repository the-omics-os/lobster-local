"""
PlotManager: Extracted component for managing Plotly visualizations.

This module provides the PlotManager class for managing Plotly figure storage,
metadata tracking, and workspace export. Extracted from DataManagerV2 to
improve separation of concerns and testability.

All methods return 3-tuples: (result, stats, AnalysisStep|None) to follow
Lobster's standard service pattern and enable provenance tracking.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from lobster.core.analysis_ir import AnalysisStep

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

logger = logging.getLogger(__name__)

# Lazy-loaded plotly modules
_plotly_go = None
_plotly_io = None


def _ensure_plotly():
    """Lazily import plotly with a helpful error message."""
    global _plotly_go, _plotly_io
    if _plotly_go is None:
        try:
            import plotly.graph_objects as _plotly_go  # type: ignore
            import plotly.io as _plotly_io  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "plotly is required for PlotManager operations. "
                "Install it with `pip install plotly`."
            ) from exc
    return _plotly_go, _plotly_io


class SuppressKaleidoLogging:
    """Context manager to suppress Kaleido's verbose logging during PNG export."""

    def __init__(self):
        self.original_level = logging.WARNING

    def __enter__(self):
        # Suppress kaleido and plotly logging
        logging.getLogger("kaleido").setLevel(logging.ERROR)
        logging.getLogger("plotly").setLevel(logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original levels
        logging.getLogger("kaleido").setLevel(self.original_level)
        logging.getLogger("plotly").setLevel(self.original_level)
        return False


class PlotManager:
    """
    Manages Plotly figure storage, metadata tracking, and workspace export.

    This class is the single source of truth for plot management within
    a Lobster workspace. It handles:
    - Plot storage with comprehensive metadata
    - FIFO buffer with configurable max history
    - Thread-safe save operations with rate limiting
    - Visualization state tracking for agents
    - Workspace export (HTML/PNG)

    All public methods return 3-tuples: (result, stats, AnalysisStep|None)
    to follow Lobster's standard service pattern.

    Thread Safety:
        - `_save_lock`: Prevents concurrent saves
        - `_counter_lock`: Protects plot_counter increment
        - Rate limiting: Minimum 2s between saves (configurable)

    Attributes:
        workspace_path: Path to the workspace directory
        max_plots_history: Maximum number of plots to retain
        enable_ir: Whether to generate AnalysisStep IR for provenance
    """

    def __init__(
        self,
        workspace_path: Path,
        max_plots_history: int = 50,
        min_save_interval: float = 2.0,
        enable_ir: bool = True,
    ):
        """
        Initialize PlotManager.

        Args:
            workspace_path: Path to the workspace directory for plot export
            max_plots_history: Maximum number of plots to keep in history
            min_save_interval: Minimum seconds between save operations
            enable_ir: Whether to generate AnalysisStep IR for provenance
        """
        self.workspace_path = Path(workspace_path)
        self.max_plots_history = max_plots_history
        self.enable_ir = enable_ir

        # Plot storage
        self.latest_plots: List[Dict[str, Any]] = []
        self.plot_counter: int = 0
        self._counter_lock = threading.Lock()

        # Save safety mechanisms
        self._save_lock = threading.Lock()
        self._save_in_progress = False
        self._last_save_time = 0.0
        self._min_save_interval = min_save_interval

        # Visualization state management (for Visualization Expert Agent)
        self.visualization_state: Dict[str, Any] = {
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

        logger.debug(f"PlotManager initialized with workspace: {workspace_path}")

    # ========================================
    # CORE PLOT MANAGEMENT
    # ========================================

    def add_plot(
        self,
        plot: "Figure",
        title: Optional[str] = None,
        source: Optional[str] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        analysis_params: Optional[Dict[str, Any]] = None,
        modalities: Optional[Dict[str, Any]] = None,
        current_dataset: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], Optional[AnalysisStep]]:
        """
        Add a plot to the collection with comprehensive metadata.

        Args:
            plot: Plotly Figure object
            title: Optional title for the plot
            source: Optional source identifier (e.g., service name)
            dataset_info: Optional information about the dataset used
            analysis_params: Optional parameters used for the analysis
            modalities: Optional dict of modalities for context (from DataManagerV2)
            current_dataset: Optional current dataset name for context

        Returns:
            Tuple of (plot_id, stats, AnalysisStep|None)

        Raises:
            ValueError: If plot is not a Plotly Figure
        """
        try:
            go_module, _ = _ensure_plotly()
            if not isinstance(plot, go_module.Figure):
                raise ValueError("Plot must be a plotly Figure object.")

            # Generate unique identifier with thread safety
            with self._counter_lock:
                self.plot_counter += 1
                plot_id = f"plot_{self.plot_counter}"

            # Create timestamps
            timestamp = datetime.now().isoformat()
            human_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Build dataset context
            current_dataset_info = dataset_info or {}
            if modalities and not current_dataset_info:
                # Try current_dataset first, then first available
                target_name = current_dataset or (
                    list(modalities.keys())[0] if modalities else None
                )
                if target_name and target_name in modalities:
                    adata = modalities[target_name]
                    if hasattr(adata, "shape"):
                        current_dataset_info = {
                            "data_shape": adata.shape,
                            "modality_name": target_name,
                            "n_obs": adata.n_obs if hasattr(adata, "n_obs") else adata.shape[0],
                            "n_vars": adata.n_vars if hasattr(adata, "n_vars") else adata.shape[1],
                        }

            # Create enhanced title with context
            enhanced_title = title or "Untitled"
            if current_dataset_info and "modality_name" in current_dataset_info:
                modality_name = current_dataset_info["modality_name"]
                if modality_name not in enhanced_title:
                    enhanced_title = f"{enhanced_title} ({modality_name} - {human_timestamp})"
                else:
                    enhanced_title = f"{enhanced_title} ({human_timestamp})"
            elif current_dataset_info and "data_shape" in current_dataset_info:
                shape_info = f"{current_dataset_info['data_shape'][0]}x{current_dataset_info['data_shape'][1]}"
                enhanced_title = f"{enhanced_title} (Data: {shape_info} - {human_timestamp})"
            else:
                enhanced_title = f"{enhanced_title} ({human_timestamp})"

            # Update plot title
            plot.update_layout(title=enhanced_title)

            # Create comprehensive plot entry
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
                    "has_modalities": bool(modalities),
                    "modality_names": list(modalities.keys()) if modalities else [],
                    "current_dataset": current_dataset,
                },
            }

            # Add to queue
            self.latest_plots.append(plot_entry)

            # Maintain FIFO buffer
            if len(self.latest_plots) > self.max_plots_history:
                self.latest_plots.pop(0)

            # Build stats
            stats = {
                "plot_id": plot_id,
                "title": enhanced_title,
                "source": source or "unknown",
                "plots_in_history": len(self.latest_plots),
                "timestamp": timestamp,
            }

            # Create IR if enabled
            ir = self._create_add_plot_ir(
                title=title,
                source=source,
                dataset_info=current_dataset_info,
            ) if self.enable_ir else None

            logger.info(f"Plot added: '{enhanced_title}' with ID {plot_id} from {source}")

            return plot_id, stats, ir

        except ValueError:
            raise
        except Exception as e:
            logger.exception(f"Error in add_plot: {e}")
            return "", {"error": str(e)}, None

    def clear_plots(self) -> Tuple[None, Dict[str, Any], Optional[AnalysisStep]]:
        """
        Clear all stored plots.

        Returns:
            Tuple of (None, stats, AnalysisStep|None)
        """
        cleared_count = len(self.latest_plots)
        self.latest_plots = []

        stats = {
            "cleared_count": cleared_count,
            "timestamp": datetime.now().isoformat(),
        }

        ir = self._create_clear_plots_ir() if self.enable_ir else None

        logger.info(f"Cleared {cleared_count} plots")

        return None, stats, ir

    def get_plot_by_id(
        self, plot_id: str
    ) -> Tuple[Optional["Figure"], Dict[str, Any], None]:
        """
        Get a plot by its unique ID.

        Args:
            plot_id: The unique ID of the plot

        Returns:
            Tuple of (Figure|None, stats, None)
        """
        for plot_entry in self.latest_plots:
            if plot_entry["id"] == plot_id:
                stats = {
                    "found": True,
                    "plot_id": plot_id,
                    "title": plot_entry.get("title"),
                }
                return plot_entry["figure"], stats, None

        stats = {"found": False, "plot_id": plot_id}
        return None, stats, None

    def get_latest_plots(
        self, n: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get the n most recent plots with their metadata.

        Args:
            n: Number of plots to return (None for all)

        Returns:
            Tuple of (plot_entries, stats, None)
        """
        if n is None:
            plots = self.latest_plots
        else:
            plots = self.latest_plots[-n:]

        stats = {
            "total_plots": len(self.latest_plots),
            "returned_plots": len(plots),
            "requested": n,
        }

        return plots, stats, None

    def get_plot_history(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get the complete plot history with minimal metadata (no figures).

        Returns:
            Tuple of (history_entries, stats, None)
        """
        history = [
            {
                "id": p["id"],
                "title": p["title"],
                "timestamp": p["timestamp"],
                "source": p["source"],
            }
            for p in self.latest_plots
        ]

        stats = {
            "total_plots": len(history),
            "timestamp": datetime.now().isoformat(),
        }

        return history, stats, None

    def save_plots_to_workspace(
        self,
    ) -> Tuple[List[str], Dict[str, Any], Optional[AnalysisStep]]:
        """
        Save all current plots to the workspace directory.

        Thread-safe with rate limiting to prevent excessive saves.

        Returns:
            Tuple of (saved_file_paths, stats, AnalysisStep|None)
        """
        current_time = time.time()

        # Safety check: already in progress
        if self._save_in_progress:
            logger.warning("save_plots_to_workspace already in progress, skipping")
            return [], {"skipped": True, "reason": "already_in_progress"}, None

        # Safety check: rate limiting
        if current_time - self._last_save_time < self._min_save_interval:
            logger.warning(
                f"Rate limited - last save was {current_time - self._last_save_time:.1f}s ago"
            )
            return [], {"skipped": True, "reason": "rate_limited"}, None

        # Try to acquire lock (non-blocking)
        if not self._save_lock.acquire(blocking=False):
            logger.warning("Could not acquire save lock, another save in progress")
            return [], {"skipped": True, "reason": "lock_unavailable"}, None

        try:
            self._save_in_progress = True
            self._last_save_time = current_time

            if not self.latest_plots:
                logger.info("No plots to save")
                return [], {"skipped": True, "reason": "no_plots"}, None

            plots_dir = self.workspace_path / "plots"
            plots_dir.mkdir(exist_ok=True)

            saved_files = []
            for plot_entry in self.latest_plots:
                try:
                    plot = plot_entry["figure"]
                    plot_id = plot_entry["id"]
                    plot_title = plot_entry.get("original_title", plot_entry["title"])

                    # Truncate long titles
                    if len(plot_title) > 80:
                        available_chars = 80 - 3
                        start_length = (available_chars + 1) // 2
                        end_length = available_chars // 2
                        plot_title = f"{plot_title[:start_length]}...{plot_title[-end_length:]}"

                    # Sanitize filename
                    safe_title = "".join(
                        c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                    ).rstrip()
                    safe_title = safe_title.replace(" ", "_")
                    filename_base = f"{plot_id}_{safe_title}" if safe_title else plot_id

                    # Save HTML
                    html_path = plots_dir / f"{filename_base}.html"
                    _, pio_module = _ensure_plotly()
                    pio_module.write_html(plot, html_path)
                    saved_files.append(str(html_path))

                    # Store file path in entry for UI access
                    plot_entry["file_path"] = str(html_path)

                    # BUG010 FIX: Skip PNG for large datasets (Kaleido limitation)
                    dataset_info = plot_entry.get("dataset_info", {})
                    n_cells = dataset_info.get("n_cells", 0)
                    skip_png = n_cells > 50000

                    if not skip_png:
                        png_path = plots_dir / f"{filename_base}.png"
                        try:
                            with SuppressKaleidoLogging():
                                pio_module.write_image(plot, png_path)
                            saved_files.append(str(png_path))
                        except Exception as e:
                            logger.warning(f"Could not save PNG for {plot_id}: {e}")
                    else:
                        logger.info(
                            f"Skipped PNG export for {plot_id} ({n_cells:,} cells > 50K threshold)"
                        )

                    logger.info(f"Saved plot {plot_id} to workspace")

                except Exception as e:
                    logger.error(f"Failed to save plot {plot_entry.get('id', 'unknown')}: {e}")

            stats = {
                "saved_count": len(saved_files),
                "total_plots": len(self.latest_plots),
                "workspace_path": str(plots_dir),
                "timestamp": datetime.now().isoformat(),
            }

            ir = self._create_save_plots_ir(len(saved_files)) if self.enable_ir else None

            return saved_files, stats, ir

        finally:
            self._save_in_progress = False
            self._save_lock.release()
            logger.debug("Released save lock and reset progress flag")

    # ========================================
    # VISUALIZATION STATE MANAGEMENT
    # ========================================

    def add_visualization_record(
        self, plot_id: str, metadata: Dict[str, Any]
    ) -> Tuple[None, Dict[str, Any], None]:
        """
        Track visualization creation in the visualization state.

        Args:
            plot_id: Unique identifier for the plot
            metadata: Metadata about the visualization

        Returns:
            Tuple of (None, stats, None)
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

            stats = {
                "plot_id": plot_id,
                "history_size": len(self.visualization_state["history"]),
                "registry_size": len(self.visualization_state["plot_registry"]),
            }

            logger.debug(f"Added visualization record for plot {plot_id}")

            return None, stats, None

        except Exception as e:
            logger.error(f"Failed to add visualization record: {e}")
            return None, {"error": str(e)}, None

    def get_visualization_history(
        self, limit: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get recent visualization history.

        Args:
            limit: Maximum number of records to return

        Returns:
            Tuple of (history, stats, None)
        """
        try:
            history = self.visualization_state["history"][-limit:]
            stats = {
                "total_history": len(self.visualization_state["history"]),
                "returned": len(history),
                "limit": limit,
            }
            return history, stats, None
        except Exception as e:
            logger.error(f"Failed to get visualization history: {e}")
            return [], {"error": str(e)}, None

    def get_visualization_settings(self) -> Tuple[Dict[str, Any], Dict[str, Any], None]:
        """
        Get current visualization settings.

        Returns:
            Tuple of (settings, stats, None)
        """
        settings = self.visualization_state["settings"].copy()
        stats = {"settings_count": len(settings)}
        return settings, stats, None

    def update_visualization_settings(
        self, settings: Dict[str, Any]
    ) -> Tuple[None, Dict[str, Any], None]:
        """
        Update visualization settings.

        Args:
            settings: Settings to update

        Returns:
            Tuple of (None, stats, None)
        """
        try:
            self.visualization_state["settings"].update(settings)
            stats = {
                "updated_keys": list(settings.keys()),
                "timestamp": datetime.now().isoformat(),
            }
            logger.debug(f"Updated visualization settings: {settings}")
            return None, stats, None
        except Exception as e:
            logger.error(f"Failed to update visualization settings: {e}")
            return None, {"error": str(e)}, None

    def get_plot_by_uuid(
        self, plot_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], None]:
        """
        Get plot metadata by UUID from visualization registry.

        Args:
            plot_id: UUID of the plot

        Returns:
            Tuple of (metadata|None, stats, None)
        """
        metadata = self.visualization_state["plot_registry"].get(plot_id)
        stats = {
            "found": metadata is not None,
            "plot_id": plot_id,
        }
        return metadata, stats, None

    def clear_visualization_history(self) -> Tuple[None, Dict[str, Any], None]:
        """
        Clear visualization history and registry.

        Returns:
            Tuple of (None, stats, None)
        """
        cleared_history = len(self.visualization_state["history"])
        cleared_registry = len(self.visualization_state["plot_registry"])

        self.visualization_state["history"] = []
        self.visualization_state["plot_registry"] = {}

        stats = {
            "cleared_history": cleared_history,
            "cleared_registry": cleared_registry,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Cleared visualization history and registry")

        return None, stats, None

    # ========================================
    # PROVENANCE IR HELPERS
    # ========================================

    def _create_add_plot_ir(
        self,
        title: Optional[str],
        source: Optional[str],
        dataset_info: Dict[str, Any],
    ) -> AnalysisStep:
        """Create provenance IR for add_plot operation."""
        return AnalysisStep(
            operation="plot_manager.add_plot",
            tool_name="PlotManager.add_plot",
            description=f"Add plot '{title or 'Untitled'}' from {source or 'unknown'}",
            library="plotly",
            imports=[
                "from lobster.core.plot_manager import PlotManager",
                "import plotly.graph_objects as go",
            ],
            code_template="""# Add plot to PlotManager
# (Plot figure created by analysis service)
plot_id, stats, ir = plot_manager.add_plot(
    plot=fig,
    title="{{ title }}",
    source="{{ source }}",
)
print(f"Added plot: {plot_id}")
""",
            parameters={
                "title": title,
                "source": source,
                "dataset_info": dataset_info,
            },
            parameter_schema={
                "title": {
                    "type": "string",
                    "optional": True,
                    "description": "Title for the plot",
                },
                "source": {
                    "type": "string",
                    "optional": True,
                    "description": "Source identifier (service name)",
                },
            },
            input_entities=["plot_figure"],
            output_entities=["plot_id"],
        )

    def _create_clear_plots_ir(self) -> AnalysisStep:
        """Create provenance IR for clear_plots operation."""
        return AnalysisStep(
            operation="plot_manager.clear_plots",
            tool_name="PlotManager.clear_plots",
            description="Clear all stored plots",
            library="lobster",
            imports=["from lobster.core.plot_manager import PlotManager"],
            code_template="""# Clear all plots
_, stats, ir = plot_manager.clear_plots()
print(f"Cleared {stats['cleared_count']} plots")
""",
            parameters={},
            parameter_schema={},
            input_entities=[],
            output_entities=[],
        )

    def _create_save_plots_ir(self, saved_count: int) -> AnalysisStep:
        """Create provenance IR for save_plots_to_workspace operation."""
        return AnalysisStep(
            operation="plot_manager.save_plots_to_workspace",
            tool_name="PlotManager.save_plots_to_workspace",
            description=f"Save {saved_count} plots to workspace",
            library="plotly",
            imports=[
                "from lobster.core.plot_manager import PlotManager",
            ],
            code_template="""# Save plots to workspace
saved_files, stats, ir = plot_manager.save_plots_to_workspace()
print(f"Saved {len(saved_files)} files to {stats['workspace_path']}")
""",
            parameters={"saved_count": saved_count},
            parameter_schema={},
            input_entities=["latest_plots"],
            output_entities=["saved_files"],
        )
