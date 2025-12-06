"""
Execution Context Builder for custom code execution.

This module inspects the workspace and builds a rich execution namespace
with automatic data loading for modalities, CSV files, JSON files, queues, etc.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionContextBuilder:
    """
    Build execution context with workspace data injection.

    This class inspects the workspace directory and automatically injects:
    - data_manager: DataManagerV2 instance
    - workspace_path: Path to workspace
    - adata: Loaded modality (if modality_name specified)
    - modalities: List of all available modality names
    - CSV files: Loaded as pandas DataFrames with sanitized names
    - JSON files: Loaded as Python dicts with sanitized names
    - Queue files: download_queue, publication_queue (if exist)
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize the context builder.

        Args:
            data_manager: DataManagerV2 instance for workspace access
        """
        self.data_manager = data_manager
        self.workspace_path = data_manager.workspace_path
        logger.debug(
            f"Initialized ExecutionContextBuilder with workspace: {self.workspace_path}"
        )

    def build_context(
        self, modality_name: Optional[str] = None, load_workspace_files: bool = True
    ) -> Dict[str, Any]:
        """
        Build execution namespace with workspace data.

        Args:
            modality_name: Optional specific modality to load as 'adata'
            load_workspace_files: Whether to auto-load CSV/JSON files

        Returns:
            Dictionary with execution namespace

        Example:
            >>> builder = ExecutionContextBuilder(data_manager)
            >>> context = builder.build_context(modality_name="geo_gse12345")
            >>> context.keys()
            dict_keys(['data_manager', 'workspace_path', 'modalities', 'adata', 'pd', 'Path'])
        """
        context = {}

        # Core objects (always present)
        context["data_manager"] = self.data_manager
        context["workspace_path"] = self.workspace_path
        context["modalities"] = self.data_manager.list_modalities()

        logger.debug(
            f"Building context with {len(context['modalities'])} modalities available"
        )

        # Load specific modality if requested
        if modality_name:
            if modality_name not in context["modalities"]:
                logger.warning(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {context['modalities']}"
                )
            else:
                context["adata"] = self.data_manager.get_modality(modality_name)
                logger.debug(
                    f"Loaded modality '{modality_name}' as 'adata' "
                    f"({context['adata'].n_obs} obs × {context['adata'].n_vars} vars)"
                )

        # Load workspace files
        if load_workspace_files:
            csv_data, json_data = self._load_workspace_files()
            context.update(csv_data)
            context.update(json_data)

            logger.debug(
                f"Loaded {len(csv_data)} CSV files, {len(json_data)} JSON/JSONL files"
            )

        # Standard library imports (convenience)
        context["pd"] = pd
        context["Path"] = Path

        logger.debug(f"Built context with {len(context)} keys")
        return context

    def _load_workspace_files(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Discover and load CSV/JSON files from workspace.

        Returns:
            Tuple of (csv_dict, json_dict) with sanitized variable names

        Example:
            >>> csv_data, json_data = builder._load_workspace_files()
            >>> csv_data.keys()
            dict_keys(['metadata', 'sample_info'])
            >>> json_data.keys()
            dict_keys(['config', 'download_queue', 'publication_queue'])
        """
        csv_data = {}
        json_data = {}

        # Load CSV files
        for csv_file in self.workspace_path.glob("*.csv"):
            var_name = self._sanitize_filename(csv_file.stem)
            try:
                csv_data[var_name] = pd.read_csv(csv_file)
                logger.debug(
                    f"Loaded CSV: {csv_file.name} -> {var_name} "
                    f"({csv_data[var_name].shape[0]} rows)"
                )
            except Exception as e:
                logger.warning(f"Failed to load CSV {csv_file.name}: {e}")

        # Load JSON files (exclude special files like .session.json)
        for json_file in self.workspace_path.glob("*.json"):
            if json_file.name.startswith("."):
                continue  # Skip hidden files

            var_name = self._sanitize_filename(json_file.stem)
            try:
                with open(json_file) as f:
                    json_data[var_name] = json.load(f)
                logger.debug(f"Loaded JSON: {json_file.name} -> {var_name}")
            except Exception as e:
                logger.warning(f"Failed to load JSON {json_file.name}: {e}")

        # Load JSONL queue files (special handling)
        queue_files = {
            "download_queue": "download_queue.jsonl",
            "publication_queue": "publication_queue.jsonl",
        }

        for var_name, filename in queue_files.items():
            queue_path = self.workspace_path / filename
            if queue_path.exists():
                try:
                    entries = []
                    with open(queue_path) as f:
                        for line in f:
                            line = line.strip()
                            if line:  # Skip empty lines
                                entries.append(json.loads(line))
                    json_data[var_name] = entries
                    logger.debug(
                        f"Loaded queue: {filename} -> {var_name} "
                        f"({len(entries)} entries)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load queue {filename}: {e}")

        return csv_data, json_data

    def _sanitize_filename(self, filename: str) -> str:
        """
        Convert filename to valid Python variable name.

        Args:
            filename: Original filename (without extension)

        Returns:
            Sanitized variable name

        Example:
            >>> builder._sanitize_filename("geo_gse12345_metadata")
            'geo_gse12345_metadata'
            >>> builder._sanitize_filename("sample-data")
            'sample_data'
            >>> builder._sanitize_filename("2024_results")
            'data_2024_results'
            >>> builder._sanitize_filename("my data!")
            'my_data'
        """
        # Replace hyphens and spaces with underscores
        sanitized = filename.replace("-", "_").replace(" ", "_")

        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = "data_" + sanitized

        # Remove non-alphanumeric characters (except underscores)
        sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")

        # Handle empty result (shouldn't happen, but be safe)
        if not sanitized:
            sanitized = "data"

        return sanitized
