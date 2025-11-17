"""
Professional file naming utility for bioinformatics data pipeline.

This module provides standardized naming conventions for all data files
throughout the lobster system, ensuring consistency and traceability.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class BioinformaticsFileNaming:
    """
    Professional file naming utility for bioinformatics data pipeline.

    Implements consistent naming pattern: <data_source>_<dataset_id>_<processing_step>_<timestamp>.<extension>

    Examples:
        - GEO_GSE235449_raw_matrix_20250815_120530.csv
        - GEO_GSE235449_filtered_20250815_120545.csv
        - GEO_GSE235449_normalized_20250815_120600.csv
        - GEO_GSE235449_batch_corrected_20250815_120615.csv
    """

    # Standard data sources
    DATA_SOURCES = {
        "GEO": "GEO",
        "ARRAYEXPRESS": "AE",
        "TCGA": "TCGA",
        "SRA": "SRA",
        "UPLOAD": "UPLOAD",
        "UNKNOWN": "DATA",
    }

    # Standard processing steps in typical analysis order
    PROCESSING_STEPS = {
        "raw_matrix": "raw_matrix",
        "combined": "combined",
        "filtered": "filtered",
        "normalized": "normalized",
        "scaled": "scaled",
        "batch_corrected": "batch_corrected",
        "dimensionality_reduced": "dimred",
        "clustered": "clustered",
        "annotated": "annotated",
        "differential": "differential",
        "enrichment": "enrichment",
    }

    # Extension mapping for different processing steps and data types
    EXTENSION_MAPPING = {
        "metadata": "json",
        "raw_matrix": "csv",
        "combined": "csv",
        "filtered": "csv",
        "normalized": "csv",
        "scaled": "npz",  # Compressed format for scaled data
        "batch_corrected": "h5ad",  # AnnData format for complex processed data
        "dimred": "npz",  # Numpy compressed for dimensionality reduced data
        "clustered": "h5ad",
        "annotated": "h5ad",
        "differential": "csv",
        "enrichment": "json",
        "quality_report": "json",
        "plots": "html",
    }

    @classmethod
    def generate_filename(
        self,
        data_source: str,
        dataset_id: str,
        processing_step: str,
        extension: str = None,
        timestamp: Optional[str] = None,
        custom_suffix: Optional[str] = None,
    ) -> str:
        """
        Generate a professional filename following the standard naming convention.

        Args:
            data_source: Source of the data (e.g., 'GEO', 'TCGA', 'UPLOAD')
            dataset_id: Unique identifier for the dataset (e.g., 'GSE235449')
            processing_step: Current processing step (e.g., 'raw_matrix', 'filtered')
            extension: File extension without dot (auto-selected if None based on processing step)
            timestamp: Custom timestamp string (auto-generated if None)
            custom_suffix: Additional suffix for special cases

        Returns:
            str: Professional filename following standard convention

        Examples:
            >>> BioinformaticsFileNaming.generate_filename('GEO', 'GSE235449', 'raw_matrix')
            'GEO_GSE235449_raw_matrix_20250815_120530.csv'

            >>> BioinformaticsFileNaming.generate_filename('GEO', 'GSE235449', 'scaled')
            'GEO_GSE235449_scaled_20250815_120530.npz'

            >>> BioinformaticsFileNaming.generate_filename('GEO', 'GSE235449', 'metadata')
            'GEO_GSE235449_metadata_20250815_120530.json'
        """
        # Normalize data source
        data_source = data_source.upper()
        if data_source not in self.DATA_SOURCES:
            data_source = "DATA"
        else:
            data_source = self.DATA_SOURCES[data_source]

        # Normalize dataset ID (remove spaces, special chars except underscore and hyphen)
        dataset_id = re.sub(r"[^a-zA-Z0-9_-]", "", dataset_id)

        # Normalize processing step
        processing_step = processing_step.lower().replace(" ", "_")
        if processing_step in self.PROCESSING_STEPS:
            processing_step = self.PROCESSING_STEPS[processing_step]

        # Auto-select extension based on processing step if not provided
        if extension is None:
            extension = self.EXTENSION_MAPPING.get(processing_step, "csv")

        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build filename components
        components = [data_source, dataset_id, processing_step, timestamp]

        # Add custom suffix if provided
        if custom_suffix:
            custom_suffix = re.sub(r"[^a-zA-Z0-9_-]", "", custom_suffix)
            components.append(custom_suffix)

        # Join components and add extension
        filename = "_".join(components) + f".{extension}"

        return filename

    @classmethod
    def generate_metadata_filename(self, data_filename: str) -> str:
        """
        Generate corresponding metadata filename for a data file.

        Args:
            data_filename: Name of the data file

        Returns:
            str: Metadata filename

        Example:
            >>> BioinformaticsFileNaming.generate_metadata_filename('GEO_GSE235449_raw_matrix_20250815_120530.csv')
            'GEO_GSE235449_raw_matrix_20250815_120530_metadata.json'
        """
        # Remove extension and add metadata suffix
        base_name = Path(data_filename).stem
        return f"{base_name}_metadata.json"

    @classmethod
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse a professional filename to extract its components.

        Args:
            filename: Professional filename to parse

        Returns:
            dict: Dictionary with parsed components

        Example:
            >>> BioinformaticsFileNaming.parse_filename('GEO_GSE235449_filtered_20250815_120530.csv')
            {
                'data_source': 'GEO',
                'dataset_id': 'GSE235449',
                'processing_step': 'filtered',
                'timestamp': '20250815_120530',
                'extension': 'csv'
            }
        """
        try:
            path = Path(filename)
            extension = path.suffix.lstrip(".")
            name_parts = path.stem.split("_")

            if len(name_parts) < 4:
                return {
                    "data_source": "UNKNOWN",
                    "dataset_id": "UNKNOWN",
                    "processing_step": "unknown",
                    "timestamp": "unknown",
                    "extension": extension,
                }

            # Standard format: DATA_SOURCE_DATASET_ID_PROCESSING_STEP_TIMESTAMP[_SUFFIX]
            data_source = name_parts[0]
            dataset_id = name_parts[1]

            # Handle timestamp (last two parts: YYYYMMDD_HHMMSS)
            if len(name_parts) >= 4:
                # Find timestamp pattern (YYYYMMDD_HHMMSS)
                timestamp_parts = []
                processing_parts = []

                for i, part in enumerate(name_parts[2:], 2):
                    if (
                        re.match(r"\d{8}", part)
                        and i + 1 < len(name_parts)
                        and re.match(r"\d{6}", name_parts[i + 1])
                    ):
                        # Found timestamp start
                        timestamp_parts = [part, name_parts[i + 1]]
                        processing_parts = name_parts[2:i]
                        break

                if timestamp_parts:
                    timestamp = "_".join(timestamp_parts)
                    processing_step = (
                        "_".join(processing_parts) if processing_parts else "unknown"
                    )
                else:
                    # No timestamp found, assume last part is processing step
                    processing_step = (
                        "_".join(name_parts[2:-2])
                        if len(name_parts) > 4
                        else name_parts[2]
                    )
                    timestamp = "_".join(name_parts[-2:])
            else:
                processing_step = name_parts[2]
                timestamp = name_parts[3] if len(name_parts) > 3 else "unknown"

            return {
                "data_source": data_source,
                "dataset_id": dataset_id,
                "processing_step": processing_step,
                "timestamp": timestamp,
                "extension": extension,
            }

        except Exception:
            return {
                "data_source": "UNKNOWN",
                "dataset_id": "UNKNOWN",
                "processing_step": "unknown",
                "timestamp": "unknown",
                "extension": (
                    Path(filename).suffix.lstrip(".") if "." in filename else "unknown"
                ),
            }

    @classmethod
    def find_latest_file(
        self,
        directory: Path,
        data_source: str,
        dataset_id: str,
        processing_step: str,
        extension: str = None,
    ) -> Optional[Path]:
        """
        Find the latest file for a specific dataset and processing step.

        Args:
            directory: Directory to search in
            data_source: Data source to match
            dataset_id: Dataset ID to match
            processing_step: Processing step to match
            extension: Specific extension to match (auto-detected if None)

        Returns:
            Path: Path to the latest file, or None if not found
        """
        if not directory.exists():
            return None

        # Auto-select extension if not provided
        if extension is None:
            extension = self.EXTENSION_MAPPING.get(processing_step, "*")

        # Create pattern to match with specific or wildcard extension
        if extension == "*":
            pattern = f"{data_source.upper()}_{dataset_id}_{processing_step}_*.*"
        else:
            pattern = (
                f"{data_source.upper()}_{dataset_id}_{processing_step}_*.{extension}"
            )

        matching_files = list(directory.glob(pattern))

        if not matching_files:
            return None

        # Sort by timestamp (newest first)
        def extract_timestamp(filepath):
            try:
                parsed = self.parse_filename(filepath.name)
                return parsed["timestamp"]
            except Exception:
                return "00000000_000000"

        matching_files.sort(key=extract_timestamp, reverse=True)
        return matching_files[0]

    @classmethod
    def get_processing_step_order(self, step: str) -> int:
        """
        Get the order index of a processing step for sorting purposes.

        Args:
            step: Processing step name

        Returns:
            int: Order index (lower numbers come first in pipeline)
        """
        step_order = {
            "raw_matrix": 0,
            "combined": 1,
            "filtered": 2,
            "normalized": 3,
            "scaled": 4,
            "batch_corrected": 5,
            "dimred": 6,
            "clustered": 7,
            "annotated": 8,
            "differential": 9,
            "enrichment": 10,
        }

        return step_order.get(step.lower(), 999)

    @classmethod
    def suggest_next_step(self, current_step: str) -> str:
        """
        Suggest the next logical processing step based on the current step.

        Args:
            current_step: Current processing step

        Returns:
            str: Suggested next processing step
        """
        step_progression = {
            "raw_matrix": "filtered",
            "combined": "filtered",
            "filtered": "normalized",
            "normalized": "scaled",
            "scaled": "batch_corrected",
            "batch_corrected": "dimred",
            "dimred": "clustered",
            "clustered": "annotated",
            "annotated": "differential",
        }

        return step_progression.get(current_step.lower(), "processed")
