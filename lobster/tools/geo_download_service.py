"""
GEO Download Service implementing IDownloadService interface.

This module provides a wrapper around the existing GEOService to adapt it
to the standardized IDownloadService interface for use with DownloadOrchestrator.

The wrapper handles:
- Converting queue entries to GEOService parameters
- Extracting strategy overrides and mapping them to GEO-specific formats
- Retrieving stored modalities after GEOService completes (since it returns strings)
- Generating proper AnalysisStep IR for provenance tracking
- Validating GEO-specific strategy parameters

Example usage:
    >>> from lobster.core.data_manager_v2 import DataManagerV2
    >>> from lobster.tools.download_orchestrator import DownloadOrchestrator
    >>> from lobster.tools.geo_download_service import GEODownloadService
    >>>
    >>> # Initialize orchestrator and register GEO service
    >>> dm = DataManagerV2(workspace_dir="./workspace")
    >>> orchestrator = DownloadOrchestrator(dm)
    >>> geo_service = GEODownloadService(dm)
    >>> orchestrator.register_service(geo_service)
    >>>
    >>> # Execute download from queue
    >>> modality_name, stats = orchestrator.execute_download("queue_GSE12345_abc123")
    >>> print(f"Downloaded: {modality_name}")
"""

from typing import Any, Dict, List, Optional, Tuple

import anndata as ad

from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.schemas.download_queue import DownloadQueueEntry
from lobster.tools.geo_service import GEOService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class GEODownloadService(IDownloadService):
    """
    GEO database download service implementing IDownloadService.

    Wraps existing GEOService to provide standardized interface for
    DownloadOrchestrator integration. Handles conversion between queue-based
    parameters and GEOService's internal API.

    Key Adaptation Logic:
        - GEOService.download_dataset() returns a string message, not a tuple
        - After calling GEOService, must retrieve stored modality from data_manager
        - Strategy mapping: queue strategy_name → GEO manual_strategy_override
        - Parameter extraction: strategy_params["use_intersecting_genes_only"]

    Attributes:
        data_manager (DataManagerV2): Data manager for storage and provenance
        geo_service (GEOService): Wrapped GEO service instance
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize GEO download service with data manager.

        Args:
            data_manager: DataManagerV2 instance for data storage and provenance tracking
        """
        super().__init__(data_manager)
        self.geo_service = GEOService(data_manager)
        logger.info("GEODownloadService initialized with wrapped GEOService")

    @classmethod
    def supports_database(cls, database: str) -> bool:
        """
        Check if this service handles GEO database.

        Args:
            database: Database identifier (case-insensitive)

        Returns:
            bool: True for 'geo' or 'GEO', False otherwise

        Example:
            >>> GEODownloadService.supports_database("geo")
            True
            >>> GEODownloadService.supports_database("GEO")
            True
            >>> GEODownloadService.supports_database("sra")
            False
        """
        return database.lower() == "geo"

    def supported_databases(self) -> List[str]:
        """
        Get list of databases supported by this service.

        Returns:
            List[str]: List containing ["geo"]

        Example:
            >>> service = GEODownloadService(data_manager)
            >>> service.supported_databases()
            ['geo']
        """
        return ["geo"]

    def validate_strategy(self, strategy_override: Dict[str, Any]) -> None:
        """
        Validate strategy override parameters.

        This is a wrapper around validate_strategy_params that raises ValueError
        on validation failure, as expected by DownloadOrchestrator.

        Args:
            strategy_override: Strategy override dictionary containing:
                - strategy_name: Strategy name (optional validation)
                - strategy_params: Database-specific parameters

        Raises:
            ValueError: If any validation fails

        Example:
            >>> service = GEODownloadService(data_manager)
            >>> service.validate_strategy({
            ...     "strategy_name": "MATRIX_FIRST",
            ...     "strategy_params": {"use_intersecting_genes_only": True}
            ... })
        """
        # Validate strategy_params if present
        if (
            "strategy_params" in strategy_override
            and strategy_override["strategy_params"]
        ):
            is_valid, error_msg = self.validate_strategy_params(
                strategy_override["strategy_params"]
            )
            if not is_valid:
                raise ValueError(error_msg)

        # Validate strategy_name if present
        if "strategy_name" in strategy_override:
            strategy_name = strategy_override["strategy_name"]
            supported = self.get_supported_strategies()
            if strategy_name and strategy_name not in supported:
                raise ValueError(
                    f"Strategy '{strategy_name}' not supported by GEODownloadService. "
                    f"Supported strategies: {', '.join(supported)}"
                )

    def download_dataset(
        self,
        queue_entry: DownloadQueueEntry,
        strategy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Execute GEO dataset download from queue entry.

        This method adapts the queue-based parameters to GEOService's API,
        executes the download, retrieves the stored modality, and constructs
        the required return tuple.

        Critical Implementation Notes:
            - GEOService.download_dataset() returns a STRING message
            - After calling it, must retrieve stored modality from data_manager
            - Modality name pattern: geo_{gse_id}_{adapter} (e.g., "geo_gse180759_transcriptomics_single_cell")
            - Must build stats dict and IR manually since GEOService doesn't return them

        Args:
            queue_entry: Queue entry prepared by research_agent containing:
                - dataset_id: GEO accession (e.g., "GSE180759")
                - database: Must be "geo"
                - metadata: Full GEO metadata from research_agent
                - recommended_strategy: Strategy config with strategy_name and concatenation_strategy
                - validation_result: Metadata validation results

            strategy_override: Optional dictionary to override queue entry strategy.
                If provided, should contain:
                - strategy_name: One of GEO strategies (see get_supported_strategies)
                - strategy_params: Dict with optional:
                    - use_intersecting_genes_only: bool (concatenation strategy)

        Returns:
            Tuple containing:
                - adata (AnnData): Downloaded and processed dataset
                - stats (Dict[str, Any]): Download statistics with keys:
                    - dataset_id: GEO accession ID
                    - database: "geo"
                    - modality_name: Name of stored modality
                    - shape: {n_obs: int, n_vars: int}
                    - strategy_used: Strategy name that was executed
                    - download_message: String message from GEOService
                - ir (AnalysisStep): Provenance tracking for reproducibility

        Raises:
            ValueError: If queue_entry.database is not "geo" or dataset_id invalid
            RuntimeError: If GEOService fails or modality cannot be retrieved after download

        Example:
            >>> entry = DownloadQueueEntry(
            ...     entry_id="queue_123",
            ...     dataset_id="GSE180759",
            ...     database="geo",
            ...     recommended_strategy=StrategyConfig(
            ...         strategy_name="H5_FIRST",
            ...         concatenation_strategy="auto",
            ...         confidence=0.95
            ...     )
            ... )
            >>> service = GEODownloadService(data_manager)
            >>> adata, stats, ir = service.download_dataset(entry)
            >>> print(stats)
            {
                "dataset_id": "GSE180759",
                "database": "geo",
                "modality_name": "geo_gse180759_transcriptomics_single_cell",
                "shape": {"n_obs": 12000, "n_vars": 20000},
                "strategy_used": "H5_FIRST",
                "download_message": "Successfully downloaded..."
            }
        """
        logger.info(f"Starting GEO download for dataset: {queue_entry.dataset_id}")

        # Step 1: Validate queue entry
        if queue_entry.database.lower() != "geo":
            raise ValueError(
                f"GEODownloadService only handles 'geo' database, "
                f"got '{queue_entry.database}'"
            )

        geo_id = queue_entry.dataset_id.strip().upper()
        if not geo_id.startswith("GSE"):
            raise ValueError(
                f"Invalid GEO accession ID: {queue_entry.dataset_id}. "
                f"Must be a GSE accession (e.g., GSE180759)"
            )

        logger.debug(f"Validated GEO accession: {geo_id}")

        # Step 2: Determine download strategy
        # Priority: strategy_override > queue_entry.recommended_strategy > None (auto-detect)
        strategy_name = None
        use_intersecting_genes_only = None

        if strategy_override and "strategy_name" in strategy_override:
            # Explicit override provided
            strategy_name = strategy_override["strategy_name"]
            logger.info(f"Using explicit strategy override: {strategy_name}")

            # Extract strategy parameters if present
            if "strategy_params" in strategy_override:
                params = strategy_override["strategy_params"]
                use_intersecting_genes_only = params.get(
                    "use_intersecting_genes_only", None
                )
                if use_intersecting_genes_only is not None:
                    logger.debug(
                        f"Concatenation strategy: use_intersecting_genes_only={use_intersecting_genes_only}"
                    )

        elif queue_entry.recommended_strategy:
            # Use recommended strategy from queue entry
            strategy_name = queue_entry.recommended_strategy.strategy_name
            logger.info(f"Using recommended strategy from queue: {strategy_name}")

            # Map concatenation_strategy to use_intersecting_genes_only
            concat_strategy = queue_entry.recommended_strategy.concatenation_strategy
            if concat_strategy == "intersection":
                use_intersecting_genes_only = True
            elif concat_strategy == "union":
                use_intersecting_genes_only = False
            else:  # "auto" or None
                use_intersecting_genes_only = None

            if use_intersecting_genes_only is not None:
                logger.debug(
                    f"Mapped concatenation strategy '{concat_strategy}' → "
                    f"use_intersecting_genes_only={use_intersecting_genes_only}"
                )

        else:
            # No strategy specified - GEOService will auto-detect
            logger.info("No strategy specified - using GEOService auto-detection")

        # Step 3: Call GEOService.download_dataset()
        # CRITICAL: This returns a STRING message, not a tuple
        try:
            logger.info(f"Calling GEOService.download_dataset for {geo_id}...")
            download_message = self.geo_service.download_dataset(
                geo_id=geo_id,
                manual_strategy_override=strategy_name,
                use_intersecting_genes_only=use_intersecting_genes_only,
            )
            logger.info(f"GEOService completed: {download_message[:200]}...")
        except Exception as e:
            logger.error(f"GEOService.download_dataset failed for {geo_id}: {e}")
            raise RuntimeError(
                f"Failed to download GEO dataset {geo_id}: {str(e)}"
            ) from e

        # Step 4: Retrieve stored modality from data_manager
        # GEOService stores modality internally with pattern: geo_{gse_id}_{adapter}
        modality_name = self._find_stored_modality(geo_id)
        if modality_name is None:
            raise RuntimeError(
                f"GEOService claimed success but no modality found for {geo_id}. "
                f"Available modalities: {self.data_manager.list_modalities()}"
            )

        logger.info(f"Found stored modality: {modality_name}")

        # Step 5: Retrieve AnnData object
        try:
            adata = self.data_manager.get_modality(modality_name)
            logger.info(f"Retrieved AnnData: {adata.n_obs} obs × {adata.n_vars} vars")
        except Exception as e:
            logger.error(f"Failed to retrieve modality '{modality_name}': {e}")
            raise RuntimeError(
                f"Failed to retrieve stored modality '{modality_name}': {str(e)}"
            ) from e

        # Step 6: Build statistics dictionary
        stats = {
            "dataset_id": geo_id,
            "database": "geo",
            "modality_name": modality_name,
            "shape": {"n_obs": adata.n_obs, "n_vars": adata.n_vars},
            "strategy_used": strategy_name or "auto",
            "concatenation_strategy": (
                "intersection"
                if use_intersecting_genes_only
                else "union" if use_intersecting_genes_only is False else "auto"
            ),
            "download_message": download_message,
        }

        logger.debug(f"Built stats dict: {stats}")

        # Step 7: Create AnalysisStep IR for provenance
        ir = self._create_analysis_step_ir(
            geo_id=geo_id,
            modality_name=modality_name,
            strategy_name=strategy_name,
            use_intersecting_genes_only=use_intersecting_genes_only,
            queue_entry_id=queue_entry.entry_id,
        )

        logger.info(f"Successfully completed GEO download for {geo_id}")
        return adata, stats, ir

    def _find_stored_modality(self, geo_id: str) -> Optional[str]:
        """
        Find stored modality name for a GEO dataset.

        GEOService stores modalities with pattern: geo_{gse_id}_{adapter}
        This method searches for matching modality names.

        Args:
            geo_id: GEO accession ID (e.g., "GSE180759")

        Returns:
            str: Modality name if found, None otherwise

        Example:
            >>> service._find_stored_modality("GSE180759")
            "geo_gse180759_transcriptomics_single_cell"
        """
        # Get all modalities from data_manager
        all_modalities = self.data_manager.list_modalities()

        # Search for modality matching pattern: geo_{gse_id}_*
        geo_id_lower = geo_id.lower()
        prefix = f"geo_{geo_id_lower}_"

        matching_modalities = [m for m in all_modalities if m.startswith(prefix)]

        if not matching_modalities:
            logger.warning(
                f"No modalities found with prefix '{prefix}'. "
                f"Available modalities: {all_modalities}"
            )
            return None

        if len(matching_modalities) > 1:
            logger.warning(
                f"Multiple modalities found for {geo_id}: {matching_modalities}. "
                f"Using first match: {matching_modalities[0]}"
            )

        modality_name = matching_modalities[0]
        logger.debug(f"Found matching modality: {modality_name}")
        return modality_name

    def _create_analysis_step_ir(
        self,
        geo_id: str,
        modality_name: str,
        strategy_name: Optional[str],
        use_intersecting_genes_only: Optional[bool],
        queue_entry_id: str,
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance tracking.

        Generates intermediate representation capturing all parameters
        and execution details for reproducibility.

        Args:
            geo_id: GEO accession ID
            modality_name: Name of stored modality
            strategy_name: Download strategy used (or None for auto)
            use_intersecting_genes_only: Concatenation strategy used
            queue_entry_id: Queue entry ID for traceability

        Returns:
            AnalysisStep: Provenance tracking object

        Note:
            The code_template uses only standard libraries as per Lobster conventions.
            It references the DownloadOrchestrator pattern for notebook reproduction.
        """
        # Build parameters dictionary
        parameters = {
            "queue_entry_id": queue_entry_id,
            "dataset_id": geo_id,
            "database": "geo",
            "strategy": strategy_name or "auto",
            "use_intersecting_genes_only": use_intersecting_genes_only,
        }

        # Build parameter schema for validation
        parameter_schema = {
            "queue_entry_id": {
                "type": "string",
                "required": True,
                "description": "Queue entry identifier for traceability",
            },
            "dataset_id": {
                "type": "string",
                "required": True,
                "description": "GEO accession ID (e.g., GSE180759)",
                "pattern": "^GSE\\d+$",
            },
            "database": {
                "type": "string",
                "required": True,
                "description": "Source database (must be 'geo')",
                "enum": ["geo"],
            },
            "strategy": {
                "type": "string",
                "optional": True,
                "description": "Download strategy name or 'auto' for auto-detection",
                "enum": [
                    "auto",
                    "H5_FIRST",
                    "MATRIX_FIRST",
                    "SUPPLEMENTARY_FIRST",
                    "SAMPLES_FIRST",
                    "RAW_FIRST",
                ],
            },
            "use_intersecting_genes_only": {
                "type": "boolean",
                "optional": True,
                "description": "Concatenation strategy: True=intersection, False=union, None=auto",
            },
        }

        # Create AnalysisStep IR
        ir = AnalysisStep(
            operation="geo_service.download_dataset",
            tool_name="GEODownloadService.download_dataset",
            description=f"Downloaded GEO dataset {geo_id} using orchestrated download service",
            library="lobster",
            imports=[
                "from lobster.tools.download_orchestrator import DownloadOrchestrator",
                "from lobster.tools.geo_download_service import GEODownloadService",
                "from lobster.core.data_manager_v2 import DataManagerV2",
            ],
            code_template="""# Download GEO dataset using DownloadOrchestrator
data_manager = DataManagerV2(workspace_dir="./workspace")
orchestrator = DownloadOrchestrator(data_manager)
geo_service = GEODownloadService(data_manager)
orchestrator.register_service(geo_service)

# Execute download from queue entry
modality_name, stats = orchestrator.execute_download("{{ queue_entry_id }}")
print(f"Downloaded: {modality_name}")
print(f"Stats: {stats}")

# Access downloaded data
adata = data_manager.get_modality(modality_name)
""",
            parameters=parameters,
            parameter_schema=parameter_schema,
            input_entities=[queue_entry_id],
            output_entities=[modality_name],
        )

        logger.debug(f"Created AnalysisStep IR for {geo_id}")
        return ir

    def validate_strategy_params(
        self, strategy_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate GEO-specific strategy parameters.

        Checks parameter types and allowed values for GEO download strategies.
        Unknown parameters are ignored for forward compatibility.

        Args:
            strategy_params: Dictionary of strategy-specific parameters.
                Recognized parameters:
                - use_intersecting_genes_only: bool (concatenation strategy)

        Returns:
            Tuple containing:
                - is_valid (bool): True if all parameters are valid
                - error_message (Optional[str]): Descriptive error if invalid, None if valid

        Example:
            >>> service = GEODownloadService(data_manager)
            >>>
            >>> # Valid parameters
            >>> valid, error = service.validate_strategy_params({
            ...     "use_intersecting_genes_only": True
            ... })
            >>> assert valid is True and error is None
            >>>
            >>> # Invalid parameters
            >>> valid, error = service.validate_strategy_params({
            ...     "use_intersecting_genes_only": "invalid"  # should be bool
            ... })
            >>> assert valid is False
            >>> print(error)
            "Parameter 'use_intersecting_genes_only' must be bool or None, got str"
        """
        logger.debug(f"Validating strategy parameters: {strategy_params}")

        # Check use_intersecting_genes_only parameter
        if "use_intersecting_genes_only" in strategy_params:
            value = strategy_params["use_intersecting_genes_only"]

            # Must be bool or None
            if value is not None and not isinstance(value, bool):
                error_msg = (
                    f"Parameter 'use_intersecting_genes_only' must be bool or None, "
                    f"got {type(value).__name__}"
                )
                logger.warning(f"Validation failed: {error_msg}")
                return False, error_msg

            logger.debug(f"Valid: use_intersecting_genes_only={value}")

        # All known parameters are valid (unknown parameters ignored for forward compatibility)
        logger.debug("Strategy parameters validation passed")
        return True, None

    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        Get list of download strategy names supported by GEOService.

        These strategies correspond to the pipeline types in GEOService's
        pipeline strategy engine.

        Returns:
            List[str]: Strategy names supported by GEO download service

        Example:
            >>> GEODownloadService.get_supported_strategies()
            ["H5_FIRST", "MATRIX_FIRST", "SUPPLEMENTARY_FIRST", "SAMPLES_FIRST", "RAW_FIRST"]

        Note:
            Strategy names are uppercase by convention. Order reflects preference
            (most efficient strategies first).
        """
        return [
            "H5_FIRST",  # HDF5 format (most efficient, single-file)
            "MATRIX_FIRST",  # Processed matrix files (fast, pre-computed)
            "SUPPLEMENTARY_FIRST",  # Supplementary files (flexible, comprehensive)
            "SAMPLES_FIRST",  # Individual sample files (detailed, per-sample QC)
            "RAW_FIRST",  # Raw UMI/count matrices (unprocessed, maximum control)
        ]
