"""
Content Access Service - Unified Publication Access.

This service consolidates PublicationService and UnifiedContentService into
a single capability-based routing system. It provides intelligent provider
selection, fallback cascades, and comprehensive publication access.

Phase 1: Provider registry infrastructure (complete)
Phase 2: Capability-based routing and service consolidation (complete)

Provides 10 core methods:
- Discovery: search_literature, discover_datasets, find_linked_datasets
- Metadata: extract_metadata, validate_metadata
- Content: get_abstract, get_full_content, extract_methods
- System: query_capabilities

Features three-tier cascade logic (PMC → Webpage → PDF), session caching
via DataManager, and W3C-PROV provenance tracking.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.base_provider import DatasetType, PublicationMetadata
from lobster.tools.providers.geo_provider import GEOProvider
from lobster.tools.providers.pmc_provider import PMCProvider
from lobster.tools.providers.provider_registry import ProviderRegistry
from lobster.tools.providers.pubmed_provider import PubMedProvider
from lobster.tools.providers.sra_provider import SRAProvider
from lobster.tools.providers.webpage_provider import WebpageProvider
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ContentAccessService:
    """
    Unified publication access service with capability-based routing.

    This service consolidates PublicationService and UnifiedContentService into
    a single interface with intelligent provider selection, fallback cascades,
    and comprehensive publication access.

    **Core Methods** (10 total):

    Discovery (3):
    - search_literature: Search PubMed, bioRxiv, medRxiv
    - discover_datasets: Search GEO, SRA, PRIDE (with accession detection)
    - find_linked_datasets: Find datasets linked to publications

    Metadata (2):
    - extract_metadata: Extract structured publication/dataset metadata
    - validate_metadata: Validate GEO dataset metadata quality

    Content (3):
    - get_abstract: Fast abstract retrieval (Tier 1: <500ms)
    - get_full_content: Full-text with 3-tier cascade (PMC → Webpage → PDF)
    - extract_methods: Extract methods section with software detection

    System (1):
    - query_capabilities: Query available providers and capabilities

    **Providers Registered** (5 total):
    1. AbstractProvider - Fast abstract retrieval (priority 10)
    2. PubMedProvider - Literature search, dataset linking (priority 10)
    3. GEOProvider - Dataset discovery, validation (priority 10)
    4. PMCProvider - Full-text from PMC XML (priority 10)
    5. WebpageProvider - Webpage scraping, PDF via Docling (priority 50)

    Note: DoclingService is used internally by WebpageProvider via composition,
    not registered as a separate provider.

    **Features**:
    - Three-tier cascade logic (PMC → Webpage → PDF)
    - Session caching via DataManager
    - W3C-PROV provenance tracking
    - Automatic accession detection (GSM/GSE/GDS/GPL)
    - Performance tiers (Fast/Moderate/Slow)

    Examples:
        >>> service = ContentAccessService(data_manager)
        >>>
        >>> # Literature search
        >>> results = service.search_literature("BRCA1 breast cancer")
        >>>
        >>> # Dataset discovery with accession detection
        >>> results = service.discover_datasets("GSE123456", DatasetType.GEO)
        >>>
        >>> # Full content with cascade
        >>> content = service.get_full_content("PMID:35042229")
        >>>
        >>> # Metadata validation
        >>> report = service.validate_metadata("GSE123456",
        ...     required_fields=["smoking_status", "treatment_response"])
        >>>
        >>> # Check capabilities
        >>> capabilities = service.query_capabilities()
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize ContentAccessService with provider registry.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
                         and provider initialization
        """
        self.data_manager = data_manager
        self.registry = ProviderRegistry()

        # Initialize and register all providers
        self._initialize_providers()

        # Log capability matrix for debugging
        self._log_capability_matrix()

        logger.debug(
            f"ContentAccessService initialized with "
            f"{len(self.registry.get_all_providers())} providers "
            f"(10 methods, 3-tier cascade, session caching)"
        )

    def _initialize_providers(self) -> None:
        """
        Initialize and register all publication providers.

        Providers are registered in order:
        1. AbstractProvider (priority 10) - Fast abstracts
        2. PubMedProvider (priority 10) - Literature search
        3. GEOProvider (priority 10) - GEO dataset discovery
        4. SRAProvider (priority 10) - SRA dataset discovery
        5. PMCProvider (priority 10) - PMC full-text
        6. WebpageProvider (priority 50) - Webpage scraping

        Each provider is instantiated with the DataManagerV2 instance
        for provenance tracking.
        """
        logger.debug("Initializing providers...")

        # 1. AbstractProvider - Fast abstract retrieval
        try:
            abstract_provider = AbstractProvider(data_manager=self.data_manager)
            self.registry.register_provider(abstract_provider)
        except Exception as e:
            logger.error(f"Failed to initialize AbstractProvider: {e}")

        # 2. PubMedProvider - Literature search and dataset linking
        try:
            pubmed_provider = PubMedProvider(data_manager=self.data_manager)
            self.registry.register_provider(pubmed_provider)
        except Exception as e:
            logger.error(f"Failed to initialize PubMedProvider: {e}")

        # 3. GEOProvider - Dataset discovery and validation
        try:
            geo_provider = GEOProvider(data_manager=self.data_manager)
            self.registry.register_provider(geo_provider)
        except Exception as e:
            logger.error(f"Failed to initialize GEOProvider: {e}")

        # 4. SRAProvider - SRA dataset discovery
        try:
            import pysradb  # noqa: F401 - Check if pysradb is available

            sra_provider = SRAProvider(data_manager=self.data_manager)
            self.registry.register_provider(sra_provider)
            logger.debug("SRAProvider registered successfully")
        except ImportError:
            logger.warning(
                "pysradb not available - SRA provider disabled. "
                "Install with: pip install pysradb"
            )
        except Exception as e:
            logger.error(f"Failed to initialize SRAProvider: {e}")

        # 5. PMCProvider - PMC full-text extraction
        try:
            pmc_provider = PMCProvider(data_manager=self.data_manager)
            self.registry.register_provider(pmc_provider)
        except Exception as e:
            logger.error(f"Failed to initialize PMCProvider: {e}")

        # 6. WebpageProvider - Webpage scraping (includes PDF via Docling)
        try:
            webpage_provider = WebpageProvider(data_manager=self.data_manager)
            self.registry.register_provider(webpage_provider)
        except Exception as e:
            logger.error(f"Failed to initialize WebpageProvider: {e}")

        logger.debug(
            f"Provider initialization complete: "
            f"{len(self.registry.get_all_providers())} providers registered"
        )

    def _log_capability_matrix(self) -> None:
        """
        Log the capability matrix for debugging and validation.

        The matrix shows which providers support which capabilities,
        enabling verification of provider registration and capability
        coverage.

        This is particularly useful during development and testing to
        ensure all capabilities are properly declared.
        """
        matrix = self.registry.get_capability_matrix()
        logger.debug("Provider Capability Matrix:\n" + matrix)

        # Log summary statistics
        all_providers = self.registry.get_all_providers()
        logger.debug(f"Total providers registered: {len(all_providers)}")

        # Log priority distribution
        priority_counts = {}
        for provider in all_providers:
            priority = provider.priority
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        logger.debug(f"Priority distribution: {priority_counts}")

        # Log dataset type coverage
        dataset_types = self.registry.get_supported_dataset_types()
        logger.debug(
            f"Supported dataset types: "
            f"{[dt.value for dt in dataset_types] if dataset_types else 'none'}"
        )

    def _create_research_ir(
        self,
        operation: str,
        tool_name: str,
        description: str,
        parameters: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
    ) -> AnalysisStep:
        """
        Create lightweight IR for research operations.

        These IR objects are marked as non-exportable since research operations
        (literature search, dataset discovery) don't need to appear in notebooks.
        They are tracked for provenance but excluded from notebook export.

        Args:
            operation: Operation name (e.g., "search_literature")
            tool_name: Tool name for provenance
            description: Human-readable description
            parameters: Parameters used in operation
            stats: Optional statistics dictionary

        Returns:
            AnalysisStep with exportable=False
        """
        # Build parameter schema for provenance tracking
        parameter_schema = {}
        for param_name, param_value in parameters.items():
            param_type = type(param_value).__name__
            if param_type == "NoneType":
                param_type = "Optional[Any]"
            elif isinstance(param_value, list):
                param_type = "List"
            elif isinstance(param_value, dict):
                param_type = "Dict"

            parameter_schema[param_name] = ParameterSpec(
                param_type=param_type,
                papermill_injectable=False,  # Research params not injectable
                default_value=None,
                required=False,
                description=f"Parameter for {operation}",
            )

        # Create lightweight IR
        return AnalysisStep(
            operation=f"research.{operation}",
            tool_name=tool_name,
            description=description,
            library="lobster",
            code_template="# Research operation - not included in notebook export",
            imports=[],
            parameters=parameters,
            parameter_schema=parameter_schema,
            input_entities=["query"],
            output_entities=["results"],
            execution_context={
                "timestamp": datetime.now().isoformat(),
                "service": "ContentAccessService",
                "statistics": stats or {},
            },
            validates_on_export=False,
            requires_validation=False,
            exportable=False,  # Key flag - exclude from notebook export
        )

    # ========================================================================
    # Public API Methods
    # ========================================================================

    def search_literature(
        self,
        query: str,
        max_results: int = 5,
        sources: Optional[list[str]] = None,
        filters: Optional[dict[str, any]] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any], AnalysisStep]:
        """
        Search for literature using capability-based routing.

        Routes to providers that support SEARCH_LITERATURE capability.
        Supports single or multi-provider search based on sources parameter.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            sources: Optional list of provider names to use (e.g., ["pubmed"])
                    If None, uses all providers with SEARCH_LITERATURE capability
            filters: Optional search filters (provider-specific)
            **kwargs: Additional parameters passed to providers

        Returns:
            Tuple[str, Dict[str, Any], AnalysisStep]:
                - Formatted search results with publications
                - Statistics dictionary
                - Lightweight IR for provenance (exportable=False)

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> results, stats, ir = service.search_literature("BRCA1 breast cancer")
            >>> results, stats, ir = service.search_literature("p53", sources=["pubmed"])
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        logger.info(f"Literature search: {query[:50]}...")

        # Initialize statistics
        stats = {
            "query": query[:100],
            "max_results": max_results,
            "sources": sources,
            "filters": filters,
            "provider_used": None,
            "results_count": 0,
            "execution_time_ms": 0,
        }

        start_time = time.time()

        try:
            # Get providers for SEARCH_LITERATURE capability
            providers = self.registry.get_providers_for_capability(
                ProviderCapability.SEARCH_LITERATURE
            )

            if not providers:
                error_msg = "No available providers for literature search."
                stats["error"] = error_msg
                ir = self._create_research_ir(
                    operation="search_literature",
                    tool_name="search_literature",
                    description=f"Literature search: {query[:50]}",
                    parameters={"query": query, "max_results": max_results},
                    stats=stats,
                )
                return error_msg, stats, ir

            # Filter by sources if specified
            if sources:
                # Map source names to provider classes
                source_filter = [s.lower() for s in sources]
                providers = [
                    p
                    for p in providers
                    if type(p).__name__.lower().replace("provider", "") in source_filter
                ]

                if not providers:
                    error_msg = f"No providers found for sources: {sources}"
                    stats["error"] = error_msg
                    ir = self._create_research_ir(
                        operation="search_literature",
                        tool_name="search_literature",
                        description=f"Literature search: {query[:50]}",
                        parameters={
                            "query": query,
                            "max_results": max_results,
                            "sources": sources,
                        },
                        stats=stats,
                    )
                    return error_msg, stats, ir

            # Use first provider (highest priority)
            provider = providers[0]
            provider_name = type(provider).__name__
            logger.info(f"Using provider: {provider_name}")
            stats["provider_used"] = provider_name

            # Call provider's search method
            results = provider.search_publications(
                query=query, max_results=max_results, filters=filters, **kwargs
            )

            # Count results (approximate based on string content)
            stats["results_count"] = (
                results.count("PMID:") if isinstance(results, str) else 0
            )
            stats["execution_time_ms"] = int((time.time() - start_time) * 1000)

            # Create IR for provenance
            ir = self._create_research_ir(
                operation="search_literature",
                tool_name="search_literature",
                description=f"Literature search: {query[:50]}",
                parameters={
                    "query": query,
                    "max_results": max_results,
                    "sources": sources,
                    "filters": filters,
                },
                stats=stats,
            )

            # Log to provenance (will be updated by agent to include IR)
            self.data_manager.log_tool_usage(
                tool_name="search_literature",
                parameters={
                    "query": query[:100],
                    "max_results": max_results,
                    "provider": provider_name,
                    "filters": filters,
                },
                description="Literature search via ContentAccessService",
            )

            return results, stats, ir

        except Exception as e:
            logger.error(f"Literature search error: {e}", exc_info=True)
            error_msg = f"Literature search error: {str(e)}"
            stats["error"] = str(e)
            stats["execution_time_ms"] = int((time.time() - start_time) * 1000)

            ir = self._create_research_ir(
                operation="search_literature",
                tool_name="search_literature",
                description=f"Literature search failed: {query[:50]}",
                parameters={"query": query, "max_results": max_results},
                stats=stats,
            )
            return error_msg, stats, ir

    def discover_datasets(
        self,
        query: str,
        dataset_type: "DatasetType",
        max_results: int = 5,
        filters: Optional[dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, Any], AnalysisStep]:
        """
        Search for datasets with automatic accession detection.

        Routes to appropriate provider based on dataset type. Automatically
        detects direct accessions (e.g., GSM6204600, GSE12345) and provides
        enhanced information including parent series for sample IDs.

        Args:
            query: Search query or direct accession (e.g., "GSM6204600")
            dataset_type: Type of dataset to search for (DatasetType enum)
            max_results: Maximum number of results to return (default: 5)
            filters: Optional search filters (provider-specific)

        Returns:
            Tuple[str, Dict[str, Any], AnalysisStep]:
                - Formatted dataset search results
                - Statistics dictionary
                - Lightweight IR for provenance (exportable=False)

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> # Direct accession
            >>> results, stats, ir = service.discover_datasets("GSM6204600", DatasetType.GEO)
            >>> # Text search
            >>> results, stats, ir = service.discover_datasets("single-cell RNA-seq", DatasetType.GEO)
        """
        from lobster.tools.providers.geo_utils import (
            extract_accession_info,
        )

        logger.info(f"Dataset search: {query[:50]}... for {dataset_type.value}")

        # Initialize statistics
        stats = {
            "query": query[:100],
            "dataset_type": (
                dataset_type.value
                if hasattr(dataset_type, "value")
                else str(dataset_type)
            ),
            "max_results": max_results,
            "filters": filters,
            "accession_detected": False,
            "results_count": 0,
            "execution_time_ms": 0,
        }

        start_time = time.time()

        try:
            # Check if query is a direct accession
            detected_type, normalized_accession = extract_accession_info(query)

            # If direct accession detected, use accession-specific handling
            if detected_type is not None:
                logger.info(
                    f"Detected direct accession: {normalized_accession} (type: {detected_type.value})"
                )
                stats["accession_detected"] = True
                stats["normalized_accession"] = normalized_accession
                results = self._handle_direct_accession(
                    normalized_accession, detected_type, max_results, filters
                )
            else:
                # Fall back to text-based search
                results = self._handle_text_search(
                    query, dataset_type, max_results, filters
                )

            # Count results (approximate)
            stats["results_count"] = (
                results.count("GSE") if isinstance(results, str) else 0
            )
            stats["execution_time_ms"] = int((time.time() - start_time) * 1000)

            # Create IR for provenance
            ir = self._create_research_ir(
                operation="discover_datasets",
                tool_name="discover_datasets",
                description=f"Dataset discovery: {query[:50]}",
                parameters={
                    "query": query,
                    "dataset_type": str(dataset_type),
                    "max_results": max_results,
                    "filters": filters,
                },
                stats=stats,
            )

            return results, stats, ir

        except Exception as e:
            logger.error(f"Dataset search error: {e}", exc_info=True)
            error_msg = f"Dataset search error: {str(e)}"
            stats["error"] = str(e)
            stats["execution_time_ms"] = int((time.time() - start_time) * 1000)

            ir = self._create_research_ir(
                operation="discover_datasets",
                tool_name="discover_datasets",
                description=f"Dataset discovery failed: {query[:50]}",
                parameters={"query": query, "dataset_type": str(dataset_type)},
                stats=stats,
            )
            return error_msg, stats, ir

    def _handle_direct_accession(
        self,
        accession: str,
        accession_type: "DatasetType",
        max_results: int,
        filters: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Handle direct accession searches with enhanced functionality.

        Args:
            accession: Normalized accession string
            accession_type: Detected accession type
            max_results: Maximum results
            filters: Additional filters

        Returns:
            str: Formatted search results with enhanced information
        """
        from lobster.tools.providers.base_provider import DatasetType
        from lobster.tools.providers.geo_utils import is_geo_sample_accession

        # Get provider for this dataset type
        provider = self.registry.get_provider_for_dataset_type(accession_type)

        if not provider:
            return f"No provider available for {accession_type.value} accession lookup."

        # Special handling for GEO accessions
        if accession_type == DatasetType.GEO:
            # Check if it's a GSM sample that needs parent lookup
            if is_geo_sample_accession(accession):
                logger.info(
                    f"GSM sample detected: {accession}, searching with parent series lookup"
                )

                # Use enhanced search_by_accession if available
                if hasattr(provider, "search_by_accession"):
                    results = provider.search_by_accession(
                        accession, include_parent_series=True
                    )
                else:
                    # Fallback to regular search
                    results = provider.search_publications(
                        query=accession, max_results=max_results, filters=filters
                    )
            else:
                # For GSE, GDS, GPL - use enhanced accession search if available
                if hasattr(provider, "search_by_accession"):
                    results = provider.search_by_accession(
                        accession, include_parent_series=False
                    )
                else:
                    results = provider.search_publications(
                        query=accession, max_results=max_results, filters=filters
                    )
        else:
            # Standard search for other dataset types
            results = provider.search_publications(
                query=accession, max_results=max_results, filters=filters
            )

        # Log to provenance
        self.data_manager.log_tool_usage(
            tool_name="discover_datasets",
            parameters={
                "accession": accession,
                "type": accession_type.value,
                "provider": type(provider).__name__,
            },
            description="Direct accession lookup via ContentAccessService",
        )

        return results

    def _handle_text_search(
        self,
        query: str,
        dataset_type: "DatasetType",
        max_results: int,
        filters: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Handle text-based dataset searches.

        Args:
            query: Search query text
            dataset_type: Type of dataset to search for
            max_results: Maximum results
            filters: Additional filters

        Returns:
            str: Formatted search results
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        # Get providers that support DISCOVER_DATASETS capability
        providers = self.registry.get_providers_for_capability(
            ProviderCapability.DISCOVER_DATASETS
        )

        if not providers:
            return "No providers available for dataset discovery."

        # Filter providers that support this dataset type
        supporting_providers = [
            p for p in providers if dataset_type in p.supported_dataset_types
        ]

        if not supporting_providers:
            return f"No providers support {dataset_type.value} dataset type."

        # Use first provider (highest priority)
        provider = supporting_providers[0]
        logger.info(f"Using provider: {type(provider).__name__}")

        # Call provider's search method
        results = provider.search_publications(
            query=query, max_results=max_results, filters=filters
        )

        # Log to provenance
        self.data_manager.log_tool_usage(
            tool_name="discover_datasets",
            parameters={
                "query": query[:100],
                "dataset_type": dataset_type.value,
                "max_results": max_results,
                "provider": type(provider).__name__,
                "filters": filters,
            },
            description="Text-based dataset search via ContentAccessService",
        )

        return results

    def find_linked_datasets(
        self,
        identifier: str,
        dataset_types: Optional[list["DatasetType"]] = None,
        include_related: bool = True,
    ) -> str:
        """
        Find datasets linked to a publication.

        Routes to providers that support FIND_LINKED_DATASETS capability.
        Auto-detects provider based on identifier format (PMID, DOI, etc.).

        Args:
            identifier: Publication identifier (PMID, DOI, etc.)
            dataset_types: Optional list of dataset types to filter results
            include_related: Whether to include related datasets

        Returns:
            str: Formatted linked datasets results

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> results = service.find_linked_datasets("PMID:35042229")
            >>> results = service.find_linked_datasets("10.1038/s41586-025-09686-5")
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        logger.info(f"Finding linked datasets for: {identifier}")

        try:
            # Get providers for FIND_LINKED_DATASETS capability
            providers = self.registry.get_providers_for_capability(
                ProviderCapability.FIND_LINKED_DATASETS
            )

            if not providers:
                return "No providers available for linked dataset discovery."

            # Use first provider (highest priority, typically PubMed)
            provider = providers[0]
            logger.info(f"Using provider: {type(provider).__name__}")

            # Call provider's find datasets method
            results = provider.find_datasets_from_publication(
                identifier=identifier,
                dataset_types=dataset_types,
            )

            # Log to provenance
            self.data_manager.log_tool_usage(
                tool_name="find_linked_datasets",
                parameters={
                    "identifier": identifier,
                    "dataset_types": (
                        [dt.value for dt in dataset_types] if dataset_types else None
                    ),
                    "provider": type(provider).__name__,
                },
                description="Linked dataset discovery via ContentAccessService",
            )

            return results

        except Exception as e:
            logger.error(f"Linked dataset search error: {e}", exc_info=True)
            return f"Linked dataset search error: {str(e)}"

    def extract_metadata(
        self,
        identifier: str,
        source: Optional[str] = None,
    ) -> Union["PublicationMetadata", str]:
        """
        Extract publication metadata using capability-based routing.

        Routes to appropriate provider based on identifier format or
        explicitly specified source. Returns PublicationMetadata object.

        Args:
            identifier: Publication identifier (PMID, DOI, PMC ID, URL)
            source: Optional explicit source ("pubmed", "pmc", etc.)

        Returns:
            PublicationMetadata object or error string

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> metadata = service.extract_metadata("PMID:35042229")
            >>> metadata = service.extract_metadata("10.1038/s41586-025-09686-5")
        """
        from lobster.tools.providers.base_provider import (
            ProviderCapability,
        )

        logger.info(f"Extracting metadata for: {identifier}")

        try:
            # Get providers for EXTRACT_METADATA capability
            providers = self.registry.get_providers_for_capability(
                ProviderCapability.EXTRACT_METADATA
            )

            if not providers:
                return "No providers available for metadata extraction."

            # Filter by source if specified
            if source:
                source_filter = source.lower()
                providers = [
                    p
                    for p in providers
                    if type(p).__name__.lower().replace("provider", "") == source_filter
                ]

                if not providers:
                    return f"No provider found for source: {source}"

            # Use first provider (highest priority)
            provider = providers[0]
            logger.info(f"Using provider: {type(provider).__name__}")

            # Call provider's metadata extraction
            metadata = provider.extract_publication_metadata(identifier)

            # Log to provenance
            self.data_manager.log_tool_usage(
                tool_name="extract_metadata",
                parameters={
                    "identifier": identifier,
                    "provider": type(provider).__name__,
                },
                description="Metadata extraction via ContentAccessService",
            )

            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction error: {e}", exc_info=True)
            return f"Metadata extraction error: {str(e)}"

    def get_abstract(
        self,
        identifier: str,
        force_refresh: bool = False,
    ) -> dict[str, any]:
        """
        Get publication abstract (Tier 1: fast access, 200-500ms).

        Uses AbstractProvider for optimized NCBI API access with internal caching.
        This is the fast-path for quick paper discovery and overview.

        Args:
            identifier: Publication identifier (PMID, DOI, PMC ID)
            force_refresh: Force refresh from API, bypass cache

        Returns:
            Dict with abstract metadata:
                - title: Publication title
                - abstract: Abstract text
                - authors: List of authors
                - journal: Journal name
                - year: Publication year
                - pmid: PubMed ID
                - doi: DOI if available

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> abstract = service.get_abstract("PMID:35042229")
            >>> print(abstract['title'])
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        logger.info(f"Getting abstract for: {identifier}")

        try:
            # Get providers for GET_ABSTRACT capability
            providers = self.registry.get_providers_for_capability(
                ProviderCapability.GET_ABSTRACT
            )

            if not providers:
                return {"error": "No providers available for abstract retrieval."}

            # Use first provider (highest priority, typically AbstractProvider)
            provider = providers[0]
            logger.info(f"Using provider: {type(provider).__name__}")

            # Call provider's get_abstract method
            abstract = provider.get_abstract(identifier)

            # Log to provenance
            self.data_manager.log_tool_usage(
                tool_name="get_abstract",
                parameters={
                    "identifier": identifier,
                    "force_refresh": force_refresh,
                    "provider": type(provider).__name__,
                },
                description="Abstract retrieval (Tier 1) via ContentAccessService",
            )

            return abstract

        except Exception as e:
            logger.error(f"Abstract retrieval error: {e}", exc_info=True)
            return {"error": f"Abstract retrieval error: {str(e)}"}

    def get_full_content(
        self,
        source: str,
        prefer_webpage: bool = True,
        keywords: Optional[list[str]] = None,
        max_paragraphs: int = 100,
        max_retries: int = 2,
    ) -> dict[str, any]:
        """
        Get full publication content (Tier 2) with PMC-first fallback cascade.

        Implements intelligent content extraction with automatic fallback:
        1. Check DataManager cache (fast path)
        2. For PMID/DOI: Try PMC XML (500ms, 95% accuracy, 30-40% coverage)
        3. Fallback: Resolve to URL and try WebpageProvider
        4. Final fallback: PDF extraction via DoclingService

        Args:
            source: Publication identifier (PMID, DOI, PMC ID, URL)
            prefer_webpage: Try webpage before PDF for URLs (default: True)
            keywords: Optional section keywords for targeted extraction
            max_paragraphs: Maximum paragraphs to extract
            max_retries: Retry count for transient errors

        Returns:
            Dict with full content:
                - content: Full text markdown
                - methods_text: Methods section (if available)
                - tier_used: "full_pmc_xml", "full_webpage", or "full_pdf"
                - source_type: "pmc_xml", "webpage", or "pdf"
                - extraction_time: Seconds taken
                - metadata: Dict with tables, figures, software, etc.

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> # PMC available (fast path)
            >>> content = service.get_full_content("PMID:35042229")
            >>> # Webpage extraction
            >>> content = service.get_full_content("https://www.nature.com/articles/...")
            >>> # PDF fallback
            >>> content = service.get_full_content("https://biorxiv.org/.../file.pdf")

        Performance:
            - Cache hit: <100ms
            - PMC XML: 500ms (priority path)
            - Webpage: 2-5s (fallback)
            - PDF: 3-8s (last resort)
        """
        from lobster.tools.providers.base_provider import ProviderCapability
        from lobster.tools.providers.pmc_provider import PMCNotAvailableError
        from lobster.tools.providers.publication_resolver import PublicationResolver

        start_time = time.time()
        logger.info(f"Getting full content for: {source}")

        try:
            # 1. Check DataManager cache first
            cached = self.data_manager.get_cached_publication(source)
            if cached:
                logger.info(f"Cache hit for {source}")
                cached["extraction_time"] = time.time() - start_time
                cached["tier_used"] = "full_cached"
                return cached

            # 2. For PMID/DOI identifiers: Try PMC Full Text XML (priority)
            if self._is_identifier(source):
                logger.info(
                    f"Detected identifier: {source}, trying PMC full text first..."
                )

                try:
                    # Get PMC provider from capability routing
                    pmc_providers = self.registry.get_providers_for_capability(
                        ProviderCapability.GET_FULL_CONTENT
                    )
                    pmc_provider = next(
                        (p for p in pmc_providers if type(p).__name__ == "PMCProvider"),
                        None,
                    )

                    if pmc_provider:
                        # Extract full text from PMC XML
                        pmc_result = pmc_provider.extract_full_text(source)

                        # Format result
                        content_result = {
                            "content": pmc_result.full_text,
                            "methods_text": pmc_result.methods_section,
                            "methods_markdown": pmc_result.methods_section,
                            "results_text": pmc_result.results_section,
                            "discussion_text": pmc_result.discussion_section,
                            "tier_used": "full_pmc_xml",
                            "source_type": "pmc_xml",
                            "extraction_time": time.time() - start_time,
                            "metadata": {
                                "tables": len(pmc_result.tables),
                                "figures": len(pmc_result.figures),
                                "software": pmc_result.software_tools,
                                "github_repos": pmc_result.github_repos,
                                "sections": ["methods", "results", "discussion"],
                            },
                            "title": pmc_result.title,
                            "abstract": pmc_result.abstract,
                            "pmc_id": pmc_result.pmc_id,
                            "pmid": pmc_result.pmid,
                            "doi": pmc_result.doi,
                        }

                        # Cache result
                        self.data_manager.cache_publication_content(
                            identifier=source,
                            content=content_result,
                            format="json",
                        )

                        # Log to provenance
                        self.data_manager.log_tool_usage(
                            tool_name="get_full_content",
                            parameters={
                                "source": source,
                                "tier_used": "full_pmc_xml",
                                "extraction_time": content_result["extraction_time"],
                            },
                            description="PMC XML extraction (Tier 2) via ContentAccessService",
                        )

                        logger.info(
                            f"PMC XML extraction successful in {content_result['extraction_time']:.2f}s"
                        )
                        return content_result

                except PMCNotAvailableError:
                    logger.info(
                        f"PMC full text not available for {source}, falling back..."
                    )
                except Exception as e:
                    logger.warning(f"PMC extraction failed: {e}, falling back...")

            # 3. Resolve identifiers to URLs (fallback from PMC)
            url_to_fetch = source
            if self._is_identifier(source):
                logger.info(f"Resolving identifier to URL: {source}")
                resolver = PublicationResolver()
                resolution_result = resolver.resolve(source)

                if resolution_result.is_accessible():
                    url_to_fetch = (
                        resolution_result.pdf_url or resolution_result.html_url
                    )
                    logger.info(f"Resolved to: {url_to_fetch}")
                else:
                    return {
                        "error": f"Paper is paywalled: {source}",
                        "suggestions": "Try institutional access or preprint servers",
                    }

            # 4. Extract from URL using Webpage or PDF providers
            webpage_providers = [
                p
                for p in self.registry.get_providers_for_capability(
                    ProviderCapability.GET_FULL_CONTENT
                )
                if type(p).__name__ == "WebpageProvider"
            ]

            if webpage_providers:
                webpage_provider = webpage_providers[0]
                logger.info("Using WebpageProvider for URL extraction")

                # Extract content (handles both webpage and PDF internally)
                content_result = webpage_provider.extract_content(
                    url_to_fetch,
                    keywords=keywords,
                    max_paragraphs=max_paragraphs,
                )

                content_result["extraction_time"] = time.time() - start_time
                content_result["tier_used"] = (
                    f"full_{content_result.get('source_type', 'webpage')}"
                )

                # Cache result
                self.data_manager.cache_publication_content(
                    identifier=source,
                    content=content_result,
                    format="json",
                )

                # Log to provenance
                self.data_manager.log_tool_usage(
                    tool_name="get_full_content",
                    parameters={
                        "source": source,
                        "url": url_to_fetch,
                        "tier_used": content_result["tier_used"],
                        "extraction_time": content_result["extraction_time"],
                    },
                    description="Webpage/PDF extraction (Tier 2) via ContentAccessService",
                )

                logger.info(
                    f"Content extraction successful in {content_result['extraction_time']:.2f}s"
                )
                return content_result

            return {"error": "No providers available for content extraction"}

        except Exception as e:
            logger.error(f"Full content extraction error: {e}", exc_info=True)
            return {"error": f"Content extraction error: {str(e)}"}

    def extract_methods(
        self,
        content_result: dict[str, any],
        llm: Optional[any] = None,
        include_tables: bool = True,
    ) -> dict[str, any]:
        """
        Extract structured methods information from full content.

        Extracts software tools, parameters, and statistical methods from
        the methods section. Future: Add LLM-based structured extraction.

        Args:
            content_result: Result dict from get_full_content()
            llm: Optional LLM for structured extraction (future feature)
            include_tables: Whether to include methods tables

        Returns:
            Dict with extracted methods:
                - methods_text: Raw methods section text
                - software_used: List of detected software tools
                - parameters: Extracted parameters (future)
                - statistical_methods: Detected statistical tests (future)
                - github_repos: GitHub repository URLs
                - tables: Methods-related tables (if include_tables=True)

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> content = service.get_full_content("PMID:35042229")
            >>> methods = service.extract_methods(content)
            >>> print(methods['software_used'])
        """
        logger.info("Extracting methods information")

        try:
            # Extract methods text
            methods_text = content_result.get("methods_text", "")
            metadata = content_result.get("metadata", {})

            # Extract software tools
            software_used = metadata.get("software", [])
            github_repos = metadata.get("github_repos", [])

            # Build result
            methods_result = {
                "methods_text": methods_text,
                "software_used": software_used,
                "github_repos": github_repos,
                "parameters": {},  # Future: LLM extraction
                "statistical_methods": [],  # Future: LLM extraction
            }

            # Include tables if requested
            if include_tables:
                methods_result["tables"] = metadata.get("tables", [])

            # Log to provenance
            self.data_manager.log_tool_usage(
                tool_name="extract_methods",
                parameters={
                    "software_count": len(software_used),
                    "github_repos_count": len(github_repos),
                    "include_tables": include_tables,
                },
                description="Methods extraction via ContentAccessService",
            )

            return methods_result

        except Exception as e:
            logger.error(f"Methods extraction error: {e}", exc_info=True)
            return {"error": f"Methods extraction error: {str(e)}"}

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def validate_metadata(
        self,
        dataset_id: str,
        required_fields: Optional[List[str]] = None,
        required_values: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.8,
    ) -> str:
        """
        Validate dataset metadata completeness and quality.

        Routes to MetadataValidationService for GEO datasets. Returns
        formatted validation report with completeness scores, missing fields,
        and recommendations.

        Args:
            dataset_id: Dataset identifier (e.g., 'GSE123456')
            required_fields: List of required field names (optional)
            required_values: Optional dict of field -> required values
            threshold: Minimum fraction of samples that must have each field (default: 0.8)

        Returns:
            str: Formatted validation report with recommendations

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> report = service.validate_metadata("GSE123456")
            >>> report = service.validate_metadata(
            ...     "GSE123456",
            ...     required_fields=["smoking_status", "treatment_response"]
            ... )
        """
        from lobster.tools.metadata_validation_service import MetadataValidationService

        logger.info(f"Validating metadata for: {dataset_id}")

        try:
            # Check if we have cached metadata
            metadata = self.data_manager._get_geo_metadata(dataset_id)

            if not metadata:
                return (
                    f"Dataset metadata not found for {dataset_id}. "
                    f"Please fetch metadata first using discover_datasets() or extract_metadata()."
                )

            # Extract metadata dict (handle both dict and stored format)
            if isinstance(metadata, dict):
                metadata_dict = metadata.get("metadata", metadata)
            else:
                metadata_dict = metadata

            # Initialize validation service
            validation_service = MetadataValidationService(
                data_manager=self.data_manager
            )

            # Perform validation
            if required_fields:
                validation_config = validation_service.validate_dataset_metadata(
                    metadata=metadata_dict,
                    geo_id=dataset_id,
                    required_fields=required_fields,
                    required_values=required_values,
                    threshold=threshold,
                )
            else:
                # No specific fields required - check general completeness
                # Look for common important fields
                common_fields = [
                    "tissue",
                    "cell_type",
                    "disease",
                    "treatment",
                    "age",
                    "sex",
                ]
                validation_config = validation_service.validate_dataset_metadata(
                    metadata=metadata_dict,
                    geo_id=dataset_id,
                    required_fields=common_fields,
                    threshold=threshold,
                )

            if not validation_config:
                return (
                    f"Metadata validation failed for {dataset_id}. "
                    f"Please check dataset metadata or try again."
                )

            # Format validation report
            report = validation_service.format_validation_report(
                validation_config, dataset_id
            )

            # Log to provenance
            self.data_manager.log_tool_usage(
                tool_name="validate_metadata",
                parameters={
                    "dataset_id": dataset_id,
                    "required_fields": required_fields or common_fields,
                    "threshold": threshold,
                    "recommendation": validation_config.recommendation,
                    "confidence_score": validation_config.confidence_score,
                },
                description="Metadata validation via ContentAccessService",
            )

            return report

        except Exception as e:
            logger.error(f"Metadata validation error: {e}", exc_info=True)
            return f"Metadata validation error for {dataset_id}: {str(e)}"

    def query_capabilities(self) -> str:
        """
        Query available capabilities and supported databases.

        Returns a formatted capability matrix showing which providers support
        which operations (search, discovery, metadata extraction, etc.) and
        what databases are available.

        Returns:
            str: Formatted capability matrix with provider details

        Examples:
            >>> service = ContentAccessService(data_manager)
            >>> capabilities = service.query_capabilities()
            >>> print(capabilities)
            Available Capabilities:
            - search_literature: PubMedProvider
            - discover_datasets: GEOProvider
            - get_full_content: PMCProvider, WebpageProvider
            ...
        """
        from lobster.tools.providers.base_provider import ProviderCapability

        logger.info("Querying system capabilities")

        try:
            # Get capability matrix from registry
            capability_dict = self.registry.get_capabilities_by_provider()

            # Get all providers
            all_providers = self.registry.get_all_providers()

            # Build formatted response
            lines = []
            lines.append("=" * 70)
            lines.append("LOBSTER CONTENT ACCESS SERVICE - CAPABILITY MATRIX")
            lines.append("=" * 70)
            lines.append("")

            # Section 1: Available Operations
            lines.append("📋 AVAILABLE OPERATIONS:")
            lines.append("")

            # Group capabilities by category
            capability_categories = {
                "Discovery & Search": [
                    ProviderCapability.SEARCH_LITERATURE,
                    ProviderCapability.DISCOVER_DATASETS,
                    ProviderCapability.FIND_LINKED_DATASETS,
                ],
                "Metadata & Validation": [
                    ProviderCapability.EXTRACT_METADATA,
                    ProviderCapability.VALIDATE_METADATA,
                ],
                "Content Retrieval": [
                    ProviderCapability.GET_ABSTRACT,
                    ProviderCapability.GET_FULL_CONTENT,
                    ProviderCapability.EXTRACT_METHODS,
                    ProviderCapability.EXTRACT_PDF,
                ],
                "Advanced Features": [
                    ProviderCapability.INTEGRATE_MULTI_OMICS,
                ],
            }

            for category, capabilities in capability_categories.items():
                lines.append(f"  {category}:")
                for capability in capabilities:
                    # Get providers for this capability
                    providers = self.registry.get_providers_for_capability(capability)
                    if providers:
                        provider_names = [type(p).__name__ for p in providers]
                        status = "✅"
                        lines.append(
                            f"    {status} {capability:30s} → {', '.join(provider_names)}"
                        )
                    else:
                        status = "❌"
                        lines.append(f"    {status} {capability:30s} → Not available")
                lines.append("")

            # Section 2: Registered Providers
            lines.append("🔧 REGISTERED PROVIDERS:")
            lines.append("")
            for provider in all_providers:
                provider_name = type(provider).__name__
                priority = provider.priority
                capabilities = list(provider.capabilities)
                cap_names = [str(c) for c in capabilities]

                lines.append(f"  • {provider_name} (Priority: {priority})")
                lines.append(f"    Capabilities: {', '.join(cap_names[:5])}")
                if len(cap_names) > 5:
                    lines.append(f"                  {', '.join(cap_names[5:])}")
                lines.append("")

            # Section 3: Supported Dataset Types
            lines.append("💾 SUPPORTED DATASET TYPES:")
            lines.append("")
            dataset_types = self.registry.get_supported_dataset_types()
            if dataset_types:
                for dtype in dataset_types:
                    provider = self.registry.get_provider_for_dataset_type(dtype)
                    if provider:
                        lines.append(
                            f"  ✅ {dtype.value:20s} → {type(provider).__name__}"
                        )
            else:
                lines.append("  ❌ No dataset types registered")
            lines.append("")

            # Section 4: Performance Tiers
            lines.append("⚡ PERFORMANCE TIERS:")
            lines.append("")
            lines.append("  Tier 1 (Fast): <500ms")
            lines.append("    - get_abstract: AbstractProvider, PubMedProvider")
            lines.append("    - search_literature: PubMedProvider")
            lines.append("    - discover_datasets: GEOProvider")
            lines.append("")
            lines.append("  Tier 2 (Moderate): 500ms-2s")
            lines.append("    - get_full_content (PMC): PMCProvider")
            lines.append("    - extract_metadata: PubMedProvider, GEOProvider")
            lines.append("")
            lines.append("  Tier 3 (Slow): 2-8s")
            lines.append("    - get_full_content (Webpage): WebpageProvider")
            lines.append(
                "    - get_full_content (PDF): WebpageProvider + DoclingService"
            )
            lines.append("")

            # Section 5: Cascade Logic
            lines.append("🔄 CASCADE LOGIC:")
            lines.append("")
            lines.append("  Full Content Retrieval:")
            lines.append("    1. Check DataManager cache (fastest)")
            lines.append("    2. Try PMC XML (Priority 10, 30-40% coverage)")
            lines.append("    3. Fallback: Webpage HTML (Priority 50)")
            lines.append("    4. Final fallback: PDF via Docling (Priority 100)")
            lines.append("")

            lines.append("=" * 70)

            result = "\n".join(lines)

            # Log to provenance
            self.data_manager.log_tool_usage(
                tool_name="query_capabilities",
                parameters={
                    "provider_count": len(all_providers),
                    "capability_count": len(capability_dict),
                    "dataset_types": (
                        [dt.value for dt in dataset_types] if dataset_types else []
                    ),
                },
                description="Capability query via ContentAccessService",
            )

            return result

        except Exception as e:
            logger.error(f"Capability query error: {e}", exc_info=True)
            return f"Capability query error: {str(e)}"

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _is_identifier(self, source: str) -> bool:
        """
        Check if source is an identifier (PMID, DOI, PMC) vs URL.

        Args:
            source: Source string to check

        Returns:
            bool: True if identifier, False if URL
        """
        # Check for common identifier patterns
        source_lower = source.lower()
        is_id = (
            source_lower.startswith("pmid:")
            or source_lower.startswith("doi:")
            or source_lower.startswith("pmc")
            or source.startswith("10.")  # DOI pattern
            or source.isdigit()  # Plain PMID
        )
        is_url = source_lower.startswith("http://") or source_lower.startswith(
            "https://"
        )

        return is_id and not is_url
