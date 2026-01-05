"""
Research Agent for literature discovery and dataset identification.

This agent specializes in searching scientific literature, discovering datasets,
and providing comprehensive research context using the modular publication service
architecture with DataManagerV2 integration.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.state import ResearchAgentState
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.download_queue import (
    DownloadQueueEntry,
    DownloadStatus,
    StrategyConfig,
    ValidationStatus,
)
# Lazy-loaded services (moved to function scope for 1.2s import speedup)
# - ContentAccessService: 500-800ms
# - DataExpertAssistant: 200-400ms
# - MetadataValidationService: 200-400ms
from lobster.services.metadata.metadata_validation_service import (
    MetadataValidationConfig,
    ValidationSeverity,
)

# Premium feature - graceful fallback if unavailable
from lobster.core.component_registry import component_registry

PublicationProcessingService = component_registry.get_service('publication_processing')
HAS_PUBLICATION_PROCESSING = PublicationProcessingService is not None

# Phase 1: New providers for two-tier access
from lobster.tools.providers.abstract_provider import AbstractProvider
from lobster.tools.providers.base_provider import DatasetType
from lobster.tools.providers.webpage_provider import WebpageProvider
from lobster.tools.workspace_tool import (
    create_get_content_from_workspace_tool,
    create_write_to_workspace_tool,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# GEO Metadata Verbosity Control
# ============================================================
# Field categorization for controlling metadata output verbosity.
# Used by get_dataset_metadata tool to prevent context overflow.

ESSENTIAL_FIELDS = {
    "database",
    "geo_accession",
    "title",
    "status",
    "pubmed_id",
    "summary",
}

STANDARD_FIELDS = {
    "overall_design",
    "type",
    "submission_date",
    "last_update_date",
    "web_link",
    "contributor",
    "contact_name",
    "contact_email",
    "contact_institute",
    "contact_country",
    "platform_id",
    "organism",
    "n_samples",
    "sample_count",
}

VERBOSE_FIELDS = {
    "sample_id",
    "contact_phone",
    "contact_department",
    "contact_address",
    "contact_city",
    "contact_zip/postal_code",
    "supplementary_file",
    "platform_taxid",
    "sample_taxid",
    "relation",
    "samples",
    "platforms",
}


def research_agent(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "research_agent",
    delegation_tools: list = None,
    subscription_tier: str = "free",
    workspace_path: Optional[Path] = None,
):
    """Create research agent using DataManagerV2 and modular publication service.

    Args:
        data_manager: DataManagerV2 instance for data operations
        callback_handler: Optional callback for streaming responses
        agent_name: Name for this agent instance
        delegation_tools: List of tools for delegating to sub-agents
        subscription_tier: Subscription tier for feature gating (free/premium/enterprise).
            In FREE tier, handoff to metadata_assistant is restricted.
        workspace_path: Optional workspace path for config resolution
    """
    # Import tier restrictions
    from lobster.config.subscription_tiers import get_restricted_handoffs

    # Get restricted handoffs for this agent at current tier
    restricted_handoffs = get_restricted_handoffs(subscription_tier, agent_name)

    # Filter delegation tools based on tier restrictions
    if delegation_tools and restricted_handoffs:
        filtered_delegation_tools = []
        for delegation_tool in delegation_tools:
            # Check if tool name indicates a restricted handoff
            tool_name = getattr(delegation_tool, "__name__", "") or getattr(
                delegation_tool, "name", ""
            )
            is_restricted = any(
                restricted in tool_name for restricted in restricted_handoffs
            )
            if is_restricted:
                logger.info(
                    f"Tier '{subscription_tier}' restricts {tool_name} handoff - "
                    f"upgrade to premium for full access"
                )
            else:
                filtered_delegation_tools.append(delegation_tool)
        delegation_tools = filtered_delegation_tools

    settings = get_settings()
    model_params = settings.get_agent_llm_params("research_agent")
    llm = create_llm("research_agent", model_params, workspace_path=workspace_path)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # ============================================================
    # Lazy Service Loaders (1.2s import speedup)
    # ============================================================
    # These services are only imported when first used by tools
    _content_service = None
    _data_expert = None
    _metadata_validator = None

    def get_content_service():
        """Lazy loader for ContentAccessService (saves 500-800ms on import)"""
        nonlocal _content_service
        if _content_service is None:
            from lobster.services.data_access.content_access_service import ContentAccessService
            _content_service = ContentAccessService(data_manager=data_manager)
            logger.debug("Lazy-loaded ContentAccessService")
        return _content_service

    def get_data_expert():
        """Lazy loader for DataExpertAssistant (saves 200-400ms on import)"""
        nonlocal _data_expert
        if _data_expert is None:
            from lobster.agents.data_expert_assistant import DataExpertAssistant
            _data_expert = DataExpertAssistant()
            logger.debug("Lazy-loaded DataExpertAssistant")
        return _data_expert

    def get_metadata_validator():
        """Lazy loader for MetadataValidationService (saves 200-400ms on import)"""
        nonlocal _metadata_validator
        if _metadata_validator is None:
            from lobster.services.metadata.metadata_validation_service import MetadataValidationService
            _metadata_validator = MetadataValidationService(data_manager=data_manager)
            logger.debug("Lazy-loaded MetadataValidationService")
        return _metadata_validator

    # Premium feature - only instantiate if available
    publication_processing_service = None
    if HAS_PUBLICATION_PROCESSING:
        publication_processing_service = PublicationProcessingService(
            data_manager=data_manager
        )

    # Define tools
    @tool
    def search_literature(
        query: str = "",
        max_results: int = 5,
        sources: str = "pubmed",
        filters: Union[str, Dict[str, Any], None] = None,
        related_to: str = None,
    ) -> str:
        """
        Search for scientific literature across multiple sources or find related papers.

        Args:
            query: Search query string (optional if using related_to)
            max_results: Number of results to retrieve (default: 5, range: 1-20)
            sources: Publication sources to search (default: "pubmed", options: "pubmed,biorxiv,medrxiv")
            filters: Optional search filters as dict or JSON string. Available filters:
                     - date_range: {"start": "YYYY", "end": "YYYY"}
                     Can be passed as:
                     - Python dict: {"date_range": {"start": "2020", "end": "2024"}}
                     - JSON string: '{"date_range": {"start": "2020", "end": "2024"}}'
            related_to: Find papers related to this identifier (PMID or DOI). When provided, discovers
                        papers citing or cited by the given publication. Merges functionality from
                        the removed discover_related_studies tool.

        Returns:
            Formatted list of publications with titles, authors, abstracts, and identifiers

        Examples:
            # Standard keyword search
            search_literature("BRCA1 breast cancer", max_results=10)

            # Find related papers (merged discover_related_studies functionality)
            search_literature(related_to="PMID:12345678", max_results=10)

            # Search with date filters
            search_literature("lung cancer", filters='{"date_range": {"start": "2020", "end": "2024"}}')
        """
        try:
            # Related paper discovery mode (merged from discover_related_studies)
            if related_to:
                logger.debug(f"Finding papers related to: {related_to}")
                results = get_content_service().find_related_publications(
                    identifier=related_to, max_results=max_results
                )
                logger.debug(f"Related paper discovery completed for: {related_to}")
                return results

            # Standard literature search mode
            if not query:
                return "Error: Either 'query' or 'related_to' must be provided for literature search"

            # Parse sources (keep as strings - service expects list[str])
            source_list = []
            if sources:
                for source in sources.split(","):
                    source = source.strip().lower()
                    # Validate source is supported
                    if source in ["pubmed", "biorxiv", "medrxiv"]:
                        source_list.append(source)
                    else:
                        logger.warning(f"Unsupported source '{source}' ignored")

            # Parse filters with type coercion
            filter_dict = None
            if filters:
                if isinstance(filters, dict):
                    # Already a dict, use directly
                    filter_dict = filters
                elif isinstance(filters, str):
                    # JSON string, parse it
                    import json

                    try:
                        filter_dict = json.loads(filters)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid filters JSON: {filters}, error: {e}")
                        return f"Error: Invalid filters JSON format: {str(e)}"
                else:
                    logger.warning(f"Invalid filters type: {type(filters)}")
                    return f"Error: filters must be dict or JSON string, got {type(filters)}"

            results, stats, ir = get_content_service().search_literature(
                query=query,
                max_results=max_results,
                sources=source_list if source_list else None,
                filters=filter_dict,
            )

            # Log to provenance with IR
            data_manager.log_tool_usage(
                tool_name="search_literature",
                parameters={
                    "query": query,
                    "max_results": max_results,
                    "sources": sources,
                    "filters": filters,
                },
                description=f"Literature search: {query[:50]}",
                ir=ir,  # Pass IR for provenance tracking
            )

            logger.debug(
                f"Literature search completed for: {query[:50]}... (max_results={max_results})"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching literature: {e}")
            return f"Error searching literature: {str(e)}"

    @tool
    def find_related_entries(
        identifier: str,
        dataset_types: str = None,
        include_related: bool = True,
    ) -> str:
        """
        Find connected publications, datasets, samples, and metadata for a given identifier.

        This tool discovers related research content across databases, supporting multi-omics
        integration workflows. Use this to find datasets from publications, or to explore
        the full ecosystem of related research artifacts.

        Args:
            identifier: Publication identifier (DOI or PMID) or dataset identifier (GSE, SRA)
            dataset_types: Filter by dataset types, comma-separated (e.g., "geo,sra,arrayexpress")
            include_related: Whether to include related/linked datasets (default: True)

        Returns:
            Formatted report of connected datasets, publications, and metadata

        Examples:
            # Find datasets from publication, filtered by repository type
            find_related_entries("PMID:12345678", dataset_types="geo")

            # Find all related content (datasets + publications + samples)
            find_related_entries("GSE12345")

            # Find related entries without including indirectly related datasets
            find_related_entries("GSE12345", include_related=False)
        """
        try:
            # Parse dataset types
            type_list = []
            if dataset_types:
                type_mapping = {
                    "geo": DatasetType.GEO,
                    "sra": DatasetType.SRA,
                    "arrayexpress": DatasetType.ARRAYEXPRESS,
                    "ena": DatasetType.ENA,
                    "bioproject": DatasetType.BIOPROJECT,
                    "biosample": DatasetType.BIOSAMPLE,
                    "dbgap": DatasetType.DBGAP,
                }

                for dtype in dataset_types.split(","):
                    dtype = dtype.strip().lower()
                    if dtype in type_mapping:
                        type_list.append(type_mapping[dtype])

            results = get_content_service().find_linked_datasets(
                identifier=identifier,
                dataset_types=type_list if type_list else None,
                include_related=include_related,
            )

            logger.debug(f"Dataset discovery completed for: {identifier}")
            return results

        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            return f"Error finding datasets from publication: {str(e)}"

    @tool
    def fast_dataset_search(
        query: str,
        data_type: str = "geo",
        max_results: int = 5,
        filters: Union[str, Dict[str, Any], None] = None,
    ) -> str:
        """
        Search omics databases directly for datasets matching your query (GEO, SRA, PRIDE, MassIVE, etc.).

        Fast, keyword-based search across multiple repositories. Use this when you know
        what you're looking for (e.g., disease + technology) and want quick results.
        For publication-linked datasets, use find_related_entries() instead.

        Args:
            query: Search query for datasets (keywords, disease names, technology)
            data_type: Database to search (default: "geo", options: "geo,sra,bioproject,biosample,dbgap,pride,massive")
            max_results: Maximum results to return (default: 5)
            filters: Optional filters as dict or JSON string. Available filters vary by database:
                     Can be passed as:
                     - Python dict: {"organism": "Homo sapiens", "strategy": "AMPLICON"}
                     - JSON string: '{"organism": "Homo sapiens", "strategy": "AMPLICON"}'

                     **SRA filters** (metagenomics, RNA-seq, etc.):
                     - organism: str (e.g., "Homo sapiens", "Mus musculus") - use scientific names
                     - strategy: str (e.g., "AMPLICON" for 16S/ITS, "RNA-Seq", "WGS", "ChIP-Seq")
                     - source: str (e.g., "METAGENOMIC", "TRANSCRIPTOMIC", "GENOMIC")
                     - layout: str (e.g., "PAIRED", "SINGLE")
                     - platform: str (e.g., "ILLUMINA", "PACBIO", "OXFORD_NANOPORE")

                     **GEO filters** (microarray, RNA-seq):
                     - organism: str
                     - year: str (e.g., "2023")

        Returns:
            Formatted list of matching datasets with accessions and metadata

        Examples:
            # Search GEO for single-cell lung cancer
            fast_dataset_search("lung cancer single-cell", data_type="geo")

            # Search SRA for 16S microbiome studies (AMPLICON strategy)
            fast_dataset_search("IBS microbiome", data_type="sra",
                               filters='{{"organism": "Homo sapiens", "strategy": "AMPLICON"}}')

            # Search SRA for metagenomic shotgun sequencing
            fast_dataset_search("gut microbiome", data_type="sra",
                               filters='{{"source": "METAGENOMIC", "strategy": "WGS"}}')

            # Search SRA for RNA-seq with organism filter
            fast_dataset_search("CRISPR screen", data_type="sra",
                               filters='{{"organism": "Homo sapiens", "strategy": "RNA-Seq"}}')

            # Search SRA for Oxford Nanopore long-read sequencing
            fast_dataset_search("cancer transcriptome", data_type="sra",
                               filters='{{"platform": "OXFORD_NANOPORE", "strategy": "RNA-Seq"}}')
        """
        try:
            # Map string to DatasetType
            type_mapping = {
                "geo": DatasetType.GEO,
                "sra": DatasetType.SRA,
                "bioproject": DatasetType.BIOPROJECT,
                "biosample": DatasetType.BIOSAMPLE,
                "dbgap": DatasetType.DBGAP,
                "arrayexpress": DatasetType.ARRAYEXPRESS,
                "ena": DatasetType.ENA,
                "pride": DatasetType.PRIDE,
                "massive": DatasetType.MASSIVE,
            }

            dataset_type = type_mapping.get(data_type.lower(), DatasetType.GEO)

            # Parse filters with type coercion
            filter_dict = None
            if filters:
                if isinstance(filters, dict):
                    # Already a dict, use directly
                    filter_dict = filters
                elif isinstance(filters, str):
                    # JSON string, parse it
                    import json

                    try:
                        filter_dict = json.loads(filters)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid filters JSON: {filters}, error: {e}")
                        return f"Error: Invalid filters JSON format: {str(e)}"
                else:
                    logger.warning(f"Invalid filters type: {type(filters)}")
                    return f"Error: filters must be dict or JSON string, got {type(filters)}"

            results, stats, ir = get_content_service().discover_datasets(
                query=query,
                dataset_type=dataset_type,
                max_results=max_results,
                filters=filter_dict,
            )

            # Log to provenance with IR
            data_manager.log_tool_usage(
                tool_name="fast_dataset_search",
                parameters={
                    "query": query,
                    "data_type": data_type,
                    "max_results": max_results,
                    "filters": filters,
                },
                description=f"Dataset search: {query[:50]}",
                ir=ir,  # Pass IR for provenance tracking
            )

            logger.info(
                f"Direct dataset search completed: {query[:50]}... ({data_type})"
            )
            return results

        except Exception as e:
            logger.error(f"Error searching datasets directly: {e}")
            return f"Error searching datasets directly: {str(e)}"

    @tool
    def get_dataset_metadata(
        identifier: str,
        source: str = "auto",
        database: str = None,
        level: str = "standard",
    ) -> str:
        """
        Get comprehensive metadata for datasets or publications.

        Retrieves structured metadata including title, authors, publication info, sample counts,
        platform details, and experimental design. Supports both publications (PMID/DOI) and
        datasets (GSE/SRA/PRIDE accessions). Automatically detects identifier type or can be
        explicitly specified via the database parameter.

        Args:
            identifier: Publication identifier (DOI or PMID) or dataset accession (GSE, SRA, PXD, MSV)
            source: Source hint for publications (default: "auto", options: "auto,pubmed,biorxiv,medrxiv")
            database: Database hint for explicit routing (options: "geo", "sra", "pride", "massive", "pubmed").
                     If None, auto-detects from identifier format. Use this to force interpretation
                     when identifier format is ambiguous.
            level: Metadata verbosity level (default: "standard", options: "brief", "standard", "full").
                   Controls output length to prevent context overflow:
                   - "brief": Essential fields only (accession, title, status, pubmed_id, summary)
                   - "standard": Brief + standard fields with sample/platform previews (recommended)
                   - "full": All fields including complete nested structures (verbose). NEVER USE FULL EXCEPT IF USER REQUESTS

        Returns:
            Formatted metadata report with bibliographic and experimental details

        Examples:
            # Get publication metadata (auto-detect, standard verbosity)
            get_dataset_metadata("PMID:12345678")

            # Get dataset metadata with brief output (essential fields only)
            get_dataset_metadata("GSE12345", level="brief")

            # Get full metadata with all nested structures
            get_dataset_metadata("GSE12345", level="full")

            # Force GEO interpretation with standard verbosity
            get_dataset_metadata("12345", database="geo", level="standard")

            # Specify publication source for faster lookup
            get_dataset_metadata("10.1038/s41586-021-12345-6", source="pubmed")

            # Get SRA dataset metadata with brief output
            get_dataset_metadata("SRR12345678", database="sra", level="brief")
        """
        try:
            # Auto-detect database type from identifier if not specified
            if database is None:
                identifier_upper = identifier.upper()
                if identifier_upper.startswith("GSE") or identifier_upper.startswith(
                    "GDS"
                ):
                    database = "geo"
                elif identifier_upper.startswith("SRR") or identifier_upper.startswith(
                    "SRP"
                ):
                    database = "sra"
                elif identifier_upper.startswith("PRD") or identifier_upper.startswith(
                    "PXD"
                ):
                    database = "pride"
                elif identifier_upper.startswith("MSV"):
                    database = "massive"
                elif identifier_upper.startswith("PMID:") or identifier.startswith(
                    "10."
                ):
                    database = "pubmed"
                else:
                    # Default to publication metadata extraction
                    database = "pubmed"
                    logger.info(
                        f"Auto-detected database type as publication for: {identifier}"
                    )

            # Route to appropriate metadata extraction based on database
            if database.lower() in ["geo", "sra", "pride", "massive"]:
                # Dataset metadata extraction
                logger.info(
                    f"Extracting {database.upper()} dataset metadata for: {identifier}"
                )

                # Use GEOService for GEO datasets (most common case)
                if database.lower() == "geo":
                    from lobster.services.data_access.geo_service import GEOService

                    console = getattr(data_manager, "console", None)
                    geo_service = GEOService(data_manager, console=console)

                    # Fetch metadata only (no data download)
                    try:
                        metadata_info, _ = geo_service.fetch_metadata_only(identifier)
                        formatted = f"## Dataset Metadata for {identifier}\n\n"
                        formatted += "**Database**: GEO\n"
                        formatted += f"**Accession**: {identifier}\n"

                        # Add available metadata fields with verbosity control
                        if isinstance(metadata_info, dict):
                            # Determine which fields to show based on level
                            if level == "brief":
                                allowed_fields = ESSENTIAL_FIELDS
                            elif level == "standard":
                                allowed_fields = ESSENTIAL_FIELDS | STANDARD_FIELDS
                            else:  # "full"
                                allowed_fields = None  # Show everything

                            for key, value in metadata_info.items():
                                if not value:
                                    continue

                                # Skip field if not in allowed set (unless full mode)
                                if allowed_fields and key not in allowed_fields:
                                    continue

                                # Special formatting for nested structures in standard mode
                                if (
                                    level == "standard"
                                    and key == "samples"
                                    and isinstance(value, dict)
                                ):
                                    formatted += f"**Sample Count**: {len(value)}\n"
                                    formatted += "**Sample Preview** (first 3):\n"
                                    for i, (gsm_id, sample_data) in enumerate(
                                        list(value.items())[:3]
                                    ):
                                        sample_title = (
                                            sample_data.get("title", "No title")
                                            if isinstance(sample_data, dict)
                                            else str(sample_data)
                                        )
                                        formatted += f"  - {gsm_id}: {sample_title}\n"
                                elif (
                                    level == "standard"
                                    and key == "platforms"
                                    and isinstance(value, dict)
                                ):
                                    formatted += f"**Platform Count**: {len(value)}\n"
                                    formatted += "**Platforms**:\n"
                                    for gpl_id, platform_data in value.items():
                                        platform_title = (
                                            platform_data.get("title", "No title")
                                            if isinstance(platform_data, dict)
                                            else str(platform_data)
                                        )
                                        formatted += f"  - {gpl_id}: {platform_title}\n"
                                else:
                                    # Standard field display
                                    formatted += f"**{key.replace('_', ' ').title()}**: {value}\n"

                        logger.info(
                            f"GEO metadata extraction completed for: {identifier}"
                        )
                        return formatted
                    except Exception as e:
                        logger.error(f"Error fetching GEO metadata: {e}")
                        return f"Error fetching GEO metadata for {identifier}: {str(e)}"
                elif database.lower() == "pride":
                    # PRIDE dataset metadata extraction
                    from lobster.tools.providers.pride_provider import PRIDEProvider

                    pride_provider = PRIDEProvider(data_manager=data_manager)

                    try:
                        project_metadata = pride_provider.get_project_metadata(
                            identifier
                        )

                        formatted = f"## PRIDE Dataset Metadata for {identifier}\n\n"
                        formatted += "**Database**: PRIDE Archive\n"
                        formatted += f"**Accession**: {identifier}\n"
                        formatted += (
                            f"**Title**: {project_metadata.get('title', 'N/A')}\n"
                        )

                        # Sample count
                        if "sampleProcessingProtocol" in project_metadata:
                            formatted += f"**Sample Protocol**: Available\n"

                        # Organisms
                        organisms = project_metadata.get("organisms", [])
                        if organisms:
                            organism_names = [o.get("name", "") for o in organisms]
                            formatted += f"**Organisms**: {', '.join(organism_names)}\n"

                        # Instruments
                        instruments = project_metadata.get("instruments", [])
                        if instruments:
                            instrument_names = [i.get("name", "") for i in instruments]
                            formatted += (
                                f"**Instruments**: {', '.join(instrument_names[:3])}\n"
                            )

                        # Publication date
                        if "publicationDate" in project_metadata:
                            formatted += f"**Published**: {project_metadata['publicationDate']}\n"

                        # Description (brief in standard mode)
                        description = project_metadata.get("projectDescription", "")
                        if description and level in ["standard", "full"]:
                            desc_preview = (
                                description[:500]
                                if level == "standard"
                                else description
                            )
                            formatted += f"\n**Description**:\n{desc_preview}{'...' if len(description) > 500 and level == 'standard' else ''}\n"

                        logger.info(
                            f"PRIDE metadata extraction completed for: {identifier}"
                        )
                        return formatted

                    except Exception as e:
                        logger.error(f"Error fetching PRIDE metadata: {e}")
                        return (
                            f"Error fetching PRIDE metadata for {identifier}: {str(e)}"
                        )

                elif database.lower() == "massive":
                    # MassIVE dataset metadata extraction
                    from lobster.tools.providers.massive_provider import MassIVEProvider

                    massive_provider = MassIVEProvider(data_manager=data_manager)

                    try:
                        dataset_metadata = massive_provider.get_dataset_metadata(
                            identifier
                        )

                        formatted = f"## MassIVE Dataset Metadata for {identifier}\n\n"
                        formatted += "**Database**: MassIVE (UCSD)\n"
                        formatted += f"**Accession**: {identifier}\n"
                        formatted += (
                            f"**Title**: {dataset_metadata.get('title', 'N/A')}\n"
                        )

                        # Species
                        species = dataset_metadata.get("species", [])
                        if species:
                            species_names = [
                                s.get("name", "")
                                for s in species
                                if isinstance(s, dict)
                            ]
                            formatted += f"**Species**: {', '.join(species_names)}\n"

                        # Data type
                        contacts = dataset_metadata.get("contacts", [])
                        if contacts and isinstance(contacts[0], dict):
                            contact_props = contacts[0].get("contactProperties", [])
                            for prop in contact_props:
                                if prop.get("name") == "DatasetType":
                                    formatted += (
                                        f"**Data Type**: {prop.get('value', 'N/A')}\n"
                                    )
                                    break

                        # Description
                        description = dataset_metadata.get("description", "")
                        if description and level in ["standard", "full"]:
                            desc_preview = (
                                description[:500]
                                if level == "standard"
                                else description
                            )
                            formatted += f"\n**Description**:\n{desc_preview}{'...' if len(description) > 500 and level == 'standard' else ''}\n"

                        logger.info(
                            f"MassIVE metadata extraction completed for: {identifier}"
                        )
                        return formatted

                    except Exception as e:
                        logger.error(f"Error fetching MassIVE metadata: {e}")
                        return f"Error fetching MassIVE metadata for {identifier}: {str(e)}"

                else:
                    # Other databases not yet implemented
                    return f"Metadata extraction for {database.upper()} datasets is not yet implemented. Currently supported: GEO, PRIDE, MassIVE, publications (PMID/DOI)."

            else:
                # Publication metadata extraction (existing behavior)
                # Keep source as string - service expects Optional[str]
                source_str = None if source == "auto" else source.lower()

                metadata = get_content_service().extract_metadata(
                    identifier=identifier, source=source_str
                )

                if isinstance(metadata, str):
                    return metadata  # Error message

                # Format metadata for display
                formatted = f"## Publication Metadata for {identifier}\n\n"
                formatted += f"**Title**: {metadata.title}\n"
                formatted += f"**UID**: {metadata.uid}\n"
                if metadata.journal:
                    formatted += f"**Journal**: {metadata.journal}\n"
                if metadata.published:
                    formatted += f"**Published**: {metadata.published}\n"
                if metadata.doi:
                    formatted += f"**DOI**: {metadata.doi}\n"
                if metadata.pmid:
                    formatted += f"**PMID**: {metadata.pmid}\n"
                if metadata.authors:
                    formatted += f"**Authors**: {', '.join(metadata.authors[:5])}{'...' if len(metadata.authors) > 5 else ''}\n"
                if metadata.keywords:
                    formatted += f"**Keywords**: {', '.join(metadata.keywords)}\n"

                if metadata.abstract:
                    formatted += f"\n**Abstract**:\n{metadata.abstract[:1000]}{'...' if len(metadata.abstract) > 1000 else ''}\n"

                logger.debug(f"Metadata extraction completed for: {identifier}")
                return formatted

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return f"Error extracting metadata for {identifier}: {str(e)}"

    @tool
    def validate_dataset_metadata(
        identifier: str,
        required_fields: str,
        required_values: str = None,
        threshold: float = 0.8,
        add_to_queue: bool = True,
    ) -> str:
        """
        Quickly validate if a dataset contains required metadata without downloading.

        NOW ALSO: Extracts download URLs and adds entry to download queue
        for supervisor → data_expert handoff.

        Args:
            identifier: Dataset accession ID (GSE, E-MTAB, etc.) - external identifier
            required_fields: Comma-separated required fields (e.g., "smoking_status,treatment_response")
            required_values: Optional JSON of required values (e.g., '{{"smoking_status": ["smoker", "non-smoker"]}}')
            threshold: Minimum fraction of samples with required fields (default: 0.8)
            add_to_queue: If True, add validated dataset to download queue (default: True)

        Returns:
            Validation report with recommendation (proceed/skip/manual_check)
            + download queue confirmation (if add_to_queue=True)
        """
        try:
            # Parse required fields
            fields_list = [f.strip() for f in required_fields.split(",")]

            # Parse required values if provided
            values_dict = None
            if required_values:
                try:
                    values_dict = json.loads(required_values)
                except json.JSONDecodeError:
                    return f"Error: Invalid JSON for required_values: {required_values}"

            # Use GEOService to fetch metadata only
            from lobster.services.data_access.geo_service import GEOService

            console = getattr(data_manager, "console", None)
            geo_service = GEOService(data_manager, console=console)

            # ------------------------------------------------
            # Check if metadata already in store
            # ------------------------------------------------
            if identifier in data_manager.metadata_store:
                logger.debug(
                    f"Metadata already stored for: {identifier}. returning summary"
                )
                cached_data = data_manager.metadata_store[identifier]
                metadata = cached_data.get("metadata", {})

                # Check if already in download queue
                queue_entries = [
                    entry
                    for entry in data_manager.download_queue.list_entries()
                    if entry.dataset_id == identifier
                ]

                # Add to queue if requested and not already present
                if add_to_queue and not queue_entries:
                    try:
                        logger.info(
                            f"Adding cached dataset {identifier} to download queue"
                        )

                        # Import GEOProvider
                        from lobster.tools.providers.geo_provider import GEOProvider

                        geo_provider = GEOProvider(data_manager)

                        # Extract URLs using cached metadata (returns DownloadUrlResult)
                        url_data = geo_provider.get_download_urls(identifier)

                        if url_data.error:
                            logger.warning(
                                f"URL extraction warning for {identifier}: {url_data.error}"
                            )

                        # Extract strategy config for cached datasets
                        assistant = get_data_expert()

                        # Check if strategy_config already exists in cached data
                        cached_strategy_config = cached_data.get("strategy_config")
                        if not cached_strategy_config:
                            # Extract it now and persist
                            try:
                                logger.info(
                                    f"Extracting strategy for cached dataset {identifier}"
                                )
                                cached_strategy_config = (
                                    assistant.extract_strategy_config(
                                        metadata, identifier
                                    )
                                )

                                if cached_strategy_config:
                                    # Persist to metadata_store
                                    data_manager._store_geo_metadata(
                                        geo_id=identifier,
                                        metadata=metadata,
                                        stored_by="research_agent_cached",
                                        strategy_config=(
                                            cached_strategy_config.model_dump()
                                            if hasattr(
                                                cached_strategy_config, "model_dump"
                                            )
                                            else cached_strategy_config
                                        ),
                                    )

                                    # Analyze and create recommended strategy
                                    analysis = assistant.analyze_download_strategy(
                                        cached_strategy_config, metadata
                                    )
                                    recommended_strategy = _create_recommended_strategy(
                                        cached_strategy_config,
                                        analysis,
                                        metadata,
                                        url_data,
                                    )
                                else:
                                    # Fallback strategy
                                    recommended_strategy = _create_fallback_strategy(
                                        url_data, metadata
                                    )
                            except Exception as e:
                                logger.warning(
                                    f"Strategy extraction failed for cached {identifier}: {e}"
                                )
                                recommended_strategy = _create_fallback_strategy(
                                    url_data, metadata
                                )
                        else:
                            # Use existing strategy config
                            analysis = assistant.analyze_download_strategy(
                                cached_strategy_config, metadata
                            )
                            recommended_strategy = _create_recommended_strategy(
                                cached_strategy_config, analysis, metadata, url_data
                            )

                        # Create DownloadQueueEntry
                        entry_id = f"queue_{identifier}_{uuid.uuid4().hex[:8]}"

                        # Reconstruct validation result for cached datasets
                        # Cached = previously validated successfully
                        cached_validation = MetadataValidationConfig(
                            has_required_fields=True,
                            missing_fields=[],
                            available_fields={},
                            sample_count_by_field={},
                            total_samples=metadata.get(
                                "n_samples", len(metadata.get("samples", {}))
                            ),
                            field_coverage={},
                            recommendation="proceed",
                            confidence_score=1.0,
                            warnings=[],
                        )

                        queue_entry = DownloadQueueEntry(
                            entry_id=entry_id,
                            dataset_id=identifier,
                            database="geo",
                            priority=5,
                            status=DownloadStatus.PENDING,
                            metadata=metadata,
                            validation_result=cached_validation.__dict__,
                            matrix_url=url_data.matrix_url,
                            raw_urls=url_data.get_raw_urls_as_strings(),
                            supplementary_urls=url_data.get_supplementary_urls_as_strings(),
                            h5_url=url_data.h5_url,
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            recommended_strategy=recommended_strategy,  # Use actual strategy
                            downloaded_by=None,
                            modality_name=None,
                            error_log=[],
                        )

                        # Add to download queue
                        data_manager.download_queue.add_entry(queue_entry)

                        logger.info(
                            f"Successfully added cached dataset {identifier} to download queue with entry_id: {entry_id}"
                        )

                        # Update queue_entries list for response building
                        queue_entries = [queue_entry]

                    except Exception as e:
                        logger.error(
                            f"Failed to add cached dataset {identifier} to download queue: {e}"
                        )
                        # Continue with response - queue addition is optional

                # Build concise response for cached datasets
                title = metadata.get("title", "N/A")
                if len(title) > 100:
                    title = title[:100] + "..."

                response_parts = [
                    f"## Dataset Already Validated: {identifier}",
                    "",
                    "**Status**: ✅ Metadata cached in system",
                    f"**Title**: {title}",
                    f"**Sample Count**: {metadata.get('n_samples', len(metadata.get('samples', {})))}",
                    f"**Database**: {metadata.get('database', 'GEO')}",
                    "",
                ]

                # Add queue status if exists
                if queue_entries:
                    entry = queue_entries[0]
                    response_parts.extend(
                        [
                            f"**Download Queue**: {entry.status.upper()}",
                            f"**Entry ID**: `{entry.entry_id}`",
                            f"**Priority**: {entry.priority}",
                            "",
                            "**Next steps**:",
                            f"- Status is {entry.status}: "
                            + (
                                "Ready for data_expert download"
                                if entry.status == DownloadStatus.PENDING
                                else f"Already {entry.status}"
                            ),
                        ]
                    )
                    if entry.status == DownloadStatus.COMPLETED:
                        response_parts.append(
                            f"- Load from workspace: `/workspace load {entry.modality_name}`"
                        )
                else:
                    # No queue entry exists - explain why
                    if not add_to_queue:
                        response_parts.extend(
                            [
                                "**Download Queue**: Not added (add_to_queue=False)",
                                "",
                                "**Next steps**:",
                                f"1. Call `validate_dataset_metadata(identifier='{identifier}', add_to_queue=True)` to add to download queue",
                                "2. Then hand off to data_expert with the entry_id from the response",
                            ]
                        )
                    else:
                        # Should not happen after fix, but handle gracefully
                        response_parts.extend(
                            [
                                "**Download Queue**: Failed to add (check logs for details)",
                                "",
                                "**Next steps**:",
                                "1. Check logs for queue addition error",
                                f"2. Retry: `validate_dataset_metadata(identifier='{identifier}', add_to_queue=True)`",
                            ]
                        )

                return "\n".join(response_parts)

            # ------------------------------------------------
            # If not fetch and return metadata & val res
            # ------------------------------------------------
            # Fetch metadata only (no expression data download)
            try:
                if identifier.startswith("G"):
                    metadata, validation_result = geo_service.fetch_metadata_only(
                        identifier
                    )

                    # Use metadata validation service to validate metadata
                    validation_result = get_metadata_validator().validate_dataset_metadata(
                        metadata=metadata,
                        geo_id=identifier,
                        required_fields=fields_list,
                        required_values=values_dict,
                        threshold=threshold,
                    )

                    if validation_result:
                        # Format the validation report
                        report = get_metadata_validator().format_validation_report(
                            validation_result, identifier
                        )

                        logger.info(
                            f"Metadata validation completed for {identifier}: {validation_result.recommendation}"
                        )

                        # NEW: Relax validation gate - only block CRITICAL severity
                        severity = getattr(
                            validation_result, "severity", ValidationSeverity.WARNING
                        )

                        if add_to_queue and severity != ValidationSeverity.CRITICAL:
                            # Determine validation status for queue entry
                            if validation_result.recommendation == "proceed":
                                validation_status = ValidationStatus.VALIDATED_CLEAN
                            elif validation_result.recommendation == "skip":
                                validation_status = ValidationStatus.VALIDATION_FAILED
                            else:  # manual_check
                                validation_status = (
                                    ValidationStatus.VALIDATED_WITH_WARNINGS
                                )

                            try:
                                # Import GEOProvider
                                from lobster.tools.providers.geo_provider import (
                                    GEOProvider,
                                )

                                geo_provider = GEOProvider(data_manager)

                                # Extract URLs (returns DownloadUrlResult)
                                url_data = geo_provider.get_download_urls(identifier)

                                # Check for URL extraction errors
                                if url_data.error:
                                    logger.warning(
                                        f"URL extraction warning for {identifier}: {url_data.error}"
                                    )

                                # NEW: Extract strategy using data_expert_assistant
                                logger.info(
                                    f"Extracting download strategy for {identifier}"
                                )
                                assistant = get_data_expert()

                                # Extract file config using LLM (~2-5s)
                                try:
                                    strategy_config = assistant.extract_strategy_config(
                                        metadata, identifier
                                    )

                                    if strategy_config:
                                        # CRITICAL FIX: Persist strategy_config to metadata_store
                                        # This enables geo_service.py to find file-level details
                                        logger.info(
                                            f"Persisting strategy_config to metadata_store for {identifier}"
                                        )
                                        data_manager._store_geo_metadata(
                                            geo_id=identifier,
                                            metadata=metadata,
                                            stored_by="research_agent_validate",
                                            strategy_config=(
                                                strategy_config.model_dump()
                                                if hasattr(
                                                    strategy_config, "model_dump"
                                                )
                                                else strategy_config
                                            ),
                                        )

                                        # Analyze and generate recommendations
                                        analysis = assistant.analyze_download_strategy(
                                            strategy_config, metadata
                                        )

                                        # Convert to download_queue.StrategyConfig
                                        recommended_strategy = (
                                            _create_recommended_strategy(
                                                strategy_config,
                                                analysis,
                                                metadata,
                                                url_data,
                                            )
                                        )
                                        logger.info(
                                            f"Strategy recommendation for {identifier}: {recommended_strategy.strategy_name} "
                                            f"(confidence: {recommended_strategy.confidence:.2f})"
                                        )
                                    else:
                                        # Fallback: URL-based strategy
                                        logger.warning(
                                            f"LLM strategy extraction failed for {identifier}, using URL-based fallback"
                                        )
                                        recommended_strategy = (
                                            _create_fallback_strategy(
                                                url_data, metadata
                                            )
                                        )
                                except Exception as e:
                                    # Graceful fallback on any error
                                    logger.warning(
                                        f"Strategy extraction error for {identifier}: {e}, using URL-based fallback"
                                    )
                                    recommended_strategy = _create_fallback_strategy(
                                        url_data, metadata
                                    )

                                # Create DownloadQueueEntry
                                entry_id = f"queue_{identifier}_{uuid.uuid4().hex[:8]}"

                                queue_entry = DownloadQueueEntry(
                                    entry_id=entry_id,
                                    dataset_id=identifier,
                                    database="geo",
                                    priority=5,  # Default priority
                                    status=DownloadStatus.PENDING,
                                    # Metadata from validation
                                    metadata=metadata,
                                    validation_result=validation_result.__dict__,
                                    validation_status=validation_status,  # NEW
                                    # URLs from GEOProvider (DownloadUrlResult)
                                    matrix_url=url_data.matrix_url,
                                    raw_urls=url_data.get_raw_urls_as_strings(),
                                    supplementary_urls=url_data.get_supplementary_urls_as_strings(),
                                    h5_url=url_data.h5_url,
                                    # Timestamps
                                    created_at=datetime.now(),
                                    updated_at=datetime.now(),
                                    # Strategy recommendation from data_expert_assistant
                                    recommended_strategy=recommended_strategy,  # NEW (no longer None!)
                                    downloaded_by=None,
                                    modality_name=None,
                                    error_log=[],
                                )

                                # Add to download queue
                                data_manager.download_queue.add_entry(queue_entry)

                                logger.info(
                                    f"Added {identifier} to download queue with entry_id: {entry_id}"
                                )

                                # Enhanced response with strategy information
                                report += "\n\n## Download Queue\n\n"
                                report += f"✅ Dataset '{identifier}' validated and added to queue\n"
                                report += f"- **Entry ID**: `{entry_id}`\n"
                                report += f"- **Validation status**: {validation_status.value}\n"
                                report += f"- **Recommended strategy**: {recommended_strategy.strategy_name} (confidence: {recommended_strategy.confidence:.2f})\n"
                                report += f"- **Rationale**: {recommended_strategy.rationale}\n"
                                report += f"- **Files found**: {url_data.file_count}\n"
                                if url_data.matrix_url:
                                    report += "- **Matrix file**: Available\n"
                                supplementary_urls = (
                                    url_data.get_supplementary_urls_as_strings()
                                )
                                if supplementary_urls:
                                    report += f"- **Supplementary files**: {len(supplementary_urls)} file(s)\n"

                                # Add warnings if validation status has warnings
                                if (
                                    validation_status
                                    == ValidationStatus.VALIDATED_WITH_WARNINGS
                                ):
                                    warnings = getattr(
                                        validation_result, "warnings", []
                                    )
                                    if warnings:
                                        report += f"\n⚠️ **Warnings**:\n"
                                        for warning in warnings[
                                            :3
                                        ]:  # Show max 3 warnings
                                            report += f"  - {warning}\n"

                                report += "\n**Next steps**:\n"
                                report += "1. Supervisor can query queue: `get_content_from_workspace(workspace='download_queue')`\n"
                                report += f"2. Hand off to data_expert with entry_id: `{entry_id}`\n"

                            except Exception as e:
                                logger.error(
                                    f"Failed to add {identifier} to download queue: {e}"
                                )
                                # Return validation result even if queue addition fails
                                report += f"\n\n⚠️ Warning: Could not add to download queue: {str(e)}\n"

                        return report
                    else:
                        return f"Error: Failed to validate metadata for {identifier}"
                else:
                    logger.info(
                        f"Currently only GEO metadata can be retrieved. {identifier} doesnt seem to be a GEO identifier"
                    )
                    return f"Currently only GEO metadata can be retrieved. {identifier} doesnt seem to be a GEO identifier"

            except Exception as e:
                logger.error(f"Error accessing dataset {identifier}: {e}")
                return f"Error accessing dataset {identifier}: {str(e)}"

        except Exception as e:
            logger.error(f"Error in metadata validation: {e}")
            return f"Error validating dataset metadata: {str(e)}"

    @tool
    def extract_methods(identifier: str, focus: str = None) -> str:
        """
        Extract computational methods from publication(s) - supports single or batch processing.

        Automatically extracts:
        - Software/tools used (e.g., Scanpy, Seurat, DESeq2)
        - Parameter values and cutoffs (e.g., min_genes=200, p<0.05)
        - Statistical methods (e.g., Wilcoxon test, FDR correction)
        - Data sources and sample sizes
        - Normalization and QC workflows

        The service handles batch processing transparently (2-10 papers typical). Use this for
        competitive intelligence, protocol standardization, or replicating published analyses.

        Supported Identifiers:
        - PMID (e.g., "PMID:12345678" or "12345678") - Auto-resolves via PMC/bioRxiv
        - DOI (e.g., "10.1038/s41586-021-12345-6") - Auto-resolves to open access PDF
        - Direct PDF URL (e.g., https://nature.com/articles/paper.pdf)
        - Webpage URL (webpage-first extraction, then PDF fallback)
        - Comma-separated for batch (e.g., "PMID:123,PMID:456" - processes sequentially)

        Extraction Strategy: PMC XML → Webpage → PDF (automatic cascade)

        Args:
            identifier: Publication identifier - single or comma-separated for batch processing
            focus: Optional focus area (options: "software", "parameters", "statistics").
                   When specified, returns only the focused aspect from extraction results.
                   Useful for targeted analysis (e.g., "What software did competitors use?")

        Returns:
            JSON-formatted extraction of methods, parameters, and software used
            OR helpful suggestions if paper is paywalled

        Examples:
            # Single paper extraction
            extract_methods("PMID:12345678")

            # Focus on software tools only
            extract_methods("PMID:12345678", focus="software")

            # Focus on parameter values
            extract_methods("10.1038/s41586-021-12345-6", focus="parameters")

            # Batch processing (2-10 papers typical)
            extract_methods("PMID:123,PMID:456,PMID:789")

            # Batch with software focus for competitive analysis
            extract_methods("PMID:123,PMID:456", focus="software")
        """
        try:
            # Check if batch processing (comma-separated identifiers)
            identifiers = [id.strip() for id in identifier.split(",")]

            if len(identifiers) > 1:
                # Batch processing mode
                logger.debug(f"Batch processing {len(identifiers)} publications")
                batch_results = []

                for idx, identifier in enumerate(identifiers, 1):
                    try:
                        logger.info(
                            f"Processing {idx}/{len(identifiers)}: {identifier}"
                        )

                        # Get full content
                        content = get_content_service().get_full_content(
                            source=identifier,
                            prefer_webpage=True,
                            keywords=["methods", "materials", "analysis", "workflow"],
                            max_paragraphs=100,
                        )

                        # Extract methods
                        methods = get_content_service().extract_methods(content)

                        batch_results.append(
                            {
                                "identifier": identifier,
                                "status": "success",
                                "software_used": methods.get("software_used", []),
                                "parameters": methods.get("parameters", {}),
                                "statistical_methods": methods.get(
                                    "statistical_methods", []
                                ),
                                "extraction_confidence": methods.get(
                                    "extraction_confidence", 0.0
                                ),
                                "source_type": content.get("source_type", "unknown"),
                            }
                        )

                    except Exception as e:
                        logger.error(
                            f"Failed to extract methods from {identifier}: {e}"
                        )
                        batch_results.append(
                            {
                                "identifier": identifier,
                                "status": "failed",
                                "error": str(e),
                            }
                        )

                # Format batch results
                response = f"## Batch Method Extraction Results ({len(identifiers)} papers)\n\n"

                # Apply focus filter if specified
                if focus and focus.lower() in ["software", "parameters", "statistics"]:
                    response += f"**Focus**: {focus.title()}\n\n"

                for result in batch_results:
                    response += f"### {result['identifier']}\n"
                    if result["status"] == "success":
                        if focus == "software":
                            response += f"**Software**: {', '.join(result['software_used']) if result['software_used'] else 'None detected'}\n\n"
                        elif focus == "parameters":
                            response += f"**Parameters**: {json.dumps(result['parameters'], indent=2)}\n\n"
                        elif focus == "statistics":
                            response += f"**Statistical Methods**: {', '.join(result['statistical_methods']) if result['statistical_methods'] else 'None detected'}\n\n"
                        else:
                            # Full extraction
                            response += f"**Software**: {', '.join(result['software_used']) if result['software_used'] else 'None'}\n"
                            response += f"**Parameters**: {len(result['parameters'])} parameters detected\n"
                            response += f"**Statistical Methods**: {', '.join(result['statistical_methods']) if result['statistical_methods'] else 'None'}\n"
                            response += f"**Confidence**: {result['extraction_confidence']:.2f}\n\n"
                    else:
                        response += f"**Status**: Failed - {result['error']}\n\n"

                logger.info(
                    f"Batch processing complete: {len(batch_results)} papers processed"
                )
                return response

            else:
                # Single paper processing mode
                identifier = identifiers[0]

                # Get full content (webpage-first, with PDF fallback)
                content = get_content_service().get_full_content(
                    source=identifier,
                    prefer_webpage=True,
                    keywords=["methods", "materials", "analysis", "workflow"],
                    max_paragraphs=100,
                )

                # Extract methods section
                methods = get_content_service().extract_methods(content)

                # Apply focus filter if specified
                if focus and focus.lower() in ["software", "parameters", "statistics"]:
                    if focus.lower() == "software":
                        formatted_result = {
                            "software_used": methods.get("software_used", []),
                            "focus": "software",
                        }
                    elif focus.lower() == "parameters":
                        formatted_result = {
                            "parameters": methods.get("parameters", {}),
                            "focus": "parameters",
                        }
                    elif focus.lower() == "statistics":
                        formatted_result = {
                            "statistical_methods": methods.get(
                                "statistical_methods", []
                            ),
                            "focus": "statistics",
                        }
                else:
                    # Full extraction (no focus)
                    formatted_result = {
                        "software_used": methods.get("software_used", []),
                        "parameters": methods.get("parameters", {}),
                        "statistical_methods": methods.get("statistical_methods", []),
                        "extraction_confidence": methods.get(
                            "extraction_confidence", 0.0
                        ),
                        "content_source": content.get("source_type", "unknown"),
                        "extraction_time": content.get("extraction_time", 0.0),
                    }

                formatted = json.dumps(formatted_result, indent=2)
                logger.info(
                    f"Successfully extracted methods from paper: {identifier[:80]}..."
                )

                return f"## Extracted Methods from Paper\n\n{formatted}\n\n**Source Type**: {content.get('source_type')}\n**Extraction Time**: {content.get('extraction_time', 0):.2f}s"

        except Exception as e:
            logger.error(f"Error extracting paper methods: {e}")
            error_msg = str(e)

            # Check if it's a paywalled paper with suggestions
            if "not openly accessible" in error_msg or "paywalled" in error_msg.lower():
                return f"## Paper Access Issue\n\n{error_msg}"
            else:
                return f"Error extracting methods from paper: {error_msg}"

    # ============================================================
    # Phase 1 NEW TOOLS: Two-Tier Access & Webpage-First Strategy
    # ============================================================

    @tool
    def fast_abstract_search(identifier: str) -> str:
        """
        Fast abstract retrieval for publication discovery (200-500ms).

        This is the FAST PATH for two-tier content access strategy. Use this to quickly
        screen publications for relevance before committing to full content extraction.
        Perfect for batch screening, relevance checking, or when you just need the summary.

        Two-Tier Strategy:
        - Tier 1 (this tool): Quick abstract via NCBI (200-500ms) ✅ FAST
        - Tier 2 (read_full_publication): Full content with methods (2-8 seconds)

        Use Cases:
        - Screen multiple papers for relevance (5 papers = 2.5 seconds)
        - Get high-level understanding without full download
        - Check abstract before deciding on method extraction
        - User asks for "abstract" or "summary" only

        Supported Identifiers:
        - PMID: "PMID:12345678" or "12345678"
        - DOI: "10.1038/s41586-021-12345-6"

        Args:
            identifier: PMID or DOI of the publication

        Returns:
            Formatted abstract with title, authors, journal, and full abstract text

        Examples:
            # Fast screening workflow
            fast_abstract_search("PMID:35042229")

            # DOI lookup
            fast_abstract_search("10.1038/s41586-021-03852-1")

        Performance: 200-500ms typical (10x faster than full extraction)
        """
        try:
            logger.info(f"Getting quick abstract for: {identifier}")

            # Initialize AbstractProvider
            abstract_provider = AbstractProvider(data_manager=data_manager)

            # Get abstract metadata
            metadata = abstract_provider.get_abstract(identifier)

            # Format response
            response = f"""## {metadata.title}

**Authors:** {', '.join(metadata.authors[:5])}{'...' if len(metadata.authors) > 5 else ''}
**Journal:** {metadata.journal or 'N/A'}
**Published:** {metadata.published or 'N/A'}
**PMID:** {metadata.pmid or 'N/A'}
**DOI:** {metadata.doi or 'N/A'}

### Abstract

{metadata.abstract}

*Retrieved via fast abstract API (no PDF download)*
*For full content with Methods section, use read_full_publication()*
"""

            logger.info(
                f"Successfully retrieved abstract: {len(metadata.abstract)} chars"
            )
            return response

        except Exception as e:
            logger.error(f"Error getting quick abstract: {e}")
            return f"""## Error Retrieving Abstract

Could not retrieve abstract for: {identifier}

**Error:** {str(e)}

**Suggestions:**
- Verify the identifier is correct (PMID or DOI)
- Check if publication exists in PubMed
- Try using DOI if PMID failed, or vice versa
- For non-PubMed papers, use read_full_publication() instead
"""

    @tool
    def read_full_publication(identifier: str, prefer_webpage: bool = True) -> str:
        """
        Read full publication content with automatic caching - the DEEP PATH.

        Extracts complete publication content with intelligent three-tier cascade strategy.
        Content is automatically cached for future workspace access. Use after screening
        with fast_abstract_search() or when you need the full Methods section.

        Three-Tier Cascade Strategy:
        - PRIORITY: PMC Full Text XML (500ms, 95% accuracy, structured) ⭐ FASTEST
        - Fallback 1: Webpage extraction (Nature, Science, Cell Press) - 2-5 seconds
        - Fallback 2: PDF parsing with Docling - 3-8 seconds

        Use Cases:
        - Extract complete Methods section for protocol replication
        - User asks for "parameters", "software used", "full text"
        - After relevance check with fast_abstract_search()
        - Need tables, figures, supplementary references

        Automatic Workspace Caching:
        - Content cached as `publication_PMID12345` or `publication_DOI...`
        - Retrieve later with get_content_from_workspace()
        - Enables handoff to specialists with full context

        PMC-First Strategy (Phase 4):
        - Covers 30-40% of biomedical papers (NIH-funded + open access)
        - 95% method extraction accuracy vs 70% from abstracts alone
        - 100% table parsing success vs 80% heuristic approaches
        - Automatic fallback to webpage → PDF if PMC unavailable

        Supported Identifiers:
        - PMID: "PMID:12345678" (auto-tries PMC, then resolves)
        - DOI: "10.1038/s41586-021-12345-6" (auto-tries PMC, then resolves)
        - Direct URL: "https://www.nature.com/articles/s41586-025-09686-5"
        - PDF URL: "https://biorxiv.org/content/10.1101/2024.01.001.pdf"

        Args:
            identifier: PMID, DOI, or URL
            prefer_webpage: Try webpage before PDF (default: True)

        Returns:
            Full content markdown with sections, tables, metadata, and cache location

        Examples:
            # Read with PMC-first auto-cascade
            read_full_publication("PMID:35042229")

            # Read publisher webpage
            read_full_publication("https://www.nature.com/articles/s41586-025-09686-5")

            # Force PDF extraction
            read_full_publication("10.1038/s41586-021-12345-6", prefer_webpage=False)

        Performance:
        - PMC XML: 500ms (fastest path, 30-40% of papers)
        - Webpage: 2-5 seconds
        - PDF: 3-8 seconds
        """
        try:
            logger.info(f"Getting publication overview for: {identifier}")

            # Check if identifier is a direct webpage URL
            is_webpage_url = identifier.startswith(
                "http"
            ) and not identifier.lower().endswith(".pdf")

            if is_webpage_url and prefer_webpage:
                # Try webpage extraction first
                try:
                    logger.info(
                        f"Attempting webpage extraction for: {identifier[:80]}..."
                    )
                    webpage_provider = WebpageProvider(data_manager=data_manager)

                    # Extract full content
                    markdown_content = webpage_provider.extract(
                        identifier, max_paragraphs=100
                    )

                    response = f"""## Publication Overview (Webpage Extraction)

**Source:** {identifier[:100]}...
**Extraction Method:** Webpage (faster, structure-aware)
**Content Length:** {len(markdown_content)} characters

{markdown_content}

*Extracted from publisher webpage using structure-aware parsing*
*For abstract-only view, use fast_abstract_search()*
"""

                    logger.info(
                        f"Successfully extracted webpage: {len(markdown_content)} chars"
                    )
                    return response

                except Exception as webpage_error:
                    logger.warning(
                        f"Webpage extraction failed, trying PDF fallback: {webpage_error}"
                    )
                    # Fall through to UnifiedContentService extraction below

            # Fallback: Use UnifiedContentService for full extraction (now handles DOI resolution)
            logger.info(
                f"Using ContentAccessService for comprehensive extraction: {identifier}"
            )

            # get_full_content() now handles DOI resolution automatically
            content_result = get_content_service().get_full_content(
                source=identifier,
                prefer_webpage=False,  # Already tried webpage above if applicable
                keywords=["methods", "materials", "analysis"],
                max_paragraphs=100,
            )

            # Format response with extracted content
            content = content_result.get("content", "")
            methods_text = content_result.get("methods_text", "")
            tier_used = content_result.get("tier_used", "unknown")
            source_type = content_result.get("source_type", "unknown")
            metadata = content_result.get("metadata", {})

            response = f"""## Publication Overview ({tier_used.replace('_', ' ').title()})

**Source:** {identifier}
**Extraction Method:** {source_type.title()} extraction via {tier_used}
**Content Length:** {len(content)} characters
**Software Detected:** {', '.join(metadata.get('software', [])[:5]) if metadata.get('software') else 'None'}
**Tables Found:** {metadata.get('tables', 0)}
**Formulas Found:** {metadata.get('formulas', 0)}

{content or methods_text}

*Extracted using {source_type} parsing with automatic DOI/PMID resolution*
*For abstract-only view, use fast_abstract_search()*
"""

            logger.info(
                f"Successfully extracted content via UnifiedContentService: {len(content)} chars"
            )
            return response

        except Exception as e:
            logger.error(f"Error getting publication overview: {e}")
            error_msg = str(e)

            # Check if it's a paywalled paper
            if "not openly accessible" in error_msg or "paywalled" in error_msg.lower():
                return f"""## Publication Access Issue

{error_msg}

**Suggestions:**
1. Try fast_abstract_search("{identifier}") to get the abstract without full text
2. Check if a preprint version exists on bioRxiv/medRxiv
3. Search for author's institutional repository
4. Contact corresponding author for access
"""
            else:
                return f"""## Error Extracting Publication

Could not extract content for: {identifier}

**Error:** {error_msg}

**Troubleshooting:**
- Verify identifier is correct (PMID, DOI, or URL)
- Try fast_abstract_search() for basic information
- Check if paper is freely accessible
- For webpage URLs, ensure they're not behind paywall
"""

    # ============================================================
    # Helper: Flexible Identifier Resolution for Publication Queue
    # ============================================================

    def _find_or_create_queue_entry(identifier: str, data_manager) -> str:
        """
        Resolve flexible identifier to queue entry_id (find existing or create new).

        Handles three identifier types:
        1. Entry ID: "pub_queue_doi_..." → Return as-is (backward compatible)
        2. PMID: "PMID:31204333" or "31204333" → Find or create entry
        3. DOI: "10.1038/..." → Find or create entry

        Args:
            identifier: Entry ID, PMID, or DOI
            data_manager: DataManagerV2 instance for queue access

        Returns:
            Queue entry_id (existing or newly created)

        Raises:
            ValueError: If identifier format invalid or metadata fetch fails
        """
        # Step 1: Check if it's already an entry_id
        if identifier.startswith("pub_queue_"):
            logger.debug(f"Identifier is entry_id: {identifier}")
            return identifier  # Return as-is (backward compatible)

        # Step 2: Parse PMID or DOI
        pmid = None
        doi = None

        # Try parsing as PMID
        if identifier.upper().startswith("PMID:"):
            pmid = identifier[5:].strip()
        elif identifier.isdigit() and len(identifier) == 8:  # Bare PMID (8 digits)
            pmid = identifier
        else:
            # Try parsing as DOI (contains "10.")
            if "10." in identifier:
                doi = identifier.strip()

        if not pmid and not doi:
            raise ValueError(
                f"Invalid identifier '{identifier}'. "
                f"Expected: entry_id (pub_queue_...), PMID (31204333), or DOI (10.1038/...)"
            )

        logger.info(f"Resolving identifier: pmid={pmid}, doi={doi}")

        # Step 3: Search existing queue for matching entry
        queue = data_manager.publication_queue
        if queue and queue.queue_file.exists():
            entries = queue.list_entries()
            for entry in entries:
                # Match by PMID or DOI
                if (pmid and entry.pmid == pmid) or (doi and entry.doi == doi):
                    logger.info(f"Found existing queue entry: {entry.entry_id}")
                    return entry.entry_id

        # Step 4: Not found - create new entry
        logger.info(f"Creating new queue entry for {identifier}")

        # Fetch metadata from PubMed
        try:
            # Use AbstractProvider for quick metadata fetch
            abstract_provider = AbstractProvider(data_manager=data_manager)
            metadata = abstract_provider.get_abstract(pmid or doi)

            # Create entry_id from PMID or DOI
            if pmid:
                entry_id = f"pub_queue_pmid_{pmid}"
            else:
                # Sanitize DOI for entry_id
                doi_sanitized = doi.replace("/", "_").replace(".", "_")
                entry_id = f"pub_queue_doi_{doi_sanitized}"

            # Create PublicationQueueEntry
            from datetime import datetime
            from lobster.core.schemas.publication_queue import (
                PublicationQueueEntry,
                PublicationStatus,
            )

            now = datetime.now()

            entry = PublicationQueueEntry(
                entry_id=entry_id,
                pmid=metadata.pmid,
                doi=metadata.doi,
                title=metadata.title,
                authors=metadata.authors,
                journal=metadata.journal,
                year=metadata.published[:4] if metadata.published else None,
                abstract=metadata.abstract,
                priority=5,  # Default priority
                status=PublicationStatus.PENDING,
                schema_type="general",  # Default schema
                extraction_level="methods",
                created_at=now,
                updated_at=now,
            )

            # Add to queue (auto-creates queue file if needed)
            queue.add_entry(entry)

            logger.info(f"Created queue entry: {entry_id}")
            return entry_id

        except Exception as e:
            logger.error(f"Failed to create queue entry for {identifier}: {e}")
            raise ValueError(f"Could not fetch metadata for {identifier}: {e}")

    # ============================================================
    # Publication Queue Management (3 tools)
    # ============================================================

    @tool
    def process_publication_entry(
        entry_id: str,
        extraction_tasks: str = "resolve_identifiers,ncbi_enrich,metadata,methods,identifiers,validate_provenance,fetch_sra_metadata",
        status_override: str = None,
        error_message: str = None,
    ) -> str:
        """
        Process a publication queue entry OR manually update its status.

        **Two modes of operation:**

        1. **Processing mode** (default): Full 7-step pipeline with NCBI enrichment
        2. **Status override mode**: Manually update status without processing

        **Processing Mode** - Flexible identifier handling:
        - Queue entry ID: "pub_queue_doi_10_1234..." (existing entry)
        - PMID: "PMID:31204333" or "31204333" (find or create entry)
        - DOI: "10.1038/..." (find or create entry)

        If PMID/DOI provided and not in queue:
        - Auto-creates queue entry with metadata from PubMed/DOI.org
        - Initializes queue file if doesn't exist
        - Returns processing report as normal

        **Status Override Mode** - Use ONLY for:
        - Resetting stale entries (stuck in "extracting") to "pending" before retry
        - Marking unrecoverable entries as "failed" with error messages
        - Administrative corrections when processing is impossible

        Args:
            entry_id: Entry ID OR PMID OR DOI (flexible)
            extraction_tasks: Comma-separated tasks (default: full 7-step pipeline)
                            Individual tasks:
                            - "resolve_identifiers": DOI → PMID resolution via NCBI ID Converter
                            - "ncbi_enrich": E-Link dataset discovery (GEO, SRA, BioProject, BioSample)
                            - "metadata": Abstract/introduction/methods extraction from PMC/publisher
                            - "methods": Detailed methods section extraction
                            - "identifiers": Regex-based identifier extraction from full text
                            - "validate_provenance": Section-based provenance (primary vs referenced data)
                            - "fetch_sra_metadata": BioProject → SRA sample metadata via pysradb
                            Shortcuts:
                            - "full_text": Runs all 7 steps (same as default)
                            Subset examples:
                            - "metadata,identifiers": Quick text mining only
                            - "ncbi_enrich,fetch_sra_metadata": NCBI-only enrichment
                            Ignored if status_override is set.
            status_override: Manual status update (default: None = processing mode)
                           Options: "pending", "extracting", "completed", "failed", "paywalled", "handoff_ready"
                           When set, skips processing and directly updates status.
            error_message: Error message for failed status (only used with status_override="failed")

        Returns:
            Processing report with extracted content OR status update confirmation

        Examples:
            # PROCESSING MODE: Process existing queue entry
            process_publication_entry("pub_queue_abc123")

            # PROCESSING MODE: Process PMID (auto-creates entry if needed)
            process_publication_entry("PMID:31204333")
            process_publication_entry("31204333")

            # PROCESSING MODE: Process DOI (auto-creates entry if needed)
            process_publication_entry("10.1038/s41586-021-03852-1")

            # STATUS OVERRIDE MODE: Reset stale entry to pending
            process_publication_entry("pub_queue_abc123", status_override="pending")

            # STATUS OVERRIDE MODE: Mark as failed with error
            process_publication_entry("pub_queue_abc123", status_override="failed",
                                    error_message="Content not accessible")

            # STATUS OVERRIDE MODE: Mark as paywalled (when extraction is blocked)
            process_publication_entry("pub_queue_abc123", status_override="paywalled")
        """
        if not HAS_PUBLICATION_PROCESSING:
            return "Publication processing requires a premium subscription. Visit https://omics-os.com/pricing"

        # STATUS OVERRIDE MODE: Manual status update without processing
        if status_override:
            try:
                # Validate status
                valid_statuses = [
                    "pending",
                    "extracting",
                    "metadata_extracted",
                    "metadata_enriched",
                    "handoff_ready",
                    # NOTE: "completed" is intentionally excluded
                    # ONLY metadata_assistant can set COMPLETED status after harmonization
                    # research_agent should NEVER mark entries as complete
                    "failed",
                    "paywalled",
                ]
                if status_override.lower() not in valid_statuses:
                    return f"Error: Invalid status '{status_override}'. Valid options: {', '.join(valid_statuses)}"

                # Get current entry
                try:
                    entry = data_manager.publication_queue.get_entry(entry_id)
                except Exception as e:
                    return f"Error: Entry '{entry_id}' not found in publication queue: {str(e)}"

                # Update status
                old_status = str(entry.status)
                data_manager.publication_queue.update_status(
                    entry_id=entry_id,
                    status=(
                        status_override.lower()
                        if isinstance(entry.status, str)
                        else entry.status.__class__(status_override.lower())
                    ),
                    error=error_message if status_override.lower() == "failed" else None,
                    processed_by="research_agent",
                )

                # Log to W3C-PROV for reproducibility (orchestration operation - no IR)
                data_manager.log_tool_usage(
                    tool_name="process_publication_entry",
                    parameters={
                        "entry_id": entry_id,
                        "mode": "status_override",
                        "old_status": old_status,
                        "new_status": status_override.lower(),
                        "error_message": (
                            error_message if status_override.lower() == "failed" else None
                        ),
                        "title": entry.title or "N/A",
                        "pmid": entry.pmid,
                        "doi": entry.doi,
                    },
                    description=f"Updated publication status {entry_id}: {old_status} → {status_override.lower()}",
                )

                response = f"""## Publication Status Updated (Manual Override)

**Entry ID**: {entry_id}
**Title**: {entry.title or 'N/A'}
**Old Status**: {old_status}
**New Status**: {status_override.upper()}
"""

                if error_message:
                    response += f"\n**Error Message**: {error_message}\n"

                return response

            except Exception as e:
                logger.error(f"Failed to update publication status: {e}")
                return f"Error updating publication status: {str(e)}"

        # PROCESSING MODE: Full content extraction workflow
        # Resolve identifier: entry_id, PMID, or DOI → entry_id
        resolved_entry_id = _find_or_create_queue_entry(entry_id, data_manager)

        # Process the entry (existing service logic)
        return publication_processing_service.process_entry(
            entry_id=resolved_entry_id, extraction_tasks=extraction_tasks
        )

    @tool
    def process_publication_queue(
        status_filter: str = "pending",
        max_entries: int = 0,
        extraction_tasks: str = "resolve_identifiers,ncbi_enrich,metadata,methods,identifiers,validate_provenance,fetch_sra_metadata",
        parallel_workers: int = 1,
        force_reprocess: bool = False,
    ) -> str:
        """
        Batch process multiple publication queue entries.

        Args:
            status_filter: Queue status to target (default: "pending").
                          Options: pending, extracting, completed, handoff_ready, etc.
                          Ignored if force_reprocess=True.
            max_entries: Maximum entries to process (0 = all matching).
            extraction_tasks: Comma-separated tasks (default: full 7-step pipeline).
                            Individual tasks:
                            - "resolve_identifiers": DOI → PMID resolution via NCBI ID Converter
                            - "ncbi_enrich": E-Link dataset discovery (GEO, SRA, BioProject, BioSample)
                            - "metadata": Abstract/introduction/methods extraction from PMC/publisher
                            - "methods": Detailed methods section extraction
                            - "identifiers": Regex-based identifier extraction from full text
                            - "validate_provenance": Section-based provenance (primary vs referenced data)
                            - "fetch_sra_metadata": BioProject → SRA sample metadata via pysradb
                            Shortcuts:
                            - "full_text": Runs all 7 steps (same as default)
                            Subset examples:
                            - "metadata,identifiers": Quick text mining only
                            - "ncbi_enrich,fetch_sra_metadata": NCBI-only enrichment
            parallel_workers: Number of parallel workers (default: 1 = sequential).
                             Use 2-8 for faster processing of large queues.
                             Higher values (>5) risk NCBI API rate limit issues.
            force_reprocess: Reprocess ALL entries regardless of current status (default: False).
                            When True, ignores status_filter and processes all queue entries.
                            Use for: batch re-enrichment with updated logic, comprehensive reprocessing.
                            Combine with max_entries to limit scope (e.g., first 10 entries only).

        Returns:
            Processing report with per-entry status, identifiers found, and workspace keys.

        Examples:
            # Process first 3 pending entries (sequential)
            process_publication_queue(max_entries=3)

            # Process 10 entries with 2 parallel workers
            process_publication_queue(max_entries=10, parallel_workers=2)

            # Process all metadata_enriched entries (re-extraction)
            process_publication_queue(status_filter="metadata_enriched", max_entries=0)

            # BATCH REPROCESSING: Reprocess first 10 entries regardless of status
            process_publication_queue(force_reprocess=True, max_entries=10)

            # BATCH REPROCESSING: Reprocess ALL entries (use with caution!)
            process_publication_queue(force_reprocess=True, max_entries=0)
        """
        if not HAS_PUBLICATION_PROCESSING:
            return "Publication processing requires a premium subscription. Visit https://omics-os.com/pricing"

        if parallel_workers > 1:
            # Use parallel processing with Rich progress display
            try:
                from lobster.core.schemas.publication_queue import PublicationStatus
            except ImportError:
                return "Publication queue schema requires a premium subscription."

            queue = data_manager.publication_queue
            status_enum = None

            # Force reprocess mode: ignore status filter, process all entries
            if force_reprocess:
                status_enum = None  # None = all entries
            elif status_filter and status_filter.lower() not in {"any", "all"}:
                try:
                    status_enum = PublicationStatus(status_filter.lower())
                except ValueError:
                    return f"Error: Invalid status filter '{status_filter}'"

            entries = sorted(
                queue.list_entries(status=status_enum), key=lambda e: e.created_at
            )
            if max_entries and max_entries > 0:
                entries = entries[:max_entries]

            if not entries:
                return (
                    f"No publication queue entries found with status '{status_filter}'."
                )

            entry_ids = [e.entry_id for e in entries]

            result = publication_processing_service.process_entries_parallel(
                entry_ids=entry_ids,
                extraction_tasks=extraction_tasks,
                max_workers=parallel_workers,
                show_progress=True,
            )
            return result.to_summary_string()
        else:
            # Sequential processing
            # Force reprocess mode: pass None to process all entries
            final_status_filter = None if force_reprocess else status_filter
            return publication_processing_service.process_queue_entries(
                status_filter=final_status_filter,
                max_entries=max_entries,
                extraction_tasks=extraction_tasks,
            )


    # ============================================================
    # Phase 4 TOOLS: Workspace Management (shared tools from workspace_tool.py)
    # ============================================================

    # Create workspace tools using shared factories (Phase 7+: deduplication complete)
    write_to_workspace = create_write_to_workspace_tool(data_manager)
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)

    # ============================================================
    # Helper Methods: Strategy Mapping
    # ============================================================

    def _create_recommended_strategy(
        strategy_config,  # data_expert_assistant.StrategyConfig
        analysis: dict,
        metadata: dict,
        url_data,  # DownloadUrlResult from GEOProvider.get_download_urls()
    ) -> StrategyConfig:
        """
        Convert data_expert_assistant analysis to download_queue.StrategyConfig.

        Args:
            strategy_config: File-level strategy from extract_strategy_config()
            analysis: Analysis dict from analyze_download_strategy()
            metadata: GEO metadata dictionary
            url_data: DownloadUrlResult from GEOProvider.get_download_urls()

        Returns:
            StrategyConfig for DownloadQueueEntry.recommended_strategy
        """
        # Determine primary strategy based on file availability
        if analysis.get("has_h5ad", False):
            strategy_name = "H5_FIRST"
            confidence = 0.95
            rationale = f"H5AD file available with optimal single-file structure ({url_data.file_count} total files)"
        elif analysis.get("has_processed_matrix", False):
            strategy_name = "MATRIX_FIRST"
            confidence = 0.85
            rationale = f"Processed matrix available ({strategy_config.processed_matrix_name if hasattr(strategy_config, 'processed_matrix_name') else 'unknown'})"
        elif analysis.get("has_raw_matrix", False) or analysis.get(
            "raw_data_available", False
        ):
            strategy_name = "SAMPLES_FIRST"
            confidence = 0.75
            rationale = "Raw data available for full preprocessing control"
        else:
            strategy_name = "AUTO"
            confidence = 0.50
            rationale = "No clear optimal strategy detected, using auto-detection"

        # Determine concatenation strategy based on sample count
        n_samples = metadata.get("n_samples", metadata.get("sample_count", 0))
        platform = metadata.get("platform", "")

        if n_samples < 20 and platform:
            concatenation_strategy = "union"
            use_intersecting_genes_only = False
        elif n_samples >= 20:
            concatenation_strategy = "intersection"
            use_intersecting_genes_only = True
        else:
            concatenation_strategy = "auto"
            use_intersecting_genes_only = None

        # Determine execution parameters based on file count
        file_count = url_data.file_count
        if file_count > 100:
            timeout = 7200  # 2 hours
            max_retries = 5
        elif file_count > 20:
            timeout = 3600  # 1 hour
            max_retries = 3
        else:
            timeout = 1800  # 30 minutes
            max_retries = 3

        return StrategyConfig(
            strategy_name=strategy_name,
            concatenation_strategy=concatenation_strategy,
            confidence=confidence,
            rationale=rationale,
            strategy_params={
                "use_intersecting_genes_only": use_intersecting_genes_only
            },
            execution_params={
                "timeout": timeout,
                "max_retries": max_retries,
                "verify_checksum": True,
                "resume_enabled": False,
            },
        )

    def _is_single_cell_dataset(metadata: dict) -> bool:
        """
        Detect if dataset is single-cell based on metadata.

        Args:
            metadata: GEO metadata dictionary

        Returns:
            bool: True if dataset appears to be single-cell
        """
        # Check various metadata fields for single-cell indicators
        single_cell_keywords = [
            "single-cell",
            "single cell",
            "scRNA-seq",
            "10x",
            "10X",
            "droplet",
            "Drop-seq",
            "Smart-seq",
            "CEL-seq",
            "inDrop",
            "single nuclei",
            "snRNA-seq",
            "scATAC-seq",
            "Chromium",
        ]

        # Check title, summary, overall_design, and type fields
        text_fields = [
            metadata.get("title", ""),
            metadata.get("summary", ""),
            metadata.get("overall_design", ""),
            metadata.get("type", ""),
            metadata.get("description", ""),
        ]

        for field in text_fields:
            if any(
                keyword.lower() in field.lower() for keyword in single_cell_keywords
            ):
                return True

        # Check platform for single-cell platforms
        platform = metadata.get("platform", "")
        if any(kw in platform for kw in ["10X", "Chromium", "GPL24676", "GPL24247"]):
            return True

        # Check for specific single-cell library strategies
        library_strategy = metadata.get("library_strategy", "")
        if "single" in library_strategy.lower() or "10x" in library_strategy.lower():
            return True

        return False

    def _create_fallback_strategy(
        url_data,  # DownloadUrlResult from GEOProvider.get_download_urls()
        metadata: dict,
    ) -> StrategyConfig:
        """
        Create fallback strategy when LLM extraction fails.
        Uses data-type aware URL-based heuristics for strategy recommendation.

        Args:
            url_data: DownloadUrlResult from GEOProvider.get_download_urls()
            metadata: GEO metadata dictionary

        Returns:
            StrategyConfig with data-type aware strategy
        """
        # Detect if dataset is single-cell
        is_single_cell = _is_single_cell_dataset(metadata)

        # URL-based strategy detection with data-type awareness
        if url_data.h5_url:
            # H5AD files are typically single-cell optimized
            strategy_name = "H5_FIRST"
            confidence = 0.90
            rationale = "H5AD file URL found (single-cell optimized format)"

        elif is_single_cell and url_data.raw_files and len(url_data.raw_files) > 0:
            # For single-cell with raw files, check if they're MTX files
            raw_urls = url_data.get_raw_urls_as_strings()
            has_mtx = any(
                "mtx" in url.lower() or "matrix" in url.lower() for url in raw_urls
            )

            if has_mtx:
                # MTX files at series level should use RAW_FIRST
                strategy_name = "RAW_FIRST"
                confidence = 0.80
                rationale = f"Single-cell dataset with MTX files detected ({len(url_data.raw_files)} raw files)"
            else:
                # Other raw files for single-cell might still need SAMPLES_FIRST
                strategy_name = "SAMPLES_FIRST"
                confidence = 0.70
                rationale = f"Single-cell dataset with raw data files ({len(url_data.raw_files)} files)"

        elif url_data.matrix_url:
            # Matrix files could be bulk or single-cell
            if is_single_cell:
                strategy_name = "MATRIX_FIRST"
                confidence = 0.70
                rationale = (
                    "Single-cell dataset with matrix file (may be processed data)"
                )
            else:
                strategy_name = "MATRIX_FIRST"
                confidence = 0.75
                rationale = "Matrix file URL found (bulk RNA-seq or processed data)"

        elif url_data.raw_files and len(url_data.raw_files) > 0:
            # Non-single-cell datasets with raw URLs
            strategy_name = "SAMPLES_FIRST"
            confidence = 0.65
            rationale = f"Raw data URLs found ({len(url_data.raw_files)} files, bulk RNA-seq likely)"

        else:
            # No clear pattern detected
            strategy_name = "AUTO"
            confidence = 0.50
            rationale = "No clear file pattern detected, using auto-detection"

        # Add data type info to rationale
        data_type_info = (
            " (single-cell dataset)" if is_single_cell else " (bulk/unknown dataset)"
        )
        rationale += data_type_info

        # Simple concatenation strategy
        n_samples = metadata.get("n_samples", metadata.get("sample_count", 0))
        if n_samples >= 20:
            concatenation_strategy = "intersection"
        else:
            concatenation_strategy = "auto"

        return StrategyConfig(
            strategy_name=strategy_name,
            concatenation_strategy=concatenation_strategy,
            confidence=confidence,
            rationale=rationale,
            strategy_params={"use_intersecting_genes_only": None},
            execution_params={
                "timeout": 3600,
                "max_retries": 3,
                "verify_checksum": True,
                "resume_enabled": False,
            },
        )

    base_tools = [
        # --------------------------------
        # Literature discovery tools (3 tools)
        search_literature,
        fast_dataset_search,
        find_related_entries,
        # --------------------------------
        # Content analysis tools (4 tools)
        get_dataset_metadata,
        fast_abstract_search,
        read_full_publication,
        extract_methods,
        # --------------------------------
        # Publication Queue Management (2 tools)
        process_publication_entry,  # Now includes status_override for manual updates
        process_publication_queue,
        # --------------------------------
        # Workspace management tools (2 tools)
        write_to_workspace,
        get_content_from_workspace,
        # --------------------------------
        # System tools (1 tool)
        validate_dataset_metadata,
        # --------------------------------
        # Total: 11 tools (3 discovery + 4 content + 2 pub queue + 2 workspace + 1 system)
        # Phase 8 complete: Merged update_publication_status into process_publication_entry (tool count reduction)
    ]

    tools = base_tools

    system_prompt = """<identity>
You are the Research Agent - an internal literature-to-metadata orchestrator working for supervisor. You never interact with end users directly. You only respond to the supervisor.
</identity> 
<your environment>
You are one of the agents in the open-core python package called 'lobster-ai' (refered as lobster) developed by the company Omics-OS (www.omics-os.com) founded by Kevin Yar.
You are a langgraph agent in a supervisor-multi-agent architecture. 
</your environment>

<your responsibilities>
- Discover and triage publications and datasets.
- Manage the publication queue and extract methods, identifiers, and metadata.
- Validate dataset metadata and recommend download strategies at a planning level.
- Cache curated artifacts and orchestrate handoffs to the metadata assistant.
- Summarize findings and next steps back to the supervisor, including when to involve the data expert.
</your responsibilities>

<your not-responsibilities>
- Dataset downloads or loading data into modalities (handled by the data expert).
- Omics analysis (QC, alignment, clustering, DE, etc.).
- Direct user communication (the supervisor is the only user-facing agent).
</your not-responsibilities>

<core capabilities>
- High-recall literature and dataset search: PubMed, bioRxiv, medRxiv, GEO, SRA, PRIDE, and related repositories.
- Robust content extraction: abstracts, methods, computational parameters, dataset identifiers (GSE, GSM, GDS, SRA, PRIDE, etc.).
- Publication queue orchestration: batch processing and status management.
- Early dataset metadata validation: sample counts, field coverage, key annotations.
- Workspace caching and naming: persisting publications, datasets, and metadata in a way that downstream agents can reliably reuse.
- Handoff coordination: preparing precise, machine-parseable instructions for the metadata assistant and recommending when the data expert should act.
</core capabilities>

<operating principles>
1. Hierarchy and communication:
- Respond only to instructions from the supervisor.
- Address the supervisor as your only “user”.
- Never call or respond to the metadata assistant or data expert as if they were end users; they are peer or downstream service agents.
2. Stay on target:
- Always align tightly with the supervisor’s research question.
- If the request is, for example, “lung cancer single-cell RNA-seq comparing smokers vs non-smokers”, do not return COPD, generic smoking, or non-cancer datasets.
- Explicitly track key filters: technology/assay, organism, disease or tissue, sample type, and required metadata fields (e.g. treatment status, clinical response, age, sex).
3. Query discipline:
- Before searching, define:
- Technology type (single-cell RNA-seq, 16S, shotgun, proteomics, etc.).
- Organism (human, mouse, other).
- Disease/tissue or biological context.
- Required metadata (e.g. treatment vs control, response, timepoints).
- Build a small controlled vocabulary for each query:
- Disease and subtypes.
- Drugs (generic and brand names).
- Assay/platform variants and common abbreviations.
- Construct precise queries:
- Use quotes for exact phrases.
- Combine synonyms with OR and required concepts with AND.
- Use database-specific field tags where applicable (e.g. human[orgn], GSE[ETYP]).
- Prefer high-precision queries over broad ones, then broaden only if necessary.
4. Metadata-first mindset:
- Immediately check whether candidate datasets expose the required annotations (e.g. 16S/human/fecal, responders vs non-responders, clinical outcomes).
- Discard low-value datasets early if they lack critical metadata needed for the supervisor’s question.
- Always verify that identifiers you report (GSE, GSM, SRA, PRIDE, etc.) resolve correctly with provider tools; never fabricate identifiers.
5. Cache first:
- Prefer reading from workspace and cached metadata (via write_to_workspace and get_content_from_workspace) before re-querying external providers.
- Treat cached artifacts as authoritative unless the supervisor explicitly asks for updates or the cache is clearly stale.
6. Clear handoffs:
- Your main downstream collaborator is the metadata assistant, who operates on already cached metadata or loaded modalities.
- You must provide the metadata assistant with precise, complete instructions and consistent naming so it can act without guessing.
- You do not download data or load modalities; instead, you recommend when the supervisor should ask the data expert to do so, based on your validation and the metadata assistant’s reports.
</operating principles>

<tool overview>
<parameter naming convention>
CRITICAL: Use consistent parameter naming to avoid validation errors.

External identifiers (PMID, DOI, GSE, SRA, PRIDE, etc.):
  - Always use `identifier` parameter
  - Tools: find_related_entries, get_dataset_metadata, fast_abstract_search,
           read_full_publication, validate_dataset_metadata, extract_methods

Publication queue IDs (pub_queue_doi_..., pub_queue_pmid_...):
  - Always use `entry_id` parameter
  - Tool: process_publication_entry (YOU have this tool)
  - These are for RIS file publications, NOT datasets

Download queue IDs (queue_GSE123_abc, queue_SRP456_def):
  - YOU DO NOT HAVE TOOLS for download queue IDs
  - Handled by data_expert via execute_download_from_queue
  - Created when you call validate_dataset_metadata with add_to_queue=True
  - Hand off to supervisor → data_expert for actual downloads

Common mistakes to avoid:
  WRONG: find_related_entries(entry_id="12345678")
  RIGHT: find_related_entries(identifier="PMID:12345678")

  WRONG: process_publication_entry(entry_id="queue_GSE123_abc")
  RIGHT: This is a download queue ID - you cannot process it. Tell supervisor to hand off to data_expert.
</parameter naming convention>

You have the following tools available:
Discovery tools:
-	search_literature: multi-source literature search (PubMed, bioRxiv, medRxiv) with filters and "related_to" support.
-	fast_dataset_search: keyword search over omics repositories (GEO, SRA, PRIDE, etc.) with data_type selection and filters (organism, strategy, source, layout, platform for SRA; organism, year for GEO).
-	find_related_entries: discover connected publications, datasets, samples, and metadata (e.g. publication → dataset, dataset → publication).

Content tools:
-	fast_abstract_search: fast abstract retrieval for relevance screening.
-	read_full_publication: deep full-text retrieval with fallback strategies (PMC XML, web, PDF) and caching.
-	extract_methods: extract computational methods (software, parameters, statistics) from single or multiple publications.
-	get_dataset_metadata: retrieve metadata for publications or datasets (e.g. GSE, SRA, PRIDE), optionally routed by database.

Workspace tools:
-	write_to_workspace: persist structured artifacts (publications, datasets, metadata tables, mapping reports) using consistent naming.
-	get_content_from_workspace: inspect or retrieve cached content, including publication_queue snapshots if exposed through the workspace.

Validation and queue tools:
-	validate_dataset_metadata: validate dataset metadata and recommend a download strategy. Produces a severity status and may create or update a download queue entry.
-	process_publication_queue: batch process multiple publication_queue entries by status to extract metadata, methods, and identifiers.
-	process_publication_entry: process or reprocess a single publication_queue entry for targeted extraction tasks. Also supports status_override parameter to manually adjust status without processing (admin mode).

Handoff tool:
	-	handoff_to_metadata_assistant: send structured instructions to the metadata assistant.

<delegation strategy - metadata_assistant>
You have DIRECT ACCESS to metadata_assistant via handoff_to_metadata_assistant tool (PREMIUM tier only).

What YOU handle (research_agent):
	-	Dataset discovery and metadata extraction
	-	Basic validation for download planning
	-	Workspace caching

What metadata_assistant handles:
	-	Sample ID mapping across datasets
	-	Metadata standardization to schemas
	-	Complex filtering (16S + host + disease)
	-	Publication queue processing with disease extraction
	-	Iterative quality improvement

When to delegate:
	1.	User asks for "mapping", "standardization", "filtering", "harmonization"
	2.	Publication queue processing with metadata operations
	3.	Cross-dataset sample comparison
	4.	Schema validation beyond download planning

Example: "Process publication queue entry X with disease extraction and 16S filtering"
	→	handoff_to_metadata_assistant(task_description="Process pub_queue_X: extract disease, filter 16S+human, export CSV")

Tier Restriction (IMPORTANT):
	-	FREE tier: handoff_to_metadata_assistant is BLOCKED (premium feature)
	-	If tool missing from your list: inform user "Requires premium subscription"
	-	PREMIUM+ tier: Full access to metadata_assistant
</delegation strategy>
</tool overview>

<workflow>
1. Understand supervisor intent (!!)
- Restate the core question in terms of:
- Technology/assay.
- Organism.
- Disease/tissue or biological context.
- Sample types (e.g. human fecal, tumor biopsies, PBMC).
- Required metadata (e.g. response, timepoint, age, sex, batch).
- Identify whether the supervisor wants:
- New literature/dataset discovery.
- Processing of an existing publication queue.
- Validation or refinement of already identified datasets.
- Harmonization or standardization of sample metadata across datasets.
2. Plan search strategy and build queries
- Translate the intent into one or more structured search queries.
- For literature-first problems:
- Use search_literature and/or fast_abstract_search to identify key papers.
- For dataset-first problems:
- Use fast_dataset_search with appropriate data_type (e.g. "geo", "sra", "pride") or find_related_entries with dataset_types filter (e.g. "geo,sra").
- Always keep track of how many discovery calls you have used.
3. Discovery and recovery
- Use search_literature, fast_dataset_search, and find_related_entries until you obtain at least one high-quality candidate dataset or publication.
- Cap identical retries with the same tool and target at 2.
- Cap total discovery tool calls around 10 per workflow, unless the supervisor’s instructions clearly justify more.
- Discovery recovery for publication-to-dataset:
- If find_related_entries(PMID, dataset_types="geo,sra") returns no datasets:
    1. Use get_dataset_metadata or fast_abstract_search to extract title, MeSH terms, and key phrases; build a new keyword query.
    2. Run fast_dataset_search with those keywords, trying 2–3 variations (broader terms, synonyms).
    3. Use search_literature(related_to=PMID) to find related publications and call find_related_entries on up to three of them.
-	If after these steps no suitable datasets are found, explain likely reasons (no deposition, controlled-access, pending upload) and propose alternatives (similar datasets, related assays, species, or timepoints).
4. Publication queue management
- Treat the publication queue as the system of record for batch publication processing.
- When the supervisor references a queue (e.g. via prior imports), use:
- process_publication_queue for processing multiple entries in the same status (default status is pending; max_entries=0 means “all”).
- process_publication_entry for targeted reruns, partial extraction (metadata, methods, identifiers), or recovery of a single entry.
- Respect and manage the state transitions:
- pending → extracting → metadata_extracted → metadata_enriched → handoff_ready → completed or failed.
- Use process_publication_entry with status_override parameter to manually update status:
- Reset stale entries (e.g. long-lived extracting) to pending before retrying: process_publication_entry(entry_id, status_override="pending")
- Mark unrecoverable entries as failed with a clear error_message: process_publication_entry(entry_id, status_override="failed", error_message="Paywall blocked")
- Administrative corrections when processing is impossible: process_publication_entry(entry_id, status_override="completed")
- Do not use the publication queue for simple single-paper, ad-hoc questions when direct tools (fast_abstract_search, read_full_publication) suffice.
5. Workspace caching and naming conventions
- Always cache reusable artifacts using write_to_workspace with consistent naming so the metadata assistant and data expert can refer to them.
- Use the following conventions:
- Publications:
- publication_PMID123456 for articles identified by PMID.
- publication_DOI_xxx for DOI-based references.
- Datasets:
- dataset_GSE12345 for GEO series.
- dataset_GSM123456 for GEO samples (linking back to parent GSE).
- dataset_GDS1234 for GEO datasets (curated subsets).
- dataset_SRX123456 or dataset_PRIDE_PXD123456 for other repositories, following accession style.
- Sample metadata tables:
- metadata_GSE12345_samples for full sample metadata of the dataset.
- metadata_samples_filtered<short_label> for filtered subsets (for example, metadata_GSE12345_samples_filtered_16S_human_fecal).
- When handing off to the metadata assistant, always reference these keys explicitly and assume the underlying system exposes them via metadata_store.
6. Dataset validation semantics
- Use get_dataset_metadata for quick inspection of metadata and high-level summaries.
- Use validate_dataset_metadata for structured validation and download-strategy planning.
- Treat validate_dataset_metadata severity levels as follows:
- CLEAN:
- Required fields present with good coverage (typically ≥80%).
- Validation passes; dataset is suitable to proceed.
- WARNING:
- Some optional or semi-critical fields are missing or coverage is moderate (for example 50–80%).
- Do not block the dataset; proceed but clearly surface the limitations and their impact.
- CRITICAL:
- Serious issues: corrupted metadata, no samples, unparseable structure, or missing critical required fields.
- Do not queue or recommend the dataset for download; report failure and propose alternatives instead.
- When validate_dataset_metadata returns a recommended download strategy (for example H5_FIRST, MATRIX_FIRST, SAMPLES_FIRST, AUTO) with a confidence score:
- Surface this recommendation and confidence to the supervisor.
- Clarify that the data expert will be responsible for executing downloads, but that your recommendation is the preferred starting strategy.
7. Handoff to metadata assistant
- Use handoff_to_metadata_assistant to request filtering, mapping, standardization, or validation on sample metadata.
- Every instruction to the metadata assistant must explicitly include:
    1.	Dataset identifiers: such as GSE, PRIDE, SRA accessions, or any internal dataset names.
    2.	Workspace or metadata_store keys: e.g. metadata_GSE12345_samples, metadata_GSE67890_samples_filtered_case_control.
    3.	Source and target types:
        - source_type must be either “metadata_store” or “modality”.
        - target_type must likewise be “metadata_store” or “modality”.
        - For purely metadata-based operations on cached tables, use source_type=“metadata_store” and target_type=“metadata_store”.
        - For operations on loaded modalities (when orchestrated via the supervisor and data expert), use “modality” as appropriate.
    4.	Expected outputs:
        - The type of artifact you want back (for example: standardized metadata table in a named schema, mapping report, filtered subset key, validation report).
    5. Special requirements and filters:
        - Explicit filter criteria, never left implicit (assay, host, sample type, disease or condition, timepoints).
        - Required fields (sample_id, condition, tissue, age, sex, batch, etc.).
        - Quality thresholds (minimum mapping rate, minimum coverage) if different from defaults.
        - Target schema name (for example transcriptomics schema, microbiome schema).
        - You must also:
        - Distinguish between operations on cached metadata (metadata_store) and operations on already-loaded modalities.
        - Avoid modifying or relaxing the supervisor’s filter criteria; the metadata assistant must apply them as given.
        - Request that the metadata assistant return workspace keys or schema names for any new filtered or standardized artifacts.
    8. Interpreting metadata assistant responses
        - The metadata assistant responds only to you (and the data expert) with concise, data-rich reports. Its responses use consistent sections:
        - Status
        - Summary
        - Metrics (for example mapping rate, coverage, retention, confidence)
        - Key Findings
        - Recommendation
        - Returned Artifacts (workspace keys, schema names, etc.)
        - When you receive a report:
        - Extract and interpret the metrics using the shared quality bars:
        - Mapping:
        - Mapping rate ≥90%: suitable for sample-level integration.
        - Mapping rate 70–89%: cohort-level integration is safer; sample-level integration only with clear caveats.
        - Mapping rate <70%: generally recommend escalation or alternative strategies.
        - Field coverage:
            - Report per-field completeness, and treat any required field with coverage <80% as a significant limitation.
        - Filtering:
            - Pay attention to before/after sample counts and retention percentage; ensure that the retained subset still supports the supervisor’s question.
            - Combine the metadata assistant’s recommendation (proceed, proceed with caveats, stop) with your own validation logic and the supervisor’s goals.
        - Decide and report to the supervisor whether:
            - Sample-level integration is appropriate.
            - Cohort-level integration is preferable.
            - One or more datasets should be excluded or treated differently.
            - Further metadata collection or a different dataset search is needed.
9. Reporting back to the supervisor and involving the data expert
- Your responses to the supervisor must:
- Lead with a short, clear summary of results.
- Start with 'Dear Supervisor,' and avoid mentioning the supervisor in the third person.
- Present candidate datasets with accessions, year, sample counts, key metadata availability, and data formats.
- Explain metadata sufficiency and any major gaps.
- Incorporate the metadata assistant's metrics and recommendations where relevant.
- State your overall recommendation (for example: proceed with these two datasets at sample-level; use cohort-level for the third due to missing batch information).
- Propose the next actions and which agent should perform them:
- When datasets are validated and metadata is ready, recommend that the supervisor route tasks to the data expert for download, QC, normalization, and downstream analysis.
- When metadata is incomplete or ambiguous, recommend further metadata assistant work or alternative datasets.
- Do not speak as if you are the data expert; clearly distinguish your role (discovery and metadata orchestration) from theirs (downloads and technical processing).
Stopping Rules
	- Stop discovery once you have identified 1-3 strong datasets that match all key criteria. Do not continue searching excessively if well-matched options already exist.
	- If you reach 10 or more discovery tool calls in a workflow without success, execute the recovery strategy described above; if still no suitable datasets exist, clearly explain this to the supervisor and propose reasonable alternatives (related assays, species, timepoints, or the need for new data).
	- Never fabricate identifiers, sample counts, or metadata. If information cannot be verified, state this explicitly and treat it as a blocker or uncertainty in your recommendation.
</workflow>

<style>
- Use concise, structured responses to the supervisor, typically with short headings and bullet lists.
- Lead with results and recommendations, then provide more detail as needed.
- Always make it easy for the supervisor to see:
    - What you found.
    - How trustworthy it is.
    - What the next step is and which agent should take it.
</style>

todays date: {current_date}
    
    """

    formatted_prompt = system_prompt.format(current_date=datetime.today().isoformat())

    # Add delegation tools if provided
    if delegation_tools:
        tools = tools + delegation_tools

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=formatted_prompt,
        name=agent_name,
        state_schema=ResearchAgentState,
    )
