"""
SRA Provider for Sequence Read Archive dataset discovery.

This provider wraps pysradb to enable search and metadata retrieval
from NCBI's Sequence Read Archive (SRA), focusing on transcriptomics
and metagenomics sequencing data.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetType,
    ProviderCapability,
    PublicationMetadata,
    PublicationSource,
)
from lobster.tools.providers.biopython_entrez_wrapper import BioPythonEntrezWrapper
from lobster.tools.providers.ncbi_query_builder import NCBIDatabase, NCBIQueryBuilder

logger = logging.getLogger(__name__)


class SRAProviderConfig(BaseModel):
    """Configuration for SRA provider."""

    max_results: int = Field(default=20, ge=1, le=100000)
    email: str = Field(default_factory=lambda: get_settings().NCBI_EMAIL)
    api_key: Optional[str] = None
    expand_attributes: bool = Field(
        default=False,
        description="Auto-expand sample attributes from semicolon-delimited format",
    )
    batch_size: int = Field(
        default=10000,
        ge=1,
        le=10000,
        description="Batch size for NCBI API requests (max 10,000 per NCBI limit)",
    )
    enable_quality_filters: bool = Field(
        default=True,
        description="Apply quality filters to improve result relevance (public, has data, etc.)",
    )


class SRAProviderError(Exception):
    """Base exception for SRAProvider errors."""

    pass


class SRANotFoundError(SRAProviderError):
    """SRA accession not found."""

    pass


class SRAConnectionError(SRAProviderError):
    """Failed to connect to SRA/NCBI."""

    pass


class SRAProvider(BasePublicationProvider):
    """
    SRA provider for comprehensive sequence data discovery via pysradb.

    This provider wraps pysradb to provide:
    - SRA database search for sequencing datasets
    - Metadata retrieval with sample attribute expansion
    - Accession conversion (PMID↔SRP, GSE↔SRP)
    - Publication-to-dataset linking

    Supports SRA accessions: SRP (study), SRX (experiment), SRS (sample), SRR (run)
    """

    def __init__(
        self, data_manager: DataManagerV2, config: Optional[SRAProviderConfig] = None
    ):
        """
        Initialize SRAProvider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional provider configuration
        """
        self.data_manager = data_manager
        self.config = config or SRAProviderConfig()
        self._sraweb = None  # Lazy initialization for pysradb.SRAweb

        # Phase 1.2: Biopython Bio.Entrez wrapper for NCBI API
        self.entrez_wrapper = BioPythonEntrezWrapper(
            email=self.config.email,
            api_key=self.config.api_key
        )
        self.query_builder = NCBIQueryBuilder(NCBIDatabase.SRA)

        logger.info("SRAProvider initialized with Biopython Bio.Entrez support")

    def _get_sraweb(self):
        """Lazy initialization of pysradb SRAweb connection."""
        if self._sraweb is None:
            try:
                from pysradb import SRAweb

                self._sraweb = SRAweb()
                logger.debug("pysradb SRAweb connection initialized")
            except ImportError as e:
                raise SRAConnectionError(
                    "pysradb not installed. Install with: pip install pysradb"
                ) from e
            except Exception as e:
                raise SRAConnectionError(f"Failed to initialize SRAweb: {e}") from e
        return self._sraweb

    def _apply_sra_filters(self, query: str, filters: Dict[str, str]) -> str:
        """
        Apply SRA-specific field qualifiers to query (PubMedProvider pattern).

        This method accepts RAW queries from agents and only adds structured
        SRA field qualifiers for filters. The agent is responsible for
        constructing the query (with OR, AND, etc.).

        Args:
            query: Raw query string from agent
            filters: Filter dict with keys: organism, strategy, source, layout, platform

        Returns:
            str: Query with SRA field qualifiers appended

        Examples:
            >>> _apply_sra_filters("gut microbiome", {"organism": "Homo sapiens", "strategy": "AMPLICON"})
            '(gut microbiome) AND (Homo sapiens[ORGN]) AND (AMPLICON[STRA])'

            >>> _apply_sra_filters("microbiome OR 16S", {"organism": "Homo sapiens"})
            '(microbiome OR 16S) AND (Homo sapiens[ORGN])'
        """
        # Safety wrapper (PubMedProvider line 858 pattern)
        filtered_query = f"({query})"

        # SRA field qualifiers (similar to PubMed [PDAT], [JOUR], etc.)
        if "organism" in filters and filters["organism"]:
            organism = filters["organism"]
            # Quote multi-word values
            if " " in organism:
                filtered_query += f' AND ("{organism}"[ORGN])'
            else:
                filtered_query += f" AND ({organism}[ORGN])"

        if "strategy" in filters and filters["strategy"]:
            strategy = filters["strategy"]
            filtered_query += f" AND ({strategy}[STRA])"

        if "source" in filters and filters["source"]:
            source = filters["source"]
            filtered_query += f" AND ({source}[SRC])"

        if "layout" in filters and filters["layout"]:
            layout = filters["layout"]
            filtered_query += f" AND ({layout}[LAY])"

        if "platform" in filters and filters["platform"]:
            platform = filters["platform"]
            filtered_query += f" AND ({platform}[PLAT])"

        logger.debug(f"Applied SRA filters: {query} → {filtered_query}")
        return filtered_query

    def _apply_quality_filters(self, query: str, modality_hint: Optional[str] = None) -> str:
        """
        Apply quality filters to improve result relevance.

        Phase 1.4: Hardcoded quality filters per modality to ensure:
        - Public datasets only
        - Datasets with actual data files
        - Modality-specific technical filters

        Args:
            query: Query string (with or without existing filters)
            modality_hint: Optional modality hint to apply specific filters
                          ("scrna-seq", "bulk-rna-seq", "amplicon", "16s")

        Returns:
            str: Query with quality filters applied

        Examples:
            >>> _apply_quality_filters("microbiome", modality_hint="amplicon")
            'microbiome AND "public"[Access] AND "has data"[Properties] AND "strategy amplicon"[Filter]'

            >>> _apply_quality_filters("cancer", modality_hint="scrna-seq")
            'cancer AND "public"[Access] AND "has data"[Properties] AND "library layout paired"[Filter] AND "platform illumina"[Filter]'
        """
        if not self.config.enable_quality_filters:
            logger.debug("Quality filters disabled in config")
            return query

        # Base quality filters (apply to all modalities)
        quality_query = query
        quality_query += ' AND "public"[Access]'
        quality_query += ' AND "has data"[Properties]'

        # Modality-specific filters
        if modality_hint:
            hint_lower = modality_hint.lower()

            if "scrna" in hint_lower or "single-cell" in hint_lower or "singlecell" in hint_lower:
                # scRNA-seq: prefer paired-end Illumina for quality
                quality_query += ' AND "library layout paired"[Filter]'
                quality_query += ' AND "platform illumina"[Filter]'
                logger.debug("Applied scRNA-seq quality filters")

            elif "bulk" in hint_lower and "rna" in hint_lower:
                # Bulk RNA-seq: just base filters (already applied)
                logger.debug("Applied bulk RNA-seq quality filters (base only)")

            elif "amplicon" in hint_lower or "16s" in hint_lower:
                # Amplicon (16S): ensure AMPLICON strategy
                quality_query += ' AND "strategy amplicon"[Filter]'
                logger.debug("Applied amplicon/16S quality filters")

        logger.info(f"Quality filters applied: {query} → {quality_query}")
        return quality_query

    def _ncbi_esearch(
        self, query: str, filters: Dict[str, Any], max_results: int
    ) -> List[str]:
        """
        Execute NCBI esearch to get SRA IDs using Biopython Bio.Entrez.

        Implements pagination for large result sets (>10,000 records) by making
        multiple requests with retstart offsets.

        Args:
            query: NCBI query string
            filters: Additional filters (for logging)
            max_results: Maximum number of results to retrieve

        Returns:
            List[str]: List of SRA IDs

        Raises:
            SRAProviderError: If search fails
        """
        try:
            logger.info(f"NCBI esearch query: {query}")
            logger.debug(f"Max results requested: {max_results}")

            # First request to get total count
            initial_result = self.entrez_wrapper.esearch(
                db="sra",
                term=query,
                retmax=min(max_results, self.config.batch_size),
                retstart=0
            )

            # Extract total count and initial IDs
            total_count = int(initial_result.get("Count", 0))
            all_ids = initial_result.get("IdList", [])

            logger.info(f"Found {total_count:,} total results in NCBI SRA")

            # If we need more results and there are more available, paginate
            if len(all_ids) < max_results and len(all_ids) < total_count:
                # Calculate how many more results we need
                remaining = min(max_results - len(all_ids), total_count - len(all_ids))

                # Calculate number of additional batches needed
                num_batches = (remaining + self.config.batch_size - 1) // self.config.batch_size

                logger.info(
                    f"Fetching {remaining:,} more results in {num_batches} batch(es) "
                    f"(batch_size={self.config.batch_size:,})"
                )

                # Fetch remaining batches
                for batch_num in range(num_batches):
                    offset = len(all_ids)
                    batch_size = min(self.config.batch_size, remaining - (batch_num * self.config.batch_size))

                    logger.debug(
                        f"Fetching batch {batch_num + 1}/{num_batches}: "
                        f"offset={offset:,}, size={batch_size:,}"
                    )

                    batch_result = self.entrez_wrapper.esearch(
                        db="sra",
                        term=query,
                        retmax=batch_size,
                        retstart=offset
                    )

                    batch_ids = batch_result.get("IdList", [])
                    all_ids.extend(batch_ids)

                    logger.debug(f"Batch {batch_num + 1} returned {len(batch_ids)} IDs")

                    # Stop if we got fewer IDs than expected (end of results)
                    if len(batch_ids) < batch_size:
                        break

            logger.info(f"Retrieved {len(all_ids):,} SRA IDs (requested: {max_results:,}, available: {total_count:,})")
            return all_ids

        except Exception as e:
            logger.error(f"NCBI esearch error: {e}")
            raise SRAProviderError(f"NCBI esearch failed: {str(e)}") from e

    def _ncbi_esummary(self, sra_ids: List[str]) -> pd.DataFrame:
        """
        Fetch metadata for SRA IDs using NCBI esummary with Biopython Bio.Entrez.

        Args:
            sra_ids: List of SRA ID numbers from esearch

        Returns:
            pd.DataFrame: DataFrame with SRA metadata

        Raises:
            SRAProviderError: If metadata retrieval fails
        """
        if not sra_ids:
            return pd.DataFrame()

        try:
            import xmltodict

            logger.debug(f"NCBI esummary for {len(sra_ids)} IDs")

            # Fetch via Biopython wrapper (returns raw XML for SRA)
            # Note: Bio.Entrez.read() doesn't parse SRA's complex XML-in-XML structure,
            # so we need to parse it manually with xmltodict
            xml_content = self.entrez_wrapper.efetch(
                db="sra",
                id=",".join(sra_ids),
                rettype="docsum",
                retmode="xml"
            )

            # Parse XML response
            result = xmltodict.parse(xml_content)

            # Handle None result (empty/invalid XML)
            if result is None:
                logger.warning(f"NCBI esummary returned None for {len(sra_ids)} IDs")
                return pd.DataFrame()

            # Extract DocSum entries with safe nested access
            esummary_result = result.get("eSummaryResult")
            if esummary_result is None:
                logger.warning(f"No eSummaryResult in response for {len(sra_ids)} IDs")
                return pd.DataFrame()

            doc_summaries = esummary_result.get("DocSum", [])

            # Handle single result case
            if isinstance(doc_summaries, dict):
                doc_summaries = [doc_summaries]

            # Parse into DataFrame
            records = []
            for doc in doc_summaries:
                if not isinstance(doc, dict):
                    continue

                # Extract ExpXml and Runs from Item list
                exp_xml = self._extract_item_content(doc, "ExpXml")
                runs_xml = self._extract_item_content(doc, "Runs")

                # Initialize all fields
                study_title = pd.NA
                organism = pd.NA
                library_strategy = pd.NA
                library_layout = pd.NA
                instrument_platform = pd.NA
                study_accession = pd.NA
                experiment_accession = pd.NA
                run_accession = pd.NA
                total_runs = "1"
                total_size = pd.NA

                # Parse ExpXml for metadata
                if exp_xml:
                    try:
                        # Wrap in root element to handle multiple top-level elements
                        wrapped_xml = f"<Root>{exp_xml}</Root>"
                        exp_data = xmltodict.parse(wrapped_xml)
                        root = exp_data.get("Root", {})

                        # Extract from Summary
                        summary = root.get("Summary", {})
                        study_title = summary.get("Title", pd.NA)

                        # Platform - text content, not attribute
                        platform = summary.get("Platform", {})
                        if isinstance(platform, dict):
                            instrument_platform = platform.get("#text", pd.NA)
                        elif isinstance(platform, str):
                            instrument_platform = platform
                        else:
                            instrument_platform = pd.NA

                        # Extract accessions from top-level elements
                        study = root.get("Study", {})
                        if isinstance(study, dict):
                            study_accession = study.get("@acc", pd.NA)

                        experiment = root.get("Experiment", {})
                        if isinstance(experiment, dict):
                            experiment_accession = experiment.get("@acc", pd.NA)

                        # Organism
                        org = root.get("Organism", {})
                        if isinstance(org, dict):
                            organism = org.get("@ScientificName", pd.NA)

                        # Library descriptor
                        lib_desc = root.get("Library_descriptor", {})
                        if isinstance(lib_desc, dict):
                            library_strategy = lib_desc.get("LIBRARY_STRATEGY", pd.NA)
                            # Layout can be a dict with keys like "PAIRED" or "SINGLE"
                            layout = lib_desc.get("LIBRARY_LAYOUT", {})
                            if isinstance(layout, dict):
                                # Get first key (PAIRED or SINGLE)
                                library_layout = (
                                    list(layout.keys())[0] if layout else pd.NA
                                )
                            else:
                                library_layout = layout

                    except Exception as e:
                        logger.debug(f"Error parsing ExpXml: {e}")

                # Parse Runs for run_accession if available
                if runs_xml:
                    try:
                        # Wrap in root element to handle multiple top-level elements
                        wrapped_xml = f"<Root>{runs_xml}</Root>"
                        runs_data = xmltodict.parse(wrapped_xml)
                        root = runs_data.get("Root", {})
                        run_list = root.get("Run", [])

                        # Handle single run case
                        if isinstance(run_list, dict):
                            run_list = [run_list]

                        if run_list:
                            first_run = run_list[0]
                            # Extract accessions from first run
                            run_accession = first_run.get("@acc", pd.NA)
                            experiment_accession = first_run.get("@exp_acc", pd.NA)
                            study_accession = first_run.get("@study_acc", pd.NA)
                            total_size = first_run.get("@total_bases", pd.NA)

                            total_runs = str(len(run_list))

                    except Exception as e:
                        logger.debug(f"Error parsing Runs: {e}")

                # Build record
                record = {
                    "study_accession": study_accession,
                    "experiment_accession": experiment_accession,
                    "run_accession": run_accession,
                    "study_title": study_title,
                    "organism": organism,
                    "library_strategy": library_strategy,
                    "library_layout": library_layout,
                    "instrument_platform": instrument_platform,
                    "total_runs": total_runs,
                    "total_size": total_size,
                }

                records.append(record)

            logger.debug(f"Parsed {len(records)} metadata records from esummary")
            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"NCBI esummary error: {e}")
            raise SRAProviderError(f"NCBI esummary failed: {str(e)}") from e

    def _extract_item_content(self, doc: Dict, item_name: str) -> Optional[str]:
        """
        Extract and unescape XML content from NCBI <Item> element.

        NCBI SRA esummary returns structure:
        <DocSum>
            <Item Name="ExpXml" Type="String">&lt;Summary&gt;...&lt;/Summary&gt;</Item>
            <Item Name="Runs" Type="String">&lt;Run acc="..."&gt;...&lt;/Run&gt;</Item>
        </DocSum>

        Args:
            doc: DocSum dict from xmltodict
            item_name: Name attribute to look for (e.g., "ExpXml", "Runs")

        Returns:
            Unescaped XML string or None if not found
        """
        try:
            import html

            # Get Item list from DocSum
            items = doc.get("Item", [])

            # Handle single Item case
            if isinstance(items, dict):
                items = [items]

            # Find Item with matching Name
            for item in items:
                if not isinstance(item, dict):
                    continue

                if item.get("@Name") == item_name:
                    # Extract text content
                    content = item.get("#text", "")

                    # Unescape HTML entities: &lt; → <, &gt; → >, etc.
                    if content:
                        return html.unescape(content)

            return None

        except Exception as e:
            logger.debug(f"Error extracting Item '{item_name}': {e}")
            return None

    def _extract_xml_field(
        self, doc: Dict, *path, prefix: Optional[str] = None, default: str = ""
    ) -> str:
        """
        Safely extract nested field from XML document.

        Args:
            doc: XML document as dict
            *path: Nested path to field
            prefix: Optional prefix filter (e.g., "SRP" to find study accession)
            default: Default value if field not found

        Returns:
            str: Extracted value or default
        """
        try:
            current = doc
            for key in path:
                if isinstance(current, dict):
                    current = current.get(key, {})
                elif isinstance(current, list) and len(current) > 0:
                    current = current[0].get(key, {})
                else:
                    return default

            # Handle prefix filtering for accessions
            if prefix and isinstance(current, str) and current.startswith(prefix):
                return current
            elif prefix:
                return default

            return str(current) if current else default

        except Exception:
            return default

    def _format_no_results(
        self, query: str, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format a "no results" message for failed searches.

        Args:
            query: Search query string
            filters: Optional filters that were applied

        Returns:
            str: Formatted markdown message
        """
        response = "## No SRA Results Found\n\n"
        response += f"**Query**: `{query}`\n"
        if filters:
            response += f"**Filters**: {filters}\n"
        response += "\n"
        response += "No datasets found matching your search criteria. Try:\n"
        response += "- Broadening your search terms\n"
        response += "- Adjusting or removing filters\n"
        response += "- Checking organism spelling (use scientific names: 'Homo sapiens' not 'human')\n"
        response += "- Using different keywords or strategies\n"
        return response

    @property
    def source(self) -> PublicationSource:
        """SRA uses NCBI infrastructure."""
        return PublicationSource.SRA

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return supported dataset types."""
        return [DatasetType.SRA]

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        SRA has high priority (10) as the authoritative NCBI source for
        sequencing data. Provides comprehensive dataset discovery, metadata
        extraction, and accession conversion for RNA-seq and metagenomics.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by SRA provider.

        SRA excels at sequencing dataset discovery, metadata extraction,
        and accession conversion (PMID↔SRP, GSE↔SRP). It's the authoritative
        source for raw sequencing data but doesn't provide literature search
        or full-text access.

        Supported capabilities:
        - DISCOVER_DATASETS: Search SRA database for sequencing datasets
        - FIND_LINKED_DATASETS: Link publications to SRA studies (PMID→SRP)
        - EXTRACT_METADATA: Parse SRA metadata with sample attributes
        - QUERY_CAPABILITIES: Dynamic capability discovery

        Not supported:
        - VALIDATE_METADATA: No schema validation (use metadata_assistant)
        - SEARCH_LITERATURE: No publication search (use PubMedProvider)
        - GET_ABSTRACT: No abstract retrieval
        - GET_FULL_CONTENT: No full-text access
        - EXTRACT_METHODS: No methods extraction
        - EXTRACT_PDF: No PDF processing
        - INTEGRATE_MULTI_OMICS: SRA is single-modality (sequencing only)

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        return {
            ProviderCapability.DISCOVER_DATASETS: True,
            ProviderCapability.FIND_LINKED_DATASETS: True,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.VALIDATE_METADATA: False,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.GET_ABSTRACT: False,
            ProviderCapability.GET_FULL_CONTENT: False,
            ProviderCapability.EXTRACT_METHODS: False,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def search_publications(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Search SRA database using pysradb or direct NCBI API.

        This method accepts RAW queries from agents and only adds structured
        SRA field qualifiers for filters. The agent is responsible for
        constructing the query (with OR, AND, etc.).

        Supports BOTH:
        - Accession-based lookup (e.g., "SRP033351")
        - Keyword search (e.g., "gut microbiome", "microbiome OR 16S")

        Args:
            query: Raw search query string (e.g., "gut microbiome", "SRP033351")
            max_results: Maximum number of results (default: 5)
            filters: Optional search filters:
                - organism: str (e.g., "Homo sapiens")
                - strategy: str (e.g., "RNA-Seq", "AMPLICON")
                - layout: str (e.g., "PAIRED", "SINGLE")
                - source: str (e.g., "TRANSCRIPTOMIC", "METAGENOMIC")
                - platform: str (e.g., "ILLUMINA", "PACBIO")
            **kwargs: Additional parameters
                - detailed: bool = True (fetch detailed metadata)
                - modality_hint: str = None (apply modality-specific quality filters:
                                            "scrna-seq", "bulk-rna-seq", "amplicon", "16s")

        Returns:
            str: Formatted search results

        Raises:
            SRAProviderError: If search fails

        Examples:
            # Agent constructs query with OR logic if needed
            >>> search_publications("microbiome OR 16S")

            # Agent passes raw query, provider adds structured filters
            >>> search_publications("gut microbiome", filters={"organism": "Homo sapiens"})

            # Direct accession lookup
            >>> search_publications("SRP033351")
        """
        try:
            db = self._get_sraweb()

            # Check if query is SRA accession pattern
            if self._is_sra_accession(query):
                # Direct metadata retrieval for accession
                detailed = kwargs.get("detailed", True)
                df = db.sra_metadata(query, detailed=detailed)

                # Handle None (invalid accession) or empty DataFrame
                if df is None or df.empty:
                    return (
                        f"## No SRA Results Found\n\n"
                        f"**Query**: `{query}`\n\n"
                        f"No metadata found for this SRA accession. "
                        f"Verify the accession is valid and publicly available."
                    )

                # Enrich metadata with organism/platform from NCBI if missing (Bug #2 fix)
                df = self._enrich_accession_metadata(df, query)

                # Apply filters to results if provided
                if filters and df is not None and not df.empty:
                    df = self._apply_filters(df, filters)

                return self._format_search_results(df, query, max_results, filters)
            else:
                # Path 2: Direct NCBI esearch (PubMedProvider pattern)
                logger.info(
                    f"[Phase 1] Performing direct NCBI SRA search: {query[:50]}..."
                )

                try:
                    # Apply SRA filters to raw query (no keyword splitting!)
                    if filters:
                        ncbi_query = self._apply_sra_filters(query, filters)
                    else:
                        ncbi_query = query  # Use raw query as-is

                    # Phase 1.4: Apply quality filters (public, has data, etc.)
                    modality_hint = kwargs.get("modality_hint", None)
                    ncbi_query = self._apply_quality_filters(ncbi_query, modality_hint=modality_hint)

                    # Execute esearch to get SRA IDs
                    sra_ids = self._ncbi_esearch(ncbi_query, filters or {}, max_results)

                    if not sra_ids:
                        return self._format_no_results(query, filters)

                    # Fetch metadata via esummary
                    df = self._ncbi_esummary(sra_ids)

                    if df.empty:
                        return self._format_no_results(query, filters)

                    # Apply additional DataFrame-level filters (reuse existing _apply_filters)
                    if filters:
                        df = self._apply_filters(df, filters)

                    # Format and return results
                    return self._format_search_results(df, query, max_results, filters)

                except SRAProviderError:
                    # Re-raise our own errors
                    raise
                except Exception as e:
                    logger.error(f"Direct NCBI SRA search error: {e}")
                    raise SRAProviderError(
                        f"Error searching SRA via NCBI API: {str(e)}"
                    ) from e

        except SRAProviderError:
            raise
        except Exception as e:
            logger.error(f"SRA search error: {e}")
            raise SRAProviderError(f"Error searching SRA: {str(e)}") from e

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find SRA datasets linked to a publication using pysradb.

        Args:
            identifier: PMID, DOI, or PMC ID
            dataset_types: Optional list of dataset types to filter
            **kwargs: Additional parameters

        Returns:
            str: Formatted string with linked SRA datasets

        Raises:
            SRAProviderError: If linking fails
        """
        try:
            db = self._get_sraweb()

            # Clean identifier
            clean_id = identifier.strip()

            # Determine identifier type and query
            if clean_id.startswith("PMID:") or clean_id.isdigit():
                # PubMed ID
                pmid = clean_id.replace("PMID:", "").strip()
                df = db.pubmed_to_srp(pmid)
                id_type = "PMID"
                id_value = pmid
            elif clean_id.startswith("10."):
                # DOI - pysradb doesn't have direct DOI support
                return (
                    f"## SRA Provider: DOI Linking\n\n"
                    f"**DOI**: {clean_id}\n\n"
                    f"ℹ️ DOI-based SRA linking is not directly supported by pysradb. "
                    f"Consider:\n\n"
                    f"1. **Convert DOI to PMID first** using PubMedProvider's `search_pubmed_by_doi()`\n"
                    f"2. **Use PMID to find SRA datasets** via this method\n\n"
                    f"**Example workflow**:\n"
                    f"```\n"
                    f"# Step 1: DOI → PMID\n"
                    f"pmid = pubmed_provider.search_pubmed_by_doi('{clean_id}')\n\n"
                    f"# Step 2: PMID → SRA\n"
                    f"sra_datasets = sra_provider.find_datasets_from_publication(pmid)\n"
                    f"```"
                )
            elif clean_id.startswith("PMC"):
                # PMC ID
                return (
                    f"## SRA Provider: PMC Linking\n\n"
                    f"**PMC ID**: {clean_id}\n\n"
                    f"ℹ️ PMC-based SRA linking requires conversion to PMID first. "
                    f"Use PubMedProvider to convert PMC to PMID.\n\n"
                    f"**Example workflow**:\n"
                    f"```\n"
                    f"# Step 1: PMC → PMID\n"
                    f"pmid = pubmed_provider.pmc_to_pmid('{clean_id}')\n\n"
                    f"# Step 2: PMID → SRA\n"
                    f"sra_datasets = sra_provider.find_datasets_from_publication(pmid)\n"
                    f"```"
                )
            else:
                return (
                    f"## Unsupported Identifier Format\n\n"
                    f"**Identifier**: {identifier}\n\n"
                    f"ℹ️ SRAProvider supports:\n"
                    f"- **PMID**: Numeric ID or 'PMID:12345678'\n"
                    f"- **DOI**: Requires conversion to PMID first (see guidance above)\n"
                    f"- **PMC ID**: Requires conversion to PMID first (see guidance above)"
                )

            # Format results
            if df.empty:
                return (
                    f"## No SRA Datasets Found\n\n"
                    f"**{id_type}**: {id_value}\n\n"
                    f"No SRA datasets are linked to this publication in NCBI's database. "
                    f"This could mean:\n\n"
                    f"1. The study did not deposit sequencing data to SRA\n"
                    f"2. Data was deposited to a different repository (e.g., GEO, ENA, DDBJ)\n"
                    f"3. The link has not been established in NCBI's database yet\n\n"
                    f"💡 **Tip**: Try searching GEO for this publication - many studies have both GEO and SRA accessions."
                )

            response = "## 🧬 SRA Datasets Linked to Publication\n\n"
            response += f"**{id_type}**: {id_value}\n"
            response += f"**Total Datasets**: {len(df)}\n\n"

            for idx, row in df.iterrows():
                srp = row.get("study_accession", "Unknown")
                title = row.get("study_title", "No title available")

                response += (
                    f"### {idx + 1}. [{srp}](https://www.ncbi.nlm.nih.gov/sra/{srp})\n"
                )
                response += f"**Title**: {title}\n"

                # Add key metadata if available
                if "organism" in row and row["organism"]:
                    response += f"**Organism**: {row['organism']}\n"
                if "instrument_platform" in row and row["instrument_platform"]:
                    response += f"**Platform**: {row['instrument_platform']}\n"
                if "library_strategy" in row and row["library_strategy"]:
                    response += f"**Strategy**: {row['library_strategy']}\n"
                if "total_runs" in row and row["total_runs"]:
                    response += f"**Total Runs**: {row['total_runs']}\n"

                response += "\n"

            return response

        except Exception as e:
            logger.error(f"Error finding SRA datasets for {identifier}: {e}")
            raise SRAProviderError(f"Error finding linked datasets: {str(e)}") from e

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract metadata for SRA accession using pysradb.

        Returns dataset metadata formatted as PublicationMetadata (following GEOProvider pattern).

        Args:
            identifier: SRA accession (SRP, SRX, SRS, SRR)
            **kwargs: Additional parameters
                - detailed: bool = True (fetch detailed metadata)
                - expand_attributes: bool = False (parse sample attributes)

        Returns:
            PublicationMetadata: Dataset info mapped to publication fields

        Raises:
            SRANotFoundError: If accession not found
            SRAProviderError: If extraction fails
        """
        try:
            db = self._get_sraweb()

            # Get metadata
            detailed = kwargs.get("detailed", True)
            df = db.sra_metadata(identifier, detailed=detailed)

            if df.empty:
                raise SRANotFoundError(f"SRA accession not found: {identifier}")

            # Extract metadata from first row (or aggregate if multiple runs)
            row = df.iloc[0]

            # Build title from available fields
            title = (
                row.get("study_title")
                or row.get("experiment_title")
                or f"SRA Dataset {identifier}"
            )

            # Build abstract from available descriptions
            abstract = (
                row.get("study_abstract")
                or row.get("experiment_desc")
                or row.get("sample_attribute", "")
            )

            # Extract publication date (multiple possible fields)
            published = (
                row.get("published") or row.get("received") or row.get("updated") or ""
            )

            # Build keywords from key metadata fields
            keywords = []
            if row.get("organism"):
                keywords.append(row["organism"])
            if row.get("instrument_platform") or row.get("platform"):
                keywords.append(row.get("instrument_platform", row.get("platform", "")))
            if row.get("library_strategy"):
                keywords.append(row["library_strategy"])
            if row.get("library_source"):
                keywords.append(row["library_source"])

            # Filter out empty keywords
            keywords = [k for k in keywords if k]

            # Build PublicationMetadata with creative field mapping (GEOProvider pattern)
            return PublicationMetadata(
                uid=identifier,
                title=title,
                journal="Sequence Read Archive (SRA)",  # Static repository name
                published=published,
                abstract=abstract,
                keywords=keywords,
                pmid=None,  # Could extract from linked publications if available
            )

        except SRANotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error extracting SRA metadata for {identifier}: {e}")
            raise SRAProviderError(f"Error extracting metadata: {str(e)}") from e

    def search_microbiome_datasets(
        self,
        query: str,
        amplicon_region: Optional[str] = None,  # "16S", "ITS", "18S"
        body_site: Optional[str] = None,  # "gut", "oral", "skin"
        host_organism: str = "Homo sapiens",
        max_results: int = 20,
    ) -> str:
        """
        Specialized search for microbiome/metagenomics datasets.

        This method provides enhanced filtering for:
        - Amplicon sequencing (16S rRNA, ITS, 18S rRNA)
        - Shotgun metagenomics
        - Host-associated microbiomes
        - Body site-specific studies

        Args:
            query: Search query (e.g., "inflammatory bowel disease", "obesity")
            amplicon_region: Amplicon target ("16S", "ITS", "18S") or None for shotgun
            body_site: Body site filter ("gut", "oral", "skin", "vaginal", etc.)
            host_organism: Host organism (default: "Homo sapiens")
            max_results: Maximum results to return

        Returns:
            Formatted string with microbiome-specific SRA datasets

        Examples:
            >>> # Find 16S gut microbiome studies
            >>> search_microbiome_datasets("IBD gut", amplicon_region="16S")

            >>> # Find shotgun metagenomics
            >>> search_microbiome_datasets("obesity microbiome", amplicon_region=None)
        """
        # Build enhanced query with microbiome-specific terms
        enhanced_query = query

        # Add amplicon region to query if specified
        if amplicon_region:
            amplicon_upper = amplicon_region.upper()
            if amplicon_upper not in enhanced_query.upper():
                enhanced_query = f"{query} {amplicon_upper}"

        # Add body site to query if specified
        if body_site and body_site.lower() not in enhanced_query.lower():
            enhanced_query = f"{enhanced_query} {body_site}"

        # Add microbiome/metagenomics keywords if not present
        microbiome_keywords = ["microbiome", "metagenom", "16s", "its", "amplicon"]
        if not any(
            keyword in enhanced_query.lower() for keyword in microbiome_keywords
        ):
            enhanced_query = f"{enhanced_query} microbiome"

        # Build filters
        filters = {
            "source": "METAGENOMIC",
            "organism": host_organism,
        }

        # Add strategy filter based on amplicon region
        if amplicon_region:
            filters["strategy"] = "AMPLICON"
        else:
            # Shotgun metagenomics
            filters["strategy"] = "WGS"

        # Delegate to search_publications with enhanced query and filters
        logger.info(
            f"Microbiome search: query='{enhanced_query}', "
            f"amplicon={amplicon_region}, body_site={body_site}"
        )

        # Pass modality_hint for quality filters
        modality_hint = "amplicon" if amplicon_region else None
        result = self.search_publications(
            enhanced_query, max_results=max_results, filters=filters, modality_hint=modality_hint
        )

        # Add microbiome-specific tips
        result += "\n---\n\n"
        result += "### 🦠 Microbiome Analysis Tips\n\n"

        if amplicon_region:
            result += f"- **Amplicon**: {amplicon_region} rRNA gene sequencing\n"
            result += "- **Analysis**: Consider QIIME2, mothur, or DADA2 pipelines\n"
        else:
            result += "- **Shotgun metagenomics**: Whole-genome sequencing\n"
            result += "- **Analysis**: Consider MetaPhlAn, HUMAnN, Kraken2 pipelines\n"

        if body_site:
            result += f"- **Body site**: {body_site} microbiome\n"

        result += f"- **Host**: {host_organism}\n"
        result += "\n**Next steps**: Use `get_dataset_metadata(accession)` for sample details\n"

        return result

    def _is_sra_accession(self, query: str) -> bool:
        """
        Check if query matches SRA accession patterns.

        Supports NCBI SRA, DDBJ, and ENA accession formats.

        Args:
            query: Query string to check

        Returns:
            bool: True if query matches SRA accession pattern
        """
        import re

        patterns = [
            r"^SRP\d{6,}$",  # NCBI Study
            r"^SRX\d{6,}$",  # NCBI Experiment
            r"^SRS\d{6,}$",  # NCBI Sample
            r"^SRR\d{6,}$",  # NCBI Run
            r"^DRP\d{6,}$",  # DDBJ Study
            r"^DRX\d{6,}$",  # DDBJ Experiment
            r"^DRS\d{6,}$",  # DDBJ Sample
            r"^DRR\d{6,}$",  # DDBJ Run
            r"^ERP\d{6,}$",  # ENA Study
            r"^ERX\d{6,}$",  # ENA Experiment
            r"^ERS\d{6,}$",  # ENA Sample
            r"^ERR\d{6,}$",  # ENA Run
        ]
        return any(re.match(pattern, query.strip()) for pattern in patterns)

    def _enrich_accession_metadata(self, df: pd.DataFrame, accession: str) -> pd.DataFrame:
        """
        Enrich pysradb metadata with organism and platform from direct NCBI API.

        This fixes Bug #2 where pysradb returns incomplete metadata (missing organism/platform).

        Args:
            df: DataFrame from pysradb.sra_metadata()
            accession: Original SRA accession (SRP, SRX, etc.)

        Returns:
            Enriched DataFrame with organism and instrument_platform fields
        """
        if df.empty:
            return df

        # Check if organism or instrument_platform are missing
        needs_enrichment = False
        if "organism" not in df.columns or df["organism"].isna().all():
            needs_enrichment = True
            logger.debug(f"Missing organism field for {accession}, enriching via NCBI API")
        if "instrument_platform" not in df.columns or df["instrument_platform"].isna().all():
            needs_enrichment = True
            logger.debug(f"Missing platform field for {accession}, enriching via NCBI API")

        if not needs_enrichment:
            return df

        try:
            # Get study accession for NCBI search
            study_acc = None
            for col in ["study_accession", "experiment_accession"]:
                if col in df.columns:
                    val = df[col].iloc[0]
                    if pd.notna(val):
                        study_acc = val
                        break

            if not study_acc:
                logger.warning(f"Cannot enrich {accession}: no study/experiment accession found")
                return df

            # Search via NCBI esearch to get SRA IDs
            sra_ids = self._ncbi_esearch(study_acc, {}, max_results=10)

            if not sra_ids:
                logger.warning(f"Cannot enrich {accession}: NCBI esearch found no IDs")
                return df

            # Get metadata via esummary
            ncbi_df = self._ncbi_esummary(sra_ids)

            if ncbi_df.empty:
                logger.warning(f"Cannot enrich {accession}: NCBI esummary returned empty")
                return df

            # Take first record from NCBI data (usually most relevant)
            ncbi_record = ncbi_df.iloc[0]

            # Enrich missing fields
            if "organism" not in df.columns or df["organism"].isna().all():
                if pd.notna(ncbi_record.get("organism")):
                    df["organism"] = ncbi_record["organism"]
                    logger.debug(f"Enriched organism: {ncbi_record['organism']}")

            if "instrument_platform" not in df.columns or df["instrument_platform"].isna().all():
                if pd.notna(ncbi_record.get("instrument_platform")):
                    df["instrument_platform"] = ncbi_record["instrument_platform"]
                    logger.debug(f"Enriched platform: {ncbi_record['instrument_platform']}")

            return df

        except Exception as e:
            logger.warning(f"Failed to enrich metadata for {accession}: {e}")
            return df  # Return original df on error

    def _apply_filters(self, df, filters: Dict[str, Any]):
        """
        Apply filters to SRA metadata DataFrame.

        Args:
            df: pandas DataFrame with SRA metadata
            filters: Dictionary of filter criteria

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        filtered_df = df.copy()

        # Organism filter
        if "organism" in filters and "organism" in filtered_df.columns:
            organism = filters["organism"]
            filtered_df = filtered_df[
                filtered_df["organism"].str.contains(organism, case=False, na=False)
            ]

        # Library strategy filter (RNA-Seq, WGS, AMPLICON, etc.)
        if "strategy" in filters and "library_strategy" in filtered_df.columns:
            strategy = filters["strategy"].upper()
            filtered_df = filtered_df[
                filtered_df["library_strategy"].str.upper() == strategy
            ]

        # Library layout filter (PAIRED, SINGLE)
        if "layout" in filters and "library_layout" in filtered_df.columns:
            layout = filters["layout"].upper()
            filtered_df = filtered_df[
                filtered_df["library_layout"].str.upper() == layout
            ]

        # Library source filter (TRANSCRIPTOMIC, GENOMIC, METAGENOMIC)
        if "source" in filters and "library_source" in filtered_df.columns:
            source = filters["source"].upper()
            filtered_df = filtered_df[
                filtered_df["library_source"].str.upper() == source
            ]

        # Platform filter (ILLUMINA, PACBIO, etc.)
        if "platform" in filters:
            platform = filters["platform"].upper()
            # Check both 'instrument_platform' and 'platform' columns
            if "instrument_platform" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["instrument_platform"].str.contains(
                        platform, case=False, na=False
                    )
                ]
            elif "platform" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["platform"].str.contains(platform, case=False, na=False)
                ]

        logger.debug(
            f"Applied filters: {filters} - {len(df)} → {len(filtered_df)} results"
        )

        return filtered_df

    def _format_search_results(
        self, df, query: str, max_results: int, filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format pysradb search results for display.

        Args:
            df: pandas DataFrame with SRA metadata
            query: Original search query
            max_results: Maximum number of results to display
            filters: Optional filters that were applied

        Returns:
            str: Formatted markdown string with search results
        """
        response = "## 🧬 SRA Database Search Results\n\n"
        response += f"**Query**: `{query}`\n"

        # Display applied filters
        if filters:
            response += "**Filters**: "
            filter_parts = []
            for key, value in filters.items():
                filter_parts.append(f"{key}={value}")
            response += ", ".join(filter_parts) + "\n"

        response += f"**Total Results**: {len(df):,}\n\n"

        if df.empty:
            response += "No datasets found matching your query"
            if filters:
                response += " and filters"
            response += ".\n"
            return response

        # Limit results to max_results
        df_display = df.head(max_results)

        for idx, row in df_display.iterrows():
            # Determine primary accession (prioritize study > experiment > run)
            # Can't use 'or' with pd.NA - must check explicitly
            acc = "Unknown"
            for col in ["study_accession", "experiment_accession", "run_accession"]:
                val = row.get(col)
                if pd.notna(val) and val:
                    acc = val
                    break

            # Get title, handling pd.NA
            title = row.get("study_title")
            if pd.isna(title):
                title = row.get("experiment_title", "Untitled")
            if pd.isna(title):
                title = "Untitled"

            response += f"### {idx + 1}. {title}\n"
            response += (
                f"**Accession**: [{acc}](https://www.ncbi.nlm.nih.gov/sra/{acc})\n"
            )

            # Key metadata - check pd.notna() before displaying
            if "organism" in row and pd.notna(row["organism"]):
                response += f"**Organism**: {row['organism']}\n"
            if "library_strategy" in row and pd.notna(row["library_strategy"]):
                response += f"**Strategy**: {row['library_strategy']}\n"
            if "library_layout" in row and pd.notna(row["library_layout"]):
                response += f"**Layout**: {row['library_layout']}\n"
            if "instrument_platform" in row and pd.notna(row["instrument_platform"]):
                response += f"**Platform**: {row['instrument_platform']}\n"
            if "total_size" in row and pd.notna(row["total_size"]):
                response += f"**Total Size**: {row['total_size']}\n"
            if "total_runs" in row and pd.notna(row["total_runs"]):
                response += f"**Total Runs**: {row['total_runs']}\n"

            response += "\n"

        if len(df) > max_results:
            response += f"_Showing {max_results} of {len(df)} total results._\n\n"

        return response
