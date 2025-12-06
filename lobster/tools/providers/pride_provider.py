"""
PRIDE provider implementation for proteomics dataset search and metadata extraction.

This provider implements search and metadata capabilities for PRIDE Archive
(https://www.ebi.ac.uk/pride), the largest public proteomics repository in
ProteomeXchange. Supports PXD accession-based discovery and keyword searching.

API endpoints and retry logic patterns adapted from pridepy:
https://github.com/PRIDE-Archive/pridepy
Copyright (c) 2025 PRIDE Team
Licensed under Apache-2.0
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.providers.base_provider import (
    BasePublicationProvider,
    DatasetType,
    ProviderCapability,
    PublicationMetadata,
    PublicationSource,
)
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


class PRIDEProviderConfig(BaseModel):
    """Configuration for PRIDE provider."""

    # PRIDE API settings
    base_url: str = "https://www.ebi.ac.uk/pride/ws/archive/v2"
    max_results: int = Field(default=100, ge=1, le=1000)
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=0.5, ge=0.1, le=5.0)

    # Result processing settings
    include_file_info: bool = True
    cache_results: bool = True


class PRIDESearchResult(BaseModel):
    """Result from PRIDE project search."""

    count: int
    projects: List[Dict[str, Any]]
    page: int = 0
    page_size: int = 100


class PRIDEProvider(BasePublicationProvider):
    """
    PRIDE Archive provider for proteomics dataset search and metadata extraction.

    Implements PRIDE REST API v2 for:
    - Project search by keywords and filters
    - Project metadata retrieval
    - File listing and categorization
    - Link extraction for dataset downloads
    """

    def __init__(
        self, data_manager: DataManagerV2, config: Optional[PRIDEProviderConfig] = None
    ):
        """
        Initialize PRIDE provider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        self.config = config or PRIDEProviderConfig()

        # Initialize cache directory
        self.cache_dir = self.data_manager.cache_dir / "pride"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create SSL context for secure connections
        self.ssl_context = create_ssl_context()

        logger.debug(f"Initialized PRIDE provider with base URL: {self.config.base_url}")

    @property
    def source(self) -> PublicationSource:
        """Return PRIDE as the publication source."""
        return PublicationSource.PRIDE

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by PRIDE."""
        return [DatasetType.PRIDE]

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        PRIDE has high priority (10) as the authoritative source for proteomics datasets.

        Returns:
            int: Priority 10 (high priority)
        """
        return 10

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by PRIDE provider.

        PRIDE excels at proteomics dataset discovery and metadata extraction.

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        return {
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.DISCOVER_DATASETS: True,
            ProviderCapability.FIND_LINKED_DATASETS: True,
            ProviderCapability.EXTRACT_METADATA: True,
            ProviderCapability.VALIDATE_METADATA: True,
            ProviderCapability.QUERY_CAPABILITIES: True,
            ProviderCapability.GET_ABSTRACT: False,
            ProviderCapability.GET_FULL_CONTENT: False,
            ProviderCapability.EXTRACT_METHODS: False,
            ProviderCapability.EXTRACT_PDF: False,
            ProviderCapability.INTEGRATE_MULTI_OMICS: False,
        }

    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate PRIDE accession format (PXD followed by 6 digits).

        Uses centralized AccessionResolver for pattern matching.

        Args:
            identifier: Potential PRIDE accession

        Returns:
            bool: True if valid PXD accession
        """
        from lobster.core.identifiers import get_accession_resolver

        resolver = get_accession_resolver()
        return resolver.validate(identifier, database="PRIDE")

    # =========================================================================
    # CORE API METHODS
    # =========================================================================

    def search_publications(
        self,
        query: str,
        max_results: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Search PRIDE projects by keyword.

        Args:
            query: Search keyword
            max_results: Maximum number of results
            filters: Optional filters (organism, instrument, etc.)
            **kwargs: Additional parameters (page, sort_direction)

        Returns:
            str: Formatted search results
        """
        try:
            page = kwargs.get("page", 0)
            page_size = min(max_results, self.config.max_results)

            # Build search request
            url = f"{self.config.base_url}/search/projects"
            params = {
                "keyword": query,
                "pageSize": page_size,
                "page": page,
                "sortDirection": kwargs.get("sort_direction", "DESC"),
                "sortFields": kwargs.get("sort_fields", "submissionDate"),
            }

            # Add filters if provided
            if filters:
                filter_str = self._build_filter_string(filters)
                if filter_str:
                    params["filter"] = filter_str

            # Execute request with retry
            response_data = self._make_api_request(url, params)

            # Parse results
            projects = response_data.get("_embedded", {}).get("projects", [])
            page_info = response_data.get("page", {})

            # Format response
            result = f"Found {page_info.get('totalElements', len(projects))} PRIDE projects for query: '{query}'\n\n"

            if not projects:
                return result + "No projects found. Try different keywords or filters."

            result += f"Showing {len(projects)} projects:\n\n"

            for i, project in enumerate(projects[:10], 1):
                accession = project.get("accession", "Unknown")
                title = project.get("title", "No title")
                organisms = project.get("organisms", [])
                organism_str = (
                    ", ".join([org.get("name", "") for org in organisms[:2]])
                    if organisms
                    else "Unknown"
                )

                result += f"{i}. **{accession}** - {title[:80]}\n"
                result += f"   Organism: {organism_str}\n"

                if "publicationDate" in project:
                    result += f"   Published: {project['publicationDate']}\n"

                result += "\n"

            if len(projects) > 10:
                result += f"... and {len(projects) - 10} more projects\n"

            return result

        except Exception as e:
            logger.error(f"Error searching PRIDE projects: {e}")
            return f"Error searching PRIDE: {str(e)}"

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find PRIDE datasets linked to a publication (by PMID or DOI).

        Args:
            identifier: PMID or DOI
            dataset_types: Expected to include DatasetType.PRIDE
            **kwargs: Additional parameters

        Returns:
            str: Formatted list of linked datasets
        """
        try:
            # Search PRIDE using publication identifier as keyword
            # PRIDE API doesn't have direct PMID linking like GEO E-Link,
            # so we search and filter by reference

            url = f"{self.config.base_url}/search/projects"
            params = {
                "keyword": identifier,
                "pageSize": 50,
                "page": 0,
            }

            response_data = self._make_api_request(url, params)
            projects = response_data.get("_embedded", {}).get("projects", [])

            # Filter projects that actually reference this publication
            linked_projects = []
            for project in projects:
                references = project.get("references", [])
                for ref in references:
                    ref_pubmed = ref.get("pubmedId", "")
                    ref_doi = ref.get("doi", "")

                    if identifier in [ref_pubmed, ref_doi]:
                        linked_projects.append(project)
                        break

            if not linked_projects:
                return f"No PRIDE datasets found linked to {identifier}"

            # Format results
            result = f"Found {len(linked_projects)} PRIDE dataset(s) linked to {identifier}:\n\n"

            for i, project in enumerate(linked_projects, 1):
                accession = project.get("accession", "Unknown")
                title = project.get("title", "No title")

                result += f"{i}. **{accession}** - {title[:80]}\n"
                result += f"   URL: https://www.ebi.ac.uk/pride/archive/projects/{accession}\n\n"

            return result

        except Exception as e:
            logger.error(f"Error finding linked PRIDE datasets: {e}")
            return f"Error finding linked datasets: {str(e)}"

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract metadata from a PRIDE project.

        Args:
            identifier: PXD accession
            **kwargs: Additional parameters

        Returns:
            PublicationMetadata: Standardized metadata
        """
        try:
            project = self.get_project_metadata(identifier)

            # Extract publication references
            references = project.get("references", [])
            pmid = None
            doi = None

            if references:
                # Use first reference
                ref = references[0]
                pmid = ref.get("pubmedId")
                doi = ref.get("doi")

            # Extract keywords
            keywords = project.get("keywords", [])
            project_tags = project.get("projectTags", [])
            all_keywords = keywords + project_tags

            return PublicationMetadata(
                uid=identifier,
                title=project.get("title", ""),
                journal=references[0].get("referenceLine", "") if references else None,
                published=project.get("publicationDate"),
                doi=doi,
                pmid=pmid,
                abstract=project.get("projectDescription", ""),
                authors=self._extract_authors(project),
                keywords=all_keywords,
            )

        except Exception as e:
            logger.error(f"Error extracting PRIDE metadata: {e}")
            # Return minimal metadata
            return PublicationMetadata(
                uid=identifier, title=f"PRIDE Project {identifier}", authors=[]
            )

    # =========================================================================
    # PRIDE-SPECIFIC METHODS
    # =========================================================================

    def get_project_metadata(self, accession: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a PRIDE project.

        Args:
            accession: PXD accession (e.g., "PXD012345")

        Returns:
            Dict containing project metadata

        Raises:
            ValueError: If accession not found or API error
        """
        if not self.validate_identifier(accession):
            raise ValueError(f"Invalid PRIDE accession format: {accession}")

        try:
            url = f"{self.config.base_url}/projects/{accession}"
            project_data = self._make_api_request(url)

            logger.debug(f"Retrieved metadata for PRIDE project {accession}")
            return project_data

        except Exception as e:
            logger.error(f"Error getting PRIDE project metadata: {e}")
            raise ValueError(f"Failed to retrieve PRIDE project {accession}: {str(e)}")

    def get_project_files(
        self, accession: str, file_category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get file list for a PRIDE project.

        Args:
            accession: PXD accession
            file_category: Optional filter by category (RAW, PEAK, SEARCH, RESULT, FASTA, OTHER)

        Returns:
            List of file dictionaries with name, size, type, URLs
        """
        try:
            # Use the project-specific files endpoint with pagination
            url = f"{self.config.base_url}/projects/{accession}/files"
            params = {
                "pageSize": 100,
                "page": 0,
                "sortDirection": "ASC",
                "sortFields": "fileName",
            }

            response_data = self._make_api_request(url, params)

            # Response has _embedded wrapper
            files = []
            if isinstance(response_data, dict) and "_embedded" in response_data:
                files = response_data["_embedded"].get("files", [])
            elif isinstance(response_data, list):
                files = response_data

            # Filter by category if requested
            if file_category:
                files = [
                    f
                    for f in files
                    if f.get("fileCategory", {}).get("value", "")
                    == file_category.upper()
                ]

            logger.debug(f"Retrieved {len(files)} files for {accession}")
            return files

        except Exception as e:
            logger.error(f"Error getting PRIDE project files: {e}")
            return []

    def get_ftp_urls(
        self, accession: str, file_category: Optional[str] = None
    ) -> List[Tuple[str, str]]:
        """
        Extract FTP download URLs from project files.

        Args:
            accession: PXD accession
            file_category: Optional filter by category

        Returns:
            List of tuples: (filename, ftp_url)
        """
        files = self.get_project_files(accession, file_category)

        ftp_urls = []
        for file_info in files:
            filename = file_info.get("fileName", "")
            locations = file_info.get("publicFileLocations", [])

            # Find FTP location
            for location in locations:
                if location.get("name") == "FTP Protocol":
                    ftp_url = location.get("value", "")
                    if ftp_url:
                        ftp_urls.append((filename, ftp_url))
                        break

        return ftp_urls

    def get_download_urls(self, accession: str) -> "DownloadUrlResult":
        """
        Get download URLs for a PRIDE project as a typed DownloadUrlResult.

        Args:
            accession: PXD accession (e.g., "PXD012345")

        Returns:
            DownloadUrlResult with categorized file URLs:
              - raw_files: Instrument RAW files
              - processed_files: mzML, peak lists, other processed data
              - search_files: Search engine outputs
              - metadata_files: FASTA databases

        Example:
            >>> provider = PRIDEProvider(data_manager)
            >>> result = provider.get_download_urls("PXD012345")
            >>> print(len(result.raw_files))
            >>> print(len(result.processed_files))
        """
        from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult

        try:
            files = self.get_project_files(accession)

            # Build categorized file lists
            raw_files = []
            processed_files = []
            search_files = []
            metadata_files = []  # For FASTA files

            for file_info in files:
                filename = file_info.get("fileName", "")
                category = file_info.get("fileCategory", {}).get("value", "")
                locations = file_info.get("publicFileLocations", [])

                # Extract FTP URL
                ftp_url = None
                for location in locations:
                    if location.get("name") == "FTP Protocol":
                        ftp_url = location.get("value", "")
                        break

                if not ftp_url:
                    continue

                download_file = DownloadFile(
                    url=ftp_url,
                    filename=filename,
                    size_bytes=file_info.get("fileSizeBytes"),
                    file_type=category.lower() if category else "other",
                )

                # Route to appropriate category
                if category == "RAW":
                    raw_files.append(download_file)
                elif category == "RESULT":
                    processed_files.append(download_file)
                elif category == "SEARCH":
                    search_files.append(download_file)
                elif category == "FASTA":
                    metadata_files.append(download_file)
                else:
                    processed_files.append(download_file)

            return DownloadUrlResult(
                accession=accession,
                database="pride",
                raw_files=raw_files,
                processed_files=processed_files,
                search_files=search_files,
                metadata_files=metadata_files,
                recommended_strategy="processed" if processed_files else "raw",
            )

        except Exception as e:
            logger.error(f"Error getting download URLs for {accession}: {e}")
            return DownloadUrlResult(
                accession=accession,
                database="pride",
                error=str(e),
            )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _make_api_request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make API request to PRIDE with retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Dict: Parsed JSON response

        Raises:
            ValueError: If request fails after retries
        """
        if params:
            query_string = urllib.parse.urlencode(params)
            full_url = f"{url}?{query_string}"
        else:
            full_url = url

        for attempt in range(self.config.max_retry):
            try:
                logger.debug(f"PRIDE API request: {full_url}")

                request = urllib.request.Request(
                    full_url, headers={"Accept": "application/json"}
                )

                with urllib.request.urlopen(
                    request, timeout=30, context=self.ssl_context
                ) as response:
                    data = response.read().decode("utf-8")
                    return json.loads(data)

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise ValueError(f"Resource not found: {url}")
                elif e.code == 429:
                    # Rate limit - wait longer
                    wait_time = self.config.sleep_time * (2**attempt)
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.config.max_retry}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.warning(
                        f"HTTP error {e.code}: {e.reason}, attempt {attempt + 1}/{self.config.max_retry}"
                    )
                    if attempt < self.config.max_retry - 1:
                        time.sleep(self.config.sleep_time * (attempt + 1))

            except urllib.error.URLError as e:
                logger.warning(
                    f"URL error: {e.reason}, attempt {attempt + 1}/{self.config.max_retry}"
                )
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time * (attempt + 1))

            except Exception as e:
                logger.error(f"Unexpected error in PRIDE API request: {e}")
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time)
                else:
                    raise

        raise ValueError(
            f"Failed to retrieve data from PRIDE after {self.config.max_retry} attempts"
        )

    def _build_filter_string(self, filters: Dict[str, Any]) -> str:
        """
        Build PRIDE API filter string from dictionary.

        Args:
            filters: Filter parameters

        Returns:
            str: Formatted filter string

        Example:
            {"organism": "human", "instrument": "orbitrap"}
            -> "organism==human,instrument==orbitrap"
        """
        filter_parts = []

        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values: organism==human,organism==mouse
                for v in value:
                    filter_parts.append(f"{key}=={v}")
            else:
                filter_parts.append(f"{key}=={value}")

        return ",".join(filter_parts)

    def _extract_authors(self, project: Dict[str, Any]) -> List[str]:
        """Extract author names from project metadata."""
        authors = []

        # Extract from submitters
        submitters = project.get("submitters", [])
        for submitter in submitters:
            name = f"{submitter.get('firstName', '')} {submitter.get('lastName', '')}".strip()
            if name:
                authors.append(name)

        # Extract from lab PIs
        lab_pis = project.get("labPIs", [])
        for pi in lab_pis:
            name = f"{pi.get('firstName', '')} {pi.get('lastName', '')}".strip()
            if name and name not in authors:
                authors.append(name)

        return authors
