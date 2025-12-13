"""
MassIVE provider implementation for proteomics/metabolomics dataset search and metadata extraction.

This provider implements search and metadata capabilities for MassIVE
(https://massive.ucsd.edu), UCSD's public mass spectrometry repository
supporting proteomics and metabolomics data. Uses PROXI v0.1 standardized API.

PROXI specification: https://github.com/HUPO-PSI/proxi-schemas
MassIVE documentation: https://ccms-ucsd.github.io/MassIVEDocumentation/
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
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


class MassIVEProviderConfig(BaseModel):
    """Configuration for MassIVE provider."""

    # MassIVE PROXI API settings
    base_url: str = "https://massive.ucsd.edu/ProteoSAFe/proxi/v0.1"
    max_results: int = Field(default=100, ge=1, le=1000)
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=1.0, ge=0.1, le=5.0)  # MassIVE is slower

    # Result processing settings
    include_file_info: bool = True
    cache_results: bool = True


class MassIVESearchResult(BaseModel):
    """Result from MassIVE dataset search."""

    count: int
    datasets: List[Dict[str, Any]]
    page_number: int = 0
    result_type: str = "datasets"


class MassIVEProvider(BasePublicationProvider):
    """
    MassIVE provider for proteomics/metabolomics dataset search and metadata extraction.

    Implements PROXI v0.1 API for:
    - Dataset search by keywords
    - Dataset metadata retrieval via PROXI standard
    - File listing for downloads
    - Support for both proteomics and metabolomics data
    """

    def __init__(
        self,
        data_manager: DataManagerV2,
        config: Optional[MassIVEProviderConfig] = None,
    ):
        """
        Initialize MassIVE provider.

        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        self.config = config or MassIVEProviderConfig()

        # Initialize cache directory
        self.cache_dir = self.data_manager.cache_dir / "massive"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create SSL context
        self.ssl_context = create_ssl_context()

        logger.debug(
            f"Initialized MassIVE provider with PROXI API: {self.config.base_url}"
        )

    @property
    def source(self) -> PublicationSource:
        """Return MassIVE as the publication source."""
        return PublicationSource.MASSIVE

    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by MassIVE."""
        return [DatasetType.MASSIVE]

    @property
    def priority(self) -> int:
        """
        Return provider priority for capability-based routing.

        MassIVE has medium priority (20) - large repository but less structured than PRIDE.

        Returns:
            int: Priority 20 (medium-high priority)
        """
        return 20

    def get_supported_capabilities(self) -> Dict[str, bool]:
        """
        Return capabilities supported by MassIVE provider.

        MassIVE provides dataset discovery and metadata extraction via PROXI API.

        Returns:
            Dict[str, bool]: Capability support mapping
        """
        return {
            ProviderCapability.SEARCH_LITERATURE: False,
            ProviderCapability.DISCOVER_DATASETS: True,
            ProviderCapability.FIND_LINKED_DATASETS: False,  # No PMID linking in PROXI
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
        Validate MassIVE accession format (MSV followed by 9 digits).

        Uses centralized AccessionResolver for pattern matching.

        Args:
            identifier: Potential MassIVE accession

        Returns:
            bool: True if valid MSV accession
        """
        from lobster.core.identifiers import get_accession_resolver

        resolver = get_accession_resolver()
        return resolver.validate(identifier, database="MassIVE")

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
        Search MassIVE datasets by keyword via PROXI API.

        Args:
            query: Search keyword
            max_results: Maximum number of results
            filters: Optional filters (currently limited in PROXI v0.1)
            **kwargs: Additional parameters (page_number)

        Returns:
            str: Formatted search results
        """
        try:
            page_number = kwargs.get("page_number", 0)
            page_size = min(max_results, 100)  # PROXI typically limits to 100

            # PROXI datasets endpoint
            url = f"{self.config.base_url}/datasets"
            params = {
                "pageSize": page_size,
                "pageNumber": page_number,
            }

            # Add keyword filter if supported (PROXI spec)
            if query:
                params["filter"] = query

            # Execute request (MassIVE PROXI v0.1 may not support this endpoint)
            try:
                response_data = self._make_api_request(url, params)
                # PROXI returns array of dataset objects
                datasets = response_data if isinstance(response_data, list) else []
            except Exception as e:
                # MassIVE's PROXI implementation is incomplete - fallback message
                logger.warning(f"MassIVE PROXI search not available: {e}")
                return (
                    "MassIVE dataset search via PROXI API is not currently supported.\n\n"
                    "To use MassIVE datasets, please provide a direct MSV accession number:\n"
                    "  Example: MSV000083067, MSV000085232\n\n"
                    "You can search for datasets at: https://massive.ucsd.edu/ProteoSAFe/datasets.jsp"
                )

            # Filter by keyword in title/description if API doesn't support it
            if query and not any("filter" in str(p) for p in params):
                query_lower = query.lower()
                datasets = [
                    d
                    for d in datasets
                    if query_lower in d.get("title", "").lower()
                    or query_lower in d.get("description", "").lower()
                ]

            datasets = datasets[:max_results]

            # Format response
            result = (
                f"Found {len(datasets)} MassIVE dataset(s) for query: '{query}'\n\n"
            )

            if not datasets:
                return result + "No datasets found. Try different keywords."

            result += f"Showing {len(datasets)} datasets:\n\n"

            for i, dataset in enumerate(datasets[:10], 1):
                accession = dataset.get("accession", "Unknown")
                title = dataset.get("title", "No title")

                # Extract data type
                data_type = "Unknown"
                contacts = dataset.get("contacts", [])
                if contacts and isinstance(contacts[0], dict):
                    contact_info = contacts[0].get("contactProperties", [])
                    for prop in contact_info:
                        if prop.get("name") == "DatasetType":
                            data_type = prop.get("value", "Unknown")
                            break

                result += f"{i}. **{accession}** - {title[:80]}\n"
                result += f"   Type: {data_type}\n"

                # Add species if available
                species = dataset.get("species", [])
                if species:
                    species_names = [
                        s.get("name", "") for s in species if isinstance(s, dict)
                    ]
                    if species_names:
                        result += f"   Species: {', '.join(species_names[:2])}\n"

                result += "\n"

            if len(datasets) > 10:
                result += f"... and {len(datasets) - 10} more datasets\n"

            return result

        except Exception as e:
            logger.error(f"Error searching MassIVE datasets: {e}")
            return f"Error searching MassIVE: {str(e)}"

    def find_datasets_from_publication(
        self,
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs,
    ) -> str:
        """
        Find MassIVE datasets linked to a publication.

        Note: PROXI v0.1 has limited publication linking. This searches
        by identifier in title/description as fallback.

        Args:
            identifier: PMID or DOI
            dataset_types: Expected to include DatasetType.MASSIVE
            **kwargs: Additional parameters

        Returns:
            str: Formatted list of linked datasets
        """
        try:
            # PROXI doesn't have direct PMID linking
            # Search using identifier as keyword
            return self.search_publications(query=identifier, max_results=50, **kwargs)

        except Exception as e:
            logger.error(f"Error finding linked MassIVE datasets: {e}")
            return f"Error finding linked datasets: {str(e)}"

    def extract_publication_metadata(
        self, identifier: str, **kwargs
    ) -> PublicationMetadata:
        """
        Extract metadata from a MassIVE dataset.

        Args:
            identifier: MSV accession
            **kwargs: Additional parameters

        Returns:
            PublicationMetadata: Standardized metadata
        """
        try:
            dataset = self.get_dataset_metadata(identifier)

            # Extract publication info if available
            publications = dataset.get("publications", [])
            pmid = None
            doi = None

            if publications:
                pub = publications[0]
                pmid = pub.get("pubmedId")
                doi = pub.get("id") if pub.get("id", "").startswith("10.") else None

            # Extract keywords from contacts
            keywords = []
            contacts = dataset.get("contacts", [])
            for contact in contacts:
                contact_props = contact.get("contactProperties", [])
                for prop in contact_props:
                    if prop.get("name") in ["Keywords", "DatasetType"]:
                        value = prop.get("value", "")
                        if value:
                            keywords.append(value)

            return PublicationMetadata(
                uid=identifier,
                title=dataset.get("title", ""),
                journal=(
                    publications[0].get("referenceLine", "") if publications else None
                ),
                published=dataset.get("publicationDate"),
                doi=doi,
                pmid=pmid,
                abstract=dataset.get("description", ""),
                authors=[],  # PROXI doesn't provide author details
                keywords=keywords,
            )

        except Exception as e:
            logger.error(f"Error extracting MassIVE metadata: {e}")
            return PublicationMetadata(
                uid=identifier, title=f"MassIVE Dataset {identifier}", authors=[]
            )

    # =========================================================================
    # MASSIVE-SPECIFIC METHODS
    # =========================================================================

    def get_dataset_metadata(self, accession: str) -> Dict[str, Any]:
        """
        Get comprehensive metadata for a MassIVE dataset via PROXI.

        Args:
            accession: MSV accession (e.g., "MSV000012345")

        Returns:
            Dict containing dataset metadata

        Raises:
            ValueError: If accession not found or API error
        """
        if not self.validate_identifier(accession):
            raise ValueError(f"Invalid MassIVE accession format: {accession}")

        try:
            url = f"{self.config.base_url}/datasets/{accession}"
            dataset_data = self._make_api_request(url)

            logger.debug(f"Retrieved metadata for MassIVE dataset {accession}")
            return dataset_data

        except Exception as e:
            logger.error(f"Error getting MassIVE dataset metadata: {e}")
            raise ValueError(
                f"Failed to retrieve MassIVE dataset {accession}: {str(e)}"
            )

    def get_ftp_urls(self, accession: str) -> List[Tuple[str, str]]:
        """
        Extract FTP download URLs for a MassIVE dataset.

        Args:
            accession: MSV accession

        Returns:
            List of tuples: (filename, ftp_url)

        Note:
            MassIVE FTP structure: ftp://massive.ucsd.edu/v06/MSV{accession}/
            PROXI API may not provide direct file URLs, need to construct
        """
        try:
            # MassIVE FTP base
            ftp_base = f"ftp://massive.ucsd.edu/v06/{accession}"

            # PROXI doesn't provide file listing in v0.1
            # Would need to implement FTP directory listing or use web API
            logger.warning(f"MassIVE FTP URL construction for {accession}: {ftp_base}")
            logger.warning(
                "PROXI v0.1 doesn't provide file listings - need FTP directory scan"
            )

            return [(accession, ftp_base)]

        except Exception as e:
            logger.error(f"Error getting MassIVE FTP URLs: {e}")
            return []

    def get_download_urls(self, accession: str) -> "DownloadUrlResult":
        """
        Get download URLs for a MassIVE dataset as a typed DownloadUrlResult.

        Note: MassIVE PROXI v0.1 doesn't provide file listings via API.
        File URLs are populated during FTP directory scan at download time.

        Args:
            accession: MSV accession (e.g., "MSV000012345")

        Returns:
            DownloadUrlResult with FTP base URL (file lists empty until FTP scan)

        Example:
            >>> provider = MassIVEProvider(data_manager)
            >>> result = provider.get_download_urls("MSV000012345")
            >>> print(result.ftp_base)  # FTP directory for scan
        """
        from lobster.core.schemas.download_urls import DownloadUrlResult

        try:
            ftp_base = f"ftp://massive.ucsd.edu/v06/{accession}"

            return DownloadUrlResult(
                accession=accession,
                database="massive",
                ftp_base=ftp_base,
                # Files empty - populated during FTP directory scan at download time
                recommended_strategy="raw",
            )

        except Exception as e:
            logger.error(f"Error getting download URLs for {accession}: {e}")
            return DownloadUrlResult(
                accession=accession,
                database="massive",
                error=str(e),
            )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _make_api_request(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make API request to MassIVE PROXI with retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Parsed JSON response (can be dict or list)

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
                logger.debug(f"MassIVE PROXI request: {full_url}")

                request = urllib.request.Request(
                    full_url, headers={"Accept": "application/json"}
                )

                with urllib.request.urlopen(
                    request, timeout=60, context=self.ssl_context
                ) as response:
                    data = response.read().decode("utf-8")
                    return json.loads(data)

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise ValueError(f"Resource not found: {url}")
                elif e.code == 429:
                    wait_time = self.config.sleep_time * (2**attempt)
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.config.max_retry}"
                    )
                    time.sleep(wait_time)
                elif e.code >= 500:
                    # Server error - retry with backoff
                    wait_time = self.config.sleep_time * (2 ** (attempt + 1))
                    logger.warning(
                        f"Server error {e.code}, waiting {wait_time}s before retry {attempt + 1}/{self.config.max_retry}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error {e.code}: {e.reason}")
                    raise

            except urllib.error.URLError as e:
                logger.warning(
                    f"URL error: {e.reason}, attempt {attempt + 1}/{self.config.max_retry}"
                )
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time * (attempt + 1))

            except Exception as e:
                logger.error(f"Unexpected error in MassIVE API request: {e}")
                if attempt < self.config.max_retry - 1:
                    time.sleep(self.config.sleep_time)
                else:
                    raise

        raise ValueError(
            f"Failed to retrieve data from MassIVE after {self.config.max_retry} attempts"
        )
