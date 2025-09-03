"""
GEO provider implementation for direct GEO DataSets search.

This provider implements direct search capabilities for NCBI's GEO DataSets database
using E-utilities, supporting all query patterns from the official API examples. hi
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from pydantic import BaseModel, Field
from enum import Enum

from lobster.tools.providers.base_provider import (
    BasePublicationProvider, 
    PublicationSource, 
    DatasetType, 
    PublicationMetadata, 
    DatasetMetadata
)
from lobster.tools.providers.ncbi_query_builder import (
    NCBIDatabase,
    GEOQueryBuilder,
    build_geo_query
)
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger
from lobster.config.settings import get_settings

logger = get_logger(__name__)


class GEOEntryType(str, Enum):
    """GEO database entry types."""
    SERIES = "gse"  # Series (experiments)
    DATASET = "gds"  # Curated datasets
    PLATFORM = "gpl"  # Platforms
    SAMPLE = "gsm"  # Individual samples


class GEOSearchFilters(BaseModel):
    """Filters for GEO dataset searches."""
    entry_types: Optional[List[str]] = None  # gse, gds, gpl, gsm
    organisms: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None  # {"start": "2023/01/01", "end": "2024/12/31"}
    supplementary_file_types: Optional[List[str]] = None  # ["cel", "txt", "h5"]
    max_results: int = Field(default=20, ge=1, le=5000)
    use_history: bool = True


class GEOProviderConfig(BaseModel):
    """Configuration for GEO provider."""
    
    # GEO search settings
    max_results: int = Field(default=20, ge=1, le=1000)
    default_entry_types: List[str] = Field(default=["gse", "gds"])
    
    # NCBI API settings
    email: str = "kevin.yar@homara.ai"
    api_key: Optional[str] = ""
    max_retry: int = Field(default=3, ge=1, le=10)
    sleep_time: float = Field(default=0.5, ge=0.1, le=5.0)
    
    # Result processing settings
    include_summaries: bool = True
    include_publication_links: bool = True
    cache_results: bool = True


class GEOSearchResult(BaseModel):
    """Result from GEO search with history server support."""
    count: int
    ids: List[str]
    web_env: Optional[str] = None
    query_key: Optional[str] = None
    summaries: Optional[List[Dict[str, Any]]] = None
    query: Optional[str] = None


class GEOProvider(BasePublicationProvider):
    """
    GEO provider for direct dataset search via NCBI E-utilities.
    
    This provider implements all patterns from official GEO API examples:
    - Series search by date range
    - Organism and platform filtering  
    - Supplementary file type search
    - History server support (WebEnv/query_key)
    - Comprehensive result metadata extraction
    """
    
    def __init__(
        self, 
        data_manager: DataManagerV2,
        config: Optional[GEOProviderConfig] = None
    ):
        """
        Initialize GEO provider.
        
        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        settings = get_settings()
        
        # Create config with API key from settings if not provided
        if config is None:
            self.config = GEOProviderConfig(api_key=settings.NCBI_API_KEY)
        else:
            # If config provided but no API key, update it with settings
            if config.api_key is None:
                self.config = config.model_copy(update={'api_key': settings.NCBI_API_KEY})
            else:
                self.config = config
        
        self.query_builder = GEOQueryBuilder()
        
        # Initialize XML parser
        try:
            import xmltodict
            self.parse_xml = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict. Install with: pip install xmltodict"
            )
        
        # NCBI E-utilities base URLs
        self.base_url_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_url_esummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        self.base_url_elink = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        
        # Session cache for WebEnv management
        self._session_cache: Dict[str, Tuple[str, str, datetime]] = {}
    
    @property
    def source(self) -> PublicationSource:
        """Return GEO as the publication source."""
        return PublicationSource.GEO
    
    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by GEO."""
        return [DatasetType.GEO]
    
    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate GEO identifiers.
        
        Args:
            identifier: Identifier to validate (GSE, GDS, GPL, etc.)
            
        Returns:
            bool: True if identifier is valid GEO format
        """
        identifier = identifier.strip().upper()
        
        # Check for GEO accession patterns
        geo_patterns = [
            r'^GSE\d+$',  # Series
            r'^GDS\d+$',  # Dataset
            r'^GPL\d+$',  # Platform
            r'^GSM\d+$',  # Sample
        ]
        
        return any(re.match(pattern, identifier) for pattern in geo_patterns)
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Return features supported by GEO provider."""
        return {
            "literature_search": False,  # GEO is for datasets, not literature
            "dataset_discovery": True,
            "metadata_extraction": True,
            "full_text_access": False,
            "advanced_filtering": True,
            "date_filtering": True,
            "organism_filtering": True,
            "platform_filtering": True,
            "file_type_filtering": True
        }
    
    def search_publications(
        self,
        query: str,
        max_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Search GEO DataSets (not publications - this searches datasets directly).
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default: 20)
            filters: Optional filters for GEO search
            **kwargs: Additional parameters
            
        Returns:
            str: Formatted search results
        """
        logger.info(f"GEO DataSets search: {query[:50]}...")
        
        try:
            # Convert filters to GEOSearchFilters if provided
            geo_filters = self._convert_filters(filters) if filters else None
            
            # Set max results
            if geo_filters:
                geo_filters.max_results = min(max_results, self.config.max_results)
            else:
                geo_filters = GEOSearchFilters(max_results=min(max_results, self.config.max_results))
            
            # Perform GEO search
            search_result = self.search_geo_datasets(query, geo_filters)
            
            if search_result.count == 0:
                return f"No GEO datasets found for query: {query}"
            
            # Get detailed summaries if enabled
            if self.config.include_summaries and search_result.ids:
                summaries = self.get_dataset_summaries(search_result)
                search_result.summaries = summaries
            
            # Format results
            formatted_results = self.format_geo_search_results(search_result, query)
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="geo_datasets_search",
                parameters={
                    "query": query[:100],
                    "max_results": geo_filters.max_results if geo_filters else max_results,
                    "filters": filters
                },
                description="Direct GEO DataSets search"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.exception(f"GEO search error: {e}")
            raise TypeError(f"GEO search error: {str(e)}")
    
    def find_datasets_from_publication(
        self, 
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs
    ) -> str:
        """
        Find GEO datasets by accession (not from publication).
        
        Args:
            identifier: GEO accession (GSE, GDS, etc.)
            dataset_types: Types of datasets to search for
            **kwargs: Additional parameters
            
        Returns:
            str: Formatted dataset information
        """
        logger.info(f"Finding GEO dataset: {identifier}")
        
        try:
            if not self.validate_identifier(identifier):
                return f"Invalid GEO identifier: {identifier}. Please provide GSE, GDS, GPL, or GSM accession."
            
            # Search for the specific accession
            query = f"{identifier}[ACCN]"
            filters = GEOSearchFilters(max_results=1)
            
            search_result = self.search_geo_datasets(query, filters)
            
            if search_result.count == 0:
                return f"GEO dataset not found: {identifier}"
            
            # Get detailed summary
            summaries = self.get_dataset_summaries(search_result)
            
            if not summaries:
                return f"Could not retrieve details for: {identifier}"
            
            summary = summaries[0]
            
            # Format detailed response
            response = self._format_detailed_dataset_report(summary, identifier)
            
            # Log the lookup
            self.data_manager.log_tool_usage(
                tool_name="geo_dataset_lookup",
                parameters={
                    "identifier": identifier,
                    "dataset_types": [dt.value for dt in dataset_types] if dataset_types else None
                },
                description="GEO dataset lookup by accession"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error finding GEO dataset: {e}")
            return f"Error finding GEO dataset: {str(e)}"
    
    def extract_publication_metadata(
        self, 
        identifier: str,
        **kwargs
    ) -> PublicationMetadata:
        """
        Extract metadata from a GEO dataset (returns dataset info as publication-like metadata).
        
        Args:
            identifier: GEO accession
            **kwargs: Additional parameters
            
        Returns:
            PublicationMetadata: Dataset metadata formatted as publication metadata
        """
        try:
            if not self.validate_identifier(identifier):
                raise ValueError(f"Invalid GEO identifier: {identifier}")
            
            # Search for the dataset
            query = f"{identifier}[ACCN]"
            filters = GEOSearchFilters(max_results=1)
            
            search_result = self.search_geo_datasets(query, filters)
            
            if search_result.count == 0:
                raise ValueError(f"GEO dataset not found: {identifier}")
            
            # Get summary
            summaries = self.get_dataset_summaries(search_result)
            
            if not summaries:
                raise ValueError(f"Could not retrieve summary for: {identifier}")
            
            summary = summaries[0]
            
            # Convert to PublicationMetadata format
            return PublicationMetadata(
                uid=identifier,
                title=summary.get('title', f"GEO Dataset {identifier}"),
                journal="Gene Expression Omnibus (GEO)",
                published=summary.get('PDAT', summary.get('date')),
                abstract=summary.get('summary', summary.get('description', '')),
                keywords=[summary.get('GPL', ''), summary.get('taxon', '')],
                pmid=summary.get('PubMedIds', {}).get('int', '') if 'PubMedIds' in summary else None
            )
            
        except Exception as e:
            logger.error(f"Error extracting GEO metadata: {e}")
            # Return minimal metadata on error
            return PublicationMetadata(
                uid=identifier,
                title=f"Error extracting metadata: {str(e)}",
                abstract=f"Could not retrieve GEO dataset metadata for: {identifier}"
            )
    
    # Core GEO-specific methods
    
    def search_geo_datasets(
        self, 
        query: str, 
        filters: Optional[GEOSearchFilters] = None
    ) -> GEOSearchResult:
        """
        Execute eSearch against GEO DataSets database.
        
        Args:
            query: Search query string
            filters: Optional search filters
            
        Returns:
            GEOSearchResult: Search results with IDs and WebEnv
        """
        # Build the complete query using the new query builder
        if filters:
            # Convert filters to dict format for the query builder
            filter_dict = {}
            if filters.organisms:
                filter_dict['organism'] = filters.organisms
            if filters.platforms:
                filter_dict['platform'] = filters.platforms
            if filters.entry_types:
                # Add entry types as special filters
                for et in filters.entry_types:
                    # Use the enum value (e.g., "gse") not the enum object
                    filter_dict[et.value if isinstance(et, GEOEntryType) else et] = True
            if filters.date_range:
                filter_dict['date_range'] = filters.date_range
            if filters.supplementary_file_types:
                filter_dict['supplementary'] = filters.supplementary_file_types
            
            complete_query = self.query_builder.build_query(query, filter_dict)
            max_results = filters.max_results
            use_history = filters.use_history
        else:
            complete_query = self.query_builder.build_query(query)
            max_results = self.config.max_results
            use_history = True
        
        # Validate query syntax
        if not self.query_builder.validate_query(complete_query):
            raise ValueError(f"Invalid query syntax: {complete_query}")
        
        logger.info(f"Executing GEO search: {complete_query}")
        
        # Build eSearch URL
        url_params = {
            'db': 'gds',  # GEO DataSets database
            'term': complete_query,
            'retmode': 'json',
            'retmax': str(max_results),
            'tool': 'lobster',
            'email': self.config.email
        }
        
        if use_history:
            url_params['usehistory'] = 'y'
        
        if self.config.api_key:
            url_params['api_key'] = self.config.api_key
        
        url = f"{self.base_url_esearch}?" + urllib.parse.urlencode(url_params)
        
        # Execute request with retry logic
        response_data = self._execute_request_with_retry(url)
        
        # Parse eSearch response
        try:
            json_data = json.loads(response_data)
            esearch_result = json_data.get('esearchresult', {})
            
            count = int(esearch_result.get('count', '0'))
            ids = esearch_result.get('idlist', [])
            web_env = esearch_result.get('webenv')
            query_key = esearch_result.get('querykey')
            
            # Cache WebEnv session if available
            if web_env and query_key:
                cache_key = f"{web_env}:{query_key}"
                self._session_cache[cache_key] = (web_env, query_key, datetime.now())
            
            return GEOSearchResult(
                count=count,
                ids=ids,
                web_env=web_env,
                query_key=query_key,
                query=complete_query
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing eSearch response: {e}")
            raise ValueError(f"Invalid eSearch response: {e}")
    
    def get_dataset_summaries(
        self,
        search_result: GEOSearchResult
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries using eSummary with WebEnv.
        
        Args:
            search_result: Result from search_geo_datasets
            
        Returns:
            List[Dict[str, Any]]: List of dataset summaries
        """
        if not search_result.ids:
            return []
        
        # Build eSummary URL
        url_params = {
            'db': 'gds',
            'retmode': 'json',
            'tool': 'lobster',
            'email': self.config.email
        }
        
        # Use WebEnv if available for efficiency
        if search_result.web_env and search_result.query_key:
            url_params['webenv'] = search_result.web_env
            url_params['query_key'] = search_result.query_key
        else:
            # Fallback to ID list
            url_params['id'] = ','.join(search_result.ids)
        
        if self.config.api_key:
            url_params['api_key'] = self.config.api_key
        
        url = f"{self.base_url_esummary}?" + urllib.parse.urlencode(url_params)
        
        # Execute request
        response_data = self._execute_request_with_retry(url)
        
        # Parse eSummary response
        try:
            json_data = json.loads(response_data)
            result = json_data.get('result', {})
            
            summaries = []
            for uid in search_result.ids:
                if uid in result:
                    summary = result[uid]
                    summary['uid'] = uid
                    summaries.append(summary)
            
            return summaries
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing eSummary response: {e}")
            return []
    
    def link_to_pubmed(self, geo_ids: List[str]) -> Dict[str, List[str]]:
        """
        Find PubMed IDs linked to GEO entries using eLink.
        
        Args:
            geo_ids: List of GEO dataset IDs
            
        Returns:
            Dict[str, List[str]]: Mapping of GEO ID to PubMed IDs
        """
        if not geo_ids:
            return {}
        
        linked = {}
        
        # Build eLink URL
        url_params = {
            'dbfrom': 'gds',
            'db': 'pubmed',
            'id': ','.join(geo_ids),
            'retmode': 'json',
            'tool': 'lobster',
            'email': self.config.email
        }
        
        if self.config.api_key:
            url_params['api_key'] = self.config.api_key
        
        url = f"{self.base_url_elink}?" + urllib.parse.urlencode(url_params)
        
        try:
            response_data = self._execute_request_with_retry(url)
            json_data = json.loads(response_data)
            
            if "linksets" in json_data:
                for linkset in json_data["linksets"]:
                    geo_id = linkset.get("ids", [None])[0]
                    if geo_id and "linksetdbs" in linkset:
                        pmids = []
                        for db in linkset["linksetdbs"]:
                            if db["dbto"] == "pubmed":
                                pmids.extend(db.get("links", []))
                        if pmids:
                            linked[geo_id] = pmids
                            
        except Exception as e:
            logger.warning(f"Error finding linked PubMed IDs: {e}")
        
        return linked
    
    # Helper methods
    
    def _execute_request_with_retry(self, url: str) -> str:
        """Execute HTTP request with retry logic."""
        retry = 0
        sleep_time = self.config.sleep_time
        
        while retry <= self.config.max_retry:
            try:
                with urllib.request.urlopen(url) as response:
                    return response.read().decode("utf-8")
                    
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.config.max_retry:
                    logger.warning(f"Rate limit hit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    sleep_time *= 2
                    retry += 1
                else:
                    raise e
            except Exception as e:
                if retry < self.config.max_retry:
                    logger.warning(f"Request failed, retrying: {e}")
                    time.sleep(sleep_time)
                    retry += 1
                else:
                    raise e
        
        raise Exception(f"Request failed after {self.config.max_retry} retries")
    
    def _convert_filters(self, filters: Dict[str, Any]) -> GEOSearchFilters:
        """Convert generic filters to GEOSearchFilters."""
        geo_filters = GEOSearchFilters()
        
        if 'organisms' in filters:
            geo_filters.organisms = filters['organisms']
        
        if 'platforms' in filters:
            geo_filters.platforms = filters['platforms']
        
        if 'entry_types' in filters:
            # Convert strings to GEOEntryType enums
            entry_types = []
            for et in filters['entry_types']:
                if isinstance(et, str):
                    try:
                        entry_types.append(GEOEntryType(et.lower()))
                    except ValueError:
                        continue
                elif isinstance(et, GEOEntryType):
                    entry_types.append(et)
            geo_filters.entry_types = entry_types
        
        if 'date_range' in filters:
            geo_filters.date_range = filters['date_range']
        
        if 'supplementary_file_types' in filters:
            geo_filters.supplementary_file_types = filters['supplementary_file_types']
        
        if 'max_results' in filters:
            geo_filters.max_results = filters['max_results']
        
        return geo_filters
    
    def format_geo_search_results(
        self, 
        search_result: GEOSearchResult, 
        original_query: str
    ) -> str:
        """Format GEO search results for display."""
        response = "## 🧬 GEO DataSets Search Results\n\n"
        response += f"**Query**: `{search_result.query or original_query}`\n"
        response += f"**Total Results**: {search_result.count:,}\n"
        response += f"**Showing**: {len(search_result.ids)} datasets\n\n"
        
        if not search_result.summaries:
            # Just show IDs if no summaries available
            response += "### Dataset IDs Found\n"
            for i, dataset_id in enumerate(search_result.ids[:10], 1):
                response += f"{i}. **GDS{dataset_id}** - [View on GEO](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc=GDS{dataset_id})\n"
            
            if len(search_result.ids) > 10:
                response += f"\n... and {len(search_result.ids) - 10} more datasets.\n"
        else:
            # Show detailed summaries
            for i, summary in enumerate(search_result.summaries, 1):
                response += f"### Dataset {i}/{len(search_result.summaries)}\n"
                
                accession = summary.get('Accession', f"GDS{summary.get('uid', '')}")
                response += f"**Accession**: [{accession}](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={accession})\n"
                response += f"**Title**: {summary.get('title', 'N/A')}\n"
                
                if summary.get('summary'):
                    desc = summary['summary'][:300]
                    if len(summary['summary']) > 300:
                        desc += "..."
                    response += f"**Description**: {desc}\n"
                
                if summary.get('taxon'):
                    response += f"**Organism**: {summary['taxon']}\n"
                
                if summary.get('GPL'):
                    response += f"**Platform**: {summary['GPL']}\n"
                
                if summary.get('n_samples'):
                    response += f"**Samples**: {summary['n_samples']}\n"
                
                if summary.get('PDAT'):
                    response += f"**Date**: {summary['PDAT']}\n"
                
                response += "\n---\n\n"
        
        # Add helpful tips
        response += "### 💡 Next Steps\n"
        response += "- Use `download_geo_dataset('GSE123456')` to download specific datasets\n"
        response += "- Refine your search with filters like organisms, platforms, or date ranges\n"
        response += "- Check the GEO website for supplementary files and detailed protocols\n"
        
        return response
    
    def _format_detailed_dataset_report(
        self, 
        summary: Dict[str, Any], 
        accession: str
    ) -> str:
        """Format detailed information for a specific GEO dataset."""
        response = f"## 📊 GEO Dataset Details: {accession}\n\n"
        
        # Basic information
        response += "### Basic Information\n"
        response += f"**Accession**: [{accession}](https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={accession})\n"
        response += f"**Title**: {summary.get('title', 'N/A')}\n"
        
        if summary.get('summary'):
            response += f"**Description**: {summary['summary']}\n"
        
        response += "\n### Dataset Metadata\n"
        
        if summary.get('taxon'):
            response += f"**Organism**: {summary['taxon']}\n"
        
        if summary.get('GPL'):
            platform_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={summary['GPL']}"
            response += f"**Platform**: [{summary['GPL']}]({platform_url})\n"
        
        if summary.get('n_samples'):
            response += f"**Sample Count**: {summary['n_samples']}\n"
        
        if summary.get('PDAT'):
            response += f"**Publication Date**: {summary['PDAT']}\n"
        
        if summary.get('entryType'):
            response += f"**Entry Type**: {summary['entryType']}\n"
        
        # Publication links
        if summary.get('PubMedIds'):
            response += "\n### Related Publications\n"
            pmid = summary['PubMedIds'].get('int', '')
            if pmid:
                pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                response += f"**PubMed**: [{pmid}]({pmid_url})\n"
        
        # Download and access information
        response += "\n### 💾 Access Options\n"
        response += f"- **GEO Browser**: [View dataset]({f'https://www.ncbi.nlm.nih.gov/sites/GDSbrowser?acc={accession}'})\n"
        response += f"- **Download**: Use `download_geo_dataset('{accession}')` command\n"
        response += f"- **FTP Access**: Check GEO FTP site for raw files\n"
        
        return response
