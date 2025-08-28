"""
PubMed provider implementation for publication search and dataset discovery.

This provider implements the BasePublicationProvider interface for PubMed/NCBI
and provides comprehensive literature search and omics dataset discovery capabilities.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from lobster.tools.providers.base_provider import (
    BasePublicationProvider, 
    PublicationSource, 
    DatasetType, 
    PublicationMetadata, 
    DatasetMetadata
)
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class NCBIDatabase:
    """NCBI database identifiers."""
    PUBMED = "pubmed"
    GEO = "gds"  # GEO DataSets
    SRA = "sra"
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"


class PubMedProviderConfig(BaseModel):
    """Configuration for PubMed provider."""
    
    # PubMed settings
    top_k_results: int = Field(default=5, ge=1, le=100)
    max_query_length: int = Field(default=500, ge=100, le=1000)
    doc_content_chars_max: int = Field(default=6000, ge=1000, le=20000)
    
    # NCBI API settings
    email: str = "kevin.yar@homara.ai"
    api_key: Optional[str] = ""
    max_retry: int = Field(default=5, ge=1, le=10)
    sleep_time: float = Field(default=0.3, ge=0.1, le=5.0)
    
    # Dataset discovery settings
    include_geo_datasets: bool = True
    include_sra_datasets: bool = True
    include_supplementary: bool = True
    
    # Method extraction settings
    extract_parameters: bool = True
    extract_github_links: bool = True
    extract_software_tools: bool = True


class PubMedProvider(BasePublicationProvider):
    """
    PubMed provider for literature search and NCBI dataset discovery.
    
    This provider implements comprehensive PubMed/NCBI functionality including:
    - Advanced literature search with parameter extraction
    - Direct NCBI Entrez access for dataset discovery  
    - Support for GEO, SRA, and other omics databases
    - Computational methodology extraction
    - GitHub repository and software tool identification
    """
    
    def __init__(
        self, 
        data_manager: DataManagerV2,
        config: Optional[PubMedProviderConfig] = None
    ):
        """
        Initialize PubMed provider.
        
        Args:
            data_manager: DataManagerV2 instance for provenance tracking
            config: Optional configuration, uses defaults if not provided
        """
        self.data_manager = data_manager
        self.config = config or PubMedProviderConfig()
        
        # Initialize XML parser
        try:
            import xmltodict
            self.parse = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict. Install with: pip install xmltodict"
            )
        
        # NCBI E-utilities base URLs
        self.base_url_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        self.base_url_efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        self.base_url_esummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        self.base_url_elink = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"
    
    @property
    def source(self) -> PublicationSource:
        """Return PubMed as the publication source."""
        return PublicationSource.PUBMED
    
    @property
    def supported_dataset_types(self) -> List[DatasetType]:
        """Return list of dataset types supported by NCBI."""
        return [
            DatasetType.GEO,
            DatasetType.SRA,
            DatasetType.BIOPROJECT,
            DatasetType.BIOSAMPLE,
            DatasetType.DBGAP
        ]
    
    def validate_identifier(self, identifier: str) -> bool:
        """
        Validate PubMed/DOI identifiers.
        
        Args:
            identifier: Identifier to validate (PMID or DOI)
            
        Returns:
            bool: True if identifier is valid
        """
        identifier = identifier.strip()
        
        # Check for DOI format
        if identifier.startswith("10."):
            return True
        
        # Check for PMID format (numeric)
        if identifier.isdigit():
            return True
        
        # Check for PMID with prefix
        if identifier.upper().startswith("PMID:"):
            return identifier[5:].strip().isdigit()
        
        return False
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Return features supported by PubMed provider."""
        return {
            "literature_search": True,
            "dataset_discovery": True,
            "metadata_extraction": True,
            "full_text_access": False,
            "advanced_filtering": True,
            "computational_methods": True,
            "github_extraction": True
        }
    
    def search_publications(
        self,
        query: str,
        max_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Search PubMed for publications.
        
        Args:
            query: Search query string
            max_results: Maximum number of results (default: 5)
            filters: Optional filters (date_range, journal, etc.)
            **kwargs: Additional parameters
            
        Returns:
            str: Formatted search results
        """
        logger.info(f"PubMed search: {query[:50]}...")
        
        try:
            # Apply configuration limits
            k_results = min(max_results, self.config.top_k_results)
            truncated_query = query[:self.config.max_query_length]
            
            # Apply filters if provided
            if filters:
                truncated_query = self._apply_search_filters(truncated_query, filters)
            
            # Load results using NCBI API
            results = list(self._load_with_params(truncated_query, k_results))
            
            if not results:
                return "No PubMed results found for your query."
            
            # Convert to PublicationMetadata objects
            pub_metadata = []
            for result in results:
                metadata = self._convert_to_metadata(result)
                pub_metadata.append(metadata)
            
            # Use base class formatting
            formatted_results = self.format_search_results(pub_metadata, query)
            
            # Add dataset discovery hints
            formatted_results += "\n\n💡 **Tip**: Use `find_datasets_from_publication()` with any PMID/DOI to discover associated datasets.\n"
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="pubmed_search",
                parameters={
                    "query": query[:100],
                    "max_results": k_results,
                    "filters": filters
                },
                description="PubMed literature search"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.exception(f"PubMed search error: {e}")
            return f"PubMed search error: {str(e)}"
    
    def find_datasets_from_publication(
        self, 
        identifier: str,
        dataset_types: Optional[List[DatasetType]] = None,
        **kwargs
    ) -> str:
        """
        Find datasets associated with a PubMed publication.
        
        Args:
            identifier: DOI or PMID of the publication
            dataset_types: Types of datasets to search for
            **kwargs: Additional parameters (include_related, etc.)
            
        Returns:
            str: Formatted list of discovered datasets
        """
        logger.info(f"Finding datasets from publication: {identifier}")
        
        try:
            include_related = kwargs.get('include_related', True)
            
            # Determine identifier type and build query
            if identifier.startswith("10."):
                query = f"{identifier}[DOI]"
                is_doi = True
            elif identifier.isdigit() or identifier.upper().startswith("PMID:"):
                pmid = identifier.replace("PMID:", "").strip()
                query = f"{pmid}[PMID]"
                is_doi = False
            else:
                return f"Invalid identifier format: {identifier}. Please provide DOI or PMID."
            
            # Get the main publication
            results = list(self._load_with_params(query, 1))
            
            if not results:
                return f"Publication not found: {identifier}"
            
            article = results[0]
            pmid = article.get('uid')
            
            # Extract datasets using comprehensive approach
            datasets = self._extract_dataset_accessions(article.get('Summary', ''))
            
            # Use NCBI E-link to find linked datasets
            if pmid and include_related:
                linked_datasets = self._find_linked_datasets(pmid)
                datasets.update(linked_datasets)
            
            # Check for supplementary materials if enabled
            if self.config.include_supplementary:
                supp_datasets = self._check_supplementary_materials(article)
                datasets.update(supp_datasets)
            
            # Filter by requested dataset types if specified
            if dataset_types:
                filtered_datasets = {}
                type_map = {
                    DatasetType.GEO: 'GEO',
                    DatasetType.SRA: 'SRA',
                    DatasetType.ARRAYEXPRESS: 'ArrayExpress',
                    DatasetType.ENA: 'ENA'
                }
                for dtype in dataset_types:
                    if dtype in type_map and type_map[dtype] in datasets:
                        filtered_datasets[type_map[dtype]] = datasets[type_map[dtype]]
                datasets = filtered_datasets
            
            # Format comprehensive response
            response = self._format_comprehensive_dataset_report(
                article, datasets, identifier if is_doi else None, pmid
            )
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="find_datasets_from_publication",
                parameters={
                    "identifier": identifier,
                    "dataset_types": [dt.value for dt in dataset_types] if dataset_types else None,
                    "include_related": include_related
                },
                description="Dataset discovery from publication"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            return f"Error finding datasets from publication: {str(e)}"
    
    def extract_publication_metadata(
        self, 
        identifier: str,
        **kwargs
    ) -> PublicationMetadata:
        """
        Extract standardized metadata from a PubMed publication.
        
        Args:
            identifier: DOI or PMID of the publication
            **kwargs: Additional parameters
            
        Returns:
            PublicationMetadata: Standardized publication metadata
        """
        try:
            # Build query based on identifier type
            if identifier.startswith("10."):
                query = f"{identifier}[DOI]"
            elif identifier.isdigit() or identifier.upper().startswith("PMID:"):
                pmid = identifier.replace("PMID:", "").strip()
                query = f"{pmid}[PMID]"
            else:
                raise ValueError(f"Invalid identifier: {identifier}")
            
            # Get publication data
            results = list(self._load_with_params(query, 1))
            
            if not results:
                raise ValueError(f"Publication not found: {identifier}")
            
            article = results[0]
            return self._convert_to_metadata(article)
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return minimal metadata on error
            return PublicationMetadata(
                uid=identifier,
                title=f"Error extracting metadata: {str(e)}",
                abstract=f"Could not retrieve publication metadata for: {identifier}"
            )
    
    def extract_computational_methods(
        self,
        identifier: str,
        method_type: str = "all",
        include_parameters: bool = True
    ) -> str:
        """
        Extract computational methods and parameters from a publication.
        
        Args:
            identifier: DOI or PMID of the publication
            method_type: Type of methods to extract
            include_parameters: Whether to extract parameter values
            
        Returns:
            str: Extracted methods with parameters and citations
        """
        logger.info(f"Extracting computational methods from: {identifier}")
        
        try:
            # Get publication metadata first
            metadata = self.extract_publication_metadata(identifier)
            
            if not metadata.abstract:
                return f"No abstract available for method extraction: {identifier}"
            
            # Extract methods using pattern matching
            methods = self._extract_methods_from_text(
                metadata.abstract, method_type, include_parameters
            )
            
            # Format response
            response = f"## Computational Methods Extracted from {identifier}\n\n"
            response += f"**Title**: {metadata.title}\n"
            response += f"**PMID**: {metadata.pmid or 'N/A'}\n"
            response += f"**DOI**: {metadata.doi or 'N/A'}\n\n"
            
            # Add extracted methods
            for mtype, mlist in methods.items():
                if mlist and mtype not in ['parameters', 'tools']:
                    response += f"### {mtype.replace('_', ' ').title()}\n"
                    for method in mlist:
                        response += f"- {method['text']}\n"
                        if method.get('value'):
                            response += f"  * Extracted value: {method['value']}\n"
                    response += "\n"
            
            # Add tools
            if methods.get('tools'):
                response += "### Software Tools\n"
                unique_tools = list(set(m['text'] for m in methods['tools']))
                for tool in unique_tools:
                    response += f"- {tool}\n"
                response += "\n"
            
            # Add GitHub repositories
            github_repos = self._extract_github_repos(metadata.abstract)
            if github_repos:
                response += "### GitHub Repositories\n"
                for repo in github_repos:
                    response += f"- {repo}\n"
                response += "\n"
            
            # Add parameter summary
            if methods.get('parameters') and include_parameters:
                response += "### Extracted Parameters\n"
                for param, value in methods['parameters'].items():
                    response += f"- {param}: {value}\n"
                response += "\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
            return f"Error extracting computational methods: {str(e)}"
    
    # Helper methods (refactored from original PubMedService)
    
    def _apply_search_filters(self, query: str, filters: Dict[str, Any]) -> str:
        """Apply search filters to PubMed query."""
        filtered_query = query
        
        if 'date_range' in filters:
            date_range = filters['date_range']
            if isinstance(date_range, dict):
                start = date_range.get('start')
                end = date_range.get('end')
                if start:
                    filtered_query += f" AND {start}[PDAT]"
                if end:
                    filtered_query += f" AND {end}[PDAT]"
        
        if 'journal' in filters:
            filtered_query += f" AND {filters['journal']}[JOUR]"
        
        if 'author' in filters:
            filtered_query += f" AND {filters['author']}[AUTH]"
        
        if 'publication_type' in filters:
            filtered_query += f" AND {filters['publication_type']}[PTYP]"
        
        return filtered_query
    
    def _load_with_params(self, query: str, top_k_results: int) -> Iterator[dict]:
        """Load PubMed results with specific parameters."""
        url = (
            self.base_url_esearch +
            f"db=pubmed&term={urllib.parse.quote(query)}" +
            f"&retmode=json&retmax={top_k_results}&usehistory=y"
        )
        
        if self.config.api_key:
            url += f"&api_key={self.config.api_key}"
        
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)
        
        if int(json_text["esearchresult"].get("count", "0")) == 0:
            return
        
        webenv = json_text["esearchresult"]["webenv"]
        for uid in json_text["esearchresult"]["idlist"]:
            yield self.retrieve_article(uid, webenv)
    
    def retrieve_article(self, uid: str, webenv: str) -> dict:
        """Retrieve article metadata."""
        url = (
            self.base_url_efetch +
            f"db=pubmed&retmode=xml&id={uid}&webenv={webenv}"
        )
        
        if self.config.api_key:
            url += f"&api_key={self.config.api_key}"
        
        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.config.max_retry:
                    time.sleep(self.config.sleep_time)
                    self.config.sleep_time *= 2
                    retry += 1
                else:
                    raise e
        
        xml_text = result.read().decode("utf-8")
        text_dict = self.parse(xml_text)
        return self._parse_article(uid, text_dict)
    
    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        """Parse article metadata from XML."""
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"]["Article"]
        except KeyError:
            try:
                ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
            except KeyError:
                return {
                    "uid": uid,
                    "Title": "Could not parse article",
                    "Published": "",
                    "Summary": "Article data could not be parsed."
                }
        
        # Extract abstract
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = []
        
        if isinstance(abstract_text, list):
            for txt in abstract_text:
                if isinstance(txt, dict) and "#text" in txt:
                    label = txt.get("@Label", "")
                    text_content = txt["#text"]
                    summaries.append(f"{label}: {text_content}" if label else text_content)
                elif isinstance(txt, str):
                    summaries.append(txt)
        elif isinstance(abstract_text, dict) and "#text" in abstract_text:
            summaries.append(abstract_text["#text"])
        elif isinstance(abstract_text, str):
            summaries.append(abstract_text)
        
        summary = "\n".join(summaries) if summaries else "No abstract available"
        
        # Get dates
        pub_date = ""
        if "ArticleDate" in ar:
            a_d = ar["ArticleDate"]
            if isinstance(a_d, list):
                a_d = a_d[0]
            pub_date = f"{a_d.get('Year', '')}-{a_d.get('Month', '')}-{a_d.get('Day', '')}"
        elif "Journal" in ar and "JournalIssue" in ar["Journal"]:
            p_d = ar["Journal"]["JournalIssue"].get("PubDate", {})
            pub_date = f"{p_d.get('Year', '')}-{p_d.get('Month', '')}-{p_d.get('Day', '')}"
        
        # Get journal
        journal = ar.get("Journal", {}).get("Title", "")
        
        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "Journal": journal,
            "Copyright Information": ar.get("Abstract", {}).get("CopyrightInformation", ""),
            "Summary": summary
        }
    
    def _convert_to_metadata(self, article: dict) -> PublicationMetadata:
        """Convert article dict to PublicationMetadata."""
        return PublicationMetadata(
            uid=article.get('uid', ''),
            title=article.get('Title', ''),
            journal=article.get('Journal'),
            published=article.get('Published'),
            pmid=article.get('uid'),
            abstract=article.get('Summary')
        )
    
    def _extract_dataset_accessions(self, text: str) -> Dict[str, List[str]]:
        """Extract various dataset accessions from text."""
        accessions = {
            'GEO': [],
            'SRA': [],
            'ArrayExpress': [],
            'ENA': [],
            'Platforms': []
        }
        
        # Define patterns for different databases
        patterns = {
            'GEO': r'GSE\d+',
            'GEO_Sample': r'GSM\d+',
            'SRA': r'SR[APRSX]\d+',
            'ArrayExpress': r'E-\w+-\d+',
            'ENA': r'PR[JD][NE][AB]\d+',
            'Platforms': r'GPL\d+'
        }
        
        for name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if name == 'GEO_Sample':
                    accessions['GEO'].extend(matches)
                else:
                    key = name if name != 'Platforms' else name
                    accessions[key].extend(matches)
        
        # Deduplicate
        for key in accessions:
            accessions[key] = list(set(accessions[key]))
        
        return accessions
    
    def _find_linked_datasets(self, pmid: str) -> Dict[str, List[str]]:
        """Find datasets linked to a PubMed article using E-link."""
        linked = {'GEO': [], 'SRA': []}
        
        # Link to GEO
        url_geo = (
            self.base_url_elink +
            f"dbfrom=pubmed&db=gds&id={pmid}&retmode=json"
        )
        
        if self.config.api_key:
            url_geo += f"&api_key={self.config.api_key}"
        
        try:
            result = urllib.request.urlopen(url_geo)
            text = result.read().decode("utf-8")
            json_response = json.loads(text)
            
            if "linksets" in json_response:
                for linkset in json_response["linksets"]:
                    if "linksetdbs" in linkset:
                        for db in linkset["linksetdbs"]:
                            if db["dbto"] == "gds":
                                for link_id in db.get("links", []):
                                    linked['GEO'].append(f"GDS{link_id}")
                                    
        except Exception as e:
            logger.warning(f"Error finding linked GEO datasets: {e}")
        
        return linked
    
    def _check_supplementary_materials(self, article: Dict) -> Dict[str, List[str]]:
        """Check for dataset mentions in supplementary materials."""
        # This would require additional API calls or web scraping
        # For now, return empty dict as placeholder
        return {'Supplementary': []}
    
    def _format_comprehensive_dataset_report(
        self,
        article: Dict,
        datasets: Dict[str, List[str]],
        doi: Optional[str],
        pmid: Optional[str]
    ) -> str:
        """Format a comprehensive dataset discovery report."""
        response = "## 📊 Dataset Discovery Report\n\n"
        
        # Publication info
        response += "### 📄 Publication Information\n"
        response += f"**Title**: {article.get('Title', 'N/A')}\n"
        response += f"**PMID**: {pmid or 'N/A'}\n"
        response += f"**DOI**: {doi or 'N/A'}\n"
        response += f"**Journal**: {article.get('Journal', 'N/A')}\n"
        response += f"**Published**: {article.get('Published', 'N/A')}\n\n"
        
        # Datasets found
        total_datasets = sum(len(v) for v in datasets.values())
        
        if total_datasets == 0:
            response += "### ⚠️ No Datasets Found\n"
            response += "No dataset accessions were identified in this publication.\n"
            response += "Consider:\n"
            response += "- Checking the supplementary materials manually\n"
            response += "- Contacting the authors for data availability\n"
            response += "- Searching for related publications\n"
        else:
            response += f"### ✅ Found {total_datasets} Dataset(s)\n\n"
            
            if datasets.get('GEO'):
                response += "#### Gene Expression Omnibus (GEO)\n"
                for acc in datasets['GEO']:
                    response += f"- **{acc}** - [View on GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc})\n"
                response += "\n"
            
            if datasets.get('SRA'):
                response += "#### Sequence Read Archive (SRA)\n"
                for acc in datasets['SRA']:
                    response += f"- **{acc}** - [View on SRA](https://www.ncbi.nlm.nih.gov/sra/{acc})\n"
                response += "\n"
            
            if datasets.get('ArrayExpress'):
                response += "#### ArrayExpress\n"
                for acc in datasets['ArrayExpress']:
                    response += f"- **{acc}** - [View on ArrayExpress](https://www.ebi.ac.uk/arrayexpress/experiments/{acc}/)\n"
                response += "\n"
            
            if datasets.get('Platforms'):
                response += "#### Platform Information\n"
                for acc in datasets['Platforms']:
                    response += f"- **{acc}** - Platform/Array design\n"
                response += "\n"
        
        # Recommendations
        response += "### 💡 Next Steps\n"
        if datasets.get('GEO'):
            response += f"1. Download GEO dataset: Use `download_geo_dataset('{datasets['GEO'][0]}')`\n"
        response += "2. Extract methods: Use `extract_computational_methods()` for this publication\n"
        response += "3. Find similar studies: Search for related datasets with similar methodology\n"
        
        return response
    
    def _extract_methods_from_text(
        self, 
        text: str, 
        method_type: str, 
        include_parameters: bool
    ) -> Dict[str, List]:
        """Extract computational methods from text using pattern matching."""
        methods = {
            'preprocessing': [],
            'quality_control': [],
            'normalization': [],
            'clustering': [],
            'differential_expression': [],
            'trajectory': [],
            'integration': [],
            'tools': [],
            'parameters': {}
        }
        
        # Define method patterns for extraction
        method_patterns = {
            'preprocessing': [
                r'filter(?:ed|ing)?\s+(?:cells?\s+)?(?:with\s+)?(\d+(?:,\d+)?\s+(?:genes?|UMIs?|counts?))',
                r'(?:minimum|min)\s+(?:of\s+)?(\d+)\s+(?:genes?|features?)',
                r'(?:removed?|excluded?)\s+(?:cells?\s+)?(?:with\s+)?(?:less\s+than\s+)?(\d+)',
                r'mitochondrial\s+(?:gene\s+)?(?:content|percentage|threshold)\s*[<>]?\s*(\d+(?:\.\d+)?%?)'
            ],
            'quality_control': [
                r'quality\s+control\s+(?:parameters?|thresholds?|criteria)',
                r'(?:doublet|multiplet)\s+(?:detection|removal|filtering)',
                r'(?:MAD|median\s+absolute\s+deviation)\s*=?\s*(\d+(?:\.\d+)?)',
            ],
            'normalization': [
                r'(?:normalized?|normalization)\s+(?:using|with|by)\s+(\w+)',
                r'(SCTransform|LogNormalize|TPM|CPM|RPKM|FPKM)',
                r'scale\s*factor\s*[=:]?\s*(\d+(?:\.\d+)?(?:e[+-]?\d+)?)'
            ],
            'clustering': [
                r'(?:resolution|clustering\s+resolution)\s*[=:]?\s*(\d+(?:\.\d+)?)',
                r'k\s*[=:]?\s*(\d+)\s+(?:neighbors?|nearest)',
                r'(?:Louvain|Leiden|k-means|hierarchical)\s+clustering'
            ],
            'tools': [
                r'(Seurat|Scanpy|Cell\s*Ranger|STAR|kallisto|salmon)',
                r'(?:version|v)\s*(\d+(?:\.\d+)*)',
                r'(?:github\.com/[\w-]+/[\w-]+)'
            ]
        }
        
        # Extract methods from text
        if method_type == "all":
            search_types = method_patterns.keys()
        else:
            search_types = [method_type] if method_type in method_patterns else []
        
        for mtype in search_types:
            if mtype not in method_patterns:
                continue
                
            for pattern in method_patterns[mtype]:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    method_info = {
                        'type': mtype,
                        'pattern_matched': pattern,
                        'text': match.group(0),
                        'value': match.groups()[0] if match.groups() else None
                    }
                    methods[mtype].append(method_info)
                    
                    # Extract parameter values if requested
                    if include_parameters and match.groups():
                        param_key = f"{mtype}_{pattern.split('(')[0].strip()}"
                        methods['parameters'][param_key] = match.groups()[0]
        
        return methods
    
    def _extract_github_repos(self, text: str) -> List[str]:
        """Extract GitHub repositories from text."""
        github_pattern = r'github\.com/([\w-]+)/([\w-]+)'
        github_matches = re.finditer(github_pattern, text, re.IGNORECASE)
        return [f"https://github.com/{m.group(1)}/{m.group(2)}" 
                for m in github_matches]
