"""
Enhanced PubMed and NCBI Entrez service for scientific literature and dataset discovery.

This service provides comprehensive functionality for searching PubMed articles
and discovering omics datasets from NCBI databases including GEO, SRA, and others.
Now optimized for DataManagerV2 integration.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from datetime import datetime
from enum import Enum

from langchain_core.documents import Document
from pydantic import BaseModel, Field, model_validator

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class OmicsDataType(Enum):
    """Enum for different omics data types - modular design for extensibility."""
    GEO = "geo"  # Gene Expression Omnibus
    SRA = "sra"  # Sequence Read Archive
    DBGAP = "dbgap"  # Database of Genotypes and Phenotypes
    BIOPROJECT = "bioproject"  # BioProject
    BIOSAMPLE = "biosample"  # BioSample
    PROTEOMICS = "proteomics"  # ProteomicsDB
    METABOLOMICS = "metabolomics"  # Metabolomics Workbench


class NCBIDatabase(Enum):
    """NCBI database identifiers."""
    PUBMED = "pubmed"
    GEO = "gds"  # GEO DataSets
    SRA = "sra"
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"


class EnhancedPubMedServiceConfig(BaseModel):
    """Enhanced configuration for PubMed and NCBI services."""
    
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


class PubMedService(BaseModel):
    """
    Enhanced service for PubMed literature search and NCBI dataset discovery.
    
    This service provides:
    - Advanced PubMed search with parameter extraction
    - Direct NCBI Entrez access for dataset discovery
    - Modular support for multiple omics data types
    - Computational methodology extraction
    - GitHub repository and software tool identification
    """

    parse: Any
    data_manager: Any
    config: EnhancedPubMedServiceConfig = EnhancedPubMedServiceConfig()

    # NCBI E-utilities base URLs
    base_url_esearch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    base_url_esummary: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
    base_url_elink: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?"

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate required packages."""
        try:
            import xmltodict
            values["parse"] = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict. Install with: pip install xmltodict"
            )
        return values

    def find_datasets_for_study(
        self,
        query: str,
        data_type: OmicsDataType = OmicsDataType.GEO,
        filters: Optional[Dict[str, str]] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Find omics datasets directly from NCBI databases for a given study/query.
        
        Args:
            query: Search query (e.g., "single-cell RNA-seq macrophage")
            data_type: Type of omics data to search for
            filters: Additional filters (e.g., {"organism": "human", "year": "2023"})
            top_k: Number of results to retrieve
            
        Returns:
            str: Formatted list of discovered datasets with metadata
        """
        k_results = top_k if top_k is not None else self.config.top_k_results
        
        logger.info(f"Searching for {data_type.value} datasets with query: {query[:50]}...")
        
        try:
            # Map data type to NCBI database
            db_map = {
                OmicsDataType.GEO: NCBIDatabase.GEO,
                OmicsDataType.SRA: NCBIDatabase.SRA,
                OmicsDataType.BIOPROJECT: NCBIDatabase.BIOPROJECT,
                OmicsDataType.BIOSAMPLE: NCBIDatabase.BIOSAMPLE
            }
            
            if data_type not in db_map:
                return f"Data type {data_type.value} not yet supported for direct search"
            
            database = db_map[data_type]
            
            # Build enhanced query with filters
            enhanced_query = self._build_enhanced_query(query, data_type, filters)
            
            # Search the appropriate NCBI database
            datasets = self._search_ncbi_database(database, enhanced_query, k_results)
            
            if not datasets:
                return f"No {data_type.value} datasets found for query: {query}"
            
            # Format results based on data type
            formatted_results = self._format_dataset_results(datasets, data_type)
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="find_datasets_for_study",
                parameters={
                    "query": query[:100],
                    "data_type": data_type.value,
                    "filters": filters,
                    "top_k": k_results
                },
                description=f"Searched for {data_type.value} datasets"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            return f"Error searching for datasets: {str(e)}"

    def extract_computational_methods(
        self,
        doi_or_pmid: str,
        method_type: str = "all",
        include_parameters: bool = True
    ) -> str:
        """
        Extract computational methods and parameters from a publication.
        
        Args:
            doi_or_pmid: DOI or PMID of the publication
            method_type: Type of methods to extract (e.g., "preprocessing", "clustering", "all")
            include_parameters: Whether to extract specific parameter values
            
        Returns:
            str: Extracted methods with parameters and citations
        """
        logger.info(f"Extracting computational methods from: {doi_or_pmid}")
        
        try:
            # Determine if input is DOI or PMID
            is_doi = doi_or_pmid.startswith("10.")
            
            # Get the publication
            if is_doi:
                query = f"{doi_or_pmid}[DOI]"
            else:
                query = f"{doi_or_pmid}[PMID]"
            
            results = self._load_with_params(query, 1)
            
            if not results:
                return f"Publication not found: {doi_or_pmid}"
            
            article = results[0]
            summary = article.get('Summary', '')
            
            # Extract methods based on type
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
            
            # Extract methods from summary
            if method_type == "all":
                search_types = method_patterns.keys()
            else:
                search_types = [method_type] if method_type in method_patterns else []
            
            for mtype in search_types:
                if mtype not in method_patterns:
                    continue
                    
                for pattern in method_patterns[mtype]:
                    matches = re.finditer(pattern, summary, re.IGNORECASE)
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
            
            # Look for GitHub repositories
            github_pattern = r'github\.com/([\w-]+)/([\w-]+)'
            github_matches = re.finditer(github_pattern, summary, re.IGNORECASE)
            github_repos = [f"https://github.com/{m.group(1)}/{m.group(2)}" 
                           for m in github_matches]
            
            # Format the response
            response = f"## Computational Methods Extracted from {doi_or_pmid}\n\n"
            response += f"**Title**: {article.get('Title', 'N/A')}\n"
            response += f"**PMID**: {article.get('uid', 'N/A')}\n\n"
            
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
            if methods['tools']:
                response += "### Software Tools\n"
                unique_tools = list(set(m['text'] for m in methods['tools']))
                for tool in unique_tools:
                    response += f"- {tool}\n"
                response += "\n"
            
            # Add GitHub repositories
            if github_repos:
                response += "### GitHub Repositories\n"
                for repo in set(github_repos):
                    response += f"- {repo}\n"
                response += "\n"
            
            # Add parameter summary
            if methods['parameters'] and include_parameters:
                response += "### Extracted Parameters\n"
                for param, value in methods['parameters'].items():
                    response += f"- {param}: {value}\n"
                response += "\n"
            
            # Add recommendation for finding associated datasets
            response += "### Associated Datasets\n"
            response += "Use `find_geo_from_doi` or `find_datasets_for_study` to discover datasets.\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
            return f"Error extracting computational methods: {str(e)}"

    def find_geo_from_publication(
        self,
        doi: Optional[str] = None,
        pmid: Optional[str] = None,
        include_related: bool = True
    ) -> str:
        """
        Enhanced GEO dataset discovery from publications.
        
        Args:
            doi: DOI of the publication
            pmid: PubMed ID of the publication
            include_related: Include related datasets from linked publications
            
        Returns:
            str: Comprehensive list of GEO and other omics datasets
        """
        logger.info(f"Finding datasets from publication: DOI={doi}, PMID={pmid}")
        
        try:
            # Build query
            if doi:
                query = f"{doi}[DOI]"
            elif pmid:
                query = f"{pmid}[PMID]"
            else:
                return "Please provide either DOI or PMID"
            
            # Get the main publication
            results = self._load_with_params(query, 1)
            
            if not results:
                return f"Publication not found: {doi or pmid}"
            
            article = results[0]
            pmid = article.get('uid')
            
            # Extract accessions from abstract
            datasets = self._extract_dataset_accessions(article.get('Summary', ''))
            
            # Use NCBI E-link to find linked datasets
            if pmid and include_related:
                linked_datasets = self._find_linked_datasets(pmid)
                datasets.update(linked_datasets)
            
            # Check for supplementary materials
            if self.config.include_supplementary:
                supp_datasets = self._check_supplementary_materials(article)
                datasets.update(supp_datasets)
            
            # Format comprehensive response
            response = self._format_comprehensive_dataset_report(
                article, datasets, doi, pmid
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
            return f"Error finding datasets from publication: {str(e)}"

    def _build_enhanced_query(
        self,
        query: str,
        data_type: OmicsDataType,
        filters: Optional[Dict[str, str]] = None
    ) -> str:
        """Build enhanced query with filters for NCBI database search."""
        enhanced_query = query
        
        # Add data type specific enhancements
        if data_type == OmicsDataType.GEO:
            enhanced_query += " AND (GSE[ACCN] OR GSM[ACCN])"
            if filters:
                if 'organism' in filters:
                    enhanced_query += f" AND {filters['organism']}[ORGN]"
                if 'year' in filters:
                    enhanced_query += f" AND {filters['year']}[PDAT]"
                if 'platform' in filters:
                    enhanced_query += f" AND {filters['platform']}[PLAT]"
                    
        elif data_type == OmicsDataType.SRA:
            enhanced_query += " AND (SRP[ACCN] OR SRX[ACCN] OR SRR[ACCN])"
            
        return enhanced_query

    def _search_ncbi_database(
        self,
        database: NCBIDatabase,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """Search specific NCBI database using E-utilities."""
        url = (
            self.base_url_esearch +
            f"db={database.value}&term={urllib.parse.quote(query)}" +
            f"&retmode=json&retmax={max_results}&usehistory=y"
        )
        
        if self.config.api_key:
            url += f"&api_key={self.config.api_key}"
        
        logger.debug(f"NCBI search URL: {url}")
        
        try:
            result = urllib.request.urlopen(url)
            text = result.read().decode("utf-8")
            json_response = json.loads(text)
            
            if int(json_response["esearchresult"].get("count", "0")) == 0:
                return []
            
            # Get detailed information for each result
            webenv = json_response["esearchresult"]["webenv"]
            id_list = json_response["esearchresult"]["idlist"]
            
            # Fetch summaries for all IDs
            datasets = self._fetch_dataset_summaries(database, id_list, webenv)
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error searching NCBI database: {e}")
            return []

    def _fetch_dataset_summaries(
        self,
        database: NCBIDatabase,
        id_list: List[str],
        webenv: str
    ) -> List[Dict]:
        """Fetch detailed summaries for dataset IDs."""
        if not id_list:
            return []
        
        ids_str = ",".join(id_list)
        url = (
            self.base_url_esummary +
            f"db={database.value}&id={ids_str}&retmode=json&webenv={webenv}"
        )
        
        if self.config.api_key:
            url += f"&api_key={self.config.api_key}"
        
        try:
            result = urllib.request.urlopen(url)
            text = result.read().decode("utf-8")
            json_response = json.loads(text)
            
            summaries = []
            if "result" in json_response:
                for uid in id_list:
                    if uid in json_response["result"]:
                        summary = json_response["result"][uid]
                        summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error fetching dataset summaries: {e}")
            return []

    def _format_dataset_results(
        self,
        datasets: List[Dict],
        data_type: OmicsDataType
    ) -> str:
        """Format dataset results based on data type."""
        if not datasets:
            return f"No {data_type.value} datasets found"
        
        response = f"## Found {len(datasets)} {data_type.value.upper()} Datasets\n\n"
        
        for i, dataset in enumerate(datasets, 1):
            response += f"### Dataset {i}\n"
            
            if data_type == OmicsDataType.GEO:
                response += f"**Accession**: {dataset.get('Accession', 'N/A')}\n"
                response += f"**Title**: {dataset.get('title', 'N/A')}\n"
                response += f"**Summary**: {dataset.get('summary', 'N/A')[:500]}...\n"
                response += f"**Platform**: {dataset.get('GPL', 'N/A')}\n"
                response += f"**Samples**: {dataset.get('n_samples', 'N/A')}\n"
                response += f"**Organism**: {dataset.get('taxon', 'N/A')}\n"
                response += f"**Date**: {dataset.get('PDAT', 'N/A')}\n"
                
            elif data_type == OmicsDataType.SRA:
                response += f"**Accession**: {dataset.get('Run', 'N/A')}\n"
                response += f"**Title**: {dataset.get('Title', 'N/A')}\n"
                response += f"**Platform**: {dataset.get('Platform', 'N/A')}\n"
                response += f"**Total Spots**: {dataset.get('TotalSpots', 'N/A')}\n"
                response += f"**Total Bases**: {dataset.get('TotalBases', 'N/A')}\n"
                
            response += "\n---\n\n"
        
        response += f"\n💡 **To download**: Use the accession numbers with appropriate download tools.\n"
        
        return response

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
        # For now, return empty dict
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

    # Keep the original methods for backward compatibility but enhance them
    def search_pubmed(
        self,
        query: str,
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None,
        max_query_length: Optional[int] = None
    ) -> str:
        """Enhanced PubMed search with better result formatting."""
        k_results = top_k_results if top_k_results is not None else self.config.top_k_results
        content_max = doc_content_chars_max if doc_content_chars_max is not None else self.config.doc_content_chars_max
        query_max = max_query_length if max_query_length is not None else self.config.max_query_length
        
        logger.info(f"Enhanced PubMed search: {query[:50]}...")
        
        try:
            truncated_query = query[:query_max]
            results = self._load_with_params(truncated_query, k_results)
            
            if not results:
                return "No PubMed results found for your query."
            
            # Enhanced formatting with links and better structure
            docs = []
            for i, result in enumerate(results, 1):
                pmid = result.get('uid', 'N/A')
                title = result.get('Title', 'N/A')
                
                doc = f"### Result {i}/{len(results)}\n"
                doc += f"**Title**: {title}\n"
                doc += f"**PMID**: [{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)\n"
                doc += f"**Journal**: {result.get('Journal', 'N/A')}\n"
                doc += f"**Published**: {result.get('Published', 'N/A')}\n"
                
                # Check for datasets in abstract
                summary = result.get('Summary', 'No abstract available')
                datasets_found = self._extract_dataset_accessions(summary)
                
                if any(datasets_found.values()):
                    doc += "**Datasets Found**: "
                    dataset_list = []
                    for dtype, accs in datasets_found.items():
                        if accs:
                            dataset_list.append(f"{dtype}: {', '.join(accs[:3])}")
                    doc += "; ".join(dataset_list) + "\n"
                
                doc += f"**Abstract**:\n{summary[:1000]}..."
                docs.append(doc)
            
            formatted_results = "\n\n---\n\n".join(docs)
            
            # Add search summary
            header = f"## PubMed Search Results\n"
            header += f"**Query**: {query[:100]}{'...' if len(query) > 100 else ''}\n"
            header += f"**Results**: {len(results)} papers found\n\n"
            
            formatted_results = header + formatted_results
            
            if len(formatted_results) > content_max:
                formatted_results = formatted_results[:content_max] + "\n\n[Results truncated]"
            
            # Log the search
            self.data_manager.log_tool_usage(
                tool_name="search_pubmed_enhanced",
                parameters={
                    "query": query[:100],
                    "top_k_results": k_results,
                    "doc_content_chars_max": content_max
                },
                description="Enhanced PubMed literature search"
            )
            
            return formatted_results
            
        except Exception as e:
            logger.exception(f"PubMed search error: {e}")
            return f"PubMed search error: {str(e)}"

    def find_geo_from_doi(
        self,
        doi: str,
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None
    ) -> str:
        """Enhanced version that uses the new comprehensive method."""
        return self.find_geo_from_publication(doi=doi, include_related=True)

    def find_geo_from_pmid(
        self,
        pmid: str,
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None
    ) -> str:
        """Enhanced version that uses the new comprehensive method."""
        return self.find_geo_from_publication(pmid=pmid, include_related=True)

    # Keep other original methods with minimal changes for compatibility
    def _load_with_params(self, query: str, top_k_results: int) -> List[dict]:
        """Load PubMed results with specific parameters."""
        return list(self._lazy_load_with_params(query, top_k_results))

    def _lazy_load_with_params(self, query: str, top_k_results: int) -> Iterator[dict]:
        """Lazy load PubMed results."""
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
