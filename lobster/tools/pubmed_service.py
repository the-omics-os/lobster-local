"""
PubMed service for retrieving scientific literature data.

This service provides functionality to search and retrieve articles from PubMed,
supporting bioinformaticians with literature-based evidence and information.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, model_validator

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PubMedServiceConfig(BaseModel):
    """Configuration class for PubMed service parameters."""
    
    top_k_results: int = 3
    max_query_length: int = 300
    doc_content_chars_max: int = 4000
    email: str = "genie_ai@example.com"
    api_key: str = ""
    max_retry: int = 5
    sleep_time: float = 0.2
    
    def model_dump_safe(self) -> Dict[str, Any]:
        """Return config as dict, hiding sensitive information."""
        config_dict = self.model_dump()
        if config_dict.get('api_key'):
            config_dict['api_key'] = '***'
        return config_dict


class PubMedService(BaseModel):
    """
    Service for interacting with PubMed API.

    This service allows searching PubMed for scientific articles, retrieving metadata,
    abstracts, and supporting various bioinformatics research tasks such as:
    - Finding GEO dataset references from DOIs
    - Identifying marker genes for specific cell types or diseases
    - Retrieving protocol information for specific techniques
    - Finding relevant papers for specific bioinformatics analyses
    
    The service supports dynamic parameter configuration for flexible usage.
    """

    parse: Any  #: :meta private:
    data_manager: Any
    config: PubMedServiceConfig = PubMedServiceConfig()

    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the python package exists in environment."""
        try:
            import xmltodict
            values["parse"] = xmltodict.parse
        except ImportError:
            raise ImportError(
                "Could not import xmltodict python package. "
                "Please install it with `pip install xmltodict`."
            )
        return values

    def update_config(self, **kwargs) -> None:
        """
        Update service configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
                - top_k_results: Number of results to retrieve (default: 3)
                - max_query_length: Maximum query length (default: 300)
                - doc_content_chars_max: Maximum content length (default: 4000)
                - email: Contact email for API (default: "genie_ai@example.com")
                - api_key: NCBI API key (default: "")
                - max_retry: Maximum retry attempts (default: 5)
                - sleep_time: Sleep time between retries (default: 0.2)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated PubMed config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def search_pubmed(
        self, 
        query: str, 
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None,
        max_query_length: Optional[int] = None
    ) -> str:
        """
        Search PubMed for articles matching the query and return formatted results.

        Args:
            query: The search query string
            top_k_results: Number of results to retrieve (overrides config default)
            doc_content_chars_max: Maximum content length (overrides config default)
            max_query_length: Maximum query length (overrides config default)

        Returns:
            str: Formatted results from PubMed search
        """
        # Use provided parameters or fall back to config defaults
        k_results = top_k_results if top_k_results is not None else self.config.top_k_results
        content_max = doc_content_chars_max if doc_content_chars_max is not None else self.config.doc_content_chars_max
        query_max = max_query_length if max_query_length is not None else self.config.max_query_length
        
        logger.info(f"Starting PubMed search with query: {query[:50]}...")
        logger.debug(f"Full query: {query}")
        logger.debug(f"Parameters: top_k={k_results}, content_max={content_max}, query_max={query_max}")

        try:
            # Truncate query if too long
            truncated_query = query[:query_max]
            if len(query) > query_max:
                logger.warning(f"Query truncated from {len(query)} to {query_max} characters")
            
            logger.info(f"Retrieving top {k_results} results from PubMed")
            
            # Retrieve the top-k results for the query
            results = self._load_with_params(truncated_query, k_results)
            
            if not results:
                logger.info("No PubMed results found for query")
                return "No PubMed results found for your query."
            
            logger.info(f"Found {len(results)} PubMed articles")
                
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="search_pubmed",
                parameters={
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "top_k_results": k_results,
                    "doc_content_chars_max": content_max
                },
                description="Retrieved scientific literature from PubMed database"
            )
            
            # Format the results
            docs = []
            for i, result in enumerate(results):
                logger.debug(f"Processing result {i+1}: PMID {result.get('uid', 'N/A')}")
                doc = f"**PMID**: {result.get('uid', 'N/A')}\n"
                doc += f"**Title**: {result.get('Title', 'N/A')}\n"
                doc += f"**Published**: {result.get('Published', 'N/A')}\n"
                doc += f"**Summary**:\n{result.get('Summary', 'No abstract available')}"
                docs.append(doc)

            # Join the results and limit the character count
            formatted_results = "\n\n---\n\n".join(docs)
            
            if len(formatted_results) > content_max:
                logger.warning(f"Results truncated from {len(formatted_results)} to {content_max} characters")
                formatted_results = formatted_results[:content_max]
            
            logger.info("PubMed search completed successfully")
            return formatted_results if docs else "No relevant PubMed results were found"
            
        except Exception as ex:
            logger.exception(f"PubMed search error: {ex}")
            return f"PubMed search encountered an error: {ex}"

    def find_geo_from_doi(
        self, 
        doi: str,
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None
    ) -> str:
        """
        Find GEO accession numbers mentioned in publications with the given DOI.
        
        Args:
            doi: The DOI of the publication
            top_k_results: Number of results to retrieve (overrides config default)
            doc_content_chars_max: Maximum content length (overrides config default)
            
        Returns:
            str: GEO accession numbers found, if any
        """
        # Use provided parameters or fall back to config defaults
        k_results = top_k_results if top_k_results is not None else self.config.top_k_results
        content_max = doc_content_chars_max if doc_content_chars_max is not None else self.config.doc_content_chars_max
        
        logger.info(f"Searching for GEO accessions related to DOI: {doi}")
        
        # Validate the DOI format
        import re
        doi_pattern = re.compile(r'^10\.\d{4,9}/[-._;()/:A-Z0-9]+$', re.IGNORECASE)
        
        # Clean the DOI if it contains a URL
        if "doi.org" in doi:
            doi = doi.split("doi.org/")[-1]
        
        # Check if this is a valid DOI format (typically starts with 10.)
        if not doi_pattern.match(doi):
            logger.warning(f"Invalid DOI format: {doi}")
            return f"Invalid DOI format: '{doi}'. Please provide a properly formatted DOI, which typically starts with '10.' (e.g., 10.1038/s41586-021-03659-0)."
            
        # Query to find the article by DOI
        query = f"{doi}[DOI]"
        
        try:
            # First get the basic article metadata
            results = self._load_with_params(query, k_results)
            
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="find_geo_from_doi",
                parameters={
                    "doi": doi,
                    "top_k_results": k_results,
                    "doc_content_chars_max": content_max
                },
                description="Searched for GEO accession numbers related to a publication DOI"
            )
            
            if not results:
                return f"No publications found with DOI {doi}"
            
            # Extract PMIDs for further detailed retrieval
            pmids = [result.get('uid') for result in results if result.get('uid')]
            
            # Extract GEO accession numbers from full articles
            geo_pattern = re.compile(r'GSE\d+')
            sra_pattern = re.compile(r'SRP\d+|SRR\d+|SRX\d+')  # Additional patterns for SRA data
            array_pattern = re.compile(r'GPL\d+')  # Pattern for GEO platform accessions
            
            geo_accessions = []
            sra_accessions = []
            platform_accessions = []
            
            # Search for accessions in the abstracts first
            for result in results:
                summary = result.get('Summary', '')
                # Look for GEO dataset accessions
                geo_matches = geo_pattern.findall(summary)
                if geo_matches:
                    geo_accessions.extend(geo_matches)
                
                # Look for SRA accessions
                sra_matches = sra_pattern.findall(summary)
                if sra_matches:
                    sra_accessions.extend(sra_matches)
                    
                # Look for platform accessions
                platform_matches = array_pattern.findall(summary)
                if platform_matches:
                    platform_accessions.extend(platform_matches)
            
            # Format detailed publication info with or without accessions
            publications = []
            for result in results:
                pub_info = f"Title: {result.get('Title', 'N/A')}\n"
                pub_info += f"PMID: {result.get('uid', 'N/A')}\n"
                pub_info += f"Journal: {result.get('Journal', 'N/A')}\n"
                pub_info += f"Publication Date: {result.get('Published', 'N/A')}\n"
                
                # Add any accessions found in this specific article
                summary = result.get('Summary', '')
                article_geo = geo_pattern.findall(summary)
                article_sra = sra_pattern.findall(summary)
                article_platform = array_pattern.findall(summary)
                
                if article_geo:
                    pub_info += f"GEO Accessions in this article: {', '.join(set(article_geo))}\n"
                if article_sra:
                    pub_info += f"SRA Accessions in this article: {', '.join(set(article_sra))}\n"
                if article_platform:
                    pub_info += f"GEO Platform Accessions in this article: {', '.join(set(article_platform))}\n"
                    
                publications.append(pub_info)
                
            # Prepare the response
            response = f"Analyzed publication(s) with DOI {doi}:\n\n"
            
            # Deduplicate and add all found accessions to the report
            all_geo = list(set(geo_accessions))
            all_sra = list(set(sra_accessions))
            all_platform = list(set(platform_accessions))
            
            if all_geo:
                response += f"Found {len(all_geo)} GEO dataset accession(s):\n"
                response += ", ".join(all_geo) + "\n\n"
            
            if all_sra:
                response += f"Found {len(all_sra)} SRA accession(s):\n"
                response += ", ".join(all_sra) + "\n\n"
                
            if all_platform:
                response += f"Found {len(all_platform)} GEO platform accession(s):\n"
                response += ", ".join(all_platform) + "\n\n"
                
            if not all_geo and not all_sra and not all_platform:
                response += "No GEO or SRA accession numbers were found in the publication abstracts.\n\n"
            
            # Add publication details
            response += "Publication Details:\n\n"
            response += "\n\n---\n\n".join(publications)
            
            # Limit response length
            if len(response) > content_max:
                logger.warning(f"Response truncated from {len(response)} to {content_max} characters")
                response = response[:content_max]
            
            return response
                
        except Exception as ex:
            logger.error(f"Error finding GEO from DOI: {ex}")
            return f"Error searching for GEO accessions related to DOI {doi}: {ex}"

    def find_marker_genes(
        self, 
        cell_type: str, 
        disease: Optional[str] = None,
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None
    ) -> str:
        """
        Find marker genes for specific cell types, optionally in the context of a disease.
        
        Args:
            cell_type: The cell type to find markers for (e.g., "T cells", "macrophages")
            disease: Optional disease context (e.g., "rheumatoid arthritis", "cancer")
            top_k_results: Number of results to retrieve (overrides config default)
            doc_content_chars_max: Maximum content length (overrides config default)
            
        Returns:
            str: Information about marker genes for the specified cell type
        """
        # Use provided parameters or fall back to config defaults
        k_results = top_k_results if top_k_results is not None else self.config.top_k_results
        content_max = doc_content_chars_max if doc_content_chars_max is not None else self.config.doc_content_chars_max
        
        logger.info(f"Searching for marker genes for {cell_type}" + 
                   (f" in {disease}" if disease else ""))
        
        # Construct query based on input parameters
        query = f"{cell_type} marker genes"
        if disease:
            query += f" {disease}"
            
        query += " single-cell RNA-seq"
        
        try:
            results = self._load_with_params(query, k_results)
            
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="find_marker_genes_literature",
                parameters={
                    "cell_type": cell_type, 
                    "disease": disease if disease else "None",
                    "top_k_results": k_results,
                    "doc_content_chars_max": content_max
                },
                description=f"Searched for marker genes for {cell_type}" + 
                           (f" in {disease}" if disease else "")
            )
            
            if not results:
                return f"No literature found for marker genes of {cell_type}" + \
                       (f" in {disease}" if disease else "")
                
            # Format the results
            papers = [
                f"**Title**: {result.get('Title', 'N/A')}\n"
                f"**PMID**: {result.get('uid', 'N/A')}\n"
                f"**Summary**:\n{result.get('Summary', 'No abstract available')}"
                for result in results
            ]
            
            formatted_results = f"Found literature about marker genes for {cell_type}" + \
                               (f" in {disease}" if disease else "") + ":\n\n" + \
                               "\n\n---\n\n".join(papers)
            
            # Limit response length
            if len(formatted_results) > content_max:
                logger.warning(f"Results truncated from {len(formatted_results)} to {content_max} characters")
                formatted_results = formatted_results[:content_max]
                
            return formatted_results
                
        except Exception as ex:
            logger.error(f"Error finding marker genes: {ex}")
            return f"Error searching for marker genes: {ex}"

    def find_protocol_information(
        self, 
        technique: str,
        top_k_results: Optional[int] = None,
        doc_content_chars_max: Optional[int] = None
    ) -> str:
        """
        Find protocol information for specific bioinformatics techniques.
        
        Args:
            technique: The technique to find protocol information for
            top_k_results: Number of results to retrieve (overrides config default)
            doc_content_chars_max: Maximum content length (overrides config default)
            
        Returns:
            str: Protocol information for the specified technique
        """
        # Use provided parameters or fall back to config defaults
        k_results = top_k_results if top_k_results is not None else self.config.top_k_results
        content_max = doc_content_chars_max if doc_content_chars_max is not None else self.config.doc_content_chars_max
        
        logger.info(f"Searching for protocol information for {technique}")
        
        query = f"{technique} protocol bioinformatics methodology"
        
        try:
            results = self._load_with_params(query, k_results)
            
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="find_protocol_information",
                parameters={
                    "technique": technique,
                    "top_k_results": k_results,
                    "doc_content_chars_max": content_max
                },
                description=f"Searched for protocol information for {technique}"
            )
            
            if not results:
                return f"No protocol information found for {technique}"
                
            # Format the results
            protocols = [
                f"**Title**: {result.get('Title', 'N/A')}\n"
                f"**PMID**: {result.get('uid', 'N/A')}\n"
                f"**Published**: {result.get('Published', 'N/A')}\n"
                f"**Summary**:\n{result.get('Summary', 'No abstract available')}"
                for result in results
            ]
            
            formatted_results = f"Found protocol information for {technique}:\n\n" + \
                               "\n\n---\n\n".join(protocols)
            
            # Limit response length
            if len(formatted_results) > content_max:
                logger.warning(f"Results truncated from {len(formatted_results)} to {content_max} characters")
                formatted_results = formatted_results[:content_max]
                
            return formatted_results
                
        except Exception as ex:
            logger.error(f"Error finding protocol information: {ex}")
            return f"Error searching for protocol information: {ex}"

    def _load_with_params(self, query: str, top_k_results: int) -> List[dict]:
        """
        Load PubMed results with specific parameters.
        
        Args:
            query: The search query string
            top_k_results: Number of results to retrieve
            
        Returns:
            List[dict]: List of document metadata dictionaries
        """
        return list(self._lazy_load_with_params(query, top_k_results))

    def _lazy_load_with_params(self, query: str, top_k_results: int) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query with specific parameters.
        Return an iterator of dictionaries containing the document metadata.
        
        Args:
            query: The search query string
            top_k_results: Number of results to retrieve
            
        Returns:
            Iterator[dict]: Iterator of document metadata dictionaries
        """
        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + urllib.parse.quote(query)
            + f"&retmode=json&retmax={top_k_results}&usehistory=y"
        )
        if self.config.api_key:
            url += f"&api_key={self.config.api_key}"
            
        logger.debug(f"PubMed search URL: {url}")
        
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)

        # Check if any results were found
        if int(json_text["esearchresult"].get("count", "0")) == 0:
            logger.info(f"No PubMed results found for query: {query[:50]}...")
            return
            
        webenv = json_text["esearchresult"]["webenv"]
        for uid in json_text["esearchresult"]["idlist"]:
            yield self.retrieve_article(uid, webenv)

    # Legacy methods for backward compatibility
    def lazy_load(self, query: str) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        
        Args:
            query: The search query string
            
        Returns:
            Iterator[dict]: Iterator of document metadata dictionaries
        """
        return self._lazy_load_with_params(query, self.config.top_k_results)

    def load(self, query: str) -> List[dict]:
        """
        Search PubMed for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        
        Args:
            query: The search query string
            
        Returns:
            List[dict]: List of document metadata dictionaries
        """
        return self._load_with_params(query, self.config.top_k_results)

    def _dict2document(self, doc: dict) -> Document:
        """
        Convert a dictionary to a Document object.
        
        Args:
            doc: Dictionary containing document data
            
        Returns:
            Document: LangChain Document object
        """
        summary = doc.pop("Summary")
        return Document(page_content=summary, metadata=doc)

    def lazy_load_docs(self, query: str) -> Iterator[Document]:
        """
        Search PubMed and return documents as LangChain Document objects.
        
        Args:
            query: The search query string
            
        Returns:
            Iterator[Document]: Iterator of Document objects
        """
        for d in self.lazy_load(query=query):
            yield self._dict2document(d)

    def load_docs(self, query: str) -> List[Document]:
        """
        Search PubMed and return documents as a list of LangChain Document objects.
        
        Args:
            query: The search query string
            
        Returns:
            List[Document]: List of Document objects
        """
        return list(self.lazy_load_docs(query))

    def retrieve_article(self, uid: str, webenv: str) -> dict:
        """
        Retrieve article metadata for a specific PubMed ID.
        
        Args:
            uid: PubMed ID
            webenv: Web environment string from PubMed API
            
        Returns:
            dict: Article metadata
        """
        url = (
            self.base_url_efetch
            + "db=pubmed&retmode=xml&id="
            + uid
            + "&webenv="
            + webenv
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
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    logger.warning(
                        f"Too Many Requests from PubMed, "
                        f"waiting for {self.config.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.config.sleep_time)
                    self.config.sleep_time *= 2
                    retry += 1
                else:
                    logger.error(f"HTTP error while retrieving article {uid}: {e}")
                    raise e

        xml_text = result.read().decode("utf-8")
        text_dict = self.parse(xml_text)
        return self._parse_article(uid, text_dict)

    def _parse_article(self, uid: str, text_dict: dict) -> dict:
        """
        Parse article metadata from PubMed XML response.
        
        Args:
            uid: PubMed ID
            text_dict: Dictionary containing parsed XML data
            
        Returns:
            dict: Structured article metadata
        """
        try:
            ar = text_dict["PubmedArticleSet"]["PubmedArticle"]["MedlineCitation"][
                "Article"
            ]
        except KeyError:
            try:
                ar = text_dict["PubmedArticleSet"]["PubmedBookArticle"]["BookDocument"]
            except KeyError:
                logger.warning(f"Unexpected PubMed XML structure for article {uid}")
                return {
                    "uid": uid,
                    "Title": "Could not parse article data",
                    "Published": "",
                    "Summary": "The article data could not be parsed due to unexpected structure."
                }
                
        abstract_text = ar.get("Abstract", {}).get("AbstractText", [])
        summaries = []
        
        # Handle different abstract text structures
        if isinstance(abstract_text, list):
            for txt in abstract_text:
                if isinstance(txt, dict) and "#text" in txt and "@Label" in txt:
                    summaries.append(f"{txt['@Label']}: {txt['#text']}")
                elif isinstance(txt, dict) and "#text" in txt:
                    summaries.append(txt["#text"])
                elif isinstance(txt, str):
                    summaries.append(txt)
        elif isinstance(abstract_text, dict):
            if "#text" in abstract_text:
                summaries.append(abstract_text["#text"])
            else:
                for key, value in abstract_text.items():
                    if key != "@Label" and isinstance(value, str):
                        summaries.append(value)
        elif isinstance(abstract_text, str):
            summaries.append(abstract_text)
            
        summary = "\n".join(summaries) if summaries else "No abstract available"
        
        # Get publication date
        pub_date = ""
        try:
            # Try to get ArticleDate first
            if "ArticleDate" in ar:
                a_d = ar["ArticleDate"]
                if isinstance(a_d, list):
                    a_d = a_d[0]
                pub_date = "-".join(
                    [
                        a_d.get("Year", ""),
                        a_d.get("Month", ""),
                        a_d.get("Day", ""),
                    ]
                )
            # Try PubDate from Journal/JournalIssue/PubDate as fallback
            elif "Journal" in ar and "JournalIssue" in ar["Journal"] and "PubDate" in ar["Journal"]["JournalIssue"]:
                p_d = ar["Journal"]["JournalIssue"]["PubDate"]
                pub_date = "-".join(
                    [
                        p_d.get("Year", ""),
                        p_d.get("Month", ""),
                        p_d.get("Day", ""),
                    ]
                )
        except Exception as e:
            logger.warning(f"Error parsing publication date for article {uid}: {e}")
            
        # Get journal information
        journal = ""
        try:
            if "Journal" in ar and "Title" in ar["Journal"]:
                journal = ar["Journal"]["Title"]
        except Exception as e:
            logger.warning(f"Error parsing journal for article {uid}: {e}")

        return {
            "uid": uid,
            "Title": ar.get("ArticleTitle", ""),
            "Published": pub_date,
            "Journal": journal,
            "Copyright Information": ar.get("Abstract", {}).get(
                "CopyrightInformation", ""
            ),
            "Summary": summary,
        }
