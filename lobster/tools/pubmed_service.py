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


class PubMedService(BaseModel):
    """
    Service for interacting with PubMed API.

    This service allows searching PubMed for scientific articles, retrieving metadata,
    abstracts, and supporting various bioinformatics research tasks such as:
    - Finding GEO dataset references from DOIs
    - Identifying marker genes for specific cell types or diseases
    - Retrieving protocol information for specific techniques
    - Finding relevant papers for specific bioinformatics analyses
    """

    parse: Any  #: :meta private:
    data_manager: Any

    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_efetch: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    max_retry: int = 5
    sleep_time: float = 0.2

    # Default values for the parameters
    top_k_results: int = 3
    MAX_QUERY_LENGTH: int = 300
    doc_content_chars_max: int = 4000
    email: str = "genie_ai@example.com"
    api_key: str = ""

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

    def search_pubmed(self, query: str) -> str:
        """
        Search PubMed for articles matching the query and return formatted results.

        Args:
            query: The search query string

        Returns:
            str: Formatted results from PubMed search
        """
        logger.info(f"Starting PubMed search with query: {query[:50]}...")
        logger.debug(f"Full query: {query}")
        logger.debug(f"Query length: {len(query)} (max: {self.MAX_QUERY_LENGTH})")

        try:
            # Truncate query if too long
            truncated_query = query[: self.MAX_QUERY_LENGTH]
            if len(query) > self.MAX_QUERY_LENGTH:
                logger.warning(f"Query truncated from {len(query)} to {self.MAX_QUERY_LENGTH} characters")
            
            logger.info(f"Retrieving top {self.top_k_results} results from PubMed")
            
            # Retrieve the top-k results for the query
            results = self.load(truncated_query)
            
            if not results:
                logger.info("No PubMed results found for query")
                return "No PubMed results found for your query."
            
            logger.info(f"Found {len(results)} PubMed articles")
                
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="search_pubmed",
                parameters={"query": query[:100] + "..." if len(query) > 100 else query},
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
            
            if len(formatted_results) > self.doc_content_chars_max:
                logger.warning(f"Results truncated from {len(formatted_results)} to {self.doc_content_chars_max} characters")
                formatted_results = formatted_results[:self.doc_content_chars_max]
            
            logger.info("PubMed search completed successfully")
            return formatted_results if docs else "No relevant PubMed results were found"
            
        except Exception as ex:
            logger.exception(f"PubMed search error: {ex}")
            return f"PubMed search encountered an error: {ex}"

    def find_geo_from_doi(self, doi: str) -> str:
        """
        Find GEO accession numbers mentioned in publications with the given DOI.
        
        Args:
            doi: The DOI of the publication
            
        Returns:
            str: GEO accession numbers found, if any
        """
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
            results = self.load(query)
            
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="find_geo_from_doi",
                parameters={"doi": doi},
                description="Searched for GEO accession numbers related to a publication DOI"
            )
            
            if not results:
                return f"No publications found with DOI {doi}"
            
            # Extract PMIDs for further detailed retrieval
            pmids = [result.get('uid') for result in results if result.get('uid')]
            
            # Extract GEO accession numbers from full articles
            import re
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
            
            return response
                
        except Exception as ex:
            logger.error(f"Error finding GEO from DOI: {ex}")
            return f"Error searching for GEO accessions related to DOI {doi}: {ex}"

    def find_marker_genes(self, cell_type: str, disease: Optional[str] = None) -> str:
        """
        Find marker genes for specific cell types, optionally in the context of a disease.
        
        Args:
            cell_type: The cell type to find markers for (e.g., "T cells", "macrophages")
            disease: Optional disease context (e.g., "rheumatoid arthritis", "cancer")
            
        Returns:
            str: Information about marker genes for the specified cell type
        """
        logger.info(f"Searching for marker genes for {cell_type}" + 
                   (f" in {disease}" if disease else ""))
        
        # Construct query based on input parameters
        query = f"{cell_type} marker genes"
        if disease:
            query += f" {disease}"
            
        query += " single-cell RNA-seq"
        
        try:
            results = self.load(query)
            
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="find_marker_genes_literature",
                parameters={"cell_type": cell_type, "disease": disease if disease else "None"},
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
            
            return f"Found literature about marker genes for {cell_type}" + \
                   (f" in {disease}" if disease else "") + ":\n\n" + \
                   "\n\n---\n\n".join(papers)[: self.doc_content_chars_max]
                
        except Exception as ex:
            logger.error(f"Error finding marker genes: {ex}")
            return f"Error searching for marker genes: {ex}"

    def find_protocol_information(self, technique: str) -> str:
        """
        Find protocol information for specific bioinformatics techniques.
        
        Args:
            technique: The technique to find protocol information for
            
        Returns:
            str: Protocol information for the specified technique
        """
        logger.info(f"Searching for protocol information for {technique}")
        
        query = f"{technique} protocol bioinformatics methodology"
        
        try:
            results = self.load(query)
            
            # Log the search in data manager
            self.data_manager.log_tool_usage(
                tool_name="find_protocol_information",
                parameters={"technique": technique},
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
            
            return f"Found protocol information for {technique}:\n\n" + \
                   "\n\n---\n\n".join(protocols)[: self.doc_content_chars_max]
                
        except Exception as ex:
            logger.error(f"Error finding protocol information: {ex}")
            return f"Error searching for protocol information: {ex}"

    def lazy_load(self, query: str) -> Iterator[dict]:
        """
        Search PubMed for documents matching the query.
        Return an iterator of dictionaries containing the document metadata.
        
        Args:
            query: The search query string
            
        Returns:
            Iterator[dict]: Iterator of document metadata dictionaries
        """
        url = (
            self.base_url_esearch
            + "db=pubmed&term="
            + urllib.parse.quote(query)
            + f"&retmode=json&retmax={self.top_k_results}&usehistory=y"
        )
        if self.api_key != "":
            url += f"&api_key={self.api_key}"
            
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

    def load(self, query: str) -> List[dict]:
        """
        Search PubMed for documents matching the query.
        Return a list of dictionaries containing the document metadata.
        
        Args:
            query: The search query string
            
        Returns:
            List[dict]: List of document metadata dictionaries
        """
        return list(self.lazy_load(query))

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
        if self.api_key != "":
            url += f"&api_key={self.api_key}"

        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    logger.warning(
                        f"Too Many Requests from PubMed, "
                        f"waiting for {self.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
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
