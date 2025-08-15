"""
Simplified Method Agent for literature research.

Following LangGraph 0.2.x multi-agent template pattern.
"""

import re
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

from ..config.settings import get_settings
from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


def method_expert(
    data_manager: DataManager, 
    callback_handler=None, 
    agent_name: str = "method_expert_agent",
    handoff_tools: List = None
):
    """Create method agent following template pattern."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('method_agent')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Define tools
    @tool
    def search_pubmed(
        query: str,
        top_k_results: int = 3,
        doc_content_chars_max: int = 4000,
        max_query_length: int = 300
    ) -> str:
        """
        Search PubMed for relevant scientific publications.
        
        Args:
            query: Search query string
            top_k_results: Number of results to retrieve (default: 3, range: 1-20)
            doc_content_chars_max: Maximum content length (default: 4000, range: 1000-10000)
            max_query_length: Maximum query length (default: 300, range: 100-500)
        """
        try:
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            results = pubmed_service.search_pubmed(
                query=query,
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                max_query_length=max_query_length
            )
            logger.info(f"PubMed search completed for: {query[:50]}... (k={top_k_results})")
            return results
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return f"Error searching PubMed: {str(e)}"

    @tool
    def find_method_parameters_from_doi(
        doi: str,
        top_k_results: int = 5,
        doc_content_chars_max: int = 6000
    ) -> str:
        """
        Extract method parameters and protocols from a publication DOI.
        
        Args:
            doi: Publication DOI (e.g., "10.1038/s41586-021-03659-0")
            top_k_results: Number of results to retrieve (default: 5, range: 1-10)
            doc_content_chars_max: Maximum content length (default: 6000, range: 2000-10000)
        """
        try:
            if not doi.startswith("10."):
                return "Invalid DOI format. DOI should start with '10.'"
            
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            
            # Search for the publication with enhanced parameters
            results = pubmed_service.search_pubmed(
                query=f"DOI:{doi}",
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )
            
            # Store DOI-based parameters in metadata
            if "parameters" in results.lower() or "methods" in results.lower():
                data_manager.current_metadata[f'methods_from_doi_{doi}'] = results
                
            logger.info(f"Method search completed for DOI: {doi} (k={top_k_results})")
            return results
        except Exception as e:
            logger.error(f"Error finding methods from DOI: {e}")
            return f"Error finding method parameters: {str(e)}"

    @tool
    def find_marker_genes(
        query: str,
        top_k_results: int = 5,
        doc_content_chars_max: int = 5000
    ) -> str:
        """
        Find marker genes for cell types from literature.
        
        Args:
            query: Query with cell_type parameter (e.g., 'cell_type=T_cell disease=cancer')
            top_k_results: Number of results to retrieve (default: 5, range: 1-15)
            doc_content_chars_max: Maximum content length (default: 5000, range: 2000-8000)
        """
        try:
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            
            # Parse parameters from query
            cell_type_match = re.search(r'cell[_\s]type[=\s]+([^,\s]+)', query)
            disease_match = re.search(r'disease[=\s]+([^,\s]+)', query)
            
            if not cell_type_match:
                return "Please specify cell_type parameter (e.g., 'cell_type=T_cell')"
            
            cell_type = cell_type_match.group(1).strip()
            disease = disease_match.group(1).strip() if disease_match else None
            
            results = pubmed_service.find_marker_genes(
                cell_type=cell_type,
                disease=disease,
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )
            logger.info(f"Marker gene search completed for {cell_type} (k={top_k_results})")
            return results
            
        except Exception as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"

    @tool
    def find_protocol_information(
        technique: str,
        top_k_results: int = 4,
        doc_content_chars_max: int = 5000
    ) -> str:
        """
        Find protocol information for bioinformatics techniques.
        
        Args:
            technique: The bioinformatics technique (e.g., "single-cell RNA-seq clustering")
            top_k_results: Number of results to retrieve (default: 4, range: 1-10)
            doc_content_chars_max: Maximum content length (default: 5000, range: 2000-8000)
        """
        try:
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            results = pubmed_service.find_protocol_information(
                technique=technique,
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )
            logger.info(f"Protocol search completed for: {technique} (k={top_k_results})")
            return results
        except Exception as e:
            logger.error(f"Error finding protocol info: {e}")
            return f"Error finding protocol information: {str(e)}"

    base_tools = [
        search_pubmed,
        find_method_parameters_from_doi,
        find_marker_genes,
        find_protocol_information
    ]
    
    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])
    
    system_prompt = """
You are a computational methods specialist. Your role is to find best practice parameters for bioinformatics analyses from scientific literature.

<Task>
Given a specific computational method question, you will:
1. Search relevant publications using PubMed
2. Extract methodology parameters (resolutions, thresholds, etc.)
3. Find GitHub repositories or supplementary materials
4. Provide clear recommendations with supporting evidence
5. Store important method parameters in the data manager for reproducibility
</Task>

<Available Tools>
- search_pubmed: Search for relevant publications with configurable parameters (top_k_results, doc_content_chars_max, max_query_length)
- find_method_parameters_from_doi: Extract method parameters from a DOI with enhanced retrieval settings
- find_marker_genes: Find marker genes for cell types with configurable result limits and content size
- find_protocol_information: Find protocol details for bioinformatics techniques with flexible parameters

Note: All tools now support dynamic parameter configuration for flexible literature retrieval.
Use higher top_k_results values (5-10) for comprehensive searches, lower values (1-3) for focused queries.
Adjust doc_content_chars_max based on detail needs: 2000-4000 for summaries, 6000-8000 for detailed analysis.
</Available Tools>

<Data Management>
- You DO NOT handle data files directly
- If GEO datasets are mentioned in papers, note them for the data expert
- Your focus is on finding and documenting computational methods
- Store important parameters in the data manager metadata
</Data Management>

<Guidelines>
When providing parameter recommendations:
- Focus on finding concrete parameter values used in similar analyses
- Include specific citations (DOI or PMID) for your recommendations
- Mention any GitHub repositories or supplementary materials found
- Explain why certain parameters are recommended based on the literature
- Provide ranges when multiple studies use different values
- Note if parameters vary by dataset type or analysis goal

<Response Format>
Provide your findings in clear, structured text that includes:
- Recommended parameter values with justification
- Citations to supporting literature
- Any relevant GitHub repositories or protocols
- Context about when different parameters should be used
- If datasets are mentioned, note their identifiers for the data expert

Today's date is {date}. Max iterations before timeout: {max_iterations}
""".format(
        date=date.today(),
        max_iterations=3
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
