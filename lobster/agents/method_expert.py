"""
Simplified Method Agent for literature research.

Following LangGraph 0.2.x multi-agent template pattern.
"""

import re
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock

from config.settings import get_settings
from datetime import date
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
    llm = ChatBedrock(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Define tools
    @tool
    def search_pubmed(query: str) -> str:
        """Search PubMed for relevant scientific publications."""
        try:
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            results = pubmed_service.search_pubmed(query)
            logger.info(f"PubMed search completed for: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return f"Error searching PubMed: {str(e)}"

    @tool
    def find_geo_from_doi(doi: str) -> str:
        """Find GEO accession numbers from a DOI."""
        try:
            if not doi.startswith("10."):
                return "Invalid DOI format. DOI should start with '10.'"
            
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            results = pubmed_service.find_geo_from_doi(doi)
            logger.info(f"GEO search completed for DOI: {doi}")
            return results
        except Exception as e:
            logger.error(f"Error finding GEO from DOI: {e}")
            return f"Error finding GEO datasets: {str(e)}"

    @tool
    def find_marker_genes(query: str) -> str:
        """Find marker genes for cell types from literature."""
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
                disease=disease
            )
            logger.info(f"Marker gene search completed for {cell_type}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"

    @tool
    def find_protocol_information(technique: str) -> str:
        """Find protocol information for bioinformatics techniques."""
        try:
            from ..tools import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            results = pubmed_service.find_protocol_information(technique)
            logger.info(f"Protocol search completed for: {technique}")
            return results
        except Exception as e:
            logger.error(f"Error finding protocol info: {e}")
            return f"Error finding protocol information: {str(e)}"

    base_tools = [
        search_pubmed,
        find_geo_from_doi,
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
</Task>

<Available Tools>
- search_pubmed: Search for relevant publications
- find_geo_from_doi: Find GEO datasets from DOI
- find_marker_genes: Find marker genes for cell types from literature
- find_protocol_information: Find protocol details for bioinformatics techniques
</Available Tools>

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
