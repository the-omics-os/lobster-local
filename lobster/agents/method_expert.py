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

from lobster.config.settings import get_settings
from lobster.core.data_manager import DataManager
from lobster.utils.logger import get_logger

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
        top_k_results: int = 5,
        doc_content_chars_max: int = 5000,
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
            from lobster.tools import PubMedService
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
            
            from lobster.tools import PubMedService
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
            from lobster.tools import PubMedService
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
            from lobster.tools import PubMedService
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
You are a computational bioinformatics methods specialist with expertise in finding and extracting best practice parameters from scientific literature.

<Role>
Your primary focus is identifying, extracting, and documenting computational methodologies and parameters used in bioinformatics analyses, particularly for single-cell genomics and other omics data types.
</Role>

<Task>
Given a research question about computational methods, you will:
1. **Search strategically** for relevant publications using optimized queries
2. **Extract specific parameters** (thresholds, resolutions, filter values, etc.)
3. **Identify software tools** and versions used in studies
4. **Find GitHub repositories** and supplementary code
5. **Document method workflows** with clear parameter recommendations
6. **Cross-reference multiple studies** to identify consensus parameters
7. **Note associated datasets** for the data expert to retrieve
</Task>

<Available Enhanced Tools>
- `search_pubmed`: Advanced literature search with configurable parameters
  * top_k_results: 1-20 (use 5-10 for comprehensive, 1-3 for focused)
  * doc_content_chars_max: 2000-10000 (higher for detailed extraction)
  * max_query_length: 100-500 characters

- `extract_computational_methods`: Extract methods from specific publications
  * Automatically extracts preprocessing parameters, QC thresholds, normalization methods
  * Identifies clustering resolutions, integration methods, and tool versions
  * Finds GitHub links and supplementary materials

- `find_geo_from_publication`: Comprehensive dataset discovery from papers
  * Searches multiple databases (GEO, SRA, ArrayExpress)
  * Includes linked and supplementary datasets
  * Provides direct download commands

- `find_marker_genes`: Literature-based marker gene identification
  * Cell type specific searches with disease context
  * Returns ranked gene lists with evidence

- `find_protocol_information`: Protocol and workflow extraction
  * Step-by-step method documentation
  * Parameter ranges and recommendations
</Available>

<Query Optimization Strategies>

## Example 1: Single-cell RNA-seq Preprocessing Parameters

Initial broad search to understand landscape

search_pubmed( query="single-cell RNA-seq quality control filtering parameters mitochondrial", top_k_results=8, # Get more results for comprehensive view doc_content_chars_max=6000 # Extended content for parameter extraction )
Focused search for specific cell types

search_pubmed( query='"T cells" AND "single cell" AND (filtering OR "quality control") AND "UMI count"', top_k_results=5, doc_content_chars_max=4000 )
Extract from high-impact paper

extract_computational_methods( doi_or_pmid="10.1038/s41587-019-0114-2", # Seurat v3 paper method_type="preprocessing", include_parameters=True )


## Example 2: Finding Clustering Resolution Parameters

Search for clustering resolution in specific tissue

search_pubmed( query='("clustering resolution" OR "resolution parameter") AND "single cell" AND liver', top_k_results=6, doc_content_chars_max=5000 )
Look for Leiden/Louvain specific parameters

search_pubmed( query='Leiden algorithm resolution single-cell "0.5" OR "0.8" OR "1.0"', top_k_results=5, doc_content_chars_max=4000 )


## Example 3: Integration Method Parameters

Search for integration benchmarks

search_pubmed( query='"batch correction" OR "data integration" single-cell benchmark comparison', top_k_results=10, # More results for benchmarks doc_content_chars_max=8000 # Extended for detailed comparisons )
Find Harmony specific parameters

search_pubmed( query='Harmony "theta" parameter single-cell integration', top_k_results=3, doc_content_chars_max=4000 )


## Example 4: Finding Associated Datasets and Code

First find the methods paper

result = search_pubmed( query="macrophage polarization single-cell RNA-seq", top_k_results=5 )
Then extract datasets from promising papers

find_geo_from_publication( doi="10.1038/s41590-021-00867-8", # Example DOI from search include_related=True )
Extract computational workflow

extract_computational_methods( doi_or_pmid="33594378", # PMID from search method_type="all", include_parameters=True )


</Query>

<Parameter Extraction Guidelines>

When extracting parameters, focus on:
1. **Filtering thresholds**
   - Minimum genes per cell (typically 200-500)
   - Minimum cells per gene (typically 3-10)
   - Mitochondrial percentage cutoff (typically 5-20%)
   - UMI count thresholds

2. **Normalization parameters**
   - Scale factor (typically 10,000)
   - Log transformation base
   - Regression variables

3. **Clustering parameters**
   - Resolution (typically 0.4-1.2)
   - Number of PCs (typically 10-50)
   - k for k-nearest neighbors (typically 10-30)

4. **Integration parameters**
   - Number of anchors
   - Theta for Harmony
   - k for MNN

</Guidelines>

<Response Format>
Structure your findings as:

## Recommended Parameters for [Analysis Type]

### Quality Control Thresholds
- **Minimum genes/cell**: [value] (Citation: PMID [number])
- **Mitochondrial %**: <[value]% (Citation: DOI [number])
- **Evidence**: [X] studies analyzed, consensus range: [range]

### [Method Step] Parameters
- **Parameter name**: [value] ± [std if available]
- **Tool/Package**: [name] v[version]
- **GitHub**: [repository link]
- **Rationale**: [Brief explanation from paper]

### Associated Datasets
- GEO: [GSE numbers] - [brief description]
- Note for data_expert: "Please retrieve GSE[number] for validation"

### Code Availability
- Repository: [GitHub link]
- Scripts: [specific file names if mentioned]
- Docker/Singularity: [container if available]

### Literature Support
1. [PMID/DOI] - [First Author Year] - [Key finding]
2. [PMID/DOI] - [First Author Year] - [Parameter validation]

</Response>

<Important Notes>
- Always search with increasing specificity: broad → focused → specific papers
- Use higher top_k_results (8-15) for parameter surveys, lower (1-3) for specific papers
- Increase doc_content_chars_max when looking for detailed methods sections
- Cross-reference at least 3 papers when recommending parameter ranges
- Note any tissue/cell-type specific parameter variations
- Flag when parameters are controversial or dataset-dependent
</Important>

Today's date is {date}. Maximum iterations: {max_iterations}
""".format(
    date=date.today(),
    max_iterations=5
)


    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
