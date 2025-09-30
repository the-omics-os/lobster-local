"""
Method Expert Agent for computational methodology extraction and parameter analysis.

This agent specializes in extracting and analyzing computational parameters from
scientific publications, focusing on best practice parameter identification and
methodological analysis using the modular DataManagerV2 and publication service.
"""

import re
from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from lobster.config.llm_factory import create_llm

from datetime import date

from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.publication_service import PublicationService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def method_expert(
    data_manager: DataManagerV2, 
    callback_handler=None, 
    agent_name: str = "method_expert_agent",
    handoff_tools: List = None
):
    """Create method expert agent focused on parameter extraction using DataManagerV2."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('method_expert_agent')
    llm = create_llm('method_expert_agent', model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Initialize publication service for method extraction
    publication_service = PublicationService(data_manager=data_manager)
    
    # Define tools

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
            
            from lobster.tools.pubmed_service import PubMedService
            
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            
            # Search for the publication with enhanced parameters
            results = pubmed_service.search_pubmed(
                query=f"DOI:{doi}",
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max
            )
            
            # Store DOI-based parameters in metadata using DataManagerV2's tool logging
            if "parameters" in results.lower() or "methods" in results.lower():
                data_manager.log_tool_usage(
                    tool_name="find_method_parameters_from_doi",
                    parameters={"doi": doi, "found_parameters": True},
                    description=f"Found method parameters for DOI: {doi}"
                )
                
            logger.info(f"Method search completed for DOI: {doi} (k={top_k_results})")
            return results
        except Exception as e:
            logger.error(f"Error finding methods from DOI: {e}")
            return f"Error finding method parameters: {str(e)}"

    @tool
    def extract_computational_methods(
        identifier: str,
        method_type: str = "all",
        include_parameters: bool = True
    ) -> str:
        """
        Extract computational methods and parameters from a publication.
        
        Args:
            identifier: Publication identifier (DOI or PMID)
            method_type: Type of methods to extract ("all", "preprocessing", "clustering", "normalization", etc.)
            include_parameters: Whether to extract specific parameter values (default: True)
        """
        try:
            results = publication_service.extract_computational_methods(
                doi_or_pmid=identifier,
                method_type=method_type,
                include_parameters=include_parameters
            )
            
            logger.info(f"Method extraction completed for: {identifier}")
            return results
            
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
            return f"Error extracting computational methods: {str(e)}"

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
            from lobster.tools.pubmed_service import PubMedService
            
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

    @tool
    def find_method_parameters_for_modality(
        modality_name: str,
        analysis_type: str,
        top_k_results: int = 5
    ) -> str:
        """
        Find method parameters specifically for a loaded modality.
        
        Args:
            modality_name: Name of the loaded modality to find parameters for
            analysis_type: Type of analysis (e.g., 'clustering', 'differential_expression', 'quality_control')
            top_k_results: Number of results to retrieve
            
        Returns:
            str: Method parameters specific to the modality type
        """
        try:
            # Get the modality to understand its characteristics
            try:
                adata = data_manager.get_modality(modality_name)
                metrics = data_manager.get_quality_metrics(modality_name)
            except ValueError:
                return f"Modality '{modality_name}' not found. Available modalities: {data_manager.list_modalities()}"
            
            # Determine modality type from adapter info
            adapter_info = data_manager.get_adapter_info()
            modality_type = "single_cell"  # Default
            
            for adapter_name, info in adapter_info.items():
                if modality_name in adapter_name or info['modality_name'] in modality_name.lower():
                    if 'single_cell' in info['schema']['modality']:
                        modality_type = "single-cell RNA-seq"
                    elif 'bulk' in info['schema']['modality']:
                        modality_type = "bulk RNA-seq"
                    elif 'proteomics' in info['schema']['modality']:
                        modality_type = "proteomics"
                    break
            
            # Build search query based on modality and analysis type
            search_query = f"{modality_type} {analysis_type} parameters methods"
            
            # Add data characteristics to search
            if adata.n_obs > 1000 and adata.n_vars > 10000:
                search_query += " large-scale"
            elif adata.n_obs < 100:
                search_query += " small-scale"
            
            # Search for relevant methods using DataManagerV2
            from lobster.tools.pubmed_service import PubMedService
            pubmed_service = PubMedService(parse=None, data_manager=data_manager)
            
            results = pubmed_service.search_pubmed(
                query=search_query,
                top_k_results=top_k_results,
                doc_content_chars_max=6000
            )
            
            # Add modality context to results
            context_info = f"""
## Modality Context for '{modality_name}'
- **Data type**: {modality_type}
- **Shape**: {adata.n_obs} obs × {adata.n_vars} vars
- **Analysis type**: {analysis_type}
- **Key metrics**: {len([k for k, v in metrics.items() if isinstance(v, (int, float))])} calculated

## Literature-Based Method Parameters:
"""
            
            return context_info + results
            
        except Exception as e:
            logger.error(f"Error finding modality parameters: {e}")
            return f"Error finding method parameters: {str(e)}"

    @tool
    def analyze_methodology(
        identifiers: str,
        analysis_focus: str = "parameters"
    ) -> str:
        """
        Analyze methodologies across multiple publications for parameter consensus.
        
        Args:
            identifiers: Comma-separated list of DOIs or PMIDs
            analysis_focus: Focus of analysis ("parameters", "tools", "workflows", "all")
        """
        try:
            identifier_list = [id.strip() for id in identifiers.split(',')]
            results = []
            
            for identifier in identifier_list:
                if not identifier:
                    continue
                    
                methods = publication_service.extract_computational_methods(
                    doi_or_pmid=identifier,
                    method_type="all",
                    include_parameters=True
                )
                results.append(f"### Analysis of {identifier}\n{methods}\n")
            
            # Combine results with analysis header
            combined_results = f"## Comparative Methodology Analysis\n"
            combined_results += f"**Publications analyzed**: {len(identifier_list)}\n"
            combined_results += f"**Analysis focus**: {analysis_focus}\n\n"
            combined_results += "\n---\n\n".join(results)
            
            logger.info(f"Methodology analysis completed for {len(identifier_list)} publications")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error analyzing methodology: {e}")
            return f"Error analyzing methodology: {str(e)}"

    base_tools = [
        find_method_parameters_from_doi,
        extract_computational_methods,
        find_protocol_information,
        find_method_parameters_for_modality,
        analyze_methodology
    ]
    
    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])
    
    system_prompt = """
You are a computational bioinformatics methods specialist focused on extracting and analyzing computational parameters and methodologies from scientific publications.

<Role>
Your specialized expertise lies in deep analysis of computational methodologies, parameter extraction, and best practice identification. You work with publications that have already been discovered by the research agent to extract actionable computational insights.
</Role>

<Task>
Given publications or research questions about computational methods, you will:
1. **Extract specific parameters** (thresholds, resolutions, filter values, etc.) from publications
2. **Identify software tools** and versions used in studies  
3. **Document method workflows** with clear parameter recommendations
4. **Cross-reference multiple studies** to identify consensus parameters
5. **Analyze methodological approaches** across different studies
6. **Provide modality-specific parameter recommendations** using DataManagerV2
7. **Generate parameter validation strategies** using available datasets
</Task>

<Available Method Extraction Tools>
- `find_method_parameters_from_doi`: Extract methods from specific publications  
  * Automatically extracts preprocessing parameters, QC thresholds, normalization methods
  * Identifies clustering resolutions, integration methods, and tool versions
  * Finds GitHub links and supplementary materials

- `extract_computational_methods`: Enhanced method extraction from publications
  * method_type: "all", "preprocessing", "clustering", "normalization", etc.
  * include_parameters: Extract specific parameter values
  * Comprehensive parameter identification and documentation

- `find_protocol_information`: Protocol and workflow extraction
  * Step-by-step method documentation  
  * Parameter ranges and recommendations
  * Technique-specific best practices

- `find_method_parameters_for_modality`: Context-aware parameter search (DataManagerV2)
  * Searches for parameters specific to loaded modality characteristics
  * Considers data dimensions and type for relevant parameter extraction
  * Provides modality-specific recommendations

- `analyze_methodology`: Comparative methodology analysis
  * Analyze methods across multiple publications
  * Identify parameter consensus and variations
  * Generate best practice recommendations

<DataManagerV2 Integration>
When working with DataManagerV2, you can provide modality-specific parameter recommendations:


# Find parameters for a specific loaded modality
find_method_parameters_for_modality("geo_gse12345", "quality_control")
find_method_parameters_for_modality("proteomics_sample1", "normalization")


This provides context-aware parameter recommendations based on the actual data characteristics.

<Query Optimization Strategies>

## Example 1: Single-cell RNA-seq Preprocessing Parameters


# Initial broad search to understand landscape
search_pubmed(
    query="single-cell RNA-seq quality control filtering parameters mitochondrial", 
    top_k_results=8,  # Get more results for comprehensive view
    doc_content_chars_max=6000  # Extended content for parameter extraction
)

# Focused search for specific cell types
search_pubmed(
    query='"T cells" AND "single cell" AND (filtering OR "quality control") AND "UMI count"',
    top_k_results=5,
    doc_content_chars_max=4000
)


## Example 2: Proteomics Method Parameters


# Search for MS proteomics normalization methods
search_pubmed(
    query="mass spectrometry proteomics normalization missing values imputation",
    top_k_results=6,
    doc_content_chars_max=5000
)

# Find peptide-to-protein mapping methods
search_pubmed(
    query="peptide protein identification FDR threshold proteomics",
    top_k_results=4,
    doc_content_chars_max=4000
)


<Parameter Extraction Guidelines>

When extracting parameters, focus on modality-specific requirements:

**Transcriptomics (Single-cell):**
1. **Quality Control**: min_genes_per_cell (200-500), max_pct_mt (5-20%)
2. **Normalization**: target_sum (10,000), log_base (natural log)
3. **Feature Selection**: n_top_genes (2000-5000)
4. **Clustering**: resolution (0.4-1.2), n_pcs (10-50)

**Transcriptomics (Bulk):**
1. **Quality Control**: min_reads_per_sample (1M-10M)
2. **Normalization**: TMM, DESeq2, or RPKM/TPM
3. **Batch Correction**: ComBat, limma, or Harmony

**Proteomics:**
1. **Quality Control**: max_missing_per_protein (70-80%), min_peptides_per_protein (1-2)
2. **Normalization**: median, quantile, or VSN
3. **Missing Value Handling**: imputation methods (KNN, MinProb, etc.)

<Response Format>
Structure your findings as:

## Recommended Parameters for [Analysis Type] - [Modality Type]

### Quality Control Thresholds
- **Parameter**: value (Citation: PMID/DOI)
- **Evidence**: X studies analyzed, consensus range: [range]

### Method-Specific Parameters  
- **Tool/Package**: name version
- **Key parameters**: values with rationale
- **GitHub**: repository links

### Modality-Specific Considerations
- **Data characteristics**: What affects parameter choice
- **Common variations**: Tissue/condition-specific adjustments

### Associated Datasets for Validation
- **GEO**: GSE numbers with descriptions
- **Note for data_expert**: "Please retrieve GSE[number] for parameter validation"

### Literature Support
1. PMID/DOI - First Author Year - Key finding
2. PMID/DOI - First Author Year - Parameter validation

</Response>

<Important Notes>
- **Search with increasing specificity**: broad → focused → specific papers
- **Use higher top_k_results** (8-15) for parameter surveys, lower (1-3) for specific papers
- **Cross-reference multiple studies** when recommending parameter ranges
- **Consider modality characteristics** when using DataManagerV2
- **Flag controversial or dataset-dependent parameters**
- **Always provide literature citations** for recommended parameters

Today's date is {date}. Maximum iterations: 5
""".format(
    date=date.today()
)

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
