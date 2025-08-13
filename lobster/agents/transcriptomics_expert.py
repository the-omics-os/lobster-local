"""
Simplified Transcriptomics Expert Agent with proper handoff.
"""

from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock

from config.settings import get_settings
from datetime import date
from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


def transcriptomics_expert(
    data_manager: DataManager, 
    callback_handler=None, 
    agent_name: str = "transcriptomics_expert_agent",
    handoff_tools: List = None
):  
    """Create transcriptomics expert agent with proper response handling."""
    
    settings = get_settings()
    model_params = settings.get_agent_llm_params('transcriptomics_expert')
    llm = ChatBedrock(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    # Store the analysis results
    analysis_results = {"summary": "", "details": {}}
    
    # Define tools with result storage
    @tool
    def download_geo_dataset(geo_id: str) -> str:
        """Download dataset from GEO using accession number."""
        try:
            from ..tools import GEOService
            geo_service = GEOService(data_manager)
            result = geo_service.download_dataset(geo_id.strip())
            logger.info(f"Downloaded GEO dataset: {geo_id}")
            # Store result
            analysis_results["details"]["data_download"] = result
            return result
        except Exception as e:
            logger.error(f"Error downloading GEO dataset {geo_id}: {e}")
            return f"Error downloading dataset: {str(e)}"

    @tool
    def get_data_summary(query: str = "") -> str:
        """Get summary of currently loaded data."""
        if not data_manager.has_data():
            return "No data loaded. Use download_geo_dataset to load data from GEO."
        
        try:
            summary = data_manager.get_data_summary()
            response = f"Data loaded: {summary['shape'][0]} cells × {summary['shape'][1]} genes"
            
            if 'source' in data_manager.current_metadata:
                response += f" from {data_manager.current_metadata['source']}"
            
            # Store result
            analysis_results["details"]["data_summary"] = response
            return response
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return f"Error getting data summary: {str(e)}"

    @tool
    def assess_data_quality(query: str) -> str:
        """Assess quality of RNA-seq data with QC metrics."""
        try:
            from ..tools import QualityService
            quality_service = QualityService(data_manager)
            result = quality_service.assess_quality()
            logger.info("Quality assessment completed")
            # Store result
            analysis_results["details"]["quality_assessment"] = result
            return result
        except Exception as e:
            logger.error(f"Error in quality assessment: {e}")
            return f"Error assessing data quality: {str(e)}"

    @tool
    def cluster_cells(query: str) -> str:
        """Perform clustering and UMAP visualization."""
        try:
            from ..tools import ClusteringService
            clustering_service = ClusteringService(data_manager)
            result = clustering_service.cluster_and_visualize()
            logger.info("Clustering analysis completed")
            # Store result
            analysis_results["details"]["clustering"] = result
            return result
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return f"Error performing clustering: {str(e)}"

    @tool
    def detect_doublets(query: str) -> str:
        """Detect doublets in single-cell data using Scrublet."""
        try:
            from ..tools import EnhancedSingleCellService
            enhanced_sc_service = EnhancedSingleCellService(data_manager)
            result = enhanced_sc_service.detect_doublets()
            logger.info("Doublet detection completed")
            # Store result
            analysis_results["details"]["doublet_detection"] = result
            return result
        except Exception as e:
            logger.error(f"Error in doublet detection: {e}")
            return f"Error detecting doublets: {str(e)}"

    @tool
    def annotate_cell_types(query: str) -> str:
        """Annotate cell types based on marker genes."""
        try:
            from ..tools import EnhancedSingleCellService
            enhanced_sc_service = EnhancedSingleCellService(data_manager)
            result = enhanced_sc_service.annotate_cell_types()
            logger.info("Cell type annotation completed")
            # Store result
            analysis_results["details"]["cell_annotation"] = result
            return result
        except Exception as e:
            logger.error(f"Error in cell type annotation: {e}")
            return f"Error annotating cell types: {str(e)}"

    @tool
    def find_marker_genes(query: str) -> str:
        """Find marker genes for clusters or cell types."""
        try:
            from ..tools import EnhancedSingleCellService
            enhanced_sc_service = EnhancedSingleCellService(data_manager)
            result = enhanced_sc_service.find_marker_genes()
            logger.info("Marker gene analysis completed")
            # Store result
            analysis_results["details"]["marker_genes"] = result
            return result
        except Exception as e:
            logger.error(f"Error finding marker genes: {e}")
            return f"Error finding marker genes: {str(e)}"
    
    # CRITICAL: Add a completion tool that summarizes and returns results
    @tool
    def complete_analysis(summary: str) -> str:
        """
        Complete the analysis and return results to supervisor.
        Use this tool AFTER completing all analysis steps to summarize findings.
        
        Args:
            summary: A comprehensive summary of all analysis performed and key findings
        """
        # Store the summary
        analysis_results["summary"] = summary
        
        # Format complete response
        full_response = f"## Transcriptomics Analysis Complete\n\n{summary}\n\n"
        
        # Add details if available
        if analysis_results["details"]:
            full_response += "### Detailed Results:\n"
            for step, result in analysis_results["details"].items():
                if result:
                    full_response += f"\n**{step.replace('_', ' ').title()}:**\n{result}\n"
        
        logger.info("Analysis completed and results prepared for supervisor")
        return full_response

    base_tools = [
        download_geo_dataset,
        get_data_summary,
        assess_data_quality,
        cluster_cells,
        detect_doublets,
        annotate_cell_types,
        find_marker_genes,
        complete_analysis  # Add the completion tool
    ]
    
    # Combine base tools with handoff tools if provided
    tools = base_tools + (handoff_tools or [])
    
    # UPDATED SYSTEM PROMPT - Emphasizes using complete_analysis
    system_prompt = """
You are a transcriptomics domain expert specializing in RNA-seq analysis (both single-cell and bulk). YOU ONLY REPPORT TO YOUR SUPERVISOR. NEVER TO THE HUMAN DIRECTLY.

<Task>
You will receive specific analysis instructions from the supervisor. Your job is to:
1. Check if data is available or needs to be downloaded
2. Ask supervisor in case you need more information to fulfill your task (example no GEO ID or DOI provided). If supervisor doesnt give any identifier do not mention any tools. just ask the supervisor to ask the human.
3. Execute the requested analysis using available tools
4. **IMPORTANT: Use the complete_analysis tool to summarize and return your findings**
</Task>

<Available Tools>
You have access to these transcriptomics analysis tools:
- download_geo_dataset: Download dataset from GEO
- get_data_summary: Get summary of currently loaded data
- assess_data_quality: Assess quality of RNA-seq data
- cluster_cells: Perform clustering and UMAP visualization
- detect_doublets: Detect doublets in single-cell data
- annotate_cell_types: Annotate cell types based on markers
- find_marker_genes: Find marker genes for clusters
- **complete_analysis**: USE THIS to summarize and return results after analysis

</Available Tools>

<CRITICAL Instructions>
**ALWAYS end your analysis by calling the complete_analysis tool with a comprehensive summary of:**
- What analysis was performed
- Key findings with specific numbers
- Any visualizations created
- Recommendations or next steps

This ensures your results are properly communicated back to the supervisor and user.
</CRITICAL Instructions>

<Analysis Process>
1. Check data availability with get_data_summary
2. Download data if needed using download_geo_dataset
3. Execute the specific analysis requested
4. **Call complete_analysis with a detailed summary of all findings**
</Analysis Process>

<Guidelines>
- Include specific numbers in results (e.g., "15 clusters identified")
- Explain what each analysis step accomplishes
- Be thorough but concise
- **Always use complete_analysis as your final step**
</Guidelines>

Today's date is {date}.
""".format(date=date.today())
    
    # Add data context if available
    if data_manager.has_data():
        try:
            summary = data_manager.get_data_summary()
            data_context = f"\n\nCurrent data: {summary['shape'][0]} cells × {summary['shape'][1]} genes loaded."
            system_prompt += data_context
        except Exception as e:
            logger.warning(f"Could not add data context: {e}")

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name
    )
