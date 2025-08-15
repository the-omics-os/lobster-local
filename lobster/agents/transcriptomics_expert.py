"""
Transcriptomics Expert Agent (MVP - PDAC Reproduction Scope)
Now includes pseudocode placeholders for reproducing key preprocessing steps
from the PDAC paper's pipeline (GSE194247 & GSE235449).
"""

from typing import List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from datetime import date

from .state import TranscriptomicsExpertState
from ..config.settings import get_settings
from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

def transcriptomics_expert(
    data_manager: DataManager, 
    callback_handler=None, 
    agent_name: str = "transcriptomics_expert_agent",
    handoff_tools: List = None
):  
    settings = get_settings()
    model_params = settings.get_agent_llm_params('transcriptomics_expert')
    llm = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])
    
    analysis_results = {"summary": "", "details": {}}
    
    # -------------------------
    # BASIC TOOLS
    # -------------------------
    @tool
    def check_data_status() -> str:
        """Check if data is loaded and ready for analysis."""
        if not data_manager.has_data():
            return "No data loaded. Please ask the data expert to load a dataset first."
        try:
            summary = data_manager.get_data_summary()
            dataset_id = data_manager.current_metadata.get('dataset_id', 'Unknown')
            response = f"Data ready for analysis: {summary['shape'][0]} cells × {summary['shape'][1]} genes"
            response += f"\nDataset ID: {dataset_id}"
            analysis_results["details"]["data_status"] = response
            return response
        except Exception as e:
            logger.error(f"Error checking data status: {e}")
            return f"Error checking data status: {str(e)}"

    @tool
    def assess_data_quality(query: str) -> str:
        """Run basic QC metrics (cell/gene counts, mitochondrial %, etc.)."""
        try:
            from ..tools import QualityService
            result = QualityService(data_manager).assess_quality()
            analysis_results["details"]["quality_assessment"] = result
            return result
        except Exception as e:
            return f"Error in QC: {str(e)}"

    # -------------------------
    # NEW - PREPROCESSING TOOLS FROM PDAC PAPER
    # -------------------------
    @tool
    def ambient_RNA_correction(
        contamination_fraction: float = 0.1,
        empty_droplet_threshold: int = 100,
        method: str = "simple_decontamination",
        save_corrected: bool = True
    ) -> str:
        """
        Correct ambient RNA contamination using advanced decontamination methods.
        
        Args:
            contamination_fraction: Expected fraction of ambient RNA (0.05-0.2 typical)
            empty_droplet_threshold: Minimum UMI count to consider droplet as cell-containing
            method: Method to use ('simple_decontamination', 'quantile_based')
            save_corrected: Whether to save corrected data to workspace
        """
        try:
            from ..tools.preprocessing_service import PreprocessingService
            preprocessing_service = PreprocessingService(data_manager)
            result = preprocessing_service.correct_ambient_rna(
                contamination_fraction=contamination_fraction,
                empty_droplet_threshold=empty_droplet_threshold,
                method=method,
                save_corrected=save_corrected
            )
            analysis_results["details"]["ambient_correction"] = result
            return result
        except Exception as e:
            logger.error(f"Error in ambient RNA correction: {e}")
            return f"Error in ambient RNA correction: {str(e)}"

    @tool
    def filter_and_normalize_cells(
        min_genes_per_cell: int = 200,
        max_genes_per_cell: int = 5000,
        min_cells_per_gene: int = 3,
        max_mito_percent: float = 20.0,
        max_ribo_percent: float = 50.0,
        normalization_method: str = "log1p",
        target_sum: int = 10000,
        save_filtered: bool = True
    ) -> str:
        """
        Filter cells by QC thresholds and normalize gene expression.
        
        Args:
            min_genes_per_cell: Minimum number of genes expressed per cell
            max_genes_per_cell: Maximum number of genes expressed per cell
            min_cells_per_gene: Minimum number of cells expressing each gene
            max_mito_percent: Maximum mitochondrial gene percentage
            max_ribo_percent: Maximum ribosomal gene percentage  
            normalization_method: Normalization method ('log1p', 'sctransform_like')
            target_sum: Target sum for normalization (e.g., 10,000)
            save_filtered: Whether to save filtered data to workspace
        """
        try:
            from ..tools.preprocessing_service import PreprocessingService
            preprocessing_service = PreprocessingService(data_manager)
            result = preprocessing_service.filter_and_normalize_cells(
                min_genes_per_cell=min_genes_per_cell,
                max_genes_per_cell=max_genes_per_cell,
                min_cells_per_gene=min_cells_per_gene,
                max_mito_percent=max_mito_percent,
                max_ribo_percent=max_ribo_percent,
                normalization_method=normalization_method,
                target_sum=target_sum,
                save_filtered=save_filtered
            )
            analysis_results["details"]["filter_normalize"] = result
            return result
        except Exception as e:
            logger.error(f"Error in filtering/normalization: {e}")
            return f"Error in filtering/normalization: {str(e)}"

    @tool
    def integrate_and_batch_correct(
        batch_key: str = "sample",
        integration_method: str = "harmony",
        n_pca_components: int = 50,
        n_integration_features: int = 2000,
        theta: float = 2.0,
        lambda_param: float = 1.0,
        save_integrated: bool = True
    ) -> str:
        """
        Integrate datasets and correct for batch effects using advanced methods.
        
        Args:
            batch_key: Column name in obs containing batch information
            integration_method: Method to use ('harmony', 'scanorama_like', 'simple_scaling')
            n_pca_components: Number of PCA components for integration
            n_integration_features: Number of highly variable genes for integration
            theta: Harmony diversity clustering penalty parameter
            lambda_param: Harmony ridge regression penalty parameter
            save_integrated: Whether to save integrated data to workspace
        """
        try:
            from ..tools.preprocessing_service import PreprocessingService
            preprocessing_service = PreprocessingService(data_manager)
            result = preprocessing_service.integrate_and_batch_correct(
                batch_key=batch_key,
                integration_method=integration_method,
                n_pca_components=n_pca_components,
                n_integration_features=n_integration_features,
                theta=theta,
                lambda_param=lambda_param,
                save_integrated=save_integrated
            )
            analysis_results["details"]["integration_batch"] = result
            return result
        except Exception as e:
            logger.error(f"Error in integration: {e}")
            return f"Error in integration: {str(e)}"

    # -------------------------
    # EXISTING TOOLS
    # -------------------------
    @tool
    def cluster_cells(query: str) -> str:
        """Perform clustering and UMAP visualization."""
        try:
            from ..tools import ClusteringService
            result = ClusteringService(data_manager).cluster_and_visualize()
            analysis_results["details"]["clustering"] = result
            return result
        except Exception as e:
            return f"Error in clustering: {str(e)}"

    @tool
    def detect_doublets(query: str) -> str:
        """Detect doublets in single-cell data."""
        try:
            from ..tools import EnhancedSingleCellService
            result = EnhancedSingleCellService(data_manager).detect_doublets()
            analysis_results["details"]["doublet_detection"] = result
            return result
        except Exception as e:
            return f"Error detecting doublets: {str(e)}"

    @tool
    def annotate_cell_types(query: str) -> str:
        """Annotate cell types based on marker genes."""
        try:
            from ..tools import EnhancedSingleCellService
            result = EnhancedSingleCellService(data_manager).annotate_cell_types()
            analysis_results["details"]["cell_annotation"] = result
            return result
        except Exception as e:
            return f"Error annotating cell types: {str(e)}"

    @tool
    def find_marker_genes(query: str) -> str:
        """Find marker genes for clusters."""
        try:
            from ..tools import EnhancedSingleCellService
            result = EnhancedSingleCellService(data_manager).find_marker_genes()
            analysis_results["details"]["marker_genes"] = result
            return result
        except Exception as e:
            return f"Error finding marker genes: {str(e)}"

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        check_data_status,
        assess_data_quality,
        ambient_RNA_correction,
        filter_and_normalize_cells,
        detect_doublets,
        integrate_and_batch_correct,
        cluster_cells,
        annotate_cell_types,
        find_marker_genes
    ]
    tools = base_tools + (handoff_tools or [])

    # -------------------------
    # SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""
You are an expert bioinformatician specializing in single-cell RNA sequencing (scRNA-seq) analysis. 
You reason like a meticulous scientist: starting from raw count matrices and sample metadata, you plan and execute an analysis pipeline 
that ensures data quality, removes artifacts, integrates multiple datasets, and performs biologically meaningful interpretation.

You always explain your reasoning before taking each step, just as you would when working in a collaborative lab meeting. 
You adapt to different datasets, but follow good practice in single-cell analysis.

You have access to the following tools and may call them as needed in a logical order:

1. check_data_status  
   - Verify that data is loaded and accessible; inspect dimensions and basic stats.
   
2. assess_data_quality  
   - Evaluate QC metrics: gene counts, cell counts, mitochondrial/ribosomal content, sequencing depth.
   
3. ambient_RNA_correction  
   - Estimate and remove ambient RNA contamination (e.g., SoupX-style methods).  
   - This is **not** batch correction — it addresses artifactual reads from the background solution.
   
4. filter_and_normalize_cells  
   - Apply QC thresholds to filter low-quality cells (too few/too many genes, high mito%, etc.).  
   - Normalize gene expression values (log-transformation, SCTransform, or other method).
   
5. detect_doublets  
   - Identify and flag/dismiss potential doublets or multiplets.

6. integrate_and_batch_correct  
   - Integrate multiple datasets, correcting for batch effects using methods like Harmony, Seurat CCA, or MNN.  
   - Choose features for integration, run dimensionality reduction, then harmonize embeddings.
   
7. cluster_cells  
   - Perform graph-based clustering (Leiden/Louvain) on reduced embeddings.  
   - Select clustering resolution by evaluating stability and biological interpretability.

8. annotate_cell_types  
   - Assign cell type labels based on canonical marker genes or reference atlases.

9. find_marker_genes  
   - Identify differentially expressed genes that define each cluster or cell type.

---

**Reasoning Approach**  
For any dataset you work on, your thought process should be:  

1. **Understand input data**  
   - What are the raw matrices, what’s in the metadata, and what conditions or batches exist?  
   - Are there multiple datasets that need integration?  

2. **Initial inspection**  
   - Verify dimensions, number of features, basic QC metrics.  

3. **QC and preprocessing**  
   - Remove ambient RNA contamination if needed.  
   - Filter poor-quality cells.  
   - Normalize counts for downstream analysis.  
   - Identify and remove doublets.  

4. **Data integration**  
   - If multiple datasets or batches exist, perform integration and batch correction.  
   - Ensure biological variability is preserved while technical artifacts are reduced.  

5. **Dimensionality reduction & clustering**  
   - Use PCA, then a non-linear method (UMAP/t-SNE).  
   - Apply clustering; evaluate different resolutions for stability and interpretability.  

6. **Biological interpretation**  
   - Annotate clusters using known marker genes or reference datasets.  
   - Find marker genes and interpret biological roles.  

7. **Iterate if needed**  
   - If QC or clustering reveals issues, revisit earlier steps.  

---

**Critical Notes for Execution**  
- Always report intermediate statistics (cells kept/removed, contamination fraction, number of clusters).  
- When deciding parameters (e.g., Leiden resolution), explain your reasoning and justify based on biological context.  
- Integration should happen *after* QC and doublet removal but *before* final clustering.  
- Ambient RNA correction happens early, before filtering and normalization.  
- Clustering and annotation should be reproducible; note seeds and parameters.  
- Keep the workflow modular so steps can be reused for other projects.

Today's date: {date.today()}
""".strip()

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=TranscriptomicsExpertState
    )
