"""
Agent service for managing LLM-based interactions.

This service provides a layer of abstraction around the LangChain agent
setup and integration with the application's tools and services.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_aws import ChatBedrockConverse
from langchain.callbacks.base import BaseCallbackHandler

from config.settings import get_settings
from utils.logger import get_logger

from lobster.tools.geo_service import GEOService
from lobster.tools.quality_service import QualityService
from lobster.tools.clustering_service import ClusteringService
from lobster.tools.bulk_rnaseq_service import BulkRNASeqService
from lobster.tools.enhanced_singlecell_service import EnhancedSingleCellService
from services.file_upload_service import FileUploadService
from lobster.tools.pubmed_service import PubMedService

from core.data_manager import DataManager
from dotenv import load_dotenv

logger = get_logger(__name__)
settings = get_settings()

class AgentService:
    """
    Service for managing LLM-based agent interactions.
    
    This class sets up and manages the LLM agent, tools, memory, and callbacks.
    It provides a clean interface for interacting with the agent.
    """
    
    def __init__(
        self, 
        data_manager: DataManager,
        system_message: str,
        callback_handler: Optional[BaseCallbackHandler] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the agent service.
        
        Args:
            data_manager: DataManager instance
            system_message: System message for the LLM agent
            callback_handler: Optional callback handler for agent events
            chat_history: Optional list of chat messages from session state
        """
        logger.info("Starting AgentService initialization")
        logger.debug(f"System message length: {len(system_message)} characters")
        logger.debug(f"Callback handler provided: {callback_handler is not None}")
        logger.debug(f"Chat history length: {len(chat_history or [])}")
        
        self.data_manager = data_manager
        self.system_message = system_message
        self.callback_handler = callback_handler
        self.chat_history = chat_history or []
        
        logger.info("Initializing LangChain agent")
        self.agent = self._initialize_agent()
        logger.info("AgentService initialization completed successfully")
    
    def _initialize_agent(self):
        """
        Initialize the LangChain agent with tools and memory.
        
        Returns:
            Agent: LangChain agent instance
        """
        logger.info("Setting up agent with tools and LLM")
        
        # Initialize services
        geo_service = GEOService(self.data_manager)
        quality_service = QualityService(self.data_manager)
        clustering_service = ClusteringService(self.data_manager)
        
        # Set up tools
        tools = self._setup_tools(geo_service, quality_service, clustering_service)
        
        # Set up LLM
        llm = self._setup_llm()
        
        # Set up memory with existing chat history
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Load existing chat history from session state into memory
        if self.chat_history:
            logger.info(f"Loading {len(self.chat_history)} messages into agent memory")
            
            for message in self.chat_history:
                if message["role"] == "user":
                    memory.chat_memory.add_user_message(message["content"])
                elif message["role"] == "assistant":
                    memory.chat_memory.add_ai_message(message["content"])
        
        # Set up callbacks
        # Initialize agent
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": self.system_message},
            callbacks=[self.callback_handler] if self.callback_handler else None
        )
        
        return agent
    
    def _setup_tools(
        self, 
        geo_service: GEOService,
        quality_service: QualityService,
        clustering_service: ClusteringService
    ) -> List[Tool]:
        """
        Set up tools for the agent.
        """
        # Initialize additional services
        bulk_rnaseq_service = BulkRNASeqService(self.data_manager)
        enhanced_sc_service = EnhancedSingleCellService(self.data_manager)
        file_upload_service = FileUploadService(self.data_manager)
        pubmed_service = PubMedService(parse=None, data_manager=self.data_manager)
        
        # Create wrapper function for tracking tool usage
        def track_tool_usage(tool_name, description, func):
            def wrapper(*args, **kwargs):
                # Extract parameters from args/kwargs
                params = kwargs.copy()
                if args and len(args) > 0:
                    params['args'] = [str(arg) for arg in args]
                
                # Log the tool usage
                self.data_manager.log_tool_usage(
                    tool_name=tool_name,
                    parameters=params,
                    description=description
                )
                
                # Call the original function
                return func(*args, **kwargs)
            return wrapper
        
        tools = [
            # Existing tools
            Tool(
                name="download_geo_dataset",
                description="Download a dataset from GEO database using accession number (e.g., GSE109564). ONLY USE THIS IF USER GIVES YOU THE GEO ID",
                func=track_tool_usage(
                    "download_geo_dataset",
                    "Download dataset from NCBI Gene Expression Omnibus",
                    lambda query: geo_service.download_dataset(query)
                )
            ),
            Tool(
                name="assess_data_quality",
                description="Assess the quality of loaded single-cell RNA-seq data. Optionally specify parameters: min_genes, max_mt_pct, max_ribo_pct, min_housekeeping_score.",
                func=lambda query: self._parse_qc_parameters(query, quality_service)
            ),
            Tool(
                name="cluster_cells",
                description="Perform cell clustering and generate UMAP visualization. Options: resolution=X.X (clustering granularity), demo_mode (use for faster processing on large matrices), subsample_size=N (limit analysis to N cells), skip_marker_genes (skip time-consuming marker gene identification). Example: 'cluster_cells with resolution=0.8 and demo_mode' or 'cluster with demo_mode and subsample_size=2000'.",
                func=lambda query: self._parse_clustering_request(query)
            ),
            
            # New bulk RNA-seq tools
            # Tool(
            #     name="run_fastqc",
            #     description="Run FastQC quality control on uploaded FASTQ files.",
            #     func=lambda query: self._run_fastqc_from_metadata()
            # ),
            # Tool(
            #     name="run_multiqc",
            #     description="Run MultiQC to aggregate quality control results.",
            #     func=lambda _: bulk_rnaseq_service.run_multiqc()
            # ),
            # Tool(
            #     name="run_salmon_quantification",
            #     description="Run Salmon quantification on FASTQ files. Requires index path.",
            #     func=lambda query: self._run_salmon_from_query(query)
            # ),
            # Tool(
            #     name="run_deseq2_analysis",
            #     description="Run DESeq2 differential expression analysis on count data.",
            #     func=lambda query: self._run_deseq2_from_query(query)
            # ),
            Tool(
                name="run_enrichment_analysis",
                description="Run GO/KEGG pathway enrichment analysis on differential genes.",
                func=lambda query: self._run_enrichment_from_query(query)
            ),
            
            # Enhanced single-cell tools
            Tool(
                name="detect_doublets",
                description="Detect doublets in single-cell data using Scrublet. Optional parameters: expected_doublet_rate, threshold.",
                func=lambda query: self._parse_doublet_parameters(query, enhanced_sc_service)
            ),
            Tool(
                name="annotate_cell_types",
                description="Annotate cell types using marker genes.",
                func=track_tool_usage(
                    "annotate_cell_types",
                    "Annotation of cell types based on marker genes",
                    lambda _: enhanced_sc_service.annotate_cell_types()
                )
            ),
            Tool(
                name="find_marker_genes",
                description="Find marker genes for clusters or cell types.",
                func=lambda query: self._find_markers_from_query(query)
            ),
            # Tool(
            #     name="run_pathway_analysis",
            #     description="Run pathway analysis on marker genes from specific cell types.",
            #     func=track_tool_usage(
            #         "run_pathway_analysis",
            #         "Pathway analysis on marker genes from cell types",
            #         lambda query: enhanced_sc_service.run_pathway_analysis()
            #     )
            # ),
            
            # PubMed integration tools
            # Tool(
            #     name="search_pubmed",
            #     description="Search PubMed for scientific literature. Provide a query to find relevant publications.",
            #     func=track_tool_usage(
            #         "search_pubmed",
            #         "Searched PubMed for scientific literature",
            #         lambda query: pubmed_service.search_pubmed(query)
            #     )
            # ),
            # Tool(
            #     name="find_geo_from_doi",
            #     description="Find GEO accession numbers mentioned in a publication by DOI. REQUIREMENTS: 1) Input MUST be a properly formatted DOI (typically starting with '10.') 2) DO NOT use this tool with PubMed URLs or any URL format 3) If the user provides a PubMed URL instead of a DOI, DO NOT use this tool - ask the user to provide the DOI instead 4) The DOI must be in the standard format (e.g., 10.1038/s41586-021-03659-0) without any additional text",
            #     func=track_tool_usage(
            #         "find_geo_from_doi",
            #         "Located GEO datasets referenced in publication by DOI",
            #         lambda doi: pubmed_service.find_geo_from_doi(doi)
            #     )
            # ),
            # Tool(
            #     name="find_literature_marker_genes",
            #     description="Find literature about marker genes for specific cell types or diseases. Format: 'cell_type=T cells, disease=cancer' (disease is optional).",
            #     func=lambda query: self._parse_marker_genes_literature_query(query)
            # ),
            # Tool(
            #     name="find_protocol_information",
            #     description="Find protocol information for a specific bioinformatics technique. Provide the technique name to retrieve relevant protocols.",
            #     func=track_tool_usage(
            #         "find_protocol_information",
            #         "Retrieved protocol information for bioinformatics technique",
            #         lambda technique: pubmed_service.find_protocol_information(technique)
            #     )
            # ),
            
            # Data inspection tool
            Tool(
                name="get_data_summary",
                description="Get a summary of the currently loaded data, including shape, metadata, processing history, and available plots. Use this to understand what data is currently available for analysis.",
                func=track_tool_usage(
                    "get_data_summary",
                    "Retrieved summary of currently loaded data",
                    lambda _: self._get_data_summary()
                )
            ),
            
            # # Plot regeneration tool
            # Tool(
            #     name="regenerate_plot",
            #     description="Regenerate a plot with a new unique ID to avoid Streamlit ID conflicts. Use this if you encounter an error about 'multiple plotly_chart elements with the same auto-generated ID'. Optionally specify a plot_id, otherwise regenerates the latest plot.",
            #     func=track_tool_usage(
            #         "regenerate_plot",
            #         "Regenerated a plot with new ID",
            #         lambda query: self.regenerate_plot(plot_id=query if query else None)
            #     )
            # ),
        ]
        
        return tools
    
    def _setup_llm(self, max_tokens=4096):
        """
        Set up the LLM for the agent.
        
        Returns:
            BaseLLM: LangChain LLM instance
        """
        load_dotenv()
        # Get API key from settings or environment
        
        model_params = {
            "model_id": settings.LLM_MODEL,
            "temperature": settings.LLM_TEMPERATURE,
            "region_name": settings.REGION,
            "aws_access_key_id": settings.AWS_BEDROCK_ACCESS_KEY,
            "aws_secret_access_key": settings.AWS_BEDROCK_SECRET_ACCESS_KEY,
            "max_tokens": settings.LLM_MODEL_MAX_TOKENS
        }        

        # Initialize the LLM
        llm = ChatBedrockConverse(**model_params)
        
        return llm
    
    def _parse_qc_parameters(self, query: str, quality_service) -> str:
        """
        Parse quality control parameters from query string.
        
        Args:
            query: User query potentially containing QC parameters
            quality_service: Quality service instance
            
        Returns:
            str: Quality assessment results
        """
        # Default values
        min_genes = 500
        max_mt_pct = 20.0
        max_ribo_pct = 50.0
        min_housekeeping_score = 1.0
        
        # Extract parameters if provided
        import re
        
        min_genes_match = re.search(r'min_genes[=\s]+([0-9]+)', query.lower())
        if min_genes_match:
            try:
                min_genes = int(min_genes_match.group(1))
                logger.info(f"Using custom min_genes: {min_genes}")
            except ValueError:
                logger.warning(f"Could not parse min_genes from {min_genes_match.group(1)}")
        
        max_mt_match = re.search(r'max_mt_pct[=\s]+([0-9.]+)', query.lower())
        if max_mt_match:
            try:
                max_mt_pct = float(max_mt_match.group(1))
                logger.info(f"Using custom max_mt_pct: {max_mt_pct}")
            except ValueError:
                logger.warning(f"Could not parse max_mt_pct from {max_mt_match.group(1)}")
        
        max_ribo_match = re.search(r'max_ribo_pct[=\s]+([0-9.]+)', query.lower())
        if max_ribo_match:
            try:
                max_ribo_pct = float(max_ribo_match.group(1))
                logger.info(f"Using custom max_ribo_pct: {max_ribo_pct}")
            except ValueError:
                logger.warning(f"Could not parse max_ribo_pct from {max_ribo_match.group(1)}")
        
        min_hk_match = re.search(r'min_housekeeping_score[=\s]+([0-9.]+)', query.lower())
        if min_hk_match:
            try:
                min_housekeeping_score = float(min_hk_match.group(1))
                logger.info(f"Using custom min_housekeeping_score: {min_housekeeping_score}")
            except ValueError:
                logger.warning(f"Could not parse min_housekeeping_score from {min_hk_match.group(1)}")
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="assess_data_quality",
            parameters={
                "min_genes": min_genes,
                "max_mt_pct": max_mt_pct,
                "max_ribo_pct": max_ribo_pct,
                "min_housekeeping_score": min_housekeeping_score
            },
            description="Quality assessment of single-cell RNA-seq data"
        )
        
        # Run quality assessment with parsed parameters
        return quality_service.assess_quality(
            min_genes=min_genes,
            max_mt_pct=max_mt_pct,
            max_ribo_pct=max_ribo_pct,
            min_housekeeping_score=min_housekeeping_score
        )
    
    def _parse_clustering_request(self, query: str) -> str:
        """
        Parse clustering request and extract parameters.
        
        Args:
            query: User query string
            
        Returns:
            str: Clustering results
        """
        # Try to extract parameters
        resolution = None
        demo_mode = False
        subsample_size = None
        skip_steps = []
        
        import re
        # Extract resolution
        res_match = re.search(r'resolution[=\s]+([0-9.]+)', query.lower())
        if res_match:
            try:
                resolution = float(res_match.group(1))
                logger.info(f"Extracted resolution: {resolution}")
            except ValueError:
                logger.warning(f"Could not parse resolution from {res_match.group(1)}")
        
        # Check for demo mode flag
        if re.search(r'demo[_\s]?mode|fast|quick', query.lower()):
            demo_mode = True
            logger.info("Demo mode enabled")
        
        # Extract subsample size
        sample_match = re.search(r'subsample[_\s]?size[=\s]+([0-9]+)', query.lower())
        if sample_match:
            try:
                subsample_size = int(sample_match.group(1))
                logger.info(f"Using subsample size: {subsample_size}")
            except ValueError:
                logger.warning(f"Could not parse subsample_size from {sample_match.group(1)}")
        
        # Check for steps to skip
        if 'skip_marker_genes' in query.lower() or 'skip markers' in query.lower():
            skip_steps.append('marker_genes')
            logger.info("Skipping marker gene identification")
        
        # Call the clustering service with extracted parameters
        clustering_service = ClusteringService(self.data_manager)
        
        # Set progress callback if UI has one available
        if hasattr(self, 'callback_handler') and self.callback_handler and hasattr(self.callback_handler, 'progress_callback'):
            clustering_service.set_progress_callback(self.callback_handler.progress_callback)
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="cluster_cells",
            parameters={
                "resolution": resolution if resolution is not None else "default",
                "demo_mode": demo_mode,
                "subsample_size": subsample_size,
                "skip_steps": skip_steps
            },
            description="Performed cell clustering and UMAP visualization" + 
                      (" (demo mode)" if demo_mode else "")
        )
        
        # Estimate processing time if demo_mode is specified
        if demo_mode:
            # Get current data dimensions
            n_cells = self.data_manager.current_data.shape[0]
            n_genes = self.data_manager.current_data.shape[1]
            
            # Get time estimates
            time_estimates = clustering_service.estimate_processing_time(n_cells, n_genes)
            logger.info(f"Processing time estimates: Standard: {time_estimates['standard']:.2f}s, Demo: {time_estimates['demo']:.2f}s")
        
        # Run clustering with all extracted parameters
        return clustering_service.cluster_and_visualize(
            resolution=resolution,
            demo_mode=demo_mode,
            subsample_size=subsample_size,
            skip_steps=skip_steps
        )
        
    def _run_fastqc_from_metadata(self) -> str:
        """Run FastQC using files from metadata."""
        bulk_service = BulkRNASeqService(self.data_manager)
        
        if 'fastq_files' in self.data_manager.current_metadata:
            file_info = self.data_manager.current_metadata['fastq_files']['files']
            file_paths = [f['path'] for f in file_info]
            
            # Check if files exist (needed for tests)
            valid_paths = []
            for path in file_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"File not found: {path}")
            
            if valid_paths:
                # Log tool usage for reproducibility
                self.data_manager.log_tool_usage(
                    tool_name="run_fastqc",
                    parameters={"file_count": len(valid_paths)},
                    description="Quality control assessment on FASTQ files using FastQC"
                )
                
                return bulk_service.run_fastqc(valid_paths)
            else:
                return "No valid FASTQ files found. Files may have been moved or deleted."
        else:
            return "No FASTQ files found. Please upload FASTQ files first."

    def _run_salmon_from_query(self, query: str) -> str:
        """Parse Salmon quantification request."""
        bulk_service = BulkRNASeqService(self.data_manager)
        
        # Extract index path from query
        import re
        index_match = re.search(r'index[=\s]+([^\s]+)', query.lower())
        
        if not index_match:
            return "Please specify the Salmon index path (e.g., 'index=/path/to/index')"
        
        index_path = index_match.group(1)
        
        if 'fastq_files' in self.data_manager.current_metadata:
            file_info = self.data_manager.current_metadata['fastq_files']['files']
            file_paths = [f['path'] for f in file_info]
            sample_names = [f['filename'].replace('.fastq', '').replace('.fq', '') for f in file_info]
            
            # Log tool usage for reproducibility
            self.data_manager.log_tool_usage(
                tool_name="run_salmon_quantification",
                parameters={
                    "index_path": index_path,
                    "sample_count": len(sample_names)
                },
                description="Transcript quantification using Salmon"
            )
            
            return bulk_service.run_salmon_quantification(file_paths, index_path, sample_names)
        else:
            return "No FASTQ files found. Please upload FASTQ files first."

    def _run_deseq2_from_query(self, query: str) -> str:
        """Parse DESeq2 analysis request."""
        bulk_service = BulkRNASeqService(self.data_manager)
        
        # Extract design formula if provided
        import re
        design_match = re.search(r'design[=\s]+([^,\s]+)', query.lower())
        design_formula = design_match.group(1) if design_match else "~ condition"
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="run_deseq2_analysis",
            parameters={"design_formula": design_formula},
            description="Differential expression analysis using DESeq2"
        )
        
        return bulk_service.run_deseq2_analysis(design_formula=design_formula)

    def _run_enrichment_from_query(self, query: str) -> str:
        """Parse enrichment analysis request."""
        bulk_service = BulkRNASeqService(self.data_manager)
        
        # Extract analysis type
        analysis_type = "GO"
        if "kegg" in query.lower():
            analysis_type = "KEGG"
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="run_enrichment_analysis",
            parameters={"analysis_type": analysis_type},
            description=f"Pathway enrichment analysis using {analysis_type} databases"
        )
        
        return bulk_service.run_enrichment_analysis(analysis_type=analysis_type)

    def _find_markers_from_query(self, query: str) -> str:
        """Parse marker gene finding request."""
        enhanced_sc_service = EnhancedSingleCellService(self.data_manager)
        
        # Extract cell type or cluster from query
        import re
        cell_type_match = re.search(r'cell[_\s]type[=\s]+([^\s,]+)', query.lower())
        cluster_match = re.search(r'cluster[=\s]+([^\s,]+)', query.lower())
        
        cell_type = cell_type_match.group(1) if cell_type_match else None
        cluster = cluster_match.group(1) if cluster_match else None
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="find_marker_genes",
            parameters={
                "cell_type": cell_type,
                "cluster": cluster
            },
            description="Identification of marker genes" + 
                       (f" for cell type {cell_type}" if cell_type else "") +
                       (f" for cluster {cluster}" if cluster else "")
        )
        
        return enhanced_sc_service.find_marker_genes(cell_type=cell_type, cluster=cluster)
        
    def _parse_doublet_parameters(self, query: str, enhanced_sc_service) -> str:
        """
        Parse doublet detection parameters from query string.
        
        Args:
            query: User query potentially containing doublet detection parameters
            enhanced_sc_service: Enhanced single-cell service instance
            
        Returns:
            str: Doublet detection results
        """
        # Default values
        expected_doublet_rate = 0.025  # Default from publication
        threshold = None  # Default is None, which uses the automatic threshold
        
        # Extract parameters if provided
        import re
        
        rate_match = re.search(r'expected_doublet_rate[=\s]+([0-9.]+)', query.lower())
        if rate_match:
            try:
                expected_doublet_rate = float(rate_match.group(1))
                logger.info(f"Using custom expected_doublet_rate: {expected_doublet_rate}")
            except ValueError:
                logger.warning(f"Could not parse expected_doublet_rate from {rate_match.group(1)}")
        
        threshold_match = re.search(r'threshold[=\s]+([0-9.]+)', query.lower())
        if threshold_match:
            try:
                threshold = float(threshold_match.group(1))
                logger.info(f"Using custom threshold: {threshold}")
            except ValueError:
                logger.warning(f"Could not parse threshold from {threshold_match.group(1)}")
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="detect_doublets",
            parameters={
                "expected_doublet_rate": expected_doublet_rate,
                "threshold": threshold
            },
            description="Detection of cell doublets in single-cell RNA-seq data using Scrublet"
        )
        
        # Run doublet detection with parsed parameters
        return enhanced_sc_service.detect_doublets(
            expected_doublet_rate=expected_doublet_rate,
            threshold=threshold
        )
        
    def _parse_marker_genes_literature_query(self, query: str) -> str:
        """
        Parse query for literature about marker genes.
        
        Args:
            query: User query string like 'cell_type=T cells, disease=cancer'
            
        Returns:
            str: Literature results for marker genes
        """
        logger.info(f"Parsing marker genes literature query: {query}")
        
        # Extract cell type and optional disease
        import re
        cell_type_match = re.search(r'cell[_\s]type[=\s]+([^,]+)', query)
        disease_match = re.search(r'disease[=\s]+([^,]+)', query)
        
        if not cell_type_match: 
            return "Please specify a cell type using format 'cell_type=X'"
            
        cell_type = cell_type_match.group(1).strip()
        disease = disease_match.group(1).strip() if disease_match else None
        
        # Get the PubMed service instance
        pubmed_service = PubMedService(parse=None, data_manager=self.data_manager, api_key=settings.NCBI_API_KEY)
        
        # Log tool usage for reproducibility
        self.data_manager.log_tool_usage(
            tool_name="find_literature_marker_genes",
            parameters={
                "cell_type": cell_type,
                "disease": disease if disease else "None"
            },
            description=f"Searched literature for marker genes of {cell_type}" +
                       (f" in {disease}" if disease else "")
        )
        
        return pubmed_service.find_marker_genes(cell_type=cell_type, disease=disease)

        
    def update_memory_from_chat_history(self, chat_history: List[Dict[str, Any]]) -> None:
        """
        Update the agent's memory with the current chat history.
        
        Args:
            chat_history: List of chat messages from session state
        """
        if not chat_history:
            return
            
        # Clear existing memory and reload from chat history
        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'chat_memory'):
            logger.info(f"Updating agent memory with {len(chat_history)} messages")
            # Reset memory
            self.agent.memory.chat_memory.clear()
            
            # Reload from chat history
            
            for message in chat_history:
                if message["role"] == "user":
                    self.agent.memory.chat_memory.add_user_message(message["content"])
                elif message["role"] == "assistant":
                    self.agent.memory.chat_memory.add_ai_message(message["content"])
    
    def regenerate_plot(self, plot_id: str = None) -> str:
        """
        Regenerate a specific plot or the latest plot.
        
        Args:
            plot_id: ID of the plot to regenerate (if None, regenerates the latest)
            
        Returns:
            str: Result message indicating success or failure
        """
        if not plot_id and not self.data_manager.latest_plots:
            return "No plots available to regenerate."
        
        try:
            # Get the plot entry to regenerate
            if plot_id:
                # Find plot by ID
                plot_entry = None
                for entry in self.data_manager.latest_plots:
                    if entry["id"] == plot_id:
                        plot_entry = entry
                        break
                if not plot_entry:
                    return f"Plot with ID {plot_id} not found."
            else:
                # Use the latest plot
                plot_entry = self.data_manager.latest_plots[-1]
            
            # Generate a new unique ID for the regenerated plot
            new_plot = plot_entry["figure"]
            title = plot_entry["title"]
            source = plot_entry["source"]
            
            # Add regenerated plot with new ID
            new_id = self.data_manager.add_plot(new_plot, title=f"{title} (Regenerated)", source=source)
            
            return f"Plot '{title}' has been regenerated with new ID: {new_id}"
            
        except Exception as e:
            logger.exception(f"Error regenerating plot: {e}")
            return f"Error regenerating plot: {str(e)}"
    
    def _get_data_summary(self) -> str:
        """
        Get a comprehensive summary of the currently loaded data and analysis state.
        
        Returns:
            str: Formatted summary of current data, metadata, plots and analysis history
        """
        if not self.data_manager.has_data():
            return "No data is currently loaded. Use the download_geo_dataset tool with a valid GEO ID to load data."
            
        # Get basic data summary
        data_summary = self.data_manager.get_data_summary()
        
        # Build a comprehensive response
        response = "## Current Data Summary\n\n"
        
        # Data dimensions and basic info
        response += f"**Data Dimensions:** {data_summary['shape'][0]} cells × {data_summary['shape'][1]} genes\n"
        response += f"**Memory Usage:** {data_summary['memory_usage']}\n"
        
        # Sample preview
        if data_summary.get('sample_names'):
            response += f"**Sample Preview:** {', '.join(data_summary['sample_names'])}\n"
        
        # Source information
        if 'source' in self.data_manager.current_metadata:
            response += f"**Data Source:** {self.data_manager.current_metadata['source']}\n"
            
        # Metadata summary
        response += "\n## Available Metadata\n\n"
        if data_summary.get('metadata_keys'):
            for key in data_summary['metadata_keys']:
                value = self.data_manager.current_metadata.get(key)
                if isinstance(value, (str, int, float, bool)):
                    response += f"- **{key}:** {value}\n"
                elif isinstance(value, dict) and len(str(value)) < 100:
                    response += f"- **{key}:** {value}\n"
                elif isinstance(value, (list, dict)):
                    response += f"- **{key}:** [Complex data structure with {len(value)} elements]\n"
                else:
                    response += f"- **{key}:** [Data available]\n"
        else:
            response += "No additional metadata available.\n"
        
        # Processing history
        response += "\n## Recent Processing Steps\n\n"
        if data_summary.get('processing_log'):
            for step in data_summary['processing_log']:
                response += f"- {step}\n"
        else:
            response += "No processing steps recorded yet.\n"
            
        # Available visualizations
        response += "\n## Available Visualizations\n\n"
        if self.data_manager.latest_plots:
            for i, _ in enumerate(self.data_manager.latest_plots):
                response += f"- Plot {i+1}\n"
        else:
            response += "No visualizations have been generated yet.\n"
            
        # Analysis state based on AnnData
        response += "\n## Current Analysis State\n\n"
        if self.data_manager.adata is not None:
            # Check for common analysis results in AnnData
            available_analyses = []
            
            if 'leiden' in self.data_manager.adata.obs.columns:
                n_clusters = len(self.data_manager.adata.obs['leiden'].unique())
                available_analyses.append(f"Clustering complete: {n_clusters} clusters identified")
                
            if 'cell_type' in self.data_manager.adata.obs.columns:
                n_cell_types = len(self.data_manager.adata.obs['cell_type'].unique())
                available_analyses.append(f"Cell type annotation complete: {n_cell_types} cell types identified")
                
            if 'doublet_score' in self.data_manager.adata.obs.columns:
                n_doublets = sum(self.data_manager.adata.obs.get('predicted_doublet', [False]))
                available_analyses.append(f"Doublet detection complete: {n_doublets} potential doublets identified")
                
            if 'X_umap' in self.data_manager.adata.obsm:
                available_analyses.append("UMAP dimensional reduction available")
                
            if 'rank_genes_groups' in self.data_manager.adata.uns:
                available_analyses.append("Marker gene analysis complete")
                
            if available_analyses:
                for analysis in available_analyses:
                    response += f"- {analysis}\n"
            else:
                response += "Basic data loaded, no advanced analyses completed yet.\n"
        else:
            response += "Data loaded but not processed into analysis-ready format yet.\n"
            
        # Recent tool usage
        if self.data_manager.tool_usage_history:
            response += "\n## Recent Tool Usage\n\n"
            # Get the last 3 tools used
            recent_tools = self.data_manager.tool_usage_history[-3:]
            for tool_entry in reversed(recent_tools):
                response += f"- {tool_entry['tool']} ({tool_entry['timestamp']}): {tool_entry['description']}\n"
                
        return response
    
    def run_agent(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Run the agent with a user query.
        
        Args:
            query: User query string
            chat_history: Optional updated chat history to sync memory
            
        Returns:
            str: Agent response
        """
        logger.info(f"Running agent with query: {query[:50]}...")
        
        # Update memory with latest chat history if provided
        if chat_history:
            self.update_memory_from_chat_history(chat_history)
        
        try:
            response = self.agent.run(query)
            return response
        except Exception as e:
            logger.exception(f"Error running agent: {e}")
            return f"I encountered an error: {str(e)}"
