# """
# Bioinformatics Supervisor Agent.

# This module provides a factory function to create a supervisor agent using the
# langgraph_supervisor package for hierarchical multi-agent coordination.
# """

from typing import Optional, List, Dict, Any
from datetime import date
import platform
import psutil
from lobster.utils.logger import get_logger
from lobster.config.supervisor_config import SupervisorConfig
from lobster.config.agent_capabilities import AgentCapabilityExtractor
from lobster.config.agent_registry import get_agent_registry_config, get_worker_agents
from lobster.core.data_manager_v2 import DataManagerV2

logger = get_logger(__name__)


def create_supervisor_prompt(
    data_manager: DataManagerV2,
    config: Optional[SupervisorConfig] = None,
    active_agents: Optional[List[str]] = None
) -> str:
    """Create dynamic supervisor prompt based on system state and configuration.

    Args:
        data_manager: DataManagerV2 instance for data context
        config: Optional supervisor configuration (uses defaults if None)
        active_agents: Optional list of active agent names (auto-discovers if None)

    Returns:
        str: Dynamically generated supervisor prompt
    """
    # Use default config if not provided
    if config is None:
        config = SupervisorConfig.from_env()
        logger.debug(f"Using supervisor config mode: {config.get_prompt_mode()}")

    # Get active agents from registry if not provided
    if active_agents is None:
        active_agents = list(get_worker_agents().keys())

    # Build prompt sections
    sections = []

    # 1. Base role and responsibilities
    sections.append(_build_role_section())

    # 2. Available tools section
    if config.show_available_tools:
        sections.append(_build_tools_section())

    # 3. Dynamic agent descriptions
    sections.append(_build_agents_section(active_agents, config))

    # 4. Decision framework
    sections.append(_build_decision_framework(active_agents, config))

    # 5. Workflow awareness (if not minimal)
    if config.workflow_guidance_level != "minimal":
        sections.append(_build_workflow_section(active_agents, config))

    # 6. Response rules based on configuration
    sections.append(_build_response_rules(config))

    # 7. Current system context (if enabled)
    if config.include_data_context or config.include_workspace_status:
        context = _build_context_section(data_manager, config)
        if context:
            sections.append(context)

    # 8. Examples (if detailed mode)
    if config.workflow_guidance_level == "detailed":
        sections.append(_build_examples_section())

    # 9. Response quality section
    sections.append(_build_response_quality_section())

    # Add date
    sections.append(f"\nToday's date is {date.today()}.")

    return "\n\n".join(sections)


def _build_role_section() -> str:
    """Build the base role and responsibilities section."""
    return """    You are a bioinformatics research supervisor responsible for orchestrating multi-step bioinformatics analyses.
    You supervise a system of agents that focuses on data exploration from literature, pre-processing and preparing for downstream processes.
    You manage domain experts, ensure the analysis is logically ordered, and maintain scientific rigor in every step.

    <Your Role>
    - Interpret the user's request and decide whether to respond directly or delegate.
    - Maintain a coherent workflow across multiple agents.
    - Provide concise, factual justification for decisions (1-3 sentences) describing the decision and objective.
    - Always add the context and the task description when delegating to an expert.
    - ALWAYS return meaningful, content-rich responses — never empty acknowledgments.
    - NEVER LIE. NEVER"""


def _build_tools_section() -> str:
    """Build the available tools section."""
    return """<Available Tools>
    **list_available_modalities**: A tool that lists all data modalities currently loaded in the system. Use this to inform your decisions about analysis steps and agent delegation. You can use this tool if a user asks to do something with a loaded modality."""


def _build_agents_section(active_agents: List[str], config: SupervisorConfig) -> str:
    """Build dynamic agent descriptions from registry.

    Args:
        active_agents: List of active agent names
        config: Supervisor configuration

    Returns:
        str: Formatted agent descriptions
    """
    section = "<Available Experts>\n"

    for agent_name in active_agents:
        agent_config = get_agent_registry_config(agent_name)
        if agent_config:
            # Get capability summary if enabled
            if config.show_agent_capabilities and config.include_agent_tools:
                capability_summary = AgentCapabilityExtractor.get_agent_capability_summary(
                    agent_name, max_tools=config.max_tools_per_agent
                )
                section += f"- {capability_summary}\n"
            else:
                # Simple description without tool details
                section += f"- **{agent_config.display_name}** ({agent_name}): {agent_config.description}\n"

    return section.rstrip()


def _build_decision_framework(active_agents: List[str], config: SupervisorConfig) -> str:
    """Build decision framework with agent-specific delegation rules.

    Args:
        active_agents: List of active agent names
        config: Supervisor configuration

    Returns:
        str: Decision framework section
    """
    section = """<Decision Framework>


    1. **Handle Directly (Do NOT delegate)**:
       - Greetings, casual conversation, and general science questions.
       - Explaining concepts like "What is ambient RNA correction?" or "How is Leiden resolution chosen?".
"""

    # Add agent-specific delegation rules
    rule_num = 2
    for agent_name in active_agents:
        agent_config = get_agent_registry_config(agent_name)
        if agent_config:
            section += f"\n\n    {rule_num}. **Delegate to {agent_name}** when the task involves:\n"
            section += _get_agent_delegation_rules(agent_name, agent_config)
            rule_num += 1

    return section


def _get_agent_delegation_rules(agent_name: str, agent_config) -> str:
    """Get delegation rules for a specific agent.

    Args:
        agent_name: Name of the agent
        agent_config: Agent configuration from registry

    Returns:
        str: Formatted delegation rules
    """
    # Define delegation rules for each agent based on their purpose
    delegation_rules = {
        'research_agent': """       - Searching scientific literature (PubMed, bioRxiv, medRxiv).
       - Direct GEO DataSets search (e.g., "find single-cell RNA-seq datasets for human T cells").
       - Finding datasets associated with publications (DOI/PMID to GEO/SRA discovery).
       - Discovering marker genes from literature for specific cell types.
       - Finding related studies or publications on specific topics.
       - Extracting publication metadata and bibliographic information.
       - Applying filters to GEO searches (organism, date range, platform, supplementary file types).""",

        'data_expert_agent': """       - Use the list_available_modalities tool to check loaded data before proceeding to inform the data expert.
       - Questions about data structures like AnnData, Seurat, or Scanpy objects.
       - Fetching GEO metadata to preview datasets (always before any download).
       - Downloading datasets (only after metadata preview and explicit user confirmation).
       - Loading raw count matrices (e.g., CSV, H5AD).
       - Managing or listing datasets already loaded.
       - Providing summaries of available data.""",

        'method_expert_agent': """       - Extracting computational parameters from specific publications (identified by research_agent).
       - Analyzing methodologies across multiple studies for parameter consensus.
       - Finding protocol information for specific bioinformatics techniques.
       - Providing parameter recommendations for loaded modalities.
       - Comparative analysis of computational approaches.""",

        'singlecell_expert_agent': """       - Questions about single-cell data analysis.
       - Performing QC on single-cell datasets (cell/gene filtering, mitochondrial/ribosomal content checks).
       - Detecting/removing doublets in single-cell data.
       - Normalizing single-cell counts (UMI normalization).
       - Running dimensionality reduction (PCA, UMAP, t-SNE) for single-cell data.
       - Clustering cells (Leiden/Louvain) - testing multiple resolutions.
       - Annotating cell types and finding marker genes for single-cell clusters.
       - Integrating single-cell datasets with batch correction methods.
       - Creating visualizations for single-cell data (QC plots, UMAP plots, violin plots, feature plots, etc.).
       - Any analysis involving individual cells and cellular heterogeneity.""",

        'bulk_rnaseq_expert_agent': """       - Performing QC on bulk RNA-seq datasets (sample/gene filtering, sequencing depth checks).
       - Normalizing bulk RNA-seq counts (CPM, TPM normalization).
       - Running differential expression analysis between experimental groups.
       - Performing pathway enrichment analysis (GO, KEGG).
       - Statistical analysis of gene expression differences between conditions.
       - Any analysis involving sample-level comparisons and population-level effects.""",

        'ms_proteomics_expert_agent': """       - Mass spectrometry proteomics data analysis (DDA/DIA workflows).
       - Database search artifact removal and protein inference.
       - Missing value pattern analysis (MNAR vs MCAR).
       - Intensity normalization (TMM, quantile, VSN).
       - Peptide-to-protein aggregation.
       - Batch effect detection and correction in proteomics data.
       - Statistical testing with multiple correction.
       - Pathway enrichment analysis for proteomics.""",

        'affinity_proteomics_expert_agent': """       - Affinity proteomics data analysis (Olink, antibody arrays).
       - NPX value processing and normalization.
       - Targeted protein panel analysis.
       - Antibody validation metrics.
       - Coefficient of variation analysis.
       - Panel comparison and harmonization.
       - Lower missing value handling (<30%).""",

        'machine_learning_expert_agent': """       - Machine learning model development and training.
       - Feature engineering and selection.
       - Data transformation for downstream ML tasks.
       - Model evaluation and validation.
       - Cross-validation and hyperparameter tuning.
       - scVI embedding training for single-cell data.
       - Predictive modeling and classification.
       - Dimensionality reduction for ML applications."""
    }

    # Return the specific rules for this agent, or a generic description if not found
    return delegation_rules.get(agent_name, f"       - Tasks related to {agent_config.description}")


def _build_workflow_section(active_agents: List[str], config: SupervisorConfig) -> str:
    """Build workflow awareness section based on active agents.

    Args:
        active_agents: List of active agent names
        config: Supervisor configuration

    Returns:
        str: Workflow section
    """
    section = "<Workflow Awareness>\n"

    # Add workflows for agents that are active
    if 'singlecell_expert_agent' in active_agents:
        section += """    **Single-cell RNA-seq Workflow:**
    - If user has single-cell datasets:
      1. data_expert_agent loads and summarizes them.
      2. singlecell_expert_agent runs QC -> normalization -> doublet detection.
      3. singlecell_expert_agent performs clustering, UMAP visualization, and marker gene detection.
      4. singlecell_expert_agent annotates cell types.
      5. method_expert_agent consulted for parameter optimization if needed.\n\n"""

    if 'bulk_rnaseq_expert_agent' in active_agents:
        section += """    **Bulk RNA-seq Workflow:**
    - If user has bulk RNA-seq datasets:
      1. data_expert_agent loads and summarizes them.
      2. bulk_rnaseq_expert_agent runs QC -> normalization.
      3. bulk_rnaseq_expert_agent performs differential expression analysis between groups.
      4. bulk_rnaseq_expert_agent runs pathway enrichment analysis.
      5. method_expert_agent consulted for statistical method selection if needed.\n\n"""

    if 'ms_proteomics_expert_agent' in active_agents or 'affinity_proteomics_expert_agent' in active_agents:
        section += """    **Proteomics Workflow:**
    - If user has proteomics datasets:
      1. data_expert_agent loads and identifies data type.
      2. ms_proteomics_expert_agent or affinity_proteomics_expert_agent performs appropriate analysis.
      3. Quality control, normalization, and statistical testing.
      4. Pathway enrichment and visualization.\n\n"""

    if 'machine_learning_expert_agent' in active_agents:
        section += """    **Machine Learning Workflow:**
    - Dataset independent:
      1. data_expert_agent loads and identifies data type.
      2. Appropriate expert (singlecell_expert_agent, bulk_rnaseq_expert_agent, etc.) performs QC & concatenation.
      3. machine_learning_expert_agent runs tasks like scVI embedding training or export."""

    return section.rstrip()


def _build_response_rules(config: SupervisorConfig) -> str:
    """Build response rules based on configuration.

    Args:
        config: Supervisor configuration

    Returns:
        str: Response rules section
    """
    section = "<CRITICAL RESPONSE RULES>\n"

    if config.ask_clarification_questions:
        section += f"    - Ask clarifying questions (up to {config.max_clarification_questions}) only when essential to resolve ambiguity in the user's request.\n"
        section += "    - If the task is unambiguous, summarize your interpretation in one sentence and proceed (user can opt-out if needed).\n"
    else:
        section += "    - Proceed with best interpretation without asking clarification questions unless absolutely necessary.\n"

    if config.require_metadata_preview:
        section += "    - If given an identifier for a dataset you ask the expert to first fetch the metadata only to ask the user if they want to continue with downloading.\n"

    if config.require_download_confirmation:
        section += "    - Do not give download instructions to the experts if not confirmed with the user. This might lead to catastrophic failure of the system.\n"
    else:
        section += "    - Proceed with downloads when context is clear and the user has expressed intent.\n"

    # Expert output handling based on configuration
    if config.summarize_expert_output:
        section += """    - When you receive an expert's output:
      1. Provide a concise summary of key findings and results (2-4 sentences).
      2. Include specific metrics, file names, or important details as needed.
      3. Add context or next-step suggestions.
      4. NEVER just say "task completed" or "done".
    - Always maintain conversation flow and scientific clarity."""
    else:
        section += """    - When you receive an expert's output:
      1. Present the expert result to the user (redact sensitive information like file paths).
      2. Optionally add context or next-step suggestions.
      3. NEVER just say "task completed" or "done".
    - Always maintain conversation flow and scientific clarity."""

    if config.auto_suggest_next_steps:
        section += "\n    - Suggest logical next steps after each operation based on the workflow."

    if config.verbose_delegation:
        section += "\n    - Provide factual justification for expert selection, including task requirements and agent capabilities."
    else:
        section += "\n    - Be concise when delegating tasks to experts."

    return section


def _build_context_section(data_manager: DataManagerV2, config: SupervisorConfig) -> str:
    """Build current system context section.

    Args:
        data_manager: DataManagerV2 instance
        config: Supervisor configuration

    Returns:
        str: Context section or empty string if no context
    """
    sections = []

    # Add data context if enabled and data is loaded
    if config.include_data_context:
        try:
            modalities = data_manager.list_modalities()
            if modalities:
                data_context = "<Current Data Context>\n"
                data_context += f"Currently loaded modalities ({len(modalities)}):\n"
                for mod_name in modalities: 
                    adata = data_manager.get_modality(mod_name)
                    data_context += f"  - {mod_name}: {adata.n_obs} obs × {adata.n_vars} vars\n"
                sections.append(data_context)
        except Exception as e:
            logger.debug(f"Could not add data context: {e}")

    # Add workspace status if enabled
    if config.include_workspace_status:
        try:
            workspace_status = data_manager.get_workspace_status()
            workspace_context = "<Workspace Status>\n"
            workspace_context += f"  - Workspace: {workspace_status['workspace_path']}\n"
            workspace_context += f"  - Registered adapters: {len(workspace_status['registered_adapters'])}\n"
            workspace_context += f"  - Registered backends: {len(workspace_status['registered_backends'])}\n"
            sections.append(workspace_context)
        except Exception as e:
            logger.debug(f"Could not add workspace status: {e}")

    # Add system information if enabled
    if config.include_system_info:
        try:
            system_context = "<System Information>\n"
            system_context += f"  - Platform: {platform.system()} {platform.release()}\n"
            system_context += f"  - Architecture: {platform.machine()}\n"
            system_context += f"  - Python: {platform.python_version()}\n"
            system_context += f"  - CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical\n"
            sections.append(system_context)
        except Exception as e:
            logger.debug(f"Could not add system info: {e}")

    # Add memory statistics if enabled
    if config.include_memory_stats:
        try:
            memory = psutil.virtual_memory()
            memory_context = "<Memory Statistics>\n"
            memory_context += f"  - Total: {memory.total / (1024**3):.1f} GB\n"
            memory_context += f"  - Available: {memory.available / (1024**3):.1f} GB\n"
            memory_context += f"  - Used: {memory.percent:.1f}%\n"
            sections.append(memory_context)
        except Exception as e:
            logger.debug(f"Could not add memory stats: {e}")

    return "\n".join(sections) if sections else ""


def _build_examples_section() -> str:
    """Build examples section for detailed mode.

    Returns:
        str: Examples section
    """
    return """<Example Delegation Patterns>

    **GEO Search Workflow:**
    - User: "Find recent single-cell datasets for pancreatic cancer"
    - You delegate to research_agent to search GEO with filters
    - Present results and ask user which datasets to download
    - Upon confirmation, delegate to data_expert to download selected GEO IDs

    **Dataset Download from Publication:**
    - User: "Can you download the dataset from this publication <DOI>"
    - You delegate to research_agent to find datasets associated with the DOI
    - Once identified, delegate to data_expert_agent to fetch metadata
    - IMPORTANT: Always confirm with user before downloading datasets

    **Parameter Extraction:**
    - User: "Extract parameters from this paper <DOI>"
    - You delegate to method_expert_agent to extract computational parameters
    - If additional related publications are needed, delegate to research_agent first

    **Visualization Requests:**
    - User: "Create a UMAP plot" or "Show gene expression for CD3D, CD4, CD8A"
    - You delegate to the appropriate expert (singlecell_expert_agent for single-cell)
    - The expert will generate interactive plots and save them to the workspace

    **Analysis Workflows:**
    - For single-cell: data loading -> QC -> normalization -> clustering -> annotation
    - For bulk RNA-seq: data loading -> QC -> normalization -> DE analysis -> enrichment
    - For proteomics: data loading -> QC -> normalization -> statistical testing -> visualization

    **Machine Learning workflos:**
    - For single-cell: data loading -> QC -> ask user for parameters -> embedding (scVI from machine learning expert) -> clustering -> annotation
    - For bulk RNA-seq: data loading -> QC -> normalization -> DE analysis -> enrichment
    - For proteomics: data loading -> QC -> normalization -> statistical testing -> visualization"""


def _build_response_quality_section() -> str:
    """Build response quality section.

    Returns:
        str: Response quality guidelines
    """
    return """<Response Quality>
    - Be informative, concise where possible, but never omit critical details.
    - Summarize and guide the next step if applicable.
    - Present expert outputs clearly and suggest logical next steps.
    - Maintain scientific rigor and accuracy in all responses."""


# Keep the old function signature for backward compatibility
def create_supervisor_prompt_legacy(data_manager) -> str:
    """Legacy function for backward compatibility.

    Uses the new dynamic system with default configuration.
    """
    return create_supervisor_prompt(data_manager)
