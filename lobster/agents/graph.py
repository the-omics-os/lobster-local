"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using langgraph_supervisor package for hierarchical multi-agent coordination.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_core.tools import tool
 
from lobster.agents.langgraph_supervisor import create_supervisor
from lobster.agents.langgraph_supervisor import create_forward_message_tool

from langchain_aws import ChatBedrockConverse

from lobster.agents.supervisor import create_supervisor_prompt
from lobster.agents.state import OverallState
from lobster.tools.handoff_tool import create_custom_handoff_tool
from lobster.config.settings import get_settings
from lobster.config.supervisor_config import SupervisorConfig
from lobster.config.agent_registry import get_worker_agents, import_agent_factory
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger
from typing import Optional

logger = get_logger(__name__)

def create_bioinformatics_graph(
    data_manager: DataManagerV2,
    checkpointer: InMemorySaver = None,
    store: InMemoryStore = None,
    callback_handler=None,
    manual_model_params: dict = None,
    supervisor_config: Optional[SupervisorConfig] = None
):
    """Create the bioinformatics multi-agent graph using langgraph_supervisor.
    
    Note: When invoking this graph, set the recursion_limit in the config to prevent
    hitting the default limit of 25. Example:
        config = {"recursion_limit": 100, ...}
        graph.invoke(input, config)
    """
    logger.debug("Creating bioinformatics multi-agent graph")
    
    # Get model configuration for the supervisor
    settings = get_settings()

    #ensure this for later
    if manual_model_params:
        # Use provided manual model parameters if available
        model_params = manual_model_params
    else:
        model_params = settings.get_agent_llm_params('supervisor')

    supervisor_model = ChatBedrockConverse(**model_params)
    
    if callback_handler and hasattr(supervisor_model, 'with_config'):
        supervisor_model = supervisor_model.with_config(callbacks=[callback_handler])
    
    # Create worker agents dynamically from registry
    agents = []
    handoff_tools = []    

    # Get all worker agents from the registry
    worker_agents = get_worker_agents()

    for agent_name, agent_config in worker_agents.items():
        # Import the factory function dynamically
        factory_function = import_agent_factory(agent_config.factory_function)

        # Create the agent
        agent = factory_function(
            data_manager=data_manager,
            callback_handler=callback_handler,
            agent_name=agent_config.name,
            handoff_tools=None
        )
        agents.append(agent)

        # Create handoff tool if configured
        if agent_config.handoff_tool_name and agent_config.handoff_tool_description:
            handoff_tool = create_custom_handoff_tool(
                agent_name=agent_config.name,
                name=agent_config.handoff_tool_name,
                description=agent_config.handoff_tool_description
            )
            handoff_tools.append(handoff_tool)        

        logger.debug(f"Created agent: {agent_config.display_name} ({agent_config.name})")


    # Supervisor only tools
    @tool
    def list_available_modalities() -> str:
        """
        List all currently loaded modalities and their details.
        
        Returns:
            str: Formatted list of available modalities
        """
        try:
            modalities = data_manager.list_modalities()
            
            if not modalities:
                return "No modalities currently loaded. Use download_geo_dataset or upload_data_file to load data."
            
            response = f"Currently loaded modalities ({len(modalities)}):\n\n"
            
            for mod_name in modalities:
                adata = data_manager.get_modality(mod_name)
                response += f"**{mod_name}**:\n"
                response += f"  - Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                response += f"  - Obs columns: {len(adata.obs.columns)} ({', '.join(list(adata.obs.columns)[:3])}...)\n"
                response += f"  - Var columns: {len(adata.var.columns)} ({', '.join(list(adata.var.columns)[:3])}...)\n"
                if adata.layers:
                    response += f"  - Layers: {', '.join(list(adata.layers.keys()))}\n"
                response += "\n"
            
            # Add workspace information
            workspace_status = data_manager.get_workspace_status()
            response += f"Workspace: {workspace_status['workspace_path']}\n"
            response += f"Available adapters: {len(workspace_status['registered_adapters'])}\n"
            response += f"Available backends: {len(workspace_status['registered_backends'])}"
            
            return response
                
        except Exception as e:
            logger.error(f"Error listing available data: {e}")
            return f"Error listing available data: {str(e)}"    
    
    # Get list of active agents that were successfully created
    active_agent_names = [agent.name for agent in agents]

    # Create supervisor prompt with configuration and active agents
    system_prompt = create_supervisor_prompt(
        data_manager=data_manager,
        config=supervisor_config,
        active_agents=active_agent_names
    )
    
    #add forwarding tool for supervisor. This is useful when the supervisor determines that the worker's response is sufficient and doesn't require further processing or summarization by the supervisor itself.
    forwarding_tool = create_forward_message_tool("supervisor")

    # UPDATED CONFIGURATION - Changed output_mode
    workflow = create_supervisor(
        agents=agents,
        model=supervisor_model,
        prompt=system_prompt,
        supervisor_name="supervisor",
        state_schema=OverallState,
        add_handoff_messages=True,
        add_handoff_back_messages=True,        
        include_agent_name='inline',
        # Change from "full_history" to "messages" or "last_message"
        # output_mode="full_history",  # This ensures the actual messages are returned
        output_mode="last_message",  # This ensures the actual messages are returned
        tools=handoff_tools + [list_available_modalities] + [forwarding_tool]  # Supervisor-only tools (handoff tools are auto-created)
              
    )
    
    # Compile the graph with the provided checkpointer
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
        # debug=True  # Enable debug mode for better visibility
    )
    
    logger.debug("Bioinformatics multi-agent graph created successfully")
    return graph
