"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using langgraph_supervisor package for hierarchical multi-agent coordination.
"""

from langgraph.checkpoint.memory import InMemorySaver
 
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from langchain_aws import ChatBedrockConverse

from lobster.agents.supervisor import create_supervisor_prompt
from lobster.agents.state import OverallState
from lobster.config.settings import get_settings
from lobster.config.agent_registry import get_worker_agents, import_agent_factory
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger
from lobster.tools.handoff_tool import create_custom_handoff_tool

logger = get_logger(__name__)

def create_bioinformatics_graph(
    data_manager: DataManagerV2,
    checkpointer: InMemorySaver = None,
    callback_handler=None,
    manual_model_params: dict = None
):
    """Create the bioinformatics multi-agent graph using langgraph_supervisor."""
    logger.info("Creating bioinformatics multi-agent graph")
    
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
        
        logger.info(f"Created agent: {agent_config.display_name} ({agent_config.name})")
    
    # UPDATED SUPERVISOR PROMPT - More explicit about response handling
    system_prompt = create_supervisor_prompt(data_manager)
    
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
        include_agent_name='inline',
        # Change from "full_history" to "messages" or "last_message"
        output_mode="full_history",  # This ensures the actual messages are returned
        # output_mode="last_message",  # This ensures the actual messages are returned
        tools=handoff_tools  # Use dynamically created handoff tools
    )
    
    # Compile the graph with the provided checkpointer
    graph = workflow.compile(
        checkpointer=checkpointer,
        # debug=True  # Enable debug mode for better visibility
    )
    
    logger.info("Bioinformatics multi-agent graph created successfully")
    return graph
