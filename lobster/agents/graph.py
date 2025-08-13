"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using langgraph_supervisor package for hierarchical multi-agent coordination.
"""

from langgraph.checkpoint.memory import MemorySaver

from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from langchain_aws import ChatBedrock

from .supervisor import create_supervisor_prompt
from .transcriptomics_expert import transcriptomics_expert
from .method_expert import method_expert
from ..core.data_manager import DataManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

def create_bioinformatics_graph(
    data_manager: DataManager,
    checkpointer: MemorySaver = None,
    callback_handler=None
):
    """Create the bioinformatics multi-agent graph using langgraph_supervisor."""
    logger.info("Creating bioinformatics multi-agent graph")
    
    # Get model configuration for the supervisor
    settings = get_settings()
    model_params = settings.get_agent_llm_params('supervisor')
    supervisor_model = ChatBedrock(**model_params)
    
    if callback_handler and hasattr(supervisor_model, 'with_config'):
        supervisor_model = supervisor_model.with_config(callbacks=[callback_handler])
    
    # Create worker agents
    agents = []
    
    # Create transcriptomics expert agent
    transcriptomics_agent = transcriptomics_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name='transcriptomics_expert_agent',
        handoff_tools=None
    )
    agents.append(transcriptomics_agent)
    
    # Create method expert agent
    method_agent = method_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name='method_expert_agent',
        handoff_tools=None
    )
    agents.append(method_agent)
    
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
        add_handoff_messages=False,
        include_agent_nameand='inline',
        # Change from "full_history" to "messages" or "last_message"
        output_mode="last_message",  # This ensures the actual messages are returned
        tools=[forwarding_tool]
    )
    
    # Compile the graph with the provided checkpointer
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Bioinformatics multi-agent graph created successfully")
    return graph

# Import settings here to avoid circular imports
from config.settings import get_settings