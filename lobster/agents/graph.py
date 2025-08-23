"""
LangGraph multi-agent graph for bioinformatics analysis.

Implementation using langgraph_supervisor package for hierarchical multi-agent coordination.
"""

from langgraph.checkpoint.memory import InMemorySaver

from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from langchain_aws import ChatBedrockConverse

from lobster.agents.supervisor import create_supervisor_prompt
from lobster.agents.data_expert import data_expert
from lobster.agents.singlecell_expert import singlecell_expert
from lobster.agents.bulk_rnaseq_expert import bulk_rnaseq_expert
from lobster.agents.state import OverallState
from lobster.config.settings import get_settings
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
    from lobster.agents.method_expert import method_expert
    
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
    
    # Create worker agents
    agents = []
    
    # Create data expert agent
    data_agent = data_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name='data_expert_agent',
        handoff_tools=None
    )
    agents.append(data_agent)
    
    # Create single-cell expert agent
    singlecell_agent = singlecell_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name='singlecell_expert_agent',
        handoff_tools=None
    )
    agents.append(singlecell_agent)
    
    # Create bulk RNA-seq expert agent
    bulk_rnaseq_agent = bulk_rnaseq_expert(
        data_manager=data_manager,
        callback_handler=callback_handler,
        agent_name='bulk_rnaseq_expert_agent',
        handoff_tools=None
    )
    agents.append(bulk_rnaseq_agent)
    
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
        state_schema=OverallState,
        add_handoff_messages=False,
        include_agent_nameand='inline',
        # Change from "full_history" to "messages" or "last_message"
        output_mode="full_history",  # This ensures the actual messages are returned
        # output_mode="last_message",  # This ensures the actual messages are returned
        tools=[
            create_custom_handoff_tool(agent_name='data_expert_agent',
                                       name="handoff_to_data_expert",
                                       description="Assign data fetching/download tasks to the data expert"),
            create_custom_handoff_tool(agent_name='singlecell_expert_agent',
                                       name="handoff_to_singlecell_expert",
                                       description="Assign single-cell RNA-seq analysis tasks to the single-cell expert"),
            create_custom_handoff_tool(agent_name='bulk_rnaseq_expert_agent',
                                       name="handoff_to_bulk_rnaseq_expert",
                                       description="Assign bulk RNA-seq analysis tasks to the bulk RNA-seq expert"),
            create_custom_handoff_tool(agent_name='method_expert_agent',
                                       name="handoff_to_method_expert",
                                       description="Assign literature/method tasks to the method expert")
            # forwarding_tool
            ]
    )
    
    # Compile the graph with the provided checkpointer
    graph = workflow.compile(
        checkpointer=checkpointer
        )
    
    logger.info("Bioinformatics multi-agent graph created successfully")
    return graph
