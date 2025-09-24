"""
Enhanced Handoff Tools for expert-to-expert collaboration.

⚠️  **CURRENTLY DISABLED FOR SUPERVISOR-MEDIATED FLOW** ⚠️

This module provides advanced handoff tools with typed context passing,
validation, and automatic return flow management for FUTURE implementation
of direct sub-agent to sub-agent handoffs.

**Current State:** All functionality is preserved but commented out to maintain
supervisor-mediated workflow. This ensures the supervisor maintains central
control over all agent interactions.

**Future Implementation:** When direct sub-agent handoffs are needed, this
module can be re-enabled by uncommenting the relevant sections and updating
agent imports.

**Architecture Decision:** Currently using User → Supervisor → Agent A → Supervisor → Agent B
instead of User → Supervisor → Agent A → Agent B (direct handoff).
"""

# COMMENTED OUT: Enhanced handoff tools for future direct sub-agent communication
# This entire module is preserved for future implementation when direct handoffs are needed.
#
# To re-enable:
# 1. Uncomment the code below
# 2. Update agent imports to include the handoff tools
# 3. Test the complete handoff workflow
# 4. Update system prompts to reflect direct handoff capabilities

# import json
# import logging
# from typing import Dict, List, Optional, Any, Type, Union, Callable
# from dataclasses import dataclass
# 
# from langchain_core.tools import tool, BaseTool
# from langchain_core.messages import AIMessage, ToolMessage
# from langgraph.prebuilt import InjectedState
# from langchain_core.tools import InjectedToolCallId
# from langgraph.types import Command
# from typing_extensions import Annotated
# 
# from .expert_handoff_manager import (
#     expert_handoff_manager,
#     create_handoff_context,
#     create_handoff_result,
#     HandoffContext,
#     HandoffResult
# )
# 
# logger = logging.getLogger(__name__)
# 
# 
# # Context schema definitions for common handoff types
# SCVI_CONTEXT_SCHEMA = {
#     "modality_name": str,
#     "n_latent": int,
#     "batch_key": Optional[str],
#     "max_epochs": int,
#     "use_gpu": bool
# }
# 
# PSEUDOBULK_CONTEXT_SCHEMA = {
#     "modality_name": str,
#     "groupby": str,
#     "layer": Optional[str],
#     "method": str,
#     "min_cells": int
# }
# 
# DATA_LOADING_SCHEMA = {
#     "file_path": str,
#     "file_type": str,
#     "delimiter": Optional[str],
#     "header": Optional[Union[int, bool]]
# }
# 
# METHOD_CONTEXT_SCHEMA = {
#     "publication_id": str,
#     "method_name": str,
#     "parameters_needed": List[str],
#     "context": str
# }
# 
# 
# def validate_context_schema(context: Dict[str, Any], schema: Dict[str, Type]) -> Dict[str, Any]:
#     """
#     Validate context data against a schema.
# 
#     Args:
#         context: Context data to validate
#         schema: Schema definition with field names and types
# 
#     Returns:
#         Validated context data
# 
#     Raises:
#         ValueError: If validation fails
#     """
#     validated = {}
#     errors = []
# 
#     for field_name, field_type in schema.items():
#         if field_name in context:
#             value = context[field_name]
# 
#             # Handle Optional types
#             if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
#                 # Check if it's Optional (Union with None)
#                 args = field_type.__args__
#                 if len(args) == 2 and type(None) in args:
#                     actual_type = args[0] if args[1] is type(None) else args[1]
#                     if value is not None and not isinstance(value, actual_type):
#                         errors.append(f"Field '{field_name}' must be {actual_type.__name__} or None, got {type(value).__name__}")
#                     validated[field_name] = value
#                 else:
#                     # Handle other Union types
#                     if not any(isinstance(value, t) for t in args):
#                         type_names = [t.__name__ for t in args]
#                         errors.append(f"Field '{field_name}' must be one of {type_names}, got {type(value).__name__}")
#                     validated[field_name] = value
#             else:
#                 # Regular type checking
#                 if not isinstance(value, field_type):
#                     errors.append(f"Field '{field_name}' must be {field_type.__name__}, got {type(value).__name__}")
#                 validated[field_name] = value
#         else:
#             # Check if field is required (not Optional)
#             if not (hasattr(field_type, '__origin__') and field_type.__origin__ is Union and type(None) in field_type.__args__):
#                 errors.append(f"Required field '{field_name}' is missing")
# 
#     if errors:
#         raise ValueError(f"Context validation failed: {'; '.join(errors)}")
# 
#     return validated
# 
# 
# def create_expert_handoff_tool(
#     from_expert: str,
#     to_expert: str,
#     task_type: str,
#     context_schema: Optional[Dict[str, Type]] = None,
#     return_to_sender: bool = True,
#     custom_description: Optional[str] = None,
#     progress_tracking: bool = False
# ) -> BaseTool:
#     """
#     Create enhanced handoff tool for expert-to-expert collaboration.
# 
#     Args:
#         from_expert: Source expert agent name
#         to_expert: Target expert agent name
#         task_type: Type of task being handed off (e.g., 'scvi_training', 'pseudobulk_analysis')
#         context_schema: Schema for context data validation
#         return_to_sender: Whether to return control to sending expert after completion
#         custom_description: Custom description for the handoff tool
#         progress_tracking: Whether to integrate with progress tracking
# 
#     Returns:
#         BaseTool: Enhanced handoff tool with validation and context preservation
#     """
# 
#     # Generate tool name and description
#     tool_name = f"handoff_{from_expert}_to_{to_expert}_{task_type}"
# 
#     if custom_description:
#         description = custom_description
#     else:
#         description = f"Hand off {task_type} task from {from_expert} to {to_expert} with context preservation"
# 
#     @tool(tool_name, description=description)
#     def expert_handoff_tool(
#         task_description: Annotated[str, "Detailed description of what the target expert should do"],
#         context: Annotated[Dict[str, Any], "Context data for the handoff task"],
#         state: Annotated[Dict[str, Any], InjectedState],
#         tool_call_id: Annotated[str, InjectedToolCallId],
#     ) -> Command:
#         """Enhanced expert handoff tool with validation and context preservation."""
# 
#         try:
#             # Validate context if schema provided
#             if context_schema:
#                 validated_context = validate_context_schema(context, context_schema)
#             else:
#                 validated_context = context.copy()
# 
#             # Create handoff context
#             handoff_context = create_handoff_context(
#                 from_expert=from_expert,
#                 to_expert=to_expert,
#                 task_type=task_type,
#                 parameters=validated_context,
#                 return_expectations=context.get("return_expectations", {
#                     "success": bool,
#                     "result_data": Dict[str, Any],
#                     "error_message": Optional[str]
#                 })
#             )
# 
#             # Add task description to context
#             handoff_context.parameters["task_description"] = task_description
# 
#             # Log handoff initiation
#             logger.info(f"Initiating expert handoff: {from_expert} -> {to_expert} "
#                        f"(task: {task_type}, id: {handoff_context.handoff_id})")
# 
#             # Create progress tracking if enabled
#             if progress_tracking:
#                 # TODO: Integrate with progress manager when background tasks are implemented
#                 logger.debug(f"Progress tracking enabled for handoff {handoff_context.handoff_id}")
# 
#             # Use expert handoff manager to create the handoff command
#             handoff_command = expert_handoff_manager.create_context_preserving_handoff(
#                 to_expert=to_expert,
#                 context=handoff_context,
#                 return_to_sender=return_to_sender
#             )
# 
#             # Create ToolMessage for LangGraph compliance
#             tool_message = ToolMessage(
#                 content=f"Successfully initiated handoff to {to_expert} for {task_type}",
#                 name=tool_name,
#                 tool_call_id=tool_call_id
#             )
# 
#             # Add ToolMessage to the Command update
#             updated_messages = state.get("messages", []) + [tool_message]
#             handoff_command.update["messages"] = updated_messages
# 
#             return handoff_command
# 
#         except ValueError as e:
#             logger.error(f"Handoff validation failed: {e}")
# 
#             # Return error message instead of proceeding with invalid handoff
#             error_response = f"❌ Handoff validation failed: {str(e)}"
# 
#             # Create ToolMessage for LangGraph compliance
#             tool_message = ToolMessage(
#                 content=error_response,
#                 name=tool_name,
#                 tool_call_id=tool_call_id
#             )
# 
#             return Command(
#                 goto="__end__",
#                 update={
#                     "messages": state["messages"] + [
#                         tool_message,
#                         AIMessage(content=error_response)
#                     ],
#                     "handoff_error": str(e)
#                 }
#             )
# 
#         except Exception as e:
#             logger.error(f"Unexpected error in expert handoff: {e}")
# 
#             error_response = f"❌ Handoff failed due to unexpected error: {str(e)}"
# 
#             # Create ToolMessage for LangGraph compliance
#             tool_message = ToolMessage(
#                 content=error_response,
#                 name=tool_name,
#                 tool_call_id=tool_call_id
#             )
# 
#             return Command(
#                 goto="__end__",
#                 update={
#                     "messages": state["messages"] + [
#                         tool_message,
#                         AIMessage(content=error_response)
#                     ],
#                     "handoff_error": str(e)
#                 }
#             )
# 
#     # Register the handoff tool with the manager
#     expert_handoff_manager.register_handoff(from_expert, to_expert, expert_handoff_tool)
# 
#     return expert_handoff_tool
# 
# 
# def create_handoff_completion_tool(expert_name: str) -> BaseTool:
#     """
#     Create a tool for experts to complete handoff operations and return results.
# 
#     Args:
#         expert_name: Name of the expert that will use this completion tool
# 
#     Returns:
#         BaseTool: Handoff completion tool
#     """
# 
#     tool_name = f"complete_handoff_{expert_name}"
#     description = f"Complete handoff operation and return results to the requesting expert"
# 
#     @tool(tool_name, description=description)
#     def handoff_completion_tool(
#         success: Annotated[bool, "Whether the handoff task was completed successfully"],
#         result_data: Annotated[Dict[str, Any], "Results from the handoff operation"],
#         error_message: Annotated[Optional[str], "Error message if the operation failed"] = None,
#         state: Annotated[Dict[str, Any], InjectedState] = None,
#         tool_call_id: Annotated[str, InjectedToolCallId] = None,
#     ) -> Command:
#         """Tool for completing handoff operations."""
# 
#         try:
#             # Get handoff context from state
#             handoff_context_data = state.get("handoff_context")
#             if not handoff_context_data:
#                 logger.error("No handoff context found in state")
# 
#                 # Create ToolMessage for LangGraph compliance
#                 error_msg = "❌ Error: No handoff context found"
#                 tool_message = ToolMessage(
#                     content=error_msg,
#                     name=tool_name,
#                     tool_call_id=tool_call_id
#                 )
# 
#                 return Command(
#                     goto="__end__",
#                     update={
#                         "messages": state["messages"] + [
#                             tool_message,
#                             AIMessage(content=error_msg)
#                         ]
#                     }
#                 )
# 
#             handoff_id = handoff_context_data.get("handoff_id")
#             if not handoff_id:
#                 logger.error("No handoff ID found in context")
# 
#                 # Create ToolMessage for LangGraph compliance
#                 error_msg = "❌ Error: No handoff ID found"
#                 tool_message = ToolMessage(
#                     content=error_msg,
#                     name=tool_name,
#                     tool_call_id=tool_call_id
#                 )
# 
#                 return Command(
#                     goto="__end__",
#                     update={
#                         "messages": state["messages"] + [
#                             tool_message,
#                             AIMessage(content=error_msg)
#                         ]
#                     }
#                 )
# 
#             # Create handoff result
#             result = create_handoff_result(
#                 handoff_id=handoff_id,
#                 success=success,
#                 result_data=result_data,
#                 error_message=error_message
#             )
# 
#             # Complete handoff using manager
#             completion_command = expert_handoff_manager.complete_handoff(
#                 handoff_id=handoff_id,
#                 result=result,
#                 state=state
#             )
# 
#             # Create ToolMessage for LangGraph compliance
#             success_msg = f"✅ Handoff {handoff_id} completed successfully" if success else f"❌ Handoff {handoff_id} completed with errors"
#             tool_message = ToolMessage(
#                 content=success_msg,
#                 name=tool_name,
#                 tool_call_id=tool_call_id
#             )
# 
#             # Add ToolMessage to the completion command
#             if "messages" not in completion_command.update:
#                 completion_command.update["messages"] = state.get("messages", [])
# 
#             # Insert ToolMessage at the beginning of new messages
#             existing_messages = completion_command.update["messages"]
#             completion_command.update["messages"] = existing_messages + [tool_message]
# 
#             # Clean up handoff data
#             expert_handoff_manager.cleanup_handoff(handoff_id)
# 
#             logger.info(f"Completed handoff {handoff_id} with success: {success}")
# 
#             return completion_command
# 
#         except Exception as e:
#             logger.error(f"Error completing handoff: {e}")
# 
#             # Create ToolMessage for LangGraph compliance
#             error_msg = f"❌ Error completing handoff: {str(e)}"
#             tool_message = ToolMessage(
#                 content=error_msg,
#                 name=tool_name,
#                 tool_call_id=tool_call_id
#             )
# 
#             return Command(
#                 goto="__end__",
#                 update={
#                     "messages": state["messages"] + [
#                         tool_message,
#                         AIMessage(content=error_msg)
#                     ],
#                     "handoff_completion_error": str(e)
#                 }
#             )
# 
#     return handoff_completion_tool
# 
# 
# # Pre-defined handoff tools for common expert collaborations
# def create_singlecell_to_ml_handoff_tool() -> BaseTool:
#     """Create handoff tool for Single Cell Expert -> ML Expert (scVI training)."""
#     return create_expert_handoff_tool(
#         from_expert="singlecell_expert",
#         to_expert="machine_learning_expert",
#         task_type="scvi_training",
#         context_schema=SCVI_CONTEXT_SCHEMA,
#         return_to_sender=True,
#         custom_description="Hand off scVI embedding training task to Machine Learning Expert"
#     )
# 
# 
# def create_singlecell_to_bulk_handoff_tool() -> BaseTool:
#     """Create handoff tool for Single Cell Expert -> Bulk RNA-seq Expert (pseudobulk analysis)."""
#     return create_expert_handoff_tool(
#         from_expert="singlecell_expert",
#         to_expert="bulk_rnaseq_expert",
#         task_type="pseudobulk_analysis",
#         context_schema=PSEUDOBULK_CONTEXT_SCHEMA,
#         return_to_sender=True,
#         custom_description="Hand off pseudobulk analysis task to Bulk RNA-seq Expert"
#     )
# 
# 
# def create_data_to_research_handoff_tool() -> BaseTool:
#     """Create handoff tool for Data Expert -> Research Agent (dataset search)."""
#     return create_expert_handoff_tool(
#         from_expert="data_expert",
#         to_expert="research_agent",
#         task_type="dataset_search",
#         context_schema=DATA_LOADING_SCHEMA,
#         return_to_sender=True,
#         custom_description="Hand off dataset search and metadata extraction to Research Agent"
#     )
# 
# 
# def create_research_to_method_handoff_tool() -> BaseTool:
#     """Create handoff tool for Research Agent -> Method Expert (parameter extraction)."""
#     return create_expert_handoff_tool(
#         from_expert="research_agent",
#         to_expert="method_expert",
#         task_type="parameter_extraction",
#         context_schema=METHOD_CONTEXT_SCHEMA,
#         return_to_sender=True,
#         custom_description="Hand off parameter extraction from publications to Method Expert"
#     )
# 
# 
# # Registry of pre-defined handoff tools
# PREDEFINED_HANDOFF_TOOLS = {
#     "singlecell_to_ml_scvi": create_singlecell_to_ml_handoff_tool,
#     "singlecell_to_bulk_pseudobulk": create_singlecell_to_bulk_handoff_tool,
#     "data_to_research_search": create_data_to_research_handoff_tool,
#     "research_to_method_params": create_research_to_method_handoff_tool,
# }
# 
# 
# def get_predefined_handoff_tool(handoff_type: str) -> Optional[BaseTool]:
#     """
#     Get a pre-defined handoff tool by type.
# 
#     Args:
#         handoff_type: Type of handoff tool to retrieve
# 
#     Returns:
#         BaseTool or None if not found
#     """
#     if handoff_type in PREDEFINED_HANDOFF_TOOLS:
#         return PREDEFINED_HANDOFF_TOOLS[handoff_type]()
#     return None
# 
# 
# def list_available_handoff_types() -> List[str]:
#     """List all available pre-defined handoff types."""
#     return list(PREDEFINED_HANDOFF_TOOLS.keys())


# FUTURE IMPLEMENTATION STUBS - For maintaining interface compatibility
# These functions are kept as stubs so existing imports don't break

def create_expert_handoff_tool(*args, **kwargs):
    """STUB: Enhanced handoff tool creation (disabled for supervisor-mediated flow)."""
    raise NotImplementedError("Direct sub-agent handoffs are currently disabled. Use supervisor-mediated flow.")

def SCVI_CONTEXT_SCHEMA():
    """STUB: scVI context schema (disabled for supervisor-mediated flow)."""
    return {}

def create_singlecell_to_ml_handoff_tool():
    """STUB: SingleCell to ML handoff tool (disabled)."""
    raise NotImplementedError("Direct sub-agent handoffs are currently disabled. Use supervisor-mediated flow.")
