"""
Expert Handoff Manager for centralized handoff coordination between agents.

This module provides the ExpertHandoffManager class for managing expert-to-expert
handoffs with context preservation and automatic return flow management.
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langgraph.types import Command

logger = logging.getLogger(__name__)


@dataclass
class HandoffContext:
    """Context data for expert handoffs."""

    handoff_id: str
    from_expert: str
    to_expert: str
    task_type: str
    parameters: Dict[str, Any]
    return_expectations: Dict[str, Any]
    timestamp: str
    chain_position: int = 0
    parent_handoff_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HandoffContext":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class HandoffResult:
    """Result from a completed handoff operation."""

    handoff_id: str
    success: bool
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    completion_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ExpertHandoffManager:
    """
    Centralized manager for expert-to-expert handoffs with context preservation
    and automatic return flow management.
    """

    def __init__(self):
        self.handoff_registry: Dict[str, BaseTool] = {}
        self.active_handoffs: Dict[str, HandoffContext] = {}
        self.handoff_history: List[Dict[str, Any]] = []
        self.handoff_chains: Dict[str, List[str]] = {}
        self._max_chain_depth = 10  # Prevent infinite recursion

    def register_handoff(
        self, from_expert: str, to_expert: str, handoff_tool: BaseTool
    ) -> None:
        """
        Register a handoff tool between two experts.

        Args:
            from_expert: Source expert agent name
            to_expert: Target expert agent name
            handoff_tool: The handoff tool to register
        """
        key = f"{from_expert}_to_{to_expert}"
        self.handoff_registry[key] = handoff_tool
        logger.info(f"Registered handoff: {from_expert} -> {to_expert}")

    def create_context_preserving_handoff(
        self, to_expert: str, context: HandoffContext, return_to_sender: bool = True
    ) -> Command:
        """
        Create handoff command that preserves context across agents.

        Args:
            to_expert: Target expert agent name
            context: Handoff context data
            return_to_sender: Whether to return control to sending expert after completion

        Returns:
            Command object for LangGraph execution
        """
        # Store active handoff
        self.active_handoffs[context.handoff_id] = context

        # Track handoff chain to prevent loops
        chain_key = context.handoff_id
        if chain_key not in self.handoff_chains:
            self.handoff_chains[chain_key] = []

        self.handoff_chains[chain_key].append(to_expert)

        # Check for chain depth limit
        if len(self.handoff_chains[chain_key]) > self._max_chain_depth:
            logger.error(f"Handoff chain depth exceeded for {context.handoff_id}")
            raise ValueError(
                f"Maximum handoff chain depth ({self._max_chain_depth}) exceeded"
            )

        # Create detailed task description for target expert
        task_description = self._format_task_description(context)

        # Log handoff initiation
        self._log_handoff_event("initiated", context)

        return Command(
            goto=to_expert,
            update={
                "handoff_context": context.to_dict(),
                "task_description": task_description,
                "return_to_sender": return_to_sender,
                "handoff_id": context.handoff_id,
            },
        )

    def track_handoff_chain(self, handoff_id: str, chain: List[str]) -> None:
        """
        Track multi-agent handoff chains for proper return flow.

        Args:
            handoff_id: Unique identifier for the handoff
            chain: List of agent names in the handoff chain
        """
        self.handoff_chains[handoff_id] = chain.copy()
        logger.debug(f"Tracking handoff chain {handoff_id}: {' -> '.join(chain)}")

    def get_return_path(self, current_agent: str, handoff_id: str) -> Optional[str]:
        """
        Determine where to return after completing handoff task.

        Args:
            current_agent: Current agent name
            handoff_id: Handoff identifier

        Returns:
            Agent name to return to, or None if returning to supervisor
        """
        if handoff_id not in self.active_handoffs:
            logger.warning(f"No active handoff found for ID: {handoff_id}")
            return None

        context = self.active_handoffs[handoff_id]

        # For simple handoffs, return to sender
        if context.chain_position == 0:
            return context.from_expert

        # For chained handoffs, check the chain
        if handoff_id in self.handoff_chains:
            chain = self.handoff_chains[handoff_id]
            current_idx = None

            # Find current position in chain
            for i, agent in enumerate(chain):
                if agent == current_agent:
                    current_idx = i
                    break

            if current_idx is not None and current_idx > 0:
                return chain[current_idx - 1]

        # Default: return to original sender
        return context.from_expert

    def complete_handoff(
        self, handoff_id: str, result: HandoffResult, state: Dict[str, Any]
    ) -> Command:
        """
        Complete a handoff operation and return control to sender.

        Args:
            handoff_id: Handoff identifier
            result: Result data from the handoff operation
            state: Current agent state

        Returns:
            Command object to return control
        """
        if handoff_id not in self.active_handoffs:
            logger.error(f"Cannot complete unknown handoff: {handoff_id}")
            raise ValueError(f"Unknown handoff ID: {handoff_id}")

        context = self.active_handoffs[handoff_id]

        # Log completion
        self._log_handoff_event("completed", context, result.to_dict())

        # Determine return destination
        return_to = self.get_return_path(context.to_expert, handoff_id)

        if return_to is None:
            # Return to supervisor
            return Command(
                goto="__end__",
                update={
                    **state,
                    "handoff_result": result.to_dict(),
                    "handoff_completed": True,
                },
            )
        else:
            # Return to specific agent
            return Command(
                goto=return_to,
                update={
                    **state,
                    "handoff_result": result.to_dict(),
                    "handoff_completed": True,
                    "returned_from": context.to_expert,
                },
            )

    def cleanup_handoff(self, handoff_id: str) -> None:
        """
        Clean up completed handoff data.

        Args:
            handoff_id: Handoff identifier to clean up
        """
        # Move to history before cleanup
        if handoff_id in self.active_handoffs:
            context = self.active_handoffs[handoff_id]
            self.handoff_history.append(
                {
                    "handoff_id": handoff_id,
                    "context": context.to_dict(),
                    "completed_at": datetime.now().isoformat(),
                    "status": "completed",
                }
            )

            # Clean up active data
            del self.active_handoffs[handoff_id]

        if handoff_id in self.handoff_chains:
            del self.handoff_chains[handoff_id]

        logger.debug(f"Cleaned up handoff: {handoff_id}")

    def get_active_handoffs(self) -> Dict[str, HandoffContext]:
        """Get all currently active handoffs."""
        return self.active_handoffs.copy()

    def get_handoff_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get handoff history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of handoff history entries
        """
        history = self.handoff_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def _format_task_description(self, context: HandoffContext) -> str:
        """
        Format a detailed task description for the target expert.

        Args:
            context: Handoff context

        Returns:
            Formatted task description
        """
        description = f"Task: {context.task_type}\n"
        description += f"From: {context.from_expert}\n"
        description += "Parameters:\n"

        for key, value in context.parameters.items():
            description += f"  - {key}: {value}\n"

        if context.return_expectations:
            description += "Expected Results:\n"
            for key, value in context.return_expectations.items():
                description += f"  - {key}: {value}\n"

        return description

    def _log_handoff_event(
        self,
        event_type: str,
        context: HandoffContext,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log handoff events for debugging and analytics.

        Args:
            event_type: Type of event (initiated, completed, failed)
            context: Handoff context
            additional_data: Additional event data
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "handoff_id": context.handoff_id,
            "from_expert": context.from_expert,
            "to_expert": context.to_expert,
            "task_type": context.task_type,
        }

        if additional_data:
            log_entry["additional_data"] = additional_data

        self.handoff_history.append(log_entry)

        logger.info(
            f"Handoff event: {event_type} for {context.handoff_id} "
            f"({context.from_expert} -> {context.to_expert})"
        )


# Global instance for use across the application
expert_handoff_manager = ExpertHandoffManager()


def create_handoff_context(
    from_expert: str,
    to_expert: str,
    task_type: str,
    parameters: Dict[str, Any],
    return_expectations: Optional[Dict[str, Any]] = None,
) -> HandoffContext:
    """
    Helper function to create handoff context.

    Args:
        from_expert: Source expert agent name
        to_expert: Target expert agent name
        task_type: Type of task being handed off
        parameters: Task parameters
        return_expectations: Expected return data structure

    Returns:
        HandoffContext object
    """
    return HandoffContext(
        handoff_id=str(uuid.uuid4()),
        from_expert=from_expert,
        to_expert=to_expert,
        task_type=task_type,
        parameters=parameters,
        return_expectations=return_expectations or {},
        timestamp=datetime.now().isoformat(),
    )


def create_handoff_result(
    handoff_id: str,
    success: bool,
    result_data: Dict[str, Any],
    error_message: Optional[str] = None,
) -> HandoffResult:
    """
    Helper function to create handoff result.

    Args:
        handoff_id: Handoff identifier
        success: Whether the handoff was successful
        result_data: Result data from the operation
        error_message: Error message if failed

    Returns:
        HandoffResult object
    """
    return HandoffResult(
        handoff_id=handoff_id,
        success=success,
        result_data=result_data,
        error_message=error_message,
        completion_time=datetime.now().isoformat(),
    )
