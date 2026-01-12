"""State schema for Data Expert agent."""

# Import from global state.py - DataExpertState is already defined there
# and inherits from AgentState which includes all required LangGraph fields
from lobster.agents.state import DataExpertState

__all__ = ["DataExpertState"]
