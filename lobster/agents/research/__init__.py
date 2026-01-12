"""
Research Agent Module - Literature discovery and dataset identification.

Note: This agent uses PublicationProcessingService which is a PREMIUM feature.
Graceful degradation is handled within the agent factory.
"""

from lobster.agents.research.state import ResearchAgentState

try:
    from lobster.agents.research.config import (
        ESSENTIAL_FIELDS,
        STANDARD_FIELDS,
        VERBOSE_FIELDS,
    )
    from lobster.agents.research.prompts import create_research_agent_prompt
    from lobster.agents.research.research_agent import research_agent
    RESEARCH_AGENT_AVAILABLE = True
except ImportError:
    RESEARCH_AGENT_AVAILABLE = False
    research_agent = None
    create_research_agent_prompt = None
    ESSENTIAL_FIELDS = {}
    STANDARD_FIELDS = {}
    VERBOSE_FIELDS = {}

__all__ = [
    "RESEARCH_AGENT_AVAILABLE",
    "research_agent",
    "create_research_agent_prompt",
    "ESSENTIAL_FIELDS",
    "STANDARD_FIELDS",
    "VERBOSE_FIELDS",
    "ResearchAgentState",
]
