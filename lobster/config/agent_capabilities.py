"""
Agent Capability Extraction System for the Lobster platform.

This module provides automatic discovery of agent capabilities by introspecting
agent modules and extracting @tool decorated functions, enabling dynamic
supervisor prompt generation without manual updates.
"""

import inspect
import importlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from functools import lru_cache
from lobster.utils.logger import get_logger

# Import registry functions at module level for easier mocking in tests
from lobster.config.agent_registry import get_agent_registry_config, get_all_agent_names

logger = get_logger(__name__)


@dataclass
class AgentCapability:
    """Represents a capability (tool) of an agent.

    Attributes:
        tool_name: Name of the tool function
        description: Tool description from docstring
        parameters: Dict of parameter names to descriptions
        return_type: Description of return type
    """
    tool_name: str
    description: str
    parameters: Dict[str, str]
    return_type: str


@dataclass
class AgentCapabilities:
    """Container for all capabilities of an agent.

    Attributes:
        agent_name: Name of the agent
        display_name: Human-readable name
        description: Agent description from registry
        tools: List of agent capabilities
        error: Error message if capability extraction failed
    """
    agent_name: str
    display_name: str
    description: str
    tools: List[AgentCapability]
    error: Optional[str] = None


class AgentCapabilityExtractor:
    """Extracts capabilities from agent modules."""

    @staticmethod
    def _parse_docstring(docstring: str) -> Dict[str, Any]:
        """Parse a docstring to extract description and parameter info.

        Args:
            docstring: The docstring to parse

        Returns:
            Dict containing parsed information
        """
        if not docstring:
            return {"description": "No description available", "params": {}, "returns": "Unknown"}

        lines = docstring.strip().split('\n')
        description = []
        params = {}
        returns = "Unknown"
        current_section = "description"

        for line in lines:
            line = line.strip()

            # Check for section markers
            if line.lower().startswith(('args:', 'arguments:', 'parameters:')):
                current_section = "params"
                continue
            elif line.lower().startswith(('returns:', 'return:')):
                current_section = "returns"
                continue
            elif line.lower().startswith(('raises:', 'note:', 'notes:', 'example:', 'examples:')):
                # Stop parsing at other sections
                break

            # Process content based on current section
            if current_section == "description" and line:
                description.append(line)
            elif current_section == "params" and line:
                # Parse parameter lines (format: "param_name: description" or "- param_name: description")
                if ':' in line:
                    line = line.lstrip('- ').strip()
                    param_name, param_desc = line.split(':', 1)
                    param_name = param_name.strip()
                    # Remove type hints from param name if present
                    if ' ' in param_name:
                        param_name = param_name.split()[0]
                    params[param_name] = param_desc.strip()
            elif current_section == "returns" and line:
                if returns == "Unknown":
                    returns = line

        return {
            "description": ' '.join(description) if description else "No description available",
            "params": params,
            "returns": returns
        }

    @staticmethod
    def _extract_parameters(func: Callable) -> Dict[str, str]:
        """Extract parameter information from a function.

        Args:
            func: The function to analyze

        Returns:
            Dict mapping parameter names to descriptions
        """
        try:
            # Get the original function if it's wrapped
            if hasattr(func, '__wrapped__'):
                original_func = func.__wrapped__
            else:
                original_func = func

            sig = inspect.signature(original_func)
            params = {}

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                # Build parameter description
                param_desc = "No description"
                if param.annotation != inspect.Parameter.empty:
                    param_desc = f"Type: {param.annotation.__name__ if hasattr(param.annotation, '__name__') else str(param.annotation)}"
                if param.default != inspect.Parameter.empty:
                    param_desc += f" (default: {param.default})"

                params[param_name] = param_desc

            return params
        except Exception as e:
            logger.debug(f"Could not extract parameters: {e}")
            return {}

    @staticmethod
    def _is_tool_function(obj: Any) -> bool:
        """Check if an object is a @tool decorated function.

        Args:
            obj: The object to check

        Returns:
            bool: True if the object is a tool function
        """
        # Check for langchain @tool decorator
        if hasattr(obj, '__wrapped__') and hasattr(obj, '__name__'):
            # Additional check for tool-specific attributes
            if hasattr(obj, 'name') or hasattr(obj, 'description'):
                return True
            # Check if it's a callable with a docstring (likely a tool)
            if callable(obj) and obj.__doc__:
                return True
        return False

    @classmethod
    def extract_capabilities(cls, agent_name: str) -> AgentCapabilities:
        """Extract all @tool decorated functions from an agent.

        Args:
            agent_name: Name of the agent to analyze

        Returns:
            AgentCapabilities object containing all discovered capabilities
        """
        # Get agent config from registry
        config = get_agent_registry_config(agent_name)
        if not config:
            error_msg = f"Agent '{agent_name}' not found in registry"
            logger.warning(error_msg)
            return AgentCapabilities(
                agent_name=agent_name,
                display_name=agent_name,
                description="Unknown agent",
                tools=[],
                error=error_msg
            )

        capabilities = []
        error = None

        try:
            # Extract module path from factory function
            module_path = config.factory_function.rsplit('.', 1)[0]

            # Import the agent module
            module = importlib.import_module(module_path)

            # Find the factory function
            factory_name = config.factory_function.rsplit('.', 1)[1]
            if not hasattr(module, factory_name):
                raise ImportError(f"Factory function '{factory_name}' not found in module")

            # Get the factory function
            factory_func = getattr(module, factory_name)

            # Look for tool functions defined within the factory function's module
            # This requires parsing the source code of the factory function
            # For now, we'll look for tool functions at module level
            for name, obj in inspect.getmembers(module):
                if cls._is_tool_function(obj):
                    # Parse docstring
                    doc_info = cls._parse_docstring(obj.__doc__ or "")

                    # Extract parameters
                    params = cls._extract_parameters(obj)

                    # Merge parameter descriptions from docstring if available
                    for param_name, param_desc in doc_info.get("params", {}).items():
                        if param_name in params:
                            params[param_name] = param_desc

                    capability = AgentCapability(
                        tool_name=name,
                        description=doc_info["description"],
                        parameters=params,
                        return_type=doc_info["returns"]
                    )
                    capabilities.append(capability)
                    logger.debug(f"Found tool '{name}' in {agent_name}")

            # If no module-level tools found, try to analyze the factory function itself
            # This is more complex as tools are often defined inside the factory
            if not capabilities:
                logger.debug(f"No module-level tools found for {agent_name}, checking factory function")
                # This would require more sophisticated analysis
                # For now, we'll return empty capabilities

        except Exception as e:
            error = f"Error extracting capabilities: {str(e)}"
            logger.warning(f"Could not extract capabilities for {agent_name}: {e}")

        return AgentCapabilities(
            agent_name=agent_name,
            display_name=config.display_name,
            description=config.description,
            tools=capabilities,
            error=error
        )

    @classmethod
    @lru_cache(maxsize=32)
    def get_all_agent_capabilities(cls) -> Dict[str, AgentCapabilities]:
        """Get capabilities for all registered agents with caching.

        Returns:
            Dict mapping agent names to their capabilities
        """
        capabilities = {}
        for agent_name in get_all_agent_names():
            capabilities[agent_name] = cls.extract_capabilities(agent_name)
            logger.info(f"Extracted {len(capabilities[agent_name].tools)} tools for {agent_name}")

        return capabilities

    @classmethod
    def get_agent_capability_summary(cls, agent_name: str, max_tools: int = 5) -> str:
        """Get a formatted summary of an agent's capabilities.

        Args:
            agent_name: Name of the agent
            max_tools: Maximum number of tools to include in summary

        Returns:
            Formatted string describing the agent's capabilities
        """
        caps = cls.extract_capabilities(agent_name)

        if caps.error:
            return f"**{caps.display_name}**: {caps.description}"

        summary = f"**{caps.display_name}**: {caps.description}"

        if caps.tools:
            summary += "\n  Key capabilities:"
            for tool in caps.tools[:max_tools]:
                # Truncate long descriptions
                desc = tool.description
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                summary += f"\n    • {tool.tool_name}: {desc}"

            if len(caps.tools) > max_tools:
                summary += f"\n    • ...and {len(caps.tools) - max_tools} more tools"

        return summary

    @classmethod
    def clear_cache(cls):
        """Clear the capability cache.

        Useful when agents are modified during development.
        """
        cls.get_all_agent_capabilities.cache_clear()
        logger.info("Cleared agent capability cache")