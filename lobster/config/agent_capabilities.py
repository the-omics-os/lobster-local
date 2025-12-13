"""
Agent Capability Extraction System for the Lobster platform.

This module provides automatic discovery of agent capabilities by introspecting
agent modules and extracting @tool decorated functions, enabling dynamic
supervisor prompt generation without manual updates.
"""

import ast
import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, types

# Import registry functions at module level for easier mocking in tests
from lobster.config.agent_registry import get_agent_registry_config, get_all_agent_names
from lobster.utils.logger import get_logger

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
            return {
                "description": "No description available",
                "params": {},
                "returns": "Unknown",
            }

        lines = docstring.strip().split("\n")
        description = []
        params = {}
        returns = "Unknown"
        current_section = "description"

        for line in lines:
            line = line.strip()

            # Check for section markers
            if line.lower().startswith(("args:", "arguments:", "parameters:")):
                current_section = "params"
                continue
            elif line.lower().startswith(("returns:", "return:")):
                current_section = "returns"
                continue
            elif line.lower().startswith(
                ("raises:", "note:", "notes:", "example:", "examples:")
            ):
                # Stop parsing at other sections
                break

            # Process content based on current section
            if current_section == "description" and line:
                description.append(line)
            elif current_section == "params" and line:
                # Parse parameter lines (format: "param_name: description" or "- param_name: description")
                if ":" in line:
                    line = line.lstrip("- ").strip()
                    param_name, param_desc = line.split(":", 1)
                    param_name = param_name.strip()
                    # Remove type hints from param name if present
                    if " " in param_name:
                        param_name = param_name.split()[0]
                    params[param_name] = param_desc.strip()
            elif current_section == "returns" and line:
                if returns == "Unknown":
                    returns = line

        return {
            "description": (
                " ".join(description) if description else "No description available"
            ),
            "params": params,
            "returns": returns,
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
            if hasattr(func, "__wrapped__"):
                original_func = func.__wrapped__
            else:
                original_func = func

            sig = inspect.signature(original_func)
            params = {}

            for param_name, param in sig.parameters.items():
                if param_name == "self":
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
        # 1️⃣ Plain wrapped function (langchain <0.1)
        if isinstance(obj, types.FunctionType) and getattr(obj, "_tool_name", None):
            return True

        # 2️⃣ StructuredTool / Tool instance (langchain >=0.1)
        if hasattr(obj, "name") and hasattr(obj, "func"):
            return True

        return False

    @staticmethod
    def _extract_tools_from_factory(factory_func: Callable) -> List[Callable]:
        """
        Return a list of *callable* objects that are decorated with @tool
        inside *factory_func*.
        """
        source = inspect.getsource(factory_func)
        tree = ast.parse(source)

        tools: List[Callable] = []

        class ToolVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                # look for @tool
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Name) and deco.id == "tool":
                        # compile the function definition
                        func_code = compile(
                            ast.Module([node], type_ignores=[]),
                            filename="<ast>",
                            mode="exec",
                        )
                        # exec in the factory's globals so the decorator is known
                        namespace = {}
                        exec(func_code, factory_func.__globals__, namespace)
                        tools.append(namespace[node.name])
                # keep walking
                self.generic_visit(node)

        ToolVisitor().visit(tree)
        return tools

    @classmethod
    def extract_capabilities(cls, agent_name: str) -> AgentCapabilities:
        """Extract all @tool‑decorated functions from an agent."""
        config = get_agent_registry_config(agent_name)
        if not config:
            return AgentCapabilities(
                agent_name=agent_name,
                display_name=agent_name,
                description="Unknown agent",
                tools=[],
                error=f"Agent '{agent_name}' not found in registry",
            )

        capabilities: List[AgentCapability] = []
        error: Optional[str] = None

        try:
            # 1️⃣  Load the module and factory
            module_path, factory_name = config.factory_function.rsplit(".", 1)
            module = importlib.import_module(module_path)
            factory_func = getattr(module, factory_name)

            inner_tools = cls._extract_tools_from_factory(factory_func)

            for tool_obj in inner_tools:
                if not cls._is_tool_function(tool_obj):
                    continue

                # ── StructuredTool path ──────────────────────────────────────
                if hasattr(tool_obj, "name"):
                    tool_name = tool_obj.name
                    description = getattr(tool_obj, "description", "")
                    func = tool_obj.func
                else:  # fallback to plain function
                    tool_name = getattr(tool_obj, "_tool_name", tool_obj.__name__)
                    description = getattr(tool_obj, "__doc__", "") or ""
                    func = tool_obj

                doc_info = cls._parse_docstring(func.__doc__ or "")
                params = cls._extract_parameters(func)

                # Merge param descriptions from docstring
                for p_name, p_desc in doc_info.get("params", {}).items():
                    if p_name in params:
                        params[p_name] = p_desc

                capability = AgentCapability(
                    tool_name=tool_name,
                    description=doc_info.get("description", ""),
                    parameters=params,
                    return_type=doc_info.get("returns", ""),
                )
                capabilities.append(capability)
                logger.debug(f"Found tool '{tool_name}' in {agent_name}")

            # 3️⃣  Optional fallback to module‑level tools
            if not capabilities:
                for name, obj in inspect.getmembers(module):
                    if cls._is_tool_function(obj):
                        doc_info = cls._parse_docstring(obj.__doc__ or "")
                        params = cls._extract_parameters(obj)
                        for p_name, p_desc in doc_info.get("params", {}).items():
                            if p_name in params:
                                params[p_name] = p_desc

                        capability = AgentCapability(
                            tool_name=name,
                            description=doc_info.get("description", ""),
                            parameters=params,
                            return_type=doc_info.get("returns", ""),
                        )
                        capabilities.append(capability)

        except Exception as e:
            error = f"Error extracting capabilities: {e}"
            logger.warning(f"Could not extract capabilities for {agent_name}: {e}")

        return AgentCapabilities(
            agent_name=agent_name,
            display_name=config.display_name,
            description=config.description,
            tools=capabilities,
            error=error,
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
            logger.info(
                f"Extracted {len(capabilities[agent_name].tools)} tools for {agent_name}"
            )

        return capabilities

    @classmethod
    def get_agent_capability_summary(cls, agent_name: str, max_tools: int = 5) -> str:
        """Get a compressed summary of an agent's capabilities.

        Optimized for token efficiency: lists tool names without verbose descriptions.
        Tool descriptions are available in agent-specific prompts.

        Args:
            agent_name: Name of the agent
            max_tools: Maximum number of tools to include (kept for API compatibility)

        Returns:
            Compressed string with agent description and tool names only
        """
        caps = cls.extract_capabilities(agent_name)

        if caps.error:
            return f"- **{caps.display_name}** ({caps.agent_name}): {caps.description}"

        # Compressed format: agent name, description, and tool names only
        summary = f"- **{caps.display_name}** ({caps.agent_name}): {caps.description}"

        if caps.tools:
            # List tool names only (comma-separated) - descriptions in agent prompts
            tool_names = [tool.tool_name for tool in caps.tools]
            summary += f"\n  Tools: {', '.join(tool_names)}"

        return summary

    @classmethod
    def clear_cache(cls):
        """Clear the capability cache.

        Useful when agents are modified during development.
        """
        cls.get_all_agent_capabilities.cache_clear()
        logger.info("Cleared agent capability cache")
