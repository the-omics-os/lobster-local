"""
Unit tests for the Agent Capability Extraction System.
"""

import pytest
from unittest.mock import patch, MagicMock
from lobster.config.agent_capabilities import (
    AgentCapability,
    AgentCapabilities,
    AgentCapabilityExtractor
)


class TestAgentCapability:
    """Test the AgentCapability dataclass."""

    def test_agent_capability_creation(self):
        """Test creating an AgentCapability instance."""
        capability = AgentCapability(
            tool_name="test_tool",
            description="Test tool description",
            parameters={"param1": "string", "param2": "int"},
            return_type="str"
        )

        assert capability.tool_name == "test_tool"
        assert capability.description == "Test tool description"
        assert "param1" in capability.parameters
        assert capability.return_type == "str"


class TestAgentCapabilities:
    """Test the AgentCapabilities dataclass."""

    def test_agent_capabilities_creation(self):
        """Test creating an AgentCapabilities instance."""
        tool = AgentCapability(
            tool_name="test_tool",
            description="Test tool",
            parameters={},
            return_type="str"
        )

        capabilities = AgentCapabilities(
            agent_name="test_agent",
            display_name="Test Agent",
            description="Test agent description",
            tools=[tool]
        )

        assert capabilities.agent_name == "test_agent"
        assert capabilities.display_name == "Test Agent"
        assert len(capabilities.tools) == 1
        assert capabilities.error is None

    def test_agent_capabilities_with_error(self):
        """Test AgentCapabilities with an error."""
        capabilities = AgentCapabilities(
            agent_name="test_agent",
            display_name="Test Agent",
            description="Test agent description",
            tools=[],
            error="Failed to load agent module"
        )

        assert capabilities.error == "Failed to load agent module"
        assert len(capabilities.tools) == 0


class TestAgentCapabilityExtractor:
    """Test the AgentCapabilityExtractor class."""

    def test_parse_docstring_simple(self):
        """Test parsing a simple docstring."""
        docstring = """
        This is a test function.

        Args:
            param1: First parameter
            param2: Second parameter

        Returns:
            str: The result
        """

        result = AgentCapabilityExtractor._parse_docstring(docstring)

        assert "This is a test function" in result["description"]
        assert "param1" in result["params"]
        assert "param2" in result["params"]
        assert "The result" in result["returns"]

    def test_parse_docstring_no_docstring(self):
        """Test parsing with no docstring."""
        result = AgentCapabilityExtractor._parse_docstring(None)

        assert result["description"] == "No description available"
        assert result["params"] == {}
        assert result["returns"] == "Unknown"

    def test_parse_docstring_alternative_format(self):
        """Test parsing docstring with alternative format."""
        docstring = """
        Test function description.

        Parameters:
            - input_data: The input data
            - config: Configuration object

        Return:
            dict: Processing results
        """

        result = AgentCapabilityExtractor._parse_docstring(docstring)

        assert "Test function description" in result["description"]
        assert "input_data" in result["params"]
        assert "config" in result["params"]
        assert "Processing results" in result["returns"]

    def test_is_tool_function(self):
        """Test identifying tool functions."""
        # Mock a tool-decorated function
        mock_tool = MagicMock()
        mock_tool.__wrapped__ = lambda: None
        mock_tool.__name__ = "test_tool"
        mock_tool.name = "test_tool"
        mock_tool.__doc__ = "Test tool"

        assert AgentCapabilityExtractor._is_tool_function(mock_tool) is True

        # Test non-tool function
        regular_func = lambda x: x
        assert AgentCapabilityExtractor._is_tool_function(regular_func) is False

    @patch('lobster.config.agent_capabilities.get_agent_registry_config')
    def test_extract_capabilities_agent_not_found(self, mock_get_config):
        """Test extracting capabilities for non-existent agent."""
        mock_get_config.return_value = None

        result = AgentCapabilityExtractor.extract_capabilities("non_existent_agent")

        assert result.agent_name == "non_existent_agent"
        assert result.display_name == "non_existent_agent"
        assert result.description == "Unknown agent"
        assert len(result.tools) == 0
        assert "not found in registry" in result.error

    @patch('lobster.config.agent_capabilities.importlib.import_module')
    @patch('lobster.config.agent_capabilities.get_agent_registry_config')
    def test_extract_capabilities_import_error(self, mock_get_config, mock_import):
        """Test handling import errors during capability extraction."""
        mock_config = MagicMock()
        mock_config.factory_function = "module.factory"
        mock_config.display_name = "Test Agent"
        mock_config.description = "Test description"
        mock_get_config.return_value = mock_config

        mock_import.side_effect = ImportError("Module not found")

        result = AgentCapabilityExtractor.extract_capabilities("test_agent")

        assert result.agent_name == "test_agent"
        assert result.display_name == "Test Agent"
        assert "Error extracting capabilities" in result.error

    def test_get_agent_capability_summary(self):
        """Test getting agent capability summary."""
        with patch.object(AgentCapabilityExtractor, 'extract_capabilities') as mock_extract:
            # Setup mock capabilities
            tools = [
                AgentCapability(
                    tool_name=f"tool_{i}",
                    description=f"Tool {i} description that is quite long and should be truncated" * 5,
                    parameters={},
                    return_type="str"
                )
                for i in range(7)
            ]

            mock_caps = AgentCapabilities(
                agent_name="test_agent",
                display_name="Test Agent",
                description="Test agent for testing",
                tools=tools
            )

            mock_extract.return_value = mock_caps

            # Test with default max_tools
            summary = AgentCapabilityExtractor.get_agent_capability_summary("test_agent")

            assert "Test Agent" in summary
            assert "Test agent for testing" in summary
            assert "tool_0" in summary
            assert "tool_4" in summary
            assert "...and 2 more tools" in summary
            assert "tool_6" not in summary  # Should be truncated

            # Test with custom max_tools
            summary = AgentCapabilityExtractor.get_agent_capability_summary("test_agent", max_tools=2)
            assert "tool_1" in summary
            assert "tool_2" not in summary
            assert "...and 5 more tools" in summary

    def test_get_agent_capability_summary_with_error(self):
        """Test capability summary when there's an error."""
        with patch.object(AgentCapabilityExtractor, 'extract_capabilities') as mock_extract:
            mock_caps = AgentCapabilities(
                agent_name="test_agent",
                display_name="Test Agent",
                description="Test agent description",
                tools=[],
                error="Failed to load"
            )

            mock_extract.return_value = mock_caps

            summary = AgentCapabilityExtractor.get_agent_capability_summary("test_agent")

            assert "Test Agent" in summary
            assert "Test agent description" in summary
            assert "Key capabilities" not in summary

    @patch('lobster.config.agent_capabilities.get_all_agent_names')
    @patch.object(AgentCapabilityExtractor, 'extract_capabilities')
    def test_get_all_agent_capabilities(self, mock_extract, mock_get_names):
        """Test getting capabilities for all agents."""
        mock_get_names.return_value = ["agent1", "agent2"]

        mock_caps1 = AgentCapabilities(
            agent_name="agent1",
            display_name="Agent 1",
            description="First agent",
            tools=[]
        )

        mock_caps2 = AgentCapabilities(
            agent_name="agent2",
            display_name="Agent 2",
            description="Second agent",
            tools=[]
        )

        mock_extract.side_effect = [mock_caps1, mock_caps2]

        # Clear cache first
        AgentCapabilityExtractor.clear_cache()

        result = AgentCapabilityExtractor.get_all_agent_capabilities()

        assert len(result) == 2
        assert "agent1" in result
        assert "agent2" in result
        assert result["agent1"].display_name == "Agent 1"
        assert result["agent2"].display_name == "Agent 2"

    def test_clear_cache(self):
        """Test clearing the capability cache."""
        # This should not raise any errors
        AgentCapabilityExtractor.clear_cache()

        # Verify the cache is actually cleared by checking the cache_info
        cache_info = AgentCapabilityExtractor.get_all_agent_capabilities.cache_info()
        assert cache_info.currsize == 0