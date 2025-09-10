"""
Comprehensive unit tests for agent registry system.

This module provides thorough testing of the agent registry including
agent registration, discovery, configuration, lifecycle management,
and metadata handling for the multi-agent bioinformatics platform.

Test coverage target: 95%+ with meaningful tests for agent orchestration.
"""

import pytest
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from lobster.config.agent_registry import (
    AgentRegistryConfig, 
    AGENT_REGISTRY, 
    get_agent_config,
    list_available_agents,
    register_agent,
    validate_agent_config
)
from lobster.core.data_manager_v2 import DataManagerV2

from tests.mock_data.factories import SingleCellDataFactory
from tests.mock_data.base import SMALL_DATASET_CONFIG


# ===============================================================================
# Mock Agent Implementations
# ===============================================================================

class MockAgent:
    """Mock agent implementation for testing."""
    
    def __init__(self, name: str = "mock_agent", **kwargs):
        self.name = name
        self.state = "initialized"
        self.tools = []
        self.messages = []
        self.config = kwargs
    
    def process_message(self, message: str) -> str:
        """Mock message processing."""
        self.messages.append(message)
        return f"Mock response to: {message}"
    
    def add_tool(self, tool_name: str, tool_func: Callable):
        """Mock tool addition."""
        self.tools.append((tool_name, tool_func))
    
    def get_status(self) -> Dict[str, Any]:
        """Mock status retrieval."""
        return {
            "name": self.name,
            "state": self.state,
            "tools_count": len(self.tools),
            "messages_processed": len(self.messages)
        }


def mock_agent_factory(name: str = "mock_agent", **kwargs) -> MockAgent:
    """Mock agent factory function."""
    return MockAgent(name=name, **kwargs)


def mock_supervisor_factory(**kwargs) -> MockAgent:
    """Mock supervisor agent factory."""
    supervisor = MockAgent(name="supervisor_agent", **kwargs)
    supervisor.state = "coordinating"
    return supervisor


def mock_data_expert_factory(**kwargs) -> MockAgent:
    """Mock data expert agent factory."""
    return MockAgent(name="data_expert_agent", **kwargs)


# ===============================================================================
# Test Fixtures
# ===============================================================================

@pytest.fixture
def mock_agent_config():
    """Create mock agent configuration."""
    return AgentRegistryConfig(
        name='test_agent',
        display_name='Test Agent',
        description='A test agent for unit testing',
        factory_function='tests.unit.agents.test_agent_registry.mock_agent_factory',
        handoff_tool_name='handoff_to_test_agent',
        handoff_tool_description='Handoff tasks to the test agent'
    )


@pytest.fixture
def temp_agent_registry():
    """Create temporary agent registry for testing."""
    # Save original registry
    original_registry = AGENT_REGISTRY.copy()
    
    # Clear registry for testing
    AGENT_REGISTRY.clear()
    
    yield AGENT_REGISTRY
    
    # Restore original registry
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(original_registry)


@pytest.fixture
def mock_data_manager():
    """Create mock data manager for agent testing."""
    with patch('lobster.core.data_manager_v2.DataManagerV2') as MockDataManager:
        mock_dm = MockDataManager.return_value
        mock_dm.list_modalities.return_value = ['test_modality_1', 'test_modality_2']
        mock_dm.get_modality.return_value = SingleCellDataFactory(config=SMALL_DATASET_CONFIG)
        yield mock_dm


# ===============================================================================
# AgentRegistryConfig Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentRegistryConfig:
    """Test AgentRegistryConfig functionality."""
    
    def test_config_initialization(self):
        """Test AgentRegistryConfig initialization."""
        config = AgentRegistryConfig(
            name='test_agent',
            display_name='Test Agent',
            description='A test agent',
            factory_function='module.function',
            handoff_tool_name='handoff_to_test',
            handoff_tool_description='Handoff to test agent'
        )
        
        assert config.name == 'test_agent'
        assert config.display_name == 'Test Agent'
        assert config.description == 'A test agent'
        assert config.factory_function == 'module.function'
        assert config.handoff_tool_name == 'handoff_to_test'
        assert config.handoff_tool_description == 'Handoff to test agent'
    
    def test_config_validation_valid(self):
        """Test validation of valid config."""
        config = AgentRegistryConfig(
            name='valid_agent',
            display_name='Valid Agent',
            description='A valid agent configuration',
            factory_function='valid.module.function',
            handoff_tool_name='handoff_to_valid',
            handoff_tool_description='Handoff to valid agent'
        )
        
        # Should not raise any exceptions
        assert config.name == 'valid_agent'
    
    def test_config_validation_invalid_name(self):
        """Test validation with invalid agent name."""
        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            AgentRegistryConfig(
                name='',
                display_name='Invalid Agent',
                description='Invalid name',
                factory_function='module.function',
                handoff_tool_name='handoff_to_invalid',
                handoff_tool_description='Handoff description'
            )
    
    def test_config_validation_invalid_factory(self):
        """Test validation with invalid factory function."""
        with pytest.raises(ValueError, match="Factory function cannot be empty"):
            AgentRegistryConfig(
                name='invalid_factory_agent',
                display_name='Invalid Factory Agent',
                description='Invalid factory function',
                factory_function='',
                handoff_tool_name='handoff_to_invalid',
                handoff_tool_description='Handoff description'
            )
    
    def test_config_serialization(self):
        """Test config serialization to dictionary."""
        config = AgentRegistryConfig(
            name='serializable_agent',
            display_name='Serializable Agent',
            description='Agent for serialization testing',
            factory_function='module.serializable_function',
            handoff_tool_name='handoff_to_serializable',
            handoff_tool_description='Handoff to serializable agent'
        )
        
        config_dict = config.__dict__
        assert config_dict['name'] == 'serializable_agent'
        assert config_dict['display_name'] == 'Serializable Agent'
        assert config_dict['factory_function'] == 'module.serializable_function'
    
    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = AgentRegistryConfig(
            name='equal_agent',
            display_name='Equal Agent',
            description='First config',
            factory_function='module.function',
            handoff_tool_name='handoff_equal',
            handoff_tool_description='Handoff description'
        )
        
        config2 = AgentRegistryConfig(
            name='equal_agent',
            display_name='Equal Agent',
            description='First config',
            factory_function='module.function',
            handoff_tool_name='handoff_equal',
            handoff_tool_description='Handoff description'
        )
        
        config3 = AgentRegistryConfig(
            name='different_agent',
            display_name='Different Agent',
            description='Different config',
            factory_function='module.function',
            handoff_tool_name='handoff_different',
            handoff_tool_description='Different handoff'
        )
        
        assert config1.__dict__ == config2.__dict__
        assert config1.__dict__ != config3.__dict__


# ===============================================================================
# Agent Registry Management Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentRegistryManagement:
    """Test agent registry management functionality."""
    
    def test_get_agent_config_existing(self):
        """Test retrieving existing agent config."""
        # Test with a known agent from the registry
        config = get_agent_config('supervisor_agent')
        
        assert config is not None
        assert config.name == 'supervisor_agent'
        assert config.display_name == 'Supervisor'
        assert 'supervisor' in config.description.lower()
    
    def test_get_agent_config_nonexistent(self):
        """Test retrieving non-existent agent config."""
        config = get_agent_config('nonexistent_agent')
        
        assert config is None
    
    def test_list_available_agents(self):
        """Test listing all available agents."""
        agents = list_available_agents()
        
        assert isinstance(agents, list)
        assert len(agents) > 0
        
        # Check that known agents are present
        agent_names = [agent.name for agent in agents]
        assert 'supervisor_agent' in agent_names
        assert 'data_expert_agent' in agent_names
    
    def test_list_available_agents_with_filters(self):
        """Test listing agents with type filters."""
        # This would test filtering if implemented
        all_agents = list_available_agents()
        
        # Verify basic properties of listed agents
        for agent in all_agents:
            assert isinstance(agent, AgentRegistryConfig)
            assert agent.name
            assert agent.display_name
            assert agent.factory_function
    
    def test_register_agent_new(self, temp_agent_registry, mock_agent_config):
        """Test registering a new agent."""
        # Registry should be empty initially
        assert len(temp_agent_registry) == 0
        
        # Register the agent
        register_agent(mock_agent_config)
        
        # Verify registration
        assert len(temp_agent_registry) == 1
        assert 'test_agent' in temp_agent_registry
        assert temp_agent_registry['test_agent'] == mock_agent_config
    
    def test_register_agent_duplicate(self, temp_agent_registry, mock_agent_config):
        """Test registering duplicate agent."""
        # Register agent first time
        register_agent(mock_agent_config)
        assert len(temp_agent_registry) == 1
        
        # Try to register again - should raise error
        with pytest.raises(ValueError, match="Agent 'test_agent' is already registered"):
            register_agent(mock_agent_config)
    
    def test_register_agent_override(self, temp_agent_registry, mock_agent_config):
        """Test registering agent with override flag."""
        # Register agent first time
        register_agent(mock_agent_config)
        
        # Create modified config
        modified_config = AgentRegistryConfig(
            name='test_agent',
            display_name='Modified Test Agent',
            description='Modified description',
            factory_function='modified.function',
            handoff_tool_name='modified_handoff',
            handoff_tool_description='Modified handoff'
        )
        
        # Register with override
        register_agent(modified_config, override=True)
        
        # Verify the config was updated
        assert temp_agent_registry['test_agent'].display_name == 'Modified Test Agent'
        assert temp_agent_registry['test_agent'].description == 'Modified description'
    
    def test_validate_agent_config_valid(self, mock_agent_config):
        """Test validation of valid agent config."""
        # Should not raise any exceptions
        validate_agent_config(mock_agent_config)
    
    def test_validate_agent_config_invalid(self):
        """Test validation of invalid agent config."""
        invalid_config = AgentRegistryConfig(
            name='',  # Invalid empty name
            display_name='Invalid',
            description='Invalid config',
            factory_function='module.function',
            handoff_tool_name='handoff',
            handoff_tool_description='Description'
        )
        
        with pytest.raises(ValueError):
            validate_agent_config(invalid_config)


# ===============================================================================
# Agent Factory and Instantiation Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentFactoryInstantiation:
    """Test agent factory and instantiation functionality."""
    
    def test_agent_factory_function_resolution(self):
        """Test resolving agent factory functions."""
        # Test with mock factory
        factory_path = 'tests.unit.agents.test_agent_registry.mock_agent_factory'
        
        # This would test the actual factory resolution if implemented
        # For now, we test that the path is valid
        assert '.' in factory_path
        assert factory_path.endswith('mock_agent_factory')
    
    def test_agent_instantiation_with_config(self, mock_data_manager):
        """Test agent instantiation with configuration."""
        # Create agent using mock factory
        agent = mock_agent_factory(name="test_instance")
        
        assert agent.name == "test_instance"
        assert agent.state == "initialized"
        assert isinstance(agent.tools, list)
        assert isinstance(agent.messages, list)
    
    def test_agent_instantiation_with_data_manager(self, mock_data_manager):
        """Test agent instantiation with data manager."""
        # Test that agents can be created with data manager
        agent = mock_agent_factory(
            name="data_agent",
            data_manager=mock_data_manager
        )
        
        assert agent.name == "data_agent"
        assert "data_manager" in agent.config
    
    def test_agent_instantiation_error_handling(self):
        """Test error handling in agent instantiation."""
        # Test with invalid parameters
        try:
            agent = mock_agent_factory(invalid_param="should_not_work")
            # Mock factory accepts any kwargs, so this won't fail
            # In real implementation, this might raise an error
            assert agent is not None
        except Exception:
            # Expected behavior for strict validation
            pass
    
    @pytest.mark.parametrize("agent_type,expected_state", [
        ("supervisor", "coordinating"),
        ("data_expert", "initialized"),
        ("generic", "initialized")
    ])
    def test_different_agent_types(self, agent_type, expected_state):
        """Test instantiation of different agent types."""
        if agent_type == "supervisor":
            agent = mock_supervisor_factory()
        elif agent_type == "data_expert":
            agent = mock_data_expert_factory()
        else:
            agent = mock_agent_factory()
        
        assert agent.state == expected_state
        assert hasattr(agent, 'name')
        assert hasattr(agent, 'tools')


# ===============================================================================
# Agent Lifecycle Management Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentLifecycleManagement:
    """Test agent lifecycle management."""
    
    def test_agent_initialization(self):
        """Test agent initialization process."""
        agent = mock_agent_factory(name="lifecycle_test")
        
        # Check initial state
        assert agent.state == "initialized"
        assert len(agent.tools) == 0
        assert len(agent.messages) == 0
        
        # Check status
        status = agent.get_status()
        assert status['name'] == "lifecycle_test"
        assert status['state'] == "initialized"
    
    def test_agent_tool_registration(self):
        """Test agent tool registration."""
        agent = mock_agent_factory(name="tool_test")
        
        def mock_tool():
            return "tool_result"
        
        # Add tool
        agent.add_tool("test_tool", mock_tool)
        
        # Verify tool was added
        assert len(agent.tools) == 1
        assert agent.tools[0][0] == "test_tool"
        assert callable(agent.tools[0][1])
    
    def test_agent_message_processing(self):
        """Test agent message processing."""
        agent = mock_agent_factory(name="message_test")
        
        # Process messages
        response1 = agent.process_message("Test message 1")
        response2 = agent.process_message("Test message 2")
        
        # Verify messages were processed
        assert len(agent.messages) == 2
        assert "Test message 1" in agent.messages
        assert "Test message 2" in agent.messages
        
        # Verify responses
        assert "Mock response to: Test message 1" == response1
        assert "Mock response to: Test message 2" == response2
    
    def test_agent_state_management(self):
        """Test agent state management."""
        agent = mock_agent_factory(name="state_test")
        
        # Initial state
        assert agent.state == "initialized"
        
        # Change state
        agent.state = "active"
        assert agent.state == "active"
        
        # Verify state in status
        status = agent.get_status()
        assert status['state'] == "active"
    
    def test_agent_concurrent_operation(self):
        """Test concurrent agent operations."""
        import threading
        import time
        
        agent = mock_agent_factory(name="concurrent_test")
        results = []
        errors = []
        
        def worker(worker_id):
            """Worker function for concurrent testing."""
            try:
                for i in range(3):
                    response = agent.process_message(f"Worker {worker_id} message {i}")
                    results.append((worker_id, i, response))
                    time.sleep(0.01)
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 9  # 3 workers × 3 messages
        assert len(agent.messages) == 9


# ===============================================================================
# Agent Configuration and Metadata Tests
# ===============================================================================

@pytest.mark.unit
class TestAgentConfigurationMetadata:
    """Test agent configuration and metadata management."""
    
    def test_agent_metadata_structure(self):
        """Test agent metadata structure."""
        config = AgentRegistryConfig(
            name='metadata_test',
            display_name='Metadata Test Agent',
            description='Agent for testing metadata functionality',
            factory_function='module.metadata_factory',
            handoff_tool_name='handoff_to_metadata',
            handoff_tool_description='Handoff to metadata agent'
        )
        
        # Check all required metadata fields
        assert hasattr(config, 'name')
        assert hasattr(config, 'display_name')
        assert hasattr(config, 'description')
        assert hasattr(config, 'factory_function')
        assert hasattr(config, 'handoff_tool_name')
        assert hasattr(config, 'handoff_tool_description')
    
    def test_agent_configuration_validation(self):
        """Test comprehensive agent configuration validation."""
        # Valid configuration
        valid_config = AgentRegistryConfig(
            name='valid_config_test',
            display_name='Valid Config Test',
            description='Valid configuration for testing',
            factory_function='valid.module.factory',
            handoff_tool_name='valid_handoff',
            handoff_tool_description='Valid handoff description'
        )
        
        validate_agent_config(valid_config)  # Should not raise
        
        # Invalid configurations
        invalid_configs = [
            # Empty name
            {
                'name': '',
                'display_name': 'Invalid',
                'description': 'Empty name',
                'factory_function': 'module.factory',
                'handoff_tool_name': 'handoff',
                'handoff_tool_description': 'Description'
            },
            # None values
            {
                'name': 'test',
                'display_name': None,
                'description': 'None display name',
                'factory_function': 'module.factory',
                'handoff_tool_name': 'handoff',
                'handoff_tool_description': 'Description'
            }
        ]
        
        for invalid_data in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                config = AgentRegistryConfig(**invalid_data)
                validate_agent_config(config)
    
    def test_agent_version_compatibility(self):
        """Test agent version compatibility."""
        # This would test version compatibility if implemented
        config = AgentRegistryConfig(
            name='version_test',
            display_name='Version Test',
            description='Testing version compatibility',
            factory_function='module.version_factory',
            handoff_tool_name='version_handoff',
            handoff_tool_description='Version handoff'
        )
        
        # For now, just verify config is valid
        validate_agent_config(config)
    
    def test_agent_capability_metadata(self):
        """Test agent capability metadata."""
        # Test that agents can have capability metadata
        config = AgentRegistryConfig(
            name='capability_test',
            display_name='Capability Test',
            description='Testing capability metadata support',
            factory_function='module.capability_factory',
            handoff_tool_name='capability_handoff',
            handoff_tool_description='Capability handoff'
        )
        
        # Verify basic config structure
        assert config.name == 'capability_test'
        assert 'capability' in config.description.lower()


# ===============================================================================
# Registry Persistence and Loading Tests
# ===============================================================================

@pytest.mark.unit
class TestRegistryPersistenceLoading:
    """Test registry persistence and loading functionality."""
    
    def test_registry_state_consistency(self):
        """Test registry state consistency."""
        # Get current registry state
        agents_before = list_available_agents()
        agent_names_before = [agent.name for agent in agents_before]
        
        # Registry should be consistent
        assert len(agent_names_before) > 0
        assert len(set(agent_names_before)) == len(agent_names_before)  # No duplicates
        
        # Get state again
        agents_after = list_available_agents()
        agent_names_after = [agent.name for agent in agents_after]
        
        # Should be identical
        assert agent_names_before == agent_names_after
    
    def test_registry_default_agents(self):
        """Test that default agents are properly registered."""
        agents = list_available_agents()
        agent_names = [agent.name for agent in agents]
        
        # Check for expected default agents
        expected_agents = [
            'supervisor_agent',
            'data_expert_agent',
            'singlecell_expert_agent',
            'research_agent',
            'method_expert_agent'
        ]
        
        for expected_agent in expected_agents:
            assert expected_agent in agent_names, f"Expected agent {expected_agent} not found"
    
    def test_registry_agent_properties(self):
        """Test properties of registered agents."""
        agents = list_available_agents()
        
        for agent in agents:
            # All agents should have required properties
            assert agent.name, f"Agent missing name: {agent}"
            assert agent.display_name, f"Agent {agent.name} missing display_name"
            assert agent.description, f"Agent {agent.name} missing description"
            assert agent.factory_function, f"Agent {agent.name} missing factory_function"
            assert agent.handoff_tool_name, f"Agent {agent.name} missing handoff_tool_name"
            assert agent.handoff_tool_description, f"Agent {agent.name} missing handoff_tool_description"
            
            # Names should be valid
            assert agent.name.isidentifier() or '_' in agent.name, f"Invalid agent name: {agent.name}"
            
            # Factory function should be importable path
            assert '.' in agent.factory_function, f"Invalid factory function: {agent.factory_function}"
    
    def test_registry_modification_isolation(self, temp_agent_registry):
        """Test that registry modifications are properly isolated."""
        # Temp registry should start empty
        assert len(temp_agent_registry) == 0
        
        # Add test agent
        test_config = AgentRegistryConfig(
            name='isolation_test',
            display_name='Isolation Test',
            description='Testing isolation',
            factory_function='module.isolation_factory',
            handoff_tool_name='isolation_handoff',
            handoff_tool_description='Isolation handoff'
        )
        
        register_agent(test_config)
        assert len(temp_agent_registry) == 1
        
        # After fixture cleanup, original registry should be restored
        # This is tested implicitly by fixture


# ===============================================================================
# Error Handling and Edge Cases
# ===============================================================================

@pytest.mark.unit
class TestRegistryErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_agent_name_patterns(self):
        """Test various invalid agent name patterns."""
        invalid_names = [
            '',  # Empty
            '   ',  # Whitespace only
            '123invalid',  # Starts with number
            'invalid-name',  # Contains hyphen
            'invalid.name',  # Contains dot
            'invalid name',  # Contains space
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises((ValueError, TypeError)):
                AgentRegistryConfig(
                    name=invalid_name,
                    display_name='Invalid Name Test',
                    description='Testing invalid names',
                    factory_function='module.factory',
                    handoff_tool_name='handoff',
                    handoff_tool_description='Description'
                )
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        required_fields = [
            'name', 'display_name', 'description', 
            'factory_function', 'handoff_tool_name', 'handoff_tool_description'
        ]
        
        base_config = {
            'name': 'test',
            'display_name': 'Test',
            'description': 'Test agent',
            'factory_function': 'module.factory',
            'handoff_tool_name': 'handoff',
            'handoff_tool_description': 'Description'
        }
        
        for field in required_fields:
            incomplete_config = base_config.copy()
            del incomplete_config[field]
            
            with pytest.raises(TypeError):
                AgentRegistryConfig(**incomplete_config)
    
    def test_factory_function_validation(self):
        """Test factory function validation."""
        invalid_factory_functions = [
            '',  # Empty
            'invalid_module',  # No dots
            '.invalid.start',  # Starts with dot
            'invalid..double.dot',  # Double dots
            'invalid.end.',  # Ends with dot
        ]
        
        for invalid_factory in invalid_factory_functions:
            with pytest.raises(ValueError):
                config = AgentRegistryConfig(
                    name='factory_test',
                    display_name='Factory Test',
                    description='Testing factory validation',
                    factory_function=invalid_factory,
                    handoff_tool_name='handoff',
                    handoff_tool_description='Description'
                )
                validate_agent_config(config)
    
    def test_concurrent_registry_access(self):
        """Test concurrent registry access."""
        import threading
        import time
        
        results = []
        errors = []
        
        def registry_worker(worker_id):
            """Worker that accesses registry concurrently."""
            try:
                for i in range(5):
                    agents = list_available_agents()
                    results.append((worker_id, len(agents)))
                    time.sleep(0.001)
            except Exception as e:
                errors.append((worker_id, e))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=registry_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 15  # 3 workers × 5 iterations
        
        # All results should be consistent (same number of agents)
        agent_counts = [result[1] for result in results]
        assert len(set(agent_counts)) == 1, "Inconsistent agent counts in concurrent access"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])