"""
Unit tests for expert handoff functionality.

This module tests the core handoff infrastructure including the ExpertHandoffManager,
enhanced handoff tools, and handoff patterns.
"""

import unittest
import uuid
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

import pytest

from lobster.tools.expert_handoff_manager import (
    ExpertHandoffManager,
    HandoffContext,
    HandoffResult,
    create_handoff_context,
    create_handoff_result
)
from lobster.tools.enhanced_handoff_tool import (
    create_expert_handoff_tool,
    validate_context_schema,
    SCVI_CONTEXT_SCHEMA
)
from lobster.tools.expert_handoff_patterns import (
    get_handoff_pattern,
    validate_handoff_pattern,
    get_context_schema_for_handoff,
    EXPERT_HANDOFF_PATTERNS
)
from lobster.config.agent_registry import (
    create_expert_handoff_tools,
    get_handoff_tools_for_agent,
    validate_handoff_compatibility
)


class TestExpertHandoffManager(unittest.TestCase):
    """Test ExpertHandoffManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = ExpertHandoffManager()
        self.test_handoff_id = str(uuid.uuid4())

    def test_handoff_manager_initialization(self):
        """Test that handoff manager initializes correctly."""
        self.assertEqual(len(self.manager.handoff_registry), 0)
        self.assertEqual(len(self.manager.active_handoffs), 0)
        self.assertEqual(len(self.manager.handoff_history), 0)
        self.assertEqual(self.manager._max_chain_depth, 10)

    def test_register_handoff(self):
        """Test handoff tool registration."""
        mock_tool = Mock()
        mock_tool.name = "test_handoff"

        self.manager.register_handoff("agent_a", "agent_b", mock_tool)

        self.assertIn("agent_a_to_agent_b", self.manager.handoff_registry)
        self.assertEqual(self.manager.handoff_registry["agent_a_to_agent_b"], mock_tool)

    def test_create_context_preserving_handoff(self):
        """Test context preserving handoff creation."""
        context = HandoffContext(
            handoff_id=self.test_handoff_id,
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            parameters={"modality_name": "test_data", "n_latent": 10},
            return_expectations={"embedding_key": "X_scvi"},
            timestamp=datetime.now().isoformat()
        )

        command = self.manager.create_context_preserving_handoff(
            to_expert="machine_learning_expert",
            context=context,
            return_to_sender=True
        )

        # Verify command structure
        self.assertEqual(command.goto, "machine_learning_expert")
        self.assertIn("handoff_context", command.update)
        self.assertIn("task_description", command.update)
        self.assertTrue(command.update["return_to_sender"])
        self.assertEqual(command.update["handoff_id"], self.test_handoff_id)

        # Verify handoff is tracked
        self.assertIn(self.test_handoff_id, self.manager.active_handoffs)

    def test_track_handoff_chain(self):
        """Test handoff chain tracking."""
        chain = ["agent_a", "agent_b", "agent_c"]
        self.manager.track_handoff_chain(self.test_handoff_id, chain)

        self.assertIn(self.test_handoff_id, self.manager.handoff_chains)
        self.assertEqual(self.manager.handoff_chains[self.test_handoff_id], chain)

    def test_get_return_path(self):
        """Test return path determination."""
        # Set up active handoff
        context = HandoffContext(
            handoff_id=self.test_handoff_id,
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            parameters={},
            return_expectations={},
            timestamp=datetime.now().isoformat()
        )
        self.manager.active_handoffs[self.test_handoff_id] = context

        # Test simple return path
        return_path = self.manager.get_return_path("machine_learning_expert", self.test_handoff_id)
        self.assertEqual(return_path, "singlecell_expert")

        # Test unknown handoff
        unknown_return = self.manager.get_return_path("unknown_agent", "unknown_id")
        self.assertIsNone(unknown_return)

    def test_complete_handoff(self):
        """Test handoff completion."""
        # Set up active handoff
        context = HandoffContext(
            handoff_id=self.test_handoff_id,
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            parameters={},
            return_expectations={},
            timestamp=datetime.now().isoformat()
        )
        self.manager.active_handoffs[self.test_handoff_id] = context

        # Create result
        result = HandoffResult(
            handoff_id=self.test_handoff_id,
            success=True,
            result_data={"embedding_key": "X_scvi", "shape": (1000, 10)}
        )

        # Complete handoff
        state = {"messages": []}
        completion_command = self.manager.complete_handoff(self.test_handoff_id, result, state)

        # Verify completion command
        self.assertEqual(completion_command.goto, "singlecell_expert")
        self.assertIn("handoff_result", completion_command.update)
        self.assertTrue(completion_command.update["handoff_completed"])

    def test_cleanup_handoff(self):
        """Test handoff cleanup."""
        # Set up active handoff
        context = HandoffContext(
            handoff_id=self.test_handoff_id,
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            parameters={},
            return_expectations={},
            timestamp=datetime.now().isoformat()
        )
        self.manager.active_handoffs[self.test_handoff_id] = context
        self.manager.handoff_chains[self.test_handoff_id] = ["machine_learning_expert"]

        # Cleanup
        self.manager.cleanup_handoff(self.test_handoff_id)

        # Verify cleanup
        self.assertNotIn(self.test_handoff_id, self.manager.active_handoffs)
        self.assertNotIn(self.test_handoff_id, self.manager.handoff_chains)
        self.assertTrue(len(self.manager.handoff_history) > 0)

    def test_max_chain_depth_protection(self):
        """Test protection against infinite handoff chains."""
        context = HandoffContext(
            handoff_id=self.test_handoff_id,
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            parameters={},
            return_expectations={},
            timestamp=datetime.now().isoformat()
        )

        # Create a deep chain that exceeds max depth
        deep_chain = [f"agent_{i}" for i in range(15)]
        self.manager.handoff_chains[self.test_handoff_id] = deep_chain

        # Should raise ValueError for chain depth exceeded
        with self.assertRaises(ValueError) as cm:
            self.manager.create_context_preserving_handoff(
                to_expert="another_agent",
                context=context,
                return_to_sender=True
            )

        self.assertIn("Maximum handoff chain depth", str(cm.exception))


class TestEnhancedHandoffTool(unittest.TestCase):
    """Test enhanced handoff tool functionality."""

    def test_validate_context_schema_success(self):
        """Test successful context validation."""
        context = {
            "modality_name": "test_data",
            "n_latent": 10,
            "batch_key": "sample_id",
            "max_epochs": 400,
            "use_gpu": False
        }

        validated = validate_context_schema(context, SCVI_CONTEXT_SCHEMA)

        self.assertEqual(validated["modality_name"], "test_data")
        self.assertEqual(validated["n_latent"], 10)
        self.assertEqual(validated["batch_key"], "sample_id")
        self.assertEqual(validated["max_epochs"], 400)
        self.assertFalse(validated["use_gpu"])

    def test_validate_context_schema_failure(self):
        """Test context validation failure."""
        # Missing required field
        context = {
            "n_latent": 10,
            "use_gpu": False
        }

        with self.assertRaises(ValueError) as cm:
            validate_context_schema(context, SCVI_CONTEXT_SCHEMA)

        self.assertIn("Required field 'modality_name' is missing", str(cm.exception))

    def test_validate_context_schema_type_error(self):
        """Test context validation type error."""
        context = {
            "modality_name": "test_data",
            "n_latent": "invalid_string",  # Should be int
            "batch_key": None,
            "max_epochs": 400,
            "use_gpu": False
        }

        with self.assertRaises(ValueError) as cm:
            validate_context_schema(context, SCVI_CONTEXT_SCHEMA)

        self.assertIn("must be int", str(cm.exception))

    @patch('lobster.tools.enhanced_handoff_tool.expert_handoff_manager')
    def test_create_expert_handoff_tool(self, mock_manager):
        """Test expert handoff tool creation."""
        # Mock the manager methods
        mock_manager.register_handoff = Mock()
        mock_manager.create_context_preserving_handoff = Mock()

        tool = create_expert_handoff_tool(
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            context_schema=SCVI_CONTEXT_SCHEMA,
            return_to_sender=True
        )

        # Verify tool was registered
        mock_manager.register_handoff.assert_called_once()

        # Verify tool properties
        self.assertTrue(hasattr(tool, 'name'))
        self.assertTrue(hasattr(tool, 'description'))


class TestExpertHandoffPatterns(unittest.TestCase):
    """Test expert handoff patterns functionality."""

    def test_get_handoff_pattern_success(self):
        """Test successful handoff pattern retrieval."""
        pattern = get_handoff_pattern("singlecell_expert", "machine_learning_expert", "scvi_training")

        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.from_expert, "singlecell_expert")
        self.assertEqual(pattern.to_expert, "machine_learning_expert")
        self.assertIn("scvi_training", pattern.task_types)

    def test_get_handoff_pattern_not_found(self):
        """Test handoff pattern not found."""
        pattern = get_handoff_pattern("nonexistent_expert", "another_expert", "unknown_task")

        self.assertIsNone(pattern)

    def test_validate_handoff_pattern_success(self):
        """Test successful handoff pattern validation."""
        is_valid = validate_handoff_pattern("singlecell_expert", "machine_learning_expert", "scvi_training")

        self.assertTrue(is_valid)

    def test_validate_handoff_pattern_failure(self):
        """Test handoff pattern validation failure."""
        is_valid = validate_handoff_pattern("nonexistent_expert", "another_expert", "unknown_task")

        self.assertFalse(is_valid)

    def test_get_context_schema_for_handoff(self):
        """Test context schema retrieval for handoff."""
        schema = get_context_schema_for_handoff("singlecell_expert", "machine_learning_expert", "scvi_training")

        self.assertIsNotNone(schema)
        self.assertIn("modality_name", schema)
        self.assertIn("n_latent", schema)

    def test_all_patterns_have_required_fields(self):
        """Test that all patterns have required fields."""
        for pattern_name, pattern in EXPERT_HANDOFF_PATTERNS.items():
            self.assertIsInstance(pattern.from_expert, str)
            self.assertIsInstance(pattern.to_expert, str)
            self.assertIsInstance(pattern.task_types, list)
            self.assertTrue(len(pattern.task_types) > 0)
            self.assertIsInstance(pattern.context_schema, dict)
            self.assertIn(pattern.return_flow, ["sender", "supervisor"])
            self.assertIsInstance(pattern.description, str)
            self.assertIsInstance(pattern.priority, int)


class TestAgentRegistryIntegration(unittest.TestCase):
    """Test agent registry integration with handoff system."""

    def test_create_expert_handoff_tools(self):
        """Test automatic handoff tool creation."""
        available_agents = [
            "singlecell_expert_agent",
            "machine_learning_expert_agent",
            "bulk_rnaseq_expert_agent"
        ]

        handoff_tools = create_expert_handoff_tools(available_agents)

        # Should have created tools for available patterns
        self.assertGreater(len(handoff_tools), 0)

        # Check specific tool exists
        expected_tool_name = "handoff_singlecell_expert_to_machine_learning_expert_scvi_training"
        self.assertIn(expected_tool_name, handoff_tools)

    def test_get_handoff_tools_for_agent(self):
        """Test getting handoff tools for specific agent."""
        available_agents = [
            "singlecell_expert_agent",
            "machine_learning_expert_agent",
            "bulk_rnaseq_expert_agent"
        ]

        tools = get_handoff_tools_for_agent("singlecell_expert_agent", available_agents)

        # Should have outgoing handoff tools for single cell expert
        self.assertGreater(len(tools), 0)

        # Tools should be for handoffs from singlecell_expert
        for tool in tools:
            self.assertTrue(hasattr(tool, 'name'))

    def test_validate_handoff_compatibility(self):
        """Test handoff compatibility validation."""
        # Test valid handoff
        is_compatible = validate_handoff_compatibility(
            "singlecell_expert_agent",
            "machine_learning_expert_agent",
            "scvi_training"
        )
        self.assertTrue(is_compatible)

        # Test invalid handoff
        is_compatible = validate_handoff_compatibility(
            "nonexistent_agent",
            "another_agent",
            "unknown_task"
        )
        self.assertFalse(is_compatible)


class TestHandoffContextAndResult(unittest.TestCase):
    """Test HandoffContext and HandoffResult data classes."""

    def test_handoff_context_creation(self):
        """Test HandoffContext creation and serialization."""
        context = create_handoff_context(
            from_expert="singlecell_expert",
            to_expert="machine_learning_expert",
            task_type="scvi_training",
            parameters={"modality_name": "test_data", "n_latent": 10}
        )

        self.assertEqual(context.from_expert, "singlecell_expert")
        self.assertEqual(context.to_expert, "machine_learning_expert")
        self.assertEqual(context.task_type, "scvi_training")
        self.assertIsInstance(context.handoff_id, str)

        # Test serialization
        context_dict = context.to_dict()
        self.assertIsInstance(context_dict, dict)
        self.assertIn("handoff_id", context_dict)

        # Test deserialization
        restored_context = HandoffContext.from_dict(context_dict)
        self.assertEqual(restored_context.from_expert, context.from_expert)

    def test_handoff_result_creation(self):
        """Test HandoffResult creation and serialization."""
        result = create_handoff_result(
            handoff_id="test-id",
            success=True,
            result_data={"embedding_key": "X_scvi", "shape": (1000, 10)}
        )

        self.assertEqual(result.handoff_id, "test-id")
        self.assertTrue(result.success)
        self.assertIsNone(result.error_message)

        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn("handoff_id", result_dict)


if __name__ == '__main__':
    unittest.main()