"""
Unit tests for the Supervisor Configuration System.
"""

import os
import pytest
from unittest.mock import patch
from lobster.config.supervisor_config import SupervisorConfig


class TestSupervisorConfig:
    """Test the SupervisorConfig class."""

    def test_default_configuration(self):
        """Test that default configuration matches current behavior."""
        config = SupervisorConfig()

        # Verify backward compatibility defaults
        assert config.ask_clarification_questions is True
        assert config.max_clarification_questions == 3
        assert config.require_download_confirmation is True
        assert config.require_metadata_preview is True
        assert config.auto_suggest_next_steps is True
        assert config.verbose_delegation is False
        assert config.include_expert_output is True
        assert config.workflow_guidance_level == "standard"
        assert config.delegation_strategy == "auto"
        assert config.error_handling == "informative"

    def test_from_env_with_no_env_vars(self):
        """Test loading from environment with no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = SupervisorConfig.from_env()

            # Should use defaults
            assert config.ask_clarification_questions is True
            assert config.max_clarification_questions == 3
            assert config.workflow_guidance_level == "standard"

    def test_from_env_with_env_vars(self):
        """Test loading from environment with env vars set."""
        env_vars = {
            'SUPERVISOR_ASK_QUESTIONS': 'false',
            'SUPERVISOR_MAX_QUESTIONS': '5',
            'SUPERVISOR_REQUIRE_CONFIRMATION': 'false',
            'SUPERVISOR_VERBOSE': 'true',
            'SUPERVISOR_WORKFLOW_GUIDANCE': 'minimal',
            'SUPERVISOR_DELEGATION_STRATEGY': 'aggressive',
            'SUPERVISOR_AUTO_DISCOVER': 'true',
        }

        with patch.dict(os.environ, env_vars):
            config = SupervisorConfig.from_env()

            assert config.ask_clarification_questions is False
            assert config.max_clarification_questions == 5
            assert config.require_download_confirmation is False
            assert config.verbose_delegation is True
            assert config.workflow_guidance_level == "minimal"
            assert config.delegation_strategy == "aggressive"
            assert config.auto_discover_agents is True

    def test_boolean_env_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('no', False),
        ]

        for value, expected in test_cases:
            with patch.dict(os.environ, {'SUPERVISOR_ASK_QUESTIONS': value}):
                config = SupervisorConfig.from_env()
                assert config.ask_clarification_questions == expected, f"Failed for value: {value}"

    def test_validation_workflow_guidance_level(self):
        """Test validation of workflow_guidance_level."""
        config = SupervisorConfig()

        # Valid values
        for level in ['minimal', 'standard', 'detailed']:
            config.workflow_guidance_level = level
            config._validate()
            assert config.workflow_guidance_level == level

        # Invalid value should default to 'standard'
        config.workflow_guidance_level = 'invalid'
        config._validate()
        assert config.workflow_guidance_level == 'standard'

    def test_validation_delegation_strategy(self):
        """Test validation of delegation_strategy."""
        config = SupervisorConfig()

        # Valid values
        for strategy in ['auto', 'conservative', 'aggressive']:
            config.delegation_strategy = strategy
            config._validate()
            assert config.delegation_strategy == strategy

        # Invalid value should default to 'auto'
        config.delegation_strategy = 'invalid'
        config._validate()
        assert config.delegation_strategy == 'auto'

    def test_validation_error_handling(self):
        """Test validation of error_handling."""
        config = SupervisorConfig()

        # Valid values
        for mode in ['silent', 'informative', 'verbose']:
            config.error_handling = mode
            config._validate()
            assert config.error_handling == mode

        # Invalid value should default to 'informative'
        config.error_handling = 'invalid'
        config._validate()
        assert config.error_handling == 'informative'

    def test_validation_numeric_constraints(self):
        """Test validation of numeric constraints."""
        config = SupervisorConfig()

        # Test max_clarification_questions
        config.max_clarification_questions = -1
        config._validate()
        assert config.max_clarification_questions == 0

        config.max_clarification_questions = 15
        config._validate()
        assert config.max_clarification_questions == 10

        config.max_clarification_questions = 5
        config._validate()
        assert config.max_clarification_questions == 5

        # Test max_tools_per_agent
        config.max_tools_per_agent = -1
        config._validate()
        assert config.max_tools_per_agent == 0

        config.max_tools_per_agent = 25
        config._validate()
        assert config.max_tools_per_agent == 20

        config.max_tools_per_agent = 10
        config._validate()
        assert config.max_tools_per_agent == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SupervisorConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['ask_clarification_questions'] is True
        assert config_dict['max_clarification_questions'] == 3
        assert config_dict['workflow_guidance_level'] == 'standard'
        assert 'auto_discover_agents' in config_dict
        assert len(config_dict) == 20  # Check all fields are present

    def test_get_prompt_mode(self):
        """Test get_prompt_mode method."""
        config = SupervisorConfig()

        # Production Mode
        config.ask_clarification_questions = False
        config.workflow_guidance_level = 'minimal'
        assert config.get_prompt_mode() == "Production Mode (Automated)"

        # Development Mode
        config.verbose_delegation = True
        config.workflow_guidance_level = 'detailed'
        assert config.get_prompt_mode() == "Development Mode (Verbose)"

        # Research Mode
        config.ask_clarification_questions = True
        config.verbose_delegation = False
        config.workflow_guidance_level = 'detailed'
        assert config.get_prompt_mode() == "Research Mode (Interactive)"

        # Standard Mode
        config.workflow_guidance_level = 'standard'
        assert config.get_prompt_mode() == "Standard Mode"

    def test_env_override_with_invalid_types(self):
        """Test that invalid environment variable types are handled gracefully."""
        env_vars = {
            'SUPERVISOR_MAX_QUESTIONS': 'not_a_number',
            'SUPERVISOR_MAX_TOOLS_PER_AGENT': 'invalid',
        }

        with patch.dict(os.environ, env_vars):
            # Should not raise an exception
            config = SupervisorConfig.from_env()

            # Should use defaults when parsing fails
            assert config.max_clarification_questions == 3
            assert config.max_tools_per_agent == 5