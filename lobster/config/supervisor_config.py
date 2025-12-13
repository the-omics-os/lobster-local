"""
Supervisor Configuration System for the Lobster platform.

This module provides a flexible configuration system for the supervisor agent,
allowing users to customize interaction styles, behavior, and context inclusion
through environment variables while maintaining backward compatibility.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SupervisorConfig:
    """Configuration for supervisor agent behavior.

    This configuration allows customization of how the supervisor interacts
    with users and delegates to expert agents. All settings have defaults
    that match the current behavior to ensure backward compatibility.
    """

    # Interaction settings
    ask_clarification_questions: bool = True
    max_clarification_questions: int = 2
    require_download_confirmation: bool = True
    require_metadata_preview: bool = True

    # Response settings
    auto_suggest_next_steps: bool = True
    verbose_delegation: bool = True
    include_expert_output: bool = True
    summarize_expert_output: bool = False

    # Context inclusion
    include_data_context: bool = True
    include_workspace_status: bool = True
    include_system_info: bool = False
    include_memory_stats: bool = False

    # Workflow settings
    workflow_guidance_level: str = "detailed"  # minimal, standard, detailed
    show_available_tools: bool = True
    show_agent_capabilities: bool = True

    # Advanced settings
    delegation_strategy: str = "auto"  # auto, conservative, aggressive
    error_handling: str = "informative"  # silent, informative, verbose

    # Planning settings
    enable_todo_planning: bool = True  # Enable todo planning tools
    require_planning_for_complex_tasks: bool = True  # Require planning for complex tasks
    workspace_metadata_level_warning: bool = True  # Warn about large metadata fetches

    # Agent discovery settings
    auto_discover_agents: bool = True
    include_agent_tools: bool = True  # Whether to list individual agent tools
    max_tools_per_agent: int = 20  # Limit tools shown per agent for brevity

    @classmethod
    def from_env(cls) -> "SupervisorConfig":
        """Load configuration from environment variables.

        Environment variables follow the pattern SUPERVISOR_<SETTING_NAME>.
        For example: SUPERVISOR_ASK_QUESTIONS, SUPERVISOR_VERBOSE, etc.

        Returns:
            SupervisorConfig: Configuration instance with environment overrides
        """
        config = cls()

        # Map environment variables to config fields
        env_mappings = {
            "SUPERVISOR_ASK_QUESTIONS": ("ask_clarification_questions", bool),
            "SUPERVISOR_MAX_QUESTIONS": ("max_clarification_questions", int),
            "SUPERVISOR_REQUIRE_CONFIRMATION": ("require_download_confirmation", bool),
            "SUPERVISOR_REQUIRE_PREVIEW": ("require_metadata_preview", bool),
            "SUPERVISOR_AUTO_SUGGEST": ("auto_suggest_next_steps", bool),
            "SUPERVISOR_VERBOSE": ("verbose_delegation", bool),
            "SUPERVISOR_INCLUDE_EXPERT_OUTPUT": ("include_expert_output", bool),
            "SUPERVISOR_SUMMARIZE_OUTPUT": ("summarize_expert_output", bool),
            "SUPERVISOR_INCLUDE_DATA": ("include_data_context", bool),
            "SUPERVISOR_INCLUDE_WORKSPACE": ("include_workspace_status", bool),
            "SUPERVISOR_INCLUDE_SYSTEM": ("include_system_info", bool),
            "SUPERVISOR_INCLUDE_MEMORY": ("include_memory_stats", bool),
            "SUPERVISOR_WORKFLOW_GUIDANCE": ("workflow_guidance_level", str),
            "SUPERVISOR_SHOW_TOOLS": ("show_available_tools", bool),
            "SUPERVISOR_SHOW_CAPABILITIES": ("show_agent_capabilities", bool),
            "SUPERVISOR_DELEGATION_STRATEGY": ("delegation_strategy", str),
            "SUPERVISOR_ERROR_HANDLING": ("error_handling", str),
            "SUPERVISOR_ENABLE_TODO_PLANNING": ("enable_todo_planning", bool),
            "SUPERVISOR_REQUIRE_PLANNING": ("require_planning_for_complex_tasks", bool),
            "SUPERVISOR_WORKSPACE_WARNING": ("workspace_metadata_level_warning", bool),
            "SUPERVISOR_AUTO_DISCOVER": ("auto_discover_agents", bool),
            "SUPERVISOR_INCLUDE_AGENT_TOOLS": ("include_agent_tools", bool),
            "SUPERVISOR_MAX_TOOLS_PER_AGENT": ("max_tools_per_agent", int),
        }

        for env_var, (field_name, field_type) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    if field_type is bool:
                        setattr(
                            config, field_name, value.lower() in ("true", "1", "yes")
                        )
                    elif field_type is int:
                        setattr(config, field_name, int(value))
                    else:
                        setattr(config, field_name, value)
                    logger.debug(
                        f"Set {field_name} to {getattr(config, field_name)} from {env_var}"
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to set {field_name} from {env_var}: {e}")

        # Validate settings
        config._validate()

        return config

    def _validate(self):
        """Validate configuration settings."""
        # Validate workflow guidance level
        valid_guidance_levels = {"minimal", "standard", "detailed"}
        if self.workflow_guidance_level not in valid_guidance_levels:
            logger.warning(
                f"Invalid workflow_guidance_level '{self.workflow_guidance_level}', using 'standard'"
            )
            self.workflow_guidance_level = "standard"

        # Validate delegation strategy
        valid_strategies = {"auto", "conservative", "aggressive"}
        if self.delegation_strategy not in valid_strategies:
            logger.warning(
                f"Invalid delegation_strategy '{self.delegation_strategy}', using 'auto'"
            )
            self.delegation_strategy = "auto"

        # Validate error handling
        valid_error_modes = {"silent", "informative", "verbose"}
        if self.error_handling not in valid_error_modes:
            logger.warning(
                f"Invalid error_handling '{self.error_handling}', using 'informative'"
            )
            self.error_handling = "informative"

        # Validate numeric constraints
        if self.max_clarification_questions < 0:
            self.max_clarification_questions = 0
        elif self.max_clarification_questions > 10:
            self.max_clarification_questions = 10

        if self.max_tools_per_agent < 0:
            self.max_tools_per_agent = 0
        elif self.max_tools_per_agent > 20:
            self.max_tools_per_agent = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            "ask_clarification_questions": self.ask_clarification_questions,
            "max_clarification_questions": self.max_clarification_questions,
            "require_download_confirmation": self.require_download_confirmation,
            "require_metadata_preview": self.require_metadata_preview,
            "auto_suggest_next_steps": self.auto_suggest_next_steps,
            "verbose_delegation": self.verbose_delegation,
            "include_expert_output": self.include_expert_output,
            "summarize_expert_output": self.summarize_expert_output,
            "include_data_context": self.include_data_context,
            "include_workspace_status": self.include_workspace_status,
            "include_system_info": self.include_system_info,
            "include_memory_stats": self.include_memory_stats,
            "workflow_guidance_level": self.workflow_guidance_level,
            "show_available_tools": self.show_available_tools,
            "show_agent_capabilities": self.show_agent_capabilities,
            "delegation_strategy": self.delegation_strategy,
            "error_handling": self.error_handling,
            "enable_todo_planning": self.enable_todo_planning,
            "require_planning_for_complex_tasks": self.require_planning_for_complex_tasks,
            "workspace_metadata_level_warning": self.workspace_metadata_level_warning,
            "auto_discover_agents": self.auto_discover_agents,
            "include_agent_tools": self.include_agent_tools,
            "max_tools_per_agent": self.max_tools_per_agent,
        }

    def get_prompt_mode(self) -> str:
        """Get a human-readable description of the current configuration mode.

        Returns:
            str: Description of the configuration mode
        """
        if (
            not self.ask_clarification_questions
            and self.workflow_guidance_level == "minimal"
        ):
            return "Production Mode (Automated)"
        elif self.verbose_delegation and self.workflow_guidance_level == "detailed":
            return "Development Mode (Verbose)"
        elif (
            self.ask_clarification_questions
            and self.workflow_guidance_level == "detailed"
        ):
            return "Research Mode (Interactive)"
        else:
            return "Standard Mode"
