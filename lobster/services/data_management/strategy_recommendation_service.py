"""
Strategy Recommendation Service for download strategy selection.

This service provides stateless, configuration-driven download strategy
recommendation and validation extracted from data_expert_assistant.py.
Follows the 3-tuple pattern (strategy, metadata, AnalysisStep).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from lobster.agents.data_expert.config import (
    PLATFORM_STRATEGY_CONFIGS,
    SUPPORTED_DOWNLOAD_STRATEGIES,
    get_strategy_config,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec

logger = logging.getLogger(__name__)


class StrategyRecommendationService:
    """
    Stateless service for download strategy recommendation and validation.

    This service analyzes platform type and file availability to recommend
    the optimal download strategy using configuration-driven logic.
    """

    def __init__(self):
        """Initialize stateless service."""
        logger.debug(f"Initializing {self.__class__.__name__}")

    def recommend_strategy(
        self,
        platform_type: str,
        file_patterns: Optional[List[str]] = None,
        has_processed: bool = False,
        has_raw: bool = False,
    ) -> Tuple[str, Dict[str, Any], AnalysisStep]:
        """
        Recommend download strategy based on platform type and available files.

        Args:
            platform_type: Detected platform (e.g., "10x", "h5ad", "kallisto")
            file_patterns: Optional list of available file names
            has_processed: Whether processed matrix files are available
            has_raw: Whether raw/UMI matrix files are available

        Returns:
            Tuple containing:
            - strategy: Recommended strategy (e.g., "MATRIX_FIRST", "H5_FIRST")
            - metadata: Dictionary with recommendation details
            - ir: AnalysisStep for provenance tracking

        Raises:
            ValueError: If platform_type is not recognized
        """
        logger.info(
            f"Recommending strategy for platform={platform_type}, "
            f"has_processed={has_processed}, has_raw={has_raw}"
        )

        # Get platform configuration
        try:
            config = get_strategy_config(platform_type)
        except ValueError as e:
            raise ValueError(f"Strategy recommendation failed: {str(e)}") from e

        # Determine best strategy based on available files
        strategy = config.default_strategy
        reasoning = []

        # Strategy selection logic
        if platform_type == "h5ad":
            strategy = "H5_FIRST"
            reasoning.append("H5AD files preferred for pre-processed data")
        elif platform_type == "10x":
            if has_processed and has_raw:
                strategy = "MATRIX_FIRST" if config.prefer_processed else "SAMPLES_FIRST"
                reasoning.append(
                    f"Both processed and raw available, "
                    f"{'preferring processed' if config.prefer_processed else 'preferring raw'}"
                )
            elif has_processed:
                strategy = "MATRIX_FIRST"
                reasoning.append("Processed matrix available, using MATRIX_FIRST")
            elif has_raw:
                strategy = "SAMPLES_FIRST"
                reasoning.append("Only raw data available, using SAMPLES_FIRST")
            else:
                # Check for .h5 files in patterns
                if file_patterns and any(".h5" in fp.lower() for fp in file_patterns):
                    strategy = "H5_FIRST"
                    reasoning.append("H5 files detected, using H5_FIRST")
                else:
                    strategy = "MATRIX_FIRST"  # Default fallback
                    reasoning.append("Using default MATRIX_FIRST strategy")
        elif platform_type in ["kallisto", "salmon", "csv", "proteomics"]:
            strategy = "SUPPLEMENTARY_FILES"
            reasoning.append(
                f"{config.display_name} platform uses supplementary files strategy"
            )
        else:
            # Unknown platform - use AUTO
            strategy = "AUTO"
            reasoning.append(f"Unknown platform '{platform_type}', using AUTO strategy")

        # Build metadata dictionary
        metadata = {
            "platform_type": platform_type,
            "strategy": strategy,
            "supported_strategies": config.supported_strategies,
            "reasoning": reasoning,
            "prefer_processed": config.prefer_processed,
            "validation_level": config.validation_level,
        }

        logger.info(
            f"Strategy recommendation: {strategy} for {platform_type}. "
            f"Reasoning: {'; '.join(reasoning)}"
        )

        # Create IR
        ir = self._create_recommend_ir(
            platform_type=platform_type,
            strategy=strategy,
            reasoning=reasoning,
        )

        return strategy, metadata, ir

    def validate_strategy(
        self, strategy: str, platform_type: str
    ) -> Tuple[bool, Optional[str], AnalysisStep]:
        """
        Validate that a strategy is compatible with a platform type.

        Args:
            strategy: Download strategy to validate (e.g., "MATRIX_FIRST")
            platform_type: Platform type (e.g., "10x", "h5ad")

        Returns:
            Tuple containing:
            - is_valid: True if strategy is compatible
            - error_message: Error message if invalid, None if valid
            - ir: AnalysisStep for provenance tracking
        """
        logger.info(f"Validating strategy={strategy} for platform={platform_type}")

        # Validate strategy is recognized
        if strategy not in SUPPORTED_DOWNLOAD_STRATEGIES:
            error_msg = (
                f"Invalid strategy '{strategy}'. "
                f"Supported: {SUPPORTED_DOWNLOAD_STRATEGIES}"
            )
            ir = self._create_validate_ir(strategy, platform_type, False, error_msg)
            return False, error_msg, ir

        # Get platform configuration
        try:
            config = get_strategy_config(platform_type)
        except ValueError as e:
            error_msg = str(e)
            ir = self._create_validate_ir(strategy, platform_type, False, error_msg)
            return False, error_msg, ir

        # Check if strategy is supported by platform
        if strategy not in config.supported_strategies and strategy != "AUTO":
            error_msg = (
                f"Strategy '{strategy}' not supported for platform '{platform_type}'. "
                f"Supported strategies: {config.supported_strategies}"
            )
            ir = self._create_validate_ir(strategy, platform_type, False, error_msg)
            return False, error_msg, ir

        # Valid
        ir = self._create_validate_ir(strategy, platform_type, True, None)
        logger.info(f"Strategy validation passed: {strategy} is valid for {platform_type}")
        return True, None, ir

    def _create_recommend_ir(
        self, platform_type: str, strategy: str, reasoning: List[str]
    ) -> AnalysisStep:
        """Create IR for recommend_strategy operation."""
        parameter_schema = {
            "platform_type": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=platform_type,
                required=True,
                validation_rule=None,
                description="Platform type for strategy recommendation",
            ),
            "strategy": ParameterSpec(
                param_type="str",
                papermill_injectable=False,
                default_value=strategy,
                required=False,
                validation_rule=None,
                description="Recommended download strategy",
            ),
        }

        code_template = """# Download Strategy Recommendation
# Recommends optimal download strategy based on platform type

platform_type = '{{ platform_type }}'

# Platform-specific strategy mapping
STRATEGY_MAP = {
    "10x": "MATRIX_FIRST",
    "h5ad": "H5_FIRST",
    "kallisto": "SUPPLEMENTARY_FILES",
    "salmon": "SUPPLEMENTARY_FILES",
}

strategy = STRATEGY_MAP.get(platform_type, "AUTO")
print(f"Recommended strategy for {platform_type}: {strategy}")
"""

        return AnalysisStep(
            operation="lobster.services.data_management.strategy_recommendation_service.recommend_strategy",
            tool_name="recommend_strategy",
            description=f"""## Strategy Recommendation

Recommends optimal download strategy for platform type.

**Platform**: {platform_type}
**Strategy**: {strategy}
**Reasoning**: {'; '.join(reasoning)}
""",
            library="lobster",
            code_template=code_template,
            imports=[],
            parameters={"platform_type": platform_type, "strategy": strategy},
            parameter_schema=parameter_schema,
            input_entities=["platform_type"],
            output_entities=["strategy"],
            execution_context={"reasoning": reasoning},
            validates_on_export=True,
            exportable=True,
        )

    def _create_validate_ir(
        self,
        strategy: str,
        platform_type: str,
        is_valid: bool,
        error_message: Optional[str],
    ) -> AnalysisStep:
        """Create IR for validate_strategy operation."""
        parameter_schema = {
            "strategy": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=strategy,
                required=True,
                validation_rule=None,
                description="Download strategy to validate",
            ),
            "platform_type": ParameterSpec(
                param_type="str",
                papermill_injectable=True,
                default_value=platform_type,
                required=True,
                validation_rule=None,
                description="Platform type for validation",
            ),
        }

        code_template = """# Download Strategy Validation
# Validates strategy compatibility with platform

strategy = '{{ strategy }}'
platform_type = '{{ platform_type }}'

# Platform compatibility check
PLATFORM_STRATEGIES = {
    "10x": ["MATRIX_FIRST", "H5_FIRST", "AUTO"],
    "h5ad": ["H5_FIRST", "AUTO"],
    "kallisto": ["SUPPLEMENTARY_FILES", "AUTO"],
}

is_valid = strategy in PLATFORM_STRATEGIES.get(platform_type, ["AUTO"])
print(f"Strategy '{strategy}' valid for {platform_type}: {is_valid}")
"""

        return AnalysisStep(
            operation="lobster.services.data_management.strategy_recommendation_service.validate_strategy",
            tool_name="validate_strategy",
            description=f"""## Strategy Validation

Validates strategy compatibility with platform type.

**Strategy**: {strategy}
**Platform**: {platform_type}
**Valid**: {is_valid}
{f'**Error**: {error_message}' if error_message else ''}
""",
            library="lobster",
            code_template=code_template,
            imports=[],
            parameters={
                "strategy": strategy,
                "platform_type": platform_type,
                "is_valid": is_valid,
            },
            parameter_schema=parameter_schema,
            input_entities=["strategy", "platform_type"],
            output_entities=["is_valid"],
            execution_context={"error_message": error_message} if error_message else {},
            validates_on_export=True,
            exportable=True,
        )
