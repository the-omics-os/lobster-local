"""
Shared custom code execution tool factory.

Creates agent-specific execute_custom_code tools with unified signature
and optional post-processing hooks. Follows workspace_tool.py pattern.

Architecture:
    Factory Pattern: create_execute_custom_code_tool() returns @tool function
    Post-Processor: Optional callback for agent-specific behavior
    Single Source: Eliminates duplication across data_expert and metadata_assistant

Usage:
    # data_expert (no post-processing)
    tool = create_execute_custom_code_tool(data_manager, service, "data_expert")

    # metadata_assistant (with metadata_store persistence)
    tool = create_execute_custom_code_tool(
        data_manager, service, "metadata_assistant",
        post_processor=metadata_store_post_processor
    )

See Also:
    - workspace_tool.py: Established factory pattern for shared tools
    - CustomCodeExecutionService: Underlying execution service
    - CLAUDE.md section 4.5: Patterns & Abstractions
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from langchain_core.tools import tool

from lobster.utils.logger import get_logger

# TYPE_CHECKING imports - only evaluated by type checkers, not at runtime
# This prevents circular imports when component_registry loads entry points
if TYPE_CHECKING:
    from lobster.core.data_manager_v2 import DataManagerV2
    from lobster.services.execution.custom_code_execution_service import (
        CodeExecutionError,
        CodeValidationError,
        CustomCodeExecutionService,
    )

logger = get_logger(__name__)


# =============================================================================
# Type Aliases
# =============================================================================

# Post-processor callback signature
# Args: (result, stats, data_manager, workspace_key, modality_name)
# Returns: Optional message to append to response
# Note: Using string annotation to avoid runtime import of DataManagerV2
PostProcessor = Callable[
    [Any, Dict[str, Any], "DataManagerV2", Optional[str], Optional[str]],
    Optional[str],
]


# =============================================================================
# Factory Function
# =============================================================================


def create_execute_custom_code_tool(
    data_manager: DataManagerV2,
    custom_code_service: CustomCodeExecutionService,
    agent_name: str = "generic",
    post_processor: Optional[PostProcessor] = None,
):
    """
    Factory function to create execute_custom_code tool with configurable behavior.

    Follows workspace_tool.py pattern for shared tools across agents.
    The underlying CustomCodeExecutionService already supports both modality_name
    (for AnnData) and workspace_keys (for metadata), so NO service changes needed.

    Args:
        data_manager: DataManagerV2 instance for provenance logging
        custom_code_service: Service instance for code execution
        agent_name: Agent identifier for logging (default: "generic")
        post_processor: Optional callback for agent-specific post-processing.
                       Signature: (result, stats, data_manager, workspace_key, modality_name) -> Optional[str]
                       Return value (if any) is appended to response.

    Returns:
        LangChain tool for executing custom Python code

    Example:
        # data_expert (no post-processing)
        execute_custom_code = create_execute_custom_code_tool(
            data_manager=data_manager,
            custom_code_service=custom_code_service,
            agent_name="data_expert",
        )

        # metadata_assistant (with metadata_store persistence)
        execute_custom_code = create_execute_custom_code_tool(
            data_manager=data_manager,
            custom_code_service=custom_code_service,
            agent_name="metadata_assistant",
            post_processor=metadata_store_post_processor,
        )
    """

    @tool
    def execute_custom_code(
        python_code: str,
        modality_name: Optional[str] = None,
        workspace_key: Optional[str] = None,
        load_workspace_files: bool = True,
        persist: bool = False,
        description: str = "Custom code execution",
    ) -> str:
        """
        Execute custom Python code with workspace context.

        **Use this tool ONLY when existing specialized tools don't cover your need.**

        PARAMETER SELECTION (choose ONE based on your data type):
        - `modality_name`: Load AnnData modality as `adata` variable (for data operations)
        - `workspace_key`: Load specific JSON/CSV file (token-efficient, for metadata operations)
        - Neither: Load all workspace files (can be slow with many files)

        SECURITY MODEL:
        - Code runs in subprocess with filtered environment (API keys NOT accessible)
        - AST validation blocks dangerous imports (subprocess, pickle, ctypes, etc.)
        - 300-second timeout enforced
        - Network access and file permissions are NOT restricted (local CLI trust model)

        AVAILABLE IN NAMESPACE:
        - WORKSPACE: Path to workspace directory (pathlib.Path)
        - OUTPUT_DIR: Recommended path for exports (workspace/exports/)
        - adata: Loaded modality (if modality_name provided)
        - Auto-loaded CSV/JSON files (if load_workspace_files=True)
        - publication_queue, download_queue (if exist in workspace)

        RETURNING RESULTS:
        Assign to `result` variable: `result = my_computation()`

        DEFENSIVE CODING REQUIRED:
        - Use data.get('key', default) instead of data['key']
        - Check `if value is not None` before .lower()/.upper()
        - Convert numpy types: int(val), float(val), list(arr)
        - Check isinstance(data, dict) before dict operations

        Example defensive pattern:
            samples = data.get('samples', [])
            for s in samples:
                disease = s.get('disease')
                if disease:
                    print(disease.lower())

        NOTEBOOK EXPORT (persist parameter):
        - persist=False (default): Code is EPHEMERAL
          * Execution logged in provenance for tracking
          * NOT included when you run /notebook export
          * Use for: debugging, quick checks, exploratory analysis

        - persist=True: Code is REPRODUCIBLE
          * Execution logged in provenance for tracking
          * Included in /notebook export as executable cell
          * Code exported AS-IS (not templated) in exact workflow order
          * Use for: production workflows, data transformations, final analyses

        PERSISTING FOR METADATA OPERATIONS:
        Return dict with samples and output_key to persist to metadata_store:
        `result = {'samples': [...], 'output_key': 'filtered_samples'}`

        Args:
            python_code: Python code to execute (multi-line supported)
            modality_name: AnnData modality to load as 'adata' (for data operations)
            workspace_key: Specific workspace file to load (for metadata operations)
            load_workspace_files: Auto-inject CSV/JSON from workspace (default: True)
            persist: Control notebook export behavior (default: False)
                    - False: Ephemeral execution (exploratory, debugging, quick checks)
                              Logged in provenance but NOT included in /notebook export
                    - True: Reproducible execution (production workflow, data transformations)
                             Included in exported Jupyter notebook for full reproducibility
                    Use True for steps that are part of your final analysis pipeline.
            description: Human-readable description of the operation

        Returns:
            Formatted string with execution results, warnings, and outputs

        Example (exploratory - no notebook export):
            >>> execute_custom_code(
            ...     modality_name="geo_gse12345_filtered",
            ...     python_code="result = adata.obs['cell_type'].value_counts().to_dict()",
            ...     persist=False,  # Ephemeral debugging
            ...     description="Count cell types in filtered dataset"
            ... )

        Example (production workflow - included in notebook):
            >>> execute_custom_code(
            ...     workspace_key="aggregated_samples",
            ...     python_code=\"\"\"
            ...     # Filter samples with complete annotations
            ...     samples = aggregated_samples['samples']
            ...     valid = [s for s in samples if s.get('body_site') and s.get('disease')]
            ...     result = {'samples': valid, 'output_key': 'reviewed_samples'}
            ...     \"\"\",
            ...     persist=True,  # Include in /notebook export for reproducibility
            ...     description="Filter samples with body_site and disease annotations"
            ... )
            # This code will appear AS-IS in exported Jupyter notebook

        Example (quick check - no persistence):
            >>> execute_custom_code(
            ...     workspace_key="sra_prjna834801_samples",
            ...     python_code="result = len(sra_prjna834801_samples['samples'])",
            ...     persist=False,  # Quick check, not part of workflow
            ...     description="Count samples in PRJNA834801"
            ... )
        """
        # Lazy import to prevent circular imports when component_registry loads entry points
        # These exceptions are needed at runtime for except clauses, but importing the
        # service module at module level triggers DataManagerV2 → component_registry → agents
        from lobster.services.execution.custom_code_execution_service import (
            CodeExecutionError,
            CodeValidationError,
        )

        # Validation: mutual exclusivity
        if modality_name and workspace_key:
            return (
                "Error: Cannot specify both `modality_name` and `workspace_key`. "
                "Choose ONE based on your data type:\n"
                "- `modality_name`: For AnnData/H5AD operations\n"
                "- `workspace_key`: For JSON/CSV metadata operations"
            )

        try:
            # Convert workspace_key to list (service expects Optional[List[str]])
            workspace_keys_list = [workspace_key] if workspace_key else None

            # Execute via service (single source of truth)
            result, stats, ir = custom_code_service.execute(
                code=python_code,
                modality_name=modality_name,
                load_workspace_files=load_workspace_files,
                workspace_keys=workspace_keys_list,
                persist=persist,
                description=description,
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="execute_custom_code",
                parameters={
                    "description": description,
                    "modality_name": modality_name,
                    "workspace_key": workspace_key,
                    "persist": persist,
                    "agent": agent_name,
                    "duration_seconds": stats.get("duration_seconds", 0),
                    "success": stats.get("success", False),
                },
                description=f"{description} ({'success' if stats['success'] else 'failed'})",
                ir=ir,
            )

            # Agent-specific post-processing (only on success)
            post_processor_msg = None
            if post_processor and stats.get("success", False):
                try:
                    post_processor_msg = post_processor(
                        result, stats, data_manager, workspace_key, modality_name
                    )
                except Exception as e:
                    logger.warning(f"Post-processor error (non-fatal): {e}")
                    post_processor_msg = f"Warning: Post-processing failed: {e}"

            # Format response
            return _format_response(result, stats, persist, post_processor_msg)

        except CodeValidationError as e:
            logger.error(f"Code validation failed: {e}")
            import json
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "validation_error",
                "stderr": str(e)
            }, indent=2)
        except CodeExecutionError as e:
            logger.error(f"Code execution failed: {e}")
            import json
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "execution_error",
                "stderr": str(e)
            }, indent=2)
        except Exception as e:
            logger.error(f"Unexpected error in execute_custom_code: {e}", exc_info=True)
            import json
            return json.dumps({
                "success": False,
                "error": str(e),
                "error_type": "unexpected_error",
                "stderr": str(e)
            }, indent=2)

    return execute_custom_code


# =============================================================================
# Response Formatting
# =============================================================================


def _format_response(
    result: Any,
    stats: Dict[str, Any],
    persist: bool,
    post_processor_msg: Optional[str] = None,
) -> str:
    """
    Format unified JSON response for all agents (DeepAgents pattern).

    Returns structured JSON instead of markdown for programmatic parsing by LLMs.
    Enables retry logic, error pattern matching, and conditional branching.

    Args:
        result: Execution result value
        stats: Execution statistics from service
        persist: Whether execution was persisted to provenance
        post_processor_msg: Optional message from agent-specific post-processor

    Returns:
        JSON string with structured execution result
    """
    import json

    # Build typed response object
    response_obj = {
        "success": True,
        "duration_seconds": stats.get("duration_seconds", 0),
        "result": result,
        "result_type": stats.get("result_type", type(result).__name__ if result is not None else None),
        "stdout": stats.get("stdout_preview", ""),
        "stderr": stats.get("stderr_preview", ""),
        "stdout_full_path": stats.get("stdout_full_path"),
        "stderr_full_path": stats.get("stderr_full_path"),
        "warnings": stats.get("warnings", []),
        "persisted": persist,
        "post_processor_message": post_processor_msg,
    }

    return json.dumps(response_obj, indent=2)


# =============================================================================
# Pre-built Post-Processors
# =============================================================================


def metadata_store_post_processor(
    result: Any,
    stats: Dict[str, Any],
    data_manager: DataManagerV2,
    workspace_key: Optional[str],
    modality_name: Optional[str],
) -> Optional[str]:
    """
    Post-processor for metadata_assistant: Persist results to metadata_store.

    Handles two patterns:
    1. In-place update: If workspace_key exists in metadata_store, update its 'samples'
    2. New key creation: If result contains {'samples': [...], 'output_key': 'name'}

    This enables the metadata_assistant workflow where users filter/transform samples
    and persist them for later export via write_to_workspace.

    Args:
        result: Execution result (should be dict with 'samples' key for persistence)
        stats: Execution statistics (unused, but required by signature)
        data_manager: DataManagerV2 instance with metadata_store
        workspace_key: Original workspace key that was loaded (for in-place updates)
        modality_name: Modality name (unused for metadata operations)

    Returns:
        Message describing what was persisted, or None if nothing persisted

    Example Result Formats:
        # Pattern 1: In-place update (workspace_key="existing_key")
        result = {"samples": [{"id": 1}, {"id": 2}]}

        # Pattern 2: New key creation
        result = {"samples": [...], "output_key": "filtered_samples"}
    """
    if not isinstance(result, dict):
        return None

    # Pattern 1: In-place update of existing workspace_key
    if workspace_key and workspace_key in data_manager.metadata_store:
        if "samples" in result:
            data_manager.metadata_store[workspace_key]["samples"] = result["samples"]
            count = len(result["samples"])
            logger.info(
                f"Persisted {count} samples to metadata_store['{workspace_key}']"
            )
            return f"Persisted {count} samples to metadata_store['{workspace_key}']"

    # Pattern 2: Create new key with output_key
    elif "samples" in result and "output_key" in result:
        output_key = result["output_key"]
        data_manager.metadata_store[output_key] = {
            "samples": result["samples"],
            "filter_criteria": result.get("filter_criteria", "custom"),
            "stats": result.get("stats", {}),
        }
        count = len(result["samples"])
        logger.info(f"Created metadata_store['{output_key}'] with {count} samples")
        return (
            f"Created metadata_store['{output_key}'] with {count} samples\n"
            f"Export with: write_to_workspace(identifier='{output_key}', "
            f"workspace='metadata', output_format='csv')"
        )

    return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "create_execute_custom_code_tool",
    "metadata_store_post_processor",
    "PostProcessor",
]
