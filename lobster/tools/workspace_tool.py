"""
Workspace content retrieval tool for accessing cached research content.

This module provides a factory function for creating get_content_from_workspace tool
that can be shared between multiple agents (research_agent, supervisor).
"""

import json

from langchain_core.tools import tool

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.tools.workspace_content_service import (
    ContentType,
    RetrievalLevel,
    WorkspaceContentService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def create_get_content_from_workspace_tool(data_manager: DataManagerV2):
    """
    Factory function to create get_content_from_workspace tool with data_manager closure.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for retrieving workspace content
    """

    @tool
    def get_content_from_workspace(
        identifier: str = None,
        workspace: str = None,
        level: str = "summary",
        status_filter: str = None,
    ) -> str:
        """
        Retrieve cached research content from workspace with flexible detail levels.

        Reads previously cached publications, datasets, and metadata from workspace
        directories. Supports listing all content, filtering by workspace, and
        extracting specific details (summary, methods, samples, platform, full metadata).

        Workspace categories:
        - "literature": Publications, papers, abstracts
        - "data": Dataset metadata, GEO records
        - "metadata": Validation results, sample mappings
        - "download_queue": Pending/completed download tasks

        Detail Levels:
        - "summary": Key-value pairs, high-level overview (default)
        - "methods": Methods section (for publications)
        - "samples": Sample IDs list (for datasets)
        - "platform": Platform information (for datasets)
        - "metadata": Full metadata (for any content)
        - "github": GitHub repositories (for publications)
        - "validation": Validation results (for download_queue)
        - "strategy": Download strategy (for download_queue)

        For download_queue workspace:
        - identifier=None: List all entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific entry
        - status_filter: "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED"
        - level: "summary" (basic info) | "metadata" (full entry details)

        Args:
            identifier: Content identifier to retrieve (None = list all)
            workspace: Filter by workspace category (None = all workspaces)
            level: Detail level to extract (default: "summary")
            status_filter: Status filter for download_queue (optional)

        Returns:
            Formatted content based on detail level or list of cached items

        Examples:
            # List all cached content
            get_content_from_workspace()

            # List content in specific workspace
            get_content_from_workspace(workspace="literature")

            # Read publication methods section
            get_content_from_workspace(
                identifier="publication_PMID12345",
                workspace="literature",
                level="methods"
            )

            # Get dataset sample IDs
            get_content_from_workspace(
                identifier="dataset_GSE12345",
                workspace="data",
                level="samples"
            )

            # Get full metadata
            get_content_from_workspace(
                identifier="metadata_GSE12345_samples",
                workspace="metadata",
                level="metadata"
            )

            # List pending downloads
            get_content_from_workspace(
                workspace="download_queue",
                status_filter="PENDING"
            )

            # Get download queue entry details
            get_content_from_workspace(
                identifier="queue_entry_123",
                workspace="download_queue",
                level="metadata"
            )
        """
        try:
            # Initialize workspace service
            workspace_service = WorkspaceContentService(data_manager=data_manager)

            # Map workspace strings to ContentType enum
            workspace_to_content_type = {
                "literature": ContentType.PUBLICATION,
                "data": ContentType.DATASET,
                "metadata": ContentType.METADATA,
                "download_queue": ContentType.DOWNLOAD_QUEUE,
            }

            # Map level strings to RetrievalLevel enum
            level_to_retrieval = {
                "summary": RetrievalLevel.SUMMARY,
                "methods": RetrievalLevel.METHODS,
                "samples": RetrievalLevel.SAMPLES,
                "platform": RetrievalLevel.PLATFORM,
                "metadata": RetrievalLevel.FULL,
                "github": None,  # Special case, handle separately
                "validation": None,  # Special case for download_queue
                "strategy": None,  # Special case for download_queue
            }

            # Validate detail level using enum keys
            if level not in level_to_retrieval:
                valid = list(level_to_retrieval.keys())
                return f"Error: Invalid detail level '{level}'. Valid options: {', '.join(valid)}"

            # Validate workspace if provided
            if workspace and workspace not in workspace_to_content_type:
                valid_ws = list(workspace_to_content_type.keys())
                return f"Error: Invalid workspace '{workspace}'. Valid options: {', '.join(valid_ws)}"

            # List mode: Use service instead of manual scanning
            if identifier is None:
                # Special handling for download_queue workspace
                if workspace == "download_queue":
                    entries = workspace_service.list_download_queue_entries(
                        status_filter=status_filter
                    )

                    if not entries:
                        if status_filter:
                            return f"No download queue entries found with status '{status_filter}'."
                        return "Download queue is empty."

                    # Format response based on level
                    if level == "summary":
                        response = f"## Download Queue ({len(entries)} entries)\n\n"
                        for entry in entries:
                            response += (
                                f"- **{entry['entry_id']}**: {entry['dataset_id']} "
                            )
                            response += (
                                f"({entry['status']}, priority {entry['priority']})\n"
                            )
                            if entry.get("modality_name"):
                                response += (
                                    f"  └─> Loaded as: {entry['modality_name']}\n"
                                )
                        return response

                    elif level == "metadata":
                        # Full details for all entries
                        response = f"## Download Queue Entries ({len(entries)})\n\n"
                        for entry in entries:
                            response += _format_queue_entry_full(entry) + "\n\n"
                        return response

                logger.info("Listing all cached workspace content")

                # Determine content type filter
                content_type_filter = None
                if workspace:
                    content_type_filter = workspace_to_content_type[workspace]

                # Use service to list content (replaces manual glob + JSON reading)
                all_cached = workspace_service.list_content(
                    content_type=content_type_filter
                )

                if not all_cached:
                    filter_msg = f" in workspace '{workspace}'" if workspace else ""
                    return f"No cached content found{filter_msg}. Use write_to_workspace() to cache content first."

                # Format list response (same output format)
                response = f"## Cached Workspace Content ({len(all_cached)} items)\n\n"
                for item in all_cached:
                    response += f"- **{item['identifier']}**\n"
                    response += (
                        f"  - Workspace: {item.get('_content_type', 'unknown')}\n"
                    )
                    response += f"  - Type: {item.get('content_type', 'unknown')}\n"
                    response += f"  - Cached: {item.get('cached_at', 'unknown')}\n\n"
                return response

            # Handle download_queue retrieval mode
            if workspace == "download_queue":
                try:
                    entry = workspace_service.read_download_queue_entry(identifier)
                except FileNotFoundError as e:
                    return f"Error: {str(e)}"

                if level == "summary":
                    return _format_queue_entry_summary(entry)
                elif level == "metadata":
                    return json.dumps(entry, indent=2, default=str)
                elif level == "validation":
                    if entry.get("validation_result"):
                        return json.dumps(entry["validation_result"], indent=2)
                    return "No validation result available for this entry."
                elif level == "strategy":
                    if entry.get("recommended_strategy"):
                        return json.dumps(entry["recommended_strategy"], indent=2)
                    return "No recommended strategy available for this entry."

            # Retrieve mode: Handle "github" level specially (not in RetrievalLevel enum)
            if level == "github":
                # Try each content type if workspace not specified
                if workspace:
                    content_types_to_try = [workspace_to_content_type[workspace]]
                else:
                    content_types_to_try = list(ContentType)

                cached_content = None
                for content_type in content_types_to_try:
                    try:
                        # GitHub requires full content retrieval
                        cached_content = workspace_service.read_content(
                            identifier=identifier,
                            content_type=content_type,
                            level=RetrievalLevel.FULL,
                        )
                        break
                    except FileNotFoundError:
                        continue

                if not cached_content:
                    workspace_filter = (
                        f" in workspace '{workspace}'" if workspace else ""
                    )
                    return f"Error: Identifier '{identifier}' not found{workspace_filter}. Available content:\n{get_content_from_workspace(workspace=workspace)}"

                # Extract GitHub repos from data
                data = cached_content.get("data", {})
                if "github_repos" in data:
                    repos = data["github_repos"]
                    response = f"## GitHub Repositories for {identifier}\n\n"
                    response += f"**Found**: {len(repos)} repositories\n\n"
                    for repo in repos:
                        response += f"- {repo}\n"
                    return response
                else:
                    return f"No GitHub repositories found for '{identifier}'. This detail level is typically for publications with code."

            # Standard level retrieval using service with automatic filtering
            retrieval_level = level_to_retrieval[level]

            # Try each content type if workspace not specified
            if workspace:
                content_types_to_try = [workspace_to_content_type[workspace]]
            else:
                content_types_to_try = list(ContentType)

            cached_content = None
            found_content_type = None

            for content_type in content_types_to_try:
                try:
                    # Use service with level-based filtering (replaces manual if/elif)
                    cached_content = workspace_service.read_content(
                        identifier=identifier,
                        content_type=content_type,
                        level=retrieval_level,  # Service handles filtering automatically
                    )
                    found_content_type = content_type
                    break
                except FileNotFoundError:
                    continue

            if not cached_content:
                workspace_filter = f" in workspace '{workspace}'" if workspace else ""
                return f"Error: Identifier '{identifier}' not found{workspace_filter}. Available content:\n{get_content_from_workspace(workspace=workspace)}"

            # Format response based on level (service already filtered content)
            if level == "summary":
                data = cached_content.get("data", {})
                response = f"""## Summary: {identifier}

**Workspace**: {found_content_type.value if found_content_type else 'unknown'}
**Content Type**: {cached_content.get('content_type', 'unknown')}
**Cached At**: {cached_content.get('cached_at', 'unknown')}

**Data Overview**:
"""
                # Format data as key-value pairs
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            value_str = (
                                f"{type(value).__name__} with {len(value)} items"
                            )
                        else:
                            value_str = str(value)[:100]
                            if len(str(value)) > 100:
                                value_str += "..."
                        response += f"- **{key}**: {value_str}\n"
                return response

            elif level == "methods":
                # Service already filtered to methods fields
                if "methods" in cached_content:
                    return f"## Methods Section\n\n{cached_content['methods']}"
                else:
                    return f"No methods section found for '{identifier}'. This detail level is typically for publications."

            elif level == "samples":
                # Service already filtered to samples fields
                if "samples" in cached_content:
                    sample_list = cached_content["samples"]
                    response = f"## Sample IDs for {identifier}\n\n"
                    response += f"**Total Samples**: {len(sample_list)}\n\n"
                    response += "\n".join(f"- {sample}" for sample in sample_list[:50])
                    if len(sample_list) > 50:
                        response += f"\n\n... and {len(sample_list) - 50} more samples"
                    return response
                elif "obs_columns" in cached_content:
                    # For modalities, show obs_columns
                    return f"## Sample Information for {identifier}\n\n**N Observations**: {cached_content.get('n_obs')}\n**Obs Columns**: {', '.join(cached_content['obs_columns'])}"
                else:
                    return f"No sample information found for '{identifier}'. This detail level is typically for datasets."

            elif level == "platform":
                # Service already filtered to platform fields
                if "platform" in cached_content:
                    return f"## Platform Information\n\n{json.dumps(cached_content['platform'], indent=2)}"
                elif "var_columns" in cached_content:
                    # For modalities, show var_columns
                    return f"## Platform/Feature Information for {identifier}\n\n**N Variables**: {cached_content.get('n_vars')}\n**Var Columns**: {', '.join(cached_content['var_columns'])}"
                else:
                    return f"No platform information found for '{identifier}'. This detail level is typically for datasets."

            elif level == "metadata":
                # Service already filtered (full metadata)
                return f"## Full Metadata for {identifier}\n\n```json\n{json.dumps(cached_content, indent=2)}\n```"

            else:
                return f"Unsupported detail level: {level}"

        except Exception as e:
            logger.error(f"Error retrieving workspace content: {e}")
            return f"Error retrieving content from workspace: {str(e)}"

    def _format_queue_entry_summary(entry: dict) -> str:
        """Format queue entry as summary."""

        summary = f"## Download Queue Entry: {entry['entry_id']}\n\n"
        summary += f"**Dataset**: {entry['dataset_id']}\n"
        summary += f"**Database**: {entry['database']}\n"
        summary += f"**Status**: {entry['status']}\n"
        summary += f"**Priority**: {entry['priority']}/10\n"
        summary += f"**Created**: {entry['created_at']}\n"

        if entry.get("modality_name"):
            summary += f"**Modality**: {entry['modality_name']}\n"

        if entry.get("validation_result"):
            summary += "\n**Validation**: Available (use level='validation' to view)\n"

        if entry.get("recommended_strategy"):
            strategy = entry["recommended_strategy"]
            summary += f"\n**Recommended Strategy**: {strategy['strategy_name']}\n"
            summary += f"  - Confidence: {strategy['confidence']:.2%}\n"

        if entry.get("error_log"):
            summary += f"\n**Errors**: {len(entry['error_log'])} error(s) logged\n"

        return summary

    def _format_queue_entry_full(entry: dict) -> str:
        """Format queue entry with full details."""
        full = _format_queue_entry_summary(entry)

        # Add URLs if present
        urls = []
        if entry.get("matrix_url"):
            urls.append(f"- Matrix: {entry['matrix_url']}")
        if entry.get("h5_url"):
            urls.append(f"- H5: {entry['h5_url']}")
        if entry.get("raw_urls"):
            urls.append(f"- Raw: {len(entry['raw_urls'])} file(s)")

        if urls:
            full += "\n**URLs**:\n" + "\n".join(urls) + "\n"

        return full

    return get_content_from_workspace
