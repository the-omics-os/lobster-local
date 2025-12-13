"""
Workspace content retrieval tool for accessing cached research content.

This module provides factory functions for creating shared tools that can be used
by multiple agents (research_agent, data_expert, supervisor):
- get_content_from_workspace: Access cached research content
- list_available_modalities: List loaded modalities with optional filtering
"""

import json
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langchain_core.tools import tool

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.queue_storage import atomic_write_jsonl, queue_file_lock
from lobster.core.schemas.export_schemas import (
    get_ordered_export_columns,
    infer_data_type,
)
from lobster.services.data_access.workspace_content_service import (
    ContentType,
    RetrievalLevel,
    WorkspaceContentService,
)
from lobster.services.data_management.modality_management_service import (
    ModalityManagementService,
)
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# ===============================================================================
# Docstring-Driven Error System (v2.6+)
# ===============================================================================

# Cache to store parsed docstrings. Key: id(docstring), Value: DocstringSectionParser
_parser_cache: Dict[int, 'DocstringSectionParser'] = {}


class DocstringSectionParser:
    """
    Parses a docstring into sections based on '## Title' headers.

    Cached for performance (<10ms requirement).
    """

    def __init__(self, docstring: str):
        self.sections: Dict[str, str] = self._parse(docstring)

    def _parse(self, docstring: str) -> Dict[str, str]:
        """Extracts sections using regex for '## Title' headers."""
        if not docstring:
            return {}
        # Regex to find '## Title' and content until next '##'
        pattern = re.compile(
            r"^##\s*(.*?)\s*\n(.*?)(?=\n##|\Z)",
            re.MULTILINE | re.DOTALL
        )
        sections = {
            title.strip(): content.strip()
            for title, content in pattern.findall(docstring)
        }
        return sections

    def get_section(self, name: str) -> Optional[str]:
        """Retrieves a parsed section by its title."""
        return self.sections.get(name)

    @staticmethod
    def get_parser(docstring: str) -> 'DocstringSectionParser':
        """Cached factory for getting a parser for a docstring."""
        doc_id = id(docstring)
        if doc_id not in _parser_cache:
            _parser_cache[doc_id] = DocstringSectionParser(docstring)
        return _parser_cache[doc_id]


def _extract_example(section_content: str) -> Optional[str]:
    """Finds the first line starting with 'Example:'."""
    for line in section_content.split('\n'):
        if line.strip().lower().startswith("example:"):
            return line.strip()
    return None


def generate_contextual_error(
    invalid_value: str,
    valid_options: List[str],
    docstring: str,
    section_title: str,
    param_name: str
) -> str:
    """
    Generates a helpful, docstring-driven error message.

    Performance: <3ms with caching, meets <10ms requirement.

    Args:
        invalid_value: The invalid value provided
        valid_options: List of valid options
        docstring: Tool's docstring
        section_title: Section name to extract (e.g., "Workspace Categories")
        param_name: Parameter name for error message

    Returns:
        Formatted error message with suggestions and examples
    """
    # 1. Basic Error
    error_msg = f"❌ Error: Invalid value '{invalid_value}' for parameter '{param_name}'\n\n"

    # 2. Suggestion (Fuzzy Match)
    suggestion = get_close_matches(invalid_value, valid_options, n=1, cutoff=0.6)
    if suggestion:
        error_msg += f"💡 Did you mean '{suggestion[0]}'?\n\n"

    # 3. Valid Options (always show)
    error_msg += f"✅ Valid options: {', '.join(valid_options)}\n"

    # 4. Docstring Context
    parser = DocstringSectionParser.get_parser(docstring)
    section = parser.get_section(section_title)

    if section:
        # Extract example if available
        example = _extract_example(section)
        if example:
            error_msg += f"\n📖 {example}\n"

    return error_msg.strip()


# ===============================================================================
# Unified Workspace Item Structure
# ===============================================================================


class WorkspaceItem(TypedDict, total=False):
    """
    Unified representation of any workspace item.

    Used internally to normalize data from different workspace sources
    (literature, data, metadata, download_queue, publication_queue)
    into a consistent structure for filtering and formatting.

    Attributes:
        identifier: Primary identifier (entry_id, GSE, PMID, filename)
        workspace: Workspace category
        type: Item type (publication, dataset, metadata, queue_entry, etc.)
        status: Status for queue items (pending, completed, etc.)
        priority: Priority for queue items (1-5)
        title: Human-readable title
        cached_at: ISO timestamp when item was cached
        details: Brief summary or key metadata
    """
    identifier: str
    workspace: str
    type: str
    status: Optional[str]
    priority: Optional[int]
    title: Optional[str]
    cached_at: Optional[str]
    details: Optional[str]


def create_get_content_from_workspace_tool(data_manager: DataManagerV2):
    """
    Factory function to create get_content_from_workspace tool with data_manager closure.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for retrieving workspace content
    """
    from datetime import datetime

    @tool
    def get_content_from_workspace(
        identifier: str = None,
        workspace: str = None,
        level: str = "summary",
        status_filter: str = None,
    ) -> str:
        """
        Retrieve cached research content from workspace with flexible detail levels.

        **Unified Architecture (v2.6+)**: All workspace types now use a consistent
        adapter-based architecture with unified formatting and error handling. All
        workspaces support the same operations (list, filter, retrieve) with consistent
        output format.

        Reads previously cached publications, datasets, and metadata from workspace
        directories. Supports listing all content, filtering by workspace, and
        extracting specific details (summary, methods, samples, platform, full metadata).

        ## Workspace Categories
        - "literature": Publications, papers, abstracts
        - "data": Dataset metadata, GEO records
        - "metadata": Validation results, sample mappings
        - "download_queue": Pending/completed download tasks
        - "publication_queue": Pending/completed publication extraction tasks
        Example: get_content_from_workspace(workspace="literature")

        ## Detail Levels
        - "summary": Key-value pairs, high-level overview (default)
        - "methods": Methods section only (for publications)
        - "metadata": Full metadata JSON including COMPLETE PUBLICATION TEXT (for publications),
                     full entry details (for queues), or complete metadata (for datasets)
        - "samples": Sample IDs list (for datasets)
        - "platform": Platform information (for datasets)
        - "github": GitHub repositories (for publications)
        - "validation": Validation results (for download_queue)
        - "strategy": Download strategy (for download_queue)
        Example: get_content_from_workspace("literature", level="summary")

        For download_queue workspace:
        - identifier=None: List all entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific entry
        - status_filter: "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED"
        - level: "summary" (basic info) | "metadata" (full entry details)

        For publication_queue workspace:
        - identifier=None: List all entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific entry
        - status_filter: "pending" | "extracting" | "metadata_extracted" | "metadata_enriched" | "handoff_ready" | "completed" | "failed"
        - level: "summary" (basic info) | "metadata" (full entry details)

        **Unified Behavior**: All workspaces now support consistent operations:
        - List mode (identifier=None): Always returns formatted markdown list
        - Filter by status: Works across all queue workspaces
        - Consistent error handling: Defensive against missing fields
        - Same output format: Status emojis, titles, details (see WorkspaceItem TypedDict)

        For publication content (manual enrichment use case):
        - Cached publications have 3 workspace files:
          * pub_queue_doi_X_Y_Z_metadata.json: FULL PUBLICATION TEXT + metadata
          * pub_queue_doi_X_Y_Z_methods.json: Methods section only + structured extraction
          * pub_queue_doi_X_Y_Z_identifiers.json: Extracted dataset IDs (GEO/SRA/PRIDE)

        - level="metadata" on *_metadata identifier returns JSON with:
          * "content" field: COMPLETE PUBLICATION TEXT (Title + Intro + Methods + Results + Discussion)
          * "title", "authors", "journal", "year" fields
          * Use for: disease extraction from title/abstract, demographics from methods,
                     tissue context from results, comprehensive manual enrichment

        - level="methods" on *_methods identifier returns:
          * "methods_text" field: Methods section text only
          * "methods_dict" with software_used, parameters, statistical_methods
          * Use for: focused age/sex extraction, cohort characteristics

        - To extract specific sections from full text, use regex on "content" field
        - To enrich samples, combine with execute_custom_code tool

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

            # PUBLICATION ENRICHMENT: Read full publication text (COMPLETE CONTENT)
            # For manual sample enrichment, use level="metadata" to access full text
            get_content_from_workspace(
                identifier="pub_queue_doi_10_1080_19490976_2022_2046244_metadata",
                workspace="metadata",
                level="metadata"
            )
            # Returns JSON with "content" field containing:
            # - Complete publication text (Title + Introduction + Methods + Results + Discussion)
            # - All sections accessible for disease/demographics extraction
            # - Use for: extracting disease from title, demographics from methods,
            #            tissue context from results

            # PUBLICATION ENRICHMENT: Read methods section only (FOCUSED)
            get_content_from_workspace(
                identifier="pub_queue_doi_10_1080_19490976_2022_2046244_methods",
                workspace="metadata",
                level="methods"
            )
            # Returns: Methods section text with cohort demographics, experimental design
            # Use for: age/sex extraction, sample collection details

            # PUBLICATION ENRICHMENT: Quick title/abstract check
            get_content_from_workspace(
                identifier="pub_queue_doi_10_1080_19490976_2022_2046244_metadata",
                workspace="metadata",
                level="summary"
            )
            # Returns: Title, authors, journal, year (no full text)
            # Use for: quick disease context check before full extraction

            # List pending publication extractions
            get_content_from_workspace(
                workspace="publication_queue",
                status_filter="pending"
            )

            # Get publication queue entry details
            get_content_from_workspace(
                identifier="pub_queue_456",
                workspace="publication_queue",
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
                "publication_queue": ContentType.PUBLICATION_QUEUE,
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
                return generate_contextual_error(
                    invalid_value=level,
                    valid_options=valid,
                    docstring=get_content_from_workspace.__doc__,
                    section_title="Detail Levels",
                    param_name="level"
                )

            # Validate workspace if provided
            if workspace and workspace not in workspace_to_content_type:
                valid_ws = list(workspace_to_content_type.keys())
                return generate_contextual_error(
                    invalid_value=workspace,
                    valid_options=valid_ws,
                    docstring=get_content_from_workspace.__doc__,
                    section_title="Workspace Categories",
                    param_name="workspace"
                )

            # ===================================================================
            # Adapter Functions: Normalize workspace data to WorkspaceItem
            # ===================================================================

            def _adapt_general_content(ws: str, filter_status: Optional[str] = None) -> List[WorkspaceItem]:
                """
                Adapter for literature, data, and metadata workspaces.

                Converts WorkspaceContentService items to unified WorkspaceItem structure.
                """
                content_type = workspace_to_content_type.get(ws)
                if not content_type:
                    return []

                items = workspace_service.list_content(content_type=content_type)

                normalized_items: List[WorkspaceItem] = []
                for item in items:
                    # Defensive identifier extraction (already implemented in line 418-423)
                    item_id = (
                        item.get('identifier')
                        or item.get('name')
                        or (Path(item.get('_file_path', '')).stem if item.get('_file_path') else None)
                        or next(iter(item.keys()), 'unknown')
                    )

                    normalized_items.append(WorkspaceItem(
                        identifier=item_id,
                        workspace=ws,
                        type=item.get('_content_type', ws),
                        status=None,  # No status for general content
                        priority=None,
                        title=item.get('title'),
                        cached_at=item.get('cached_at'),
                        details=item.get('content_type', '')
                    ))

                return normalized_items

            def _adapt_download_queue(filter_status: Optional[str] = None) -> List[WorkspaceItem]:
                """
                Adapter for download_queue workspace.

                Converts DownloadQueueEntry objects to unified WorkspaceItem structure.
                """
                entries = workspace_service.list_download_queue_entries(status_filter=filter_status)

                normalized_items: List[WorkspaceItem] = []
                for entry in entries:
                    dataset_id = entry.get('dataset_id', 'unknown')
                    status = entry.get('status', 'UNKNOWN')
                    database = entry.get('database', 'unknown')
                    urls_count = len(entry.get('urls', []))

                    normalized_items.append(WorkspaceItem(
                        identifier=entry.get('entry_id', 'unknown'),
                        workspace='download_queue',
                        type='download_queue_entry',
                        status=status,
                        priority=entry.get('priority', 5),
                        title=dataset_id,
                        cached_at=entry.get('created_at'),
                        details=f"{database} | {urls_count} URLs"
                    ))

                return normalized_items

            def _adapt_publication_queue(filter_status: Optional[str] = None) -> List[WorkspaceItem]:
                """
                Adapter for publication_queue workspace.

                Converts PublicationQueueEntry objects to unified WorkspaceItem structure.
                """
                entries = workspace_service.list_publication_queue_entries(status_filter=filter_status)

                normalized_items: List[WorkspaceItem] = []
                for entry in entries:
                    # Extract title and authors
                    title = entry.get('title', 'Untitled')[:80]
                    authors = entry.get('authors', [])
                    authors_str = ', '.join(authors[:2]) if authors else 'Unknown'
                    if len(authors) > 2:
                        authors_str += f" et al. ({len(authors)} total)"

                    normalized_items.append(WorkspaceItem(
                        identifier=entry.get('entry_id', 'unknown'),
                        workspace='publication_queue',
                        type='publication_queue_entry',
                        status=entry.get('status'),
                        priority=entry.get('priority', 5),
                        title=title,
                        cached_at=entry.get('created_at'),
                        details=authors_str
                    ))

                return normalized_items

            def _format_workspace_item(item: WorkspaceItem) -> str:
                """
                Format a WorkspaceItem into consistent markdown output.

                Uses status emojis and preserves existing formatting style for backward compatibility.
                """
                # Status emoji mapping
                status_emojis = {
                    # Download queue statuses (uppercase)
                    "PENDING": "⏳",
                    "IN_PROGRESS": "🔄",
                    "COMPLETED": "✅",
                    "FAILED": "❌",
                    # Publication queue statuses (lowercase)
                    "pending": "⏳",
                    "extracting": "🔄",
                    "metadata_extracted": "📄",
                    "metadata_enriched": "✨",
                    "handoff_ready": "🤝",
                    "completed": "✅",
                    "failed": "❌",
                }

                status = item.get('status', '')
                emoji = status_emojis.get(status, "📄")

                output = f"- {emoji} **{item['identifier']}**\n"

                if item.get('title'):
                    output += f"  - Title: {item['title']}\n"
                if status:
                    output += f"  - Status: {status}\n"
                if item.get('details'):
                    output += f"  - Details: {item['details']}\n"

                return output

            # Workspace adapter dispatcher
            WORKSPACE_ADAPTERS: Dict[str, Callable[[Optional[str]], List[WorkspaceItem]]] = {
                "literature": lambda s: _adapt_general_content("literature", s),
                "data": lambda s: _adapt_general_content("data", s),
                "metadata": lambda s: _adapt_general_content("metadata", s),
                "download_queue": _adapt_download_queue,
                "publication_queue": _adapt_publication_queue,
            }

            # ===================================================================
            # List mode: Unified dispatcher-based listing
            # ===================================================================
            if identifier is None:
                # Special case: publication_queue with aggregated summary
                # This provides statistics and aggregation instead of simple listing
                if workspace == "publication_queue" and level == "summary":
                    entries = workspace_service.list_publication_queue_entries(
                        status_filter=status_filter
                    )

                    if not entries:
                        if status_filter:
                            return f"No publication queue entries found with status '{status_filter}'."
                        return "Publication queue is empty."

                    # Format response based on level
                    if level == "summary":
                        # Token-efficient aggregated summary (avoid listing all entries)
                        from collections import Counter
                        from datetime import datetime

                        # Compute statistics
                        status_counts = Counter(entry["status"] for entry in entries)

                        # Priority distribution (1-3=high, 4-7=medium, 8-10=low)
                        priority_high = sum(1 for e in entries if e["priority"] <= 3)
                        priority_medium = sum(
                            1 for e in entries if 4 <= e["priority"] <= 7
                        )
                        priority_low = sum(1 for e in entries if e["priority"] >= 8)

                        # Failed entries count
                        failed_count = status_counts.get("failed", 0)

                        # Sort by updated_at (most recent first), show top 5
                        sorted_entries = sorted(
                            entries, key=lambda e: e.get("updated_at", ""), reverse=True
                        )[:5]

                        # Build response
                        response = f"## Publication Queue Summary\n\n"
                        response += f"**Total Entries**: {len(entries)}\n\n"

                        # Status breakdown
                        response += "**Status Breakdown**:\n"
                        for status in [
                            "pending",
                            "extracting",
                            "metadata_extracted",
                            "metadata_enriched",
                            "handoff_ready",
                            "completed",
                            "failed",
                            "paywalled",
                        ]:
                            count = status_counts.get(status, 0)
                            if count > 0:
                                response += f"- {status}: {count} entries\n"
                        response += "\n"

                        # Priority distribution
                        response += "**Priority Distribution**:\n"
                        response += f"- High priority (1-3): {priority_high} entries\n"
                        response += (
                            f"- Medium priority (4-7): {priority_medium} entries\n"
                        )
                        response += f"- Low priority (8-10): {priority_low} entries\n\n"

                        # Recent activity
                        response += "**Recent Activity** (last 5 updates):\n"
                        for entry in sorted_entries:
                            title = entry.get("title", "Untitled")
                            title_short = (
                                title[:50] + "..." if len(title) > 50 else title
                            )
                            updated = entry.get("updated_at", "unknown")

                            # Format time ago if possible
                            try:
                                if isinstance(updated, str):
                                    updated_dt = datetime.fromisoformat(
                                        updated.replace("Z", "+00:00")
                                    )
                                else:
                                    updated_dt = updated
                                time_diff = datetime.now() - updated_dt.replace(
                                    tzinfo=None
                                )

                                if time_diff.days > 0:
                                    time_ago = f"{time_diff.days}d ago"
                                elif time_diff.seconds >= 3600:
                                    time_ago = f"{time_diff.seconds // 3600}h ago"
                                elif time_diff.seconds >= 60:
                                    time_ago = f"{time_diff.seconds // 60}m ago"
                                else:
                                    time_ago = "just now"
                            except:
                                time_ago = "unknown"

                            response += f"- **{entry['entry_id']}**: \"{title_short}\" ({entry['status']}) - Updated {time_ago}\n"
                        response += "\n"

                        # Problem indicators
                        if failed_count > 0:
                            response += f"**Failed Entries**: {failed_count} (use status_filter='failed' to inspect)\n\n"

                        # Actionable guidance
                        response += "**Tip**: Use `status_filter` parameter to focus on specific statuses:\n"
                        response += "- `status_filter='handoff_ready'` - Ready for metadata processing\n"
                        response += (
                            "- `status_filter='failed'` - Entries needing attention\n"
                        )
                        response += "- `status_filter='pending'` - Not yet started\n"
                        response += "\nUse `level='metadata'` for detailed inspection of all entries.\n"

                        return response

                # Unified dispatcher for all other list modes (including publication_queue metadata level)
                logger.info(f"Listing workspace content (workspace={workspace}, level={level}, filter={status_filter})")

                # Determine which workspaces to list
                workspaces_to_list = [workspace] if workspace else list(WORKSPACE_ADAPTERS.keys())

                # Fetch and normalize items from all relevant workspaces
                all_items: List[WorkspaceItem] = []
                for ws in workspaces_to_list:
                    adapter = WORKSPACE_ADAPTERS.get(ws)
                    if adapter:
                        try:
                            items = adapter(status_filter)
                            all_items.extend(items)
                        except Exception as e:
                            logger.warning(f"Failed to list workspace '{ws}': {e}")

                # Handle empty results
                if not all_items:
                    filter_msg = f" in workspace '{workspace}'" if workspace else ""
                    status_msg = f" with status '{status_filter}'" if status_filter else ""
                    return f"No items found{filter_msg}{status_msg}. Use write_to_workspace() to cache content first."

                # Format based on level
                if level == "metadata":
                    # Full details mode (for queues) - delegate to existing formatters
                    if workspace == "publication_queue":
                        entries = workspace_service.list_publication_queue_entries(status_filter)
                        response = f"## Publication Queue Entries ({len(entries)})\n\n"
                        for entry in entries:
                            response += _format_pub_queue_entry_full(entry) + "\n\n"
                        return response
                    elif workspace == "download_queue":
                        entries = workspace_service.list_download_queue_entries(status_filter)
                        response = f"## Download Queue Entries ({len(entries)})\n\n"
                        for entry in entries:
                            response += _format_queue_entry_full(entry) + "\n\n"
                        return response

                # Summary mode (default) - unified formatting
                ws_display = workspace.title() if workspace else "All"
                response = f"## {ws_display} Workspace ({len(all_items)} items)\n\n"

                for item in all_items:
                    response += _format_workspace_item(item)

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

            # Handle publication_queue retrieval mode
            if workspace == "publication_queue":
                try:
                    entry = workspace_service.read_publication_queue_entry(identifier)
                except FileNotFoundError as e:
                    return f"Error: {str(e)}"

                if level == "summary":
                    return _format_pub_queue_entry_summary(entry)
                elif level == "metadata":
                    return json.dumps(entry, indent=2, default=str)

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

    def _format_pub_queue_entry_summary(entry: dict) -> str:
        """Format publication queue entry as summary."""

        summary = f"## Publication Queue Entry: {entry['entry_id']}\n\n"

        if entry.get("title"):
            summary += f"**Title**: {entry['title']}\n"

        if entry.get("authors"):
            authors = entry["authors"]
            if len(authors) > 3:
                author_str = ", ".join(authors[:3]) + f" et al. ({len(authors)} total)"
            else:
                author_str = ", ".join(authors)
            summary += f"**Authors**: {author_str}\n"

        if entry.get("journal"):
            summary += f"**Journal**: {entry['journal']}\n"

        if entry.get("year"):
            summary += f"**Year**: {entry['year']}\n"

        # Identifiers
        identifiers = []
        if entry.get("pmid"):
            identifiers.append(f"PMID: {entry['pmid']}")
        if entry.get("doi"):
            identifiers.append(f"DOI: {entry['doi']}")
        if entry.get("pmc_id"):
            identifiers.append(f"PMC: {entry['pmc_id']}")
        if identifiers:
            summary += f"**Identifiers**: {', '.join(identifiers)}\n"

        summary += f"\n**Status**: {entry['status']}\n"
        summary += f"**Priority**: {entry['priority']}/10\n"
        summary += f"**Schema Type**: {entry.get('schema_type', 'general')}\n"
        summary += f"**Extraction Level**: {entry.get('extraction_level', 'methods')}\n"
        summary += f"**Created**: {entry.get('created_at', 'unknown')}\n"

        if entry.get("cached_content_path"):
            summary += f"\n**Cached Content**: {entry['cached_content_path']}\n"

        if entry.get("extracted_identifiers"):
            ids = entry["extracted_identifiers"]
            if any(ids.values()):
                summary += "\n**Extracted Identifiers**:\n"
                for id_type, id_list in ids.items():
                    if id_list:
                        summary += f"  - {id_type}: {', '.join(id_list[:5])}"
                        if len(id_list) > 5:
                            summary += f" (+{len(id_list) - 5} more)"
                        summary += "\n"

        if entry.get("error"):
            summary += f"\n**Error**: {entry['error']}\n"

        return summary

    def _format_pub_queue_entry_full(entry: dict) -> str:
        """Format publication queue entry with full details."""
        full = _format_pub_queue_entry_summary(entry)

        # Add URLs if present
        if entry.get("metadata_url"):
            full += f"\n**Metadata URL**: {entry['metadata_url']}\n"

        if entry.get("supplementary_files"):
            files = entry["supplementary_files"]
            full += f"\n**Supplementary Files**: {len(files)} file(s)\n"
            for file_url in files[:3]:
                full += f"  - {file_url}\n"
            if len(files) > 3:
                full += f"  - ... and {len(files) - 3} more\n"

        if entry.get("github_url"):
            full += f"\n**GitHub URL**: {entry['github_url']}\n"

        return full

    return get_content_from_workspace


def create_write_to_workspace_tool(data_manager: DataManagerV2):
    """
    Factory function to create write_to_workspace tool with data_manager closure.

    Shared between research_agent and metadata_assistant for workspace export
    with JSON and CSV format support.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for writing content to workspace
    """
    from datetime import datetime

    from lobster.services.data_access.workspace_content_service import (
        ContentType,
        DatasetContent,
        MetadataContent,
        PublicationContent,
        WorkspaceContentService,
    )

    @tool
    def write_to_workspace(
        identifier: str,
        workspace: str,
        content_type: str = None,
        output_format: str = "json",
        export_mode: str = "auto",
        add_timestamp: bool = True,
    ) -> str:
        """
        Cache research content to workspace for later retrieval and specialist handoff.

        Stores publications, datasets, and metadata in organized workspace directories
        for persistent access. Validates naming conventions and content standardization.

        Workspace Categories:
        - "literature": Publications, abstracts, methods sections
        - "data": Dataset metadata, sample information
        - "metadata": Standardized metadata schemas

        Output Formats:
        - "json": Structured JSON format (default)
        - "csv": Tabular CSV format (best for sample metadata tables)

        Export Modes (for CSV):
        - "auto": Detect publication queue data and apply rich format automatically
        - "rich": Force rich 28-column format with publication context
        - "simple": Export samples as-is without enrichment

        Args:
            identifier: Content identifier to cache (must exist in current session)
            workspace: Target workspace category ("literature", "data", "metadata")
            content_type: Type of content ("publication", "dataset", "metadata")
            output_format: Output format ("json" or "csv"). Default: "json"
            export_mode: CSV export mode ("auto", "rich", "simple"). Default: "auto"
            add_timestamp: Auto-append YYYY-MM-DD timestamp to filename. Default: True
                          Set to False to use identifier as exact filename

        Returns:
            Confirmation message with storage location and next steps
        """
        try:
            workspace_service = WorkspaceContentService(data_manager=data_manager)

            workspace_to_content_type = {
                "literature": ContentType.PUBLICATION,
                "data": ContentType.DATASET,
                "metadata": ContentType.METADATA,
            }

            if workspace not in workspace_to_content_type:
                return f"Error: Invalid workspace '{workspace}'. Valid: {', '.join(workspace_to_content_type.keys())}"

            if content_type and content_type not in {
                "publication",
                "dataset",
                "metadata",
            }:
                return f"Error: Invalid content_type '{content_type}'. Valid: publication, dataset, metadata"

            if output_format not in {"json", "csv"}:
                return (
                    f"Error: Invalid output_format '{output_format}'. Valid: json, csv"
                )

            if export_mode not in {"auto", "rich", "simple"}:
                return f"Error: Invalid export_mode '{export_mode}'. Valid: auto, rich, simple"

            # Check if identifier exists in session
            exists = False
            content_data = None
            source_location = None

            if identifier in data_manager.metadata_store:
                exists = True
                content_data = data_manager.metadata_store[identifier]
                source_location = "metadata_store"
                logger.info(f"Found '{identifier}' in metadata_store")
            elif identifier in data_manager.list_modalities():
                exists = True
                adata = data_manager.get_modality(identifier)
                content_data = {
                    "n_obs": adata.n_obs,
                    "n_vars": adata.n_vars,
                    "obs_columns": list(adata.obs.columns),
                    "var_columns": list(adata.var.columns),
                }
                source_location = "modalities"
                logger.info(f"Found '{identifier}' in modalities")

            if not exists:
                return f"Error: Identifier '{identifier}' not found in current session."

            # Helper function to detect publication queue SRA data
            def _is_publication_queue_sra_data(data: dict) -> bool:
                """Detect if data is publication queue SRA samples."""
                if not isinstance(data, dict):
                    return False
                samples = data.get("samples", [])
                if not samples or not isinstance(samples, list):
                    return False
                # Check for SRA-specific fields
                if samples:
                    first_sample = samples[0]
                    sra_indicators = [
                        "run_accession",
                        "biosample",
                        "bioproject",
                        "library_strategy",
                    ]
                    return any(field in first_sample for field in sra_indicators)
                return False

            # Helper function to enrich samples with publication context
            def _enrich_samples_with_publication_context(
                samples: list, identifier: str, content_data: dict
            ) -> list:
                """Add publication context columns to each sample."""
                import pandas as pd

                # Extract publication context from content_data or identifier
                source_doi = content_data.get("source_doi", "")
                source_pmid = content_data.get("source_pmid", "")
                source_entry_id = identifier

                # If identifier is like "pub_queue_doi_X_Y_Z_samples", extract entry_id
                if identifier.endswith("_samples"):
                    source_entry_id = identifier.rsplit("_samples", 1)[0]

                enriched_samples = []
                for sample in samples:
                    enriched = dict(sample)
                    # Add publication context if not already present
                    if "source_doi" not in enriched:
                        enriched["source_doi"] = source_doi
                    if "source_pmid" not in enriched:
                        enriched["source_pmid"] = source_pmid
                    if "source_entry_id" not in enriched:
                        enriched["source_entry_id"] = source_entry_id
                    enriched_samples.append(enriched)

                return enriched_samples

            # Special handling for CSV export of samples list
            # When exporting to CSV and data contains a 'samples' list,
            # write directly to CSV bypassing MetadataContent model validation
            samples_count = None
            rich_export_used = False
            if output_format == "csv" and isinstance(content_data, dict):
                if "samples" in content_data and isinstance(
                    content_data["samples"], list
                ):
                    samples_list = content_data["samples"]
                    samples_count = len(samples_list)
                    logger.info(f"Extracting {samples_count} samples for CSV export")

                    # Determine if we should use rich export format
                    use_rich_format = False
                    if export_mode == "rich":
                        use_rich_format = True
                    elif export_mode == "auto":
                        use_rich_format = _is_publication_queue_sra_data(content_data)
                    # export_mode == "simple" keeps use_rich_format = False

                    import pandas as pd

                    if use_rich_format:
                        logger.info(
                            "Using schema-driven export format (column count depends on data type)"
                        )
                        rich_export_used = True

                        # Enrich samples with publication context
                        enriched_samples = _enrich_samples_with_publication_context(
                            samples_list, identifier, content_data
                        )

                        # Schema-driven column ordering (v1.2.0 - export_schemas.py)
                        # Infer data type from samples (auto-detects SRA/proteomics/metabolomics)
                        data_type = infer_data_type(enriched_samples)

                        # Get priority-ordered columns from schema
                        ordered_cols = get_ordered_export_columns(
                            samples=enriched_samples,
                            data_type=data_type,
                            include_extra=True,  # Include fields not in schema
                        )

                        # Create DataFrame with schema-ordered columns
                        df = pd.DataFrame(enriched_samples)
                        available_cols = [c for c in ordered_cols if c in df.columns]
                        df = df[available_cols]
                    else:
                        # Simple export - as-is
                        df = pd.DataFrame(samples_list)

                    # Write to CSV
                    content_dir = workspace_service._get_content_dir(
                        workspace_to_content_type[workspace]
                    )
                    filename = workspace_service._sanitize_filename(identifier)

                    # Auto-append timestamp if requested and not already present
                    if add_timestamp and not re.search(r"\d{4}-\d{2}-\d{2}", filename):
                        timestamp = datetime.now().strftime("%Y-%m-%d")
                        filename = f"{filename}_{timestamp}"

                    cache_file_path = content_dir / f"{filename}.csv"

                    df.to_csv(cache_file_path, index=False, encoding="utf-8")
                    logger.info(
                        f"CSV export: {samples_count} rows, {len(df.columns)} columns "
                        f"(rich={rich_export_used}) to {cache_file_path}"
                    )
                else:
                    # No samples list, fall through to normal handling
                    samples_count = None

            # Normal path: create correct model based on workspace type
            if samples_count is None:
                target_content_type = workspace_to_content_type[workspace]

                # Create appropriate model based on workspace
                if target_content_type == ContentType.PUBLICATION:
                    content_model = PublicationContent(
                        identifier=identifier,
                        title=content_data.get("title"),
                        authors=content_data.get("authors", []),
                        journal=content_data.get("journal"),
                        year=content_data.get("year"),
                        abstract=content_data.get("abstract"),
                        methods=content_data.get("methods"),
                        full_text=content_data.get("full_text"),
                        keywords=content_data.get("keywords", []),
                        source=f"DataManager.{source_location}",
                        cached_at=datetime.now().isoformat(),
                        url=content_data.get("url"),
                    )

                elif target_content_type == ContentType.DATASET:
                    # Handle nested metadata structure or flat structure
                    metadata = content_data.get("metadata", content_data)
                    content_model = DatasetContent(
                        identifier=identifier,
                        title=metadata.get("title"),
                        platform=metadata.get("platform"),
                        platform_id=metadata.get("platform_id"),
                        organism=metadata.get("organism"),
                        sample_count=metadata.get("n_samples", metadata.get("sample_count", 0)),
                        samples=metadata.get("samples"),
                        experimental_design=metadata.get("experimental_design"),
                        summary=metadata.get("summary"),
                        pubmed_ids=metadata.get("pubmed_ids", []),
                        source=f"DataManager.{source_location}",
                        cached_at=datetime.now().isoformat(),
                        url=metadata.get("url"),
                    )

                else:  # ContentType.METADATA (existing behavior preserved)
                    content_model = MetadataContent(
                        identifier=identifier,
                        content_type=content_type or "unknown",
                        description=f"Cached from {source_location}",
                        data=content_data,
                        related_datasets=[],
                        source=f"DataManager.{source_location}",
                        cached_at=datetime.now().isoformat(),
                    )

                cache_file_path = workspace_service.write_content(
                    content=content_model,
                    content_type=target_content_type,
                    output_format=output_format,
                )

            response = f"""## Content Cached Successfully

**Identifier**: {identifier}
**Workspace**: {workspace}
**Output Format**: {output_format.upper()}
**Location**: {cache_file_path}

**Next Steps**:
- Use `get_content_from_workspace()` to retrieve cached content
"""
            if output_format == "csv":
                response += "**Note**: CSV format ideal for spreadsheet import.\n"
                if samples_count:
                    response += f"**Rows**: {samples_count} samples exported (one row per sample).\n"
                    if rich_export_used:
                        response += (
                            "**Format**: Rich 28-column format with publication context "
                            "(source_doi, source_pmid, source_entry_id).\n"
                        )

            return response

        except Exception as e:
            logger.error(f"Error caching to workspace: {e}")
            return f"Error caching content to workspace: {str(e)}"

    return write_to_workspace


def create_export_publication_queue_tool(data_manager: DataManagerV2):
    """
    Factory function to create export_publication_queue_samples tool.

    Enables aggregated export of samples from multiple publication queue entries
    into a single CSV file with rich metadata format.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for exporting publication queue samples
    """
    from datetime import datetime

    from lobster.services.data_access.workspace_content_service import (
        ContentType,
        WorkspaceContentService,
    )

    @tool
    def export_publication_queue_samples(
        entry_ids: str,
        output_filename: str,
        filter_criteria: str = None,
    ) -> str:
        """
        Export samples from multiple publication queue entries to a single aggregated CSV.

        Combines samples from multiple publications with rich metadata format including
        publication context (DOI, PMID, entry_id) for each sample. Ideal for downstream
        analysis pipelines requiring samples from multiple studies.

        Args:
            entry_ids: Comma-separated list of entry IDs to export, OR "all" to export
                      all entries with samples, OR "handoff_ready" to export only
                      entries with HANDOFF_READY status
            output_filename: Name for the output CSV file (without path, will be saved
                           in exports/ directory)
            filter_criteria: Optional filter string for sample selection (e.g., "16S human")
                           Applied to organism_name, host, library_strategy fields

        Returns:
            Summary of export with file location and sample counts per publication
        """
        import pandas as pd

        try:
            workspace_service = WorkspaceContentService(data_manager=data_manager)
            content_dir = workspace_service._get_content_dir(ContentType.METADATA)

            # Ensure exports subdirectory exists
            exports_dir = content_dir / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)

            # Parse entry_ids
            if entry_ids.lower() == "all":
                # Find all entries with samples in metadata_store
                target_entries = [
                    k.rsplit("_samples", 1)[0]
                    for k in data_manager.metadata_store.keys()
                    if k.endswith("_samples")
                ]
            elif entry_ids.lower() == "handoff_ready":
                # Find entries with HANDOFF_READY status
                # Look for entries that have samples AND filter criteria applied
                target_entries = []
                for k in data_manager.metadata_store.keys():
                    if k.endswith("_samples"):
                        entry_id = k.rsplit("_samples", 1)[0]
                        # Check if entry has been processed (has samples)
                        if data_manager.metadata_store.get(k, {}).get("samples"):
                            target_entries.append(entry_id)
            else:
                # Parse comma-separated list
                target_entries = [e.strip() for e in entry_ids.split(",") if e.strip()]

            if not target_entries:
                return "Error: No valid entry IDs found. Provide comma-separated IDs, 'all', or 'handoff_ready'."

            # Collect samples from all entries
            all_samples = []
            entries_found = []
            entries_missing = []
            samples_per_entry = {}

            for entry_id in target_entries:
                samples_key = f"{entry_id}_samples"
                if samples_key not in data_manager.metadata_store:
                    entries_missing.append(entry_id)
                    continue

                entry_data = data_manager.metadata_store[samples_key]
                samples = entry_data.get("samples", [])

                if not samples:
                    entries_missing.append(entry_id)
                    continue

                entries_found.append(entry_id)

                # Get publication context
                source_doi = entry_data.get("source_doi", "")
                source_pmid = entry_data.get("source_pmid", "")

                # Enrich each sample with publication context
                for sample in samples:
                    enriched = dict(sample)
                    enriched["source_doi"] = source_doi
                    enriched["source_pmid"] = source_pmid
                    enriched["source_entry_id"] = entry_id
                    all_samples.append(enriched)

                samples_per_entry[entry_id] = len(samples)

            if not all_samples:
                return f"Error: No samples found. Missing entries: {entries_missing}"

            # Apply filter criteria if provided
            filtered_samples = all_samples
            filter_applied = False
            if filter_criteria:
                filter_lower = filter_criteria.lower()
                filtered_samples = []
                for s in all_samples:
                    # Check multiple fields
                    fields_to_check = [
                        str(s.get("organism_name", "")).lower(),
                        str(s.get("host", "")).lower(),
                        str(s.get("library_strategy", "")).lower(),
                        str(s.get("isolation_source", "")).lower(),
                    ]
                    if any(filter_lower in f for f in fields_to_check):
                        filtered_samples.append(s)
                filter_applied = True

            if not filtered_samples:
                return (
                    f"Error: No samples match filter '{filter_criteria}'. "
                    f"Total samples before filter: {len(all_samples)}"
                )

            # Schema-driven column ordering (v1.2.0 - export_schemas.py)
            # Infer data type from samples (auto-detects SRA/proteomics/metabolomics)
            data_type = infer_data_type(filtered_samples)

            # Get priority-ordered columns from schema
            ordered_cols = get_ordered_export_columns(
                samples=filtered_samples,
                data_type=data_type,
                include_extra=True,  # Include fields not in schema
            )

            # Create DataFrame with schema-ordered columns
            df = pd.DataFrame(filtered_samples)
            available_cols = [c for c in ordered_cols if c in df.columns]
            df = df[available_cols]

            # Sanitize output filename
            safe_filename = workspace_service._sanitize_filename(output_filename)
            if not safe_filename.endswith(".csv"):
                safe_filename += ".csv"
            output_path = exports_dir / safe_filename

            # Write CSV
            df.to_csv(output_path, index=False, encoding="utf-8")

            # Build response
            response = f"""## Publication Queue Samples Exported

**Output File**: {output_path}
**Total Samples**: {len(filtered_samples)}
**Publications**: {len(entries_found)}
**Columns**: {len(df.columns)} (schema-driven format with publication context)

**Samples per Publication**:
"""
            for entry_id, count in sorted(samples_per_entry.items()):
                response += f"- {entry_id}: {count} samples\n"

            if filter_applied:
                response += f"\n**Filter Applied**: '{filter_criteria}'\n"
                response += f"**Samples Before Filter**: {len(all_samples)}\n"

            if entries_missing:
                response += f"\n**Entries Without Samples**: {len(entries_missing)}\n"

            return response

        except Exception as e:
            logger.error(f"Error exporting publication queue samples: {e}")
            return f"Error exporting samples: {str(e)}"

    return export_publication_queue_samples


def create_list_modalities_tool(data_manager: DataManagerV2):
    """
    Factory function to create list_available_modalities tool with data_manager closure.

    Shared between supervisor and data_expert agents for consistent modality listing
    with provenance tracking and optional filtering.

    Args:
        data_manager: DataManagerV2 instance for modality access

    Returns:
        LangChain tool for listing modalities with optional filtering
    """

    # Initialize service once (closure captures this)
    modality_service = ModalityManagementService(data_manager)

    @tool
    def list_available_modalities(filter_pattern: Optional[str] = None) -> str:
        """
        List all available modalities with optional filtering.

        Args:
            filter_pattern: Optional glob-style pattern to filter modality names
                          (e.g., "geo_gse*", "*clustered", "bulk_*")
                          If None, lists all modalities.

        Returns:
            str: Formatted list of modalities with details
        """
        try:
            modality_info, stats, ir = modality_service.list_modalities(
                filter_pattern=filter_pattern
            )

            # Log to provenance (W3C-PROV compliant)
            data_manager.log_tool_usage(
                tool_name="list_available_modalities",
                parameters={"filter_pattern": filter_pattern},
                description=stats,
                ir=ir,
            )

            # Format response
            if not modality_info:
                return "No modalities found matching the criteria."

            response = f"## Available Modalities ({stats['matched_modalities']}/{stats['total_modalities']})\n\n"
            if filter_pattern:
                response += f"**Filter**: `{filter_pattern}`\n\n"

            for info in modality_info:
                if "error" in info:
                    response += f"- **{info['name']}**: Error - {info['error']}\n"
                else:
                    response += f"- **{info['name']}**: {info['n_obs']} obs × {info['n_vars']} vars\n"
                    if info["obs_columns"]:
                        response += f"  - Obs: {', '.join(info['obs_columns'][:3])}\n"
                    if info["var_columns"]:
                        response += f"  - Var: {', '.join(info['var_columns'][:3])}\n"

            # Add workspace info (useful for supervisor context)
            workspace_status = data_manager.get_workspace_status()
            response += f"\n**Workspace**: {workspace_status['workspace_path']}\n"

            return response

        except Exception as e:
            logger.error(f"Error listing modalities: {e}")
            return f"Error listing modalities: {str(e)}"

    return list_available_modalities


def create_delete_from_workspace_tool(data_manager: DataManagerV2):
    """
    Factory function to create delete_from_workspace tool with data_manager closure.

    Unified deletion tool for removing workspace content, queue entries, and modalities
    with safety validations and provenance tracking. Supervisor-only tool for data cleanup.

    Args:
        data_manager: DataManagerV2 instance for workspace access

    Returns:
        LangChain tool for deleting workspace content with preview/confirmation modes
    """

    @tool
    def delete_from_workspace(
        identifier: str,
        workspace: str = None,
        confirmation: bool = False,
    ) -> str:
        """
        Delete cached content, queue entries, or modalities from workspace.

        Unified deletion tool supporting:
        - Cached workspace files (literature, data, metadata)
        - Queue entries (download_queue, publication_queue)
        - Modalities (loaded datasets)

        Safety features:
        - Preview mode (confirmation=False): Shows what will be deleted without executing
        - Execution mode (confirmation=True): Performs actual deletion after preview
        - Auto-detects workspace category from identifier format
        - Validates existence before deletion
        - Tracks all deletions via provenance logging

        Workspace Categories (auto-detected):
        - "literature": Publications, papers (pub_*, publication_*, pmid_*)
        - "data": Dataset metadata (dataset_*, geo_*, gse_*)
        - "metadata": Validation results, sample mappings (metadata_*, samples_*)
        - "download_queue": Pending/completed download tasks (queue_*, dlq_*)
        - "publication_queue": Publication extraction tasks (pub_queue_*)
        - "modality": Loaded datasets in memory (any modality name)

        Args:
            identifier: Content identifier or modality name to delete
            workspace: Optional workspace category override
                      If None, auto-detects from identifier format
            confirmation: If True, executes deletion. If False, shows preview only (default: False)

        Returns:
            Preview of deletion actions (confirmation=False) or
            Confirmation of completed deletions (confirmation=True)

        Examples:
            # Preview deletion (safe - no actual deletion)
            delete_from_workspace(
                identifier="pub_queue_doi_10_1234_5678_metadata",
                confirmation=False
            )

            # Execute deletion after preview
            delete_from_workspace(
                identifier="pub_queue_doi_10_1234_5678_metadata",
                confirmation=True
            )

            # Delete queue entry
            delete_from_workspace(
                identifier="queue_entry_123",
                workspace="download_queue",
                confirmation=True
            )

            # Delete modality from memory
            delete_from_workspace(
                identifier="geo_gse12345_clustered",
                workspace="modality",
                confirmation=True
            )

            # Delete all cached files for a publication
            delete_from_workspace(
                identifier="pub_queue_doi_10_1234_5678",
                confirmation=True
            )
        """
        try:
            workspace_service = WorkspaceContentService(data_manager=data_manager)

            # Auto-detect workspace category if not specified
            if workspace is None:
                workspace = _auto_detect_workspace(identifier)
                if workspace is None:
                    # Could not auto-detect - check if it's a modality
                    if identifier in data_manager.list_modalities():
                        workspace = "modality"
                    else:
                        return (
                            f"Error: Could not auto-detect workspace type for '{identifier}'.\n\n"
                            f"Please specify workspace explicitly:\n"
                            f"  • literature, data, metadata (for cached files)\n"
                            f"  • download_queue, publication_queue (for queue entries)\n"
                            f"  • modality (for loaded datasets)\n\n"
                            f"Example: delete_from_workspace(identifier='{identifier}', workspace='modality', confirmation=True)"
                        )

            # Collect items to delete
            items_to_delete = []
            deletion_summary = {
                "identifier": identifier,
                "workspace": workspace,
                "items_found": 0,
                "items_deleted": 0,
                "errors": [],
            }

            # Handle modality deletion
            if workspace == "modality":
                if identifier in data_manager.list_modalities():
                    items_to_delete.append({
                        "type": "modality",
                        "identifier": identifier,
                        "location": "memory",
                    })
                    deletion_summary["items_found"] = 1
                else:
                    return f"Error: Modality '{identifier}' not found in memory.\n\nAvailable modalities: {', '.join(data_manager.list_modalities())}"

            # Handle download_queue deletion
            elif workspace == "download_queue":
                try:
                    entry = workspace_service.read_download_queue_entry(identifier)
                    items_to_delete.append({
                        "type": "queue_entry",
                        "identifier": identifier,
                        "workspace": "download_queue",
                        "details": f"{entry['dataset_id']} ({entry['status']})",
                    })
                    deletion_summary["items_found"] = 1
                except FileNotFoundError:
                    return f"Error: Download queue entry '{identifier}' not found."

            # Handle publication_queue deletion
            elif workspace == "publication_queue":
                try:
                    entry = workspace_service.read_publication_queue_entry(identifier)
                    title = entry.get("title", "Untitled")
                    items_to_delete.append({
                        "type": "queue_entry",
                        "identifier": identifier,
                        "workspace": "publication_queue",
                        "details": f"{title[:60]}... ({entry['status']})",
                    })
                    deletion_summary["items_found"] = 1
                except FileNotFoundError:
                    return f"Error: Publication queue entry '{identifier}' not found."

            # Handle cached workspace files (literature, data, metadata)
            else:
                # Map workspace strings to ContentType enum
                workspace_to_content_type = {
                    "literature": ContentType.PUBLICATION,
                    "data": ContentType.DATASET,
                    "metadata": ContentType.METADATA,
                }

                if workspace not in workspace_to_content_type:
                    return f"Error: Invalid workspace '{workspace}'. Valid: {', '.join(workspace_to_content_type.keys())}, download_queue, publication_queue, modality"

                content_type = workspace_to_content_type[workspace]

                # Find all files matching identifier (may be prefix match)
                content_dir = workspace_service._get_content_dir(content_type)
                if not content_dir.exists():
                    return f"Error: Workspace directory '{workspace}' does not exist."

                # Look for exact match or prefix match
                sanitized_id = workspace_service._sanitize_filename(identifier)
                matching_files = []

                for file_path in content_dir.glob(f"{sanitized_id}*"):
                    if file_path.is_file():
                        matching_files.append(file_path)

                if not matching_files:
                    return f"Error: No cached files found matching '{identifier}' in workspace '{workspace}'."

                for file_path in matching_files:
                    items_to_delete.append({
                        "type": "cached_file",
                        "identifier": file_path.stem,
                        "workspace": workspace,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                    })
                deletion_summary["items_found"] = len(matching_files)

            # Preview mode - show what will be deleted
            if not confirmation:
                response = f"## Deletion Preview for '{identifier}'\n\n"
                response += f"**Workspace**: {workspace}\n"
                response += f"**Items Found**: {deletion_summary['items_found']}\n\n"

                if deletion_summary["items_found"] == 0:
                    response += "⚠️ No items found to delete.\n"
                else:
                    response += "**Items that will be deleted:**\n\n"
                    for item in items_to_delete:
                        if item["type"] == "modality":
                            try:
                                adata = data_manager.get_modality(item["identifier"])
                                response += f"- **{item['identifier']}** (modality)\n"
                                response += f"  - Shape: {adata.n_obs} obs × {adata.n_vars} vars\n"
                                response += f"  - Location: {item['location']}\n"
                            except Exception:
                                response += f"- **{item['identifier']}** (modality)\n"
                        elif item["type"] == "queue_entry":
                            response += f"- **{item['identifier']}** (queue entry)\n"
                            response += f"  - Workspace: {item['workspace']}\n"
                            response += f"  - Details: {item['details']}\n"
                        elif item["type"] == "cached_file":
                            size_mb = item['size'] / (1024 * 1024)
                            response += f"- **{item['identifier']}** (cached file)\n"
                            response += f"  - Workspace: {item['workspace']}\n"
                            response += f"  - Path: {item['path']}\n"
                            response += f"  - Size: {size_mb:.2f} MB\n"

                    response += f"\n**⚠️ PREVIEW ONLY - No deletion performed**\n"
                    response += f"**To execute deletion**: Set `confirmation=True`\n"

                return response

            # Execution mode - perform actual deletion
            else:
                response = f"## Deleting '{identifier}' from workspace '{workspace}'\n\n"

                for item in items_to_delete:
                    try:
                        if item["type"] == "modality":
                            # Use ModalityManagementService for modality deletion
                            modality_service = ModalityManagementService(data_manager)
                            success, stats, ir = modality_service.remove_modality(
                                item["identifier"]
                            )
                            if success:
                                data_manager.log_tool_usage(
                                    tool_name="delete_from_workspace",
                                    parameters={
                                        "identifier": identifier,
                                        "workspace": workspace,
                                        "type": "modality",
                                    },
                                    description=f"Deleted modality: {item['identifier']}",
                                    ir=ir,
                                )
                                deletion_summary["items_deleted"] += 1
                                response += f"✓ Deleted modality: {item['identifier']}\n"
                            else:
                                deletion_summary["errors"].append(
                                    f"Failed to delete modality: {item['identifier']}"
                                )
                                response += f"✗ Failed to delete modality: {item['identifier']}\n"

                        elif item["type"] == "queue_entry":
                            # Delete queue entry file (with atomic operations to prevent race conditions)
                            queue_type = item["workspace"]
                            queue_file = (
                                data_manager.workspace_path / f"{queue_type}.jsonl"
                            )
                            lock_file = queue_file.parent / f".{queue_file.name}.lock"

                            if queue_file.exists():
                                # Use InterProcessFileLock to prevent concurrent access
                                from lobster.core.queue_storage import InterProcessFileLock

                                with InterProcessFileLock(lock_file):
                                    # Read all entries
                                    entries = []
                                    with open(queue_file, "r") as f:
                                        for line in f:
                                            entry = json.loads(line.strip())
                                            if entry.get("entry_id") != item["identifier"]:
                                                entries.append(entry)

                                    # Atomic write back without deleted entry
                                    atomic_write_jsonl(
                                        target_path=queue_file,
                                        entries=entries,
                                        serializer=lambda x: x,  # Already dict
                                    )

                                data_manager.log_tool_usage(
                                    tool_name="delete_from_workspace",
                                    parameters={
                                        "identifier": identifier,
                                        "workspace": workspace,
                                        "type": "queue_entry",
                                    },
                                    description=f"Deleted queue entry: {item['identifier']}",
                                )
                                deletion_summary["items_deleted"] += 1
                                response += f"✓ Deleted queue entry: {item['identifier']}\n"
                            else:
                                deletion_summary["errors"].append(
                                    f"Queue file not found: {queue_file}"
                                )
                                response += f"✗ Queue file not found: {queue_file}\n"

                        elif item["type"] == "cached_file":
                            # Delete cached file
                            file_path = Path(item["path"])
                            if file_path.exists():
                                file_path.unlink()
                                data_manager.log_tool_usage(
                                    tool_name="delete_from_workspace",
                                    parameters={
                                        "identifier": identifier,
                                        "workspace": workspace,
                                        "type": "cached_file",
                                        "path": str(file_path),
                                    },
                                    description=f"Deleted cached file: {item['identifier']}",
                                )
                                deletion_summary["items_deleted"] += 1
                                response += f"✓ Deleted file: {item['identifier']}\n"
                            else:
                                deletion_summary["errors"].append(
                                    f"File not found: {file_path}"
                                )
                                response += f"✗ File not found: {file_path}\n"

                    except Exception as e:
                        deletion_summary["errors"].append(str(e))
                        response += f"✗ Error deleting {item['identifier']}: {str(e)}\n"

                # Summary
                response += f"\n**Summary**:\n"
                response += f"- Items found: {deletion_summary['items_found']}\n"
                response += f"- Items deleted: {deletion_summary['items_deleted']}\n"
                if deletion_summary["errors"]:
                    response += f"- Errors: {len(deletion_summary['errors'])}\n"
                    for error in deletion_summary["errors"]:
                        response += f"  - {error}\n"

                return response

        except Exception as e:
            logger.error(f"Error in delete_from_workspace: {e}")
            return f"Error deleting from workspace: {str(e)}"

    def _auto_detect_workspace(identifier: str) -> Optional[str]:
        """Auto-detect workspace category from identifier format.

        Returns None if pattern doesn't match any known workspace type,
        requiring explicit workspace parameter from caller.
        """
        identifier_lower = identifier.lower()

        # Queue patterns
        if identifier_lower.startswith(("queue_", "dlq_")):
            return "download_queue"
        if identifier_lower.startswith("pub_queue_"):
            return "publication_queue"

        # Publication patterns
        if identifier_lower.startswith(("pub_", "publication_", "pmid_", "doi_")):
            return "literature"

        # Dataset patterns
        if identifier_lower.startswith(("dataset_", "geo_", "gse_", "gsm_", "gpl_")):
            return "data"

        # Metadata patterns
        if identifier_lower.startswith(("metadata_", "samples_", "validation_")):
            return "metadata"

        # Return None instead of guessing - safer to error
        return None

    return delete_from_workspace
