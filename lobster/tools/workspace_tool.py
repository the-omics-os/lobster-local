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
    harmonize_samples_for_export,
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
# Centralized Exports Directory (v1.0+)
# ===============================================================================

EXPORTS_DIR_NAME = "exports"  # Single location for all user-facing exports

# Maximum items to display in list mode (prevents context overflow)
MAX_LIST_ITEMS = 50


def _get_exports_directory(workspace_path: Path, create: bool = True) -> Path:
    """
    Get the centralized exports directory for all CSV/TSV/Excel exports.

    This provides a single, predictable location for users to find exported files,
    regardless of which agent or tool created them.

    Args:
        workspace_path: Base workspace path
        create: Whether to create directory if it doesn't exist (default: True)

    Returns:
        Path to exports directory (workspace/exports/)
    """
    exports_dir = workspace_path / EXPORTS_DIR_NAME
    if create and not exports_dir.exists():
        exports_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created exports directory: {exports_dir}")
    return exports_dir


def _check_deprecated_export_location(workspace_path: Path) -> Optional[List[Path]]:
    """
    Check for files in deprecated metadata/exports/ location.

    Returns list of files if found, None otherwise. Logs warning if deprecated
    location contains files.
    """
    old_export_dir = workspace_path / "metadata" / "exports"
    if old_export_dir.exists():
        old_files = list(old_export_dir.glob("*"))
        if old_files:
            logger.warning(
                f"Found {len(old_files)} file(s) in deprecated location: {old_export_dir}. "
                f"New exports go to {workspace_path / EXPORTS_DIR_NAME}. "
                "Consider migrating: mv workspace/metadata/exports/* workspace/exports/"
            )
            return old_files
    return None


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
        pattern: str = None,
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
        - "literature": Publications, papers, abstracts (cached research content)
        - "data": Dataset metadata, GEO records (cached dataset information)
        - "metadata": Validation results, sample mappings (processed metadata)
        - "exports": Centralized export files (CSV/TSV exports, user-facing files)
        - "download_queue": Dataset downloads from databases (GEO/SRA/PRIDE/etc.) - tracks file download progress
        - "publication_queue": Publication processing pipeline (RIS imports, PubMed papers, full-text extraction) - tracks literature mining workflows
        Example: get_content_from_workspace(workspace="literature")

        ## Detail Levels
        - "summary": Key-value pairs, high-level overview (default)
        - "methods": Methods section only (for publications)
        - "metadata": **Smart hybrid view for queues** (Statistics + Head 10 entries + Guidance).
                     For single items: Full JSON including COMPLETE PUBLICATION TEXT (for pubs),
                     full entry details (for queue entries), or complete metadata (for datasets).
                     NOTE: Queue listing uses smart truncation to prevent context overflow.
        - "samples": Sample IDs list (for datasets)
        - "platform": Platform information (for datasets)
        - "github": GitHub repositories (for publications)
        - "validation": Validation results (for download_queue)
        - "strategy": Download strategy (for download_queue)
        Example: get_content_from_workspace("literature", level="summary")

        **Queue Selection Guide**:
        - Use "download_queue" when tracking dataset downloads from databases (GEO, SRA, PRIDE, etc.)
        - Use "publication_queue" when tracking publication processing workflows (RIS files, PubMed searches, full-text extraction, identifier extraction)

        For download_queue workspace:
        - identifier=None: List all entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific entry
        - status_filter: "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED"
        - level="summary": Stats + top 5 recent entries
        - level="metadata": **Smart hybrid** (Stats + head 10 entries + guidance)

        For publication_queue workspace (publication/paper/literature processing):
        - Tracks: RIS file imports, PubMed paper extraction, full-text retrieval, dataset identifier mining
        - identifier=None: List all publication entries (filtered by status_filter if provided)
        - identifier=<entry_id>: Retrieve specific publication entry
        - status_filter: "pending" | "extracting" | "metadata_extracted" | "metadata_enriched" | "handoff_ready" | "completed" | "failed"
        - level="summary": Stats + top 5 recent publications
        - level="metadata": **Smart hybrid** (Stats + head 10 publications + guidance)

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
            identifier: **EXACT LOOKUP** - Complete filename or ID for single item retrieval (e.g., "harmonized_metadata.csv", "GSE123456"). Use this for accessing specific files.
            workspace: Filter by workspace category (None = all workspaces)
            level: Detail level to extract (default: "summary")
            status_filter: Status filter for download_queue/publication_queue (optional)
            pattern: **GLOB FILTER** - Wildcard pattern for filtering lists (e.g., "harmonized*", "GSE*"). Only use with wildcards (*?[]), NOT for exact filenames.

        Returns:
            Formatted content based on detail level or list of cached items

        Examples:
            # ===== EXACT LOOKUPS (use identifier parameter) =====
            # Retrieve specific export file by exact filename
            get_content_from_workspace(identifier="harmonized_metadata.csv", workspace="exports")
            get_content_from_workspace(identifier="harmonized_metadata_170_datasets.csv", workspace="exports")

            # Retrieve specific dataset by accession
            get_content_from_workspace(identifier="GSE123456", workspace="data")

            # ===== GLOB FILTERING (use pattern parameter) =====
            # List all files starting with "harmonized"
            get_content_from_workspace(pattern="harmonized*", workspace="exports")

            # List all GEO datasets
            get_content_from_workspace(pattern="GSE*", workspace="data")

            # ===== LISTING (no identifier or pattern) =====
            # List all cached content
            get_content_from_workspace()

            # List content in specific workspace
            get_content_from_workspace(workspace="literature")

            # List metadata with pattern filter (avoids loading all 1000+ items)
            get_content_from_workspace(workspace="metadata", pattern="aggregated_*")
            get_content_from_workspace(workspace="metadata", pattern="sra_*")
            get_content_from_workspace(workspace="metadata", pattern="pub_queue_*")

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

            # Check publication processing queue state
            get_content_from_workspace(
                workspace="publication_queue"
            )

            # Check dataset download queue state
            get_content_from_workspace(
                workspace="download_queue"
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
                "exports": ContentType.EXPORTS,
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

            def _adapt_general_content(ws: str, filter_status: Optional[str] = None, filter_pattern: Optional[str] = None) -> List[WorkspaceItem]:
                """
                Adapter for literature, data, and metadata workspaces.

                Converts WorkspaceContentService items to unified WorkspaceItem structure.

                Args:
                    ws: Workspace name
                    filter_status: Not used for general content (only for queues)
                    filter_pattern: Glob pattern to filter by identifier (e.g., "aggregated_*", "sra_*")
                """
                content_type = workspace_to_content_type.get(ws)
                if not content_type:
                    return []

                items = workspace_service.list_content(content_type=content_type)

                # DEFENSIVE: Should never happen after service fix, but be safe
                if items is None:
                    logger.warning("list_content returned None (service bug)")
                    return []

                # Apply glob pattern filtering if provided
                if filter_pattern:
                    from fnmatch import fnmatch
                    filtered_items = []
                    for item in items:
                        item_id = (
                            item.get('identifier')
                            or item.get('name')
                            or (Path(item.get('_file_path', '')).stem if item.get('_file_path') else None)
                            or ''
                        )
                        if fnmatch(item_id.lower(), filter_pattern.lower()):
                            filtered_items.append(item)
                    items = filtered_items
                    logger.debug(f"Pattern '{filter_pattern}' matched {len(items)} items in '{ws}' workspace")

                normalized_items: List[WorkspaceItem] = []
                for item in items:
                    # Defensive identifier extraction (safe fallback pattern)
                    item_id = (
                        item.get('identifier')
                        or item.get('name')
                        or (Path(item.get('_file_path', '')).stem if item.get('_file_path') else None)
                        or 'unknown'  # Safe fallback (removed fragile item.keys() call)
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

                # DEFENSIVE: Should never happen after service fix, but be safe
                if entries is None:
                    logger.warning("list_download_queue_entries returned None (service bug)")
                    return []

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

                # DEFENSIVE: Should never happen after service fix, but be safe
                if entries is None:
                    logger.warning("list_publication_queue_entries returned None (service bug)")
                    return []

                normalized_items: List[WorkspaceItem] = []
                for entry in entries:
                    # Extract title and authors (defensive None handling)
                    title = (entry.get('title') or 'Untitled')[:80]
                    authors = entry.get('authors') or []
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

            def _adapt_exports(filter_status: Optional[str] = None, filter_pattern: Optional[str] = None) -> List[WorkspaceItem]:
                """
                Adapter for exports workspace.

                Converts export files to unified WorkspaceItem structure.
                Uses WorkspaceContentService.list_export_files() to scan workspace/exports/ directory.
                """
                # Use pattern if provided, otherwise default to all files
                pattern = filter_pattern or "*"
                export_files = workspace_service.list_export_files(pattern=pattern)

                # DEFENSIVE: Should never happen, but be safe
                if export_files is None:
                    logger.warning("list_export_files returned None (service bug)")
                    return []

                normalized_items: List[WorkspaceItem] = []
                for file_info in export_files:
                    # Extract file metadata
                    file_name = file_info.get('name', 'unknown')
                    file_path = file_info.get('path')
                    file_size = file_info.get('size', 0)
                    modified_time = file_info.get('modified', 'unknown')
                    category = file_info.get('category', 'unknown')

                    # Format size for display
                    size_mb = file_size / (1024 * 1024) if file_size else 0
                    size_str = f"{size_mb:.2f} MB" if size_mb > 0.01 else f"{file_size} bytes"

                    normalized_items.append(WorkspaceItem(
                        identifier=file_name,
                        workspace='exports',
                        type='export_file',
                        status=None,  # No status for export files
                        priority=None,
                        title=file_name,
                        cached_at=modified_time,
                        details=f"{size_str} | {category}"
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

            def _render_queue_metadata_hybrid(
                entries: List[Dict[str, Any]],
                queue_type: str,
                entry_formatter: Callable[[Dict[str, Any]], str],
                head_size: int = 10
            ) -> str:
                """
                Render smart hybrid view for queue metadata (statistics + head + guidance).

                This pattern prevents context overflow by showing:
                1. Statistics overview (status distribution, priority for publication queue)
                2. Head of N most recent entries with full details
                3. Guidance message if more entries available

                Args:
                    entries: Queue entries to render
                    queue_type: "publication" or "download" (for display titles and hints)
                    entry_formatter: Function to format individual entries (e.g., _format_pub_queue_entry_full)
                    head_size: Number of head entries to show (default: 10)

                Returns:
                    Formatted markdown response with statistics, head entries, and guidance
                """
                total = len(entries)
                response = f"## {queue_type.title()} Queue Metadata ({total} total)\n\n"

                # 1. Statistics Overview
                status_counts: Dict[str, int] = {}
                for entry in entries:
                    status = entry.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1

                response += "### Statistics Overview\n"
                response += "**Status Distribution**:\n"
                for status, count in sorted(status_counts.items()):
                    response += f"- {status}: {count}\n"

                # Add priority stats for publication queue only
                if queue_type == "publication":
                    priority_high = sum(1 for e in entries if e.get("priority", 5) <= 3)
                    priority_medium = sum(1 for e in entries if 4 <= e.get("priority", 5) <= 7)
                    priority_low = sum(1 for e in entries if e.get("priority", 5) >= 8)
                    response += f"\n**Priority**: High: {priority_high}, Medium: {priority_medium}, Low: {priority_low}\n\n"
                else:
                    response += "\n"

                # 2. Head: Show most recent entries with full details
                sorted_entries = sorted(
                    entries, key=lambda e: e.get("updated_at", ""), reverse=True
                )[:head_size]

                response += f"### Recent Entries (head {len(sorted_entries)})\n\n"
                for entry in sorted_entries:
                    response += entry_formatter(entry) + "\n\n"

                # 3. Guidance if more available
                remaining = total - head_size
                if remaining > 0:
                    response += f"---\n**Tip**: {remaining} more entries available. "
                    if queue_type == "publication":
                        response += "Use `status_filter` to narrow scope (e.g., `status_filter='handoff_ready'`).\n"
                    else:
                        response += "Use `status_filter` to narrow scope (e.g., `status_filter='PENDING'`).\n"

                return response

            # Workspace adapter dispatcher
            # Signature: adapter(status_filter, pattern) -> List[WorkspaceItem]
            WORKSPACE_ADAPTERS: Dict[str, Callable[[Optional[str], Optional[str]], List[WorkspaceItem]]] = {
                "literature": lambda s, p: _adapt_general_content("literature", s, p),
                "data": lambda s, p: _adapt_general_content("data", s, p),
                "metadata": lambda s, p: _adapt_general_content("metadata", s, p),
                "exports": lambda s, p: _adapt_exports(s, p),  # Uses pattern for file filtering
                "download_queue": lambda s, p: _adapt_download_queue(s),  # Queues ignore pattern
                "publication_queue": lambda s, p: _adapt_publication_queue(s),  # Queues ignore pattern
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
                            title = entry.get("title") or "Untitled"
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
                            items = adapter(status_filter, pattern)
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
                    # Smart hybrid mode using helper (Statistics + Head + Guidance)
                    if workspace == "publication_queue":
                        entries = workspace_service.list_publication_queue_entries(status_filter)

                        # DEFENSIVE: Validate entries before rendering
                        if entries is None or not isinstance(entries, list):
                            logger.error(f"Invalid entries type: {type(entries)} (expected list)")
                            return "Error: Unable to retrieve publication queue entries. Please check logs."

                        return _render_queue_metadata_hybrid(
                            entries=entries,
                            queue_type="publication",
                            entry_formatter=_format_pub_queue_entry_full,
                        )

                    elif workspace == "download_queue":
                        entries = workspace_service.list_download_queue_entries(status_filter)

                        # DEFENSIVE: Validate entries before rendering
                        if entries is None or not isinstance(entries, list):
                            logger.error(f"Invalid entries type: {type(entries)} (expected list)")
                            return "Error: Unable to retrieve download queue entries. Please check logs."

                        return _render_queue_metadata_hybrid(
                            entries=entries,
                            queue_type="download",
                            entry_formatter=_format_queue_entry_full,
                        )

                # Summary mode (default) - unified formatting with truncation
                total_items = len(all_items)

                # Apply truncation to prevent context overflow
                if total_items > MAX_LIST_ITEMS:
                    displayed_items = all_items[:MAX_LIST_ITEMS]
                    truncation_msg = (
                        f"\n---\n**Note**: Showing {MAX_LIST_ITEMS} of {total_items} total items.\n\n"
                        f"**To narrow results**:\n"
                        f"- Use `identifier='exact_filename.csv'` for exact file lookup\n"
                        f"- Use `pattern='prefix*'` for wildcard filtering\n"
                    )
                else:
                    displayed_items = all_items
                    truncation_msg = ""

                ws_display = workspace.title() if workspace else "All"
                response = f"## {ws_display} Workspace ({len(displayed_items)}/{total_items} items)\n\n"

                for item in displayed_items:
                    response += _format_workspace_item(item)

                response += truncation_msg
                return response

            # Handle exports workspace retrieval mode (flexible filename matching)
            if workspace == "exports" and identifier:
                # Normalize identifier (strip extension if present for stem matching)
                identifier_stem = identifier
                for ext in ['.csv', '.tsv', '.xlsx', '.json']:
                    if identifier_stem.lower().endswith(ext):
                        identifier_stem = identifier_stem[:-len(ext)]
                        break

                # Find matching file in exports directory
                exports_dir = Path(data_manager.workspace_path) / "exports"
                if not exports_dir.exists():
                    return "Error: exports directory does not exist."

                # Try exact match first (with extension), then stem match (without extension)
                matching_files = []
                for f in exports_dir.iterdir():
                    if f.is_file():
                        if f.name == identifier or f.stem == identifier_stem:
                            matching_files.append(f)

                if not matching_files:
                    # Return helpful error with available files
                    available = [f.name for f in exports_dir.iterdir() if f.is_file()][:10]
                    available_str = ', '.join(available) if available else 'none'
                    return (
                        f"Error: No export file matching '{identifier}' found.\n\n"
                        f"**Available files**: {available_str}"
                    )

                if len(matching_files) > 1:
                    return (
                        f"Error: Multiple files match '{identifier}': "
                        f"{[f.name for f in matching_files]}. Please specify exact filename with extension."
                    )

                matched_file = matching_files[0]
                file_size = matched_file.stat().st_size
                size_mb = file_size / (1024 * 1024)
                size_str = f"{size_mb:.2f} MB" if size_mb > 0.01 else f"{file_size} bytes"

                # Return based on level
                if level == "summary":
                    return (
                        f"## Export File: {matched_file.name}\n\n"
                        f"**Path**: {matched_file}\n"
                        f"**Size**: {size_str}\n"
                    )
                elif level == "metadata":
                    return (
                        f"## Export File: {matched_file.name}\n\n"
                        f"**Full Path**: {matched_file}\n"
                        f"**Size**: {size_str}\n\n"
                        f"**Usage**: Use `execute_custom_code` with `load_workspace_files=True` "
                        f"to access this file as variable `{matched_file.stem.replace('-', '_')}`."
                    )
                else:
                    # Default: return path info
                    return f"**File**: {matched_file.name}\n**Path**: {matched_file}"

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
                    # Contextual hint instead of recursive call (prevents LangChain deprecation warning)
                    if workspace == "publication_queue":
                        hint = "Use `get_content_from_workspace(workspace='publication_queue')` to list entries."
                    elif workspace == "literature":
                        hint = "Publication identifiers follow pattern: `publication_<PMID>` or `pub_queue_doi_<doi>`."
                    else:
                        hint = f"Use `get_content_from_workspace(workspace='{workspace or 'literature'}')` to list available items."
                    return f"Error: Identifier '{identifier}' not found{workspace_filter}.\n\n**Suggestion**: {hint}"

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
                # Contextual hint instead of recursive call (prevents LangChain deprecation warning)
                if workspace == "publication_queue":
                    hint = "Use `get_content_from_workspace(workspace='publication_queue')` to list entries by status."
                elif workspace == "download_queue":
                    hint = "Use `get_content_from_workspace(workspace='download_queue')` to list pending downloads."
                elif workspace == "metadata":
                    hint = "Metadata identifiers follow pattern: `sra_<PRJNA>_samples` or `pub_queue_<doi>_metadata`."
                elif workspace == "literature":
                    hint = "Publication identifiers follow pattern: `publication_<PMID>` or `pub_queue_doi_<doi>`."
                elif workspace == "data":
                    hint = "Dataset identifiers follow pattern: `dataset_<GSE>` or `geo_<GSE>`."
                else:
                    hint = f"Use `get_content_from_workspace(workspace='{workspace or 'all'}')` to list available items."
                return f"Error: Identifier '{identifier}' not found{workspace_filter}.\n\n**Suggestion**: {hint}"

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
        - "strict": MIMARKS-compliant export (excludes non-schema columns + deduplicates by run_accession)

        Args:
            identifier: Content identifier to cache (must exist in current session)
            workspace: Target workspace category ("literature", "data", "metadata")
            content_type: Type of content ("publication", "dataset", "metadata")
            output_format: Output format ("json" or "csv"). Default: "json"
            export_mode: CSV export mode ("auto", "rich", "simple", "strict"). Default: "auto"
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
                "exports": ContentType.EXPORTS,
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

            if export_mode not in {"auto", "rich", "simple", "strict"}:
                return f"Error: Invalid export_mode '{export_mode}'. Valid: auto, rich, simple, strict"

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
                return (
                    f"Error: Identifier '{identifier}' not found in current session.\n\n"
                    f"**Suggestion**: Use `get_content_from_workspace()` to list available items, "
                    f"or check if data is loaded in modalities with `list_available_modalities()`."
                )

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

                    # Map publication_* to source_* fields (v2.0 harmonization)
                    # Priority: existing source_* > publication_* from sample > content_data fallback
                    enriched["source_doi"] = (
                        enriched.get("source_doi")
                        or enriched.get("publication_doi")
                        or source_doi
                        or ""
                    )
                    enriched["source_pmid"] = (
                        enriched.get("source_pmid")
                        or enriched.get("publication_pmid")
                        or source_pmid
                        or ""
                    )
                    enriched["source_entry_id"] = (
                        enriched.get("source_entry_id")
                        or enriched.get("publication_entry_id")
                        or source_entry_id
                        or ""
                    )
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

                    if use_rich_format or export_mode == "strict":
                        format_name = "strict MIMARKS" if export_mode == "strict" else "schema-driven"
                        logger.info(
                            f"Using {format_name} export format (column count depends on data type)"
                        )
                        rich_export_used = True

                        # Enrich samples with publication context
                        enriched_samples = _enrich_samples_with_publication_context(
                            samples_list, identifier, content_data
                        )

                        # v2.0: Apply harmonization pipeline (column merge, disease extraction, sparse removal)
                        harmonized_samples, harmonization_stats, provenance_log = harmonize_samples_for_export(
                            enriched_samples,
                            remove_sparse=True,
                            sparse_threshold=0.05,
                            remove_constant=True,
                            track_provenance=True,
                        )
                        logger.info(
                            f"Harmonization: {harmonization_stats['columns_before']} → "
                            f"{harmonization_stats['columns_after']} columns, "
                            f"disease coverage {harmonization_stats['disease_coverage_before']:.1f}% → "
                            f"{harmonization_stats['disease_coverage_after']:.1f}%, "
                            f"{harmonization_stats['provenance_log_size']} transformations tracked"
                        )

                        # Schema-driven column ordering (v1.2.0 - export_schemas.py)
                        # Infer data type from samples (auto-detects SRA/proteomics/metabolomics)
                        data_type = infer_data_type(harmonized_samples)

                        # Strict mode: exclude non-schema columns (MIMARKS compliance)
                        # Rich mode: include extra columns for flexibility
                        strict_export = export_mode == "strict"
                        ordered_cols = get_ordered_export_columns(
                            samples=harmonized_samples,
                            data_type=data_type,
                            include_extra=not strict_export,
                        )

                        # Create DataFrame with schema-ordered columns
                        df = pd.DataFrame(harmonized_samples)

                        # Deduplication by run_accession (Bug 2 fix - DataBioMix)
                        # Multiple publications can reference same BioProject/samples
                        duplicates_removed = 0
                        if "run_accession" in df.columns:
                            original_count = len(df)
                            df = df.drop_duplicates(subset=["run_accession"], keep="first")
                            duplicates_removed = original_count - len(df)
                            if duplicates_removed > 0:
                                logger.info(
                                    f"Deduplication: removed {duplicates_removed} duplicate samples "
                                    f"by run_accession ({original_count} → {len(df)})"
                                )
                                samples_count = len(df)  # Update count after dedup

                        available_cols = [c for c in ordered_cols if c in df.columns]
                        df = df[available_cols]

                        # v2.0: Dual-file export (ALWAYS export both strict + rich)
                        workspace_path = Path(data_manager.workspace_path)
                        exports_dir = _get_exports_directory(workspace_path, create=True)
                        base_filename = workspace_service._sanitize_filename(identifier)

                        # Generate timestamp if requested (date + time format)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") if add_timestamp else ""

                        # Check for deprecated location
                        _check_deprecated_export_location(workspace_path)

                        # Export 1: Rich file (current df with ~90 columns)
                        # Timestamp after suffix: filename_rich_2026-01-09_143052.csv
                        rich_filename = f"{base_filename}_rich_{timestamp}.csv" if timestamp else f"{base_filename}_rich.csv"
                        rich_file_path = exports_dir / rich_filename
                        df.to_csv(rich_file_path, index=False, encoding="utf-8")
                        logger.info(
                            f"CSV export (rich): {len(df)} rows, {len(df.columns)} columns → {rich_file_path.name}"
                        )

                        # Export 2: Strict file (34 MIMARKS columns only)
                        strict_cols = get_ordered_export_columns(
                            samples=harmonized_samples,
                            data_type=data_type,
                            include_extra=False,  # MIMARKS only
                        )
                        available_strict = [c for c in strict_cols if c in df.columns]
                        df_strict = df[available_strict]
                        strict_filename = f"{base_filename}_strict_{timestamp}.csv" if timestamp else f"{base_filename}_strict.csv"
                        strict_file_path = exports_dir / strict_filename
                        df_strict.to_csv(strict_file_path, index=False, encoding="utf-8")
                        logger.info(
                            f"CSV export (strict): {len(df_strict)} rows, {len(df_strict.columns)} columns → {strict_file_path.name}"
                        )

                        # Export 3: Provenance log (harmonization audit trail)
                        if provenance_log:
                            prov_filename = f"{base_filename}_harmonization_log_{timestamp}.tsv" if timestamp else f"{base_filename}_harmonization_log.tsv"
                            prov_log_path = exports_dir / prov_filename
                            prov_df = pd.DataFrame(provenance_log)
                            prov_df.to_csv(prov_log_path, sep="\t", index=False, encoding="utf-8")
                            logger.info(
                                f"Provenance log: {len(provenance_log)} transformations → {prov_log_path.name}"
                            )
                        else:
                            prov_log_path = None

                        # Primary file path for backward compatibility (points to rich)
                        cache_file_path = rich_file_path

                        # Update metadata_store with all paths + stats + provenance (Phase 6)
                        if identifier in data_manager.metadata_store:
                            # Convert lists to tuples for hashable storage (Bug 5 fix)
                            def make_hashable(obj):
                                """Recursively convert lists to tuples for hashable storage."""
                                if isinstance(obj, list):
                                    return tuple(make_hashable(item) for item in obj)
                                elif isinstance(obj, dict):
                                    return {k: make_hashable(v) for k, v in obj.items()}
                                return obj

                            update_dict = {
                                "csv_export_path": str(rich_file_path),
                                "csv_export_filename": rich_file_path.name,
                                "csv_export_strict_path": str(strict_file_path),
                                "csv_export_strict_filename": strict_file_path.name,
                                "harmonization_stats": make_hashable(harmonization_stats),
                            }
                            if prov_log_path:
                                update_dict["provenance_log_path"] = str(prov_log_path)
                                update_dict["provenance_log_filename"] = prov_log_path.name
                            data_manager.metadata_store[identifier].update(update_dict)
                            logger.info(f"Updated metadata_store['{identifier}'] with dual export paths + stats + provenance")

                        # Build success response and return (don't continue to normal path)
                        response = f"""## CSV Export Complete (Dual-File v2.0)

**Identifier**: {identifier}
**Samples**: {len(df)} rows
**Harmonization**: {harmonization_stats['columns_before']} → {harmonization_stats['columns_after']} columns
**Disease Coverage**: {harmonization_stats['disease_coverage_before']:.1f}% → {harmonization_stats['disease_coverage_after']:.1f}%

**Files Created**:
1. **Rich Export**: {rich_file_path.name} ({len(df.columns)} columns)
2. **Strict Export**: {strict_file_path.name} ({len(df_strict.columns)} columns)
3. **Provenance Log**: {prov_log_path.name if prov_log_path else 'N/A'} ({harmonization_stats['provenance_log_size']} transformations)

**Location**: {exports_dir}

**Next Steps**:
- Use files for downstream analysis
- Check harmonization_log.tsv for transformation audit trail
- Use strict.csv for MIMARKS-compliant submissions
"""
                        return response

                    else:
                        # Simple export - as-is (no harmonization)
                        df = pd.DataFrame(samples_list)

                        workspace_path = Path(data_manager.workspace_path)
                        exports_dir = _get_exports_directory(workspace_path, create=True)
                        filename = workspace_service._sanitize_filename(identifier)

                        # Generate timestamp if requested (date + time format)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S") if add_timestamp else ""

                        # Add timestamp before extension: filename_2026-01-09_143052.csv
                        final_filename = f"{filename}_{timestamp}.csv" if timestamp else f"{filename}.csv"
                        cache_file_path = exports_dir / final_filename

                        # Check for deprecated location
                        _check_deprecated_export_location(workspace_path)

                        df.to_csv(cache_file_path, index=False, encoding="utf-8")
                        logger.info(
                            f"CSV export (simple): {samples_count} rows, {len(df.columns)} columns → {cache_file_path}"
                        )

                        # Update metadata_store with CSV path tracking (Task 4)
                        if identifier in data_manager.metadata_store:
                            data_manager.metadata_store[identifier]["csv_export_path"] = str(cache_file_path)
                            data_manager.metadata_store[identifier]["csv_export_filename"] = cache_file_path.name
                            logger.info(f"Updated metadata_store['{identifier}'] with csv_export_path")
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
"""
            if output_format == "csv":
                response += f"**File Name**: {cache_file_path.name}\n"
                response += f"**File Path**: {str(cache_file_path)}\n\n"
                response += "**Note**: CSV format ideal for spreadsheet import.\n"
                if samples_count:
                    response += f"**Rows**: {samples_count} samples exported (one row per sample).\n"
                    if rich_export_used:
                        response += (
                            "**Format**: Rich 28-column format with publication context "
                            "(source_doi, source_pmid, source_entry_id).\n"
                        )
                    response += f"\n**Quick Access**: The file is now auto-loaded in `execute_custom_code` as variable `{workspace_service._sanitize_filename(identifier).replace('-', '_')}`\n"

            response += "\n**Next Steps**:\n"
            response += "- Use `get_content_from_workspace(workspace='exports')` to list all export files\n"
            response += "- Use `execute_custom_code(load_workspace_files=True)` to access the exported data\n"

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
        max_entries: int = 100,
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
            max_entries: Maximum number of entries to process for "all" or "handoff_ready"
                        modes (default: 100). Use specific entry_ids for larger exports.

        Returns:
            Summary of export with file location and sample counts per publication
        """
        import pandas as pd

        try:
            workspace_service = WorkspaceContentService(data_manager=data_manager)
            workspace_path = Path(data_manager.workspace_path)

            # Use centralized exports directory (v1.0+)
            exports_dir = _get_exports_directory(workspace_path, create=True)

            # Check for deprecated location and warn user
            _check_deprecated_export_location(workspace_path)

            # Parse entry_ids with bounded iteration
            truncated_warning = ""
            if entry_ids.lower() == "all":
                # Find all entries with samples in metadata_store (bounded)
                all_sample_keys = [
                    k for k in data_manager.metadata_store.keys()
                    if k.endswith("_samples")
                ]
                total_available = len(all_sample_keys)

                # Apply limit and sort by key (most recent first if keys have timestamps)
                limited_keys = sorted(all_sample_keys, reverse=True)[:max_entries]
                target_entries = [k.rsplit("_samples", 1)[0] for k in limited_keys]

                if total_available > max_entries:
                    truncated_warning = (
                        f"\n**Note**: Limited to {max_entries} of {total_available} available entries. "
                        f"Use specific entry_ids or increase max_entries for more."
                    )

            elif entry_ids.lower() == "handoff_ready":
                # Find entries with HANDOFF_READY status (bounded)
                all_handoff_entries = []
                for k in data_manager.metadata_store.keys():
                    if k.endswith("_samples"):
                        entry_id = k.rsplit("_samples", 1)[0]
                        # Check if entry has been processed (has samples)
                        if data_manager.metadata_store.get(k, {}).get("samples"):
                            all_handoff_entries.append(entry_id)

                total_available = len(all_handoff_entries)
                target_entries = sorted(all_handoff_entries, reverse=True)[:max_entries]

                if total_available > max_entries:
                    truncated_warning = (
                        f"\n**Note**: Limited to {max_entries} of {total_available} handoff_ready entries. "
                        f"Use specific entry_ids or increase max_entries for more."
                    )
            else:
                # Parse comma-separated list (no limit for explicit IDs)
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

            # Sanitize output filename and add timestamp to prevent overwrites
            safe_filename = workspace_service._sanitize_filename(output_filename)
            # Remove .csv extension if present (we'll add it back with timestamp)
            if safe_filename.endswith(".csv"):
                safe_filename = safe_filename[:-4]

            # Add timestamp: filename_2026-01-09_143052.csv
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            final_filename = f"{safe_filename}_{timestamp}.csv"
            output_path = exports_dir / final_filename

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

            # Add truncation warning if applicable
            if truncated_warning:
                response += truncated_warning

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
                    available = data_manager.list_modalities()
                    available_str = ", ".join(available) if available else "none loaded"
                    return (
                        f"Error: Modality '{identifier}' not found in memory.\n\n"
                        f"**Available modalities**: {available_str}\n\n"
                        f"**Suggestion**: Use `list_available_modalities()` to see all loaded datasets."
                    )

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
                    return (
                        f"Error: Download queue entry '{identifier}' not found.\n\n"
                        f"**Suggestion**: Use `get_content_from_workspace(workspace='download_queue')` "
                        f"to list available entries."
                    )

            # Handle publication_queue deletion
            elif workspace == "publication_queue":
                try:
                    entry = workspace_service.read_publication_queue_entry(identifier)
                    title = entry.get("title") or "Untitled"
                    items_to_delete.append({
                        "type": "queue_entry",
                        "identifier": identifier,
                        "workspace": "publication_queue",
                        "details": f"{title[:60]}... ({entry['status']})",
                    })
                    deletion_summary["items_found"] = 1
                except FileNotFoundError:
                    return (
                        f"Error: Publication queue entry '{identifier}' not found.\n\n"
                        f"**Suggestion**: Use `get_content_from_workspace(workspace='publication_queue')` "
                        f"to list available entries."
                    )

            # Handle cached workspace files (literature, data, metadata)
            else:
                # Map workspace strings to ContentType enum
                workspace_to_content_type = {
                    "literature": ContentType.PUBLICATION,
                    "data": ContentType.DATASET,
                    "metadata": ContentType.METADATA,
                    "exports": ContentType.EXPORTS,
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
