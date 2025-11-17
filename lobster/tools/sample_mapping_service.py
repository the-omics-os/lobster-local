"""
Sample Mapping Service for Cross-Dataset Sample ID Harmonization.

This service provides fuzzy matching, pattern-based matching, and metadata-supported
matching strategies to map sample IDs across different datasets for multi-omics integration.

Used by metadata_assistant agent for cross-dataset sample ID mapping.

Phase 3 implementation for research agent refactoring.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from lobster.core.data_manager_v2 import DataManagerV2

# Lazy RapidFuzz import with fallback
try:
    from rapidfuzz import fuzz

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    fuzz = None

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Matching Results
# =============================================================================


class SampleMatch(BaseModel):
    """Represents a successful sample ID match between datasets.

    Attributes:
        source_id: Sample ID from source dataset
        target_id: Sample ID from target dataset
        confidence_score: Match confidence (0.0-1.0)
        match_strategy: Strategy used ("exact", "fuzzy", "pattern", "metadata")
        metadata_support: Dict of metadata fields that supported the match
    """

    source_id: str = Field(..., description="Sample ID from source dataset")
    target_id: str = Field(..., description="Sample ID from target dataset")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Match confidence (0.0-1.0)"
    )
    match_strategy: str = Field(
        ..., description="Strategy used (exact/fuzzy/pattern/metadata)"
    )
    metadata_support: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Metadata fields supporting the match"
    )


class UnmappedSample(BaseModel):
    """Represents a sample that could not be mapped.

    Attributes:
        sample_id: Unmapped sample ID
        dataset: Dataset the sample belongs to
        reason: Reason for failure to map
        best_candidates: List of potential matches with low confidence
    """

    sample_id: str = Field(..., description="Unmapped sample ID")
    dataset: str = Field(..., description="Dataset the sample belongs to")
    reason: str = Field(..., description="Reason for failure to map")
    best_candidates: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="List of (sample_id, confidence) tuples for near-matches",
    )


class SampleMappingResult(BaseModel):
    """Complete result of sample mapping operation.

    Attributes:
        exact_matches: List of exact matches
        fuzzy_matches: List of fuzzy matches (confidence >= threshold)
        unmapped: List of unmapped samples
        summary: Summary statistics
        warnings: List of warnings encountered
    """

    exact_matches: List[SampleMatch] = Field(
        default_factory=list, description="Exact matches"
    )
    fuzzy_matches: List[SampleMatch] = Field(
        default_factory=list, description="Fuzzy matches"
    )
    unmapped: List[UnmappedSample] = Field(
        default_factory=list, description="Unmapped samples"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")


# =============================================================================
# SampleMappingService
# =============================================================================


class SampleMappingService:
    """Service for cross-dataset sample ID mapping with multiple strategies.

    This service implements four matching strategies:
    1. Exact match - Case-insensitive exact matching
    2. Fuzzy match - RapidFuzz token_set_ratio (if available)
    3. Pattern match - Regex prefix/suffix removal
    4. Metadata match - Metadata-supported matching

    Follows Lobster architecture:
    - Receives DataManagerV2 in __init__
    - Stateless matching operations
    - Returns Pydantic result models
    """

    def __init__(
        self, data_manager: DataManagerV2, min_confidence: float = 0.75
    ) -> None:
        """Initialize SampleMappingService.

        Args:
            data_manager: DataManagerV2 instance for modality access
            min_confidence: Minimum confidence threshold for fuzzy matches (0.0-1.0)
        """
        self.data_manager = data_manager
        self.min_confidence = min_confidence

        logger.debug(
            f"SampleMappingService initialized with min_confidence={min_confidence}"
        )
        if not RAPIDFUZZ_AVAILABLE:
            logger.warning(
                "RapidFuzz not available - fuzzy matching will use fallback method"
            )

    # =========================================================================
    # Matching Strategies
    # =========================================================================

    def _exact_match(
        self, source_ids: List[str], target_ids: List[str]
    ) -> Tuple[List[SampleMatch], Set[str], Set[str]]:
        """Perform case-insensitive exact matching.

        Args:
            source_ids: List of source sample IDs
            target_ids: List of target sample IDs

        Returns:
            Tuple of (matches, matched_source_ids, matched_target_ids)
        """
        matches = []
        matched_source = set()
        matched_target = set()

        # Create case-insensitive lookup
        target_lookup = {tid.lower(): tid for tid in target_ids}

        for source_id in source_ids:
            source_lower = source_id.lower()
            if source_lower in target_lookup:
                target_id = target_lookup[source_lower]
                matches.append(
                    SampleMatch(
                        source_id=source_id,
                        target_id=target_id,
                        confidence_score=1.0,
                        match_strategy="exact",
                        metadata_support={},
                    )
                )
                matched_source.add(source_id)
                matched_target.add(target_id)

        logger.debug(f"Exact matching: {len(matches)} matches found")
        return matches, matched_source, matched_target

    def _fuzzy_match(
        self,
        source_ids: List[str],
        target_ids: List[str],
        matched_target: Set[str],
    ) -> List[SampleMatch]:
        """Perform fuzzy matching using RapidFuzz (if available).

        Args:
            source_ids: List of unmapped source sample IDs
            target_ids: List of unmapped target sample IDs
            matched_target: Set of already matched target IDs to exclude

        Returns:
            List of fuzzy matches above confidence threshold
        """
        if not RAPIDFUZZ_AVAILABLE:
            logger.debug("Fuzzy matching skipped - RapidFuzz not available")
            return []

        matches = []
        available_targets = [tid for tid in target_ids if tid not in matched_target]

        for source_id in source_ids:
            best_score = 0.0
            best_match = None

            for target_id in available_targets:
                # Use token_set_ratio for flexible matching
                score = fuzz.token_set_ratio(source_id, target_id) / 100.0

                if score > best_score:
                    best_score = score
                    best_match = target_id

            # Only accept matches above confidence threshold
            if best_match and best_score >= self.min_confidence:
                matches.append(
                    SampleMatch(
                        source_id=source_id,
                        target_id=best_match,
                        confidence_score=best_score,
                        match_strategy="fuzzy",
                        metadata_support={},
                    )
                )
                matched_target.add(best_match)

        logger.debug(f"Fuzzy matching: {len(matches)} matches found")
        return matches

    def _pattern_match(
        self,
        source_ids: List[str],
        target_ids: List[str],
        matched_target: Set[str],
    ) -> List[SampleMatch]:
        """Perform pattern-based matching with common prefix/suffix removal.

        Common patterns:
        - Prefix: "Sample_", "GSM", "SRR", "Control_", "Treatment_"
        - Suffix: "_Rep1", "_Rep2", "_Batch1", "_1", "_2"

        Args:
            source_ids: List of unmapped source sample IDs
            target_ids: List of unmapped target sample IDs
            matched_target: Set of already matched target IDs to exclude

        Returns:
            List of pattern matches
        """
        matches = []
        available_targets = [tid for tid in target_ids if tid not in matched_target]

        # Common prefixes and suffixes
        common_prefixes = [
            r"Sample[-_]?",
            r"GSM\d+[-_]?",
            r"SRR\d+[-_]?",
            r"Control[-_]?",
            r"Treatment[-_]?",
            r"Patient[-_]?",
        ]
        common_suffixes = [
            r"[-_]?Rep\d+",
            r"[-_]?Batch\d+",
            r"[-_]?\d+",
            r"[-_]?[AB]",
        ]

        def normalize_id(sample_id: str) -> str:
            """Remove common prefixes and suffixes."""
            normalized = sample_id
            for prefix in common_prefixes:
                normalized = re.sub(f"^{prefix}", "", normalized, flags=re.IGNORECASE)
            for suffix in common_suffixes:
                normalized = re.sub(f"{suffix}$", "", normalized, flags=re.IGNORECASE)
            return normalized.lower()

        # Create normalized lookup
        target_normalized = {normalize_id(tid): tid for tid in available_targets}

        for source_id in source_ids:
            source_norm = normalize_id(source_id)
            if source_norm in target_normalized:
                target_id = target_normalized[source_norm]
                matches.append(
                    SampleMatch(
                        source_id=source_id,
                        target_id=target_id,
                        confidence_score=0.9,  # High confidence for pattern match
                        match_strategy="pattern",
                        metadata_support={"normalized_id": source_norm},
                    )
                )
                matched_target.add(target_id)

        logger.debug(f"Pattern matching: {len(matches)} matches found")
        return matches

    def _metadata_match(
        self,
        source_ids: List[str],
        target_ids: List[str],
        source_metadata: Optional[Dict[str, Dict[str, Any]]],
        target_metadata: Optional[Dict[str, Dict[str, Any]]],
        matched_target: Set[str],
    ) -> List[SampleMatch]:
        """Perform metadata-supported matching.

        Args:
            source_ids: List of unmapped source sample IDs
            target_ids: List of unmapped target sample IDs
            source_metadata: Dict mapping source IDs to metadata
            target_metadata: Dict mapping target IDs to metadata
            matched_target: Set of already matched target IDs to exclude

        Returns:
            List of metadata-supported matches
        """
        if not source_metadata or not target_metadata:
            logger.debug("Metadata matching skipped - metadata not provided")
            return []

        matches = []
        available_targets = [tid for tid in target_ids if tid not in matched_target]

        # Fields to check for matching
        metadata_fields = ["condition", "treatment", "tissue", "timepoint", "batch"]

        for source_id in source_ids:
            if source_id not in source_metadata:
                continue

            source_meta = source_metadata[source_id]
            best_match = None
            best_support = {}

            for target_id in available_targets:
                if target_id not in target_metadata:
                    continue

                target_meta = target_metadata[target_id]
                support = {}

                # Check metadata field alignment
                for field in metadata_fields:
                    if field in source_meta and field in target_meta:
                        if source_meta[field] == target_meta[field]:
                            support[field] = source_meta[field]

                # Require at least 2 matching fields
                if len(support) >= 2:
                    if len(support) > len(best_support):
                        best_match = target_id
                        best_support = support

            if best_match:
                # Confidence based on number of matching fields
                confidence = min(0.95, 0.7 + (len(best_support) * 0.05))
                matches.append(
                    SampleMatch(
                        source_id=source_id,
                        target_id=best_match,
                        confidence_score=confidence,
                        match_strategy="metadata",
                        metadata_support=best_support,
                    )
                )
                matched_target.add(best_match)

        logger.debug(f"Metadata matching: {len(matches)} matches found")
        return matches

    # =========================================================================
    # Core Methods
    # =========================================================================

    def map_samples_by_id(
        self,
        source_identifier: str,
        target_identifier: str,
        strategies: Optional[List[str]] = None,
        source_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        target_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SampleMappingResult:
        """Map sample IDs between source and target datasets.

        Args:
            source_identifier: Source dataset identifier
            target_identifier: Target dataset identifier
            strategies: List of strategies to use (default: all available)
            source_metadata: Optional source metadata dict
            target_metadata: Optional target metadata dict

        Returns:
            SampleMappingResult with matches, unmapped, and summary

        Raises:
            ValueError: If datasets not found or have no samples
        """
        # Validate datasets exist
        if source_identifier not in self.data_manager.list_modalities():
            raise ValueError(f"Source dataset '{source_identifier}' not found")
        if target_identifier not in self.data_manager.list_modalities():
            raise ValueError(f"Target dataset '{target_identifier}' not found")

        # Get sample IDs
        source_adata = self.data_manager.get_modality(source_identifier)
        target_adata = self.data_manager.get_modality(target_identifier)

        source_ids = source_adata.obs_names.tolist()
        target_ids = target_adata.obs_names.tolist()

        if not source_ids or not target_ids:
            raise ValueError("One or both datasets have no samples")

        logger.info(
            f"Mapping samples: {len(source_ids)} source → {len(target_ids)} target"
        )

        # Default to all strategies
        if strategies is None:
            strategies = ["exact", "fuzzy", "pattern", "metadata"]

        # Initialize results
        all_matches = []
        matched_source = set()
        matched_target = set()
        warnings = []

        # Strategy 1: Exact matching
        if "exact" in strategies:
            exact_matches, matched_src, matched_tgt = self._exact_match(
                source_ids, target_ids
            )
            all_matches.extend(exact_matches)
            matched_source.update(matched_src)
            matched_target.update(matched_tgt)

        # Get unmapped IDs for subsequent strategies
        unmapped_source = [sid for sid in source_ids if sid not in matched_source]
        unmapped_target = [tid for tid in target_ids if tid not in matched_target]

        # Strategy 2: Fuzzy matching
        if "fuzzy" in strategies and unmapped_source:
            if RAPIDFUZZ_AVAILABLE:
                fuzzy_matches = self._fuzzy_match(
                    unmapped_source, unmapped_target, matched_target
                )
                all_matches.extend(fuzzy_matches)
                matched_source.update([m.source_id for m in fuzzy_matches])
            else:
                warnings.append(
                    "Fuzzy matching skipped - RapidFuzz not installed. "
                    "Install with: pip install rapidfuzz"
                )

        # Update unmapped
        unmapped_source = [sid for sid in source_ids if sid not in matched_source]

        # Strategy 3: Pattern matching
        if "pattern" in strategies and unmapped_source:
            pattern_matches = self._pattern_match(
                unmapped_source, unmapped_target, matched_target
            )
            all_matches.extend(pattern_matches)
            matched_source.update([m.source_id for m in pattern_matches])

        # Update unmapped
        unmapped_source = [sid for sid in source_ids if sid not in matched_source]

        # Strategy 4: Metadata matching
        if "metadata" in strategies and unmapped_source:
            metadata_matches = self._metadata_match(
                unmapped_source,
                unmapped_target,
                source_metadata,
                target_metadata,
                matched_target,
            )
            all_matches.extend(metadata_matches)
            matched_source.update([m.source_id for m in metadata_matches])

        # Final unmapped samples
        unmapped_source = [sid for sid in source_ids if sid not in matched_source]
        unmapped = [
            UnmappedSample(
                sample_id=sid,
                dataset=source_identifier,
                reason="No match found above confidence threshold",
                best_candidates=[],
            )
            for sid in unmapped_source
        ]

        # Separate exact from fuzzy/pattern/metadata matches
        exact_matches = [m for m in all_matches if m.match_strategy == "exact"]
        fuzzy_matches = [m for m in all_matches if m.match_strategy != "exact"]

        # Summary statistics
        summary = {
            "total_source_samples": len(source_ids),
            "total_target_samples": len(target_ids),
            "exact_matches": len(exact_matches),
            "fuzzy_matches": len(fuzzy_matches),
            "unmapped": len(unmapped),
            "mapping_rate": len(matched_source) / len(source_ids) if source_ids else 0,
        }

        logger.info(
            f"Mapping complete: {summary['exact_matches']} exact, "
            f"{summary['fuzzy_matches']} fuzzy, {summary['unmapped']} unmapped"
        )

        return SampleMappingResult(
            exact_matches=exact_matches,
            fuzzy_matches=fuzzy_matches,
            unmapped=unmapped,
            summary=summary,
            warnings=warnings,
        )

    def format_mapping_report(self, result: SampleMappingResult) -> str:
        """Generate human-readable markdown report from mapping result.

        Args:
            result: SampleMappingResult to format

        Returns:
            Formatted markdown report
        """
        report_lines = [
            "# Sample Mapping Report\n",
            "## Summary",
            f"- **Total Source Samples**: {result.summary['total_source_samples']}",
            f"- **Total Target Samples**: {result.summary['total_target_samples']}",
            f"- **Exact Matches**: {result.summary['exact_matches']}",
            f"- **Fuzzy Matches**: {result.summary['fuzzy_matches']}",
            f"- **Unmapped Samples**: {result.summary['unmapped']}",
            f"- **Mapping Rate**: {result.summary['mapping_rate']:.1%}\n",
        ]

        if result.exact_matches:
            report_lines.append("## Exact Matches")
            for match in result.exact_matches[:10]:  # Show first 10
                report_lines.append(f"- {match.source_id} → {match.target_id}")
            if len(result.exact_matches) > 10:
                report_lines.append(
                    f"- ... and {len(result.exact_matches) - 10} more\n"
                )
            else:
                report_lines.append("")

        if result.fuzzy_matches:
            report_lines.append("## Fuzzy/Pattern Matches")
            for match in result.fuzzy_matches[:10]:
                report_lines.append(
                    f"- {match.source_id} → {match.target_id} "
                    f"({match.confidence_score:.2f}, {match.match_strategy})"
                )
            if len(result.fuzzy_matches) > 10:
                report_lines.append(
                    f"- ... and {len(result.fuzzy_matches) - 10} more\n"
                )
            else:
                report_lines.append("")

        if result.unmapped:
            report_lines.append("## Unmapped Samples")
            for unmapped in result.unmapped[:10]:
                report_lines.append(f"- {unmapped.sample_id} ({unmapped.reason})")
            if len(result.unmapped) > 10:
                report_lines.append(f"- ... and {len(result.unmapped) - 10} more\n")
            else:
                report_lines.append("")

        if result.warnings:
            report_lines.append("## Warnings")
            for warning in result.warnings:
                report_lines.append(f"- ⚠️ {warning}\n")

        return "\n".join(report_lines)
