"""
Sample Grouping Service for individual-to-sample relationship management.

This service groups samples by individual_id and validates consistency across
samples from the same individual. Essential for longitudinal studies and
multi-sample datasets where multiple samples come from the same individual.

Addresses DataBioMix pain point: "The sample can be correctly matched to an
individual, with relevant metadata available."
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.sra import SRASampleSchema

logger = logging.getLogger(__name__)


class SampleGroupingService:
    """
    Service for grouping samples by individual and validating consistency.

    This service provides:
    1. Grouping samples by individual_id
    2. Sorting samples within each individual by timepoint
    3. Validation of cross-sample consistency (organism, platform)
    4. Detection of duplicate timepoints
    5. Analysis of longitudinal coverage

    Use Cases:
    - Preparing data for longitudinal microbiome analysis
    - Validating sample metadata before pipeline execution
    - Identifying missing timepoints or duplicate entries
    - Grouping samples for differential analysis
    """

    def __init__(self):
        """Initialize the service."""
        self.logger = logging.getLogger(__name__)

    def group_by_individual(
        self,
        samples: List[SRASampleSchema],
        sort_by_timepoint: bool = True,
    ) -> Dict[str, List[SRASampleSchema]]:
        """
        Group samples by individual_id.

        Samples without individual_id are grouped under "unknown_{biosample}".

        Args:
            samples: List of validated SRASampleSchema instances
            sort_by_timepoint: If True, sort samples within each group by timepoint_numeric

        Returns:
            Dict mapping individual_id to list of samples

        Examples:
            >>> service = SampleGroupingService()
            >>> groups = service.group_by_individual(samples)
            >>> groups["P042"]
            [<SRASampleSchema day=0>, <SRASampleSchema day=13>]
        """
        groups: Dict[str, List[SRASampleSchema]] = defaultdict(list)

        for sample in samples:
            ind_id = sample.individual_id or f"unknown_{sample.biosample}"
            groups[ind_id].append(sample)

        # Sort each group by timepoint_numeric if requested
        if sort_by_timepoint:
            for ind_id in groups:
                groups[ind_id].sort(key=lambda s: s.timepoint_numeric or 0)

        self.logger.info(
            f"Grouped {len(samples)} samples into {len(groups)} individuals"
        )

        return dict(groups)

    def validate_individual_consistency(
        self,
        samples: List[SRASampleSchema],
    ) -> ValidationResult:
        """
        Validate that samples from the same individual have consistent metadata.

        Checks:
        1. Organism consistency within individual
        2. Duplicate timepoints (same individual, same timepoint)
        3. Library strategy consistency
        4. Platform consistency

        Args:
            samples: List of validated SRASampleSchema instances

        Returns:
            ValidationResult with warnings for inconsistencies

        Examples:
            >>> result = service.validate_individual_consistency(samples)
            >>> result.warnings
            ["Individual P042: Duplicate timepoints detected"]
        """
        result = ValidationResult()
        groups = self.group_by_individual(samples, sort_by_timepoint=False)

        for ind_id, ind_samples in groups.items():
            if ind_id.startswith("unknown_"):
                # Skip validation for unlinked samples
                continue

            # Check organism consistency
            organisms = set(s.organism_name for s in ind_samples if s.organism_name)
            if len(organisms) > 1:
                result.add_warning(
                    f"Individual {ind_id}: Inconsistent organism_name across samples: {organisms}"
                )

            # Check for duplicate timepoints
            timepoints = [s.timepoint for s in ind_samples if s.timepoint]
            if timepoints and len(timepoints) != len(set(timepoints)):
                duplicates = [t for t in timepoints if timepoints.count(t) > 1]
                result.add_warning(
                    f"Individual {ind_id}: Duplicate timepoints detected: {set(duplicates)}"
                )

            # Check library strategy consistency
            strategies = set(
                s.library_strategy for s in ind_samples if s.library_strategy
            )
            if len(strategies) > 1:
                result.add_warning(
                    f"Individual {ind_id}: Multiple library strategies: {strategies}"
                )

            # Check platform consistency
            platforms = set(s.instrument for s in ind_samples if s.instrument)
            if len(platforms) > 1:
                result.add_info(
                    f"Individual {ind_id}: Multiple platforms used: {platforms}"
                )

        # Summary
        if not result.warnings:
            result.add_info(f"All {len(groups)} individuals have consistent metadata")

        return result

    def get_longitudinal_summary(
        self,
        samples: List[SRASampleSchema],
    ) -> Dict[str, Any]:
        """
        Generate summary statistics for longitudinal analysis.

        Provides insights into:
        - Number of individuals with multiple timepoints
        - Distribution of timepoints per individual
        - Coverage of timepoint ranges

        Args:
            samples: List of validated SRASampleSchema instances

        Returns:
            Dictionary with longitudinal statistics

        Examples:
            >>> stats = service.get_longitudinal_summary(samples)
            >>> stats["individuals_with_multiple_timepoints"]
            45
            >>> stats["avg_samples_per_individual"]
            3.2
        """
        groups = self.group_by_individual(samples)

        # Samples per individual distribution
        samples_per_individual = {
            ind_id: len(samps) for ind_id, samps in groups.items()
        }

        # Individuals with multiple timepoints
        multi_timepoint_individuals = [
            ind_id
            for ind_id, samps in groups.items()
            if len(samps) > 1 and not ind_id.startswith("unknown_")
        ]

        # Timepoint range analysis
        timepoint_ranges = {}
        for ind_id, samps in groups.items():
            numeric_timepoints = [
                s.timepoint_numeric for s in samps if s.timepoint_numeric is not None
            ]
            if numeric_timepoints:
                timepoint_ranges[ind_id] = {
                    "min": min(numeric_timepoints),
                    "max": max(numeric_timepoints),
                    "range": max(numeric_timepoints) - min(numeric_timepoints),
                    "count": len(numeric_timepoints),
                }

        # Calculate statistics
        known_individuals = [
            ind_id for ind_id in groups if not ind_id.startswith("unknown_")
        ]
        unknown_samples = sum(
            len(samps)
            for ind_id, samps in groups.items()
            if ind_id.startswith("unknown_")
        )

        return {
            "total_samples": len(samples),
            "total_individuals": len(groups),
            "known_individuals": len(known_individuals),
            "unknown_individuals": len(groups) - len(known_individuals),
            "unknown_samples": unknown_samples,
            "individuals_with_multiple_timepoints": len(multi_timepoint_individuals),
            "avg_samples_per_individual": (
                sum(samples_per_individual.values()) / len(groups) if groups else 0
            ),
            "max_samples_per_individual": (
                max(samples_per_individual.values()) if groups else 0
            ),
            "samples_per_individual_distribution": dict(samples_per_individual),
            "timepoint_ranges": timepoint_ranges,
        }

    def find_individuals_missing_baseline(
        self,
        samples: List[SRASampleSchema],
        baseline_values: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Find individuals who don't have a baseline sample.

        Baseline is defined as timepoint_numeric = 0 or the minimum timepoint.

        Args:
            samples: List of validated SRASampleSchema instances
            baseline_values: List of timepoint values considered baseline (default: [0])

        Returns:
            List of individual_ids missing baseline samples
        """
        if baseline_values is None:
            baseline_values = [0, 0.0]

        groups = self.group_by_individual(samples)
        missing_baseline = []

        for ind_id, ind_samples in groups.items():
            if ind_id.startswith("unknown_"):
                continue

            # Check if any sample has baseline timepoint
            has_baseline = any(
                s.timepoint_numeric in baseline_values
                for s in ind_samples
                if s.timepoint_numeric is not None
            )

            if not has_baseline and len(ind_samples) > 1:
                # Only flag if individual has multiple samples (longitudinal)
                missing_baseline.append(ind_id)

        return missing_baseline

    def merge_individual_metadata(
        self,
        samples: List[SRASampleSchema],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge metadata across samples from the same individual.

        For longitudinal data, individual-level metadata (age, sex, disease_status)
        should be consistent. This function identifies and reports any
        discrepancies while creating a unified individual profile.

        Args:
            samples: List of validated SRASampleSchema instances

        Returns:
            Dict mapping individual_id to merged metadata profile
        """
        groups = self.group_by_individual(samples)
        individual_profiles = {}

        for ind_id, ind_samples in groups.items():
            if ind_id.startswith("unknown_"):
                continue

            # Collect all values for individual-level fields
            profile = {
                "individual_id": ind_id,
                "sample_count": len(ind_samples),
                "run_accessions": [s.run_accession for s in ind_samples],
            }

            # Age - should be same across samples
            ages = set(s.age for s in ind_samples if s.age)
            profile["age"] = list(ages)[0] if len(ages) == 1 else None
            profile["age_conflict"] = len(ages) > 1

            # Sex - should be same across samples
            sexes = set(s.sex for s in ind_samples if s.sex)
            profile["sex"] = list(sexes)[0] if len(sexes) == 1 else None
            profile["sex_conflict"] = len(sexes) > 1

            # Health status - may change over time
            health_statuses = [s.health_status for s in ind_samples if s.health_status]
            profile["health_statuses"] = list(set(health_statuses))

            # Timepoints
            timepoints = sorted(
                [
                    s.timepoint_numeric
                    for s in ind_samples
                    if s.timepoint_numeric is not None
                ]
            )
            profile["timepoints"] = timepoints
            profile["timepoint_range"] = (
                timepoints[-1] - timepoints[0] if len(timepoints) >= 2 else 0
            )

            individual_profiles[ind_id] = profile

        return individual_profiles
