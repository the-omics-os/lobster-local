"""
SRA sample metadata schema for validation and harmonization.

This module provides Pydantic schemas for validating SRA sample metadata
extracted from NCBI's SRA Run Selector API. It follows the established
pattern from transcriptomics.py, proteomics.py, and metagenomics.py.

The schema is modality-agnostic and supports all library strategies:
- AMPLICON (16S, ITS, etc.)
- RNA-Seq (bulk, single-cell)
- WGS (whole genome shotgun)
- ChIP-Seq, ATAC-seq, etc.

Used by metadata_assistant agent for:
- Sample extraction validation (prevent malformed data)
- Pre-download validation (ensure URLs exist)
- Cross-database harmonization
"""

import logging
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator

from lobster.core.interfaces.validator import ValidationResult

logger = logging.getLogger(__name__)


# =============================================================================
# Heuristic Extraction Candidates
# =============================================================================
# These field name candidates are used to automatically extract individual-level
# metadata from the diverse field naming conventions across SRA datasets.

INDIVIDUAL_ID_CANDIDATES = [
    "pid",
    "participant_id",
    "host_subject_id",
    "subject_id",
    "patient_id",
    "donor_id",
    "individual_id",
    "subject",
    "individual",
    "sample_name",  # Sometimes contains individual ID
    "host_id",
    "person_id",
]

TIMEPOINT_CANDIDATES = [
    "sample_day",
    "timepoint",
    "time_point",
    "tp",
    "visit",
    "day",
    "collection_day",
    "sampling_day",
    "time-point",
    "visit_number",
    "week",
    "month",
    "follow_up",
]

AGE_CANDIDATES = [
    "age",
    "host_age",
    "age_years",
    "age_at_collection",
    "patient_age",
    "subject_age",
]

SEX_CANDIDATES = [
    "sex",
    "gender",
    "host_sex",
    "patient_sex",
    "subject_sex",
]

HEALTH_STATUS_CANDIDATES = [
    "health_status",
    "disease",
    "disease_status",
    "diagnosis",
    "condition",
    "health_state",
    "disease_state",
    "phenotype",
    "case_control",
    "group",
]

BODY_SITE_CANDIDATES = [
    "body_site",
    "body_product",
    "tissue",
    "sample_type",
    "isolation_source",
    "sample_source",
]


# =============================================================================
# Sample Quality Flags
# =============================================================================


class SampleQualityFlag(str, Enum):
    """
    Flags indicating sample quality/usability issues.

    These flags are used for soft filtering - samples are not excluded,
    but flagged for user review. Each flag represents a potential data
    quality concern or filtering criterion.
    """

    # Missing critical metadata
    MISSING_INDIVIDUAL_ID = "missing_individual_id"
    MISSING_TIMEPOINT = "missing_timepoint"
    MISSING_BODY_SITE = "missing_body_site"
    MISSING_HEALTH_STATUS = "missing_health_status"

    # Data quality concerns
    NO_DOWNLOAD_URL = "no_download_url"
    INCOMPLETE_METADATA = "incomplete_metadata"
    CONFLICTING_INDIVIDUAL_IDS = "conflicting_individual_ids"

    # Filtering criteria (soft flags - user decides)
    NON_HUMAN_HOST = "non_human_host"
    CONTROL_SAMPLE = "control_sample"
    MOCK_COMMUNITY = "mock_community"
    NEGATIVE_CONTROL = "negative_control"
    ENVIRONMENTAL_SAMPLE = "environmental_sample"


class SRASampleSchema(BaseModel):
    """
    Pydantic schema for SRA sample metadata validation.

    This schema validates raw SRA sample records extracted from NCBI's
    SRA Run Selector API. It's modality-agnostic and supports all library
    strategies (AMPLICON, RNA-Seq, WGS, ChIP-Seq, etc.).

    SRA samples contain 71 fields from NCBI. This schema explicitly defines
    critical fields and stores the remaining fields in additional_metadata.

    Attributes:
        run_accession: SRA run accession (SRR*)
        experiment_accession: SRA experiment accession (SRX*)
        sample_accession: SRA sample accession (SRS*)
        study_accession: SRA study accession (SRP*)
        bioproject: BioProject accession (PRJ*)
        biosample: BioSample accession (SAM*)
        library_strategy: Sequencing strategy (AMPLICON, RNA-Seq, etc.)
        library_source: Library source (GENOMIC, TRANSCRIPTOMIC, etc.)
        library_selection: Library selection method (PCR, RANDOM, etc.)
        library_layout: SINGLE or PAIRED
        organism_name: Organism name (e.g., "Homo sapiens")
        organism_taxid: NCBI Taxonomy ID
        instrument: Sequencing instrument
        instrument_model: Instrument model
        public_url: NCBI public download URL
        ncbi_url: NCBI direct download URL
        aws_url: AWS S3 download URL
        gcp_url: GCP download URL
        study_title: Study title (optional)
        experiment_title: Experiment title (optional)
        sample_title: Sample title (optional)
        env_medium: Environmental medium (optional, critical for microbiome)
        env_broad_scale: Environmental broad scale (optional)
        env_local_scale: Environmental local scale (optional)
        collection_date: Sample collection date (optional)
        geo_loc_name: Geographic location (optional)
        total_spots: Total spots (optional)
        total_size: Total size in bytes (optional)
        run_total_spots: Run total spots (optional)
        run_total_bases: Run total bases (optional)
        additional_metadata: All other SRA fields (71 total)

    Examples:
        >>> sample_dict = {
        ...     "run_accession": "SRR21960766",
        ...     "experiment_accession": "SRX17944370",
        ...     "sample_accession": "SRS15461891",
        ...     "study_accession": "SRP403291",
        ...     "bioproject": "PRJNA891765",
        ...     "biosample": "SAMN31357800",
        ...     "library_strategy": "AMPLICON",
        ...     "library_source": "METAGENOMIC",
        ...     "library_selection": "PCR",
        ...     "library_layout": "PAIRED",
        ...     "organism_name": "human metagenome",
        ...     "organism_taxid": "646099",
        ...     "instrument": "Illumina MiSeq",
        ...     "instrument_model": "Illumina MiSeq",
        ...     "public_url": "https://sra-downloadb.be-md.ncbi.nlm.nih.gov/...",
        ...     # ... 50+ additional fields
        ... }
        >>> validated = SRASampleSchema.from_dict(sample_dict)
        >>> validated.has_download_url()
        True
    """

    # Core SRA identifiers (REQUIRED)
    run_accession: str = Field(..., description="SRA run accession (SRR*)")
    experiment_accession: str = Field(..., description="SRA experiment (SRX*)")
    sample_accession: str = Field(..., description="SRA sample (SRS*)")
    study_accession: str = Field(..., description="SRA study (SRP*)")

    # BioProject/BioSample linkage (REQUIRED)
    bioproject: str = Field(..., description="BioProject accession (PRJ*)")
    biosample: str = Field(..., description="BioSample accession (SAM*)")

    # Library metadata (REQUIRED)
    library_strategy: str = Field(..., description="Sequencing strategy")
    library_source: str = Field(..., description="Library source")
    library_selection: str = Field(..., description="Library selection method")
    library_layout: str = Field(..., description="SINGLE or PAIRED")

    # Organism (REQUIRED)
    organism_name: str = Field(..., description="Organism name")
    organism_taxid: str = Field(..., description="NCBI Taxonomy ID")

    # Instrument (REQUIRED)
    instrument: str = Field(..., description="Sequencing instrument")
    instrument_model: str = Field(..., description="Instrument model")

    # Download URLs (at least ONE required - validated separately)
    public_url: Optional[str] = Field(None, description="NCBI public URL")
    ncbi_url: Optional[str] = Field(None, description="NCBI direct URL")
    aws_url: Optional[str] = Field(None, description="AWS S3 URL")
    gcp_url: Optional[str] = Field(None, description="GCP URL")

    # Study metadata (OPTIONAL)
    study_title: Optional[str] = None
    experiment_title: Optional[str] = None
    sample_title: Optional[str] = None

    # Environmental context (OPTIONAL - critical for microbiome)
    env_medium: Optional[str] = Field(None, description="Environmental medium")
    env_broad_scale: Optional[str] = None
    env_local_scale: Optional[str] = None
    collection_date: Optional[str] = None
    geo_loc_name: Optional[str] = None

    # Sequencing metrics (OPTIONAL)
    total_spots: Optional[str] = None
    total_size: Optional[str] = None
    run_total_spots: Optional[str] = None
    run_total_bases: Optional[str] = None

    # =========================================================================
    # Individual/Subject Tracking (OPTIONAL - heuristically extracted)
    # =========================================================================
    # These fields are auto-populated from common SRA attribute names
    # (pid, host_subject_id, sample_day, etc.) during from_dict() processing.

    individual_id: Optional[str] = Field(
        None,
        description="Subject/patient identifier linking multiple samples to one individual. "
        "Auto-extracted from fields like pid, host_subject_id, participant_id.",
    )
    individual_id_sources: Optional[Dict[str, str]] = Field(
        None,
        description="All individual ID fields found in source data (for conflict detection). "
        "Maps field name to value, e.g., {'pid': 'P042', 'host_subject_id': 'P042'}.",
    )
    timepoint: Optional[str] = Field(
        None,
        description="Timepoint identifier (e.g., 'baseline', 'day13', 'week4'). "
        "Auto-extracted from fields like sample_day, timepoint, tp, visit.",
    )
    timepoint_numeric: Optional[float] = Field(
        None,
        description="Numeric timepoint value for ordering (e.g., days from baseline). "
        "Parsed from timepoint string when possible.",
    )

    # =========================================================================
    # Clinical/Demographic (OPTIONAL - from manuscript or SRA attributes)
    # =========================================================================

    age: Optional[str] = Field(
        None, description="Subject age (may be range like '30-40' or exact)"
    )
    sex: Optional[str] = Field(
        None, description="Subject sex (M/F/male/female/Unknown)"
    )
    health_status: Optional[str] = Field(
        None, description="Health/disease status (e.g., 'CRC', 'healthy', 'IBD')"
    )
    disease_stage: Optional[str] = Field(
        None, description="Disease stage or severity (e.g., 'Stage III', 'mild')"
    )
    treatment: Optional[str] = Field(
        None, description="Treatment received (e.g., 'chemotherapy', 'untreated')"
    )
    body_site: Optional[str] = Field(
        None,
        description="Standardized body site (e.g., 'fecal', 'biopsy', 'oral'). "
        "Complements env_medium with more specific anatomical location.",
    )

    # Additional fields stored in additional_metadata
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="All other SRA fields (71 total fields)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "run_accession": "SRR21960766",
                "experiment_accession": "SRX17944370",
                "sample_accession": "SRS15461891",
                "study_accession": "SRP403291",
                "bioproject": "PRJNA891765",
                "biosample": "SAMN31357800",
                "library_strategy": "AMPLICON",
                "library_source": "METAGENOMIC",
                "library_selection": "PCR",
                "library_layout": "PAIRED",
                "organism_name": "human metagenome",
                "organism_taxid": "646099",
                "instrument": "Illumina MiSeq",
                "instrument_model": "Illumina MiSeq",
                "public_url": "https://sra-downloadb.be-md.ncbi.nlm.nih.gov/...",
                "env_medium": "Stool",
                "collection_date": "2017",
            }
        }

    @field_validator("library_strategy")
    @classmethod
    def validate_library_strategy(cls, v: str) -> str:
        """
        Validate library strategy is recognized.

        Logs a warning for uncommon strategies but allows them.
        """
        # Common strategies - not exhaustive
        known_strategies = {
            "AMPLICON",
            "RNA-Seq",
            "WGS",
            "WXS",
            "ChIP-Seq",
            "ATAC-seq",
            "Bisulfite-Seq",
            "Hi-C",
            "FAIRE-seq",
            "MBD-Seq",
            "MRE-Seq",
            "MeDIP-Seq",
            "DNase-Hypersensitivity",
            "Tn-Seq",
            "VALIDATION",
            "OTHER",
        }
        if v not in known_strategies:
            logger.warning(f"Uncommon library_strategy: '{v}'")
        return v

    @field_validator("library_layout")
    @classmethod
    def validate_library_layout(cls, v: str) -> str:
        """
        Validate library layout is SINGLE or PAIRED.

        Raises:
            ValueError: If layout is not SINGLE or PAIRED
        """
        if v.upper() not in {"SINGLE", "PAIRED"}:
            raise ValueError(f"library_layout must be 'SINGLE' or 'PAIRED', got '{v}'")
        return v.upper()

    @field_validator("run_accession")
    @classmethod
    def validate_run_accession(cls, v: str) -> str:
        """Validate run accession format (SRR* or ERR* or DRR*)."""
        if not re.match(r"^[SED]RR\d+$", v):
            logger.warning(f"Unexpected run_accession format: '{v}'")
        return v

    @field_validator("bioproject")
    @classmethod
    def validate_bioproject(cls, v: str) -> str:
        """Validate BioProject accession format (PRJNA*, PRJEB*, PRJDB*)."""
        if not re.match(r"^PRJ[NED][A-Z]\d+$", v):
            logger.warning(f"Unexpected bioproject format: '{v}'")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SRASampleSchema":
        """
        Create schema from dict, handling 71+ fields flexibly with heuristic extraction.

        Extracts known fields into Pydantic fields, applies heuristic extraction
        for individual-level metadata (individual_id, timepoint, age, sex, etc.),
        and stores remaining fields in additional_metadata.

        Heuristic Extraction:
            - individual_id: Extracted from pid, host_subject_id, participant_id, etc.
            - timepoint: Extracted from sample_day, timepoint, tp, visit, etc.
            - timepoint_numeric: Parsed from timepoint string when possible
            - age, sex, health_status, body_site: From common field name variants

        Conflict Detection:
            When multiple individual ID fields exist with different values,
            all are stored in individual_id_sources for manual review.

        Args:
            data: SRA sample dictionary (71 fields from NCBI API)

        Returns:
            SRASampleSchema: Validated schema instance with heuristically extracted fields

        Examples:
            >>> sample = {
            ...     "run_accession": "SRR001",
            ...     "pid": "P042",
            ...     "sample_day": "13",
            ...     ...
            ... }
            >>> validated = SRASampleSchema.from_dict(sample)
            >>> validated.individual_id
            'P042'
            >>> validated.timepoint
            '13'
            >>> validated.timepoint_numeric
            13.0
        """
        known_fields = set(cls.model_fields.keys()) - {
            "additional_metadata",
            "individual_id",
            "individual_id_sources",
            "timepoint",
            "timepoint_numeric",
            "age",
            "sex",
            "health_status",
            "body_site",
        }
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # =====================================================================
        # Heuristic Extraction for Individual ID (with conflict detection)
        # =====================================================================
        found_individual_ids: Dict[str, str] = {}
        for candidate in INDIVIDUAL_ID_CANDIDATES:
            if candidate in data and data[candidate]:
                found_individual_ids[candidate] = str(data[candidate])

        if found_individual_ids:
            # Use first match as primary
            primary_field = next(iter(found_individual_ids))
            schema_data["individual_id"] = found_individual_ids[primary_field]

            # Store all sources for conflict detection
            unique_values = set(found_individual_ids.values())
            if len(unique_values) > 1 or len(found_individual_ids) > 1:
                # Multiple fields found - store all for review
                schema_data["individual_id_sources"] = found_individual_ids

        # =====================================================================
        # Heuristic Extraction for Timepoint
        # =====================================================================
        for candidate in TIMEPOINT_CANDIDATES:
            if candidate in data and data[candidate]:
                timepoint_str = str(data[candidate])
                schema_data["timepoint"] = timepoint_str

                # Try to extract numeric value
                try:
                    # Look for numbers in the string
                    match = re.search(r"-?\d+\.?\d*", timepoint_str)
                    if match:
                        schema_data["timepoint_numeric"] = float(match.group())
                except (ValueError, AttributeError):
                    pass  # Could not parse numeric value
                break

        # =====================================================================
        # Heuristic Extraction for Clinical/Demographic Fields
        # =====================================================================
        for candidate in AGE_CANDIDATES:
            if candidate in data and data[candidate]:
                schema_data["age"] = str(data[candidate])
                break

        for candidate in SEX_CANDIDATES:
            if candidate in data and data[candidate]:
                schema_data["sex"] = str(data[candidate])
                break

        for candidate in HEALTH_STATUS_CANDIDATES:
            if candidate in data and data[candidate]:
                schema_data["health_status"] = str(data[candidate])
                break

        for candidate in BODY_SITE_CANDIDATES:
            if candidate in data and data[candidate]:
                schema_data["body_site"] = str(data[candidate])
                break

        # =====================================================================
        # Store remaining fields in additional_metadata
        # =====================================================================
        # Exclude both known_fields AND heuristic candidate fields from additional
        heuristic_candidates = set(
            INDIVIDUAL_ID_CANDIDATES
            + TIMEPOINT_CANDIDATES
            + AGE_CANDIDATES
            + SEX_CANDIDATES
            + HEALTH_STATUS_CANDIDATES
            + BODY_SITE_CANDIDATES
        )
        all_known = (
            known_fields
            | heuristic_candidates
            | {
                "individual_id",
                "individual_id_sources",
                "timepoint",
                "timepoint_numeric",
                "age",
                "sex",
                "health_status",
                "body_site",
            }
        )

        additional = {k: v for k, v in data.items() if k not in all_known}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)

    def has_download_url(self) -> bool:
        """
        Check if at least one download URL is available.

        Returns:
            bool: True if any download URL is present

        Examples:
            >>> sample = SRASampleSchema(...)
            >>> sample.has_download_url()
            True
        """
        return bool(self.public_url or self.ncbi_url or self.aws_url or self.gcp_url)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation, including additional_metadata.

        Returns:
            Dict[str, Any]: Complete sample dictionary with all 71 fields

        Examples:
            >>> validated = SRASampleSchema.from_dict(sample_dict)
            >>> reconstructed = validated.to_dict()
            >>> len(reconstructed)
            71  # All original fields preserved
        """
        base_dict = self.model_dump(exclude={"additional_metadata"}, exclude_none=True)
        if self.additional_metadata:
            base_dict.update(self.additional_metadata)
        return base_dict


def validate_sra_sample(sample: Dict[str, Any]) -> ValidationResult:
    """
    Validate single SRA sample using Pydantic schema.

    Returns ValidationResult following existing pattern from validation.py.
    Uses errors for critical issues, warnings for non-critical issues.

    Args:
        sample: SRA sample dictionary (71 fields from NCBI API)

    Returns:
        ValidationResult with errors/warnings/info

    Examples:
        >>> sample = {"run_accession": "SRR001", "library_strategy": "AMPLICON", ...}
        >>> result = validate_sra_sample(sample)
        >>> result.is_valid  # True if no errors
        True
        >>> len(result.warnings)  # May have warnings
        1
        >>> result.summary()
        'Validation completed with 1 warning(s)'
    """
    result = ValidationResult()

    try:
        validated = SRASampleSchema.from_dict(sample)

        # Critical check: At least one download URL must be present
        if not validated.has_download_url():
            result.add_error(
                f"Sample {validated.run_accession}: No download URLs available. "
                f"At least one of (public_url, ncbi_url, aws_url, gcp_url) is required."
            )

        # Warn about missing environmental context (important for microbiome filtering)
        # Check both env_medium AND body_site since sample type info may be in either field
        # (e.g., MIxS uses env_medium, clinical samples often use tissue → body_site)
        if validated.library_strategy == "AMPLICON":
            if not (validated.env_medium or validated.body_site):
                result.add_warning(
                    f"Sample {validated.run_accession}: Missing sample type info. "
                    f"Neither 'env_medium' nor 'body_site' found (checked: tissue, isolation_source, etc.). "
                    f"This field is important for microbiome filtering (fecal vs tissue samples)."
                )

        # Debug: successful validation (changed from INFO to reduce log pollution)
        logger.debug(
            f"Sample {validated.run_accession} validated successfully "
            f"(library_strategy: {validated.library_strategy})"
        )

    except ValidationError as e:
        # Pydantic validation failed - critical errors
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            result.add_error(f"Field '{field}': {msg}")

    except Exception as e:
        # Unexpected error
        result.add_error(f"Unexpected validation error: {str(e)}")
        logger.error(f"Unexpected error validating SRA sample: {e}", exc_info=True)

    return result


def validate_sra_samples_batch(samples: List[Dict[str, Any]]) -> ValidationResult:
    """
    Validate list of SRA samples and return aggregated results.

    This function validates each sample individually and merges all
    ValidationResults into a single aggregated result with batch-level
    statistics.

    Args:
        samples: List of SRA sample dictionaries

    Returns:
        ValidationResult: Aggregated validation results with batch statistics

    Examples:
        >>> samples = [
        ...     {"run_accession": "SRR001", ...},
        ...     {"run_accession": "SRR002", ...},
        ...     {"run_accession": "SRR003", ...}  # Missing required field
        ... ]
        >>> result = validate_sra_samples_batch(samples)
        >>> result.metadata["total_samples"]
        3
        >>> result.metadata["valid_samples"]
        2
        >>> result.metadata["validation_rate"]
        66.67
    """
    aggregated = ValidationResult()

    # Validate each sample
    for idx, sample in enumerate(samples):
        sample_result = validate_sra_sample(sample)
        aggregated = aggregated.merge(sample_result)

    # Add batch-level statistics
    valid_samples = len(samples) - len(
        [e for e in aggregated.errors if "Field" in e or "No download URLs" in e]
    )

    aggregated.metadata["total_samples"] = len(samples)
    aggregated.metadata["valid_samples"] = valid_samples
    aggregated.metadata["validation_rate"] = (
        (valid_samples / len(samples) * 100) if samples else 0.0
    )
    aggregated.metadata["error_count"] = len(aggregated.errors)
    aggregated.metadata["warning_count"] = len(aggregated.warnings)

    # Add summary info message
    if samples:
        aggregated.add_info(
            f"Batch validation complete: {valid_samples}/{len(samples)} samples valid "
            f"({aggregated.metadata['validation_rate']:.1f}%)"
        )

    return aggregated


def is_valid_sra_sample_key(ws_key: str) -> tuple[bool, Optional[str]]:
    """
    Validate SRA sample workspace key format.

    Validates that workspace key follows the expected pattern:
    'sra_<bioproject_id>_samples' where bioproject_id matches PRJNA*/PRJEB*/PRJDB*.

    Args:
        ws_key: Workspace key string (e.g., "sra_PRJNA891765_samples")

    Returns:
        (is_valid, reason): True if valid, reason string if invalid

    Examples:
        >>> is_valid_sra_sample_key("sra_PRJNA891765_samples")
        (True, None)
        >>> is_valid_sra_sample_key("pub_queue_doi_123_metadata.json")
        (False, "does not start with 'sra_'")
        >>> is_valid_sra_sample_key("sra_INVALID_samples")
        (False, "invalid project ID format: INVALID")
    """
    if not ws_key.startswith("sra_"):
        return False, "does not start with 'sra_'"

    if not ws_key.endswith("_samples"):
        return False, "does not end with '_samples'"

    # Extract project ID (e.g., "PRJNA891765" from "sra_PRJNA891765_samples")
    parts = ws_key.split("_")
    if len(parts) < 3:
        return False, "invalid format (expected 'sra_<project>_samples')"

    project_id = "_".join(parts[1:-1])  # Handle multi-part IDs

    # Validate project ID format (PRJNA, PRJEB, PRJDB + digits)
    # Common patterns:
    # - PRJNA891765 (NCBI)
    # - PRJEB12345 (ENA/EBI)
    # - PRJDB6789 (DDBJ)
    if not re.match(r"^PRJ[NED][A-Z]\d+$", project_id):
        return False, f"invalid project ID format: {project_id}"

    return True, None


# =============================================================================
# Sample Completeness Scoring
# =============================================================================


def compute_sample_completeness(
    sample: SRASampleSchema,
) -> Tuple[float, List[SampleQualityFlag]]:
    """
    Compute completeness score (0-100) and list of quality flags for a sample.

    This function evaluates sample metadata quality based on:
    1. Presence of critical fields (individual_id, timepoint, body_site)
    2. Download URL availability (critical for processing)
    3. Detection of potential filtering criteria (controls, non-human samples)

    Scoring weights:
    - download_url: 30 points (critical - cannot process without it)
    - individual_id: 20 points (critical for longitudinal analysis)
    - timepoint: 15 points (important for longitudinal studies)
    - body_site/env_medium: 15 points (important for filtering)
    - health_status: 10 points (useful but often missing)
    - age/sex: 5 points each (nice to have)

    Flags are for soft filtering - samples are not excluded, but flagged
    for user review.

    Args:
        sample: Validated SRASampleSchema instance

    Returns:
        Tuple of (score, flags):
        - score: Float 0-100 indicating metadata completeness
        - flags: List of SampleQualityFlag indicating quality concerns

    Examples:
        >>> sample = SRASampleSchema.from_dict({...})
        >>> score, flags = compute_sample_completeness(sample)
        >>> score
        65.0
        >>> flags
        [<SampleQualityFlag.MISSING_INDIVIDUAL_ID: 'missing_individual_id'>]
    """
    flags: List[SampleQualityFlag] = []
    score = 100.0

    # =========================================================================
    # Critical Field Checks (weighted scoring)
    # =========================================================================

    # Download URL (30 points) - Critical
    if not sample.has_download_url():
        flags.append(SampleQualityFlag.NO_DOWNLOAD_URL)
        score -= 30

    # Individual ID (20 points) - Critical for longitudinal
    if not sample.individual_id:
        flags.append(SampleQualityFlag.MISSING_INDIVIDUAL_ID)
        score -= 20

    # Timepoint (15 points) - Important for longitudinal
    if not sample.timepoint:
        flags.append(SampleQualityFlag.MISSING_TIMEPOINT)
        score -= 15

    # Body site / Environmental medium (15 points)
    if not (sample.env_medium or sample.body_site):
        flags.append(SampleQualityFlag.MISSING_BODY_SITE)
        score -= 15

    # Health status (10 points)
    if not sample.health_status:
        flags.append(SampleQualityFlag.MISSING_HEALTH_STATUS)
        score -= 10

    # Age (5 points)
    if not sample.age:
        score -= 5

    # Sex (5 points)
    if not sample.sex:
        score -= 5

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    if sample.individual_id_sources:
        unique_values = set(sample.individual_id_sources.values())
        if len(unique_values) > 1:
            flags.append(SampleQualityFlag.CONFLICTING_INDIVIDUAL_IDS)
            # Don't reduce score - this is a warning, not a quality issue

    # =========================================================================
    # Soft Filtering Flags (don't affect score - user decides)
    # =========================================================================

    # Non-human host detection
    if sample.organism_name:
        organism_lower = sample.organism_name.lower()
        if "sapiens" not in organism_lower and "human" not in organism_lower:
            # Check if it's a human-associated metagenome (e.g., "human metagenome")
            # which is valid for microbiome studies
            if "metagenome" not in organism_lower:
                flags.append(SampleQualityFlag.NON_HUMAN_HOST)

    # Control sample detection (check sample_title and experiment_title)
    control_keywords = ["control", "blank", "negative", "empty", "water"]
    mock_keywords = ["mock", "zymo", "synthetic", "artificial"]

    titles_to_check = [
        sample.sample_title or "",
        sample.experiment_title or "",
        sample.study_title or "",
    ]
    combined_title = " ".join(titles_to_check).lower()

    for keyword in control_keywords:
        if keyword in combined_title:
            flags.append(SampleQualityFlag.CONTROL_SAMPLE)
            break

    for keyword in mock_keywords:
        if keyword in combined_title:
            flags.append(SampleQualityFlag.MOCK_COMMUNITY)
            break

    # Environmental sample detection (not host-associated)
    if sample.env_medium:
        env_lower = sample.env_medium.lower()
        environmental_keywords = ["soil", "water", "sediment", "air", "ocean", "river"]
        for keyword in environmental_keywords:
            if keyword in env_lower:
                flags.append(SampleQualityFlag.ENVIRONMENTAL_SAMPLE)
                break

    # =========================================================================
    # Overall incompleteness flag
    # =========================================================================

    if score < 50:
        flags.append(SampleQualityFlag.INCOMPLETE_METADATA)

    return max(score, 0.0), flags


def compute_batch_completeness_stats(
    samples: List[SRASampleSchema],
) -> Dict[str, Any]:
    """
    Compute completeness statistics for a batch of samples.

    This function aggregates completeness scores and flags across all samples,
    providing a summary useful for quality assessment and filtering decisions.

    Args:
        samples: List of validated SRASampleSchema instances

    Returns:
        Dictionary containing:
        - total_samples: Total number of samples
        - avg_completeness: Average completeness score (0-100)
        - completeness_distribution: {"high": [...], "medium": [...], "low": [...]}
        - flag_counts: Count of each flag type across all samples
        - unique_individuals: Number of unique individual_ids
        - samples_per_individual: Distribution of samples per individual
        - flagged_sample_ids: Dict mapping flag name to list of run_accessions

    Examples:
        >>> samples = [SRASampleSchema.from_dict(s) for s in raw_samples]
        >>> stats = compute_batch_completeness_stats(samples)
        >>> stats["avg_completeness"]
        72.5
        >>> stats["unique_individuals"]
        45
    """
    if not samples:
        return {
            "total_samples": 0,
            "avg_completeness": 0.0,
            "completeness_distribution": {"high": [], "medium": [], "low": []},
            "flag_counts": {},
            "unique_individuals": 0,
            "samples_per_individual": {},
            "flagged_sample_ids": {},
        }

    scores: List[float] = []
    completeness_distribution: Dict[str, List[str]] = {
        "high": [],  # >= 80
        "medium": [],  # 50-79
        "low": [],  # < 50
    }
    flag_counts: Dict[str, int] = {}
    flagged_sample_ids: Dict[str, List[str]] = {}
    individuals: Dict[str, List[str]] = {}  # individual_id -> [run_accessions]

    for sample in samples:
        score, flags = compute_sample_completeness(sample)
        scores.append(score)

        # Categorize by completeness
        if score >= 80:
            completeness_distribution["high"].append(sample.run_accession)
        elif score >= 50:
            completeness_distribution["medium"].append(sample.run_accession)
        else:
            completeness_distribution["low"].append(sample.run_accession)

        # Count flags
        for flag in flags:
            flag_name = flag.value
            flag_counts[flag_name] = flag_counts.get(flag_name, 0) + 1
            if flag_name not in flagged_sample_ids:
                flagged_sample_ids[flag_name] = []
            flagged_sample_ids[flag_name].append(sample.run_accession)

        # Track individuals
        ind_id = sample.individual_id or f"unknown_{sample.biosample}"
        if ind_id not in individuals:
            individuals[ind_id] = []
        individuals[ind_id].append(sample.run_accession)

    # Compute samples per individual distribution
    samples_per_individual: Dict[str, int] = {}
    for ind_id, run_accessions in individuals.items():
        count = len(run_accessions)
        key = str(count) if count <= 5 else "6+"
        samples_per_individual[key] = samples_per_individual.get(key, 0) + 1

    return {
        "total_samples": len(samples),
        "avg_completeness": sum(scores) / len(scores),
        "completeness_distribution": completeness_distribution,
        "flag_counts": flag_counts,
        "unique_individuals": len(individuals),
        "samples_per_individual": samples_per_individual,
        "flagged_sample_ids": flagged_sample_ids,
    }
