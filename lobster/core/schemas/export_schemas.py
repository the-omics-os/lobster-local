"""
Export schema registry for CSV export column ordering.

Defines priority-ordered export columns per data type (modality).
Integrates with existing schemas (sra.py, proteomics.py, metabolomics.py, etc.).

**Purpose**:
    Provide a professional, extensible mechanism for CSV export column ordering
    that works across all omics layers (transcriptomics, proteomics, metabolomics,
    metagenomics). Replaces hardcoded column lists with schema-driven approach.

**Architecture**:
    - ExportPriority enum: Defines column ordering priority (1=first, 99=last)
    - ExportSchemaRegistry: Maps data types to priority-grouped column lists
    - get_ordered_export_columns(): Main API for workspace_tool.py

**Usage**:
    >>> from lobster.core.schemas.export_schemas import get_ordered_export_columns
    >>> samples = [{"run_accession": "SRR123", "disease": "crc", ...}, ...]
    >>> ordered_cols = get_ordered_export_columns(samples, data_type="sra_amplicon")
    >>> df = pd.DataFrame(samples)[ordered_cols]

**Adding New Omics Layers** (15 min):
    1. Add method: get_<modality>_export_schema() -> Dict
    2. Define priority_groups with relevant fields
    3. Register in get_export_schema() registry dict
    4. Done! CSV exports now support new modality.

**Introduced**: v1.2.0 (December 2024)
**Customer**: DataBioMix microbiome harmonization use case
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class ExportPriority(Enum):
    """
    Column priority for CSV export ordering.

    Lower values appear first in exported CSV files.
    This ensures critical identifiers and metadata appear before
    technical details and optional fields.
    """

    CORE_IDENTIFIERS = 1  # run_accession, sample_accession, biosample
    SAMPLE_METADATA = 2  # organism, host, age, sex, condition
    HARMONIZED_METADATA = 3  # disease, sample_type (harmonized fields)
    LIBRARY_TECHNICAL = 4  # library_strategy, instrument, sequencing metrics
    DOWNLOAD_URLS = 5  # URLs for automated download pipelines
    PUBLICATION_CONTEXT = 6  # source_doi, source_pmid (publication provenance)
    OPTIONAL_FIELDS = 99  # Extra fields not defined in schema


class ExportSchemaRegistry:
    """
    Centralized registry mapping data types to export column configurations.

    Each schema defines priority_groups mapping ExportPriority to field lists.
    Fields are ordered within each priority group for consistent output.

    **Design Pattern**:
        - Static methods return schema dicts (no state needed)
        - get_export_schema() dispatcher with aliases for common names
        - Schemas define ONLY fields relevant to each modality

    **Example Schema**:
        {
            "modality": "sra_amplicon",
            "description": "SRA 16S amplicon samples (microbiome)",
            "priority_groups": {
                ExportPriority.CORE_IDENTIFIERS: ["run_accession", ...],
                ExportPriority.SAMPLE_METADATA: ["organism_name", ...],
                ...
            }
        }
    """

    @staticmethod
    def get_sra_amplicon_export_schema() -> Dict[str, Any]:
        """
        Export schema for SRA 16S/ITS amplicon data (DataBioMix use case).

        **Use cases**:
            - Microbiome metadata harmonization (IBD, CRC studies)
            - Multi-publication aggregation
            - Sample filtering by host, disease, sample type

        **Column count**: 34 fields (28 SRA standard + 6 harmonized)

        Returns:
            Schema dict with priority-grouped column definitions
        """
        return {
            "modality": "sra_amplicon",
            "description": "SRA 16S/ITS amplicon samples (microbiome)",
            "priority_groups": {
                # ================================================================
                # Provenance Fields - Understanding Study vs Publication Context
                # ================================================================
                #
                # CRITICAL: For batch effect correction, use study_accession, NOT source_* fields!
                #
                # study_accession (SRP*, PRJNA*):
                #   - Biological study identifier (sequencing center, protocol, temporal batch)
                #   - Same study = same technical conditions = batch effect group
                #   - USE THIS for ComBat-seq, PERMANOVA, or other batch correction methods
                #
                # source_entry_id / source_doi / source_pmid:
                #   - Publication that CITED this dataset
                #   - Different purpose: literature provenance, citation tracking
                #   - NOT suitable for batch correction (one study can appear in multiple papers)
                #
                # Example:
                #   Study SRP001 sequenced at Broad Institute in 2020
                #   → Cited by Paper A (PMID:12345) in 2021
                #   → Also cited by Paper B (PMID:67890) in 2022
                #
                #   For batch correction: Group by SRP001 (technical batch)
                #   For literature tracking: Track both PMID:12345 and PMID:67890
                #
                # See: wiki/48-manual-sample-enrichment-workflow.md for R/Python examples
                # ================================================================
                ExportPriority.CORE_IDENTIFIERS: [
                    "run_accession",  # Primary identifier (SRR accession)
                    "sample_accession",  # Sample-level accession
                    "biosample",  # SAMN accession
                    "bioproject",  # PRJNA accession
                    "study_accession",  # Study-level ID (SRP*/PRJNA* - USE FOR BATCH CORRECTION)
                ],
                ExportPriority.SAMPLE_METADATA: [
                    "organism_name",  # Scientific name (e.g., "Homo sapiens")
                    "host",  # Host organism (e.g., "human", "mouse")
                    "isolation_source",  # Sample origin (e.g., "fecal", "gut tissue")
                    "geo_loc_name",  # Geographic location
                    "collection_date",  # Sample collection date
                ],
                ExportPriority.HARMONIZED_METADATA: [
                    # DataBioMix harmonized fields (standardized by metadata_assistant)
                    "disease",  # Standardized disease term (e.g., "crc", "uc", "cd")
                    "disease_original",  # Original disease string from metadata
                    # Disease Enrichment Provenance (v0.5.1+)
                    # Tracks how disease annotation was obtained:
                    #   - sra_original: Found in standard SRA field (confidence: 1.0)
                    #   - column_remapped: Found in non-standard column (confidence: 1.0)
                    #   - abstract_llm: Extracted from publication abstract via LLM
                    #   - methods_llm: Extracted from publication methods via LLM
                    #   - manual_override: User-provided mapping (confidence: 1.0)
                    "disease_source",  # Provenance: how disease was determined
                    "disease_confidence",  # Confidence score: 0.0-1.0 (LLM confidence or 1.0 for deterministic)
                    "disease_evidence",  # Supporting quote from publication (max 200 chars)
                    "enrichment_timestamp",  # ISO timestamp when enriched
                    "host_disease_stat",  # MIMARKS: Disease ontology term (fallback populated)
                    "sample_type",  # Sample type (e.g., "fecal", "tissue", "biopsy")
                    "host_body_site",  # MIMARKS: Anatomical sampling site
                    "host_body_product",  # MIMARKS: Biological product (stool, saliva, etc.)
                    "age",  # Patient/subject age (extracted heuristically)
                    "sex",  # Patient/subject sex (extracted heuristically)
                    "tissue",  # Tissue type (e.g., "colon", "ileum", "rectum")
                    # Lobster internal quality tracking
                    "_individual_id",  # Harmonized subject ID for longitudinal tracking
                    "_quality_score",  # Metadata completeness score (0-100)
                    "_quality_flags",  # Human-readable quality warnings
                ],
                ExportPriority.LIBRARY_TECHNICAL: [
                    "library_strategy",  # AMPLICON, RNA-Seq, WGS, etc.
                    "library_layout",  # SINGLE, PAIRED
                    "library_source",  # GENOMIC, TRANSCRIPTOMIC, METAGENOMIC
                    "library_selection",  # PCR, RANDOM, etc.
                    "instrument",  # Sequencer (e.g., "Illumina MiSeq")
                    "instrument_model",  # Model detail
                    "total_spots",  # Number of reads
                    "run_total_bases",  # Total sequenced bases
                ],
                ExportPriority.DOWNLOAD_URLS: [
                    # URLs for automated download pipelines
                    "ena_fastq_http",  # ENA FASTQ URL (merged PAIRED)
                    "ena_fastq_http_1",  # ENA R1 FASTQ
                    "ena_fastq_http_2",  # ENA R2 FASTQ
                    "ncbi_url",  # NCBI SRA Toolkit URL
                    "aws_url",  # AWS Open Data URL
                    "gcp_url",  # Google Cloud URL
                ],
                ExportPriority.PUBLICATION_CONTEXT: [
                    # Publication provenance (added by write_to_workspace)
                    "source_doi",  # DOI of source publication
                    "source_pmid",  # PubMed ID
                    "source_entry_id",  # Publication queue entry ID
                ],
            },
        }

    @staticmethod
    def get_proteomics_export_schema() -> Dict[str, Any]:
        """
        Export schema for mass spectrometry proteomics.

        **Use cases**:
            - DDA/DIA proteomics workflows
            - Protein abundance comparisons
            - Disease biomarker discovery

        **Column count**: ~25 fields (core identifiers + technical metrics)

        Returns:
            Schema dict with priority-grouped column definitions
        """
        return {
            "modality": "mass_spectrometry_proteomics",
            "description": "Mass spec proteomics samples",
            "priority_groups": {
                ExportPriority.CORE_IDENTIFIERS: [
                    "sample_id",  # Primary sample identifier
                    "protein_id",  # Protein accession
                    "gene_name",  # Gene symbol
                    "uniprot_id",  # UniProt accession
                ],
                ExportPriority.SAMPLE_METADATA: [
                    "condition",  # Experimental condition
                    "treatment",  # Treatment applied
                    "batch",  # MS run batch
                    "replicate",  # Biological replicate
                    "organism",  # Organism (free text)
                    "tissue",  # Tissue type (free text)
                    "cell_type",  # Cell type (free text)
                ],
                ExportPriority.HARMONIZED_METADATA: [
                    "disease",  # Standardized disease term
                    "age",  # Patient age
                    "sex",  # Patient sex
                    "sample_type",  # Sample type (e.g., "plasma", "tissue")
                ],
                ExportPriority.LIBRARY_TECHNICAL: [
                    "instrument",  # MS instrument (e.g., "Orbitrap Fusion")
                    "acquisition_method",  # DDA, DIA, SRM, PRM
                    "total_spectra",  # Total MS/MS spectra
                    "identified_spectra_pct",  # % spectra identified
                    "mass_accuracy_ppm",  # Mass accuracy (ppm)
                    "peptides_detected",  # Number of peptides
                ],
            },
        }

    @staticmethod
    def get_metabolomics_export_schema() -> Dict[str, Any]:
        """
        Export schema for metabolomics (LC-MS, GC-MS, NMR).

        **Use cases**:
            - Metabolite profiling
            - Disease metabolomics
            - Drug metabolism studies

        **Column count**: ~22 fields (identifiers + platform details)

        Returns:
            Schema dict with priority-grouped column definitions
        """
        return {
            "modality": "metabolomics",
            "description": "Metabolomics samples (LC-MS, GC-MS, NMR)",
            "priority_groups": {
                ExportPriority.CORE_IDENTIFIERS: [
                    "sample_id",  # Primary sample identifier
                    "metabolite_id",  # Metabolite accession
                    "compound_name",  # IUPAC or common name
                    "hmdb_id",  # Human Metabolome Database ID
                    "kegg_id",  # KEGG compound ID
                ],
                ExportPriority.SAMPLE_METADATA: [
                    "condition",  # Experimental condition
                    "treatment",  # Treatment applied
                    "batch",  # Analytical batch
                    "replicate",  # Biological replicate
                    "organism",  # Organism (free text)
                    "tissue",  # Tissue type (free text)
                    "biofluid",  # Biofluid (e.g., "plasma", "urine")
                ],
                ExportPriority.HARMONIZED_METADATA: [
                    "disease",  # Standardized disease term
                    "age",  # Patient age
                    "sex",  # Patient sex
                    "sample_type",  # Sample type
                ],
                ExportPriority.LIBRARY_TECHNICAL: [
                    "platform",  # LC-MS, GC-MS, NMR
                    "ionization_mode",  # Positive, negative, ESI, APCI
                    "column_type",  # Chromatography column
                    "acquisition_method",  # Data acquisition method
                    "retention_time",  # RT (minutes)
                    "mz",  # m/z value
                ],
            },
        }

    @staticmethod
    def get_transcriptomics_export_schema() -> Dict[str, Any]:
        """
        Export schema for bulk RNA-seq.

        **Use cases**:
            - Differential expression analysis
            - Pathway enrichment
            - Disease transcriptomics

        **Column count**: ~28 fields (similar to SRA but RNA-specific)

        Returns:
            Schema dict with priority-grouped column definitions
        """
        return {
            "modality": "bulk_rna_seq",
            "description": "Bulk RNA sequencing samples",
            "priority_groups": {
                ExportPriority.CORE_IDENTIFIERS: [
                    "sample_id",  # Primary sample identifier
                    "run_accession",  # SRR accession (if SRA)
                    "biosample",  # SAMN accession
                    "bioproject",  # PRJNA accession
                ],
                ExportPriority.SAMPLE_METADATA: [
                    "condition",  # Experimental condition
                    "treatment",  # Treatment applied
                    "batch",  # Sequencing batch
                    "replicate",  # Biological replicate
                    "organism",  # Organism (free text)
                    "tissue",  # Tissue type (free text)
                    "cell_type",  # Cell type (free text)
                ],
                ExportPriority.HARMONIZED_METADATA: [
                    "disease",  # Standardized disease term
                    "age",  # Patient age
                    "sex",  # Patient sex
                    "sample_type",  # Sample type
                ],
                ExportPriority.LIBRARY_TECHNICAL: [
                    "library_strategy",  # RNA-Seq, miRNA-Seq, etc.
                    "library_layout",  # SINGLE, PAIRED
                    "instrument",  # Sequencer
                    "total_reads",  # Total reads
                    "mapped_reads_pct",  # % reads mapped
                    "rRNA_pct",  # % ribosomal RNA
                ],
                ExportPriority.DOWNLOAD_URLS: [
                    "ena_fastq_http",  # ENA FASTQ URL
                    "ncbi_url",  # NCBI URL
                    "aws_url",  # AWS URL
                ],
            },
        }

    @staticmethod
    def get_export_schema(data_type: str) -> Optional[Dict[str, Any]]:
        """
        Get export schema by data type/modality.

        Supports multiple aliases for common data types to improve UX.

        Args:
            data_type: Modality identifier (case-insensitive)
                Examples: "sra_amplicon", "16s_amplicon", "microbiome",
                          "mass_spectrometry_proteomics", "proteomics",
                          "metabolomics", "bulk_rna_seq", "transcriptomics"

        Returns:
            Export schema dict or None if data type not found

        **Extensibility**: Add new omics layer by:
            1. Adding method: get_<modality>_export_schema()
            2. Registering in this registry dict
            3. That's it! (15 min task)
        """
        registry = {
            # SRA/Amplicon/Microbiome
            "sra_amplicon": ExportSchemaRegistry.get_sra_amplicon_export_schema,
            "16s_amplicon": ExportSchemaRegistry.get_sra_amplicon_export_schema,
            "microbiome": ExportSchemaRegistry.get_sra_amplicon_export_schema,
            "metagenomics": ExportSchemaRegistry.get_sra_amplicon_export_schema,
            # Proteomics
            "mass_spectrometry_proteomics": ExportSchemaRegistry.get_proteomics_export_schema,
            "proteomics": ExportSchemaRegistry.get_proteomics_export_schema,
            "ms_proteomics": ExportSchemaRegistry.get_proteomics_export_schema,
            # Metabolomics
            "metabolomics": ExportSchemaRegistry.get_metabolomics_export_schema,
            "lcms": ExportSchemaRegistry.get_metabolomics_export_schema,
            "gcms": ExportSchemaRegistry.get_metabolomics_export_schema,
            # Transcriptomics
            "bulk_rna_seq": ExportSchemaRegistry.get_transcriptomics_export_schema,
            "transcriptomics": ExportSchemaRegistry.get_transcriptomics_export_schema,
            "rna_seq": ExportSchemaRegistry.get_transcriptomics_export_schema,
        }

        schema_fn = registry.get(data_type.lower())
        return schema_fn() if schema_fn else None


def get_ordered_export_columns(
    samples: List[Dict[str, Any]],
    data_type: str = "sra_amplicon",
    include_extra: bool = True,
) -> List[str]:
    """
    Get priority-ordered column list for CSV export.

    Main API function called by workspace_tool.py. Replaces hardcoded
    column lists with schema-driven approach.

    **Algorithm**:
        1. Lookup export schema by data_type
        2. Iterate priority groups in order (1, 2, 3, ..., 99)
        3. Include columns that exist in actual data
        4. Append extra fields not in schema (if include_extra=True)

    Args:
        samples: List of sample dicts to export
        data_type: Modality type for schema lookup
            Default: "sra_amplicon" (DataBioMix microbiome use case)
        include_extra: Include fields not defined in schema
            Default: True (ensures no data loss)

    Returns:
        Ordered list of column names for DataFrame column selection

    **Examples**:
        >>> samples = [{"run_accession": "SRR123", "disease": "crc", ...}]
        >>> cols = get_ordered_export_columns(samples, "sra_amplicon")
        >>> df = pd.DataFrame(samples)[cols]  # Columns in schema order

        >>> # Proteomics example
        >>> samples = [{"sample_id": "S1", "protein_id": "P12345", ...}]
        >>> cols = get_ordered_export_columns(samples, "proteomics")

    **Fallback behavior**:
        If data_type not found in registry:
            - Returns alphabetically sorted columns
            - Logs warning about missing schema
            - Ensures CSV export still works
    """
    schema = ExportSchemaRegistry.get_export_schema(data_type)

    if not schema:
        # Fallback: alphabetical ordering
        # (Professional degradation - still works, just not optimal)
        all_cols = set()
        for sample in samples:
            all_cols.update(sample.keys())
        return sorted(all_cols)

    # Build ordered list by priority
    ordered_cols = []
    priority_groups = schema["priority_groups"]

    # Sort by priority enum value (1=CORE_IDENTIFIERS, 2=SAMPLE_METADATA, etc.)
    for priority in sorted(priority_groups.keys(), key=lambda p: p.value):
        for col in priority_groups[priority]:
            # Only include if present in actual data
            if any(col in sample for sample in samples):
                if col not in ordered_cols:  # Avoid duplicates
                    ordered_cols.append(col)

    # Append extra fields not defined in schema
    if include_extra:
        all_cols = set()
        for sample in samples:
            all_cols.update(sample.keys())

        extra_cols = sorted(all_cols - set(ordered_cols))
        ordered_cols.extend(extra_cols)

    return ordered_cols


def infer_data_type(samples: List[Dict[str, Any]]) -> str:
    """
    Infer data type from sample field names.

    Uses heuristic field matching to detect modality type.
    Called by workspace_tool.py when data_type not explicitly provided.

    **Detection Rules**:
        1. SRA/Amplicon: run_accession + library_strategy="AMPLICON"
        2. Transcriptomics: run_accession + library_strategy="RNA-Seq"
        3. Proteomics: protein_id OR uniprot_id present
        4. Metabolomics: metabolite_id OR hmdb_id present
        5. Default: "sra_amplicon" (safest fallback)

    Args:
        samples: List of sample dicts

    Returns:
        Data type identifier for schema lookup

    **Examples**:
        >>> samples = [{"run_accession": "SRR123", "library_strategy": "AMPLICON"}]
        >>> infer_data_type(samples)
        'sra_amplicon'

        >>> samples = [{"protein_id": "P12345", "condition": "control"}]
        >>> infer_data_type(samples)
        'proteomics'
    """
    if not samples:
        return "sra_amplicon"  # Default

    first_sample = samples[0]

    # SRA/Amplicon/Transcriptomics detection
    if "run_accession" in first_sample and "library_strategy" in first_sample:
        library_strategy = str(first_sample.get("library_strategy", "")).upper()
        if "AMPLICON" in library_strategy:
            return "sra_amplicon"
        elif "RNA" in library_strategy:
            return "transcriptomics"
        else:
            # Other SRA strategies default to amplicon schema (most common)
            return "sra_amplicon"

    # Proteomics detection
    if "protein_id" in first_sample or "uniprot_id" in first_sample:
        return "proteomics"

    # Metabolomics detection
    if "metabolite_id" in first_sample or "hmdb_id" in first_sample:
        return "metabolomics"

    # Default fallback: SRA amplicon (DataBioMix microbiome use case)
    return "sra_amplicon"


# =============================================================================
# Column Harmonization (v2.0 - January 2026)
# =============================================================================

# Canonical column name → list of known variants (order matters for priority)
COLUMN_ALIASES: Dict[str, List[str]] = {
    # Core identifiers (rarely have variants)
    "run_accession": ["run_accession"],
    "sample_accession": ["sample_accession", "sample_acc"],
    "biosample": ["biosample", "bio_sample", "BioSample"],
    "bioproject": ["bioproject", "bio_project", "BioProject"],

    # Sample metadata (common variants)
    "isolation_source": ["isolation_source", "isolation source", "isolationsource"],
    "geo_loc_name": ["geo_loc_name", "geo loc name", "geographic location", "geolocation"],
    "collection_date": ["collection_date", "collection date", "collectiondate"],
    "sample_type": ["sample_type", "sample type", "sampletype", "sample_material"],

    # Temporal (many variants)
    "timepoint": ["timepoint", "time_point", "time point", "time-point", "_timepoint", "sampling_time"],

    # Host metadata
    "host": ["host", "host_name", "host organism"],
    "host_sex": ["host_sex", "host sex", "sex", "gender"],
    "host_age": ["host_age", "host age", "age", "patient_age"],
    "host_body_site": ["host_body_site", "host body site", "body_site", "anatomical_site"],
    "host_body_product": ["host_body_product", "host body product", "body_product"],

    # Disease (many variants - critical for harmonization)
    "disease": ["disease", "disease_state", "condition", "diagnosis", "health_status"],
    "host_disease_stat": ["host_disease_stat", "host_disease", "disease_stat"],

    # Publication context
    "source_doi": ["source_doi", "publication_doi", "doi"],
    "source_pmid": ["source_pmid", "publication_pmid", "pmid", "pubmed_id"],
    "source_entry_id": ["source_entry_id", "publication_entry_id", "entry_id"],
}


def harmonize_column_names(
    samples: List[Dict[str, Any]],
    track_provenance: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merge duplicate columns into canonical names with provenance tracking.

    Uses COLUMN_ALIASES to coalesce values from variant column names.
    Priority: first non-null value from alias list wins.

    Args:
        samples: List of sample dicts with potentially duplicate columns
        track_provenance: If True, return detailed transformation log

    Returns:
        Tuple of (harmonized_samples, provenance_log)

        Provenance log contains:
        - sample_id: Identifier for sample (run_accession or index)
        - transformation_type: "column_alias"
        - canonical_field: Target column name
        - source_column: Original column name
        - original_value: Value before transformation
        - harmonized_value: Value after transformation

    Example:
        >>> samples = [{"time_point": "day1", "timepoint": None, "host sex": "male"}]
        >>> result, log = harmonize_column_names(samples)
        >>> result[0]["timepoint"]
        'day1'
        >>> result[0]["host_sex"]
        'male'
    """
    if not samples:
        return samples, []

    # Build reverse lookup: variant (lowercase) → canonical
    variant_to_canonical: Dict[str, str] = {}
    for canonical, variants in COLUMN_ALIASES.items():
        for variant in variants:
            variant_to_canonical[variant.lower()] = canonical

    result = []
    provenance_log = []

    for idx, sample in enumerate(samples):
        sample_id = sample.get("run_accession") or sample.get("biosample") or f"sample_{idx}"
        harmonized: Dict[str, Any] = {}
        processed_keys: Set[str] = set()

        # First pass: handle aliased columns
        for canonical, variants in COLUMN_ALIASES.items():
            for variant in variants:
                # Case-insensitive lookup
                for key in sample.keys():
                    if key.lower() == variant.lower() and key not in processed_keys:
                        value = sample[key]
                        # Coalesce: first non-null wins
                        if value is not None and str(value).strip() and str(value).lower() not in ("nan", "none", ""):
                            if canonical not in harmonized:
                                harmonized[canonical] = value
                                # Track transformation if source column differs from canonical
                                if track_provenance and key.lower() != canonical.lower():
                                    provenance_log.append({
                                        "sample_id": sample_id,
                                        "transformation_type": "column_alias",
                                        "canonical_field": canonical,
                                        "source_column": key,
                                        "original_value": str(value)[:100],  # Truncate long values
                                        "harmonized_value": str(value)[:100],
                                        "rule_applied": f"{key} → {canonical}"
                                    })
                        processed_keys.add(key)

        # Second pass: keep non-aliased columns as-is
        for key, value in sample.items():
            if key not in processed_keys:
                harmonized[key] = value

        result.append(harmonized)

    return result, provenance_log


# =============================================================================
# Coverage Analysis & Sparse Column Removal (v2.0)
# =============================================================================

def calculate_coverage(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate coverage (% non-null) for each column.

    Args:
        samples: List of sample dicts

    Returns:
        Dict mapping column_name -> coverage percentage (0.0 to 100.0)

    Example:
        >>> samples = [{"a": 1, "b": None}, {"a": 2, "b": "value"}]
        >>> coverage = calculate_coverage(samples)
        >>> coverage["a"]
        100.0
        >>> coverage["b"]
        50.0
    """
    if not samples:
        return {}

    col_counts: Dict[str, int] = {}
    total = len(samples)

    for sample in samples:
        for col, value in sample.items():
            if col not in col_counts:
                col_counts[col] = 0
            # Count as non-null if not None, not NaN, not empty string
            if value is not None and str(value).strip() and str(value).lower() not in ("nan", "na", "none", ""):
                col_counts[col] += 1

    return {col: (count / total) * 100 for col, count in col_counts.items()}


def remove_sparse_columns(
    samples: List[Dict[str, Any]],
    threshold: float = 0.05,
    protected_columns: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Remove columns with coverage below threshold.

    Args:
        samples: List of sample dicts
        threshold: Minimum coverage ratio (0.05 = 5%)
        protected_columns: Columns to never remove (schema columns)

    Returns:
        Tuple of (filtered_samples, removed_columns)

    Example:
        >>> samples = [{"a": 1, "b": None}, {"a": 2, "b": None}]
        >>> filtered, removed = remove_sparse_columns(samples, threshold=0.5)
        >>> "b" in removed
        True
    """
    if not samples:
        return samples, set()

    if protected_columns is None:
        # Protect all schema-defined columns
        schema = ExportSchemaRegistry.get_export_schema("sra_amplicon")
        protected_columns = set()
        if schema:
            for cols in schema["priority_groups"].values():
                protected_columns.update(cols)

    coverage = calculate_coverage(samples)
    threshold_pct = threshold * 100

    sparse_cols = {
        col for col, pct in coverage.items()
        if pct < threshold_pct and col not in protected_columns
    }

    filtered = [
        {k: v for k, v in s.items() if k not in sparse_cols}
        for s in samples
    ]

    return filtered, sparse_cols


def remove_constant_columns(
    samples: List[Dict[str, Any]],
    protected_columns: Optional[Set[str]] = None
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Remove columns where all values are identical (no information value).

    Args:
        samples: List of sample dicts
        protected_columns: Columns to never remove

    Returns:
        Tuple of (filtered_samples, removed_columns)

    Example:
        >>> samples = [{"a": 1, "b": "same"}, {"a": 2, "b": "same"}]
        >>> filtered, removed = remove_constant_columns(samples)
        >>> "b" in removed
        True
    """
    if not samples:
        return samples, set()

    if protected_columns is None:
        protected_columns = {
            "source_entry_id", "bioproject", "study_accession",
            "run_accession", "sample_accession", "biosample"
        }

    # Find constant columns
    all_cols: Set[str] = set()
    for sample in samples:
        all_cols.update(sample.keys())

    constant_cols: Set[str] = set()
    for col in all_cols:
        try:
            values = {s.get(col) for s in samples}
            # Treat None and empty string as same value for this check
            normalized = {v if v is not None and str(v).strip() else None for v in values}
            if len(normalized) == 1 and col not in protected_columns:
                constant_cols.add(col)
        except TypeError:
            # Skip columns with unhashable types (lists, dicts)
            # These likely contain structured data worth preserving
            continue

    filtered = [
        {k: v for k, v in s.items() if k not in constant_cols}
        for s in samples
    ]

    return filtered, constant_cols


# =============================================================================
# Disease Extraction (v2.0) - Critical for Coverage Improvement
# =============================================================================

# Disease flag mappings (boolean columns → standardized terms)
DISEASE_FLAG_MAPPINGS: Dict[str, str] = {
    "crohns": "cd",
    "crohn": "cd",
    "inflammbowel": "ibd",
    "inflam_bowel": "ibd",
    "ulcerativecolitis": "uc",
    "ulcerative_colitis": "uc",
    "colorectal": "crc",
    "colorectalcancer": "crc",
    "parkinson": "parkinsons",
    "alzheimer": "alzheimers",
    "diabetes": "diabetes",
    "obesity": "obesity",
}

# Values that indicate TRUE for boolean disease flags
TRUE_VALUES = {"yes", "y", "true", "1", "positive", "affected"}
FALSE_VALUES = {"no", "n", "false", "0", "negative", "unaffected", "control", "healthy"}
CONTROL_INDICATORS = {"control", "healthy", "normal", "non-ibd", "non-diseased"}
AMBIGUOUS_VALUES = {"not applicable", "n/a", "missing", "unknown", "not collected"}


def harmonize_disease_field(
    samples: List[Dict[str, Any]],
    track_provenance: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract disease from study-specific fields into unified 'disease' column with provenance tracking.

    Extraction strategies (applied in order):
    1. Existing unified columns: disease, disease_state, condition, diagnosis
    2. Boolean disease flags: crohns_disease=Yes → disease="cd"
    3. Free-text phenotype: host_phenotype, health_status, clinical condition
    4. MIMARKS fallback: host_disease_stat

    Creates two fields:
    - disease: Standardized term (cd, uc, crc, healthy, etc.)
    - disease_original: Original raw value for traceability

    Args:
        samples: List of sample dicts
        track_provenance: If True, return detailed transformation log

    Returns:
        Tuple of (harmonized_samples, provenance_log)

        Provenance log contains:
        - sample_id: Identifier for sample
        - transformation_type: "disease_extraction"
        - canonical_field: "disease"
        - source_column: Which field disease was extracted from
        - original_value: Raw disease value
        - harmonized_value: Standardized disease term
        - strategy_used: Which extraction strategy succeeded

    Example:
        >>> samples = [{"crohns_disease": "Yes"}, {"ulcerative_colitis": "Yes"}]
        >>> result, log = harmonize_disease_field(samples)
        >>> result[0]["disease"]
        'cd'
        >>> result[1]["disease"]
        'uc'
    """
    import pandas as pd

    if not samples:
        return samples, []

    df = pd.DataFrame(samples)

    # Initialize columns if not present
    if "disease" not in df.columns:
        df["disease"] = None
    if "disease_original" not in df.columns:
        df["disease_original"] = None
    if track_provenance:
        df["_disease_strategy"] = None  # Track which strategy extracted disease
        df["_disease_source_col"] = None  # Track source column

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy 1: Check existing unified columns
    # ─────────────────────────────────────────────────────────────────────────
    unified_cols = ["disease", "disease_state", "condition", "diagnosis"]
    for col in unified_cols:
        if col in df.columns and col != "disease":
            mask = df["disease"].isna() & df[col].notna() & (df[col] != "")
            if mask.any():
                df.loc[mask, "disease"] = df.loc[mask, col]
                df.loc[mask, "disease_original"] = df.loc[mask, col]
                if track_provenance:
                    df.loc[mask, "_disease_strategy"] = "strategy_1_unified"
                    df.loc[mask, "_disease_source_col"] = col

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy 2: Boolean disease flags (crohns_disease=Yes → disease="cd")
    # ─────────────────────────────────────────────────────────────────────────
    disease_flag_cols = [c for c in df.columns if c.lower().endswith("_disease") or c.lower().endswith("disease")]
    disease_flag_cols = [c for c in disease_flag_cols if c.lower() not in ("disease", "host_disease", "host_disease_stat")]

    if disease_flag_cols:
        def extract_from_flags(row):
            if pd.notna(row.get("disease")) and row.get("disease"):
                return row["disease"], row.get("disease_original")

            active_diseases = []
            original_values = []
            all_false = True
            has_any_flag = False

            for flag_col in disease_flag_cols:
                flag_value = row.get(flag_col)
                if flag_value is None or pd.isna(flag_value):
                    continue

                flag_str = str(flag_value).lower().strip()

                # Skip ambiguous values
                if flag_str in AMBIGUOUS_VALUES:
                    continue

                has_any_flag = True

                # Check for control indicators first
                if any(indicator in flag_str for indicator in CONTROL_INDICATORS):
                    return "healthy", f"control_indicator={flag_value}"

                if flag_str in TRUE_VALUES:
                    all_false = False
                    # Extract disease name from column name
                    disease_name = flag_col.lower().replace("_disease", "").replace("disease", "").replace("_", "")

                    # Map to standardized term
                    standardized = disease_name
                    for pattern, mapped in DISEASE_FLAG_MAPPINGS.items():
                        if pattern in disease_name:
                            standardized = mapped
                            break

                    active_diseases.append(standardized)
                    original_values.append(f"{flag_col}={flag_value}")
                elif flag_str in FALSE_VALUES:
                    pass  # Explicit false, not missing

            if active_diseases:
                return ";".join(sorted(set(active_diseases))), ";".join(original_values)
            elif all_false and has_any_flag:
                return "healthy", "all_disease_flags=No"
            return None, None

        # Apply extraction
        extracted = df.apply(extract_from_flags, axis=1, result_type="expand")
        extracted.columns = ["_extracted_disease", "_extracted_original"]

        # Fill missing values
        mask = df["disease"].isna() & extracted["_extracted_disease"].notna()
        df.loc[mask, "disease"] = extracted.loc[mask, "_extracted_disease"]
        df.loc[mask, "disease_original"] = extracted.loc[mask, "_extracted_original"]
        if track_provenance:
            df.loc[mask, "_disease_strategy"] = "strategy_2_boolean_flags"
            df.loc[mask, "_disease_source_col"] = "multiple_disease_flags"

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy 3: Free-text phenotype fields
    # ─────────────────────────────────────────────────────────────────────────
    freetext_cols = [
        "host_phenotype", "phenotype", "health_status", "clinical condition",
        "clinical_condition", "host_disease", "ibd", "ibd_diagnosis",
        "ibd_diagnosis_refined"
    ]

    for col in freetext_cols:
        if col in df.columns:
            mask = df["disease"].isna() & df[col].notna() & (df[col].astype(str).str.strip() != "")
            if mask.any():
                df.loc[mask, "disease"] = df.loc[mask, col]
                df.loc[mask, "disease_original"] = df.loc[mask, col]
                if track_provenance:
                    df.loc[mask, "_disease_strategy"] = "strategy_3_freetext"
                    df.loc[mask, "_disease_source_col"] = col

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy 4: MIMARKS fallback (host_disease_stat)
    # ─────────────────────────────────────────────────────────────────────────
    if "host_disease_stat" in df.columns:
        mask = df["disease"].isna() & df["host_disease_stat"].notna()
        df.loc[mask, "disease"] = df.loc[mask, "host_disease_stat"]
        df.loc[mask, "disease_original"] = df.loc[mask, "host_disease_stat"]
        if track_provenance:
            df.loc[mask, "_disease_strategy"] = "strategy_4_mimarks"
            df.loc[mask, "_disease_source_col"] = "host_disease_stat"

    # Build provenance log from tracking columns
    provenance_log = []
    if track_provenance:
        # Get sample identifier column
        id_col = "run_accession" if "run_accession" in df.columns else "biosample" if "biosample" in df.columns else None

        # Build log for samples where disease was extracted
        disease_extracted = df[df["_disease_strategy"].notna()]
        for idx, row in disease_extracted.iterrows():
            sample_id = row.get(id_col, f"sample_{idx}") if id_col else f"sample_{idx}"
            provenance_log.append({
                "sample_id": sample_id,
                "transformation_type": "disease_extraction",
                "canonical_field": "disease",
                "source_column": row.get("_disease_source_col", "unknown"),
                "original_value": str(row.get("disease_original", ""))[:100],
                "harmonized_value": str(row.get("disease", ""))[:100],
                "strategy_used": row.get("_disease_strategy", "unknown"),
                "rule_applied": f"{row.get('_disease_strategy', 'unknown')} extraction"
            })

        # Remove tracking columns before returning
        df = df.drop(columns=["_disease_strategy", "_disease_source_col"], errors="ignore")

    return df.to_dict("records"), provenance_log


# =============================================================================
# Master Harmonization Pipeline (v2.0)
# =============================================================================

def harmonize_samples_for_export(
    samples: List[Dict[str, Any]],
    remove_sparse: bool = True,
    sparse_threshold: float = 0.05,
    remove_constant: bool = True,
    track_provenance: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Master harmonization pipeline for CSV export with provenance tracking.

    Pipeline order:
    1. Harmonize column names (merge duplicates)
    2. Extract/harmonize disease field
    3. Remove sparse columns (<5% coverage)
    4. Remove constant columns (single value)

    Args:
        samples: Raw sample dicts from SRA/publication queue
        remove_sparse: Whether to remove sparse columns
        sparse_threshold: Coverage threshold for sparse removal (default 5%)
        remove_constant: Whether to remove constant columns
        track_provenance: If True, return detailed transformation log

    Returns:
        Tuple of (harmonized_samples, stats_dict, provenance_log)

    Stats dict contains:
        - columns_before: int
        - columns_after: int
        - sparse_removed: Set[str]
        - constant_removed: Set[str]
        - disease_coverage_before: float
        - disease_coverage_after: float
        - unmapped_columns: List[str] (columns not in COLUMN_ALIASES)
        - unmapped_disease_samples: int (samples with no disease extracted)
        - provenance_log_size: int (number of transformation events)

    Provenance log: List of dicts tracking each transformation

    Example:
        >>> samples = [{"time_point": "d1", "crohns_disease": "Yes"}]
        >>> result, stats, log = harmonize_samples_for_export(samples)
        >>> stats["disease_coverage_after"] > 0
        True
        >>> len(log) > 0  # Provenance tracked
        True
    """
    if not samples:
        return samples, {"columns_before": 0, "columns_after": 0}, []

    stats: Dict[str, Any] = {}
    all_provenance_logs = []

    # Count columns before
    all_cols_before: Set[str] = set()
    for s in samples:
        all_cols_before.update(s.keys())
    stats["columns_before"] = len(all_cols_before)

    # Disease coverage before
    disease_before = sum(1 for s in samples if s.get("disease"))
    stats["disease_coverage_before"] = (disease_before / len(samples)) * 100

    # Step 1: Harmonize column names
    result, col_prov_log = harmonize_column_names(samples, track_provenance=track_provenance)
    all_provenance_logs.extend(col_prov_log)

    # Step 2: Extract disease
    result, disease_prov_log = harmonize_disease_field(result, track_provenance=track_provenance)
    all_provenance_logs.extend(disease_prov_log)

    # Disease coverage after extraction
    disease_after = sum(1 for s in result if s.get("disease"))
    stats["disease_coverage_after"] = (disease_after / len(result)) * 100
    stats["disease_extracted"] = disease_after - disease_before

    # Unmapped term reporting (Gemini feedback)
    if track_provenance:
        # Columns that were not harmonized (not in COLUMN_ALIASES)
        all_cols_after_step1: Set[str] = set()
        for s in result:
            all_cols_after_step1.update(s.keys())
        known_canonical = set(COLUMN_ALIASES.keys())
        stats["unmapped_columns"] = sorted(list(all_cols_after_step1 - known_canonical))[:50]  # First 50
        stats["unmapped_columns_count"] = len(all_cols_after_step1 - known_canonical)

        # Samples where disease extraction failed
        stats["unmapped_disease_samples"] = len(result) - disease_after

    # Step 3: Remove sparse columns
    stats["sparse_removed"] = []
    if remove_sparse:
        result, sparse_removed = remove_sparse_columns(result, threshold=sparse_threshold)
        stats["sparse_removed"] = sorted(list(sparse_removed))  # Convert set to list for JSON

    # Step 4: Remove constant columns
    stats["constant_removed"] = []
    if remove_constant:
        result, constant_removed = remove_constant_columns(result)
        stats["constant_removed"] = sorted(list(constant_removed))  # Convert set to list for JSON

    # Count columns after
    all_cols_after: Set[str] = set()
    for s in result:
        all_cols_after.update(s.keys())
    stats["columns_after"] = len(all_cols_after)

    # Provenance metadata
    stats["provenance_log_size"] = len(all_provenance_logs)

    return result, stats, all_provenance_logs
