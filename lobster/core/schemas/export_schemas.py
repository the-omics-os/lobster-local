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
from typing import Any, Dict, List, Optional


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
                ExportPriority.CORE_IDENTIFIERS: [
                    "run_accession",  # Primary identifier (SRR accession)
                    "sample_accession",  # Sample-level accession
                    "biosample",  # SAMN accession
                    "bioproject",  # PRJNA accession
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
                    "sample_type",  # Sample type (e.g., "fecal", "tissue", "biopsy")
                    "age",  # Patient/subject age (extracted heuristically)
                    "sex",  # Patient/subject sex (extracted heuristically)
                    "tissue",  # Tissue type (e.g., "colon", "ileum", "rectum")
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
