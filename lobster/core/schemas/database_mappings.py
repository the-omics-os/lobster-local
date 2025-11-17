"""
Cross-database accession field mappings and validation patterns.

This module provides centralized mappings between omics data fields and external
database accession identifiers, enabling proper validation and cross-referencing
across bioinformatics databases.

Used by:
- Schema validators (transcriptomics, proteomics, metabolomics, metagenomics)
- Data retrieval services (GEO, PRIDE, MetaboLights, etc.)
- Metadata standardization and harmonization

Phase: Week 3 Days 1-2 of multi-omics schema refactoring (4-week project)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatabaseAccession:
    """
    Metadata for a cross-database accession field.

    Attributes:
        field_name: Field name in schema (e.g., 'bioproject_accession')
        database_name: Human-readable database name
        database_url_template: URL template with {accession} placeholder
        prefix_pattern: Regex pattern for the accession prefix
        full_pattern: Complete regex pattern for validation
        example: Example valid accession
        description: Field purpose and usage notes
        modalities: Which data modalities use this field
        required: Whether this field is required (vs optional)
    """

    field_name: str
    database_name: str
    database_url_template: str
    prefix_pattern: str
    full_pattern: str
    example: str
    description: str
    modalities: List[str]
    required: bool = False


# =============================================================================
# NCBI Database Accessions
# =============================================================================

NCBI_ACCESSIONS = {
    "bioproject_accession": DatabaseAccession(
        field_name="bioproject_accession",
        database_name="NCBI BioProject",
        database_url_template="https://www.ncbi.nlm.nih.gov/bioproject/{accession}",
        prefix_pattern=r"PRJNA",
        full_pattern=r"^PRJNA\d{6,}$",
        example="PRJNA123456",
        description="NCBI BioProject provides organizational framework for genomic data",
        modalities=["transcriptomics", "proteomics", "metabolomics", "metagenomics"],
        required=False,
    ),
    "biosample_accession": DatabaseAccession(
        field_name="biosample_accession",
        database_name="NCBI BioSample",
        database_url_template="https://www.ncbi.nlm.nih.gov/biosample/{accession}",
        prefix_pattern=r"SAMN",
        full_pattern=r"^SAMN\d{8,}$",
        example="SAMN12345678",
        description="NCBI BioSample stores sample metadata and provenance",
        modalities=["transcriptomics", "proteomics", "metabolomics", "metagenomics"],
        required=False,
    ),
    "sra_study_accession": DatabaseAccession(
        field_name="sra_study_accession",
        database_name="NCBI Sequence Read Archive (Study)",
        database_url_template="https://www.ncbi.nlm.nih.gov/sra/{accession}",
        prefix_pattern=r"SRP",
        full_pattern=r"^SRP\d{6,}$",
        example="SRP123456",
        description="SRA Study accession for sequencing project",
        modalities=["transcriptomics", "metagenomics"],
        required=False,
    ),
    "sra_experiment_accession": DatabaseAccession(
        field_name="sra_experiment_accession",
        database_name="NCBI Sequence Read Archive (Experiment)",
        database_url_template="https://www.ncbi.nlm.nih.gov/sra/{accession}",
        prefix_pattern=r"SRX",
        full_pattern=r"^SRX\d{6,}$",
        example="SRX123456",
        description="SRA Experiment accession for library preparation",
        modalities=["transcriptomics", "metagenomics"],
        required=False,
    ),
    "sra_run_accession": DatabaseAccession(
        field_name="sra_run_accession",
        database_name="NCBI Sequence Read Archive (Run)",
        database_url_template="https://www.ncbi.nlm.nih.gov/sra/{accession}",
        prefix_pattern=r"SRR",
        full_pattern=r"^SRR\d{6,}$",
        example="SRR123456",
        description="SRA Run accession for sequencing run (FASTQ files)",
        modalities=["transcriptomics", "metagenomics"],
        required=False,
    ),
}

# =============================================================================
# GEO (Gene Expression Omnibus) Accessions
# =============================================================================

GEO_ACCESSIONS = {
    "geo_accession": DatabaseAccession(
        field_name="geo_accession",
        database_name="NCBI Gene Expression Omnibus",
        database_url_template="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        prefix_pattern=r"GSE",
        full_pattern=r"^GSE\d{3,}$",
        example="GSE194247",
        description="GEO Series accession for gene expression datasets",
        modalities=["transcriptomics"],
        required=False,
    ),
}

# =============================================================================
# Proteomics Database Accessions
# =============================================================================

PROTEOMICS_ACCESSIONS = {
    "pride_accession": DatabaseAccession(
        field_name="pride_accession",
        database_name="ProteomeXchange/PRIDE",
        database_url_template="https://www.ebi.ac.uk/pride/archive/projects/{accession}",
        prefix_pattern=r"PXD",
        full_pattern=r"^PXD\d{6}$",
        example="PXD012345",
        description="PRIDE Archive accession for mass spectrometry proteomics data",
        modalities=["proteomics"],
        required=False,
    ),
    "massive_accession": DatabaseAccession(
        field_name="massive_accession",
        database_name="MassIVE",
        database_url_template="https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task={accession}",
        prefix_pattern=r"MSV",
        full_pattern=r"^MSV\d{9}$",
        example="MSV000012345",
        description="MassIVE repository accession for proteomics datasets",
        modalities=["proteomics"],
        required=False,
    ),
}

# =============================================================================
# Metabolomics Database Accessions
# =============================================================================

METABOLOMICS_ACCESSIONS = {
    "metabolights_accession": DatabaseAccession(
        field_name="metabolights_accession",
        database_name="MetaboLights",
        database_url_template="https://www.ebi.ac.uk/metabolights/{accession}",
        prefix_pattern=r"MTBLS",
        full_pattern=r"^MTBLS\d{1,}$",
        example="MTBLS1234",
        description="MetaboLights study accession for metabolomics data",
        modalities=["metabolomics"],
        required=False,
    ),
    "metabolomics_workbench_accession": DatabaseAccession(
        field_name="metabolomics_workbench_accession",
        database_name="Metabolomics Workbench",
        database_url_template="https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID={accession}",
        prefix_pattern=r"ST",
        full_pattern=r"^ST\d{6}$",
        example="ST001234",
        description="Metabolomics Workbench study accession",
        modalities=["metabolomics"],
        required=False,
    ),
}

# =============================================================================
# Metagenomics Database Accessions
# =============================================================================

METAGENOMICS_ACCESSIONS = {
    "mgnify_accession": DatabaseAccession(
        field_name="mgnify_accession",
        database_name="MGnify (EBI Metagenomics)",
        database_url_template="https://www.ebi.ac.uk/metagenomics/studies/{accession}",
        prefix_pattern=r"MGYS",
        full_pattern=r"^MGYS\d{8}$",
        example="MGYS00001234",
        description="MGnify study accession for metagenomic analysis",
        modalities=["metagenomics"],
        required=False,
    ),
    "qiita_accession": DatabaseAccession(
        field_name="qiita_accession",
        database_name="Qiita",
        database_url_template="https://qiita.ucsd.edu/study/description/{accession}",
        prefix_pattern=r"\d",
        full_pattern=r"^\d{1,}$",
        example="10317",
        description="Qiita study ID for microbiome data",
        modalities=["metagenomics"],
        required=False,
    ),
}

# =============================================================================
# Cross-Platform Accessions (Used by Multiple Modalities)
# =============================================================================

CROSS_PLATFORM_ACCESSIONS = {
    "arrayexpress_accession": DatabaseAccession(
        field_name="arrayexpress_accession",
        database_name="ArrayExpress",
        database_url_template="https://www.ebi.ac.uk/arrayexpress/experiments/{accession}",
        prefix_pattern=r"E-[A-Z]{4}",
        full_pattern=r"^E-[A-Z]{4}-\d{1,}$",
        example="E-MTAB-12345",
        description="ArrayExpress experiment accession (alternative to GEO)",
        modalities=["transcriptomics"],
        required=False,
    ),
    "publication_doi": DatabaseAccession(
        field_name="publication_doi",
        database_name="Digital Object Identifier",
        database_url_template="https://doi.org/{accession}",
        prefix_pattern=r"10\.",
        full_pattern=r"^10\.\d{4,}/[^\s]+$",
        example="10.1038/nature12345",
        description="DOI for associated publication",
        modalities=["transcriptomics", "proteomics", "metabolomics", "metagenomics"],
        required=False,
    ),
}

# =============================================================================
# Consolidated Database Mapping Registry
# =============================================================================

# Combine all accession types into single registry
DATABASE_ACCESSION_REGISTRY: Dict[str, DatabaseAccession] = {
    **NCBI_ACCESSIONS,
    **GEO_ACCESSIONS,
    **PROTEOMICS_ACCESSIONS,
    **METABOLOMICS_ACCESSIONS,
    **METAGENOMICS_ACCESSIONS,
    **CROSS_PLATFORM_ACCESSIONS,
}


def get_accessions_for_modality(modality: str) -> Dict[str, DatabaseAccession]:
    """
    Get all database accessions applicable to a specific modality.

    Args:
        modality: Data modality (transcriptomics, proteomics, metabolomics, metagenomics)

    Returns:
        Dict[str, DatabaseAccession]: Filtered accession registry for modality

    Example:
        >>> accessions = get_accessions_for_modality("proteomics")
        >>> print(list(accessions.keys()))
        ['bioproject_accession', 'biosample_accession', 'pride_accession',
         'massive_accession', 'publication_doi']
    """
    return {
        field_name: accession
        for field_name, accession in DATABASE_ACCESSION_REGISTRY.items()
        if modality in accession.modalities
    }


def validate_accession(field_name: str, value: str) -> bool:
    """
    Validate an accession value against its expected pattern.

    Args:
        field_name: Accession field name (e.g., 'bioproject_accession')
        value: Accession value to validate

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> validate_accession("bioproject_accession", "PRJNA123456")
        True
        >>> validate_accession("bioproject_accession", "GSE123456")
        False
    """
    if field_name not in DATABASE_ACCESSION_REGISTRY:
        return False

    accession = DATABASE_ACCESSION_REGISTRY[field_name]
    return bool(re.match(accession.full_pattern, value))


def get_accession_url(field_name: str, value: str) -> Optional[str]:
    """
    Generate database URL for an accession value.

    Args:
        field_name: Accession field name
        value: Accession value

    Returns:
        Optional[str]: Database URL or None if invalid

    Example:
        >>> get_accession_url("bioproject_accession", "PRJNA123456")
        'https://www.ncbi.nlm.nih.gov/bioproject/PRJNA123456'
    """
    if not validate_accession(field_name, value):
        return None

    accession = DATABASE_ACCESSION_REGISTRY[field_name]
    return accession.database_url_template.format(accession=value)


def list_required_accessions(modality: str) -> List[str]:
    """
    Get list of required accession fields for a modality.

    Args:
        modality: Data modality

    Returns:
        List[str]: Required accession field names

    Note:
        Currently all cross-database accessions are optional.
        This function exists for future extensibility.
    """
    accessions = get_accessions_for_modality(modality)
    return [
        field_name for field_name, accession in accessions.items() if accession.required
    ]


# =============================================================================
# Database-Specific Validation Patterns
# =============================================================================


def validate_ncbi_accession(value: str, prefix: str) -> bool:
    """
    Validate NCBI-style accessions (prefix + numeric).

    Args:
        value: Accession value
        prefix: Expected prefix (e.g., 'PRJNA', 'SRR')

    Returns:
        bool: True if matches NCBI pattern
    """
    pattern = f"^{prefix}\\d{{6,}}$"
    return bool(re.match(pattern, value))


def validate_doi(value: str) -> bool:
    """
    Validate Digital Object Identifier (DOI) format.

    Args:
        value: DOI string

    Returns:
        bool: True if valid DOI format

    Note:
        DOI format: 10.{registrant}/{suffix}
        Registrant is 4+ digits, suffix can contain any characters
    """
    return bool(re.match(r"^10\.\d{4,}/[^\s]+$", value))


def validate_geo_accession(value: str) -> bool:
    """
    Validate GEO Series accession format.

    Args:
        value: GEO accession

    Returns:
        bool: True if valid GSE format

    Note:
        GEO Series IDs: GSE followed by 3+ digits
    """
    return bool(re.match(r"^GSE\d{3,}$", value))


# =============================================================================
# Export Summary
# =============================================================================


def get_database_summary() -> Dict[str, int]:
    """
    Get summary statistics of database accession coverage.

    Returns:
        Dict[str, int]: Count of accessions per modality

    Example:
        >>> summary = get_database_summary()
        >>> print(summary)
        {'transcriptomics': 9, 'proteomics': 5, 'metabolomics': 4, 'metagenomics': 5}
    """
    modalities = ["transcriptomics", "proteomics", "metabolomics", "metagenomics"]
    return {
        modality: len(get_accessions_for_modality(modality)) for modality in modalities
    }


if __name__ == "__main__":
    # Print database mapping summary
    print("=== Cross-Database Accession Mapping Summary ===\n")

    summary = get_database_summary()
    print("Accessions per Modality:")
    for modality, count in summary.items():
        print(f"  {modality}: {count} accession types")

    print(f"\nTotal Unique Accession Types: {len(DATABASE_ACCESSION_REGISTRY)}")

    print("\n=== Example Validation ===\n")
    test_cases = [
        ("bioproject_accession", "PRJNA123456", True),
        ("bioproject_accession", "GSE123456", False),
        ("geo_accession", "GSE194247", True),
        ("pride_accession", "PXD012345", True),
        ("publication_doi", "10.1038/nature12345", True),
    ]

    for field, value, expected in test_cases:
        result = validate_accession(field, value)
        status = "✓" if result == expected else "✗"
        print(f"{status} validate_accession('{field}', '{value}') = {result}")
