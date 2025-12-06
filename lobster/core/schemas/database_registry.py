"""
Single source of truth for supported databases.

This module defines the canonical list of databases supported by Lobster.
All code that needs to validate or enumerate databases should import from here.

Example:
    >>> from lobster.core.schemas.database_registry import SupportedDatabase
    >>> SupportedDatabase.is_valid("geo")
    True
    >>> SupportedDatabase.GEO.value
    'geo'
"""

from enum import Enum
from typing import Dict, Optional, Set

from pydantic import BaseModel


class DatabaseCategory(str, Enum):
    """Categories of supported databases."""

    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    METAGENOMICS = "metagenomics"
    GENERAL = "general"


class SupportedDatabase(str, Enum):
    """
    Canonical list of supported databases.

    This is the single source of truth for all database names in Lobster.
    Use this enum instead of hardcoding database names.

    Example:
        >>> if database.lower() in SupportedDatabase.values():
        ...     print("Valid database")
        >>> db = SupportedDatabase.from_string("geo")
        >>> db.display_name  # "NCBI GEO"
    """

    # Transcriptomics
    GEO = "geo"
    SRA = "sra"
    ARRAYEXPRESS = "arrayexpress"
    ENA = "ena"

    # Proteomics
    PRIDE = "pride"
    MASSIVE = "massive"
    PROTEOMEXCHANGE = "proteomexchange"

    # Metabolomics
    METABOLIGHTS = "metabolights"

    # Metagenomics
    MGNIFY = "mgnify"
    QIITA = "qiita"

    # General / Multi-omics
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"
    DBGAP = "dbgap"
    EGA = "ega"
    EBI = "ebi"

    @classmethod
    def values(cls) -> Set[str]:
        """Get set of all database values (lowercase)."""
        return {db.value for db in cls}

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid database name."""
        return value.lower().strip() in cls.values()

    @classmethod
    def from_string(cls, value: str) -> Optional["SupportedDatabase"]:
        """
        Get enum from string value (case-insensitive).

        Args:
            value: Database name string

        Returns:
            SupportedDatabase enum or None if not found
        """
        value_lower = value.lower().strip()
        for db in cls:
            if db.value == value_lower:
                return db
        return None

    @classmethod
    def get_by_category(cls, category: DatabaseCategory) -> Set["SupportedDatabase"]:
        """Get all databases in a specific category."""
        return {
            db
            for db in cls
            if DATABASE_METADATA.get(db, {}).get("category") == category
        }


class DatabaseInfo(BaseModel):
    """Metadata about a supported database."""

    name: str
    display_name: str
    category: DatabaseCategory
    base_url: str
    description: str


# Database metadata registry
DATABASE_METADATA: Dict[SupportedDatabase, DatabaseInfo] = {
    SupportedDatabase.GEO: DatabaseInfo(
        name="geo",
        display_name="NCBI GEO",
        category=DatabaseCategory.TRANSCRIPTOMICS,
        base_url="https://www.ncbi.nlm.nih.gov/geo/",
        description="Gene Expression Omnibus - transcriptomics data repository",
    ),
    SupportedDatabase.SRA: DatabaseInfo(
        name="sra",
        display_name="NCBI SRA",
        category=DatabaseCategory.TRANSCRIPTOMICS,
        base_url="https://www.ncbi.nlm.nih.gov/sra/",
        description="Sequence Read Archive - raw sequencing data",
    ),
    SupportedDatabase.ARRAYEXPRESS: DatabaseInfo(
        name="arrayexpress",
        display_name="ArrayExpress",
        category=DatabaseCategory.TRANSCRIPTOMICS,
        base_url="https://www.ebi.ac.uk/arrayexpress/",
        description="EBI ArrayExpress - functional genomics data",
    ),
    SupportedDatabase.ENA: DatabaseInfo(
        name="ena",
        display_name="ENA",
        category=DatabaseCategory.TRANSCRIPTOMICS,
        base_url="https://www.ebi.ac.uk/ena/",
        description="European Nucleotide Archive - sequencing data",
    ),
    SupportedDatabase.PRIDE: DatabaseInfo(
        name="pride",
        display_name="PRIDE",
        category=DatabaseCategory.PROTEOMICS,
        base_url="https://www.ebi.ac.uk/pride/",
        description="PRIDE Archive - proteomics data repository",
    ),
    SupportedDatabase.MASSIVE: DatabaseInfo(
        name="massive",
        display_name="MassIVE",
        category=DatabaseCategory.PROTEOMICS,
        base_url="https://massive.ucsd.edu/",
        description="MassIVE - mass spectrometry data repository",
    ),
    SupportedDatabase.PROTEOMEXCHANGE: DatabaseInfo(
        name="proteomexchange",
        display_name="ProteomeXchange",
        category=DatabaseCategory.PROTEOMICS,
        base_url="http://www.proteomexchange.org/",
        description="ProteomeXchange - federated proteomics data",
    ),
    SupportedDatabase.METABOLIGHTS: DatabaseInfo(
        name="metabolights",
        display_name="MetaboLights",
        category=DatabaseCategory.METABOLOMICS,
        base_url="https://www.ebi.ac.uk/metabolights/",
        description="MetaboLights - metabolomics data repository",
    ),
    SupportedDatabase.MGNIFY: DatabaseInfo(
        name="mgnify",
        display_name="MGnify",
        category=DatabaseCategory.METAGENOMICS,
        base_url="https://www.ebi.ac.uk/metagenomics/",
        description="MGnify - metagenomics analysis platform",
    ),
    SupportedDatabase.QIITA: DatabaseInfo(
        name="qiita",
        display_name="Qiita",
        category=DatabaseCategory.METAGENOMICS,
        base_url="https://qiita.ucsd.edu/",
        description="Qiita - microbiome data management",
    ),
    SupportedDatabase.BIOPROJECT: DatabaseInfo(
        name="bioproject",
        display_name="BioProject",
        category=DatabaseCategory.GENERAL,
        base_url="https://www.ncbi.nlm.nih.gov/bioproject/",
        description="NCBI BioProject - project-level metadata",
    ),
    SupportedDatabase.BIOSAMPLE: DatabaseInfo(
        name="biosample",
        display_name="BioSample",
        category=DatabaseCategory.GENERAL,
        base_url="https://www.ncbi.nlm.nih.gov/biosample/",
        description="NCBI BioSample - sample-level metadata",
    ),
    SupportedDatabase.DBGAP: DatabaseInfo(
        name="dbgap",
        display_name="dbGaP",
        category=DatabaseCategory.GENERAL,
        base_url="https://www.ncbi.nlm.nih.gov/gap/",
        description="Database of Genotypes and Phenotypes - controlled access",
    ),
    SupportedDatabase.EGA: DatabaseInfo(
        name="ega",
        display_name="EGA",
        category=DatabaseCategory.GENERAL,
        base_url="https://ega-archive.org/",
        description="European Genome-phenome Archive - controlled access",
    ),
    SupportedDatabase.EBI: DatabaseInfo(
        name="ebi",
        display_name="EBI",
        category=DatabaseCategory.GENERAL,
        base_url="https://www.ebi.ac.uk/",
        description="European Bioinformatics Institute - general",
    ),
}


def get_database_info(database: SupportedDatabase) -> Optional[DatabaseInfo]:
    """Get metadata for a database."""
    return DATABASE_METADATA.get(database)


def get_display_name(database_value: str) -> str:
    """Get display name for a database value string."""
    db = SupportedDatabase.from_string(database_value)
    if db and db in DATABASE_METADATA:
        return DATABASE_METADATA[db].display_name
    return database_value.upper()
