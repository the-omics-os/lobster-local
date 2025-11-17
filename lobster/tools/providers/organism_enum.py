"""
Centralized organism name mapping for NCBI database queries.

This module provides a standardized mapping between common organism names and
NCBI scientific names. It prevents query failures caused by incorrect organism
names and ensures consistency across all providers (SRAProvider, PubMedProvider,
GEOProvider).

Pattern inspired by SRAgent repository research (2025-01-15).
"""

from enum import Enum
from typing import Optional


class OrganismEnum(Enum):
    """
    Enumeration of commonly sequenced organisms with NCBI scientific names.

    This enum provides authoritative mappings for organism names used in
    bioinformatics databases. It supports:
    - Common name → scientific name conversion
    - Case-insensitive matching
    - Underscore/space handling
    - Validation before API calls

    Usage:
        >>> to_sci_name("human")
        '"Homo sapiens"'
        >>> to_sci_name("mouse")
        '"Mus musculus"'
        >>> validate_organism("rat")
        True

    Adding new organisms:
        1. Add entry to enum: COMMON_NAME = "Scientific Name"
        2. Run tests to verify
        3. Update CLAUDE.md documentation
    """

    # Mammals - Model Organisms
    HUMAN = "Homo sapiens"
    MOUSE = "Mus musculus"
    RAT = "Rattus norvegicus"
    HAMSTER = "Cricetulus griseus"
    GUINEA_PIG = "Cavia porcellus"
    RABBIT = "Oryctolagus cuniculus"
    PIG = "Sus scrofa"
    COW = "Bos taurus"
    SHEEP = "Ovis aries"
    DOG = "Canis lupus familiaris"
    CAT = "Felis catus"
    HORSE = "Equus caballus"

    # Primates
    RHESUS_MACAQUE = "Macaca mulatta"
    CYNOMOLGUS_MACAQUE = "Macaca fascicularis"
    CHIMPANZEE = "Pan troglodytes"
    GORILLA = "Gorilla gorilla"

    # Model Organisms - Non-Mammalian
    ZEBRAFISH = "Danio rerio"
    FRUIT_FLY = "Drosophila melanogaster"
    MOSQUITO = "Anopheles gambiae"
    C_ELEGANS = "Caenorhabditis elegans"
    XENOPUS = "Xenopus laevis"
    CHICKEN = "Gallus gallus"

    # Plants
    ARABIDOPSIS = "Arabidopsis thaliana"
    RICE = "Oryza sativa"
    MAIZE = "Zea mays"
    TOMATO = "Solanum lycopersicum"
    SOYBEAN = "Glycine max"
    TOBACCO = "Nicotiana tabacum"
    WHEAT = "Triticum aestivum"

    # Fungi
    YEAST = "Saccharomyces cerevisiae"
    FISSION_YEAST = "Schizosaccharomyces pombe"
    ASPERGILLUS = "Aspergillus fumigatus"
    CANDIDA = "Candida albicans"

    # Bacteria
    E_COLI = "Escherichia coli"
    SALMONELLA = "Salmonella enterica"
    PSEUDOMONAS = "Pseudomonas aeruginosa"
    BACILLUS = "Bacillus subtilis"
    MYCOBACTERIUM = "Mycobacterium tuberculosis"
    STAPHYLOCOCCUS = "Staphylococcus aureus"
    STREPTOCOCCUS = "Streptococcus pneumoniae"

    # Viruses
    HIV = "Human immunodeficiency virus 1"
    SARS_COV_2 = "Severe acute respiratory syndrome coronavirus 2"
    INFLUENZA = "Influenza A virus"
    HEPATITIS_C = "Hepatitis C virus"

    # Marine Organisms
    SEA_URCHIN = "Strongylocentrotus purpuratus"

    # Additional Model Organisms
    SLIME_MOLD = "Dictyostelium discoideum"


def to_sci_name(organism: str) -> str:
    """
    Convert organism name to NCBI scientific name with proper quoting.

    This function:
    1. Normalizes input (uppercase, replace spaces with underscores)
    2. Looks up in OrganismEnum
    3. Returns quoted scientific name for NCBI queries

    Args:
        organism: Common organism name (case-insensitive, spaces/underscores allowed)
                  Examples: "human", "Mouse", "fruit fly", "FRUIT_FLY"

    Returns:
        Quoted scientific name for NCBI query syntax
        Example: '"Homo sapiens"'

    Raises:
        ValueError: If organism not found in OrganismEnum

    Examples:
        >>> to_sci_name("human")
        '"Homo sapiens"'
        >>> to_sci_name("Mouse")
        '"Mus musculus"'
        >>> to_sci_name("fruit fly")
        '"Drosophila melanogaster"'
        >>> to_sci_name("unknown_organism")
        ValueError: Organism 'unknown_organism' not found in OrganismEnum...
    """
    # Normalize: uppercase + replace spaces with underscores
    organism_str = organism.strip().replace(" ", "_").upper()

    try:
        enum_member = OrganismEnum[organism_str]
        # Return with quotes for NCBI query syntax
        return f'"{enum_member.value}"'
    except KeyError:
        # Provide helpful error message with suggestions
        available = ", ".join([e.name.lower().replace("_", " ") for e in OrganismEnum])
        raise ValueError(
            f"Organism '{organism}' not found in OrganismEnum. "
            f"Available organisms: {available[:200]}... "
            f"(see organism_enum.py for full list)"
        )


def validate_organism(organism: str) -> bool:
    """
    Validate if an organism name exists in OrganismEnum.

    Use this for graceful error handling without raising exceptions.

    Args:
        organism: Common organism name (case-insensitive)

    Returns:
        True if organism is valid, False otherwise

    Examples:
        >>> validate_organism("human")
        True
        >>> validate_organism("unknown")
        False
    """
    organism_str = organism.strip().replace(" ", "_").upper()
    return organism_str in OrganismEnum.__members__


def get_scientific_name(organism: str) -> Optional[str]:
    """
    Get scientific name without NCBI query quotes.

    Useful for display purposes or non-NCBI contexts.

    Args:
        organism: Common organism name (case-insensitive)

    Returns:
        Scientific name without quotes, or None if not found

    Examples:
        >>> get_scientific_name("human")
        'Homo sapiens'
        >>> get_scientific_name("unknown")
        None
    """
    organism_str = organism.strip().replace(" ", "_").upper()
    try:
        return OrganismEnum[organism_str].value
    except KeyError:
        return None


def list_organisms() -> list[str]:
    """
    Get list of all supported organism names.

    Returns:
        List of common organism names (lowercase with spaces)

    Example:
        >>> organisms = list_organisms()
        >>> "human" in organisms
        True
        >>> len(organisms) >= 45
        True
    """
    return [e.name.lower().replace("_", " ") for e in OrganismEnum]


def list_organisms_with_scientific() -> dict[str, str]:
    """
    Get mapping of common names to scientific names.

    Returns:
        Dictionary mapping common name → scientific name

    Example:
        >>> mapping = list_organisms_with_scientific()
        >>> mapping["human"]
        'Homo sapiens'
    """
    return {
        e.name.lower().replace("_", " "): e.value
        for e in OrganismEnum
    }
