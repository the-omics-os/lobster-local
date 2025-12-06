"""
Utilities for GEO and other omics database accession detection and processing.

This module provides functions to detect and classify different types of
omics database accessions using the existing DatasetType enum.
"""

import re
from enum import Enum
from typing import Optional, Tuple

from lobster.tools.providers.base_provider import DatasetType


class GEOAccessionType(Enum):
    """Specific GEO accession subtypes."""

    SERIES = "gse"  # GSE - Series (experiments)
    SAMPLE = "gsm"  # GSM - Individual samples
    DATASET = "gds"  # GDS - Curated datasets
    PLATFORM = "gpl"  # GPL - Platforms


def detect_accession_type(query: str) -> Optional[DatasetType]:
    """
    Detect the dataset type from an accession string.

    Uses centralized AccessionResolver for pattern matching.

    Args:
        query: The query string to analyze

    Returns:
        DatasetType if accession detected, None if text query
    """
    if not query or not isinstance(query, str):
        return None

    from lobster.core.identifiers import get_accession_resolver

    resolver = get_accession_resolver()

    # Check if GEO
    if resolver.is_geo_identifier(query):
        return DatasetType.GEO

    # Check if SRA/ENA/DDBJ
    if resolver.is_sra_identifier(query):
        db = resolver.detect_database(query)
        if db and "ENA" in db:
            return DatasetType.ENA
        elif db and "DDBJ" in db:
            return DatasetType.SRA  # DDBJ uses SRA DatasetType
        return DatasetType.SRA

    # Check BioProject
    field = resolver.detect_field(query)
    if field and "bioproject" in field:
        db = resolver.detect_database(query)
        if db and "ENA" in db:
            return DatasetType.ENA
        return DatasetType.BIOPROJECT

    # Check BioSample
    if field and "biosample" in field:
        return DatasetType.BIOSAMPLE

    # Check ArrayExpress
    if field == "arrayexpress_accession":
        return DatasetType.ARRAYEXPRESS

    # dbGaP accessions (not in AccessionResolver, keep pattern)
    query_upper = query.strip().upper()
    if re.match(r"^PHS\d+", query_upper):
        return DatasetType.DBGAP

    return None


def detect_geo_accession_subtype(query: str) -> Optional[GEOAccessionType]:
    """
    Detect specific GEO accession subtype.

    Uses centralized AccessionResolver for pattern matching.

    Args:
        query: The query string to analyze

    Returns:
        GEOAccessionType if GEO accession detected, None otherwise
    """
    if not query or not isinstance(query, str):
        return None

    from lobster.core.identifiers import get_accession_resolver

    resolver = get_accession_resolver()
    field = resolver.detect_field(query)

    if field == "geo_accession":
        return GEOAccessionType.SERIES
    elif field == "geo_sample_accession":
        return GEOAccessionType.SAMPLE
    elif field == "geo_dataset_accession":
        return GEOAccessionType.DATASET
    elif field == "geo_platform_accession":
        return GEOAccessionType.PLATFORM

    return None


def is_direct_accession(query: str) -> bool:
    """
    Determine if query is a direct accession rather than text search.

    Args:
        query: The query string to analyze

    Returns:
        True if query appears to be a direct accession
    """
    return detect_accession_type(query) is not None


def is_geo_sample_accession(query: str) -> bool:
    """
    Specifically check if query is a GEO sample (GSM) accession.

    Args:
        query: The query string to analyze

    Returns:
        True if query is a GSM accession
    """
    return detect_geo_accession_subtype(query) == GEOAccessionType.SAMPLE


def is_geo_series_accession(query: str) -> bool:
    """
    Specifically check if query is a GEO series (GSE) accession.

    Args:
        query: The query string to analyze

    Returns:
        True if query is a GSE accession
    """
    return detect_geo_accession_subtype(query) == GEOAccessionType.SERIES


def extract_accession_info(query: str) -> Tuple[Optional[DatasetType], Optional[str]]:
    """
    Extract accession type and normalized accession string.

    Args:
        query: The query string to analyze

    Returns:
        Tuple of (DatasetType, normalized_accession) or (None, None) if not an accession
    """
    dataset_type = detect_accession_type(query)
    if dataset_type is None:
        return None, None

    # Normalize the accession (uppercase, stripped)
    normalized = query.strip().upper()

    return dataset_type, normalized


def get_ncbi_geo_url(accession: str) -> str:
    """
    Generate NCBI GEO URL for a given accession.

    Args:
        accession: GEO accession (GSE, GSM, GDS, GPL)

    Returns:
        Complete NCBI GEO URL
    """
    normalized = accession.strip().upper()
    return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={normalized}"


def get_ncbi_sra_url(accession: str) -> str:
    """
    Generate NCBI SRA URL for a given accession.

    Args:
        accession: SRA accession

    Returns:
        Complete NCBI SRA URL
    """
    normalized = accession.strip().upper()
    return f"https://www.ncbi.nlm.nih.gov/sra/{normalized}"
