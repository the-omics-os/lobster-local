"""
Unified identifier resolution for biobank accessions.

This module provides centralized accession validation and extraction,
eliminating pattern duplication across providers.

Example:
    >>> from lobster.core.identifiers import get_accession_resolver
    >>> resolver = get_accession_resolver()
    >>> resolver.detect_database("GSE12345")
    'NCBI Gene Expression Omnibus'
"""

from lobster.core.identifiers.accession_resolver import (
    AccessionResolver,
    get_accession_resolver,
    reset_resolver,
)

__all__ = [
    "AccessionResolver",
    "get_accession_resolver",
    "reset_resolver",
]
