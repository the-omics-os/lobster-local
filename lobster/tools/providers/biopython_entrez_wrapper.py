"""
Biopython Bio.Entrez wrapper utility for NCBI API access.

This module provides a thin wrapper around Biopython's Bio.Entrez for
consistent NCBI E-utilities access across all providers (SRA, PubMed, GEO).

Pattern inspired by SRAgent repository research (2025-01-15):
- Built-in rate limiting (3 req/s → 10 req/s with API key)
- Automatic XML parsing
- Exponential backoff for HTTP 429
- Proper error handling

Key differences from raw urllib + xmltodict:
✅ Bio.Entrez handles rate limiting automatically
✅ Bio.Entrez parses XML to dicts natively
✅ Bio.Entrez respects NCBI API keys
✅ Bio.Entrez has built-in retry logic
"""

import logging
from typing import Any, Dict, List, Optional

from lobster.config.settings import get_settings

logger = logging.getLogger(__name__)

# Lazy import Bio.Entrez (only when needed)
_Bio_Entrez = None


def _get_bio_entrez():
    """Lazy import of Bio.Entrez."""
    global _Bio_Entrez
    if _Bio_Entrez is None:
        try:
            from Bio import Entrez

            _Bio_Entrez = Entrez
            logger.debug("Bio.Entrez loaded successfully")
        except ImportError as e:
            raise ImportError(
                "Biopython not installed. Install with: pip install biopython"
            ) from e
    return _Bio_Entrez


class BioPythonEntrezWrapper:
    """
    Wrapper for Biopython Bio.Entrez with standardized configuration.

    This wrapper:
    - Configures email (required by NCBI)
    - Configures API key (optional but recommended)
    - Handles rate limiting automatically (Bio.Entrez built-in)
    - Provides consistent error handling
    - Supports all E-utilities: esearch, esummary, efetch, elink

    Usage:
        >>> wrapper = BioPythonEntrezWrapper()
        >>> result = wrapper.esearch(db="sra", term="microbiome", retmax=20)
        >>> ids = result["IdList"]
    """

    def __init__(
        self, email: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        """
        Initialize Bio.Entrez wrapper.

        Args:
            email: Email for NCBI (required by NCBI ToS). Defaults to settings.
            api_key: NCBI API key (optional). Defaults to settings.

        Raises:
            ImportError: If Biopython not installed
        """
        self._entrez = _get_bio_entrez()

        # Configure from settings if not provided
        settings = get_settings()
        self._entrez.email = email or settings.NCBI_EMAIL or "your-email@example.com"
        self._entrez.api_key = api_key or settings.NCBI_API_KEY

        # Rate limiting (Bio.Entrez handles this automatically)
        # - Without API key: 3 requests/second (0.34s delay)
        # - With API key: 10 requests/second (0.1s delay)
        if self._entrez.api_key:
            logger.debug("Bio.Entrez configured with API key (10 req/s rate limit)")
        else:
            logger.debug("Bio.Entrez configured without API key (3 req/s rate limit)")

        logger.debug(f"Bio.Entrez email: {self._entrez.email}")

    def esearch(
        self,
        db: str,
        term: str,
        retmax: int = 20,
        retstart: int = 0,
        usehistory: str = "n",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute NCBI esearch (search database for IDs).

        Args:
            db: Database name (e.g., "sra", "pubmed", "gds")
            term: Search query string with field qualifiers
            retmax: Maximum number of results (default: 20)
            retstart: Offset for pagination (default: 0)
            usehistory: "y" to use history server for large result sets
            **kwargs: Additional parameters for Bio.Entrez.esearch

        Returns:
            Dict with keys: Count, IdList, QueryKey, WebEnv (if usehistory="y")

        Raises:
            Exception: If NCBI API call fails

        Example:
            >>> result = wrapper.esearch(db="sra", term="microbiome", retmax=20)
            >>> ids = result["IdList"]  # List of SRA IDs
            >>> count = int(result["Count"])  # Total results
        """
        try:
            logger.debug(
                f"Bio.Entrez.esearch: db={db}, term={term[:50]}..., retmax={retmax}"
            )

            handle = self._entrez.esearch(
                db=db,
                term=term,
                retmax=retmax,
                retstart=retstart,
                usehistory=usehistory,
                **kwargs,
            )

            result = self._entrez.read(handle)
            handle.close()

            logger.debug(
                f"Bio.Entrez.esearch result: {result.get('Count', 0)} total, {len(result.get('IdList', []))} returned"
            )

            return result

        except Exception as e:
            logger.error(f"Bio.Entrez.esearch error: {e}")
            raise

    def esummary(self, db: str, id: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute NCBI esummary (get summaries for IDs).

        Args:
            db: Database name (e.g., "sra", "pubmed", "gds")
            id: Comma-separated ID list or single ID
            **kwargs: Additional parameters for Bio.Entrez.esummary

        Returns:
            List of summary dicts (one per ID)

        Raises:
            Exception: If NCBI API call fails

        Example:
            >>> summaries = wrapper.esummary(db="sra", id="123456,789012")
            >>> for summary in summaries:
            ...     print(summary.get("Title"))
        """
        try:
            # Handle both single ID and comma-separated list
            id_str = str(id)
            id_count = len(id_str.split(","))

            logger.debug(f"Bio.Entrez.esummary: db={db}, id_count={id_count}")

            handle = self._entrez.esummary(db=db, id=id_str, **kwargs)

            result = self._entrez.read(handle)
            handle.close()

            # Bio.Entrez returns list of dicts
            if not isinstance(result, list):
                result = [result]

            logger.debug(f"Bio.Entrez.esummary result: {len(result)} summaries")

            return result

        except Exception as e:
            logger.error(f"Bio.Entrez.esummary error: {e}")
            raise

    def efetch(
        self, db: str, id: str, rettype: str = "xml", retmode: str = "xml", **kwargs
    ) -> str:
        """
        Execute NCBI efetch (fetch full records).

        Args:
            db: Database name (e.g., "sra", "pubmed", "gds")
            id: Comma-separated ID list or single ID
            rettype: Return type (e.g., "xml", "fasta", "gb")
            retmode: Return mode (e.g., "xml", "text")
            **kwargs: Additional parameters for Bio.Entrez.efetch

        Returns:
            str: Raw content from NCBI (usually XML)

        Raises:
            Exception: If NCBI API call fails

        Example:
            >>> xml_content = wrapper.efetch(db="sra", id="123456", rettype="xml")
        """
        try:
            id_str = str(id)
            id_count = len(id_str.split(","))

            logger.debug(
                f"Bio.Entrez.efetch: db={db}, id_count={id_count}, rettype={rettype}"
            )

            handle = self._entrez.efetch(
                db=db, id=id_str, rettype=rettype, retmode=retmode, **kwargs
            )

            content = handle.read()
            handle.close()

            logger.debug(f"Bio.Entrez.efetch result: {len(content)} bytes")

            return content

        except Exception as e:
            logger.error(f"Bio.Entrez.efetch error: {e}")
            raise

    def elink(self, dbfrom: str, db: str, id: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute NCBI elink (find linked records across databases).

        Args:
            dbfrom: Source database (e.g., "gds" for GEO)
            db: Target database (e.g., "pubmed" for PubMed)
            id: Comma-separated ID list or single ID
            **kwargs: Additional parameters for Bio.Entrez.elink

        Returns:
            List of linkset dicts

        Raises:
            Exception: If NCBI API call fails

        Example:
            >>> links = wrapper.elink(dbfrom="gds", db="pubmed", id="123456")
            >>> for link in links:
            ...     linked_ids = link["LinkSetDb"][0]["Link"]
        """
        try:
            id_str = str(id)
            id_count = len(id_str.split(","))

            logger.debug(
                f"Bio.Entrez.elink: dbfrom={dbfrom}, db={db}, id_count={id_count}"
            )

            handle = self._entrez.elink(dbfrom=dbfrom, db=db, id=id_str, **kwargs)

            result = self._entrez.read(handle)
            handle.close()

            logger.debug(f"Bio.Entrez.elink result: {len(result)} linksets")

            return result

        except Exception as e:
            logger.error(f"Bio.Entrez.elink error: {e}")
            raise


# Module-level convenience instance (optional)
_default_wrapper: Optional[BioPythonEntrezWrapper] = None


def get_default_wrapper() -> BioPythonEntrezWrapper:
    """
    Get or create default Bio.Entrez wrapper (singleton pattern).

    Returns:
        BioPythonEntrezWrapper: Default wrapper configured from settings
    """
    global _default_wrapper
    if _default_wrapper is None:
        _default_wrapper = BioPythonEntrezWrapper()
    return _default_wrapper
