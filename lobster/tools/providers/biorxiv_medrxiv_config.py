"""
Configuration for bioRxiv and medRxiv provider.

This module provides configuration options for the unified bioRxiv/medRxiv provider,
which handles preprint metadata retrieval, full-text JATS XML parsing, and dataset
discovery from both bioRxiv and medRxiv servers.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BioRxivMedRxivConfig(BaseModel):
    """
    Configuration for bioRxiv/medRxiv provider.

    This configuration controls API access, rate limiting, pagination, and JATS XML
    parsing behavior for the unified bioRxiv/medRxiv provider.

    Attributes:
        base_url: Base URL for bioRxiv REST API (default: https://api.biorxiv.org/)
        default_server: Default server to use when not specified ("biorxiv" | "medrxiv")
        max_results: Maximum number of results to return in search (1-1000)
        page_size: Number of results per API page (API enforced: 100)
        requests_per_second: Rate limit for API requests (default: 1.0 req/s)
        max_retry: Maximum number of retry attempts for failed requests
        backoff_delay: Base delay in seconds for exponential backoff (1s, 2s, 4s, etc.)
        timeout: Request timeout in seconds
        fetch_jatsxml: Whether to fetch JATS XML for full-text extraction
        parse_methods: Whether to parse methods section from JATS XML
        parse_tables: Whether to parse tables from JATS XML
        parse_software: Whether to detect software tools from JATS XML

    Examples:
        >>> config = BioRxivMedRxivConfig()  # Use defaults
        >>> config = BioRxivMedRxivConfig(default_server="medrxiv", max_results=50)
        >>> config = BioRxivMedRxivConfig(fetch_jatsxml=False)  # Metadata only
    """

    # API settings
    base_url: str = Field(
        default="https://api.biorxiv.org/",
        description="Base URL for bioRxiv REST API",
    )
    default_server: Literal["biorxiv", "medrxiv"] = Field(
        default="biorxiv", description="Default server when not specified"
    )

    # Search and pagination settings
    max_results: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of results to return in search",
    )
    page_size: int = Field(
        default=100,
        ge=1,
        le=100,
        description="Results per page (API enforced limit: 100)",
    )

    # Rate limiting and retry settings
    requests_per_second: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Rate limit for API requests (polite: 1 req/s)",
    )
    max_retry: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts for failed requests"
    )
    backoff_delay: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Base delay for exponential backoff (seconds)",
    )
    timeout: int = Field(
        default=30, ge=5, le=120, description="Request timeout in seconds"
    )

    # JATS XML parsing settings
    fetch_jatsxml: bool = Field(
        default=True, description="Fetch JATS XML for full-text extraction"
    )
    parse_methods: bool = Field(
        default=True, description="Parse methods section from JATS XML"
    )
    parse_tables: bool = Field(default=True, description="Parse tables from JATS XML")
    parse_software: bool = Field(
        default=True, description="Detect software tools from JATS XML"
    )

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True
