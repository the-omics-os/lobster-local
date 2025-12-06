"""
Standardized download URL schema for all providers.

This module defines the canonical return type for all provider.get_download_urls() methods,
ensuring consistent interface for DownloadOrchestrator regardless of database source.

Example:
    >>> from lobster.core.schemas.download_urls import DownloadFile, DownloadUrlResult
    >>> result = DownloadUrlResult(
    ...     accession="GSE12345",
    ...     database="geo",
    ...     primary_files=[DownloadFile(url="ftp://...", filename="matrix.txt.gz")]
    ... )
    >>> result.get_all_urls()
    ['ftp://...']
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DownloadFile(BaseModel):
    """
    Single downloadable file with metadata.

    This is the atomic unit of download - represents one file that can be
    downloaded, validated, and stored.
    """

    url: str = Field(..., description="Download URL (HTTP, FTP, or S3)")
    filename: str = Field(..., description="Target filename for saving")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    checksum: Optional[str] = Field(None, description="MD5 or SHA256 checksum")
    checksum_type: Optional[str] = Field(
        "md5", description="Checksum algorithm (md5, sha256)"
    )
    file_type: Optional[str] = Field(
        None, description="File type category (raw, processed, metadata, etc.)"
    )
    description: Optional[str] = Field(None, description="Human-readable description")

    def to_simple_url(self) -> str:
        """Return just the URL for backward compatibility."""
        return self.url

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "url": self.url,
            "filename": self.filename,
            "size": self.size_bytes,
            "md5": self.checksum if self.checksum_type == "md5" else None,
        }


class DownloadUrlResult(BaseModel):
    """
    Standardized download URL structure for all providers.

    This is the canonical return type for all provider.get_download_urls() methods.
    Provides consistent interface for DownloadOrchestrator regardless of database.

    Attributes:
        accession: Dataset accession (e.g., GSE12345, PXD012345, SRP123456)
        database: Source database name (should be from SupportedDatabase)
        primary_files: Main data files (expression matrix, protein intensities, etc.)
        raw_files: Raw data files (FASTQ, .raw, .mzML)
        processed_files: Processed result files (counts, search results)
        supplementary_files: Supplementary materials
        metadata_files: Metadata and annotation files
        ftp_base: FTP base directory URL for bulk downloads
        total_size_bytes: Total estimated download size
        recommended_strategy: Suggested download strategy for this dataset

    Example:
        >>> result = DownloadUrlResult(
        ...     accession="GSE12345",
        ...     database="geo",
        ...     primary_files=[
        ...         DownloadFile(url="ftp://...", filename="matrix.txt.gz", size_bytes=1024)
        ...     ],
        ...     total_size_bytes=1024,
        ... )
        >>> queue_fields = result.to_queue_entry_fields()
        >>> entry = DownloadQueueEntry(**queue_fields, entry_id="...", source="...")
    """

    # Required fields
    accession: str = Field(..., description="Dataset accession (e.g., GSE12345)")
    database: str = Field(
        ..., description="Source database (geo, sra, pride, massive, etc.)"
    )

    # URL categories (all optional, depends on what database provides)
    primary_files: List[DownloadFile] = Field(
        default_factory=list,
        description="Primary data files (expression matrix, protein intensities)",
    )
    raw_files: List[DownloadFile] = Field(
        default_factory=list,
        description="Raw data files (FASTQ, .raw, .mzML)",
    )
    processed_files: List[DownloadFile] = Field(
        default_factory=list,
        description="Processed result files (counts, search results)",
    )
    supplementary_files: List[DownloadFile] = Field(
        default_factory=list,
        description="Supplementary materials",
    )
    metadata_files: List[DownloadFile] = Field(
        default_factory=list,
        description="Metadata and annotation files",
    )

    # Additional metadata
    ftp_base: Optional[str] = Field(None, description="FTP base directory URL")
    total_size_bytes: Optional[int] = Field(None, description="Total download size")
    recommended_strategy: Optional[str] = Field(
        None, description="Recommended download strategy (matrix, raw, h5ad, etc.)"
    )

    # Database-specific metadata
    mirror: Optional[str] = Field(
        None, description="Download mirror (ena, ncbi, ddbj, etc.)"
    )
    layout: Optional[str] = Field(
        None, description="Data layout (SINGLE, PAIRED for SRA)"
    )
    platform: Optional[str] = Field(None, description="Sequencing/analysis platform")
    run_count: Optional[int] = Field(None, description="Number of runs/samples")

    # Error handling (for failed URL extraction)
    error: Optional[str] = Field(
        None, description="Error message if URL extraction failed"
    )

    # PRIDE-specific: search engine output files
    search_files: List[DownloadFile] = Field(
        default_factory=list,
        description="Search engine output files (PRIDE-specific)",
    )

    @property
    def file_count(self) -> int:
        """Get total count of all files."""
        return len(self.get_all_files())

    @property
    def matrix_url(self) -> Optional[str]:
        """Get primary matrix URL for backward compatibility."""
        for f in self.primary_files:
            if f.file_type == "matrix" or f.filename.endswith((".txt", ".txt.gz")):
                return f.url
        return self.primary_files[0].url if self.primary_files else None

    @property
    def h5_url(self) -> Optional[str]:
        """Get H5AD/H5 URL for backward compatibility."""
        for f in self.primary_files:
            if f.filename.endswith((".h5ad", ".h5")):
                return f.url
        return None

    def get_raw_urls_as_strings(self) -> List[str]:
        """Get raw file URLs as simple string list (GEO compatibility)."""
        return [f.url for f in self.raw_files]

    def get_raw_urls_as_dicts(self) -> List[Dict[str, Any]]:
        """Get raw file URLs as list of dicts (SRA compatibility)."""
        return [f.to_dict() for f in self.raw_files]

    def get_supplementary_urls_as_strings(self) -> List[str]:
        """Get supplementary file URLs as simple string list."""
        return [f.url for f in self.supplementary_files]

    def get_all_files(self) -> List[DownloadFile]:
        """Get flat list of all download files."""
        return (
            self.primary_files
            + self.raw_files
            + self.processed_files
            + self.supplementary_files
            + self.metadata_files
            + self.search_files
        )

    def get_all_urls(self) -> List[str]:
        """Get flat list of all download URLs."""
        return [f.url for f in self.get_all_files()]

    def get_total_size(self) -> int:
        """Calculate total size from all files (or return stored value)."""
        if self.total_size_bytes is not None:
            return self.total_size_bytes

        total = 0
        for f in self.get_all_files():
            if f.size_bytes:
                total += f.size_bytes
        return total

    def to_queue_entry_fields(self) -> Dict[str, Any]:
        """
        Convert to DownloadQueueEntry-compatible fields.

        Provides backward compatibility with existing queue schema.
        The returned dict can be unpacked into DownloadQueueEntry constructor
        (along with entry_id and source).

        Returns:
            Dictionary with fields compatible with DownloadQueueEntry
        """
        # Find primary/matrix URL (first primary file or first processed file)
        matrix_url = None
        if self.primary_files:
            matrix_url = self.primary_files[0].url
        elif self.processed_files:
            matrix_url = self.processed_files[0].url

        # Find H5AD/H5 URL if present
        h5_url = None
        for f in self.primary_files + self.processed_files:
            if f.filename and (
                f.filename.endswith(".h5ad") or f.filename.endswith(".h5")
            ):
                h5_url = f.url
                break

        # Convert raw_files to simple URL list
        raw_urls = [f.url for f in self.raw_files]

        # Convert supplementary to URL list
        supplementary_urls = [f.url for f in self.supplementary_files]

        return {
            "dataset_id": self.accession,
            "database": self.database,
            "matrix_url": matrix_url,
            "h5_url": h5_url,
            "raw_urls": raw_urls,
            "supplementary_urls": supplementary_urls,
            "ftp_base": self.ftp_base,
            "download_size_bytes": self.total_size_bytes,
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """
        Convert to legacy dictionary format for backward compatibility.

        This returns the old-style dict that providers used to return.
        Use for gradual migration.
        """
        return {
            "accession": self.accession,
            "database": self.database,
            "raw_urls": [f.to_dict() for f in self.raw_files],
            "processed_urls": [f.to_dict() for f in self.processed_files],
            "primary_urls": [f.to_dict() for f in self.primary_files],
            "supplementary_urls": [f.url for f in self.supplementary_files],
            "ftp_base": self.ftp_base,
            "total_size_bytes": self.total_size_bytes,
            "mirror": self.mirror,
            "layout": self.layout,
            "platform": self.platform,
            "run_count": self.run_count,
            "recommended_strategy": self.recommended_strategy,
        }

    @classmethod
    def from_sra_response(cls, response: Dict[str, Any]) -> "DownloadUrlResult":
        """
        Create from SRA provider's get_download_urls() response.

        Args:
            response: Dict from sra_provider.get_download_urls()

        Returns:
            Standardized DownloadUrlResult
        """
        raw_files = []
        for item in response.get("raw_urls", []):
            if isinstance(item, dict):
                raw_files.append(
                    DownloadFile(
                        url=item.get("url", ""),
                        filename=item.get("filename", ""),
                        size_bytes=item.get("size"),
                        checksum=item.get("md5"),
                        checksum_type="md5",
                        file_type="raw",
                    )
                )
            elif isinstance(item, str):
                raw_files.append(
                    DownloadFile(
                        url=item,
                        filename=item.split("/")[-1],
                        file_type="raw",
                    )
                )

        return cls(
            accession=response.get("accession", ""),
            database=response.get("database", "sra"),
            raw_files=raw_files,
            ftp_base=response.get("ftp_base"),
            total_size_bytes=response.get("total_size_bytes"),
            mirror=response.get("mirror"),
            layout=response.get("layout"),
            platform=response.get("platform"),
            run_count=response.get("run_count"),
            recommended_strategy="FASTQ_FIRST",
        )

    @classmethod
    def from_geo_response(cls, response: Dict[str, Any]) -> "DownloadUrlResult":
        """
        Create from GEO provider's get_download_urls() response.

        Args:
            response: Dict from geo_provider.get_download_urls()

        Returns:
            Standardized DownloadUrlResult
        """
        primary_files = []
        raw_files = []
        supplementary_files = []

        # Matrix URL is primary
        if response.get("matrix_url"):
            primary_files.append(
                DownloadFile(
                    url=response["matrix_url"],
                    filename=response["matrix_url"].split("/")[-1],
                    file_type="matrix",
                )
            )

        # H5 URL is also primary
        if response.get("h5_url"):
            primary_files.append(
                DownloadFile(
                    url=response["h5_url"],
                    filename=response["h5_url"].split("/")[-1],
                    file_type="h5ad",
                )
            )

        # Raw URLs
        for url in response.get("raw_urls", []):
            if isinstance(url, str):
                raw_files.append(
                    DownloadFile(
                        url=url,
                        filename=url.split("/")[-1],
                        file_type="raw",
                    )
                )

        # Supplementary URLs
        for url in response.get("supplementary_urls", []):
            if isinstance(url, str):
                supplementary_files.append(
                    DownloadFile(
                        url=url,
                        filename=url.split("/")[-1],
                        file_type="supplementary",
                    )
                )

        # GEO provider uses "geo_id" key, support both for flexibility
        accession = response.get("accession") or response.get("geo_id", "")

        return cls(
            accession=accession,
            database="geo",
            primary_files=primary_files,
            raw_files=raw_files,
            supplementary_files=supplementary_files,
            ftp_base=response.get("ftp_base"),
            recommended_strategy="matrix" if primary_files else "raw",
            error=response.get("error"),  # Pass through any error message
        )

    @classmethod
    def from_pride_response(cls, response: Dict[str, Any]) -> "DownloadUrlResult":
        """
        Create from PRIDE provider's get_download_urls() response.

        Args:
            response: Dict from pride_provider.get_download_urls()

        Returns:
            Standardized DownloadUrlResult
        """
        raw_files = []
        processed_files = []

        # PRIDE returns List[Dict] with fileName, downloadLink
        for item in response.get("raw_urls", []):
            if isinstance(item, dict):
                raw_files.append(
                    DownloadFile(
                        url=item.get("downloadLink", item.get("url", "")),
                        filename=item.get("fileName", item.get("filename", "")),
                        size_bytes=item.get("fileSize", item.get("size")),
                        file_type="raw",
                    )
                )

        # PRIDE provider uses "processed_urls" and "result_files" - merge both
        for key in ["processed_urls", "result_files"]:
            for item in response.get(key, []):
                if isinstance(item, dict):
                    processed_files.append(
                        DownloadFile(
                            url=item.get("downloadLink", item.get("url", "")),
                            filename=item.get("fileName", item.get("filename", "")),
                            size_bytes=item.get("fileSize", item.get("size")),
                            file_type="processed",
                        )
                    )

        # PRIDE-specific: search engine output files
        search_files = []
        for item in response.get("search_files", []):
            if isinstance(item, dict):
                search_files.append(
                    DownloadFile(
                        url=item.get("downloadLink", item.get("url", "")),
                        filename=item.get("fileName", item.get("filename", "")),
                        size_bytes=item.get("fileSize", item.get("size")),
                        file_type="search",
                    )
                )

        return cls(
            accession=response.get("accession", ""),
            database="pride",
            raw_files=raw_files,
            processed_files=processed_files,
            search_files=search_files,
            ftp_base=response.get("ftp_base"),
            recommended_strategy="processed" if processed_files else "raw",
        )

    @classmethod
    def from_massive_response(cls, response: Dict[str, Any]) -> "DownloadUrlResult":
        """
        Create from MassIVE provider's get_download_urls() response.

        Args:
            response: Dict from massive_provider.get_download_urls()

        Returns:
            Standardized DownloadUrlResult
        """
        raw_files = []
        processed_files = []

        for item in response.get("raw_urls", []):
            if isinstance(item, dict):
                raw_files.append(
                    DownloadFile(
                        url=item.get("url", ""),
                        filename=item.get("filename", ""),
                        size_bytes=item.get("size"),
                        file_type="raw",
                    )
                )

        for item in response.get("result_files", []):
            if isinstance(item, dict):
                processed_files.append(
                    DownloadFile(
                        url=item.get("url", ""),
                        filename=item.get("filename", ""),
                        size_bytes=item.get("size"),
                        file_type="processed",
                    )
                )

        return cls(
            accession=response.get("accession", ""),
            database="massive",
            raw_files=raw_files,
            processed_files=processed_files,
            ftp_base=response.get("ftp_base"),
            recommended_strategy="processed" if processed_files else "raw",
        )
