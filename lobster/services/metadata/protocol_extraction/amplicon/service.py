"""
Amplicon Protocol Extraction Service for 16S/ITS microbiome studies.

This service extracts technical protocol information from publication text:
- Primer sequences (515F, 806R, 27F, 1492R, etc.)
- V-region amplified (V1-V2, V3-V4, V4, etc.)
- PCR conditions (annealing temperature, cycles)
- Sequencing parameters (read length, platform)
- Reference databases (SILVA, Greengenes, RDP)
- Bioinformatics pipelines (QIIME2, DADA2, mothur)

Used by metadata_assistant and research_agent for publication processing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from anndata import AnnData

from lobster.core.interfaces.validator import ValidationResult
from lobster.services.metadata.protocol_extraction.amplicon.details import (
    V_REGIONS,
    AmpliconProtocolDetails,
)
from lobster.services.metadata.protocol_extraction.base import (
    IProtocolExtractionService,
)

logger = logging.getLogger(__name__)


def _load_primers_from_json() -> Dict[str, Dict[str, Any]]:
    """
    Load primer database from external JSON file.

    This decouples biological constants from code, enabling non-developer updates.
    """
    json_path = Path(__file__).parent / "resources" / "primers.json"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("primers", {})
    except FileNotFoundError:
        logger.warning(f"primers.json not found at {json_path}, using empty database")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse primers.json: {e}")
        return {}


class AmpliconProtocolService(IProtocolExtractionService):
    """
    Service for extracting 16S/ITS microbiome protocol details from publication text.

    Extracts:
    - Primer sequences and names
    - V-region amplified
    - PCR conditions
    - Sequencing parameters
    - Reference databases
    - Bioinformatics pipelines

    Examples:
        >>> service = AmpliconProtocolService()
        >>> text = "The V3-V4 region was amplified using primers 515F and 806R..."
        >>> details, result = service.extract_protocol(text)
        >>> details.v_region
        'V3-V4'
        >>> details.forward_primer
        '515F'
    """

    # Supported domain aliases
    SUPPORTED_DOMAINS = {"amplicon", "16s", "its", "metagenomics"}

    def __init__(self):
        """Initialize the service."""
        self.logger = logging.getLogger(__name__)
        self.primers = _load_primers_from_json()
        self._compile_patterns()

    @property
    def domain(self) -> str:
        """Return domain identifier."""
        return "amplicon"

    @classmethod
    def supports_domain(cls, domain: str) -> bool:
        """Check if this service handles the given domain."""
        return domain.lower() in cls.SUPPORTED_DOMAINS

    def load_resources(self) -> Dict[str, Any]:
        """Load domain-specific reference data from JSON files."""
        return {
            "primers": self.primers,
            "v_regions": V_REGIONS,
        }

    def _compile_patterns(self):
        """Compile regex patterns for extraction."""

        # CRITICAL: Sort primer names by length (descending) before compiling regex.
        # This ensures specific variants like 515F-Y are matched before 515F.
        sorted_primer_names = sorted(self.primers.keys(), key=len, reverse=True)
        primer_names = "|".join(re.escape(p) for p in sorted_primer_names)

        self.primer_name_pattern = (
            re.compile(rf"\b({primer_names})\b", re.IGNORECASE)
            if primer_names
            else None
        )

        # Generic primer patterns (capture unknown primers)
        self.generic_primer_pattern = re.compile(
            r"\b(\d{2,4}[FRfr](?:-[A-Za-z]+)?)\b"  # e.g., 515F, 806R, 515F-Y
        )

        # Primer sequence pattern (DNA sequences 15-30 bp)
        self.primer_sequence_pattern = re.compile(
            r"['\"]?([ACGTMRWSYKVHDBN]{15,30})['\"]?", re.IGNORECASE
        )

        # V-region patterns
        self.v_region_pattern = re.compile(
            r"\b(V[1-9](?:\s*[-–]\s*V[1-9])?)\s*(?:region|hypervariable|variable)?\b",
            re.IGNORECASE,
        )

        # PCR conditions
        self.annealing_temp_pattern = re.compile(
            r"(?:annealing|anneal)\s*(?:temperature|temp\.?)?\s*(?:of|at|:)?\s*(\d{2})\s*[°º]?\s*C",
            re.IGNORECASE,
        )
        self.pcr_cycles_pattern = re.compile(
            r"(\d{2,3})\s*(?:PCR\s*)?cycles?", re.IGNORECASE
        )

        # Sequencing platform
        self.platform_patterns = {
            "Illumina MiSeq": re.compile(r"\bMiSeq\b", re.IGNORECASE),
            "Illumina HiSeq": re.compile(r"\bHiSeq\b", re.IGNORECASE),
            "Illumina NovaSeq": re.compile(r"\bNovaSeq\b", re.IGNORECASE),
            "Illumina NextSeq": re.compile(r"\bNextSeq\b", re.IGNORECASE),
            "Ion Torrent": re.compile(r"\bIon\s*Torrent\b", re.IGNORECASE),
            "PacBio": re.compile(r"\bPacBio|SMRT\b", re.IGNORECASE),
            "Oxford Nanopore": re.compile(
                r"\bNanopore|MinION|GridION\b", re.IGNORECASE
            ),
            "454": re.compile(r"\b454|Roche\s*454\b", re.IGNORECASE),
        }

        # Read length
        self.read_length_pattern = re.compile(
            r"(\d{2,3})\s*(?:bp|base\s*pairs?)\s*(?:paired[-\s]?end|PE|reads?)?",
            re.IGNORECASE,
        )
        self.paired_end_pattern = re.compile(
            r"\b(?:paired[-\s]?end|PE|2\s*[×x]\s*\d+)\b", re.IGNORECASE
        )

        # Reference databases
        self.database_patterns = {
            "SILVA": re.compile(r"\bSILVA\s*(v?\d+(?:\.\d+)?)?", re.IGNORECASE),
            "Greengenes": re.compile(
                r"\bGreengenes?\s*(v?\d+(?:\.\d+)?)?", re.IGNORECASE
            ),
            "RDP": re.compile(r"\bRDP\s*(v?\d+(?:\.\d+)?)?", re.IGNORECASE),
            "NCBI 16S": re.compile(r"\bNCBI\s*16S\b", re.IGNORECASE),
            "GTDB": re.compile(r"\bGTDB\s*(r?\d+(?:\.\d+)?)?", re.IGNORECASE),
            "UNITE": re.compile(r"\bUNITE\s*(v?\d+(?:\.\d+)?)?", re.IGNORECASE),  # ITS
        }

        # Bioinformatics pipelines
        self.pipeline_patterns = {
            "QIIME2": re.compile(r"\bQIIME\s*2?\s*(v?[\d.]+)?", re.IGNORECASE),
            "DADA2": re.compile(r"\bDADA2?\s*(v?[\d.]+)?", re.IGNORECASE),
            "mothur": re.compile(r"\bmothur\s*(v?[\d.]+)?", re.IGNORECASE),
            "USEARCH": re.compile(r"\bUSEARCH\s*(v?[\d.]+)?", re.IGNORECASE),
            "VSEARCH": re.compile(r"\bVSEARCH\s*(v?[\d.]+)?", re.IGNORECASE),
            "Deblur": re.compile(r"\bDeblur\s*(v?[\d.]+)?", re.IGNORECASE),
            "PICRUSt": re.compile(r"\bPICRUSt\s*2?\s*(v?[\d.]+)?", re.IGNORECASE),
            "LEfSe": re.compile(r"\bLEfSe\b", re.IGNORECASE),
            "phyloseq": re.compile(r"\bphyloseq\s*(v?[\d.]+)?", re.IGNORECASE),
        }

        # Clustering method
        self.clustering_patterns = {
            "ASV": re.compile(
                r"\b(?:ASV|amplicon\s*sequence\s*variant)s?\b", re.IGNORECASE
            ),
            "OTU": re.compile(
                r"\b(?:OTU|operational\s*taxonomic\s*unit)s?\b", re.IGNORECASE
            ),
            "zOTU": re.compile(r"\bzOTUs?\b", re.IGNORECASE),
        }
        self.clustering_threshold_pattern = re.compile(
            r"(\d{2,3})\s*%\s*(?:similarity|identity|threshold)", re.IGNORECASE
        )

    def extract_protocol(
        self, text: str, source: str = "unknown"
    ) -> Tuple[AmpliconProtocolDetails, ValidationResult]:
        """
        Extract protocol details from publication text.

        Args:
            text: Publication text (methods section preferred)
            source: Source identifier for logging

        Returns:
            Tuple of (AmpliconProtocolDetails, ValidationResult)

        Examples:
            >>> service = AmpliconProtocolService()
            >>> text = '''
            ...     The V3-V4 region of 16S rRNA gene was amplified using
            ...     primers 515F (GTGCCAGCMGCCGCGGTAA) and 806R. PCR was
            ...     performed for 30 cycles with annealing at 55°C.
            ...     Sequencing was done on Illumina MiSeq (2×250 bp).
            ... '''
            >>> details, result = service.extract_protocol(text)
            >>> details.v_region
            'V3-V4'
            >>> details.forward_primer
            '515F'
            >>> details.pcr_cycles
            30
        """
        result = ValidationResult()
        details = AmpliconProtocolDetails()

        if not text or len(text) < 50:
            result.add_warning(
                f"Text too short for protocol extraction: {len(text)} chars"
            )
            return details, result

        self.logger.info(f"Extracting protocol from {source} ({len(text)} chars)")

        # Track extraction success
        extracted_fields = []

        # 1. Extract V-region
        v_region = self._extract_v_region(text)
        if v_region:
            details.v_region = v_region
            extracted_fields.append("v_region")

        # 2. Extract primers
        primers = self._extract_primers(text)
        if primers.get("forward"):
            details.forward_primer = primers["forward"]
            details.forward_primer_sequence = primers.get("forward_sequence")
            details.forward_primer_source = primers.get("forward_source")
            details.forward_primer_warning = primers.get("forward_warning")
            extracted_fields.append("forward_primer")
        if primers.get("reverse"):
            details.reverse_primer = primers["reverse"]
            details.reverse_primer_sequence = primers.get("reverse_sequence")
            details.reverse_primer_source = primers.get("reverse_source")
            details.reverse_primer_warning = primers.get("reverse_warning")
            extracted_fields.append("reverse_primer")

        # 3. Extract PCR conditions
        pcr = self._extract_pcr_conditions(text)
        if pcr.get("annealing_temperature"):
            details.annealing_temperature = pcr["annealing_temperature"]
            extracted_fields.append("annealing_temperature")
        if pcr.get("cycles"):
            details.pcr_cycles = pcr["cycles"]
            extracted_fields.append("pcr_cycles")

        # 4. Extract sequencing parameters
        seq = self._extract_sequencing(text)
        if seq.get("platform"):
            details.platform = seq["platform"]
            extracted_fields.append("platform")
        if seq.get("read_length"):
            details.read_length = seq["read_length"]
            extracted_fields.append("read_length")
        if seq.get("paired_end") is not None:
            details.paired_end = seq["paired_end"]
            extracted_fields.append("paired_end")

        # 5. Extract reference database
        db = self._extract_database(text)
        if db.get("name"):
            details.reference_database = db["name"]
            details.database_version = db.get("version")
            extracted_fields.append("reference_database")

        # 6. Extract bioinformatics pipeline
        pipeline = self._extract_pipeline(text)
        if pipeline.get("name"):
            details.pipeline = pipeline["name"]
            details.pipeline_version = pipeline.get("version")
            extracted_fields.append("pipeline")

        # 7. Extract clustering method
        clustering = self._extract_clustering(text)
        if clustering.get("method"):
            details.clustering_method = clustering["method"]
            extracted_fields.append("clustering_method")
        if clustering.get("threshold"):
            details.clustering_threshold = clustering["threshold"]
            extracted_fields.append("clustering_threshold")

        # 8. Validate primer-region consistency
        validation_warnings = self._validate_primer_region_consistency(details)
        details.validation_warnings = validation_warnings

        # Calculate confidence
        total_possible = 12  # Number of extractable fields
        details.confidence = len(extracted_fields) / total_possible

        # Add result info
        if extracted_fields:
            result.add_info(
                f"Extracted {len(extracted_fields)} protocol fields: {', '.join(extracted_fields)}"
            )
        else:
            result.add_warning("No protocol details extracted from text")

        if validation_warnings:
            for warning in validation_warnings:
                result.add_warning(warning)

        self.logger.info(
            f"Protocol extraction complete: {len(extracted_fields)} fields, "
            f"confidence={details.confidence:.1%}, "
            f"warnings={len(validation_warnings)}"
        )

        return details, result

    def _extract_v_region(self, text: str) -> Optional[str]:
        """Extract V-region from text."""
        match = self.v_region_pattern.search(text)
        if match:
            v_region = match.group(1).upper()
            # Normalize format (e.g., "V3 - V4" -> "V3-V4")
            v_region = re.sub(r"\s*[-–]\s*", "-", v_region)
            return v_region
        return None

    def _extract_primers(self, text: str) -> Dict[str, Optional[str]]:
        """Extract primer names, sequences, and metadata."""
        result = {
            "forward": None,
            "forward_sequence": None,
            "forward_source": None,
            "forward_warning": None,
            "reverse": None,
            "reverse_sequence": None,
            "reverse_source": None,
            "reverse_warning": None,
        }

        # Find known primer names (sorted by length, longest first)
        known_matches = []
        if self.primer_name_pattern:
            known_matches = self.primer_name_pattern.findall(text)

        generic_matches = self.generic_primer_pattern.findall(text)

        all_primers = list(set(known_matches + generic_matches))

        for primer in all_primers:
            primer_upper = primer.upper()
            # Check if known primer (case-insensitive lookup)
            primer_info = None
            for key in self.primers:
                if key.upper() == primer_upper:
                    primer_info = self.primers[key]
                    primer_upper = key  # Use canonical casing from JSON
                    break

            if primer_info:
                direction = primer_info.get("direction", "")
                if direction == "forward" and not result["forward"]:
                    result["forward"] = primer_upper
                    result["forward_sequence"] = primer_info.get("sequence")
                    result["forward_source"] = primer_info.get("source")
                    result["forward_warning"] = primer_info.get("warning")
                elif direction == "reverse" and not result["reverse"]:
                    result["reverse"] = primer_upper
                    result["reverse_sequence"] = primer_info.get("sequence")
                    result["reverse_source"] = primer_info.get("source")
                    result["reverse_warning"] = primer_info.get("warning")
            else:
                # Guess direction from suffix for unknown primers
                if primer_upper.endswith("F") and not result["forward"]:
                    result["forward"] = primer_upper
                elif primer_upper.endswith("R") and not result["reverse"]:
                    result["reverse"] = primer_upper

        # Try to find sequences in text for primers without known sequences
        if result["forward"] and not result["forward_sequence"]:
            pattern = rf"{re.escape(result['forward'])}\s*[:\(]?\s*([ACGTMRWSYKVHDBN]{{15,30}})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["forward_sequence"] = match.group(1).upper()

        if result["reverse"] and not result["reverse_sequence"]:
            pattern = rf"{re.escape(result['reverse'])}\s*[:\(]?\s*([ACGTMRWSYKVHDBN]{{15,30}})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["reverse_sequence"] = match.group(1).upper()

        return result

    def _extract_pcr_conditions(self, text: str) -> Dict[str, Any]:
        """Extract PCR conditions."""
        result = {}

        # Annealing temperature
        match = self.annealing_temp_pattern.search(text)
        if match:
            result["annealing_temperature"] = float(match.group(1))

        # PCR cycles
        match = self.pcr_cycles_pattern.search(text)
        if match:
            cycles = int(match.group(1))
            if 20 <= cycles <= 45:  # Reasonable range
                result["cycles"] = cycles

        return result

    def _extract_sequencing(self, text: str) -> Dict[str, Any]:
        """Extract sequencing parameters."""
        result = {}

        # Platform
        for platform, pattern in self.platform_patterns.items():
            if pattern.search(text):
                result["platform"] = platform
                break

        # Read length
        match = self.read_length_pattern.search(text)
        if match:
            length = int(match.group(1))
            if 50 <= length <= 600:  # Reasonable range
                result["read_length"] = length

        # Paired-end
        if self.paired_end_pattern.search(text):
            result["paired_end"] = True
        elif re.search(r"\bsingle[-\s]?end|SE\b", text, re.IGNORECASE):
            result["paired_end"] = False

        return result

    def _extract_database(self, text: str) -> Dict[str, Optional[str]]:
        """Extract reference database."""
        result = {"name": None, "version": None}

        for db_name, pattern in self.database_patterns.items():
            match = pattern.search(text)
            if match:
                result["name"] = db_name
                if match.lastindex and match.group(1):
                    result["version"] = match.group(1).strip()
                break

        return result

    def _extract_pipeline(self, text: str) -> Dict[str, Optional[str]]:
        """Extract bioinformatics pipeline."""
        result = {"name": None, "version": None}

        for pipeline_name, pattern in self.pipeline_patterns.items():
            match = pattern.search(text)
            if match:
                result["name"] = pipeline_name
                if match.lastindex and match.group(1):
                    result["version"] = match.group(1).strip()
                break

        return result

    def _extract_clustering(self, text: str) -> Dict[str, Any]:
        """Extract clustering method and threshold."""
        result = {}

        # Method (ASV, OTU, zOTU)
        for method, pattern in self.clustering_patterns.items():
            if pattern.search(text):
                result["method"] = method
                break

        # Threshold
        match = self.clustering_threshold_pattern.search(text)
        if match:
            threshold = float(match.group(1))
            if 90 <= threshold <= 100:
                result["threshold"] = threshold

        return result

    def _validate_primer_region_consistency(
        self, details: AmpliconProtocolDetails
    ) -> List[str]:
        """
        Validate that extracted primers match the extracted V-region.

        Returns list of validation warnings.
        """
        warnings = []

        if not details.v_region:
            return warnings

        # Parse V-region to get expected regions (e.g., "V3-V4" -> ["V3", "V4"])
        v_region_str = details.v_region.upper()
        expected_regions = set()

        if "-" in v_region_str:
            parts = v_region_str.split("-")
            if len(parts) == 2:
                start_v = parts[0].strip()
                end_v = parts[1].strip()
                # Add all regions in range
                try:
                    start_num = int(start_v[1])
                    end_num = int(end_v[1])
                    for i in range(start_num, end_num + 1):
                        expected_regions.add(f"V{i}")
                except (ValueError, IndexError):
                    expected_regions.add(start_v)
                    expected_regions.add(end_v)
        else:
            expected_regions.add(v_region_str)

        # Check forward primer region
        if details.forward_primer and details.forward_primer in self.primers:
            primer_info = self.primers[details.forward_primer]
            primer_region = primer_info.get("region", "").upper()

            # Forward primer should target start of amplicon region
            if primer_region and primer_region not in expected_regions:
                # Skip ITS validation (different gene)
                if not primer_region.startswith("ITS"):
                    warnings.append(
                        f"Primer-region conflict: {details.forward_primer} targets {primer_region}, "
                        f"but V-region is {details.v_region}"
                    )

        # Check reverse primer region
        if details.reverse_primer and details.reverse_primer in self.primers:
            primer_info = self.primers[details.reverse_primer]
            primer_region = primer_info.get("region", "").upper()

            if primer_region and primer_region not in expected_regions:
                if not primer_region.startswith("ITS"):
                    warnings.append(
                        f"Primer-region conflict: {details.reverse_primer} targets {primer_region}, "
                        f"but V-region is {details.v_region}"
                    )

        return warnings

    def extract_from_methods_section(
        self, methods_text: str, full_text: Optional[str] = None
    ) -> Tuple[AmpliconProtocolDetails, ValidationResult]:
        """
        Extract protocol details with methods section priority.

        First extracts from methods section, then supplements from
        full text if available.

        Args:
            methods_text: Methods section text
            full_text: Optional full publication text

        Returns:
            Tuple of (AmpliconProtocolDetails, ValidationResult)
        """
        # Primary extraction from methods
        details, result = self.extract_protocol(methods_text, source="methods")

        # Supplement from full text if fields missing
        if full_text and details.confidence < 0.5:
            full_details, _ = self.extract_protocol(full_text, source="full_text")

            # Fill missing fields
            for field in [
                "forward_primer",
                "reverse_primer",
                "v_region",
                "platform",
                "pipeline",
                "reference_database",
            ]:
                if not getattr(details, field) and getattr(full_details, field):
                    setattr(details, field, getattr(full_details, field))
                    details.extraction_notes.append(f"{field} from full text")

            # Recalculate confidence
            extracted = sum(
                1
                for f in [
                    details.forward_primer,
                    details.reverse_primer,
                    details.v_region,
                    details.platform,
                    details.pipeline,
                    details.reference_database,
                    details.annealing_temperature,
                    details.pcr_cycles,
                    details.read_length,
                    details.clustering_method,
                ]
                if f is not None
            )
            details.confidence = extracted / 10

        return details, result

    def store_in_adata(
        self,
        adata: AnnData,
        details: AmpliconProtocolDetails,
        store_in_obs: bool = False,
    ) -> AnnData:
        """
        Store extracted protocol details in AnnData object.

        This helper method stores protocol details in the appropriate AnnData slots:
        - uns["protocol_details"]: Always stores the full protocol information
        - obs columns: Optionally adds schema-compatible fields to all rows

        Args:
            adata: AnnData object to modify
            details: AmpliconProtocolDetails instance with extracted protocol info
            store_in_obs: If True, also add schema-compatible fields to adata.obs
                         for all samples. Useful when all samples share the same
                         protocol details.

        Returns:
            AnnData: The modified AnnData object with protocol details stored.

        Examples:
            >>> import anndata as ad
            >>> import numpy as np
            >>> service = AmpliconProtocolService()
            >>> text = "V3-V4 region amplified using 515F and 806R on Illumina MiSeq"
            >>> details, _ = service.extract_protocol(text)
            >>> adata = ad.AnnData(np.random.rand(10, 5))
            >>> adata = service.store_in_adata(adata, details)
            >>> "protocol_details" in adata.uns
            True
            >>> adata.uns["protocol_details"]["forward_primer"]
            '515F'

            # With obs storage for sample-level metadata
            >>> adata = service.store_in_adata(adata, details, store_in_obs=True)
            >>> "forward_primer_name" in adata.obs.columns
            True
            >>> adata.obs["amplicon_region"].iloc[0]
            'V3-V4'
        """
        # Store comprehensive protocol details in uns
        adata.uns["protocol_details"] = details.to_uns_dict()

        # Optionally store schema-compatible fields in obs
        if store_in_obs:
            schema_dict = details.to_schema_dict()
            for field_name, value in schema_dict.items():
                # Add the same value to all rows (study-level metadata)
                adata.obs[field_name] = value

        self.logger.info(
            f"Stored protocol details in adata.uns"
            f"{' and adata.obs' if store_in_obs else ''}"
        )

        return adata
