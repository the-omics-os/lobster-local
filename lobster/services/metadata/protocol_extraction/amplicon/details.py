"""
Amplicon protocol details dataclass for 16S/ITS microbiome studies.

This module defines the AmpliconProtocolDetails dataclass containing all
fields specific to amplicon-based metagenomics protocols.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lobster.services.metadata.protocol_extraction.base import BaseProtocolDetails


@dataclass
class AmpliconProtocolDetails(BaseProtocolDetails):
    """
    Extracted protocol details from 16S/ITS microbiome publication text.

    Extends BaseProtocolDetails with amplicon-specific fields for primers,
    V-regions, PCR conditions, sequencing parameters, and bioinformatics.

    Examples:
        >>> details = AmpliconProtocolDetails()
        >>> details.v_region = "V3-V4"
        >>> details.forward_primer = "515F"
        >>> details.to_dict()
        {'v_region': 'V3-V4', 'forward_primer': '515F', 'confidence': 0.0}
    """

    # Primer information
    forward_primer: Optional[str] = None
    forward_primer_sequence: Optional[str] = None
    forward_primer_source: Optional[str] = None
    forward_primer_warning: Optional[str] = None
    reverse_primer: Optional[str] = None
    reverse_primer_sequence: Optional[str] = None
    reverse_primer_source: Optional[str] = None
    reverse_primer_warning: Optional[str] = None

    # V-region and target gene
    v_region: Optional[str] = None  # e.g., "V3-V4", "V4", "V1-V2"
    target_gene: str = "16S rRNA"  # 16S rRNA, ITS, 18S rRNA

    # PCR conditions
    annealing_temperature: Optional[float] = None
    pcr_cycles: Optional[int] = None
    polymerase: Optional[str] = None

    # Sequencing parameters
    platform: Optional[str] = None
    instrument: Optional[str] = None
    read_length: Optional[int] = None
    paired_end: Optional[bool] = None

    # Reference database
    reference_database: Optional[str] = None
    database_version: Optional[str] = None

    # Bioinformatics pipeline
    pipeline: Optional[str] = None
    pipeline_version: Optional[str] = None
    clustering_method: Optional[str] = None  # OTU, ASV, zOTU
    clustering_threshold: Optional[float] = None  # 97%, 99%, etc.

    # Quality metrics
    quality_filter: Optional[str] = None
    min_reads: Optional[int] = None

    @property
    def primer_set(self) -> Optional[str]:
        """
        Generate combined primer set string for MetagenomicsMetadataSchema compatibility.

        Returns:
            Optional[str]: Primer set string like "515F-806R" if both primers present,
                          single primer name if only one present, or None if neither.

        Examples:
            >>> details = AmpliconProtocolDetails(forward_primer="515F", reverse_primer="806R")
            >>> details.primer_set
            '515F-806R'
            >>> details = AmpliconProtocolDetails(forward_primer="515F")
            >>> details.primer_set
            '515F'
            >>> details = AmpliconProtocolDetails()
            >>> details.primer_set is None
            True
        """
        if self.forward_primer and self.reverse_primer:
            return f"{self.forward_primer}-{self.reverse_primer}"
        elif self.forward_primer:
            return self.forward_primer
        elif self.reverse_primer:
            return self.reverse_primer
        return None

    def to_schema_dict(self) -> Dict[str, Any]:
        """
        Convert protocol details to metagenomics schema obs-compatible field names.

        This method maps AmpliconProtocolDetails fields to the field names expected
        by the metagenomics schema for storage in adata.obs. Only non-None fields
        are included in the output.

        Field mappings:
            - forward_primer -> forward_primer_name
            - forward_primer_sequence -> forward_primer_seq
            - reverse_primer -> reverse_primer_name
            - reverse_primer_sequence -> reverse_primer_seq
            - v_region -> amplicon_region
            - target_gene -> target_gene (unchanged)
            - platform -> sequencing_platform
            - read_length -> read_length (unchanged)

        Returns:
            Dict[str, Any]: Dictionary with schema-compatible field names.

        Examples:
            >>> details = AmpliconProtocolDetails(
            ...     forward_primer="515F",
            ...     forward_primer_sequence="GTGCCAGCMGCCGCGGTAA",
            ...     reverse_primer="806R",
            ...     v_region="V3-V4",
            ...     platform="Illumina MiSeq",
            ... )
            >>> schema_dict = details.to_schema_dict()
            >>> schema_dict["forward_primer_name"]
            '515F'
            >>> schema_dict["amplicon_region"]
            'V3-V4'
            >>> schema_dict["sequencing_platform"]
            'Illumina MiSeq'
        """
        # Define field mappings: (source_field, target_field)
        field_mappings = [
            ("forward_primer", "forward_primer_name"),
            ("forward_primer_sequence", "forward_primer_seq"),
            ("reverse_primer", "reverse_primer_name"),
            ("reverse_primer_sequence", "reverse_primer_seq"),
            ("v_region", "amplicon_region"),
            ("target_gene", "target_gene"),
            ("platform", "sequencing_platform"),
            ("read_length", "read_length"),
        ]

        result: Dict[str, Any] = {}
        for source_field, target_field in field_mappings:
            value = getattr(self, source_field, None)
            if value is not None:
                result[target_field] = value

        # Add primer_set if available (derived property)
        if self.primer_set:
            result["primer_set"] = self.primer_set

        return result

    def to_uns_dict(self) -> Dict[str, Any]:
        """
        Convert protocol details to dictionary for storage in adata.uns["protocol_details"].

        This method creates a comprehensive dictionary suitable for storing study-level
        protocol information in the unstructured annotations slot of AnnData objects.
        PCR conditions are nested in a sub-dict only if any PCR field is present.

        Structure:
            - Primer info: forward_primer, forward_primer_sequence, forward_primer_source,
              forward_primer_warning, reverse_primer, reverse_primer_sequence, etc.
            - Target region: target_region (v_region), amplicon_target, target_gene
            - Pipeline info: taxonomy_database, taxonomy_database_version, pipeline,
              pipeline_version, clustering_method, clustering_threshold
            - PCR conditions (nested): annealing_temperature_celsius, pcr_cycles, polymerase
            - Extraction metadata: extraction_confidence, extraction_warnings, extraction_notes

        Returns:
            Dict[str, Any]: Dictionary ready for adata.uns["protocol_details"] storage.
                           Only includes fields/sections with non-None values.

        Examples:
            >>> details = AmpliconProtocolDetails(
            ...     forward_primer="515F",
            ...     forward_primer_sequence="GTGCCAGCMGCCGCGGTAA",
            ...     v_region="V3-V4",
            ...     target_gene="16S rRNA",
            ...     annealing_temperature=55.0,
            ...     pcr_cycles=30,
            ...     confidence=0.85,
            ... )
            >>> uns_dict = details.to_uns_dict()
            >>> uns_dict["forward_primer"]
            '515F'
            >>> uns_dict["target_region"]
            'V3-V4'
            >>> uns_dict["pcr_conditions"]["annealing_temperature_celsius"]
            55.0
            >>> uns_dict["extraction_confidence"]
            0.85
        """
        result: Dict[str, Any] = {}

        # Primer information (study-level)
        primer_fields = [
            ("forward_primer", "forward_primer"),
            ("forward_primer_sequence", "forward_primer_sequence"),
            ("forward_primer_source", "forward_primer_source"),
            ("forward_primer_warning", "forward_primer_warning"),
            ("reverse_primer", "reverse_primer"),
            ("reverse_primer_sequence", "reverse_primer_sequence"),
            ("reverse_primer_source", "reverse_primer_source"),
            ("reverse_primer_warning", "reverse_primer_warning"),
        ]
        for source_field, target_field in primer_fields:
            value = getattr(self, source_field, None)
            if value is not None:
                result[target_field] = value

        # Target region information
        if self.v_region is not None:
            result["target_region"] = self.v_region
            # Generate amplicon_target (e.g., "16S rRNA V3-V4")
            if self.target_gene:
                result["amplicon_target"] = f"{self.target_gene} {self.v_region}"

        if self.target_gene is not None:
            result["target_gene"] = self.target_gene

        # Pipeline and database information
        if self.reference_database is not None:
            result["taxonomy_database"] = self.reference_database
        if self.database_version is not None:
            result["taxonomy_database_version"] = self.database_version
        if self.pipeline is not None:
            result["pipeline"] = self.pipeline
        if self.pipeline_version is not None:
            result["pipeline_version"] = self.pipeline_version
        if self.clustering_method is not None:
            result["clustering_method"] = self.clustering_method
        if self.clustering_threshold is not None:
            result["clustering_threshold"] = self.clustering_threshold

        # PCR conditions (nested dict) - only include if at least one field is present
        pcr_conditions: Dict[str, Any] = {}
        if self.annealing_temperature is not None:
            pcr_conditions["annealing_temperature_celsius"] = self.annealing_temperature
        if self.pcr_cycles is not None:
            pcr_conditions["pcr_cycles"] = self.pcr_cycles
        if self.polymerase is not None:
            pcr_conditions["polymerase"] = self.polymerase

        if pcr_conditions:
            result["pcr_conditions"] = pcr_conditions

        # Extraction metadata
        result["extraction_confidence"] = self.confidence

        if self.validation_warnings:
            result["extraction_warnings"] = self.validation_warnings

        if self.extraction_notes:
            result["extraction_notes"] = self.extraction_notes

        return result


# V-region definitions for 16S rRNA gene
# All positions are based on the E. coli 16S rRNA gene numbering system,
# which is the standard reference for 16S rRNA gene position annotation.
V_REGIONS: Dict[str, Dict[str, int]] = {
    "V1": {"start": 69, "end": 99, "length": 30},
    "V2": {"start": 137, "end": 242, "length": 105},
    "V3": {"start": 433, "end": 497, "length": 64},
    "V4": {"start": 576, "end": 682, "length": 106},
    "V5": {"start": 822, "end": 879, "length": 57},
    "V6": {"start": 986, "end": 1043, "length": 57},
    "V7": {"start": 1117, "end": 1173, "length": 56},
    "V8": {"start": 1243, "end": 1294, "length": 51},
    "V9": {"start": 1435, "end": 1465, "length": 30},
}
