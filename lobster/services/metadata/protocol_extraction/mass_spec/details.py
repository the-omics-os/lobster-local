"""
Mass spectrometry protocol details dataclass for proteomics studies.

This module defines the MassSpecProtocolDetails dataclass containing all
fields specific to mass spectrometry-based proteomics protocols.
"""

from dataclasses import dataclass
from typing import Optional

from lobster.services.metadata.protocol_extraction.base import BaseProtocolDetails


@dataclass
class MassSpecProtocolDetails(BaseProtocolDetails):
    """
    Extracted protocol details from mass spectrometry proteomics publication text.

    Extends BaseProtocolDetails with mass spec-specific fields for acquisition,
    sample preparation, MS parameters, and database search.

    Examples:
        >>> details = MassSpecProtocolDetails()
        >>> details.acquisition_mode = "DDA"
        >>> details.instrument = "Q-Exactive"
        >>> details.to_dict()
        {'acquisition_mode': 'DDA', 'instrument': 'Q-Exactive', 'confidence': 0.0}
    """

    # Acquisition mode
    acquisition_mode: Optional[str] = None  # DDA, DIA, PRM, SRM, MRM
    instrument: Optional[str] = None  # Q-Exactive, Orbitrap Fusion, etc.
    vendor: Optional[str] = None  # Thermo, Bruker, Sciex, Waters

    # Sample preparation
    digestion_enzyme: Optional[str] = None  # Trypsin, LysC, Chymotrypsin
    digestion_time: Optional[str] = None  # e.g., "overnight", "4 hours"
    reduction_agent: Optional[str] = None  # DTT, TCEP
    alkylation_agent: Optional[str] = None  # IAA, CAA

    # Fractionation
    fractionation: Optional[str] = None  # SCX, bRP, HpH, none
    fraction_count: Optional[int] = None

    # Enrichment (for PTMs)
    enrichment: Optional[str] = None  # Phospho, Glyco, IMAC, TiO2
    ptm_type: Optional[str] = None  # Phosphorylation, Ubiquitination, etc.

    # MS parameters
    resolution: Optional[int] = None  # e.g., 70000, 120000
    ms1_scan_range: Optional[str] = None  # e.g., "350-1600 m/z"
    ms2_scan_range: Optional[str] = None
    isolation_window: Optional[float] = None  # e.g., 1.6 m/z
    collision_energy: Optional[str] = None  # e.g., "NCE 27%", "HCD 30%"

    # LC parameters
    lc_gradient_time: Optional[int] = None  # minutes
    column_type: Optional[str] = None  # e.g., "C18 reversed-phase"
    column_length: Optional[str] = None  # e.g., "25 cm"

    # Database search
    search_engine: Optional[str] = None  # MaxQuant, Proteome Discoverer, Mascot
    search_engine_version: Optional[str] = None
    fdr_threshold: Optional[float] = None  # e.g., 0.01 (1%)
    database: Optional[str] = None  # UniProt, RefSeq, custom
    database_version: Optional[str] = None

    # Quantification
    quantification_method: Optional[str] = None  # LFQ, TMT, SILAC, iTRAQ
    normalization: Optional[str] = None  # Median, quantile, etc.
