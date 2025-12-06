"""
RNA-seq protocol details dataclass for transcriptomics studies.

This module defines the RNASeqProtocolDetails dataclass containing all
fields specific to RNA sequencing-based transcriptomics protocols.
"""

from dataclasses import dataclass
from typing import Optional

from lobster.services.metadata.protocol_extraction.base import BaseProtocolDetails


@dataclass
class RNASeqProtocolDetails(BaseProtocolDetails):
    """
    Extracted protocol details from RNA-seq transcriptomics publication text.

    Extends BaseProtocolDetails with RNA-seq-specific fields for library
    preparation, sequencing parameters, and analysis pipeline.

    Examples:
        >>> details = RNASeqProtocolDetails()
        >>> details.library_prep_kit = "TruSeq Stranded mRNA"
        >>> details.strand_specificity = "reverse"
        >>> details.to_dict()
        {'library_prep_kit': 'TruSeq Stranded mRNA', 'strand_specificity': 'reverse', ...}
    """

    # RNA extraction
    extraction_kit: Optional[str] = None  # RNeasy, TRIzol, etc.
    rna_integrity: Optional[float] = None  # RIN score (0-10)

    # Library preparation
    library_prep_kit: Optional[str] = None  # TruSeq, SMARTseq, NEBNext
    strand_specificity: Optional[str] = None  # forward, reverse, unstranded
    selection_method: Optional[str] = None  # polyA, ribo-depletion, total RNA
    fragmentation: Optional[str] = None  # enzymatic, chemical, sonication
    fragment_size: Optional[int] = None  # target fragment size in bp

    # Sequencing parameters
    platform: Optional[str] = None  # Illumina, PacBio, Nanopore
    instrument: Optional[str] = None  # NovaSeq 6000, HiSeq 4000, etc.
    read_length: Optional[int] = None  # 50, 75, 100, 150 bp
    paired_end: Optional[bool] = None
    sequencing_depth: Optional[str] = None  # e.g., "30M reads/sample"

    # Analysis - alignment
    aligner: Optional[str] = None  # STAR, HISAT2, kallisto, Salmon
    aligner_version: Optional[str] = None
    reference_genome: Optional[str] = None  # GRCh38, mm10, etc.
    genome_version: Optional[str] = None

    # Analysis - annotation
    annotation: Optional[str] = None  # GENCODE, Ensembl, RefSeq
    annotation_version: Optional[str] = None

    # Analysis - quantification
    quantification: Optional[str] = None  # featureCounts, HTSeq, Salmon
    quantification_level: Optional[str] = None  # gene, transcript, exon

    # Analysis - differential expression
    de_tool: Optional[str] = None  # DESeq2, edgeR, limma
    de_tool_version: Optional[str] = None
    normalization: Optional[str] = None  # TPM, FPKM, CPM, TMM

    # Quality control
    qc_tool: Optional[str] = None  # FastQC, MultiQC, RSeQC
    trimming_tool: Optional[str] = None  # Trimmomatic, cutadapt, fastp
