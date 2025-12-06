"""
Metagenomics schema definitions for metagenomics and microbiome data.

This module defines the expected structure and metadata for metagenomics
datasets including 16S/ITS amplicon and shotgun sequencing with appropriate
validation rules for sample-level and feature-level metadata standardization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.validation import FlexibleValidator

# =============================================================================
# BIOLOGICAL METADATA FIELDS (FREE-TEXT, NOT ONTOLOGY-BASED)
# =============================================================================
# The following fields are stored as free-text strings in obs metadata:
#
# - organism      → Free-text organism name (e.g., "human gut metagenome", "mouse gut metagenome")
# - host          → Host organism (e.g., "Homo sapiens", "Mus musculus")
# - host_species  → Synonym for host (microbiome-specific)
# - body_site     → Body site (e.g., "gut", "skin", "oral")
# - tissue        → Specific tissue (e.g., "colon", "ileum", "rectum")
# - isolation_source → Sample source (e.g., "fecal", "gut tissue", "biopsy")
# - disease       → Standardized disease term (e.g., "crc", "uc", "cd", "healthy")
# - age           → Numeric age value (e.g., 45, 62)
# - sex           → Standardized sex (e.g., "male", "female", "unknown")
# - sample_type   → Sample classification (e.g., "fecal", "tissue", "biopsy")
#
# NOTE: Future enhancement will migrate to ontology-based standardization
# (NCBI Taxonomy, UBERON) via embedding service.
# See: kevin_notes/sragent_embedding_ontology_plan.md
# =============================================================================


class MetagenomicsSchema:
    """
    Schema definitions for metagenomics data modalities.

    This class provides schema definitions for metagenomics data across
    16S/ITS amplicon and shotgun sequencing with appropriate metadata
    requirements and validation rules.
    """

    @staticmethod
    def get_16s_amplicon_schema() -> Dict[str, Any]:
        """
        Get schema for 16S/ITS amplicon sequencing (taxonomic profiling).

        Returns:
            Dict[str, Any]: 16S/ITS amplicon schema definition
        """
        return {
            "modality": "16s_amplicon",
            "description": "16S/ITS amplicon sequencing (taxonomic profiling)",
            # obs: Observations (samples) metadata - DataFrame with samples as rows
            # Contains per-sample metadata including experimental conditions, sequencing metrics,
            # alpha diversity, and quality control measurements
            #
            # Example obs DataFrame:
            #            sample_id subject_id timepoint condition sample_type sequencing_platform amplicon_region sequencing_depth  n_features shannon_diversity observed_features
            # Sample_1    Sample_1 Subject_001      Day0   Healthy         Gut      Illumina MiSeq          16S V4            50000         234              4.2               189
            # Sample_2    Sample_2 Subject_001     Week4   Healthy         Gut      Illumina MiSeq          16S V4            48000         221              4.1               175
            # Sample_3    Sample_3 Subject_002      Day0   Disease         Gut      Illumina MiSeq          16S V4            52000         156              2.8               124
            # Sample_4    Sample_4 Subject_002     Week4   Disease         Gut      Illumina MiSeq          16S V4            51000         142              2.5               108
            "obs": {
                "required": [],  # Columns that must be present - flexible for diverse experimental designs
                "optional": [  # Standard sample metadata for metagenomics experiments
                    # Core identifiers
                    "sample_id",  # Unique sample identifier
                    "subject_id",  # Subject/patient ID
                    "timepoint",  # Timepoint
                    # Experimental design
                    "condition",  # Healthy, Disease, Treatment
                    "batch",  # Sequencing batch
                    "replicate",  # Biological replicate
                    # Biological metadata (free-text, restored v1.2.0)
                    "organism",  # Organism name (e.g., "human gut metagenome")
                    "host",  # Host organism (e.g., "Homo sapiens", "Mus musculus")
                    "host_species",  # Synonym for host (microbiome-specific)
                    "body_site",  # Body site (e.g., "gut", "skin", "oral cavity")
                    "tissue",  # Specific tissue (e.g., "colon", "ileum", "feces")
                    "isolation_source",  # Sample source (e.g., "fecal", "biopsy")
                    "disease",  # Disease status (e.g., "crc", "uc", "cd", "healthy")
                    "age",  # Subject age (numeric)
                    "sex",  # Subject sex (male/female/unknown)
                    "sample_type",  # Sample classification (fecal/tissue/biopsy)
                    "environment",  # For non-host samples (soil, water, built)
                    # Sequencing metadata
                    "sequencing_platform",  # Illumina MiSeq, PacBio, etc.
                    "library_strategy",  # AMPLICON
                    "library_layout",  # single-end, paired-end
                    "amplicon_region",  # 16S V4, ITS2, etc.
                    "target_gene",  # 16S, ITS, 18S, COI
                    "forward_primer_name",  # 515F
                    "reverse_primer_name",  # 806R
                    "forward_primer_seq",  # GTGCCAGCMGCCGCGGTAA
                    "reverse_primer_seq",  # GGACTACHVGGGTWTCTAAT
                    # Sequencing quality
                    "sequencing_depth",  # Total reads
                    "read_length",  # Average read length (bp)
                    "total_reads",  # Total raw reads
                    "percent_assigned",  # % reads assigned to features
                    # Feature detection
                    "n_features",  # Number of OTUs/ASVs detected
                    "total_counts",  # Total feature counts
                    # Alpha diversity metrics (calculated)
                    "shannon_diversity",  # Shannon entropy
                    "simpson_diversity",  # Simpson index
                    "observed_features",  # Observed richness
                    "chao1",  # Chao1 estimate
                    "pielou_evenness",  # Pielou's evenness
                    "faith_pd",  # Faith's phylogenetic diversity
                    # Environmental parameters (for environmental samples)
                    "ph",  # pH measurement
                    "temperature",  # Temperature (°C)
                    "salinity",  # Salinity (ppt)
                    "depth",  # Depth (m)
                    "elevation",  # Elevation (m)
                    # Geographic metadata
                    "collection_date",  # ISO 8601 date
                    "geographic_location",  # Country: Region, City
                    "latitude",  # Decimal degrees
                    "longitude",  # Decimal degrees
                    # Database accessions
                    "biosample_accession",  # SAMN identifier
                    "sra_accession",  # SRS identifier
                    "sra_run_accession",  # SRR identifier
                ],
                "types": {
                    "sample_id": "string",
                    "condition": "categorical",
                    # Biological metadata types (restored v1.2.0)
                    "organism": "string",
                    "host": "string",
                    "host_species": "string",
                    "body_site": "categorical",
                    "tissue": "string",
                    "isolation_source": "string",
                    "disease": "categorical",
                    "age": "numeric",
                    "sex": "categorical",
                    "sample_type": "categorical",
                    "sequencing_depth": "numeric",
                    "shannon_diversity": "numeric",
                    "simpson_diversity": "numeric",
                    "observed_features": "numeric",
                    "chao1": "numeric",
                    "pielou_evenness": "numeric",
                    "faith_pd": "numeric",
                    "n_features": "numeric",
                    "total_counts": "numeric",
                    "percent_assigned": "numeric",
                    "ph": "numeric",
                    "temperature": "numeric",
                    "salinity": "numeric",
                    "depth": "numeric",
                    "elevation": "numeric",
                    "latitude": "numeric",
                    "longitude": "numeric",
                },
            },
            # var: Variables (features) metadata - DataFrame with OTUs/ASVs as rows
            # Contains per-feature taxonomic annotations, sequences, and quality metrics
            #
            # Example var DataFrame:
            #                 feature_id                            sequence sequence_length Kingdom      Phylum          Class           Order         Family        Genus          Species taxonomy_confidence  n_samples  prevalence is_chimera
            # ASV_001            ASV_001  TACGTAGGTGGCAAGCGTTGTCCGGA...             252 Bacteria Firmicutes     Clostridia  Clostridiales Ruminococcaceae Ruminococcus Ruminococcus gnavus              0.95          4        1.00      False
            # ASV_002            ASV_002  TACGTAGGGGGCAAGCGTTATCCGGA...             251 Bacteria Bacteroidetes Bacteroidia Bacteroidales  Bacteroidaceae Bacteroides  Bacteroides fragilis              0.98          3        0.75      False
            # ASV_003            ASV_003  TACGGAGGGTGCAAGCGTTAATCGGA...             253 Bacteria Proteobacteria Gammaproteobacteria Enterobacterales Enterobacteriaceae Escherichia  Escherichia coli              0.99          2        0.50      False
            # ASV_004            ASV_004  TACGTAGGTGGCAAGCGTTGTCCGGA...             250 Bacteria Firmicutes     Bacilli  Lactobacillales Lactobacillaceae Lactobacillus  Lactobacillus acidophilus              0.92          3        0.75      False
            "var": {
                "required": [],  # Columns that must be present - flexible for different annotation levels
                "optional": [  # Standard feature metadata columns
                    # Primary identifiers
                    "feature_id",  # OTU_001, ASV_001, hash
                    "sequence",  # Representative DNA sequence
                    "sequence_length",  # Length in bp
                    "sequence_hash",  # MD5 hash
                    # Taxonomic hierarchy (7 levels)
                    "Kingdom",  # Bacteria, Archaea, Eukaryota, Unassigned
                    "Phylum",  # Firmicutes, Bacteroidetes, Proteobacteria
                    "Class",  # Bacilli, Clostridia, Bacteroidia
                    "Order",  # Lactobacillales, Clostridiales
                    "Family",  # Lactobacillaceae, Clostridiaceae
                    "Genus",  # Lactobacillus, Clostridium
                    "Species",  # Lactobacillus plantarum
                    # Taxonomic metadata
                    "taxonomy",  # Full string "k__Bacteria;p__Firmicutes;..."
                    "taxonomy_confidence",  # Confidence score (0-1)
                    "taxonomy_method",  # sklearn, blast, vsearch
                    "taxonomy_database",  # greengenes2, silva138, unite
                    # Feature-level statistics
                    "n_samples",  # Number of samples with detection
                    "total_counts",  # Total counts across samples
                    "mean_counts",  # Mean counts
                    "prevalence",  # Proportion of samples
                    "mean_abundance",  # Mean relative abundance
                    "max_abundance",  # Maximum relative abundance
                    # Quality flags
                    "is_chimera",  # Chimera detection flag
                    "is_contaminant",  # Contaminant flag
                    "is_singleton",  # Only in 1 sample
                    "is_doubleton",  # Only in 2 samples
                    # Phylogenetic metadata
                    "phylo_node_id",  # Node ID in tree
                    "phylo_branch_length",  # Branch length
                ],
                "types": {
                    "feature_id": "string",
                    "sequence": "string",
                    "sequence_length": "numeric",
                    "sequence_hash": "string",
                    "Kingdom": "categorical",
                    "Phylum": "categorical",
                    "Class": "categorical",
                    "Order": "categorical",
                    "Family": "categorical",
                    "Genus": "categorical",
                    "Species": "categorical",
                    "taxonomy": "string",
                    "taxonomy_confidence": "numeric",
                    "taxonomy_method": "string",
                    "taxonomy_database": "string",
                    "n_samples": "numeric",
                    "total_counts": "numeric",
                    "mean_counts": "numeric",
                    "prevalence": "numeric",
                    "mean_abundance": "numeric",
                    "max_abundance": "numeric",
                    "is_chimera": "boolean",
                    "is_contaminant": "boolean",
                    "is_singleton": "boolean",
                    "is_doubleton": "boolean",
                    "phylo_node_id": "string",
                    "phylo_branch_length": "numeric",
                },
            },
            # layers: Alternative count matrices with same dimensions as X
            # Store different transformations/versions of the count data
            # Each layer is a 2D matrix: samples x features, same shape as adata.X
            #
            # Example layers (4 samples x 4 ASVs):
            #
            # layers['counts'] (raw OTU/ASV counts):
            #          ASV_001  ASV_002  ASV_003  ASV_004
            # Sample_1    1245     856      234       89
            # Sample_2    1189     923      198       76
            # Sample_3     512     645      512      123
            # Sample_4     489     701      489      145
            #
            # layers['relative_abundance'] (proportions sum to 1 per sample):
            #          ASV_001  ASV_002  ASV_003  ASV_004
            # Sample_1   0.512    0.352    0.096    0.037
            # Sample_2   0.497    0.386    0.083    0.032
            # Sample_3   0.287    0.361    0.287    0.069
            # Sample_4   0.268    0.384    0.268    0.079
            "layers": {
                "required": [],  # No layers are strictly required (main data stored in adata.X)
                "optional": [  # Common data transformations for metagenomics
                    "counts",  # Raw OTU/ASV counts
                    "relative_abundance",  # Proportions (sum to 1 per sample)
                    "css_normalized",  # Cumulative sum scaling
                    "tss_normalized",  # Total sum scaling
                    "rarefied",  # Rarefied to even depth
                    "log_counts",  # log(counts + 1)
                    "clr",  # Centered log-ratio (for ANCOM)
                    "ilr",  # Isometric log-ratio
                    "presence_absence",  # Binary 0/1 matrix
                    "imputed",  # Imputed zeros
                ],
            },
            # obsm: Observations (samples) multidimensional annotations
            # Stores per-sample multidimensional data like embeddings or distance matrices
            # Each entry is a 2D array: samples x dimensions
            #
            # Example obsm matrices:
            #
            # obsm['X_pcoa'] (PCoA coordinates - 4 samples x 2 axes):
            #          PCo1   PCo2
            # Sample_1 -8.5    3.2
            # Sample_2 -7.8    2.9
            # Sample_3  9.2   -3.5
            # Sample_4  8.1   -3.1
            #
            # obsm['distances_bray_curtis'] (Bray-Curtis distance matrix - 4x4):
            #          Sample_1  Sample_2  Sample_3  Sample_4
            # Sample_1     0.00      0.12      0.68      0.71
            # Sample_2     0.12      0.00      0.65      0.69
            # Sample_3     0.68      0.65      0.00      0.15
            # Sample_4     0.71      0.69      0.15      0.00
            "obsm": {
                "required": [],  # No embeddings are required (generated during analysis)
                "optional": [  # Common dimensionality reduction and distance matrices
                    # Dimensionality reduction
                    "X_pcoa",  # Principal coordinate analysis
                    "X_nmds",  # Non-metric MDS
                    "X_tsne",  # t-SNE
                    "X_umap",  # UMAP
                    "X_pca",  # PCA (on CLR data)
                    # Beta diversity distance matrices
                    "distances_bray_curtis",  # Bray-Curtis dissimilarity
                    "distances_jaccard",  # Jaccard distance
                    "distances_unweighted_unifrac",  # Unweighted UniFrac
                    "distances_weighted_unifrac",  # Weighted UniFrac
                    "distances_aitchison",  # Aitchison (CLR + Euclidean)
                ],
            },
            # uns: Unstructured annotations - global metadata and analysis parameters
            # Stores dataset-level information, analysis parameters, and complex results
            # Contains nested dictionaries, arrays, or objects that don't fit obs/var structure
            #
            # Example uns structure:
            # uns = {
            #     'phylogenetic_tree': '(ASV_001:0.05,ASV_002:0.08,(ASV_003:0.03,ASV_004:0.04):0.10);',  # Newick format
            #     'taxonomy_database': 'greengenes2-2022.10',
            #     'taxonomy_method': 'sklearn',
            #     'denoising_method': 'dada2',
            #     'bioproject': 'PRJNA123456',
            #     'sra_study': 'SRP123456',
            #     'differential_abundance': {...},  # DESeq2/ANCOM results
            #     'pathway_analysis': {...},        # PICRUSt2 results
            #     'provenance': {...}               # W3C-PROV tracking
            # }
            "uns": {
                "required": [],  # No global metadata is strictly required
                "optional": [  # Common analysis metadata and computational results
                    # Phylogenetic tree (CRITICAL for UniFrac)
                    "phylogenetic_tree",  # Newick format string
                    "phylogenetic_tree_method",  # fasttree, iqtree, raxml
                    # Taxonomy database
                    "taxonomy_database",  # greengenes2-2022.10, silva-138
                    "taxonomy_database_version",  # Version number
                    "taxonomy_method",  # sklearn, blast, vsearch
                    # Feature processing (QIIME2 workflow)
                    "denoising_method",  # dada2, deblur, unoise3
                    "denoising_params",  # Dict of parameters
                    "chimera_detection",  # uchime_denovo, uchime_ref
                    "filtering_params",  # Min count, min prevalence
                    # Sequencing metadata
                    "sequencing_technology",  # Illumina MiSeq, PacBio Sequel II
                    "amplicon_target",  # 16S V4, ITS2
                    "forward_primer",  # 515F: GTGCCAGCMGCCGCGGTAA
                    "reverse_primer",  # 806R: GGACTACHVGGGTWTCTAAT
                    "target_region",  # V4 (for 16S)
                    # Study metadata
                    "title",  # Study title
                    "abstract",  # Study abstract
                    "publication_doi",  # DOI
                    # Database accessions
                    "bioproject",  # PRJNA123456
                    "biosample_accessions",  # List of SAMN IDs
                    "sra_study",  # SRP123456
                    "geo_accession",  # GSE123456 (if applicable)
                    "mgrast_id",  # MG-RAST ID
                    "qiita_study_id",  # Qiita study ID
                    # Geographic metadata
                    "collection_date_range",  # Date range
                    "geographic_location",  # Location summary
                    "latitude_range",  # [min, max]
                    "longitude_range",  # [min, max]
                    # Analysis results
                    "differential_abundance",  # DESeq2/ANCOM/ALDEx2 results
                    "pathway_analysis",  # PICRUSt2/HUMAnN3 results
                    "statistical_tests",  # PERMANOVA, ANOSIM results
                    "alpha_diversity_stats",  # Kruskal-Wallis, etc.
                    "beta_diversity_stats",  # PERMANOVA, PERMDISP results
                    # Provenance
                    "provenance",  # W3C-PROV tracking
                    "processing_date",  # ISO 8601 timestamp
                    "lobster_version",  # Lobster version
                    # Ontology mappings (embedding service results)
                    "ontology_mappings",  # Host organism/body site ontology IDs
                ],
            },
        }

    @staticmethod
    def get_shotgun_schema() -> Dict[str, Any]:
        """
        Get schema for shotgun metagenomic sequencing (functional profiling).

        Returns:
            Dict[str, Any]: Shotgun metagenomics schema definition
        """
        return {
            "modality": "shotgun_metagenomics",
            "description": "Shotgun metagenomic sequencing (functional profiling)",
            # obs: Same as amplicon, but library_strategy is "WGS" not "AMPLICON"
            "obs": {
                "required": [],
                "optional": [
                    # Same fields as 16S amplicon, except:
                    "sample_id",
                    "subject_id",
                    "timepoint",
                    "condition",
                    "batch",
                    "replicate",
                    "sample_type",
                    "environment",
                    # Biological metadata (restored v1.2.0 - same as 16S)
                    "organism",  # Homo sapiens, Mus musculus
                    "host",  # For microbiome studies
                    "host_species",  # Homo sapiens (for host-associated microbiomes)
                    "body_site",  # Gut, Skin, Oral cavity
                    "tissue",  # Colon, Duodenum, Fecal
                    "isolation_source",  # Fecal, Soil, Water
                    "disease",  # CRC, IBD, Healthy control
                    "age",  # Age in years
                    "age_unit",  # years, months, days
                    "sex",  # Male, Female, Unknown
                    # Technical metadata
                    "sequencing_platform",
                    "library_strategy",  # WGS (not AMPLICON)
                    "library_layout",
                    # No amplicon_region, target_gene, primers (not applicable)
                    "sequencing_depth",
                    "read_length",
                    "total_reads",
                    "percent_assigned",
                    "n_features",
                    "total_counts",
                    # Diversity metrics
                    "shannon_diversity",
                    "simpson_diversity",
                    "observed_features",
                    "chao1",
                    "pielou_evenness",
                    # Environmental metadata
                    "ph",
                    "temperature",
                    "salinity",
                    "depth",
                    "elevation",
                    # Temporal/spatial metadata
                    "collection_date",
                    "geographic_location",
                    "latitude",
                    "longitude",
                    # Database accessions
                    "biosample_accession",
                    "sra_accession",
                    "sra_run_accession",
                ],
                "types": {
                    "sample_id": "string",
                    "condition": "categorical",
                    # Biological metadata types (restored v1.2.0)
                    "organism": "string",
                    "host": "string",
                    "host_species": "string",
                    "body_site": "categorical",
                    "tissue": "string",
                    "isolation_source": "string",
                    "disease": "categorical",
                    "age": "numeric",
                    "sex": "categorical",
                    "sample_type": "categorical",
                    # Technical metrics
                    "sequencing_depth": "numeric",
                    "shannon_diversity": "numeric",
                    "simpson_diversity": "numeric",
                    "observed_features": "numeric",
                },
            },
            # var: Gene/pathway features (NOT OTUs)
            "var": {
                "required": [],
                "optional": [
                    # Gene identifiers
                    "feature_id",  # Gene_001, KO_001
                    "gene_name",  # dnaA, recA, etc.
                    "gene_description",  # Chromosomal replication initiator
                    # Functional annotations
                    "kegg_ko",  # KEGG Orthology (K00001)
                    "kegg_pathway",  # KEGG pathway (ko00010)
                    "kegg_module",  # KEGG module
                    "cog_category",  # COG category (L, J, K, etc.)
                    "cog_id",  # COG0001
                    "pfam_id",  # Pfam domain (PF00001)
                    "go_terms",  # Gene Ontology terms (list)
                    "ec_number",  # Enzyme Commission number
                    # Taxonomic assignment (gene-level)
                    "Kingdom",
                    "Phylum",
                    "Class",
                    "Order",
                    "Family",
                    "Genus",
                    "Species",
                    # Gene statistics
                    "n_samples",  # Samples with detection
                    "mean_abundance",  # Mean gene abundance (RPK)
                    "prevalence",  # Proportion of samples
                ],
                "types": {
                    "feature_id": "string",
                    "gene_name": "string",
                    "kegg_ko": "string",
                    "cog_category": "categorical",
                    "mean_abundance": "numeric",
                    "prevalence": "numeric",
                },
            },
            # layers: Different from amplicon (no phylogenetic data)
            "layers": {
                "required": [],
                "optional": [
                    "counts",  # Raw gene counts
                    "rpk",  # Reads per kilobase
                    "rpkm",  # RPKM normalized
                    "tpm",  # Transcripts per million
                    "relative_abundance",  # Proportions
                    "clr",  # CLR transformation
                ],
            },
            # obsm: Same dimensionality reduction, but NO phylogenetic distances
            "obsm": {
                "required": [],
                "optional": [
                    "X_pcoa",
                    "X_nmds",
                    "X_tsne",
                    "X_umap",
                    "X_pca",
                    "distances_bray_curtis",
                    "distances_jaccard",
                    "distances_aitchison",
                    # NO UniFrac (not applicable for genes)
                ],
            },
            # uns: Different from amplicon
            "uns": {
                "required": [],
                "optional": [
                    # NO phylogenetic_tree (not applicable for genes)
                    "functional_database",  # KEGG, COG, Pfam version
                    "taxonomic_profiler",  # MetaPhlAn, Kraken2, etc.
                    "assembly_method",  # megahit, metaSPAdes, etc.
                    "gene_caller",  # Prodigal, MetaGeneMark
                    # Study metadata (same as amplicon)
                    "title",
                    "abstract",
                    "publication_doi",
                    # Database accessions
                    "bioproject",
                    "biosample_accessions",
                    "sra_study",
                    "geo_accession",
                    # Analysis results
                    "differential_abundance",
                    "pathway_analysis",
                    "statistical_tests",
                    "alpha_diversity_stats",
                    "beta_diversity_stats",
                    # Provenance
                    "provenance",
                    "processing_date",
                    "lobster_version",
                    # Ontology mappings
                    "ontology_mappings",
                ],
            },
        }

    @staticmethod
    def create_validator(
        schema_type: str = "16s_amplicon",
        strict: bool = False,
        ignore_warnings: Optional[List[str]] = None,
    ) -> FlexibleValidator:
        """
        Create a validator for metagenomics data.

        Args:
            schema_type: Type of schema ('16s_amplicon' or 'shotgun')
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator

        Raises:
            ValueError: If schema_type is not recognized
        """
        if schema_type == "16s_amplicon":
            schema = MetagenomicsSchema.get_16s_amplicon_schema()
        elif schema_type == "shotgun":
            schema = MetagenomicsSchema.get_shotgun_schema()
        else:
            raise ValueError(
                f"Unknown schema type: {schema_type}. Must be '16s_amplicon' or 'shotgun'"
            )

        ignore_set = set(ignore_warnings) if ignore_warnings else set()

        # Add default ignored warnings for metagenomics
        ignore_set.update(
            [
                "Unexpected obs columns",
                "Unexpected var columns",
                "missing values",  # Less common than metabolomics but still possible
                "Very sparse data",
            ]
        )

        validator = FlexibleValidator(
            schema=schema,
            name=f"MetagenomicsValidator_{schema_type}",
            ignore_warnings=ignore_set,
        )

        # Add metagenomics-specific validation rules
        # Add cross-database accession validation (all schema types)
        validator.add_custom_rule(
            "check_cross_database_accessions",
            lambda adata: _validate_cross_database_accessions(
                adata, modality="metagenomics"
            ),
        )

        if schema_type == "16s_amplicon":
            validator.add_custom_rule("check_taxonomy_hierarchy", _validate_taxonomy)
            validator.add_custom_rule("check_phylogenetic_tree", _validate_tree)
            validator.add_custom_rule("check_sequence_quality", _validate_sequences)
        elif schema_type == "shotgun":
            validator.add_custom_rule(
                "check_functional_annotations", _validate_annotations
            )
            validator.add_custom_rule("check_gene_abundance", _validate_gene_abundance)

        return validator

    @staticmethod
    def get_recommended_qc_thresholds(
        schema_type: str = "16s_amplicon",
    ) -> Dict[str, Any]:
        """
        Get recommended quality control thresholds for metagenomics.

        Args:
            schema_type: Type of schema ('16s_amplicon' or 'shotgun')

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        if schema_type == "16s_amplicon":
            return {
                "min_reads_per_sample": 5000,
                "max_reads_per_sample": 200000,
                "min_features_per_sample": 50,
                "min_samples_per_feature": 2,
                "max_singleton_pct": 20.0,  # % features in only 1 sample
                "min_taxonomy_confidence": 0.8,
                "max_chimera_pct": 5.0,  # % chimeric sequences
                "min_sequence_length": 200,  # bp
                "max_sequence_length": 600,  # bp
                "min_shannon_diversity": 1.0,
            }
        elif schema_type == "shotgun":
            return {
                "min_reads_per_sample": 1000000,
                "min_genes_per_sample": 100000,
                "min_samples_per_gene": 2,
                "min_functional_annotation_pct": 50.0,  # % genes with KEGG/COG
                "min_taxonomic_assignment_pct": 70.0,  # % genes with taxonomy
            }
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")


def _validate_taxonomy(adata) -> "ValidationResult":
    """Validate taxonomic hierarchy consistency and completeness."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for taxonomic columns in var
    tax_levels = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    present_levels = [level for level in tax_levels if level in adata.var.columns]

    if not present_levels:
        result.add_warning(
            "No taxonomic hierarchy found. Consider adding Kingdom, Phylum, Class, "
            "Order, Family, Genus, Species columns for proper taxonomic annotation."
        )
        return result

    # Check for missing taxonomic assignments
    for level in present_levels:
        missing = adata.var[level].isna().sum()
        if missing > 0:
            missing_pct = (missing / len(adata.var)) * 100
            if missing_pct > 20:
                result.add_warning(
                    f"{missing} features ({missing_pct:.1f}%) missing {level} assignment. "
                    "Consider reviewing taxonomy assignment parameters."
                )

    # Check for "Unassigned" or "Unknown" entries
    for level in present_levels:
        if adata.var[level].dtype == "object":
            unassigned = (
                adata.var[level]
                .str.contains("unassigned|unknown", case=False, na=False)
                .sum()
            )
            if unassigned > 0:
                unassigned_pct = (unassigned / len(adata.var)) * 100
                if unassigned_pct > 30:
                    result.add_warning(
                        f"{unassigned} features ({unassigned_pct:.1f}%) have 'Unassigned' "
                        f"{level}. This is high and may indicate poor taxonomy resolution."
                    )

    # Check taxonomy confidence if available
    if "taxonomy_confidence" in adata.var.columns:
        low_confidence = (adata.var["taxonomy_confidence"] < 0.8).sum()
        if low_confidence > 0:
            low_conf_pct = (low_confidence / len(adata.var)) * 100
            if low_conf_pct > 20:
                result.add_warning(
                    f"{low_confidence} features ({low_conf_pct:.1f}%) have low taxonomy "
                    "confidence (<0.8). Consider filtering or using higher-confidence assignments."
                )

    return result


def _validate_tree(adata) -> "ValidationResult":
    """Validate phylogenetic tree format and feature matching."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check if phylogenetic tree exists
    if "phylogenetic_tree" not in adata.uns:
        result.add_info(
            "No phylogenetic tree found in uns['phylogenetic_tree']. "
            "Phylogenetic diversity metrics (UniFrac, Faith's PD) will not be available."
        )
        return result

    tree_str = adata.uns["phylogenetic_tree"]

    # Basic Newick format validation
    if not isinstance(tree_str, str):
        result.add_warning("Phylogenetic tree is not a string. Expected Newick format.")
        return result

    if not tree_str.strip().endswith(";"):
        result.add_warning("Phylogenetic tree does not end with ';' (Newick format).")

    # Check for basic Newick structure
    if "(" not in tree_str or ")" not in tree_str:
        result.add_warning(
            "Phylogenetic tree does not contain parentheses (Newick format)."
        )

    # Check if tree contains feature IDs from var.index
    # (Simple check - full validation would require parsing the tree)
    if hasattr(adata.var, "index"):
        sample_ids = list(adata.var.index[:5])  # Check first 5 feature IDs
        found_ids = sum(1 for fid in sample_ids if str(fid) in tree_str)
        if found_ids == 0:
            result.add_warning(
                "Feature IDs from var.index not found in phylogenetic tree. "
                "Tree may not match the feature table."
            )

    return result


def _validate_sequences(adata) -> "ValidationResult":
    """Validate sequence data quality and format."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check if sequences are present
    if "sequence" not in adata.var.columns:
        result.add_info(
            "No sequences found in var['sequence']. "
            "Representative sequences are recommended for reproducibility."
        )
        return result

    sequences = adata.var["sequence"]

    # Check for missing sequences
    missing = sequences.isna().sum()
    if missing > 0:
        result.add_warning(f"{missing} features missing sequences.")

    # Check sequence lengths
    if "sequence_length" in adata.var.columns:
        seq_lengths = adata.var["sequence_length"]
        short_seqs = (seq_lengths < 200).sum()
        long_seqs = (seq_lengths > 600).sum()

        if short_seqs > 0:
            result.add_warning(
                f"{short_seqs} sequences are unusually short (<200 bp). "
                "May indicate quality issues."
            )
        if long_seqs > 0:
            result.add_warning(
                f"{long_seqs} sequences are unusually long (>600 bp). "
                "May indicate chimeras or concatenated sequences."
            )

    # Check for chimeras if flag is present
    if "is_chimera" in adata.var.columns:
        chimeras = adata.var["is_chimera"].sum()
        if chimeras > 0:
            chimera_pct = (chimeras / len(adata.var)) * 100
            if chimera_pct > 5:
                result.add_warning(
                    f"{chimeras} chimeric sequences detected ({chimera_pct:.1f}%). "
                    "Consider removing chimeras with filter_chimeras()."
                )

    return result


def _validate_annotations(adata) -> "ValidationResult":
    """Validate functional annotations for shotgun metagenomics."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for functional annotation columns
    annotation_cols = ["kegg_ko", "kegg_pathway", "cog_id", "pfam_id", "ec_number"]
    present_cols = [col for col in annotation_cols if col in adata.var.columns]

    if not present_cols:
        result.add_warning(
            "No functional annotations found (KEGG, COG, Pfam, EC). "
            "Consider annotating genes with functional databases."
        )
        return result

    # Check annotation completeness
    for col in present_cols:
        missing = adata.var[col].isna().sum()
        if missing > 0:
            missing_pct = (missing / len(adata.var)) * 100
            if missing_pct > 50:
                result.add_warning(
                    f"{missing} genes ({missing_pct:.1f}%) missing {col} annotation. "
                    "Low annotation rate may limit pathway analysis."
                )

    return result


def _validate_gene_abundance(adata) -> "ValidationResult":
    """Validate gene abundance data characteristics."""

    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for negative values
    if hasattr(adata.X, "min"):
        min_val = adata.X.min()
        if min_val < 0:
            result.add_warning(
                f"Negative values found in abundance matrix (min: {min_val}). "
                "This is unusual for shotgun metagenomics data."
            )

    # Check for sparsity
    if hasattr(adata.X, "data"):  # Sparse matrix
        zero_pct = (adata.X.data == 0).sum() / adata.X.data.size * 100
    else:  # Dense matrix
        zero_pct = (adata.X == 0).sum() / adata.X.size * 100

    if zero_pct > 80:
        result.add_warning(
            f"{zero_pct:.1f}% of gene abundance values are zero. "
            "High sparsity may indicate insufficient sequencing depth or aggressive filtering."
        )

    return result


def _validate_cross_database_accessions(
    adata, modality: str = "metagenomics"
) -> "ValidationResult":
    """
    Validate cross-database accession format and structure.

    Checks database accession fields in adata.uns against expected formats
    using the database_mappings registry.

    Args:
        adata: AnnData object to validate
        modality: Data modality (transcriptomics, proteomics, metabolomics, metagenomics)

    Returns:
        ValidationResult: Validation results with accession format errors/warnings
    """
    from lobster.core.interfaces.validator import ValidationResult
    from lobster.core.schemas.database_mappings import (
        get_accession_url,
        get_accessions_for_modality,
        validate_accession,
    )

    result = ValidationResult()

    # Get expected accessions for this modality
    expected_accessions = get_accessions_for_modality(modality)

    # Check each accession field in uns
    for field_name, accession_spec in expected_accessions.items():
        if field_name in adata.uns:
            value = adata.uns[field_name]

            # Skip empty/None values
            if value is None or (isinstance(value, str) and not value.strip()):
                continue

            # Validate accession format
            if not validate_accession(field_name, value):
                result.add_warning(
                    f"Invalid {accession_spec.database_name} accession format: '{value}' "
                    f"(expected pattern: {accession_spec.example})"
                )
            else:
                # Successful validation - add info with URL
                url = get_accession_url(field_name, value)
                if url:
                    result.add_info(
                        f"Valid {accession_spec.database_name} accession: {value} ({url})"
                    )

    return result


# =============================================================================
# Pydantic Metadata Schema for Sample-Level Metadata Standardization
# =============================================================================
# This schema is used by the metadata_assistant agent for cross-dataset
# metadata harmonization, standardization, and validation.
# Phase 3 addition for metagenomics metadata operations.
# =============================================================================


class MetagenomicsMetadataSchema(BaseModel):
    """
    Pydantic schema for metagenomics sample-level metadata standardization.

    This schema defines the expected structure for sample metadata across
    metagenomics and microbiome experiments. It enforces controlled vocabularies
    and data types for consistent metadata representation across datasets.

    Used by metadata_assistant agent for:
    - Cross-dataset sample ID mapping
    - Metadata standardization and harmonization
    - Dataset completeness validation
    - Multi-omics integration preparation

    NOTE: organism, host_species, body_site, and tissue fields have been removed.
    These are now handled by the embedding-based ontology matching service.
    See module header for details.

    Attributes:
        sample_id: Unique sample identifier (required)
        subject_id: Subject/patient identifier for biological replicates
        timepoint: Timepoint or developmental stage
        condition: Experimental condition (e.g., "Healthy", "Disease", "Treatment")
        sample_type: Type of biological sample (e.g., "Gut", "Soil", "Water")
        sequencing_platform: Sequencing platform (e.g., "Illumina MiSeq", "PacBio")
        amplicon_region: Amplicon region (e.g., "16S V4", "ITS", "18S")
        sequencing_depth: Number of sequencing reads
        read_length: Sequencing read length in base pairs
        primer_set: Primer set used for amplification
        batch: Batch identifier for technical replicates
        additional_metadata: Flexible dict for custom fields
    """

    # Required fields
    sample_id: str = Field(..., description="Unique sample identifier", min_length=1)

    # Optional core fields
    subject_id: Optional[str] = Field(None, description="Subject/patient identifier")
    timepoint: Optional[str] = Field(
        None, description="Timepoint or developmental stage"
    )
    condition: str = Field(
        ..., description="Experimental condition (e.g., Healthy, Disease, Treatment)"
    )
    sample_type: str = Field(
        ...,
        description="Type of biological sample (Gut, Soil, Water, Skin, etc.)",
    )

    # NOTE: organism, host_species, body_site, tissue fields removed
    # See module header for details on embedding service integration

    sequencing_platform: str = Field(
        ..., description="Sequencing platform (e.g., Illumina MiSeq, PacBio)"
    )
    amplicon_region: str = Field(
        ..., description="Amplicon region (e.g., 16S V4, ITS, 18S)"
    )

    # Metagenomics-specific fields
    sequencing_depth: Optional[int] = Field(
        None, description="Number of sequencing reads", gt=0
    )
    read_length: Optional[int] = Field(
        None, description="Sequencing read length in base pairs", gt=0
    )
    primer_set: Optional[str] = Field(
        None, description="Primer set used for amplification"
    )
    batch: Optional[str] = Field(None, description="Batch identifier")

    # Flexible additional metadata
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional custom metadata fields"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "sample_id": "Sample_A_Rep1",
                "subject_id": "Subject_001",
                "timepoint": "Day0",
                "condition": "Healthy",
                "sample_type": "Gut",
                # organism removed - handled by embedding service
                "sequencing_platform": "Illumina MiSeq",
                "amplicon_region": "16S V4",
                "sequencing_depth": 50000,
                "read_length": 300,
                "primer_set": "515F-806R",
                "batch": "Batch1",
                "additional_metadata": {
                    "replicate": "Rep1",
                    "collection_method": "Swab",
                    "storage_condition": "-80C",
                },
            }
        }

    @field_validator("sequencing_platform")
    @classmethod
    def validate_sequencing_platform(cls, v: str) -> str:
        """Validate sequencing platform is a known platform."""
        allowed = {
            # Illumina
            "illumina_miseq",
            "illumina_hiseq",
            "illumina_nextseq",
            "illumina_novaseq",
            # PacBio
            "pacbio",
            "pacbio_sequel",
            "pacbio_revio",
            # Oxford Nanopore
            "ont",
            "minion",
            "gridion",
            "promethion",
            # Ion Torrent
            "ion_torrent",
            "ion_pgm",
            "ion_proton",
            # Other
            "454",
            "sanger",
        }
        v_lower = v.lower().replace(" ", "_").replace("-", "_")
        if v_lower not in allowed:
            # Allow unknown platforms
            return v
        # Return normalized format
        return v

    @field_validator("amplicon_region")
    @classmethod
    def validate_amplicon_region(cls, v: str) -> str:
        """Validate amplicon region format."""
        # Common amplicon regions for validation
        common_regions = {
            "16s",
            "16s_v1",
            "16s_v2",
            "16s_v3",
            "16s_v4",
            "16s_v5",
            "16s_v6",
            "16s_v7",
            "16s_v8",
            "16s_v9",
            "16s_v1v2",
            "16s_v3v4",
            "16s_v4v5",
            "its",
            "its1",
            "its2",
            "18s",
            "23s",
            "coi",
            "wgs",  # Whole genome shotgun
        }
        v_lower = v.lower().replace(" ", "_").replace("-", "_")
        if v_lower not in common_regions:
            # Allow unknown regions but return original format
            return v
        # Return uppercase for consistency
        return v.upper()

    @field_validator("sample_type")
    @classmethod
    def validate_sample_type(cls, v: str) -> str:
        """Ensure sample_type is not empty."""
        if not v or not v.strip():
            raise ValueError("sample_type cannot be empty")
        return v.strip()

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v: str) -> str:
        """Ensure condition is not empty."""
        if not v or not v.strip():
            raise ValueError("condition cannot be empty")
        return v.strip()

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, v: str) -> str:
        """Ensure sample_id is not empty and has no leading/trailing whitespace."""
        if not v or not v.strip():
            raise ValueError("sample_id cannot be empty")
        return v.strip()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary with all fields including additional_metadata
        """
        base_dict = self.model_dump(exclude={"additional_metadata"}, exclude_none=True)
        if self.additional_metadata:
            base_dict.update(self.additional_metadata)
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetagenomicsMetadataSchema":
        """
        Create schema from dictionary, automatically handling unknown fields.

        Args:
            data: Dictionary with metadata fields

        Returns:
            MetagenomicsMetadataSchema: Validated schema instance
        """
        # Extract known fields
        known_fields = set(cls.model_fields.keys()) - {"additional_metadata"}
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # Put remaining fields in additional_metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)
