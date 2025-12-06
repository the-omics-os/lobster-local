"""
Transcriptomics schema definitions for single-cell and bulk RNA-seq data.

This module defines the expected structure and metadata for transcriptomics
datasets including single-cell RNA-seq and bulk RNA-seq with appropriate
validation rules.
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
# - organism      → Free-text organism name (e.g., "Homo sapiens", "Mus musculus")
# - tissue        → Free-text tissue name (e.g., "brain", "colon", "blood")
# - cell_type     → Free-text cell type (e.g., "T cell", "B cell", "monocyte")
# - disease       → Standardized disease term (e.g., "crc", "uc", "cd", "healthy")
# - age           → Numeric age value (e.g., 45, 62)
# - sex           → Standardized sex (e.g., "male", "female", "unknown")
# - sample_type   → Sample classification (e.g., "fecal", "tissue", "blood")
#
# NOTE: Future enhancement will migrate to ontology-based standardization
# (NCBI Taxonomy, UBERON, Cell Ontology) via embedding service.
# See: kevin_notes/sragent_embedding_ontology_plan.md
# =============================================================================


class TranscriptomicsSchema:
    """
    Schema definitions for transcriptomics data modalities.

    This class provides schema definitions for both single-cell and bulk
    RNA-seq data with appropriate metadata requirements and validation rules.
    """

    @staticmethod
    def get_single_cell_schema() -> Dict[str, Any]:
        """
        Get schema for single-cell RNA-seq data.

        Returns:
            Dict[str, Any]: Single-cell RNA-seq schema definition
        """
        return {
            "modality": "single_cell_rna_seq",
            "description": "Single-cell RNA sequencing data schema",
            # obs: Observations (cells/samples) metadata - DataFrame with cells as rows
            # Contains per-cell metadata including experimental conditions, quality metrics,
            # cell annotations, and computational results (clustering, cell cycle, etc.)
            #
            # Example obs DataFrame:
            #                cell_id sample_id   batch condition cell_type  n_genes  total_counts  pct_counts_mt leiden phase
            # CELL_001       CELL_001  Sample_A  Batch1   Control    T_cell     2156          8934           12.4      0    G1
            # CELL_002       CELL_002  Sample_A  Batch1   Control    B_cell     1834          6721            8.9      1     S
            # CELL_003       CELL_003  Sample_B  Batch2 Treatment    T_cell     2401         11203           15.2      0   G2M
            # CELL_004       CELL_004  Sample_B  Batch2 Treatment  Monocyte     1923          7456           11.1      2    G1
            # CELL_005       CELL_005  Sample_B  Batch2 Treatment    T_cell     2089          9012            9.8      0     S
            "obs": {
                "required": [],  # Columns that must be present - flexible to accommodate diverse datasets
                "optional": [  # Commonly expected columns that may be present depending on analysis stage
                    "cell_id",  # Unique cell identifier
                    "sample_id",  # Sample origin
                    "batch",  # Batch identifier
                    "condition",  # Experimental condition
                    # Biological metadata (free-text, restored v1.2.0)
                    "organism",  # Organism name (e.g., "Homo sapiens")
                    "tissue",  # Tissue type (e.g., "brain", "PBMC")
                    "cell_type",  # Cell type annotation (e.g., "T cell")
                    "disease",  # Disease status (e.g., "crc", "healthy")
                    "age",  # Subject age (numeric)
                    "sex",  # Subject sex (male/female/unknown)
                    "sample_type",  # Sample classification (tissue/blood/etc.)
                    "n_genes",  # Number of genes detected
                    "n_genes_by_counts",  # Number of genes with counts > 0
                    "total_counts",  # Total UMI counts
                    "total_counts_mt",  # Total mitochondrial counts
                    "pct_counts_mt",  # Mitochondrial percentage
                    "pct_counts_ribo",  # Ribosomal percentage
                    "doublet_score",  # Doublet detection score
                    "is_doublet",  # Doublet classification
                    "leiden",  # Leiden clustering
                    "louvain",  # Louvain clustering
                    "phase",  # Cell cycle phase
                    "G1_score",  # G1 phase score
                    "S_score",  # S phase score
                    "G2M_score",  # G2/M phase score
                    # Sequencing quality metrics
                    "total_reads",  # Total sequencing reads
                    "percent_aligned",  # Alignment rate (%)
                    "percent_duplicates",  # PCR duplicate rate (%)
                    "median_insert_size",  # Fragment size (bp)
                    "fastqc_url",  # Link to FastQC report
                ],
                "types": {
                    "cell_id": "string",
                    "sample_id": "string",
                    "batch": "string",
                    "condition": "string",
                    # Biological metadata types (restored v1.2.0)
                    "organism": "string",
                    "tissue": "string",
                    "cell_type": "string",
                    "disease": "categorical",
                    "age": "numeric",
                    "sex": "categorical",
                    "sample_type": "categorical",
                    "n_genes": "numeric",
                    "n_genes_by_counts": "numeric",
                    "total_counts": "numeric",
                    "total_counts_mt": "numeric",
                    "pct_counts_mt": "numeric",
                    "pct_counts_ribo": "numeric",
                    "doublet_score": "numeric",
                    "is_doublet": "boolean",
                    "leiden": "categorical",
                    "louvain": "categorical",
                    "phase": "categorical",
                    "total_reads": "numeric",
                    "percent_aligned": "numeric",
                    "percent_duplicates": "numeric",
                    "median_insert_size": "numeric",
                    "fastqc_url": "string",
                },
            },
            # var: Variables (genes/features) metadata - DataFrame with genes as rows
            # Contains per-gene metadata including gene identifiers, genomic coordinates,
            # gene annotations, and computational metrics (expression stats, variability)
            #
            # Example var DataFrame:
            #                      gene_id gene_symbol                              gene_name chromosome         biotype  n_cells  mean_counts highly_variable    mt  ribo
            # ENSG00000139618  ENSG00000139618       BRCA2           BRCA2 DNA repair associated         13  protein_coding        4         15.2            True False False
            # ENSG00000141510  ENSG00000141510        TP53                    tumor protein p53         17  protein_coding        5         23.8            True False False
            # ENSG00000155657  ENSG00000155657         TTN                                titin          2  protein_coding        2          4.1           False False False
            # ENSG00000186092  ENSG00000186092       OR4F5  olfactory receptor family 4 subfamily F          1  protein_coding        1          1.0           False False False
            "var": {
                "required": [],  # Columns that must be present - flexible for different gene annotation levels
                "optional": [  # Standard gene metadata columns that enhance interpretability
                    "gene_id",  # Ensembl ID (primary key)
                    "gene_symbol",  # HGNC symbol
                    "gene_name",  # Full gene name
                    "chromosome",  # Genomic location
                    "start",  # Gene start position
                    "end",  # Gene end position
                    "strand",  # Genomic strand
                    "biotype",  # Gene biotype
                    "n_cells",  # Number of cells expressing
                    "n_cells_by_counts",  # Number of cells with counts > 0
                    "mean_counts",  # Mean expression across cells
                    "pct_dropout_by_counts",  # Dropout percentage
                    "total_counts",  # Total counts across all cells
                    "highly_variable",  # Highly variable gene flag
                    "means",  # Mean expression (scanpy)
                    "dispersions",  # Dispersion values
                    "dispersions_norm",  # Normalized dispersions
                    "mt",  # Mitochondrial gene flag
                    "ribo",  # Ribosomal gene flag
                ],
                "types": {  # Expected data types for validation and processing
                    "gene_id": "string",
                    "gene_symbol": "string",
                    "gene_name": "string",
                    "chromosome": "string",
                    "start": "numeric",
                    "end": "numeric",
                    "strand": "string",
                    "biotype": "categorical",
                    "n_cells": "numeric",
                    "n_cells_by_counts": "numeric",
                    "mean_counts": "numeric",
                    "pct_dropout_by_counts": "numeric",
                    "total_counts": "numeric",
                    "highly_variable": "boolean",
                    "means": "numeric",
                    "dispersions": "numeric",
                    "dispersions_norm": "numeric",
                    "mt": "boolean",
                    "ribo": "boolean",
                },
            },
            # layers: Alternative expression matrices with same dimensions as X
            # Store different transformations/versions of the count data (raw, normalized, scaled)
            # Each layer is a 2D matrix: cells x genes, same shape as adata.X
            #
            # Example layers (5 cells x 4 genes):
            #
            # layers['counts'] (raw UMI counts):
            #           BRCA2  TP53  TTN  OR4F5
            # CELL_001     12    28    0      0
            # CELL_002      8    31    5      0
            # CELL_003     18    42    2      1
            # CELL_004     16     0    8      0
            # CELL_005     22    18    1      0
            #
            # layers['normalized'] (library-size normalized):
            #           BRCA2  TP53  TTN  OR4F5
            # CELL_001   1.34  3.13  0.0   0.0
            # CELL_002   1.19  4.61  0.74  0.0
            # CELL_003   1.61  3.75  0.18  0.09
            # CELL_004   2.15  0.0   1.07  0.0
            # CELL_005   2.44  2.0   0.11  0.0
            "layers": {
                "required": [],  # No layers are strictly required (main data stored in adata.X)
                "optional": [  # Common data transformations stored as separate matrices
                    "counts",  # Raw count data
                    "normalized",  # Normalized counts
                    "log1p",  # Log-transformed normalized counts
                    "scaled",  # Z-scored expression values
                    "imputed",  # Imputed values
                ],
            },
            # obsm: Observations (cells) multidimensional annotations - matrices/arrays per cell
            # Stores per-cell multidimensional data like embeddings, coordinates, or feature sets
            # Each entry is a 2D array: cells x dimensions (e.g., cells x PC components)
            #
            # Example obsm matrices:
            #
            # obsm['X_pca'] (PCA coordinates - 5 cells x 3 PCs):
            #           PC1   PC2   PC3
            # CELL_001 -2.1   1.4   0.3
            # CELL_002  1.8  -0.9  -1.2
            # CELL_003 -0.5   2.1   0.8
            # CELL_004  2.3   0.2  -0.4
            # CELL_005 -1.5  -2.8   0.5
            #
            # obsm['X_umap'] (UMAP coordinates - 5 cells x 2 dimensions):
            #          UMAP1  UMAP2
            # CELL_001   4.2   -1.8
            # CELL_002  -3.1    2.4
            # CELL_003   1.9    3.7
            # CELL_004  -2.6   -0.9
            # CELL_005   0.6    1.2
            "obsm": {
                "required": [],  # No embeddings are required (generated during analysis)
                "optional": [  # Common dimensionality reduction and embedding results
                    "X_pca",  # PCA coordinates
                    "X_umap",  # UMAP embedding
                    "X_tsne",  # t-SNE embedding
                    "X_diffmap",  # Diffusion map
                    "X_draw_graph_fa",  # Force-directed layout
                ],
            },
            # uns: Unstructured annotations - global metadata and analysis parameters
            # Stores dataset-level information, analysis parameters, and complex results
            # Contains nested dictionaries, arrays, or objects that don't fit obs/var structure
            #
            # Example uns structure:
            # uns = {
            #     'hvg': {  # Highly variable genes analysis results
            #         'means': [15.2, 23.8, 4.1, 1.0],
            #         'dispersions': [2.1, 3.4, 0.8, 0.2],
            #         'highly_variable_genes': ['BRCA2', 'TP53']
            #     },
            #     'pca': {  # PCA analysis parameters and results
            #         'params': {'n_comps': 50, 'svd_solver': 'arpack'},
            #         'variance_ratio': [0.12, 0.08, 0.06, ...]
            #     },
            #     'neighbors': {  # Nearest neighbor graph info
            #         'params': {'n_neighbors': 15, 'method': 'umap'},
            #         'connectivities_key': 'connectivities',
            #         'distances_key': 'distances'
            #     },
            #     'leiden': {  # Leiden clustering parameters
            #         'params': {'resolution': 0.5, 'n_iterations': -1}
            #     }
            # }
            "uns": {
                "required": [],  # No global metadata is strictly required
                "optional": [  # Common analysis metadata and computational results
                    # Analysis-related metadata
                    "log1p",  # Log transformation parameters
                    "hvg",  # Highly variable genes info
                    "pca",  # PCA parameters and results
                    "neighbors",  # Nearest neighbors graph
                    "umap",  # UMAP parameters
                    "tsne",  # t-SNE parameters
                    "leiden",  # Leiden clustering parameters
                    "louvain",  # Louvain clustering parameters
                    "rank_genes_groups",  # Differential expression results
                    # Dataset provenance and metadata
                    "provenance",  # General provenance tracking
                    # GEO metadata fields (matching _extract_metadata output exactly)
                    "contact_address",  # Contact address from GEO
                    "contact_city",  # Contact city from GEO
                    "contact_country",  # Contact country from GEO
                    "contact_department",  # Contact department
                    "contact_email",  # Contact email address
                    "contact_institute",  # Contact institution from GEO
                    "contact_name",  # Contact name from GEO
                    "contact_phone",  # Contact phone number
                    "contact_zip/postal_code",  # Contact ZIP/postal code from GEO (exact field name)
                    "contributor",  # List of contributors from GEO
                    "geo_accession",  # GEO accession number (e.g., GSE194247)
                    "last_update_date",  # Last update date from GEO
                    "overall_design",  # Overall experimental design from GEO
                    "platform_id",  # Platform ID from GEO (e.g., GPL24676)
                    "platform_taxid",  # Platform taxonomy ID from GEO
                    "platforms",  # Platforms dictionary from GEO
                    "pubmed_id",  # PubMed ID from GEO
                    "relation",  # Relation field from GEO (BioProject links, etc.)
                    "sample_id",  # List of sample IDs from GEO
                    "sample_taxid",  # Sample taxonomy ID from GEO
                    "samples",  # Complete samples dictionary from GEO
                    "status",  # Publication status from GEO
                    "submission_date",  # Submission date from GEO
                    "summary",  # Dataset summary from GEO
                    "supplementary_file",  # Main supplementary file URL from GEO
                    "title",  # Dataset title from GEO
                    "type",  # Dataset type from GEO (e.g., "Expression profiling by high throughput sequencing")
                    "web_link",  # Web link or URL related to the dataset
                    # Sample-level metadata fields (from samples dictionary)
                    "channel_count",  # Number of channels per sample
                    "characteristics_ch1",  # Sample characteristics from GEO
                    "data_processing",  # Data processing steps from sample metadata
                    "data_row_count",  # Number of data rows per sample
                    "extract_protocol_ch1",  # RNA extraction protocol from sample metadata
                    "growth_protocol_ch1",  # Growth/culture protocol from sample metadata
                    "instrument_model",  # Sequencing instrument model from sample metadata
                    "library_selection",  # Library selection method from sample metadata
                    "library_source",  # Library source type from sample metadata
                    "library_strategy",  # Library strategy from sample metadata (RNA-Seq, etc.)
                    "molecule_ch1",  # Molecule type from sample metadata (polyA RNA, etc.)
                    "organism_ch1",  # Organism from sample metadata
                    "series_id",  # Series ID from sample metadata
                    "source_name_ch1",  # Source name from sample metadata
                    "supplementary_file_1",  # Sample-level supplementary files
                    "taxid_ch1",  # Taxonomy ID from sample metadata
                    "treatment_protocol_ch1",  # Treatment protocol from sample metadata
                    # Additional processing and quality information
                    "processing_info",  # Data processing pipeline info
                    "quality_metrics",  # Dataset-wide quality metrics
                    "data_source",  # Source of the data (GEO, SRA, etc.)
                    # Cross-database accessions
                    "bioproject_accession",  # NCBI BioProject (PRJNA123456)
                    "biosample_accession",  # NCBI BioSample (SAMN12345678)
                    "sra_study_accession",  # SRA Study (SRP123456)
                    "sra_experiment_accession",  # SRA Experiment (SRX123456)
                    "sra_run_accession",  # SRA Run (SRR123456)
                    "publication_doi",  # Publication DOI (10.1038/nature12345)
                    "arrayexpress_accession",  # ArrayExpress (E-MTAB-12345, optional)
                    # Ontology mappings (embedding service results)
                    "ontology_mappings",  # Organism/tissue/cell_type ontology IDs
                ],
            },
        }

    @staticmethod
    def get_bulk_rna_seq_schema() -> Dict[str, Any]:
        """
        Get schema for bulk RNA-seq data.

        Returns:
            Dict[str, Any]: Bulk RNA-seq schema definition
        """
        return {
            "modality": "bulk_rna_seq",
            "description": "Bulk RNA sequencing data schema",
            # obs: Observations (samples) metadata - DataFrame with samples as rows
            # Contains per-sample metadata including experimental design, technical metrics,
            # and quality control measurements specific to bulk RNA-seq experiments
            #
            # Example obs DataFrame:
            #            sample_id condition treatment   batch replicate   tissue organism  n_genes  total_counts  rna_integrity  mapping_rate
            # Sample_1    Sample_1   Control      None  Batch1      Rep1    Brain    Human    15234      45000000            8.2          0.85
            # Sample_2    Sample_2   Control      None  Batch1      Rep2    Brain    Human    14891      42000000            7.9          0.83
            # Sample_3    Sample_3 Treatment     Drug1  Batch2      Rep1    Brain    Human    15678      48000000            8.1          0.87
            # Sample_4    Sample_4 Treatment     Drug1  Batch2      Rep2    Brain    Human    15123      46000000            7.8          0.84
            "obs": {
                "required": [],  # Columns that must be present - flexible for diverse experimental designs
                "optional": [  # Standard sample metadata for bulk RNA-seq experiments
                    "sample_id",  # Unique sample identifier
                    "condition",  # Experimental condition
                    "treatment",  # Treatment information
                    "batch",  # Sequencing batch
                    "replicate",  # Biological replicate
                    # Biological metadata (free-text, restored v1.2.0)
                    "organism",  # Organism name (e.g., "Homo sapiens", "Mus musculus")
                    "tissue",  # Tissue type (e.g., "brain", "liver", "blood")
                    "disease",  # Disease status (e.g., "cancer", "healthy")
                    "age",  # Subject age (numeric)
                    "sex",  # Subject sex (male/female/unknown)
                    "sample_type",  # Sample classification (tissue/blood/cell_line)
                    "n_genes",  # Number of genes detected
                    "total_counts",  # Total read counts
                    "library_size",  # Library size
                    "rna_integrity",  # RNA integrity number (RIN)
                    "mapping_rate",  # Alignment rate
                    "duplication_rate",  # PCR duplication rate
                ],
                "types": {  # Expected data types for validation and processing
                    "sample_id": "string",
                    "condition": "categorical",
                    "treatment": "categorical",
                    "batch": "string",
                    "replicate": "string",
                    # Biological metadata types (restored v1.2.0)
                    "organism": "string",
                    "tissue": "string",
                    "disease": "categorical",
                    "age": "numeric",
                    "sex": "categorical",
                    "sample_type": "categorical",
                    "n_genes": "numeric",
                    "total_counts": "numeric",
                    "library_size": "numeric",
                    "rna_integrity": "numeric",
                    "mapping_rate": "numeric",
                    "duplication_rate": "numeric",
                },
            },
            # var: Variables (genes/features) metadata - DataFrame with genes as rows
            # Contains per-gene metadata including gene identifiers, genomic coordinates,
            # gene annotations, and computational metrics (expression stats, variability)
            #
            # Example var DataFrame:
            #                      gene_id gene_symbol                       gene_name chromosome  length  gc_content  n_samples  mean_counts highly_variable    mt  ribo
            # ENSG00000139618  ENSG00000139618       BRCA2    BRCA2 DNA repair associated         13   84193        0.42          4       1250.5            True False False
            # ENSG00000141510  ENSG00000141510        TP53             tumor protein p53         17   19149        0.36          4       2180.3            True False False
            # ENSG00000155657  ENSG00000155657         TTN                         titin          2  109224        0.44          2        450.2           False False False
            # ENSG00000186092  ENSG00000186092       OR4F5  olfactory receptor family 4          1    2559        0.40          1         45.0           False False False
            "var": {
                "required": [],  # Columns that must be present - flexible for different gene annotation levels
                "optional": [  # Standard gene metadata columns that enhance interpretability
                    "gene_id",  # Ensembl ID
                    "gene_symbol",  # HGNC symbol
                    "gene_name",  # Full gene name
                    "chromosome",  # Genomic location
                    "start",  # Gene start position
                    "end",  # Gene end position
                    "strand",  # Genomic strand
                    "biotype",  # Gene biotype
                    "length",  # Gene/transcript length
                    "gc_content",  # GC content
                    "n_samples",  # Number of samples expressing
                    "mean_counts",  # Mean expression across samples
                    "total_counts",  # Total counts across samples
                    "highly_variable",  # Highly variable gene flag
                    "mt",  # Mitochondrial gene flag
                    "ribo",  # Ribosomal gene flag
                ],
                "types": {  # Expected data types for validation and processing
                    "gene_id": "string",
                    "gene_symbol": "string",
                    "gene_name": "string",
                    "chromosome": "string",
                    "start": "numeric",
                    "end": "numeric",
                    "strand": "string",
                    "biotype": "categorical",
                    "length": "numeric",
                    "gc_content": "numeric",
                    "n_samples": "numeric",
                    "mean_counts": "numeric",
                    "total_counts": "numeric",
                    "highly_variable": "boolean",
                    "mt": "boolean",
                    "ribo": "boolean",
                },
            },
            # layers: Alternative expression matrices with same dimensions as X
            # Store different transformations/versions of the count data (raw, normalized, scaled)
            # Each layer is a 2D matrix: samples x genes, same shape as adata.X
            #
            # Example layers (4 samples x 4 genes):
            #
            # layers['counts'] (raw read counts):
            #           BRCA2  TP53   TTN  OR4F5
            # Sample_1   1245  2156   423     45
            # Sample_2   1189  2089   398     42
            # Sample_3   1312  2298   512     48
            # Sample_4   1256  2178   467     46
            #
            # layers['tpm'] (Transcripts Per Million):
            #           BRCA2   TP53    TTN  OR4F5
            # Sample_1   34.2   67.8   8.9   12.1
            # Sample_2   32.8   65.2   8.2   11.8
            # Sample_3   36.1   71.2   9.8   13.2
            # Sample_4   33.9   68.4   8.7   12.5
            #
            # layers['normalized'] (DESeq2 normalized):
            #           BRCA2   TP53    TTN  OR4F5
            # Sample_1  125.4  217.3  42.6   4.5
            # Sample_2  118.2  207.8  39.6   4.2
            # Sample_3  132.1  231.5  51.5   4.8
            # Sample_4  126.7  219.4  47.1   4.6
            "layers": {
                "required": [],  # No layers are strictly required (main data stored in adata.X)
                "optional": [  # Common data transformations for bulk RNA-seq analysis
                    "counts",  # Raw count data
                    "tpm",  # Transcripts per million
                    "fpkm",  # Fragments per kilobase per million
                    "rpkm",  # Reads per kilobase per million
                    "normalized",  # Normalized counts (DESeq2, etc.)
                    "log_normalized",  # Log-normalized counts
                    "vst",  # Variance stabilized transformation
                    "rlog",  # Regularized log transformation
                ],
            },
            # obsm: Observations (samples) multidimensional annotations - matrices/arrays per sample
            # Stores per-sample multidimensional data like embeddings, coordinates, or feature sets
            # Each entry is a 2D array: samples x dimensions (e.g., samples x PC components)
            #
            # Example obsm matrices:
            #
            # obsm['X_pca'] (PCA coordinates - 4 samples x 3 PCs):
            #           PC1    PC2    PC3
            # Sample_1 -12.5   8.3   2.1
            # Sample_2 -11.8   7.9   1.8
            # Sample_3  13.2  -8.7  -2.3
            # Sample_4  12.8  -8.1  -2.0
            #
            # obsm['X_umap'] (UMAP coordinates - 4 samples x 2 dimensions):
            #          UMAP1  UMAP2
            # Sample_1  -4.2    3.1
            # Sample_2  -3.8    2.9
            # Sample_3   4.1   -3.3
            # Sample_4   3.9   -3.1
            "obsm": {
                "required": [],  # No embeddings are required (generated during analysis)
                "optional": [  # Common dimensionality reduction results for sample visualization
                    "X_pca",  # PCA coordinates
                    "X_tsne",  # t-SNE embedding
                    "X_umap",  # UMAP embedding
                ],
            },
            # uns: Unstructured annotations - global metadata and analysis parameters
            # Stores dataset-level information, analysis parameters, and complex results
            # Contains nested dictionaries, arrays, or objects that don't fit obs/var structure
            #
            # Example uns structure:
            # uns = {
            #     'differential_expression': {  # DESeq2 results
            #         'Treatment_vs_Control': {
            #             'log2FoldChange': {'BRCA2': 0.23, 'TP53': 1.45, 'TTN': -0.12},
            #             'padj': {'BRCA2': 0.15, 'TP53': 0.001, 'TTN': 0.89},
            #             'significant_genes': ['TP53']
            #         }
            #     },
            #     'pathway_analysis': {  # GSEA results
            #         'KEGG_PATHWAYS': {
            #             'p53_signaling': {'pvalue': 0.002, 'genes': ['TP53', 'BRCA2']},
            #             'DNA_repair': {'pvalue': 0.01, 'genes': ['BRCA2']}
            #         }
            #     },
            #     'pca': {  # PCA analysis info
            #         'variance_ratio': [0.45, 0.23, 0.12, 0.08],
            #         'params': {'n_comps': 50}
            #     }
            # }
            "uns": {
                "required": [],  # No global metadata is strictly required
                "optional": [  # Common analysis metadata and computational results for bulk RNA-seq
                    "log1p",  # Log transformation parameters
                    "hvg",  # Highly variable genes info
                    "pca",  # PCA parameters and results
                    "differential_expression",  # DE analysis results
                    "pathway_analysis",  # Pathway enrichment results
                    "provenance",  # Provenance tracking
                    # Contact and web metadata fields
                    "contact_address",  # Contact address
                    "contact_city",  # Contact city
                    "contact_country",  # Contact country
                    "contact_department",  # Contact department
                    "contact_email",  # Contact email address
                    "contact_institute",  # Contact institution
                    "contact_name",  # Contact name
                    "contact_phone",  # Contact phone number
                    "contact_zip/postal_code",  # Contact ZIP/postal code
                    "web_link",  # Web link or URL related to the dataset
                    # Cross-database accessions
                    "bioproject_accession",  # NCBI BioProject (PRJNA123456)
                    "biosample_accession",  # NCBI BioSample (SAMN12345678)
                    "sra_study_accession",  # SRA Study (SRP123456)
                    "sra_experiment_accession",  # SRA Experiment (SRX123456)
                    "sra_run_accession",  # SRA Run (SRR123456)
                    "publication_doi",  # Publication DOI (10.1038/nature12345)
                    "arrayexpress_accession",  # ArrayExpress (E-MTAB-12345, optional)
                    # Ontology mappings (embedding service results)
                    "ontology_mappings",  # Organism/tissue/cell_type ontology IDs
                ],
            },
        }

    @staticmethod
    def get_pseudobulk_schema() -> Dict[str, Any]:
        """
        Get schema for pseudobulk aggregated single-cell data.

        Pseudobulk is a derived representation of single-cell RNA-seq data where
        cells are aggregated by sample and cell type for differential expression
        analysis using bulk RNA-seq methods.

        Returns:
            Dict[str, Any]: Pseudobulk RNA-seq schema definition
        """
        return {
            "modality": "pseudobulk_rna_seq",
            "description": "Pseudobulk aggregated single-cell RNA-seq data schema",
            # obs: Observations (sample × cell type combinations) metadata - DataFrame with pseudobulk samples as rows
            # Each observation represents aggregated counts for a specific sample and cell type combination
            # Contains metadata about the original samples, cell types, and aggregation statistics
            #
            # Example obs DataFrame:
            #                        sample_id cell_type condition   batch treatment  n_cells_aggregated  total_counts_aggregated  aggregation_method
            # Sample_A_T_cell         Sample_A    T_cell   Control  Batch1      None                 245                   450000                 sum
            # Sample_A_B_cell         Sample_A    B_cell   Control  Batch1      None                  89                   180000                 sum
            # Sample_A_Monocyte       Sample_A  Monocyte   Control  Batch1      None                 156                   320000                 sum
            # Sample_B_T_cell         Sample_B    T_cell Treatment  Batch2     Drug1                 198                   380000                 sum
            # Sample_B_B_cell         Sample_B    B_cell Treatment  Batch2     Drug1                  72                   145000                 sum
            # Sample_B_Monocyte       Sample_B  Monocyte Treatment  Batch2     Drug1                 134                   275000                 sum
            "obs": {
                "required": [
                    "sample_id",  # Original sample identifier
                    "cell_type",  # Cell type used for aggregation
                    "n_cells_aggregated",  # Number of cells aggregated for this pseudobulk sample
                ],
                "optional": [
                    "condition",  # Experimental condition
                    "treatment",  # Treatment information
                    "batch",  # Sequencing/experimental batch
                    "replicate",  # Biological replicate identifier
                    # NOTE: organism and tissue removed - handled by embedding service
                    "donor_id",  # Individual donor/subject ID
                    "age",  # Age of donor/subject
                    "sex",  # Sex of donor/subject
                    "disease_status",  # Disease/health status
                    "time_point",  # Time point in longitudinal studies
                    "total_counts_aggregated",  # Total UMI counts aggregated
                    "mean_counts_per_cell",  # Mean counts per cell in aggregation
                    "aggregation_method",  # Method used for aggregation (sum, mean, median)
                    "min_cells_threshold",  # Minimum cells threshold used
                    "aggregation_date",  # When aggregation was performed
                    "original_n_obs",  # Number of cells in original dataset
                    "pseudobulk_sample_id",  # Composite identifier for pseudobulk sample
                ],
                "types": {
                    "sample_id": "string",
                    "cell_type": "categorical",
                    "condition": "categorical",
                    "treatment": "categorical",
                    "batch": "string",
                    "replicate": "string",
                    "donor_id": "string",
                    "age": "numeric",
                    "sex": "categorical",
                    "disease_status": "categorical",
                    "time_point": "string",
                    "n_cells_aggregated": "numeric",
                    "total_counts_aggregated": "numeric",
                    "mean_counts_per_cell": "numeric",
                    "aggregation_method": "categorical",
                    "min_cells_threshold": "numeric",
                    "aggregation_date": "string",
                    "original_n_obs": "numeric",
                    "pseudobulk_sample_id": "string",
                },
            },
            # var: Variables (genes/features) metadata - DataFrame with genes as rows
            # Contains per-gene metadata similar to single-cell data, but with aggregation-specific metrics
            # Includes gene identifiers, annotations, and statistics about expression across pseudobulk samples
            #
            # Example var DataFrame:
            #                      gene_id gene_symbol                       gene_name chromosome  n_pseudobulk_samples  mean_aggregated_counts total_aggregated_counts    mt  ribo
            # ENSG00000139618  ENSG00000139618       BRCA2    BRCA2 DNA repair associated         13                     6                  12450.5                   74703 False False
            # ENSG00000141510  ENSG00000141510        TP53             tumor protein p53         17                     6                  21803.2                  130819 False False
            # ENSG00000155657  ENSG00000155657         TTN                         titin          2                     4                   4502.1                   18008 False False
            # ENSG00000186092  ENSG00000186092       OR4F5  olfactory receptor family 4          1                     2                    450.0                     900 False False
            "var": {
                "required": [],  # No gene metadata is strictly required
                "optional": [
                    "gene_id",  # Ensembl ID (primary identifier)
                    "gene_symbol",  # HGNC symbol
                    "gene_name",  # Full gene name
                    "chromosome",  # Genomic location
                    "start",  # Gene start position
                    "end",  # Gene end position
                    "strand",  # Genomic strand
                    "biotype",  # Gene biotype
                    "length",  # Gene/transcript length
                    "gc_content",  # GC content
                    "n_pseudobulk_samples",  # Number of pseudobulk samples expressing
                    "mean_aggregated_counts",  # Mean aggregated counts across samples
                    "total_aggregated_counts",  # Total aggregated counts across samples
                    "aggregation_efficiency",  # Fraction of original cells contributing
                    "highly_variable",  # Highly variable gene flag from original data
                    "mt",  # Mitochondrial gene flag
                    "ribo",  # Ribosomal gene flag
                    "original_n_cells_expressing",  # Cells expressing in original single-cell data
                    "original_mean_expression",  # Mean expression in original single-cell data
                ],
                "types": {
                    "gene_id": "string",
                    "gene_symbol": "string",
                    "gene_name": "string",
                    "chromosome": "string",
                    "start": "numeric",
                    "end": "numeric",
                    "strand": "string",
                    "biotype": "categorical",
                    "length": "numeric",
                    "gc_content": "numeric",
                    "n_pseudobulk_samples": "numeric",
                    "mean_aggregated_counts": "numeric",
                    "total_aggregated_counts": "numeric",
                    "aggregation_efficiency": "numeric",
                    "highly_variable": "boolean",
                    "mt": "boolean",
                    "ribo": "boolean",
                    "original_n_cells_expressing": "numeric",
                    "original_mean_expression": "numeric",
                },
            },
            # layers: Alternative expression matrices with same dimensions as X
            # Store different aggregated versions or processing steps of the pseudobulk data
            # Each layer is a 2D matrix: pseudobulk_samples x genes, same shape as adata.X
            #
            # Example layers (6 pseudobulk samples x 4 genes):
            #
            # layers['raw_aggregated'] (raw aggregated counts):
            #                   BRCA2  TP53   TTN  OR4F5
            # Sample_A_T_cell    1245  2156   423     45
            # Sample_A_B_cell     890  1654   301     32
            # Sample_A_Monocyte  1456  2543   578     67
            # Sample_B_T_cell    1189  2089   398     42
            # Sample_B_B_cell     823  1542   287     29
            # Sample_B_Monocyte  1378  2401   534     61
            #
            # layers['normalized'] (DESeq2 size factor normalized):
            #                   BRCA2   TP53    TTN  OR4F5
            # Sample_A_T_cell   125.4  217.3  42.6   4.5
            # Sample_A_B_cell    96.8  179.8  32.7   3.5
            # Sample_A_Monocyte 142.1  248.2  56.4   6.5
            # Sample_B_T_cell   118.2  207.8  39.6   4.2
            # Sample_B_B_cell    89.6  167.7  31.2   3.2
            # Sample_B_Monocyte 134.5  234.3  52.1   6.0
            "layers": {
                "required": [],  # No layers are strictly required (main data stored in adata.X)
                "optional": [
                    "raw_aggregated",  # Original aggregated counts before any normalization
                    "normalized",  # Normalized counts (DESeq2, TMM, etc.)
                    "log_normalized",  # Log-transformed normalized counts
                    "vst",  # Variance stabilized transformation
                    "rlog",  # Regularized log transformation
                    "tpm",  # Transcripts per million (if gene length available)
                    "cpm",  # Counts per million
                    "scaled",  # Z-scored expression values
                ],
            },
            # obsm: Observations (pseudobulk samples) multidimensional annotations
            # Stores per-pseudobulk-sample multidimensional data like embeddings or coordinates
            # Each entry is a 2D array: pseudobulk_samples x dimensions
            #
            # Example obsm matrices:
            #
            # obsm['X_pca'] (PCA coordinates - 6 pseudobulk samples x 3 PCs):
            #                   PC1    PC2    PC3
            # Sample_A_T_cell  -8.2   4.1   1.2
            # Sample_A_B_cell  -6.8   3.7   0.9
            # Sample_A_Monocyte -9.1  4.8   1.5
            # Sample_B_T_cell   7.9  -4.3  -1.1
            # Sample_B_B_cell   6.5  -3.9  -0.8
            # Sample_B_Monocyte 8.7  -5.2  -1.4
            #
            # obsm['X_umap'] (UMAP coordinates - 6 pseudobulk samples x 2 dimensions):
            #                  UMAP1  UMAP2
            # Sample_A_T_cell   -3.2   2.1
            # Sample_A_B_cell   -2.8   1.9
            # Sample_A_Monocyte -3.5   2.4
            # Sample_B_T_cell    3.1  -2.2
            # Sample_B_B_cell    2.7  -1.8
            # Sample_B_Monocyte  3.4  -2.5
            "obsm": {
                "required": [],  # No embeddings are required
                "optional": [
                    "X_pca",  # PCA coordinates for pseudobulk samples
                    "X_tsne",  # t-SNE embedding
                    "X_umap",  # UMAP embedding
                ],
            },
            # uns: Unstructured annotations - global metadata and analysis parameters
            # Stores aggregation parameters, provenance, and analysis results
            # Contains information about the aggregation process and downstream analyses
            #
            # Example uns structure:
            # uns = {
            #     'pseudobulk_params': {  # Aggregation parameters
            #         'sample_col': 'sample_id',
            #         'celltype_col': 'cell_type',
            #         'min_cells': 10,
            #         'aggregation_method': 'sum',
            #         'filter_zeros': True,
            #         'original_n_cells': 15234,
            #         'aggregation_timestamp': '2024-01-15T10:30:00Z'
            #     },
            #     'aggregation_stats': {  # Aggregation statistics
            #         'n_samples': 2,
            #         'n_cell_types': 3,
            #         'n_pseudobulk_samples': 6,
            #         'total_cells_aggregated': 894,
            #         'cells_per_sample': {'Sample_A': 490, 'Sample_B': 404},
            #         'cells_per_celltype': {'T_cell': 443, 'B_cell': 161, 'Monocyte': 290},
            #         'samples_excluded': [],
            #         'celltypes_excluded': ['NK_cell']
            #     },
            #     'differential_expression': {  # DE analysis results
            #         'T_cell_Treatment_vs_Control': {
            #             'log2FoldChange': {'BRCA2': -0.07, 'TP53': -0.04},
            #             'padj': {'BRCA2': 0.65, 'TP53': 0.82},
            #             'method': 'DESeq2'
            #         }
            #     },
            #     'original_dataset_info': {  # Information about source dataset
            #         'dataset_id': 'geo_gse12345_annotated',
            #         'original_modality': 'single_cell_rna_seq',
            #         'n_original_cells': 15234,
            #         'n_original_genes': 20000
            #     }
            # }
            "uns": {
                "required": [
                    "pseudobulk_params",  # Parameters used for aggregation
                    "aggregation_stats",  # Statistics about the aggregation process
                ],
                "optional": [
                    "differential_expression",  # DE analysis results
                    "original_dataset_info",  # Info about source single-cell dataset
                    "quality_control",  # QC metrics and filtering info
                    "provenance",  # Provenance tracking
                    "formula_design",  # Design matrix information for DE
                    "pca",  # PCA analysis parameters and results
                    "pathway_analysis",  # Pathway enrichment results
                    "cell_type_markers",  # Cell type marker genes used
                    "batch_correction",  # Batch correction parameters if applied
                    # Cross-database accessions (inherited from original single-cell)
                    "bioproject_accession",  # NCBI BioProject (PRJNA123456)
                    "biosample_accession",  # NCBI BioSample (SAMN12345678)
                    "sra_study_accession",  # SRA Study (SRP123456)
                    "sra_experiment_accession",  # SRA Experiment (SRX123456)
                    "sra_run_accession",  # SRA Run (SRR123456)
                    "publication_doi",  # Publication DOI (10.1038/nature12345)
                    "arrayexpress_accession",  # ArrayExpress (E-MTAB-12345, optional)
                    # Ontology mappings (embedding service results)
                    "ontology_mappings",  # Organism/tissue/cell_type ontology IDs
                    # Dataset metadata (inherited from original single-cell data)
                    "contact_address",
                    "contact_city",
                    "contact_country",
                    "contact_department",
                    "contact_email",
                    "contact_institute",
                    "contact_name",
                    "contact_phone",
                    "contact_zip/postal_code",
                    "contributor",
                    "geo_accession",
                    "last_update_date",
                    "overall_design",
                    "platform_id",
                    "platform_taxid",
                    "platforms",
                    "pubmed_id",
                    "relation",
                    "sample_id",
                    "sample_taxid",
                    "samples",
                    "status",
                    "submission_date",
                    "summary",
                    "supplementary_file",
                    "title",
                    "type",
                    "web_link",
                ],
                "types": {
                    "pseudobulk_params": "dict",
                    "aggregation_stats": "dict",
                    "differential_expression": "dict",
                    "original_dataset_info": "dict",
                    "quality_control": "dict",
                    "formula_design": "dict",
                },
            },
        }

    @staticmethod
    def create_validator(
        schema_type: str = "single_cell",
        strict: bool = False,
        ignore_warnings: Optional[List[str]] = None,
    ) -> FlexibleValidator:
        """
        Create a validator for transcriptomics data.

        Args:
            schema_type: Type of schema ('single_cell', 'bulk', or 'pseudobulk')
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator

        Raises:
            ValueError: If schema_type is not recognized
        """
        if schema_type == "single_cell":
            schema = TranscriptomicsSchema.get_single_cell_schema()
        elif schema_type == "bulk":
            schema = TranscriptomicsSchema.get_bulk_rna_seq_schema()
        elif schema_type == "pseudobulk":
            schema = TranscriptomicsSchema.get_pseudobulk_schema()
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        ignore_set = set(ignore_warnings) if ignore_warnings else set()

        # Add default ignored warnings for transcriptomics
        ignore_set.update(
            [
                "Unexpected obs columns",
                "Unexpected var columns",
                "missing values",
                "Very sparse data",
            ]
        )

        # Pseudobulk-specific ignored warnings
        if schema_type == "pseudobulk":
            ignore_set.add("missing values in optional fields")

        validator = FlexibleValidator(
            schema=schema,
            name=f"TranscriptomicsValidator_{schema_type}",
            ignore_warnings=ignore_set,
        )

        # Add transcriptomics-specific validation rules
        validator.add_custom_rule("check_gene_symbols", _validate_gene_symbols)
        validator.add_custom_rule("check_count_data", _validate_count_data)

        # Add cross-database accession validation (all schema types)
        validator.add_custom_rule(
            "check_cross_database_accessions",
            lambda adata: _validate_cross_database_accessions(
                adata, modality="transcriptomics"
            ),
        )

        if schema_type == "single_cell":
            validator.add_custom_rule("check_cell_metrics", _validate_cell_metrics)
        elif schema_type == "bulk":
            validator.add_custom_rule("check_sample_metrics", _validate_sample_metrics)
        elif schema_type == "pseudobulk":
            validator.add_custom_rule(
                "check_pseudobulk_structure", _validate_pseudobulk_structure
            )
            validator.add_custom_rule(
                "check_aggregation_consistency", _validate_aggregation_consistency
            )
            validator.add_custom_rule("check_cell_counts", _validate_cell_counts)
            validator.add_custom_rule(
                "check_aggregation_params", _validate_aggregation_params
            )

        return validator

    @staticmethod
    def get_recommended_qc_thresholds(
        schema_type: str = "single_cell",
    ) -> Dict[str, Any]:
        """
        Get recommended quality control thresholds.

        Args:
            schema_type: Type of schema ('single_cell', 'bulk', or 'pseudobulk')

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        if schema_type == "single_cell":
            return {
                "min_genes_per_cell": 200,
                "max_genes_per_cell": 5000,
                "min_cells_per_gene": 3,
                "max_pct_mt": 20.0,
                "max_pct_ribo": 50.0,
                "min_total_counts": 1000,
                "max_total_counts": 50000,
                "doublet_score_threshold": 0.3,
            }
        elif schema_type == "bulk":
            return {
                "min_genes_per_sample": 10000,
                "min_samples_per_gene": 2,
                "min_total_counts": 1000000,
                "min_mapping_rate": 0.7,
                "max_duplication_rate": 0.3,
                "min_rna_integrity": 6.0,
            }
        elif schema_type == "pseudobulk":
            return {
                "min_cells_per_pseudobulk": 10,  # Minimum cells for reliable aggregation
                "min_pseudobulk_samples": 3,  # Minimum samples for statistical testing
                "min_genes_per_pseudobulk": 5000,  # Minimum genes detected
                "max_zero_fraction": 0.8,  # Maximum fraction of zero values
                "min_total_aggregated_counts": 100000,  # Minimum total counts per pseudobulk sample
                "min_samples_per_celltype": 2,  # Minimum samples per cell type for DE
                "max_aggregation_imbalance": 5.0,  # Max fold difference in cell counts between conditions
            }
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")


def _validate_gene_symbols(adata) -> "ValidationResult":
    """Validate gene symbol format and uniqueness."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for gene symbols in var
    if "gene_symbol" in adata.var.columns:
        symbols = adata.var["gene_symbol"]

        # Check for duplicates
        duplicates = symbols.duplicated().sum()
        if duplicates > 0:
            result.add_warning(f"{duplicates} duplicate gene symbols found")

        # Check for missing symbols
        missing = symbols.isna().sum()
        if missing > 0:
            result.add_warning(f"{missing} missing gene symbols")

        # Basic format check (starts with letter)
        if symbols.dtype == "object":
            invalid_format = ~symbols.str.match(r"^[A-Za-z]", na=False).sum()
            if invalid_format > 0:
                result.add_warning(f"{invalid_format} gene symbols with unusual format")

    return result


def _validate_count_data(adata) -> "ValidationResult":
    """Validate count data characteristics."""
    import numpy as np

    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check for negative values
    if hasattr(adata.X, "min"):
        min_val = adata.X.min()
        if min_val < 0:
            result.add_warning(
                f"Negative values found in count matrix (min: {min_val})"
            )

    # Check for non-integer values in what should be count data
    if hasattr(adata.X, "dtype"):
        if np.issubdtype(adata.X.dtype, np.floating):
            # Check if values are effectively integers
            if hasattr(adata.X, "data"):  # Sparse matrix
                non_int_values = (adata.X.data % 1 != 0).sum()
            else:  # Dense matrix
                non_int_values = (adata.X % 1 != 0).sum()

            if non_int_values > 0:
                result.add_info(
                    f"Found {non_int_values} non-integer values (may be normalized data)"
                )

    return result


def _validate_cell_metrics(adata) -> "ValidationResult":
    """Validate single-cell specific metrics."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check mitochondrial percentage if available
    if "pct_counts_mt" in adata.obs.columns:
        mt_pct = adata.obs["pct_counts_mt"]
        high_mt = (mt_pct > 20).sum()
        if high_mt > 0:
            result.add_warning(f"{high_mt} cells with >20% mitochondrial reads")

    # Check for very low or high gene counts
    if "n_genes" in adata.obs.columns:
        n_genes = adata.obs["n_genes"]
        low_genes = (n_genes < 200).sum()
        high_genes = (n_genes > 5000).sum()

        if low_genes > 0:
            result.add_warning(f"{low_genes} cells with <200 genes detected")
        if high_genes > 0:
            result.add_warning(f"{high_genes} cells with >5000 genes detected")

    return result


def _validate_sample_metrics(adata) -> "ValidationResult":
    """Validate bulk RNA-seq specific metrics."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check mapping rate if available
    if "mapping_rate" in adata.obs.columns:
        mapping_rates = adata.obs["mapping_rate"]
        low_mapping = (mapping_rates < 0.7).sum()
        if low_mapping > 0:
            result.add_warning(f"{low_mapping} samples with <70% mapping rate")

    # Check duplication rate if available
    if "duplication_rate" in adata.obs.columns:
        dup_rates = adata.obs["duplication_rate"]
        high_dup = (dup_rates > 0.3).sum()
        if high_dup > 0:
            result.add_warning(f"{high_dup} samples with >30% duplication rate")

    return result


def _validate_pseudobulk_structure(adata) -> "ValidationResult":
    """Validate basic pseudobulk data structure."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check required observation columns
    required_obs = ["sample_id", "cell_type", "n_cells_aggregated"]
    missing_obs = [col for col in required_obs if col not in adata.obs.columns]

    if missing_obs:
        result.add_error(f"Missing required obs columns: {missing_obs}")

    # Check for composite sample identifiers (should be unique)
    if "sample_id" in adata.obs.columns and "cell_type" in adata.obs.columns:
        composite_ids = (
            adata.obs["sample_id"].astype(str)
            + "_"
            + adata.obs["cell_type"].astype(str)
        )
        duplicates = composite_ids.duplicated().sum()
        if duplicates > 0:
            result.add_error(
                f"Found {duplicates} duplicate sample_id + cell_type combinations"
            )

    # Check for required uns fields
    required_uns = ["pseudobulk_params", "aggregation_stats"]
    missing_uns = [field for field in required_uns if field not in adata.uns]

    if missing_uns:
        result.add_warning(f"Missing recommended uns fields: {missing_uns}")

    return result


def _validate_aggregation_consistency(adata) -> "ValidationResult":
    """Validate consistency of aggregation metadata."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    # Check if cell counts are reasonable
    if "n_cells_aggregated" in adata.obs.columns:
        cell_counts = adata.obs["n_cells_aggregated"]

        # Check for zero or negative cell counts
        invalid_counts = (cell_counts <= 0).sum()
        if invalid_counts > 0:
            result.add_error(
                f"Found {invalid_counts} pseudobulk samples with zero or negative cell counts"
            )

        # Check for very low cell counts
        low_counts = (cell_counts < 10).sum()
        if low_counts > 0:
            result.add_warning(
                f"Found {low_counts} pseudobulk samples with <10 cells (may be unreliable)"
            )

        # Check for extreme outliers in cell counts
        if len(cell_counts) > 1:
            q99 = cell_counts.quantile(0.99)
            q1 = cell_counts.quantile(0.01)
            if q99 > 10 * q1:
                result.add_warning(
                    f"Large variation in cell counts (99th percentile: {q99:.0f}, 1st percentile: {q1:.0f})"
                )

    return result


def _validate_cell_counts(adata) -> "ValidationResult":
    """Validate cell count distributions and balance."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    if not all(
        col in adata.obs.columns
        for col in ["sample_id", "cell_type", "n_cells_aggregated"]
    ):
        return result

    # Check balance across conditions if available
    if "condition" in adata.obs.columns:
        condition_counts = adata.obs.groupby(["condition", "cell_type"])[
            "n_cells_aggregated"
        ].sum()

        # Check if any condition-celltype combinations are missing
        conditions = adata.obs["condition"].unique()
        cell_types = adata.obs["cell_type"].unique()

        for condition in conditions:
            for cell_type in cell_types:
                if (condition, cell_type) not in condition_counts.index:
                    result.add_warning(
                        f"Missing data for condition '{condition}' and cell type '{cell_type}'"
                    )

    # Check for sufficient replication
    sample_celltype_counts = adata.obs.groupby(["sample_id", "cell_type"]).size()

    # Each sample-celltype combination should appear only once in pseudobulk
    duplicates = (sample_celltype_counts > 1).sum()
    if duplicates > 0:
        result.add_error(
            f"Found {duplicates} sample-celltype combinations with multiple entries"
        )

    return result


def _validate_aggregation_params(adata) -> "ValidationResult":
    """Validate aggregation parameters in uns."""
    from lobster.core.interfaces.validator import ValidationResult

    result = ValidationResult()

    if "pseudobulk_params" not in adata.uns:
        return result

    params = adata.uns["pseudobulk_params"]

    # Check for required parameters
    required_params = ["sample_col", "celltype_col", "min_cells", "aggregation_method"]
    missing_params = [param for param in required_params if param not in params]

    if missing_params:
        result.add_warning(f"Missing aggregation parameters: {missing_params}")

    # Validate aggregation method
    if "aggregation_method" in params:
        valid_methods = ["sum", "mean", "median"]
        if params["aggregation_method"] not in valid_methods:
            result.add_warning(
                f"Unexpected aggregation method: {params['aggregation_method']} (expected one of {valid_methods})"
            )

    # Check consistency with actual data
    if "min_cells" in params and "n_cells_aggregated" in adata.obs.columns:
        min_cells_actual = adata.obs["n_cells_aggregated"].min()
        min_cells_param = params["min_cells"]

        if min_cells_actual < min_cells_param:
            result.add_warning(
                f"Actual minimum cell count ({min_cells_actual}) is less than "
                f"specified threshold ({min_cells_param})"
            )

    return result


def _validate_cross_database_accessions(
    adata, modality: str = "transcriptomics"
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
# Pydantic Metadata Schemas for Sample-Level Metadata Standardization
# =============================================================================
# These schemas are used by the metadata_assistant agent for cross-dataset
# metadata harmonization, standardization, and validation.
# Phase 3 addition for metadata operations.
# =============================================================================


class TranscriptomicsMetadataSchema(BaseModel):
    """
    Pydantic schema for transcriptomics sample-level metadata standardization.

    This schema defines the expected structure for sample metadata across both
    single-cell and bulk RNA-seq experiments. It enforces controlled vocabularies
    and data types for consistent metadata representation across datasets.

    Used by metadata_assistant agent for:
    - Cross-dataset sample ID mapping
    - Metadata standardization and harmonization
    - Dataset completeness validation
    - Multi-omics integration preparation

    NOTE: organism, tissue, and cell_type fields have been removed.
    These are now handled by the embedding-based ontology matching service.
    See module header for details.

    Attributes:
        sample_id: Unique sample identifier (required)
        subject_id: Subject/patient identifier for biological replicates
        timepoint: Timepoint or developmental stage
        condition: Experimental condition (e.g., "Control", "Treatment")
        platform: Sequencing platform (e.g., "Illumina NovaSeq", "10x Genomics")
        sequencing_type: Type of RNA-seq ("bulk" or "single-cell")
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
        ..., description="Experimental condition (e.g., Control, Treatment)"
    )

    # NOTE: organism, tissue, cell_type fields removed - handled by embedding service
    # See module header for details on ontology integration

    platform: str = Field(..., description="Sequencing platform")
    sequencing_type: str = Field(
        ..., description="Type of RNA-seq (bulk or single-cell)"
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
                "condition": "Control",
                # organism, tissue, cell_type removed - handled by embedding service
                "platform": "Illumina NovaSeq 6000",
                "sequencing_type": "single-cell",
                "batch": "Batch1",
                "additional_metadata": {"replicate": "Rep1", "sequencing_depth": 50000},
            }
        }

    @field_validator("sequencing_type")
    @classmethod
    def validate_sequencing_type(cls, v: str) -> str:
        """Validate sequencing type is either bulk or single-cell."""
        allowed = {"bulk", "single-cell", "single_cell", "sc", "bulk_rna_seq"}
        v_lower = v.lower().replace("-", "_")
        if v_lower not in allowed:
            raise ValueError(f"sequencing_type must be one of {allowed}, got '{v}'")
        # Normalize to standard values
        if v_lower in {"single_cell", "sc"}:
            return "single-cell"
        if v_lower == "bulk_rna_seq":
            return "bulk"
        return v_lower

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
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptomicsMetadataSchema":
        """
        Create schema from dictionary, automatically handling unknown fields.

        Args:
            data: Dictionary with metadata fields

        Returns:
            TranscriptomicsMetadataSchema: Validated schema instance
        """
        # Extract known fields
        known_fields = set(cls.model_fields.keys()) - {"additional_metadata"}
        schema_data = {k: v for k, v in data.items() if k in known_fields}

        # Put remaining fields in additional_metadata
        additional = {k: v for k, v in data.items() if k not in known_fields}
        if additional:
            schema_data["additional_metadata"] = additional

        return cls(**schema_data)
