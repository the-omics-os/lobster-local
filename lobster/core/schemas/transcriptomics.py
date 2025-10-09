"""
Transcriptomics schema definitions for single-cell and bulk RNA-seq data.

This module defines the expected structure and metadata for transcriptomics
datasets including single-cell RNA-seq and bulk RNA-seq with appropriate
validation rules.
"""

from typing import Any, Dict, List, Optional

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.validation import FlexibleValidator


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
                    "cell_type",  # Annotated cell type
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
                ],
                "types": {
                    "cell_id": "string",
                    "sample_id": "string",
                    "batch": "string",
                    "condition": "string",
                    "cell_type": "categorical",
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
                    "tissue",  # Tissue type
                    "organism",  # Organism
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
                    "tissue": "categorical",
                    "organism": "string",
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
                ],
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
            schema_type: Type of schema ('single_cell' or 'bulk')
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

        validator = FlexibleValidator(
            schema=schema,
            name=f"TranscriptomicsValidator_{schema_type}",
            ignore_warnings=ignore_set,
        )

        # Add transcriptomics-specific validation rules
        validator.add_custom_rule("check_gene_symbols", _validate_gene_symbols)
        validator.add_custom_rule("check_count_data", _validate_count_data)

        if schema_type == "single_cell":
            validator.add_custom_rule("check_cell_metrics", _validate_cell_metrics)
        elif schema_type == "bulk":
            validator.add_custom_rule("check_sample_metrics", _validate_sample_metrics)

        return validator

    @staticmethod
    def get_recommended_qc_thresholds(
        schema_type: str = "single_cell",
    ) -> Dict[str, Any]:
        """
        Get recommended quality control thresholds.

        Args:
            schema_type: Type of schema ('single_cell' or 'bulk')

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
