"""
Pseudobulk schema definitions for aggregated single-cell RNA-seq data.

This module defines the expected structure and metadata for pseudobulk
datasets created by aggregating single-cell RNA-seq data at the sample
and cell type level for differential expression analysis.
"""

from typing import Any, Dict, List, Optional

from lobster.core.interfaces.validator import ValidationResult
from lobster.core.schemas.validation import FlexibleValidator


class PseudobulkSchema:
    """
    Schema definitions for pseudobulk aggregated data.

    This class provides schema definitions for pseudobulk matrices created
    by aggregating single-cell RNA-seq data at the sample and cell type level,
    enabling proper differential expression analysis.
    """

    @staticmethod
    def get_pseudobulk_schema() -> Dict[str, Any]:
        """
        Get schema for pseudobulk aggregated data.

        Returns:
            Dict[str, Any]: Pseudobulk data schema definition
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
                    "tissue",  # Tissue type
                    "organism",  # Organism
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
                    "tissue": "categorical",
                    "organism": "string",
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
        strict: bool = False, ignore_warnings: Optional[List[str]] = None
    ) -> FlexibleValidator:
        """
        Create a validator for pseudobulk data.

        Args:
            strict: Whether to use strict validation
            ignore_warnings: List of warning types to ignore

        Returns:
            FlexibleValidator: Configured validator

        Raises:
            ValueError: If schema validation fails
        """
        schema = PseudobulkSchema.get_pseudobulk_schema()

        ignore_set = set(ignore_warnings) if ignore_warnings else set()

        # Add default ignored warnings for pseudobulk data
        ignore_set.update(
            [
                "Unexpected obs columns",
                "Unexpected var columns",
                "missing values in optional fields",
            ]
        )

        validator = FlexibleValidator(
            schema=schema, name="PseudobulkValidator", ignore_warnings=ignore_set
        )

        # Add pseudobulk-specific validation rules
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
    def get_recommended_qc_thresholds() -> Dict[str, Any]:
        """
        Get recommended quality control thresholds for pseudobulk data.

        Returns:
            Dict[str, Any]: QC thresholds and recommendations
        """
        return {
            "min_cells_per_pseudobulk": 10,  # Minimum cells for reliable aggregation
            "min_pseudobulk_samples": 3,  # Minimum samples for statistical testing
            "min_genes_per_pseudobulk": 5000,  # Minimum genes detected
            "max_zero_fraction": 0.8,  # Maximum fraction of zero values
            "min_total_aggregated_counts": 100000,  # Minimum total counts per pseudobulk sample
            "min_samples_per_celltype": 2,  # Minimum samples per cell type for DE
            "max_aggregation_imbalance": 5.0,  # Max fold difference in cell counts between conditions
        }


def _validate_pseudobulk_structure(adata) -> ValidationResult:
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


def _validate_aggregation_consistency(adata) -> ValidationResult:
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


def _validate_cell_counts(adata) -> ValidationResult:
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


def _validate_aggregation_params(adata) -> ValidationResult:
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
