# Services API Reference

## Overview

The Services API provides stateless analysis services implementing scientific algorithms for bioinformatics workflows. All services follow the stateless pattern, accepting AnnData objects as input and returning a tuple of (processed_adata, statistics_dict). This design ensures reproducibility, testability, and easy integration with the agent system.

## Service Design Pattern

All services follow the standard stateless pattern:

```python
class ExampleService:
    """Stateless service for biological data analysis."""

    def __init__(self):
        """Initialize the service (no state stored)."""
        pass

    def analyze(
        self,
        adata: anndata.AnnData,
        **kwargs
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform analysis on AnnData object.

        Args:
            adata: Input AnnData object
            **kwargs: Analysis parameters

        Returns:
            Tuple[anndata.AnnData, Dict[str, Any]]: Processed data and statistics
        """
        # Process data
        processed_adata = self._process_data(adata, **kwargs)

        # Calculate statistics
        statistics = self._calculate_statistics(processed_adata, adata, **kwargs)

        return processed_adata, statistics
```

## Transcriptomics Services

### PreprocessingService

Advanced preprocessing service for single-cell RNA-seq data.

```python
class PreprocessingService:
    """
    Advanced preprocessing service for single-cell RNA-seq data.

    This stateless service provides methods for ambient RNA correction, quality control filtering,
    normalization, and batch correction/integration following best practices.
    """
```

#### Methods

##### correct_ambient_rna

```python
def correct_ambient_rna(
    self,
    adata: anndata.AnnData,
    contamination_fraction: float = 0.1,
    empty_droplet_threshold: int = 100,
    method: str = "simple_decontamination"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Correct for ambient RNA contamination using simplified decontamination methods.

**Parameters:**
- `adata` (anndata.AnnData): AnnData object with raw UMI counts
- `contamination_fraction` (float): Expected fraction of ambient RNA (0.05-0.2 typical)
- `empty_droplet_threshold` (int): Minimum UMI count to consider droplet as cell-containing
- `method` (str): Method to use ('simple_decontamination', 'quantile_based')

**Returns:**
- `Tuple[anndata.AnnData, Dict[str, Any]]`: Corrected AnnData and processing stats

##### filter_cells_and_genes

```python
def filter_cells_and_genes(
    self,
    adata: anndata.AnnData,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    max_genes_per_cell: int = None,
    max_pct_mito: float = 20.0,
    max_pct_ribo: float = None
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Filter cells and genes based on quality metrics.

**Parameters:**
- `min_genes_per_cell` (int): Minimum genes expressed per cell
- `min_cells_per_gene` (int): Minimum cells expressing each gene
- `max_genes_per_cell` (int): Maximum genes per cell (removes potential doublets)
- `max_pct_mito` (float): Maximum mitochondrial gene percentage
- `max_pct_ribo` (float): Maximum ribosomal gene percentage

##### normalize_data

```python
def normalize_data(
    self,
    adata: anndata.AnnData,
    target_sum: float = 1e4,
    normalization_method: str = "log1p",
    highly_variable_genes: bool = True,
    n_top_genes: int = 2000
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Normalize expression data and identify highly variable genes.

**Parameters:**
- `target_sum` (float): Target sum for normalization
- `normalization_method` (str): Method ('log1p', 'sqrt', 'none')
- `highly_variable_genes` (bool): Whether to identify highly variable genes
- `n_top_genes` (int): Number of highly variable genes to identify

### QualityService

Quality assessment service for single-cell data.

```python
class QualityService:
    """Service for assessing data quality with comprehensive metrics."""
```

#### Methods

##### assess_quality_comprehensive

```python
def assess_quality_comprehensive(
    self,
    adata: anndata.AnnData,
    organism: str = "human",
    include_scrublet: bool = True
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Perform comprehensive quality assessment including doublet detection.

**Parameters:**
- `organism` (str): Organism type for gene set analysis ('human', 'mouse')
- `include_scrublet` (bool): Whether to include Scrublet doublet detection

### ClusteringService

Clustering service for single-cell RNA-seq data.

```python
class ClusteringService:
    """Stateless service for clustering single-cell RNA-seq data."""
```

#### Methods

##### cluster_and_visualize

```python
def cluster_and_visualize(
    self,
    adata: anndata.AnnData,
    resolution: Optional[float] = None,
    use_rep: Optional[str] = None,
    batch_correction: bool = False,
    batch_key: Optional[str] = None,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    umap_min_dist: float = 0.5,
    random_state: int = 42
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Perform clustering and dimensionality reduction with UMAP visualization.

**Parameters:**
- `resolution` (float): Clustering resolution for Leiden algorithm
- `use_rep` (str): Representation to use for clustering ('X_pca', 'X_harmony')
- `batch_correction` (bool): Whether to apply batch correction
- `batch_key` (str): Column name for batch information
- `n_pcs` (int): Number of principal components
- `n_neighbors` (int): Number of neighbors for graph construction
- `umap_min_dist` (float): UMAP minimum distance parameter

### EnhancedSinglecellService

Enhanced single-cell analysis service with advanced features.

```python
class EnhancedSinglecellService:
    """Enhanced service for advanced single-cell analysis workflows."""
```

#### Methods

##### detect_doublets_comprehensive

```python
def detect_doublets_comprehensive(
    self,
    adata: anndata.AnnData,
    expected_doublet_rate: float = 0.1,
    use_scrublet: bool = True,
    use_doubletfinder_alternative: bool = True,
    n_neighbors: int = None,
    n_pcs: int = 30
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Comprehensive doublet detection using multiple methods.

##### find_marker_genes

```python
def find_marker_genes(
    self,
    adata: anndata.AnnData,
    groupby: str,
    method: str = "wilcoxon",
    n_genes: int = 100,
    reference: str = "rest",
    min_fold_change: float = 1.5,
    max_pval_adj: float = 0.05
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Find marker genes for clusters or groups using statistical testing.

### BulkRNAseqService

Service for bulk RNA-seq analysis with pyDESeq2 integration.

```python
class BulkRNAseqService:
    """Service for bulk RNA-seq differential expression analysis."""
```

#### Methods

##### run_deseq2_analysis

```python
def run_deseq2_analysis(
    self,
    adata: anndata.AnnData,
    design_formula: str,
    condition_col: str,
    reference_level: str = None,
    batch_col: str = None,
    min_count: int = 10,
    alpha: float = 0.05
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Run differential expression analysis using pyDESeq2.

**Parameters:**
- `design_formula` (str): R-style formula for experimental design
- `condition_col` (str): Column name for the main condition
- `reference_level` (str): Reference level for comparison
- `batch_col` (str): Column name for batch effects
- `min_count` (int): Minimum count threshold
- `alpha` (float): Significance threshold

### DifferentialFormulaService

Service for R-style formula construction and design matrix generation.

```python
class DifferentialFormulaService:
    """Service for constructing and validating R-style formulas for differential analysis."""
```

#### Methods

##### construct_formula

```python
def construct_formula(
    self,
    adata: anndata.AnnData,
    primary_condition: str,
    covariates: List[str] = None,
    interactions: List[Tuple[str, str]] = None,
    formula_type: str = "additive"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Construct and validate R-style formula for differential analysis.

**Parameters:**
- `primary_condition` (str): Main condition of interest
- `covariates` (List[str]): Additional covariates to include
- `interactions` (List[Tuple[str, str]]): Interaction terms
- `formula_type` (str): Type of formula ('additive', 'interaction')

### PseudobulkService

Service for aggregating single-cell data to pseudobulk.

```python
class PseudobulkService:
    """Service for converting single-cell data to pseudobulk for differential expression."""
```

#### Methods

##### create_pseudobulk

```python
def create_pseudobulk(
    self,
    adata: anndata.AnnData,
    sample_col: str,
    cluster_col: str = None,
    min_cells: int = 10,
    aggregation_method: str = "sum"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Convert single-cell data to pseudobulk samples.

**Parameters:**
- `sample_col` (str): Column identifying individual samples
- `cluster_col` (str): Optional column for cell type-specific pseudobulk
- `min_cells` (int): Minimum cells required per pseudobulk sample
- `aggregation_method` (str): Method for aggregation ('sum', 'mean')

## Proteomics Services

### ProteomicsPreprocessingService

Preprocessing service for proteomics data.

```python
class ProteomicsPreprocessingService:
    """Service for preprocessing proteomics data including missing value handling."""
```

#### Methods

##### handle_missing_values

```python
def handle_missing_values(
    self,
    adata: anndata.AnnData,
    missing_strategy: str = "hybrid",
    imputation_method: str = "knn",
    filter_threshold: float = 0.7,
    min_valid_values: int = 3
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Handle missing values in proteomics data with multiple strategies.

**Parameters:**
- `missing_strategy` (str): Strategy ('filter', 'impute', 'hybrid')
- `imputation_method` (str): Method for imputation ('knn', 'mice', 'mean')
- `filter_threshold` (float): Threshold for filtering features with too many missing values
- `min_valid_values` (int): Minimum valid values required per feature

##### normalize_intensities

```python
def normalize_intensities(
    self,
    adata: anndata.AnnData,
    method: str = "tmm",
    log_transform: bool = True,
    center_median: bool = True
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Normalize protein intensities using various methods.

**Parameters:**
- `method` (str): Normalization method ('tmm', 'quantile', 'vsn', 'median')
- `log_transform` (bool): Whether to apply log transformation
- `center_median` (bool): Whether to center by median

### ProteomicsQualityService

Quality assessment service for proteomics data.

```python
class ProteomicsQualityService:
    """Service for assessing proteomics data quality."""
```

#### Methods

##### assess_data_quality

```python
def assess_data_quality(
    self,
    adata: anndata.AnnData,
    cv_threshold: float = 0.3,
    missing_threshold: float = 0.5
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Comprehensive quality assessment for proteomics data.

**Parameters:**
- `cv_threshold` (float): Coefficient of variation threshold
- `missing_threshold` (float): Missing value threshold for quality flags

### ProteomicsAnalysisService

Analysis service for proteomics data.

```python
class ProteomicsAnalysisService:
    """Service for proteomics statistical analysis and pathway enrichment."""
```

#### Methods

##### perform_differential_analysis

```python
def perform_differential_analysis(
    self,
    adata: anndata.AnnData,
    group_col: str,
    reference_group: str = None,
    method: str = "limma",
    adjust_method: str = "BH"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Perform differential protein expression analysis.

**Parameters:**
- `group_col` (str): Column for grouping samples
- `reference_group` (str): Reference group for comparison
- `method` (str): Statistical method ('limma', 't-test', 'wilcoxon')
- `adjust_method` (str): Multiple testing correction method

## Utility Services

### GEOService

Service for downloading and processing GEO datasets.

```python
class GEOService:
    """Service for fetching and processing GEO datasets."""
```

#### Methods

##### fetch_metadata_only

```python
def fetch_metadata_only(
    self,
    geo_id: str,
    include_sample_info: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]
```

Fetch metadata for a GEO dataset without downloading expression data.

**Parameters:**
- `geo_id` (str): GEO accession number
- `include_sample_info` (bool): Whether to include detailed sample information

**Returns:**
- `Tuple[Dict[str, Any], Dict[str, Any]]`: Metadata and validation results

##### download_and_process

```python
def download_and_process(
    self,
    geo_id: str,
    sample_limit: Optional[int] = None,
    concatenation_strategy: str = "guided"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Download and process GEO dataset with guided concatenation.

### PublicationService

Service for literature mining and dataset discovery.

```python
class PublicationService:
    """Service for searching literature and finding associated datasets."""
```

#### Methods

##### search_literature

```python
def search_literature(
    self,
    query: str,
    max_results: int = 10,
    publication_year_range: Tuple[int, int] = None,
    journal_filter: List[str] = None
) -> Dict[str, Any]
```

Search PubMed for relevant literature.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum number of results
- `publication_year_range` (Tuple[int, int]): Year range filter
- `journal_filter` (List[str]): List of journals to include

##### find_datasets_from_publication

```python
def find_datasets_from_publication(
    self,
    pmid: str,
    dataset_types: List[str] = None
) -> Dict[str, Any]
```

Find datasets associated with a publication.

### ConcatenationService

Service for combining multiple samples or datasets.

```python
class ConcatenationService:
    """Service for concatenating samples with batch correction and validation."""
```

#### Methods

##### concatenate_samples

```python
def concatenate_samples(
    self,
    adata_list: List[anndata.AnnData],
    batch_key: str = "batch",
    batch_correction_method: str = "harmony",
    join_method: str = "outer"
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Concatenate multiple AnnData objects with batch correction.

**Parameters:**
- `adata_list` (List[anndata.AnnData]): List of AnnData objects to concatenate
- `batch_key` (str): Column name for batch information
- `batch_correction_method` (str): Method for batch correction ('harmony', 'scanorama', 'none')
- `join_method` (str): How to join variables ('outer', 'inner')

### VisualizationService

Service for creating scientific visualizations.

```python
class VisualizationService:
    """Service for creating publication-quality visualizations."""
```

#### Methods

##### create_umap_plot

```python
def create_umap_plot(
    self,
    adata: anndata.AnnData,
    color_by: str = None,
    use_raw: bool = False,
    point_size: float = 1.0,
    alpha: float = 0.8,
    color_map: str = "viridis"
) -> go.Figure
```

Create UMAP visualization with customizable styling.

##### create_volcano_plot

```python
def create_volcano_plot(
    self,
    results_df: pd.DataFrame,
    log2fc_col: str = "log2FoldChange",
    pvalue_col: str = "padj",
    significance_threshold: float = 0.05,
    fold_change_threshold: float = 1.0
) -> go.Figure
```

Create volcano plot for differential expression results.

##### create_heatmap

```python
def create_heatmap(
    self,
    adata: anndata.AnnData,
    genes: List[str],
    groupby: str = None,
    use_raw: bool = False,
    standard_scale: str = None,
    cmap: str = "RdBu_r"
) -> go.Figure
```

Create expression heatmap for selected genes.

## Advanced Services

### MLProteomicsService (ALPHA)

Machine learning service for proteomics data.

```python
class MLProteomicsService:
    """Alpha service for machine learning applications in proteomics."""
```

### MLTranscriptomicsService (ALPHA)

Machine learning service for transcriptomics data.

```python
class MLTranscriptomicsService:
    """Alpha service for machine learning applications in transcriptomics."""
```

### SCVIEmbeddingService

Service for scVI-based embeddings and batch correction.

```python
class SCVIEmbeddingService:
    """Service for scVI-based dimensionality reduction and batch correction."""
```

#### Methods

##### train_scvi_model

```python
def train_scvi_model(
    self,
    adata: anndata.AnnData,
    batch_key: str = None,
    n_latent: int = 10,
    n_epochs: int = 400,
    early_stopping: bool = True
) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

Train scVI model for dimensionality reduction and batch correction.

## Error Handling in Services

All services implement consistent error handling:

### Exception Hierarchy

```python
class ServiceError(Exception):
    """Base exception for service operations."""
    pass

class PreprocessingError(ServiceError):
    """Exception for preprocessing operations."""
    pass

class AnalysisError(ServiceError):
    """Exception for analysis operations."""
    pass

class ValidationError(ServiceError):
    """Exception for validation operations."""
    pass
```

### Error Response Pattern

```python
def handle_service_error(func):
    """Decorator for consistent service error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Service error in {func.__name__}: {e}")
            raise ServiceError(f"Operation failed: {str(e)}") from e
    return wrapper
```

## Progress Callbacks

Services support progress callbacks for long-running operations:

```python
def set_progress_callback(self, callback: Callable[[int, str], None]) -> None:
    """
    Set a callback function to report progress.

    Args:
        callback: Function accepting (progress_percent, message)
    """
    self.progress_callback = callback
```

## Service Integration Examples

### Using Services Directly

```python
from lobster.tools.preprocessing_service import PreprocessingService
from lobster.tools.clustering_service import ClusteringService

# Initialize services
preprocess = PreprocessingService()
cluster = ClusteringService()

# Process data through pipeline
filtered_adata, filter_stats = preprocess.filter_cells_and_genes(adata)
normalized_adata, norm_stats = preprocess.normalize_data(filtered_adata)
clustered_adata, cluster_stats = cluster.cluster_and_visualize(normalized_adata)
```

### Service Chain Pattern

```python
def create_analysis_pipeline(services: List, params: List[Dict]) -> Callable:
    """Create a pipeline from multiple services."""
    def pipeline(adata: anndata.AnnData) -> Tuple[anndata.AnnData, Dict]:
        current_adata = adata
        all_stats = {}

        for service, param_dict in zip(services, params):
            current_adata, stats = service(**param_dict)(current_adata)
            all_stats.update(stats)

        return current_adata, all_stats

    return pipeline
```

### Validation and Quality Control

All services include built-in validation:

```python
def validate_input(self, adata: anndata.AnnData) -> None:
    """Validate AnnData input for service operations."""
    if adata is None:
        raise ValueError("AnnData object cannot be None")
    if adata.n_obs == 0:
        raise ValueError("No observations in AnnData object")
    if adata.n_vars == 0:
        raise ValueError("No variables in AnnData object")
```

The Services API provides a comprehensive set of stateless, reproducible analysis tools that form the computational backbone of the Lobster AI system. Each service is designed to be used independently or as part of larger analysis workflows, with consistent interfaces and robust error handling throughout.