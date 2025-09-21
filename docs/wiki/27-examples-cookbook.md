# Examples Cookbook

This comprehensive cookbook provides practical code snippets, analysis recipes, and real-world solutions for common bioinformatics tasks using Lobster AI. Each example includes complete workflows, expected outputs, and troubleshooting tips.

## Table of Contents

1. [Quick Start Recipes](#quick-start-recipes)
2. [Data Loading & Management](#data-loading--management)
3. [Single-Cell Analysis Recipes](#single-cell-analysis-recipes)
4. [Bulk RNA-seq Workflows](#bulk-rna-seq-workflows)
5. [Proteomics Analysis Patterns](#proteomics-analysis-patterns)
6. [Multi-Omics Integration](#multi-omics-integration)
7. [Visualization Recipes](#visualization-recipes)
8. [Advanced Analysis Techniques](#advanced-analysis-techniques)
9. [Automation & Scripting](#automation--scripting)
10. [Performance Optimization](#performance-optimization)

---

## Quick Start Recipes

### 🚀 Basic Analysis Pipeline

```bash
# Complete single-cell analysis in 5 commands
🦞 You: "Download GSE109564 from GEO"
🦞 You: "Assess quality and filter low-quality cells"
🦞 You: "Normalize, find variable genes, and cluster cells"
🦞 You: "Find marker genes and annotate cell types"
🦞 You: "Create comprehensive visualization dashboard"
```

### 🧬 Proteomics Quick Analysis

```bash
# MS proteomics analysis pipeline
🦞 You: "Load MaxQuant proteinGroups.txt file"
🦞 You: "Perform quality control with missing value analysis"
🦞 You: "Apply log2 transformation and normalization"
🦞 You: "Run differential expression analysis treatment vs control"
🦞 You: "Generate volcano plots and pathway analysis"
```

### 📊 Bulk RNA-seq Differential Expression

```bash
# Bulk RNA-seq with complex design
🦞 You: "Load counts.csv and metadata.csv with treatment, batch, and time factors"
🦞 You: "Design matrix using formula: ~treatment + batch + time + treatment:time"
🦞 You: "Run pyDESeq2 differential expression analysis"
🦞 You: "Test specific contrasts and create visualizations"
```

---

## Data Loading & Management

### Loading Different Data Formats

#### GEO Datasets
```bash
# Download and load GEO datasets
🦞 You: "Download GSE12345 from GEO and show dataset metadata"
🦞 You: "Download multiple datasets: GSE11111, GSE22222, GSE33333"
🦞 You: "Search GEO for single-cell datasets related to cancer immunotherapy"
```

#### Local Files
```bash
# Load various file formats
🦞 You: "Load the H5AD file from /path/to/data.h5ad"
🦞 You: "Load 10X data from /path/to/10x/directory with matrix.mtx, barcodes.tsv, features.tsv"
🦞 You: "Load CSV file with first column as gene names and samples as columns"
🦞 You: "Load Excel file from sheet 'RNAseq_counts' with genes as rows"
```

#### Proteomics Files
```bash
# Load proteomics data
🦞 You: "Load MaxQuant proteinGroups.txt file from /path/to/file"
🦞 You: "Load Olink NPX data from olink_results.xlsx"
🦞 You: "Load Spectronaut output with protein intensity values"
```

### Data Management Commands

```bash
# Workspace management
🦞 You: "/files"                    # List all loaded files
🦞 You: "/data"                     # Show current dataset info
🦞 You: "/workspace"                # Show workspace status
🦞 You: "/tree"                     # Directory tree view

# Data operations
🦞 You: "/read filename.csv"        # Read and display file contents
🦞 You: "/plots"                    # List generated visualizations
🦞 You: "/export results"           # Export analysis results
```

---

## Single-Cell Analysis Recipes

### Quality Control Patterns

#### Standard QC Pipeline
```bash
🦞 You: "Calculate QC metrics including mitochondrial genes, ribosomal genes, and total UMI counts"
🦞 You: "Generate QC violin plots showing distributions across samples"
🦞 You: "Identify outlier cells with >25% mitochondrial genes or <200 total genes"
🦞 You: "Filter cells and genes based on QC thresholds"
```

#### Advanced QC
```bash
🦞 You: "Detect doublets using scrublet algorithm"
🦞 You: "Analyze batch effects using PCA and show batch contributions"
🦞 You: "Calculate and visualize sample mixing scores"
🦞 You: "Generate comprehensive QC report with all metrics"
```

### Preprocessing Recipes

#### Basic Preprocessing
```bash
🦞 You: "Normalize to 10,000 UMI per cell and log-transform"
🦞 You: "Find highly variable genes using seurat method with 2000 genes"
🦞 You: "Scale data and regress out mitochondrial gene effects"
```

#### Batch Correction
```bash
🦞 You: "Apply Harmony batch correction for samples from different batches"
🦞 You: "Use Combat for batch correction and compare before/after PCA plots"
🦞 You: "Apply scanorama integration for multiple samples"
```

### Clustering & Annotation

#### Basic Clustering
```bash
🦞 You: "Perform PCA with 50 components and generate elbow plot"
🦞 You: "Build neighbor graph with 15 neighbors and compute UMAP"
🦞 You: "Run Leiden clustering with resolution 0.5 and evaluate cluster stability"
```

#### Advanced Clustering
```bash
🦞 You: "Test multiple clustering resolutions from 0.1 to 2.0 and compare results"
🦞 You: "Perform hierarchical clustering and cut dendrogram at different levels"
🦞 You: "Use Louvain clustering and compare with Leiden results"
```

#### Cell Type Annotation
```bash
🦞 You: "Find marker genes for each cluster using Wilcoxon test"
🦞 You: "Annotate clusters using canonical immune cell markers"
🦞 You: "Use automated cell type annotation with CellTypist"
🦞 You: "Create manual annotation based on expert knowledge"
```

### Trajectory Analysis
```bash
🦞 You: "Infer pseudotime using diffusion pseudotime (DPT)"
🦞 You: "Perform RNA velocity analysis to show differentiation dynamics"
🦞 You: "Create trajectory plots showing cellular transitions"
🦞 You: "Identify genes that change along the trajectory"
```

---

## Bulk RNA-seq Workflows

### Experimental Design Recipes

#### Simple Two-Group Comparison
```bash
🦞 You: "Design matrix for treatment vs control comparison"
🦞 You: "Run DESeq2 with formula ~condition"
🦞 You: "Generate MA plot and volcano plot"
🦞 You: "Export significant genes with log2FC > 1 and FDR < 0.05"
```

#### Multi-Factor Design
```bash
🦞 You: "Create design matrix: ~treatment + sex + age + treatment:sex"
🦞 You: "Test main effect of treatment controlling for sex and age"
🦞 You: "Test treatment×sex interaction term"
🦞 You: "Generate contrast for treatment effect in females only"
```

#### Time Course Analysis
```bash
🦞 You: "Model time as continuous variable: ~condition + time + condition:time"
🦞 You: "Identify genes with linear temporal changes"
🦞 You: "Find genes with different temporal patterns between conditions"
🦞 You: "Cluster genes by temporal expression profiles"
```

#### Batch Effect Handling
```bash
🦞 You: "Include batch in design: ~batch + condition"
🦞 You: "Apply ComBat batch correction before analysis"
🦞 You: "Use RUVSeq to identify and remove unwanted variation"
🦞 You: "Compare results with and without batch correction"
```

### Statistical Analysis Patterns

#### Multiple Contrasts
```bash
🦞 You: "Define custom contrasts: early_treatment, late_treatment, time_effect"
🦞 You: "Test all pairwise comparisons between 4 conditions"
🦞 You: "Apply different FDR thresholds: 0.01, 0.05, 0.1"
🦞 You: "Compare results across different statistical methods"
```

#### Effect Size Analysis
```bash
🦞 You: "Calculate effect sizes (Cohen's d) for all significant genes"
🦞 You: "Filter results by both significance and effect size"
🦞 You: "Generate effect size distribution plots"
🦞 You: "Identify genes with large effects but moderate significance"
```

---

## Proteomics Analysis Patterns

### MS Proteomics Workflows

#### Data Preprocessing
```bash
🦞 You: "Load MaxQuant data and assess missing value patterns"
🦞 You: "Apply MNAR imputation for low-abundance proteins"
🦞 You: "Perform TMM normalization and batch correction"
🦞 You: "Filter proteins present in >50% of samples"
```

#### Differential Analysis
```bash
🦞 You: "Run limma differential analysis with empirical Bayes"
🦞 You: "Test multiple contrasts with appropriate FDR correction"
🦞 You: "Generate volcano plots with protein annotations"
🦞 You: "Export results with UniProt annotations"
```

### Affinity Proteomics (Olink)

#### Quality Assessment
```bash
🦞 You: "Calculate coefficient of variation for all proteins"
🦞 You: "Assess antibody performance metrics"
🦞 You: "Generate QC dashboard with detection frequencies"
🦞 You: "Identify failed samples and low-quality antibodies"
```

#### Statistical Analysis
```bash
🦞 You: "Perform ANOVA across multiple conditions"
🦞 You: "Run pairwise t-tests with multiple testing correction"
🦞 You: "Generate heatmap of significant proteins"
🦞 You: "Create protein correlation network"
```

---

## Multi-Omics Integration

### RNA-seq + Proteomics Integration

#### Correlation Analysis
```bash
🦞 You: "Correlate mRNA and protein levels for matched genes"
🦞 You: "Identify genes with high RNA-protein correlation (r > 0.7)"
🦞 You: "Find proteins regulated post-translationally (low correlation)"
🦞 You: "Generate RNA vs protein scatter plots"
```

#### Pathway-Level Integration
```bash
🦞 You: "Perform pathway analysis on both RNA and protein data"
🦞 You: "Identify pathways significant in both datasets"
🦞 You: "Create integrated pathway heatmaps"
🦞 You: "Generate multi-omics pathway networks"
```

### Single-Cell + Spatial Integration
```bash
🦞 You: "Map single-cell clusters to spatial locations"
🦞 You: "Identify spatial patterns of cell type distribution"
🦞 You: "Analyze cell-cell communication in spatial context"
🦞 You: "Generate integrated spatial-molecular visualizations"
```

---

## Visualization Recipes

### Basic Plotting Commands

#### Single-Cell Visualizations
```bash
🦞 You: "Create UMAP plot colored by cell type annotations"
🦞 You: "Generate violin plots of marker genes by cluster"
🦞 You: "Create feature plots showing gene expression on UMAP"
🦞 You: "Make dotplot of top marker genes by cell type"
```

#### Bulk RNA-seq Plots
```bash
🦞 You: "Create volcano plot with gene labels for top hits"
🦞 You: "Generate MA plot showing fold-change vs abundance"
🦞 You: "Make heatmap of top 50 differentially expressed genes"
🦞 You: "Create PCA plot colored by experimental conditions"
```

#### Proteomics Visualizations
```bash
🦞 You: "Generate missing value heatmap for MS data"
🦞 You: "Create volcano plot for protein differential expression"
🦞 You: "Make correlation network of significantly changed proteins"
🦞 You: "Generate QC dashboard for Olink data"
```

### Advanced Visualization Techniques

#### Interactive Dashboards
```bash
🦞 You: "Create interactive dashboard with multiple panels"
🦞 You: "Generate plotly-based exploration interface"
🦞 You: "Make filterable data tables with visualizations"
🦞 You: "Create animated plots showing temporal changes"
```

#### Publication-Ready Figures
```bash
🦞 You: "Export high-resolution figures (300 DPI) in SVG format"
🦞 You: "Create multi-panel figures with consistent styling"
🦞 You: "Generate figures with publication-appropriate fonts and colors"
🦞 You: "Export figure data for manual customization"
```

---

## Advanced Analysis Techniques

### Machine Learning Integration

#### Dimensionality Reduction
```bash
🦞 You: "Apply t-SNE with different perplexity values"
🦞 You: "Use UMAP with custom distance metrics"
🦞 You: "Perform diffusion maps for trajectory inference"
🦞 You: "Apply autoencoders for non-linear dimension reduction"
```

#### Classification and Prediction
```bash
🦞 You: "Train classifier to predict cell types from expression"
🦞 You: "Build regression model for continuous phenotypes"
🦞 You: "Perform cross-validation and assess model performance"
🦞 You: "Use feature selection to identify predictive genes"
```

### Network Analysis
```bash
🦞 You: "Build gene co-expression networks using WGCNA"
🦞 You: "Create protein-protein interaction networks"
🦞 You: "Analyze network topology and identify hubs"
🦞 You: "Perform network-based pathway analysis"
```

### Functional Analysis

#### Pathway Enrichment
```bash
🦞 You: "Run Gene Ontology enrichment analysis"
🦞 You: "Perform KEGG pathway analysis"
🦞 You: "Use Reactome for pathway annotation"
🦞 You: "Create enrichment plots with significance levels"
```

#### Gene Set Analysis
```bash
🦞 You: "Perform GSEA with custom gene sets"
🦞 You: "Test enrichment of MSigDB collections"
🦞 You: "Create leading edge analysis plots"
🦞 You: "Generate enrichment heatmaps"
```

---

## Automation & Scripting

### Batch Processing Workflows

#### Process Multiple Datasets
```python
# Python script for batch processing
#!/usr/bin/env python3

from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2
from pathlib import Path

def batch_process_datasets(dataset_paths, output_dir):
    """Process multiple datasets with standard pipeline."""

    for dataset_path in dataset_paths:
        print(f"Processing {dataset_path}")

        # Initialize fresh workspace
        workspace = Path(output_dir) / f"analysis_{dataset_path.stem}"
        data_manager = DataManagerV2(workspace_path=workspace)
        client = AgentClient(data_manager=data_manager)

        # Standard analysis pipeline
        queries = [
            f"Load data from {dataset_path}",
            "Perform quality control and filtering",
            "Normalize and find variable genes",
            "Run clustering analysis",
            "Find marker genes and annotate cell types",
            "Export results and visualizations"
        ]

        for query in queries:
            result = client.query(query)
            if not result['success']:
                print(f"Failed: {query}")
                break

        print(f"Completed analysis for {dataset_path}")

# Usage
dataset_paths = [
    Path("data/dataset1.h5ad"),
    Path("data/dataset2.h5ad"),
    Path("data/dataset3.h5ad")
]

batch_process_datasets(dataset_paths, "batch_results")
```

#### Automated Report Generation
```bash
🦞 You: "Generate automated analysis report for all loaded datasets"
🦞 You: "Create summary statistics table comparing all samples"
🦞 You: "Export standardized figure set for publication"
🦞 You: "Generate methods description for manuscript"
```

### Parameter Optimization

#### Systematic Parameter Testing
```bash
🦞 You: "Test clustering resolutions from 0.1 to 2.0 in steps of 0.1"
🦞 You: "Compare different normalization methods and show results"
🦞 You: "Optimize PCA components by testing 10, 20, 30, 40, 50"
🦞 You: "Test different QC thresholds and compare cell numbers"
```

---

## Performance Optimization

### Memory-Efficient Processing

#### Large Dataset Handling
```bash
🦞 You: "Process large dataset >100k cells using chunked analysis"
🦞 You: "Use memory-efficient file formats (H5AD, Zarr)"
🦞 You: "Apply subsampling for initial exploration"
🦞 You: "Use sparse matrix operations for memory efficiency"
```

#### Parallel Processing
```bash
🦞 You: "Run analysis using multiple CPU cores"
🦞 You: "Process samples in parallel for batch analysis"
🦞 You: "Use GPU acceleration for large matrix operations"
🦞 You: "Optimize I/O operations for network storage"
```

### Cloud Integration Patterns
```bash
# Set up cloud processing
export LOBSTER_CLOUD_KEY="your-api-key"

🦞 You: "Upload large dataset to cloud for processing"
🦞 You: "Run memory-intensive analysis on cloud infrastructure"
🦞 You: "Download results and visualizations locally"
🦞 You: "Switch between local and cloud processing seamlessly"
```

---

## Common Analysis Combinations

### Complete Single-Cell Pipeline
```bash
# Full single-cell analysis workflow
🦞 You: "Download GSE datasets for single-cell immune atlas"
🦞 You: "Merge multiple samples and batch correct"
🦞 You: "Perform comprehensive quality control"
🦞 You: "Apply clustering and cell type annotation"
🦞 You: "Generate trajectory analysis and pseudotime"
🦞 You: "Create publication dashboard"
```

### Integrated Multi-Omics Analysis
```bash
# Multi-omics integration workflow
🦞 You: "Load paired RNA-seq and proteomics data"
🦞 You: "Perform quality control on both datasets"
🦞 You: "Run differential analysis for each platform"
🦞 You: "Correlate changes across omics layers"
🦞 You: "Perform integrated pathway analysis"
🦞 You: "Generate multi-omics summary report"
```

### Time Series Analysis
```bash
# Temporal analysis workflow
🦞 You: "Load time series data with multiple time points"
🦞 You: "Model temporal patterns using spline regression"
🦞 You: "Identify genes with significant time trends"
🦞 You: "Cluster genes by temporal expression patterns"
🦞 You: "Create animated visualizations of changes"
```

---

## Troubleshooting Recipes

### Common Issues and Solutions

#### Data Loading Problems
```bash
# File format issues
🦞 You: "My CSV file has gene names in the first row - how to load correctly?"
🦞 You: "The H5AD file seems corrupted - can you validate and repair it?"
🦞 You: "Excel file has multiple sheets - load from specific sheet 'RNAseq'"

# Memory issues
🦞 You: "Dataset too large for memory - use chunked processing"
🦞 You: "Convert dense matrix to sparse format to save memory"
```

#### Analysis Issues
```bash
# Quality control
🦞 You: "No cells pass QC filters - adjust thresholds more liberally"
🦞 You: "Too many cells filtered out - show QC distribution plots"

# Clustering problems
🦞 You: "Clusters look poorly separated - try different resolution"
🦞 You: "Getting too many/few clusters - optimize parameters"

# Statistical issues
🦞 You: "No significant genes found - check sample sizes and effect sizes"
🦞 You: "P-value distribution looks biased - investigate data quality"
```

#### Performance Issues
```bash
# Speed optimization
🦞 You: "Analysis taking too long - use faster approximate methods"
🦞 You: "Enable parallel processing to speed up computation"
🦞 You: "Use cloud processing for large dataset analysis"
```

---

## Tips and Best Practices

### Natural Language Best Practices

#### Effective Query Patterns
```bash
# ✅ Good queries - specific and clear
🦞 You: "Load single-cell data from GSE12345 and perform quality control"
🦞 You: "Run differential expression between condition A and B using DESeq2"
🦞 You: "Create UMAP plot colored by cell type with cluster labels"

# ❌ Avoid vague queries
🦞 You: "Analyze my data"
🦞 You: "Make some plots"
🦞 You: "Fix the problem"
```

#### Providing Context
```bash
# ✅ Include relevant context
🦞 You: "I have Visium spatial data with 2000 spots - cluster into tissue regions"
🦞 You: "This is proteomics data from MaxQuant with 40% missing values"
🦞 You: "Time series RNA-seq with samples at 0h, 6h, 12h, 24h timepoints"
```

### Analysis Strategy Tips

1. **Start Simple**: Begin with basic analyses before complex workflows
2. **Check Quality**: Always perform QC before downstream analysis
3. **Validate Results**: Cross-check findings with different methods
4. **Document Parameters**: Keep track of analysis settings
5. **Save Checkpoints**: Export intermediate results regularly

### Performance Tips

1. **Use Appropriate Data Types**: Sparse matrices for single-cell, dense for bulk
2. **Optimize Memory**: Filter unnecessary genes/cells early
3. **Parallel Processing**: Leverage multiple cores when available
4. **Cloud Resources**: Use cloud for large-scale analyses
5. **Caching**: Reuse preprocessed data when possible

---

This cookbook provides a comprehensive collection of practical examples for using Lobster AI effectively. Each recipe can be adapted to your specific datasets and analysis needs. For more detailed tutorials, see the individual tutorial documents for [single-cell](23-tutorial-single-cell.md), [bulk RNA-seq](24-tutorial-bulk-rnaseq.md), and [proteomics](25-tutorial-proteomics.md) analysis.