# Data Analysis Workflows

## Overview

This guide provides step-by-step workflows for analyzing different types of biological data using Lobster AI. Each workflow combines natural language interaction with specialized AI agents to perform publication-quality analysis.

## Single-Cell RNA-seq Analysis Workflow

### Workflow Overview

**Goal**: Analyze single-cell RNA-seq data to identify cell types, find marker genes, and understand cellular heterogeneity.

**Agent**: Single-Cell Expert handles all aspects of scRNA-seq analysis.

**Time**: 15-30 minutes for typical dataset (10K-50K cells)

### Step 1: Data Loading and Initial Assessment

```bash
# Load your single-cell data
/read my_singlecell_data.h5ad

# Alternative: Load from multiple formats
/read counts_matrix.csv
/read filtered_feature_bc_matrix/  # 10X format
/read *.h5                        # Multiple files
```

**Natural Language Alternative**:
```
"Load my single-cell RNA-seq data from the h5ad file"
```

**Expected Output**:
- Data shape (cells × genes)
- File format confirmation
- Initial data structure summary

### Step 2: Data Quality Assessment

```bash
# Check data overview
/data

# Request quality control analysis
"Perform quality control analysis on this single-cell data"
```

**Quality Control Includes**:
- **Mitochondrial Gene Percentage**: Cell viability indicator
- **Ribosomal Gene Percentage**: Translation activity
- **Total Gene Counts**: Library complexity
- **Total UMI Counts**: Sequencing depth
- **Doublet Detection**: Multi-cell artifacts

**Expected Results**:
- Quality control metrics for each cell
- Distribution plots for QC metrics
- Recommendations for filtering thresholds

### Step 3: Data Filtering and Preprocessing

```
"Filter low-quality cells and normalize the data using standard parameters"
```

**Or specify custom parameters**:
```
"Filter cells with less than 200 genes and more than 20% mitochondrial content, then normalize using log1p transformation"
```

**Processing Steps**:
1. **Cell Filtering**: Remove low-quality cells
2. **Gene Filtering**: Remove rarely expressed genes
3. **Normalization**: Library size normalization + log1p
4. **Highly Variable Genes**: Identify most informative features

**Expected Output**:
- Filtered dataset dimensions
- Normalization parameters used
- Quality metrics after filtering

### Step 4: Dimensionality Reduction and Clustering

```
"Perform PCA, compute neighbors, and cluster the cells using Leiden algorithm"
```

**Or request comprehensive analysis**:
```
"Run the complete single-cell workflow: PCA, UMAP, clustering, and find marker genes"
```

**Analysis Steps**:
1. **Principal Component Analysis (PCA)**: Reduce dimensionality
2. **Neighborhood Graph**: Build cell-cell similarity network
3. **Leiden Clustering**: Identify cell communities
4. **UMAP Embedding**: 2D visualization

**Expected Results**:
- UMAP plot with colored clusters
- Cluster statistics and cell counts
- Quality assessment of clustering

### Step 5: Cell Type Annotation

```
"Identify the cell types in each cluster using marker genes"
```

**For specific tissue**:
```
"Annotate cell types in this liver single-cell data using known liver cell markers"
```

**Annotation Methods**:
1. **Marker Gene Analysis**: Find top genes per cluster
2. **Reference Mapping**: Compare to cell atlases
3. **Manual Annotation**: User-guided cell type assignment
4. **Automated Annotation**: ML-based cell type prediction

**Expected Results**:
- Marker genes table for each cluster
- Cell type annotations
- UMAP plot with cell type labels
- Confidence scores for annotations

### Step 6: Differential Expression Analysis

```
"Find differentially expressed genes between cell types"
```

**For specific comparison**:
```
"Compare hepatocytes and stellate cells to find differentially expressed genes"
```

**Or condition-based analysis**:
```
"Find genes differentially expressed between control and treatment conditions in each cell type"
```

**Analysis Features**:
- **Statistical Testing**: Wilcoxon rank-sum test
- **Multiple Testing Correction**: Benjamini-Hochberg FDR
- **Effect Size Filtering**: Log fold change thresholds
- **Visualization**: Volcano plots and heatmaps

### Step 7: Advanced Analysis (Optional)

#### Trajectory Analysis
```
"Perform trajectory analysis to identify developmental paths"
```

#### Pseudobulk Analysis
```
"Aggregate cells by type and perform bulk RNA-seq differential expression"
```

#### Gene Set Enrichment
```
"Perform pathway enrichment analysis on the differentially expressed genes"
```

### Complete Workflow Example

```bash
# 1. Load data
/read liver_scrnaseq.h5ad

# 2. Comprehensive analysis request
"Analyze this liver single-cell RNA-seq data: perform quality control,
filter low-quality cells, normalize, cluster cells, identify cell types,
and find marker genes for each cluster"

# 3. Specific follow-up
"Compare hepatocytes between control and fibrotic conditions"

# 4. Visualization
/plots  # View all generated plots

# 5. Save results
/save
```

## Bulk RNA-seq Analysis Workflow

### Workflow Overview

**Goal**: Analyze bulk RNA-seq data to identify differentially expressed genes between conditions.

**Agent**: Bulk RNA-seq Expert specializes in count-based differential expression analysis.

**Time**: 10-20 minutes for typical experiment

### Step 1: Data Preparation

```bash
# Load count matrix
/read counts_matrix.csv

# Load with metadata
/read counts.csv
"Load the sample metadata file to define experimental conditions"
```

**Expected Data Format**:
- Rows: Genes/transcripts
- Columns: Samples
- Raw or normalized counts

### Step 2: Experimental Design Setup

```
"Set up differential expression analysis comparing treatment vs control groups"
```

**For complex designs**:
```
"Analyze differential expression using the formula: ~condition + batch + gender"
```

**Features**:
- **R-style Formulas**: Support complex experimental designs
- **Batch Effect Handling**: Automatic detection and correction
- **Multiple Factors**: Age, gender, batch, treatment interactions
- **Contrasts**: Flexible comparison specifications

### Step 3: Quality Control

```
"Generate quality control plots and assess data distribution"
```

**QC Analysis Includes**:
- **Count Distribution**: Library size assessment
- **PCA Plots**: Sample clustering and batch effects
- **Correlation Heatmaps**: Sample relationships
- **Dispersion Plots**: Model fitting quality

### Step 4: Differential Expression with pyDESeq2

```
"Perform differential expression analysis using DESeq2"
```

**Analysis Features**:
- **Normalization**: Size factor estimation
- **Dispersion Modeling**: Gene-wise and fitted dispersions
- **Statistical Testing**: Wald test or likelihood ratio test
- **Shrinkage**: Effect size shrinkage for better estimates

**Results Include**:
- Log2 fold changes with confidence intervals
- P-values and adjusted P-values (FDR)
- Base means and dispersion estimates
- Convergence diagnostics

### Step 5: Results Visualization

```
"Create volcano plots and heatmaps for the differential expression results"
```

**Visualization Options**:
- **Volcano Plots**: Effect size vs significance
- **MA Plots**: Mean expression vs fold change
- **Heatmaps**: Top differentially expressed genes
- **PCA Plots**: Sample relationships

### Step 6: Downstream Analysis

```
"Perform pathway enrichment analysis on the upregulated genes"
```

**Advanced Analysis**:
- Gene set enrichment analysis (GSEA)
- Pathway over-representation analysis
- Gene ontology analysis
- KEGG pathway mapping

### Complete Workflow Example

```bash
# 1. Load data
/read rnaseq_counts.csv

# 2. Define experimental setup
"Analyze differential expression between high-fat diet and control mice,
accounting for batch effects and gender differences"

# 3. Request comprehensive analysis
"Perform complete bulk RNA-seq analysis: quality control, normalization,
differential expression testing, and generate volcano plots"

# 4. Follow-up analysis
"Show me the top 20 upregulated genes and their functions"

# 5. Export results
/export
```

## Mass Spectrometry Proteomics Workflow

### Workflow Overview

**Goal**: Analyze label-free quantitative proteomics data to identify differentially abundant proteins.

**Agent**: MS Proteomics Expert handles mass spectrometry data analysis.

**Time**: 20-40 minutes depending on dataset complexity

### Step 1: Data Loading

```bash
# Load MaxQuant output
/read proteinGroups.txt

# Load Spectronaut results
/read spectronaut_results.csv

# Load generic proteomics data
/read protein_intensities.csv
```

### Step 2: Data Assessment

```
"Assess the quality of this proteomics data and show missing value patterns"
```

**Quality Assessment**:
- **Missing Value Analysis**: MNAR vs MCAR patterns
- **Coefficient of Variation**: Technical and biological CV
- **Intensity Distributions**: Dynamic range assessment
- **Batch Effect Detection**: Systematic biases

### Step 3: Data Preprocessing

```
"Filter proteins with excessive missing values and normalize intensities"
```

**Preprocessing Steps**:
1. **Protein Filtering**: Remove contaminants and reverse sequences
2. **Missing Value Handling**: Imputation strategies (MNAR/MCAR)
3. **Intensity Normalization**: TMM, quantile, or VSN normalization
4. **Log Transformation**: Variance stabilization

### Step 4: Statistical Analysis

```
"Perform differential protein abundance analysis between treatment groups"
```

**Statistical Methods**:
- **Linear Models**: limma-based analysis
- **Empirical Bayes**: Moderated t-statistics
- **Multiple Testing**: FDR control
- **Effect Size Estimation**: Protein fold changes

### Step 5: Results Interpretation

```
"Identify significantly changed proteins and perform pathway analysis"
```

**Results Analysis**:
- Volcano plots for differential proteins
- Protein interaction networks
- Pathway enrichment analysis
- GO term analysis

### Complete Workflow Example

```bash
# Load MaxQuant data
/read proteinGroups.txt

# Comprehensive analysis
"Analyze this label-free proteomics data: assess data quality,
handle missing values, normalize intensities, and identify proteins
differentially abundant between control and treatment groups"

# Pathway analysis
"Perform pathway enrichment analysis on the significantly changed proteins"
```

## Affinity Proteomics Workflow

### Workflow Overview

**Goal**: Analyze targeted proteomics data from Olink panels or antibody arrays.

**Agent**: Affinity Proteomics Expert specializes in targeted protein analysis.

**Time**: 15-25 minutes for typical panel

### Step 1: Data Loading

```bash
# Load Olink NPX data
/read olink_npx_data.csv

# Load antibody array data
/read antibody_intensities.csv
```

### Step 2: Quality Assessment

```
"Assess the quality of this Olink panel data and check for batch effects"
```

**Quality Metrics**:
- **Coefficient of Variation**: Within and between batch CV
- **Detection Rates**: Protein detectability across samples
- **Control Performance**: Internal control assessment
- **Batch Effects**: Systematic biases between runs

### Step 3: Statistical Analysis

```
"Compare protein levels between disease and healthy control groups"
```

**Analysis Features**:
- **Linear Models**: Account for covariates
- **Batch Correction**: ComBat or similar methods
- **Multiple Testing**: FDR correction
- **Effect Size**: Clinical significance assessment

### Complete Workflow Example

```bash
# Load Olink data
/read olink_cardiovascular_panel.csv

# Comprehensive analysis
"Analyze this Olink cardiovascular panel data: assess quality,
check for batch effects, and identify proteins associated with
cardiovascular disease status"
```

## Multi-Omics Integration Workflow

### Workflow Overview

**Goal**: Integrate multiple data modalities for comprehensive biological insights.

**Agents**: Multiple agents coordinate for multi-modal analysis.

**Time**: 30-60 minutes depending on complexity

### Step 1: Load Multiple Datasets

```bash
# Load different modalities
/read transcriptomics_data.h5ad
/read proteomics_data.csv
/read metabolomics_data.xlsx
```

### Step 2: Data Integration

```
"Integrate the transcriptomics and proteomics data to identify
coordinated changes across molecular layers"
```

**Integration Methods**:
- **Sample Matching**: Align samples across modalities
- **Feature Integration**: Multi-omics factor analysis
- **Pathway Integration**: Combine evidence across layers
- **Network Analysis**: Multi-layer biological networks

### Step 3: Coordinated Analysis

```
"Find genes and proteins that change together in response to treatment"
```

**Results**:
- Correlation analysis across omics layers
- Pathway-level integration
- Multi-omics visualizations
- Integrated statistical models

## Literature Integration Workflow

### Workflow Overview

**Goal**: Integrate literature knowledge with experimental data analysis.

**Agent**: Research Agent and Method Expert coordinate literature integration.

### Step 1: Literature Search

```
"Find papers about single-cell RNA-seq analysis of liver fibrosis"
```

### Step 2: Parameter Extraction

```
"Extract the analysis parameters used in similar studies for my liver single-cell data"
```

### Step 3: Method Application

```
"Apply the methods from Smith et al. 2023 to analyze my data using their parameters"
```

## GEO Database Integration Workflow

### Workflow Overview

**Goal**: Download and analyze public datasets from GEO database.

**Agent**: Data Expert handles GEO integration.

### Step 1: Dataset Discovery

```
"Find GEO datasets related to liver single-cell RNA-seq"
```

### Step 2: Data Download

```
"Download GSE12345 and prepare it for analysis"
```

### Step 3: Comparative Analysis

```
"Compare my results to the downloaded GEO dataset GSE12345"
```

## Workflow Best Practices

### General Principles

1. **Start with Data Quality**: Always assess data quality before analysis
2. **Iterative Approach**: Build analysis step-by-step
3. **Parameter Documentation**: Keep track of analysis parameters
4. **Validation**: Cross-validate results with multiple methods
5. **Visualization**: Generate plots at each major step

### Quality Control Guidelines

1. **Check Data Distribution**: Ensure appropriate data characteristics
2. **Assess Missing Values**: Handle missing data appropriately
3. **Batch Effect Detection**: Look for systematic biases
4. **Outlier Identification**: Handle outliers appropriately
5. **Normalization Validation**: Verify normalization effectiveness

### Statistical Considerations

1. **Multiple Testing Correction**: Always apply appropriate corrections
2. **Effect Size Reporting**: Report both significance and effect size
3. **Confidence Intervals**: Provide uncertainty estimates
4. **Sample Size Assessment**: Ensure adequate statistical power
5. **Assumption Validation**: Check statistical model assumptions

### Reproducibility Guidelines

1. **Parameter Recording**: Document all analysis parameters
2. **Version Control**: Track software and data versions
3. **Random Seeds**: Set seeds for reproducible results
4. **Session Export**: Save complete analysis sessions
5. **Method Documentation**: Record rationale for method choices

## Troubleshooting Common Issues

### Data Loading Problems

**Issue**: File format not recognized
```
# Solution: Check file format and convert if necessary
"Convert this Excel file to a format suitable for analysis"
```

**Issue**: Large file loading slowly
```
# Solution: Use streaming or chunked loading
"Load this large dataset efficiently in chunks"
```

### Analysis Issues

**Issue**: Poor clustering results
```
# Solution: Adjust parameters or try different methods
"The clusters look over-fragmented, can you try different resolution parameters?"
```

**Issue**: No significant results
```
# Solution: Check power and adjust thresholds
"I'm not getting significant results, can you assess the statistical power and suggest improvements?"
```

### Interpretation Challenges

**Issue**: Unexpected biological results
```
# Solution: Literature validation and quality assessment
"These results seem unexpected, can you check the literature and validate the analysis?"
```

**Issue**: Complex statistical output
```
# Solution: Request explanation and visualization
"Can you explain these statistics in simpler terms and create visualizations?"
```

This comprehensive workflow guide covers the major analysis types supported by Lobster AI. Each workflow can be customized based on specific research questions and data characteristics.