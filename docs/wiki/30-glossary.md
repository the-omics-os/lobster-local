# Glossary

This comprehensive glossary defines bioinformatics, technical, and Lobster AI-specific terms used throughout the documentation and platform.

## A

**Affinity Proteomics**
Protein analysis method using antibody-based assays for targeted protein detection and quantification. Examples include Olink panels and antibody arrays. Typically has lower missing values (<30%) compared to mass spectrometry.

**Agent**
Specialized AI component in Lobster AI that handles specific analysis domains (e.g., Single-cell Expert, Proteomics Expert). Agents use natural language understanding to execute appropriate tools and workflows.

**Agent Registry**
Centralized system in Lobster AI that manages all available agents, their configurations, and handoff capabilities. Located in `lobster/config/agent_registry.py`.

**AnnData**
Annotated data structure used in Python bioinformatics for storing high-dimensional biological data. Contains expression matrix (X), observations (obs), variables (var), and additional metadata.

**Annotation**
Process of assigning biological meaning to data elements, such as identifying cell types in single-cell data or functional categories for genes.

---

## B

**Batch Effect**
Technical variation in data caused by non-biological factors such as processing date, equipment, or experimental conditions. Must be corrected to avoid confounding biological signals.

**Benjamini-Hochberg (BH)**
Multiple testing correction method that controls the False Discovery Rate (FDR). Used to adjust p-values when testing many hypotheses simultaneously.

**Bulk RNA-seq**
RNA sequencing of entire tissue samples or cell populations, providing average expression across all cells. Contrasts with single-cell RNA-seq which measures individual cells.

---

## C

**Cell Type Annotation**
Process of identifying and labeling cell populations in single-cell data based on marker gene expression patterns and biological knowledge.

**Clustering**
Computational method to group similar observations (cells, samples, genes) based on expression patterns. Common algorithms include Leiden, Louvain, and k-means.

**Coefficient of Variation (CV)**
Statistical measure of relative variability, calculated as standard deviation divided by mean. Used to assess technical reproducibility in proteomics.

**ComBat**
Batch correction method that removes batch effects while preserving biological variation. Commonly used in transcriptomics analysis.

**Count Matrix**
Two-dimensional data structure with features (genes/proteins) as rows and samples (cells/samples) as columns, containing quantified expression values.

---

## D

**DDA (Data-Dependent Acquisition)**
Mass spectrometry method where the instrument selects the most abundant ions for fragmentation and identification. Traditional approach for shotgun proteomics.

**DIA (Data-Independent Acquisition)**
Mass spectrometry method that systematically fragments all ions in predefined windows. Often provides more comprehensive and reproducible protein identification.

**DataManagerV2**
Core data management system in Lobster AI that handles multiple modalities, provenance tracking, and analysis history. Supports various data backends and formats.

**Differential Expression (DE)**
Statistical analysis to identify genes or proteins with significantly different expression levels between conditions or cell types.

**Design Matrix**
Mathematical representation of experimental design used in statistical models. Encodes relationships between samples and experimental factors.

**Doublet**
Artifact in single-cell RNA-seq where two cells are captured and sequenced together, appearing as a single cell with unusually high gene counts.

---

## E

**Empirical Bayes**
Statistical method that improves parameter estimation by borrowing information across features. Used in limma and other differential expression tools.

**Enrichment Analysis**
Statistical method to identify biological pathways or processes that are over-represented in a gene or protein list.

---

## F

**False Discovery Rate (FDR)**
Expected proportion of false positives among rejected hypotheses. Commonly controlled at 5% (0.05) in genomics studies.

**Feature Selection**
Process of selecting the most informative variables (genes/proteins) for analysis, such as highly variable genes in single-cell RNA-seq.

**Fold Change**
Ratio of expression between two conditions. Often expressed as log2 fold change where 1 represents 2-fold increase and -1 represents 2-fold decrease.

---

## G

**GEO (Gene Expression Omnibus)**
NCBI database containing high-throughput genomics data. Lobster AI can directly download and analyze GEO datasets using GSE accession numbers.

**GSEA (Gene Set Enrichment Analysis)**
Method for determining whether a defined set of genes shows statistically significant differences between biological states.

---

## H

**H5AD**
HDF5-based file format for storing AnnData objects. Standard format for single-cell data that preserves metadata and analysis results.

**Handoff**
Process in Lobster AI where the supervisor agent transfers a query to a specialized agent based on the analysis type required.

**Highly Variable Genes (HVG)**
Genes showing high variation across cells or samples, often selected for downstream analysis as they contain the most biological information.

---

## I

**Imputation**
Statistical method to estimate missing values in datasets. Particularly important in proteomics where 30-70% of values may be missing.

**Integration**
Process of combining multiple datasets or omics layers to enable joint analysis. Includes batch correction and multi-omics integration.

---

## L

**Label-Free Quantification (LFQ)**
Mass spectrometry approach that quantifies proteins without chemical labeling, using peptide intensity measurements.

**Leiden Clustering**
Community detection algorithm for clustering that often performs better than Louvain clustering for biological data.

**Limma**
R package for analyzing gene expression data using linear models and empirical Bayes methods. Popular for differential expression analysis.

**Log Transformation**
Mathematical transformation that converts multiplicative relationships to additive ones. Common preprocessing step in genomics (log2 or natural log).

---

## M

**MA Plot**
Scatter plot showing log fold change (M) vs average expression (A). Used to visualize differential expression results and identify bias.

**Marker Genes**
Genes specifically or highly expressed in particular cell types or conditions. Used for cell type identification and validation.

**MaxQuant**
Software platform for analyzing mass spectrometry proteomics data. Provides protein identification and quantification from raw MS data.

**Missing at Random (MAR)**
Missing data mechanism where missingness depends on observed data but not on the missing values themselves.

**Missing Not at Random (MNAR)**
Missing data mechanism where missingness depends on the unobserved values. Common in proteomics for low-abundance proteins.

**Modality**
In Lobster AI, a named dataset representing a specific omics measurement (e.g., "single_cell_rna", "ms_proteomics"). Managed by DataManagerV2.

**Moran's I**
Statistical measure of spatial autocorrelation used in spatial omics to identify spatially variable genes or proteins.

**MuData**
Data structure for storing and analyzing multi-modal omics data. Extends AnnData to handle multiple measurement types simultaneously.

---

## N

**Normalization**
Process of adjusting data to remove technical variation and enable comparison between samples. Methods include library size normalization, quantile normalization, and TMM.

**NPX (Normalized Protein eXpression)**
Log2-transformed and normalized protein abundance values used in Olink affinity proteomics assays.

---

## O

**Olink**
Commercial platform for targeted proteomics using proximity extension assays (PEA). Provides high-quality protein measurements with low missing values.

**Ontology**
Structured vocabulary defining relationships between biological concepts. Gene Ontology (GO) is widely used for functional annotation.

---

## P

**PCA (Principal Component Analysis)**
Dimensionality reduction technique that identifies the directions of maximum variance in data. Used for visualization and quality control.

**Pseudobulk**
Method to aggregate single-cell data to sample-level summaries, enabling population-level statistical analysis with established bulk RNA-seq methods.

**Pseudotime**
Computational measure of cell progression along a biological process, such as differentiation or cell cycle, based on expression similarity.

**pyDESeq2**
Pure Python implementation of the DESeq2 algorithm for differential expression analysis of RNA-seq data.

---

## Q

**Quality Control (QC)**
Assessment of data quality including technical metrics, batch effects, and sample integrity. Essential first step in any omics analysis.

**Quantile Normalization**
Normalization method that makes the distribution of values identical across samples by matching quantiles.

---

## R

**RNA Velocity**
Method to predict future cell states by analyzing spliced and unspliced mRNA ratios, revealing cell differentiation dynamics.

---

## S

**scanpy**
Python package for analyzing single-cell gene expression data. Provides comprehensive tools for preprocessing, visualization, and analysis.

**Service**
In Lobster AI architecture, stateless components that perform specific analysis tasks. Services receive AnnData objects and return results with statistics.

**Silhouette Score**
Measure of clustering quality that quantifies how similar objects are within clusters compared to other clusters.

**Single-cell RNA-seq (scRNA-seq)**
Sequencing technology that measures gene expression in individual cells, revealing cellular heterogeneity and rare cell types.

**Spatial Omics**
Analysis of molecular data with preserved spatial context, such as Visium spatial transcriptomics or imaging mass cytometry.

---

## T

**t-SNE**
Non-linear dimensionality reduction method that preserves local structure. Often used for visualizing high-dimensional single-cell data.

**TMM (Trimmed Mean of M-values)**
Normalization method commonly used in proteomics that assumes most proteins are not differentially expressed.

**Trajectory Analysis**
Computational method to order cells along developmental or temporal progressions, revealing biological processes over time.

---

## U

**UMAP (Uniform Manifold Approximation and Projection)**
Dimensionality reduction and visualization technique that preserves both local and global data structure. Popular for single-cell visualization.

**UMI (Unique Molecular Identifier)**
Short DNA sequence used to identify and count individual mRNA molecules, reducing PCR bias in single-cell sequencing.

---

## V

**Variance Stabilization**
Transformation that makes variance approximately constant across the range of data values. Important for statistical analysis assumptions.

**Visium**
10x Genomics platform for spatial gene expression analysis that measures transcriptomes with preserved spatial coordinates.

**Volcano Plot**
Scatter plot showing fold change vs statistical significance (-log10 p-value) for differential expression results.

---

## W

**WGCNA (Weighted Gene Co-expression Network Analysis)**
Method for constructing gene co-expression networks and identifying modules of highly correlated genes.

---

## Technical Terms (Lobster AI Specific)

**Agent Factory Function**
Python function that creates and configures an agent instance. Specified in the agent registry for dynamic agent loading.

**CLI (Command Line Interface)**
Lobster's Rich-enhanced terminal interface with orange branding, autocomplete, and real-time monitoring capabilities.

**Cloud Mode**
Operating mode where analyses are processed on cloud infrastructure. Activated by setting `LOBSTER_CLOUD_KEY` environment variable.

**Handoff Tool**
Specialized tool that allows agents to transfer tasks to other agents based on analysis requirements.

**LangGraph**
Framework used by Lobster AI for creating multi-agent workflows with state management and tool integration.

**Local Mode**
Default operating mode where all analyses are processed on the user's local machine.

**Orange Theming**
Lobster AI's signature color scheme (#e45c47) used throughout the CLI interface and visualizations.

**Provenance Tracking**
W3C-PROV compliant system in Lobster AI that records complete analysis history for reproducibility.

**Rich CLI**
Enhanced command-line interface using the Rich Python library, providing advanced formatting, progress bars, and interactive elements.

**Service Pattern**
Lobster AI's architectural pattern where stateless services handle analysis logic, returning both processed data and statistics.

**Tool**
Function available to agents for performing specific tasks. Tools follow standardized patterns for data validation, service calls, and result storage.

**Workspace**
Directory structure used by Lobster AI to organize data, results, and analysis history. Default location is `.lobster_workspace/`.

---

## Statistical Terms

**Adjusted Rand Score**
Measure of clustering performance that compares clustering results to known ground truth, corrected for chance.

**Cohen's d**
Standardized measure of effect size representing the difference between groups in terms of standard deviations.

**Empirical Bayes Moderation**
Statistical technique that improves variance estimates by borrowing information across genes, leading to more stable results.

**Hypergeometric Test**
Statistical test used in enrichment analysis to determine if a pathway or gene set is over-represented in a list of interesting genes.

**Multiple Testing Correction**
Statistical adjustment applied when testing multiple hypotheses simultaneously to control the overall error rate.

**Pearson Correlation**
Measure of linear correlation between two variables, ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).

**Spearman Correlation**
Rank-based correlation measure that captures monotonic relationships, more robust to outliers than Pearson correlation.

**Wilcoxon Rank-Sum Test**
Non-parametric statistical test used to compare expression levels between groups. Default method for single-cell differential expression.

---

This glossary provides definitions for terms commonly encountered in Lobster AI documentation and bioinformatics analysis. For additional technical details, refer to the specific tutorial documents and API documentation.