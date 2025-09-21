# Single-Cell RNA-seq Analysis Tutorial

This comprehensive tutorial demonstrates how to perform complete single-cell RNA-seq analysis using Lobster AI, from data acquisition to biological interpretation.

## Overview

In this tutorial, you will learn to:
- Download single-cell datasets from GEO using natural language
- Perform quality control and filtering
- Normalize and cluster cells
- Identify cell types and marker genes
- Create publication-ready visualizations
- Export and interpret results

## Prerequisites

- Lobster AI installed and configured (see [Installation Guide](../INSTALLATION.md))
- API keys set up in your `.env` file
- Basic understanding of single-cell RNA-seq concepts

## Tutorial Dataset

We'll use **GSE109564**, a well-characterized single-cell dataset of immune cells from the tumor microenvironment. This dataset contains:
- ~5,000 cells
- ~20,000 genes
- Multiple immune cell types
- Excellent for demonstrating clustering and annotation

## Step 1: Starting Lobster AI

Start the interactive chat interface with the enhanced CLI:

```bash
# Start Lobster AI with Rich CLI interface
lobster chat
```

You'll see the welcome screen with orange branding:

```
🦞 lobster chat

┌─ Welcome ────────────────────────────────────────────────────────────┐
│  🦞 LOBSTER by homara AI                                             │
│  Multi-Agent Bioinformatics Analysis System v2.0                    │
│                                                                      │
│  🧬 Key Tasks:                                                       │
│  • Analyze RNA-seq & genomics data                                  │
│  • Generate visualizations and plots                                │
│  • Extract insights from bioinformatics datasets                    │
│  • Access GEO & literature databases                               │
└──────────────────────────────────────────────────────────────────────┘

🦞 ~/projects ▸
```

## Step 2: Download and Load Data

Use natural language to download the dataset:

```bash
🦞 You: "Download GSE109564 from GEO - it's a single-cell RNA-seq dataset of immune cells from the tumor microenvironment"
```

**Expected Output:**
```
🦞 Lobster: I'll help you download the GSE109564 dataset from GEO...

✓ Found GSE109564: Single-cell RNA sequencing of tumor-infiltrating immune cells
✓ Dataset contains: 5,063 cells × 20,729 genes
✓ Downloaded and processed count matrix
✓ Added sample metadata and experimental design
✓ Stored as modality: 'geo_gse109564'

Dataset successfully loaded! This appears to be a high-quality single-cell dataset
from tumor-infiltrating immune cells. Ready for analysis.
```

**Verify the data was loaded:**
```bash
🦞 You: "/data"
```

This shows current dataset information including cell counts, gene counts, and basic statistics.

## Step 3: Quality Assessment

Assess the quality of the loaded dataset:

```bash
🦞 You: "Assess the quality of the loaded single-cell data and show me basic statistics"
```

**Expected Output:**
```
🦞 Lobster: I'll perform comprehensive quality assessment of your single-cell data...

✓ Quality metrics calculated for 5,063 cells
✓ Mitochondrial gene percentage: mean 8.2%, max 45%
✓ Ribosomal gene percentage: mean 12.4%
✓ Total UMI count: mean 2,847 per cell
✓ Genes detected: mean 1,234 per cell
✓ Generated quality control visualizations

Quality Assessment Summary:
- 287 cells (5.7%) have high mitochondrial content (>25%)
- 134 cells (2.6%) have very low gene counts (<200 genes)
- 89 cells (1.8%) have extremely high UMI counts (potential doublets)
- Overall data quality: EXCELLENT
- Recommended filtering: Remove 510 low-quality cells
```

**Check the generated plots:**
```bash
🦞 You: "/plots"
```

You'll see quality control plots including:
- `qc_violin_plot.html` - Distribution of QC metrics
- `qc_scatter_plot.html` - Gene count vs UMI count correlation
- `mitochondrial_gene_plot.html` - Mitochondrial gene content analysis

## Step 4: Data Preprocessing and Filtering

Filter out low-quality cells and prepare data for analysis:

```bash
🦞 You: "Filter low-quality cells based on the QC metrics, normalize the data, and identify highly variable genes"
```

**Expected Output:**
```
🦞 Lobster: I'll preprocess your single-cell data with quality-based filtering...

✓ Filtered cells: 5,063 → 4,553 cells (510 cells removed)
✓ Filtered genes: 20,729 → 18,345 genes (2,384 low-expression genes removed)
✓ Log1p normalization applied
✓ Highly variable genes identified: 2,000 genes selected
✓ Data scaled and centered for downstream analysis
✓ Stored filtered data as: 'geo_gse109564_filtered_normalized'

Preprocessing complete! Your data is now ready for clustering and dimensional reduction.
Quality-filtered dataset: 4,553 high-quality cells × 18,345 genes
```

## Step 5: Clustering and Dimensionality Reduction

Perform clustering to identify cell populations:

```bash
🦞 You: "Perform PCA, compute neighbors, run Leiden clustering, and generate UMAP visualization"
```

**Expected Output:**
```
🦞 Lobster: I'll perform clustering and dimensional reduction analysis...

✓ Principal component analysis: 50 components computed
✓ Neighbor graph constructed (15 neighbors)
✓ Leiden clustering completed: 12 clusters identified
✓ UMAP embedding generated (2D projection)
✓ Cluster validation metrics calculated
✓ Stored clustered data as: 'geo_gse109564_clustered'

Clustering Results:
- 12 distinct cell clusters identified
- Silhouette score: 0.73 (excellent separation)
- Largest cluster: 847 cells (18.6%)
- Smallest cluster: 89 cells (2.0%)
- Generated interactive UMAP visualization
```

**View the clustering results:**
```bash
🦞 You: "/plots"
```

New visualizations include:
- `umap_clusters.html` - Interactive UMAP with cluster colors
- `cluster_composition.html` - Cell count per cluster
- `pca_variance_plot.html` - Principal component variance explained

## Step 6: Marker Gene Identification

Find genes that characterize each cluster:

```bash
🦞 You: "Find marker genes for each cluster using differential expression analysis and create a heatmap"
```

**Expected Output:**
```
🦞 Lobster: I'll identify marker genes for each cluster...

✓ Differential expression analysis completed (Wilcoxon rank-sum test)
✓ 2,847 significant marker genes identified (FDR < 0.05)
✓ Top markers per cluster: 25 genes selected
✓ Expression heatmap generated
✓ Marker gene table exported
✓ Stored results as: 'geo_gse109564_markers'

Top Marker Genes by Cluster:
- Cluster 0: CD8A, CD8B, GZMB (CD8+ T cells)
- Cluster 1: CD4, IL7R, CCR7 (CD4+ T cells)
- Cluster 2: CD14, LYZ, S100A9 (Monocytes)
- Cluster 3: CD79A, MS4A1, CD19 (B cells)
- Cluster 4: KLRF1, NCR1, NKG7 (NK cells)
[... additional clusters]
```

## Step 7: Cell Type Annotation

Annotate clusters with biological cell types:

```bash
🦞 You: "Based on the marker genes, annotate the clusters with immune cell types. Use standard immune cell markers for annotation."
```

**Expected Output:**
```
🦞 Lobster: I'll annotate clusters based on marker gene expression patterns...

✓ Cell type annotation completed using marker genes
✓ 12 clusters annotated with immune cell types
✓ Annotation confidence scores calculated
✓ Updated UMAP with cell type labels
✓ Stored annotated data as: 'geo_gse109564_annotated'

Cell Type Annotations:
- Cluster 0 (847 cells): CD8+ T cells - Cytotoxic
- Cluster 1 (623 cells): CD4+ T cells - Helper
- Cluster 2 (543 cells): Monocytes - Classical
- Cluster 3 (421 cells): B cells - Mature
- Cluster 4 (298 cells): NK cells - Activated
- Cluster 5 (287 cells): CD8+ T cells - Memory
- Cluster 6 (234 cells): Dendritic cells - Conventional
- Cluster 7 (198 cells): CD4+ T cells - Regulatory
- Cluster 8 (156 cells): Macrophages - M1
- Cluster 9 (123 cells): Plasma cells
- Cluster 10 (89 cells): Dendritic cells - Plasmacytoid
- Cluster 11 (67 cells): Neutrophils

Annotation confidence: 94% (high confidence based on canonical markers)
```

## Step 8: Advanced Visualization

Create comprehensive visualizations:

```bash
🦞 You: "Create a comprehensive visualization dashboard showing UMAP with cell types, marker gene expression, and cluster statistics"
```

**Expected Output:**
```
🦞 Lobster: I'll create a comprehensive visualization dashboard...

✓ Multi-panel UMAP visualization created
✓ Marker gene expression overlays generated
✓ Cell type composition plots created
✓ Interactive dashboard with filtering options
✓ Cluster statistics summary generated

Generated Visualizations:
- umap_celltypes.html: UMAP colored by cell type
- marker_expression_umap.html: Top marker genes overlaid on UMAP
- celltype_composition.html: Cell type proportions and statistics
- expression_heatmap.html: Marker genes × cell types heatmap
- qc_by_celltype.html: Quality metrics by cell type
- comprehensive_dashboard.html: Interactive multi-panel dashboard
```

## Step 9: Export Results

Export your analysis for further use:

```bash
🦞 You: "/export results"
```

**Expected Output:**
```
🦞 Lobster: Exporting analysis results...

✓ Exported annotated data (H5AD format): geo_gse109564_annotated.h5ad
✓ Exported marker genes: marker_genes_table.csv
✓ Exported cell metadata: cell_annotations.csv
✓ Exported cluster statistics: cluster_summary.csv
✓ Exported visualizations: plots/ directory
✓ Analysis provenance: analysis_history.json

Export complete! Files saved to: /path/to/workspace/exports/
```

## Step 10: Biological Interpretation

Ask Lobster to interpret the results:

```bash
🦞 You: "Interpret the biological significance of these immune cell populations in the tumor microenvironment context"
```

**Expected Output:**
```
🦞 Lobster: Based on the single-cell analysis, here's the biological interpretation...

🧬 Tumor Microenvironment Analysis:

Key Findings:
1. **Diverse Immune Infiltration**: 11 distinct immune cell types identified
2. **T Cell Dominance**: CD8+ and CD4+ T cells comprise 32% of total cells
3. **Active Immune Response**: High expression of cytotoxic markers (GZMB, PRF1)
4. **Regulatory Elements**: Presence of Tregs (198 cells) may suppress immunity

Clinical Implications:
- High CD8+ T cell infiltration suggests active anti-tumor response
- Presence of M1 macrophages indicates pro-inflammatory environment
- Treg population may limit therapeutic efficacy
- NK cell activation (NKG7+) supports innate immunity

Recommended Follow-up:
- Pseudobulk analysis for population-level statistics
- Trajectory analysis for T cell activation states
- Ligand-receptor analysis for cell-cell communication
```

## Working with the Results

### Accessing Your Data

```bash
# Check available datasets
🦞 You: "/files"

# Read specific files
🦞 You: "/read marker_genes_table.csv"

# View workspace structure
🦞 You: "/tree"
```

### Advanced Analysis Options

```bash
# Convert to pseudobulk for statistical analysis
🦞 You: "Convert the annotated single-cell data to pseudobulk format for differential expression analysis between cell types"

# Perform pathway analysis
🦞 You: "Run pathway enrichment analysis on the marker genes for each cell type"

# Export for external tools
🦞 You: "Export the data in Seurat format for R analysis"
```

## Troubleshooting Common Issues

### Issue 1: Download Fails
```bash
🦞 You: "The GEO download failed with a timeout error"
```
**Solution**: Check internet connection and try smaller datasets first.

### Issue 2: Poor Clustering
```bash
🦞 You: "The clustering results don't look good - I see poorly separated clusters"
```
**Solution**: Adjust resolution parameter or filtering thresholds.

### Issue 3: Missing Cell Types
```bash
🦞 You: "Some clusters don't have clear cell type annotations"
```
**Solution**: Check additional marker genes or use reference-based annotation.

## Best Practices

1. **Quality Control**: Always inspect QC metrics before filtering
2. **Parameter Testing**: Try different clustering resolutions for optimal results
3. **Marker Validation**: Verify cell type annotations with literature
4. **Visualization**: Use interactive plots to explore data thoroughly
5. **Documentation**: Export analysis history for reproducibility

## Next Steps

After completing this tutorial, consider:

1. **[Bulk RNA-seq Tutorial](24-tutorial-bulk-rnaseq.md)** - Convert to pseudobulk and perform population-level analysis
2. **[Proteomics Tutorial](25-tutorial-proteomics.md)** - Integrate with proteomics data
3. **[Advanced Analysis](27-examples-cookbook.md)** - Trajectory analysis, cell-cell communication
4. **[Custom Workflows](26-tutorial-custom-agent.md)** - Create specialized analysis agents

## Summary

You have successfully:
- ✅ Downloaded and loaded a single-cell dataset from GEO
- ✅ Performed comprehensive quality control
- ✅ Filtered and normalized the data
- ✅ Identified 12 distinct immune cell populations
- ✅ Annotated clusters with biological cell types
- ✅ Generated publication-ready visualizations
- ✅ Exported results for further analysis
- ✅ Interpreted biological significance

This complete workflow demonstrates Lobster AI's power for single-cell RNA-seq analysis using natural language interactions and professional-grade bioinformatics algorithms.