# Proteomics Analysis Tutorial

This comprehensive tutorial demonstrates how to analyze both mass spectrometry and affinity proteomics data using Lobster AI's specialized proteomics platform with missing value handling, normalization strategies, and publication-ready visualizations.

## Overview

In this tutorial, you will learn to:
- Analyze mass spectrometry proteomics data (DDA/DIA workflows)
- Process affinity proteomics data (Olink panels, antibody arrays)
- Handle missing value patterns (MNAR vs MCAR)
- Perform proteomics-specific quality control
- Conduct differential protein expression analysis
- Generate proteomics visualization dashboards
- Integrate proteomics with transcriptomics data

## Prerequisites

- Lobster AI installed and configured (see [Installation Guide](../INSTALLATION.md))
- API keys set up in your `.env` file
- Understanding of proteomics data characteristics
- Familiarity with mass spectrometry or affinity proteomics concepts

## Part 1: Mass Spectrometry Proteomics Analysis

### Tutorial Dataset: DIA-MS Proteomics

We'll use a DIA-MS (Data-Independent Acquisition) dataset with:
- 24 samples (12 treatment, 12 control)
- 2,847 proteins quantified
- Typical 42% missing values
- MaxQuant output format
- Peptide-to-protein aggregation

### Step 1: Starting MS Proteomics Analysis

```bash
# Start Lobster AI
lobster chat
```

```
🦞 LOBSTER by homara AI
Multi-Agent Bioinformatics Analysis System v2.0

Ready for comprehensive proteomics analysis!
```

### Step 2: Load MS Proteomics Data

Load your MaxQuant proteinGroups.txt file:

```bash
🦞 You: "Load the mass spectrometry proteomics data from proteinGroups.txt. This is MaxQuant output from a DIA experiment comparing treatment vs control samples."
```

**Expected Output:**
```
🦞 Lobster: I'll load your MS proteomics data and assess its characteristics...

✓ Loaded MaxQuant proteinGroups.txt file
✓ Extracted LFQ intensity values for 24 samples
✓ Protein annotations parsed (gene names, descriptions)
✓ Detected 2,847 protein groups
✓ Missing value assessment: 42.3% missing (typical for MS data)
✓ Sample metadata inferred from column headers
✓ Stored as modality: 'ms_proteomics_lfq'

MS Proteomics Dataset Summary:
- Samples: 24 (treatment n=12, control n=12)
- Proteins: 2,847 protein groups
- Missing values: 42.3% (1,203 complete proteins)
- Dynamic range: 6.2 orders of magnitude
- Data type: Label-free quantification (LFQ)
- Platform: DIA-MS (MaxQuant processed)
```

### Step 3: MS Proteomics Quality Control

Assess data quality and missing value patterns:

```bash
🦞 You: "Perform comprehensive quality control including missing value pattern analysis, sample correlation assessment, and batch effect detection"
```

**Expected Output:**
```
🦞 Lobster: I'll perform MS proteomics-specific quality control analysis...

✓ Missing value pattern analysis completed
✓ MNAR vs MCAR classification performed
✓ Sample correlation matrix calculated
✓ Coefficient of variation analysis finished
✓ Batch effect assessment completed
✓ Protein abundance distribution analyzed
✓ Generated comprehensive QC dashboard

Quality Control Results:

Missing Value Analysis:
- Missing at Random (MAR): 18.7% of missing values
- Missing Not at Random (MNAR): 81.3% of missing values
- Low abundance bias confirmed (typical for MS data)
- Missing values concentrated in low-abundance proteins

Sample Quality Metrics:
- Protein identifications per sample: 1,456 - 2,203
- Sample correlations: 0.78 - 0.94 (good)
- Coefficient of variation (median): 23.4% (acceptable)
- Outlier samples: 0 (all samples passed QC)

Batch Effects:
- Batch effect detected: MINIMAL (3.2% variance explained)
- No systematic bias observed
- No correction needed
```

### Step 4: MS Data Preprocessing

Preprocess with appropriate normalization and filtering:

```bash
🦞 You: "Filter low-quality proteins, apply log2 transformation, perform median normalization, and handle missing values using appropriate imputation strategies"
```

**Expected Output:**
```
🦞 Lobster: I'll preprocess your MS proteomics data with platform-appropriate methods...

✓ Protein filtering applied (valid values in ≥50% of samples)
✓ Log2 transformation performed
✓ Median normalization completed (sample-wise centering)
✓ Missing value imputation: MNAR (left-shifted normal distribution)
✓ Missing value imputation: MAR (k-nearest neighbors)
✓ Data scaling and standardization applied
✓ Stored processed data as: 'ms_proteomics_lfq_processed'

Preprocessing Results:
- Proteins retained: 2,847 → 1,923 proteins (67.5%)
- Missing values before imputation: 42.3%
- Missing values after imputation: 0% (all imputed)
- Normalization method: Median centering + log2
- Imputation strategy: Hybrid (MNAR + MAR)
- Quality score: EXCELLENT (ready for analysis)
```

### Step 5: MS Differential Protein Expression

Perform statistical testing for differential expression:

```bash
🦞 You: "Run differential protein expression analysis comparing treatment vs control using linear models with empirical Bayes moderation and FDR correction"
```

**Expected Output:**
```
🦞 Lobster: I'll perform differential protein expression analysis using statistical models optimized for proteomics data...

✓ Linear model fitted with empirical Bayes moderation
✓ Statistical testing completed for treatment vs control
✓ Multiple testing correction applied (Benjamini-Hochberg FDR)
✓ Effect size calculations completed
✓ Confidence intervals estimated
✓ Protein annotation enriched
✓ Stored results as: 'ms_proteomics_lfq_differential_results'

Differential Expression Results:

Treatment vs Control:
- Total proteins tested: 1,923 proteins
- Significantly DE proteins (FDR < 0.05): 347 proteins (18.0%)
- Up-regulated: 198 proteins (log2FC > 0)
- Down-regulated: 149 proteins (log2FC < 0)
- Strong changes (|log2FC| > 1): 89 proteins (4.6%)

Top Regulated Proteins:
- Most up-regulated: STAT1 (log2FC = 2.34, FDR = 1.2e-12)
- Most down-regulated: MYC (log2FC = -1.87, FDR = 3.4e-9)
- Highest significance: IFNG (log2FC = 1.45, FDR = 2.1e-15)

Statistical Summary:
- Mean |log2FC|: 0.43
- Median p-value: 0.12
- FDR threshold met: 347/1923 proteins (18.0%)
```

### Step 6: MS Proteomics Visualization

Generate comprehensive visualizations:

```bash
🦞 You: "Create proteomics visualization dashboard including volcano plots, missing value heatmaps, protein correlation networks, and QC summary panels"
```

**Expected Output:**
```
🦞 Lobster: I'll create a comprehensive proteomics visualization suite...

✓ Volcano plot generated with statistical significance overlay
✓ Missing value pattern heatmap created
✓ Protein correlation network constructed
✓ PCA analysis with sample groupings completed
✓ Intensity distribution plots generated
✓ QC metrics dashboard assembled
✓ Interactive exploration panels created

Generated MS Proteomics Visualizations:
- volcano_plot_ms.html: Interactive volcano plot with protein annotations
- missing_value_heatmap.html: Missing value patterns across samples
- protein_correlation_network.html: NetworkX protein-protein correlations
- pca_samples_ms.html: PCA colored by treatment groups
- intensity_distributions.html: Protein abundance distributions
- cv_analysis.html: Coefficient of variation analysis
- qc_dashboard_ms.html: Comprehensive quality control summary
- comprehensive_proteomics_dashboard.html: Multi-panel interactive dashboard
```

## Part 2: Affinity Proteomics Analysis

### Tutorial Dataset: Olink Panel

Now we'll analyze an Olink targeted proteomics panel with:
- 96 samples (4 conditions × 24 samples each)
- 384 proteins (Olink Explore panel)
- Low missing values (18%)
- NPX (Normalized Protein eXpression) values
- High technical reproducibility

### Step 7: Load Affinity Proteomics Data

```bash
🦞 You: "Load the Olink proteomics data from the NPX file. This is targeted proteomics with 384 proteins across 96 samples from 4 different conditions."
```

**Expected Output:**
```
🦞 Lobster: I'll load your Olink affinity proteomics data...

✓ Loaded Olink NPX data file
✓ Extracted normalized protein expression values
✓ Sample metadata parsed from headers
✓ Protein annotations imported (UniProt IDs, gene names)
✓ Quality flags assessed for all measurements
✓ Stored as modality: 'olink_affinity_proteomics'

Affinity Proteomics Dataset Summary:
- Samples: 96 (4 conditions × 24 samples each)
- Proteins: 384 targeted proteins (Olink Explore panel)
- Missing values: 18.2% (excellent for affinity proteomics)
- Data type: NPX (Normalized Protein eXpression)
- Platform: Olink Proximity Extension Assay (PEA)
- Technical CV: 8.7% median (excellent reproducibility)
```

### Step 8: Affinity Proteomics Quality Assessment

```bash
🦞 You: "Assess the quality of the Olink data including antibody performance metrics, coefficient of variation analysis, and cross-sample correlations"
```

**Expected Output:**
```
🦞 Lobster: I'll perform affinity proteomics-specific quality assessment...

✓ Antibody validation metrics calculated
✓ Technical reproducibility assessment completed
✓ Cross-sample correlation analysis performed
✓ Detection frequency analysis finished
✓ Dynamic range assessment completed
✓ Quality flag validation performed

Affinity Proteomics Quality Results:

Antibody Performance:
- High-performance antibodies: 312/384 (81.3%)
- Medium-performance antibodies: 58/384 (15.1%)
- Low-performance antibodies: 14/384 (3.6%)
- Failed quality checks: 0/384 (0%)

Technical Metrics:
- Median CV within replicates: 8.7% (excellent)
- Detection frequency > 50%: 367/384 proteins (95.6%)
- Dynamic range: 4.8 log2 units (good)
- Cross-sample correlation: 0.89 - 0.97 (excellent)

Sample Quality:
- Outlier samples: 2/96 (2.1%) - flagged for review
- Missing data per sample: 12.3% - 24.1%
- Overall data quality: EXCELLENT
```

### Step 9: Affinity Proteomics Statistical Analysis

```bash
🦞 You: "Perform differential expression analysis across the 4 conditions using ANOVA with post-hoc pairwise comparisons and multiple testing correction"
```

**Expected Output:**
```
🦞 Lobster: I'll perform multi-group differential analysis for your Olink data...

✓ One-way ANOVA performed across 4 conditions
✓ Post-hoc pairwise comparisons completed (6 contrasts)
✓ Multiple testing correction applied (FDR < 0.05)
✓ Effect size calculations finished
✓ Pathway enrichment analysis performed
✓ Stored results as: 'olink_affinity_proteomics_anova_results'

Multi-Group Analysis Results:

ANOVA Results (4 conditions):
- Total proteins tested: 384 proteins
- Significantly different (FDR < 0.05): 156 proteins (40.6%)
- Strong effect size (eta² > 0.1): 89 proteins (23.2%)
- Very strong effect (eta² > 0.25): 34 proteins (8.9%)

Pairwise Comparisons (selected):
Condition A vs B: 67 DE proteins
Condition A vs C: 89 DE proteins
Condition A vs D: 124 DE proteins
Condition B vs C: 45 DE proteins
Condition B vs D: 78 DE proteins
Condition C vs D: 56 DE proteins

Top Regulated Proteins:
- Highest F-statistic: IL6 (F = 45.7, FDR = 2.1e-18)
- Largest effect size: TNF (eta² = 0.34, FDR = 5.7e-12)
- Most consistent: CRP (significant in 5/6 comparisons)
```

### Step 10: Affinity Proteomics Visualization

```bash
🦞 You: "Create comprehensive visualizations for the Olink data including multi-group comparisons, antibody performance metrics, and pathway analysis results"
```

**Expected Output:**
```
🦞 Lobster: I'll create specialized affinity proteomics visualizations...

✓ Multi-group volcano plots generated
✓ Antibody performance dashboard created
✓ Protein expression profiles across conditions plotted
✓ Pathway enrichment network visualization created
✓ Sample clustering with condition annotations completed
✓ Technical reproducibility plots generated

Generated Affinity Proteomics Visualizations:
- multigroup_volcano.html: Volcano plots for all pairwise comparisons
- antibody_performance_dashboard.html: CV analysis and detection frequencies
- protein_expression_profiles.html: Expression patterns across conditions
- pathway_network.html: Enriched pathways as interactive networks
- sample_clustering_olink.html: Hierarchical clustering with annotations
- technical_reproducibility.html: CV analysis and quality metrics
- comprehensive_olink_dashboard.html: Complete analysis dashboard
```

## Part 3: Multi-Omics Integration

### Step 11: Proteomics-Transcriptomics Integration

Integrate proteomics data with RNA-seq results:

```bash
🦞 You: "Integrate the MS proteomics results with bulk RNA-seq data from the same samples to identify concordant changes at mRNA and protein levels"
```

**Expected Output:**
```
🦞 Lobster: I'll integrate your proteomics and transcriptomics data for multi-omics analysis...

✓ Gene-protein mapping established (UniProt ↔ Ensembl)
✓ Sample correspondence verified across platforms
✓ Correlation analysis between mRNA and protein levels performed
✓ Concordance analysis for differential expression completed
✓ Multi-omics pathway analysis performed
✓ Integrated visualization dashboard created

Multi-Omics Integration Results:

Gene-Protein Correspondence:
- Matched genes/proteins: 1,456 pairs
- RNA-protein correlation (overall): r = 0.42 (typical)
- High correlation pairs (r > 0.7): 287 genes/proteins (19.7%)
- Low correlation pairs (r < 0.2): 445 genes/proteins (30.5%)

Differential Expression Concordance:
- Both RNA and protein DE (same direction): 89 genes (6.1%)
- RNA DE only: 1,234 genes (84.8%)
- Protein DE only: 258 proteins (17.7%)
- Opposite direction changes: 23 genes (1.6%)

Key Biological Insights:
- Post-translational regulation evident in 258 protein-only changes
- Transcriptional control dominant for 1,234 RNA-only changes
- Concordant changes likely represent core regulatory programs
- Opposite changes suggest negative feedback mechanisms
```

### Step 12: Pathway-Level Integration

```bash
🦞 You: "Perform pathway-level integration analysis to identify biological processes that are consistently altered across both transcriptomics and proteomics data"
```

**Expected Output:**
```
🦞 Lobster: I'll perform pathway-level multi-omics integration...

✓ Gene set enrichment analysis performed for both datasets
✓ Pathway overlap analysis completed
✓ Multi-omics pathway scoring calculated
✓ Concordance metrics computed for biological processes
✓ Integrated pathway networks constructed

Pathway-Level Integration Results:

Consistently Altered Pathways (RNA + Protein):
1. Immune system process - RNA FDR: 2.1e-15, Protein FDR: 5.7e-8
2. Inflammatory response - RNA FDR: 1.2e-12, Protein FDR: 3.4e-6
3. Cytokine signaling - RNA FDR: 6.8e-11, Protein FDR: 1.9e-5

RNA-Specific Pathways:
1. Transcription regulation - RNA FDR: 4.5e-18
2. Cell cycle - RNA FDR: 2.3e-14
3. DNA repair - RNA FDR: 8.9e-12

Protein-Specific Pathways:
1. Protein folding - Protein FDR: 1.2e-9
2. Post-translational modification - Protein FDR: 5.6e-7
3. Protein transport - Protein FDR: 2.8e-6

Integration Score: 0.67 (good concordance at pathway level)
```

## Step 13: Export and Reporting

Export comprehensive proteomics analysis results:

```bash
🦞 You: "/export proteomics-analysis"
```

**Expected Output:**
```
🦞 Lobster: Exporting comprehensive proteomics analysis results...

✓ MS proteomics results exported (differential expression tables)
✓ Olink affinity proteomics results exported (ANOVA results)
✓ Multi-omics integration results exported
✓ Quality control metrics exported
✓ Protein annotations and metadata exported
✓ Statistical summaries generated
✓ Publication-ready figures exported

Exported Proteomics Files:
📊 MS Proteomics Data:
- ms_differential_results.csv: Differential protein expression results
- ms_normalized_data.csv: Processed protein intensities
- ms_qc_metrics.csv: Quality control summaries

📊 Affinity Proteomics Data:
- olink_anova_results.csv: Multi-group statistical results
- olink_pairwise_comparisons.csv: All pairwise contrast results
- olink_antibody_performance.csv: Antibody validation metrics

📊 Multi-Omics Integration:
- multiomics_correlation_results.csv: RNA-protein correlations
- multiomics_pathway_analysis.csv: Integrated pathway results
- gene_protein_concordance.csv: Differential expression concordance

🎨 Visualizations:
- figures/ms_proteomics/: Mass spectrometry visualizations
- figures/affinity_proteomics/: Olink analysis plots
- figures/multiomics/: Integration analysis plots
- figures/publication/: High-resolution publication figures

📋 Analysis Documentation:
- proteomics_analysis_summary.txt: Complete analysis summary
- methods_proteomics.txt: Methods description for publication
- analysis_parameters.json: All analysis settings
```

## Advanced Proteomics Analysis Options

### Time-Series Proteomics

```bash
🦞 You: "Analyze temporal changes in protein expression over 6 time points using spline regression and cluster analysis"
```

### Spatial Proteomics

```bash
🦞 You: "Analyze spatial proteomics data to identify subcellular localization changes upon treatment"
```

### Protein Complex Analysis

```bash
🦞 You: "Identify protein complexes and analyze how complex compositions change between conditions"
```

### Cross-Platform Validation

```bash
🦞 You: "Validate MS proteomics findings using targeted analysis with selected reaction monitoring (SRM)"
```

## Troubleshooting Common Issues

### Issue 1: High Missing Values in MS Data
```bash
🦞 You: "My MS data has >60% missing values - is this normal?"
```
**Solution**: Normal for MS data. Use appropriate MNAR imputation and consider protein filtering strategies.

### Issue 2: Poor Sample Correlations
```bash
🦞 You: "Sample correlations are below 0.7 - what should I check?"
```
**Solution**: Check for batch effects, sample degradation, or technical issues. Consider outlier removal.

### Issue 3: Low Protein Coverage
```bash
🦞 You: "I'm only detecting 500 proteins in my MS experiment"
```
**Solution**: Check sample preparation, LC-MS settings, and database search parameters.

### Issue 4: Olink QC Failures
```bash
🦞 You: "Several Olink samples failed quality control"
```
**Solution**: Check sample integrity, pipetting accuracy, and plate effects.

## Best Practices

### MS Proteomics
1. **Missing Values**: Use MNAR-appropriate imputation for low-abundance proteins
2. **Normalization**: Apply median centering before statistical testing
3. **Filtering**: Remove proteins with <50% valid values across samples
4. **Statistics**: Use empirical Bayes methods for small sample sizes

### Affinity Proteomics
1. **Quality Control**: Monitor CV values and detection frequencies
2. **Antibody Validation**: Check antibody performance metrics regularly
3. **Technical Replicates**: Include technical replicates for CV assessment
4. **Cross-Platform**: Validate findings across multiple panels when possible

### Multi-Omics Integration
1. **Sample Matching**: Ensure proper sample correspondence across platforms
2. **Timing**: Account for different extraction and processing times
3. **Normalization**: Use platform-appropriate normalization methods
4. **Interpretation**: Consider biological vs technical correlations

## Next Steps

After completing this tutorial, explore:

1. **[Single-Cell Tutorial](23-tutorial-single-cell.md)** - Single-cell proteomics analysis
2. **[Bulk RNA-seq Tutorial](24-tutorial-bulk-rnaseq.md)** - Transcriptomics integration
3. **[Advanced Analysis Cookbook](27-examples-cookbook.md)** - Specialized proteomics workflows
4. **[Custom Agent Tutorial](26-tutorial-custom-agent.md)** - Create proteomics-specific agents

## Summary

You have successfully:
- ✅ Analyzed mass spectrometry proteomics data with proper missing value handling
- ✅ Processed affinity proteomics data with antibody performance assessment
- ✅ Performed differential protein expression analysis with appropriate statistics
- ✅ Created comprehensive proteomics visualization dashboards
- ✅ Integrated proteomics with transcriptomics data for multi-omics insights
- ✅ Exported publication-ready results and figures
- ✅ Learned platform-specific best practices and troubleshooting

This comprehensive workflow demonstrates Lobster AI's advanced proteomics analysis capabilities, handling the unique challenges of protein-level data analysis with professional-grade algorithms and specialized visualizations.