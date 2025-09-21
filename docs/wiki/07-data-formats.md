# Data Formats Guide

## Overview

Lobster AI supports a wide range of biological data formats for different omics types. This guide provides detailed specifications for supported input and output formats, including format conversion capabilities and best practices.

## Supported Input Formats

### Single-Cell RNA-seq Formats

#### H5AD (AnnData HDF5)
**Description**: Standard format for single-cell data, used by scanpy and other Python tools.

**File Extension**: `.h5ad`

**Structure**:
```
AnnData object with:
- X: Expression matrix (cells × genes)
- obs: Cell metadata (cell barcodes, QC metrics, clusters)
- var: Gene metadata (gene symbols, chromosome, biotype)
- obsm: Multi-dimensional cell annotations (PCA, UMAP coordinates)
- varm: Multi-dimensional gene annotations
- layers: Additional expression matrices (raw counts, normalized)
- uns: Unstructured annotations (parameters, plots)
```

**Example Loading**:
```bash
/read single_cell_data.h5ad
```

**Advantages**:
- Efficient storage with compression
- Preserves all analysis metadata
- Native format for scanpy workflows
- Supports both sparse and dense matrices

#### 10X Genomics Formats

**10X HDF5 Format**
- **File Extension**: `.h5`
- **Structure**: HDF5 file with matrix, features, and barcodes
- **Loading**: `/read filtered_feature_bc_matrix.h5`

**10X MTX Format**
- **Files**: `matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`
- **Structure**: Market Matrix format with separate metadata files
- **Loading**: `/read /path/to/filtered_feature_bc_matrix/`

**10X CSV Format**
- **Files**: CSV/TSV files with gene expression matrix
- **Structure**: Genes as rows, cells as columns (or transposed)

#### CSV/TSV Formats
**Structure Options**:
1. **Genes as rows, cells as columns**:
   ```
   gene_id,cell_1,cell_2,cell_3,...
   ENSG00000001,10,5,0,...
   ENSG00000002,0,15,3,...
   ```

2. **Cells as rows, genes as columns**:
   ```
   cell_id,ENSG00000001,ENSG00000002,...
   cell_1,10,0,...
   cell_2,5,15,...
   ```

**Loading**:
```bash
/read expression_matrix.csv
/read expression_matrix.tsv
```

**Auto-detection**: Lobster AI automatically detects orientation and format.

#### Excel Formats
**File Extensions**: `.xlsx`, `.xls`

**Structure**: Expression matrix with optional metadata sheets

**Example Loading**:
```bash
/read single_cell_data.xlsx
```

### Bulk RNA-seq Formats

#### Count Matrices

**CSV/TSV Count Matrix**
```
gene_id,sample_1,sample_2,sample_3,sample_4
ENSG00000001,150,200,175,220
ENSG00000002,0,5,2,8
ENSG00000003,1200,1500,1300,1800
```

**Requirements**:
- Raw or normalized counts
- Gene identifiers (Ensembl, Symbol, etc.)
- Sample identifiers as column headers

#### DESeq2 Format
**Structure**: Compatible with DESeq2 input requirements
- Integer count values (for raw counts)
- Gene metadata optional
- Sample metadata in separate file

#### Metadata Files
**Sample Metadata**:
```
sample_id,condition,batch,replicate
sample_1,control,batch1,1
sample_2,control,batch1,2
sample_3,treatment,batch2,1
sample_4,treatment,batch2,2
```

**Gene Metadata**:
```
gene_id,gene_symbol,biotype,chromosome
ENSG00000001,DDX11L1,processed_transcript,chr1
ENSG00000002,WASH7P,unprocessed_pseudogene,chr1
```

### Mass Spectrometry Proteomics Formats

#### MaxQuant Output

**proteinGroups.txt**
- **Description**: Main MaxQuant output file with protein quantification
- **Key Columns**:
  - `Protein IDs`: UniProt identifiers
  - `Gene names`: Gene symbols
  - `Intensity <sample>`: Raw protein intensities
  - `LFQ intensity <sample>`: Label-free quantified intensities
  - `Razor + unique peptides`: Peptide counts

**Loading**:
```bash
/read proteinGroups.txt
```

**peptides.txt**
- **Description**: Peptide-level quantification
- **Usage**: For peptide-level analysis or filtering

#### Spectronaut Output

**CSV/Excel Format**
- **Structure**: Protein or peptide quantification matrix
- **Key Columns**:
  - Protein/peptide identifiers
  - Sample quantifications
  - Quality metrics (CV, detection frequency)

**Loading**:
```bash
/read spectronaut_results.csv
```

#### Generic Proteomics Format

**Intensity Matrix**:
```
protein_id,sample_1,sample_2,sample_3,sample_4
P12345,1200.5,1500.2,1300.8,1800.1
Q67890,800.3,950.7,750.2,1100.4
```

**Requirements**:
- Protein identifiers (UniProt, gene symbols)
- Quantitative values (intensities, ratios)
- Missing values as NA, NaN, or empty

### Affinity Proteomics Formats

#### Olink NPX Data

**CSV Format**:
```
SampleID,UniProt,Assay,NPX,Panel
Sample_1,P12345,IL6,5.2,Inflammation
Sample_1,Q67890,TNF,4.8,Inflammation
Sample_2,P12345,IL6,5.5,Inflammation
```

**Structure**:
- **NPX Values**: Normalized protein expression
- **Panel Information**: Olink panel designation
- **UniProt IDs**: Protein identifiers
- **Assay Names**: Protein assay identifiers

#### Antibody Array Data

**Intensity Matrix**:
```
sample_id,protein_1,protein_2,protein_3,...
control_1,1500,2200,800,...
control_2,1600,2100,750,...
treatment_1,2200,3500,1200,...
```

**Metadata Requirements**:
- Antibody validation information
- Protein identifiers
- Sample annotations

### Multi-Omics Formats

#### MuData (Multi-modal AnnData)
**File Extension**: `.h5mu`

**Description**: Stores multiple omics modalities in single file

**Structure**:
```
MuData object with:
- mod['rna']: Transcriptomics AnnData
- mod['protein']: Proteomics AnnData
- mod['atac']: Chromatin accessibility AnnData
- obs: Shared sample metadata
- var: Combined feature metadata
```

**Loading**:
```bash
/read multiomics_data.h5mu
```

#### Integrated CSV Formats
**Separate Files for Each Modality**:
- `transcriptomics.csv`
- `proteomics.csv`
- `metadata.csv`

**Sample Matching**: Common sample identifiers across files

### Metadata Formats

#### Sample Metadata

**Standard Format**:
```
sample_id,condition,batch,age,gender,replicate
sample_1,control,batch1,25,female,1
sample_2,control,batch1,27,male,2
sample_3,treatment,batch2,24,female,1
```

**Required Columns**:
- `sample_id`: Unique sample identifier
- Additional columns as needed for experimental design

**Supported Data Types**:
- Categorical: condition, batch, gender
- Numerical: age, dose, time
- Date/time: collection_date, processing_time

#### Feature Metadata

**Gene Metadata**:
```
gene_id,gene_symbol,biotype,chromosome,start,end
ENSG00000001,DDX11L1,processed_transcript,chr1,11869,14409
```

**Protein Metadata**:
```
protein_id,gene_symbol,protein_name,molecular_weight
P12345,IL6,Interleukin-6,23.7
```

### GEO Database Integration

#### Automatic GEO Download
**Usage**:
```bash
"Download GSE12345 from GEO database"
```

**Supported GEO Formats**:
- Series Matrix Files (`GSE*_series_matrix.txt.gz`)
- Supplementary Files (various formats)
- Platform Annotations (`GPL*`)

**Processing**:
- Automatic format detection
- Metadata extraction
- Sample annotation processing
- Expression matrix reconstruction

#### Manual GEO Files
**Loading Downloaded Files**:
```bash
/read GSE12345_series_matrix.txt.gz
/read GSE12345_RAW.tar  # Extract and process
```

## Output Formats

### Analysis Results

#### H5AD Output
**Generated Data**:
- Processed expression matrices
- Quality control metrics
- Clustering results
- Dimensionality reduction coordinates
- Differential expression results

**Professional Naming Convention**:
```
geo_gse12345_quality_assessed.h5ad
geo_gse12345_filtered_normalized.h5ad
geo_gse12345_clustered.h5ad
geo_gse12345_annotated.h5ad
```

#### CSV Export
**Differential Expression Results**:
```
gene_id,gene_symbol,log2FoldChange,pvalue,padj,baseMean
ENSG00000001,DDX11L1,2.5,0.001,0.05,150.2
ENSG00000002,WASH7P,-1.8,0.002,0.06,89.7
```

**Cluster Annotations**:
```
cell_id,cluster,cell_type,confidence
cell_1,0,Hepatocyte,0.95
cell_2,1,Stellate_Cell,0.87
```

### Visualization Outputs

#### Interactive HTML Plots
**Format**: Plotly HTML files
**Features**:
- Zoom, pan, hover information
- Publication-quality rendering
- Embedded metadata

**Example Files**:
- `plot_1_UMAP_clusters.html`
- `plot_2_volcano_plot.html`
- `plot_3_quality_metrics.html`

#### Static Image Exports
**Formats**: PNG, PDF, SVG
**Usage**: Publications and presentations
**Resolution**: High-resolution (300+ DPI)

### Session Exports

#### Complete Data Package
**Format**: ZIP archive
**Contents**:
- All processed data files (H5AD format)
- Generated plots (HTML and PNG)
- Analysis metadata and parameters
- Technical summary report
- Provenance information

**Structure**:
```
lobster_analysis_package_20240115_143022.zip
├── modalities/
│   ├── dataset_processed.h5ad
│   ├── dataset_processed.csv
│   └── dataset_metadata.json
├── plots/
│   ├── plot_1_clusters.html
│   ├── plot_1_clusters.png
│   └── index.json
├── technical_summary.md
├── workspace_status.json
└── provenance.json
```

#### Session State
**Format**: JSON metadata
**Content**: Analysis parameters, tool usage history, session information

## Format Conversion Capabilities

### Automatic Conversion

Lobster AI automatically handles format conversion during loading:

#### Single-Cell Conversions
- **CSV/Excel → AnnData**: Matrix orientation detection and conversion
- **10X → AnnData**: MTX format to AnnData with metadata
- **H5 → AnnData**: 10X HDF5 to AnnData format

#### Bulk RNA-seq Conversions
- **CSV → Count Matrix**: Proper gene/sample orientation
- **Excel → Multiple Sheets**: Extract expression and metadata

#### Proteomics Conversions
- **MaxQuant → Standard Matrix**: Extract relevant columns
- **Wide → Long Format**: Reshape for analysis tools
- **Missing Value Handling**: Consistent NA representation

### Manual Conversion Requests

```bash
# Convert Excel to CSV
"Convert this Excel file to CSV format for analysis"

# Reshape data matrix
"Transpose this matrix so genes are rows and samples are columns"

# Extract specific columns
"Extract only the LFQ intensity columns from this MaxQuant file"

# Merge files
"Combine the expression data with the sample metadata file"
```

## Data Validation and Quality Checks

### Automatic Validation

#### Structure Validation
- **Matrix Dimensions**: Consistent row/column counts
- **Data Types**: Numeric values in expression matrices
- **Identifiers**: Valid gene/protein/sample IDs
- **Missing Values**: Appropriate handling of NA values

#### Content Validation
- **Expression Ranges**: Biologically reasonable values
- **Count Data**: Non-negative values for count matrices
- **Metadata Consistency**: Matching sample identifiers
- **Format Compliance**: Standard field requirements

### Quality Assessments

#### Single-Cell Data
- **Gene Detection**: Minimum genes per cell
- **Cell Quality**: Mitochondrial content, doublet detection
- **Library Complexity**: UMI and gene count distributions

#### Bulk RNA-seq Data
- **Library Sizes**: Total count distributions
- **Gene Detection**: Expressed genes per sample
- **Batch Effects**: PCA-based assessment

#### Proteomics Data
- **Missing Value Patterns**: MNAR vs MCAR assessment
- **Coefficient of Variation**: Technical reproducibility
- **Dynamic Range**: Protein intensity distributions

## Best Practices

### Data Preparation

#### File Organization
```
project/
├── raw_data/
│   ├── expression_matrix.csv
│   ├── sample_metadata.csv
│   └── gene_annotations.csv
├── processed_data/
└── results/
```

#### Naming Conventions
- **Descriptive Names**: Include data type, condition, date
- **No Spaces**: Use underscores instead of spaces
- **Version Control**: Include version numbers for iterations

#### Metadata Standards
- **Complete Annotations**: All relevant experimental factors
- **Consistent Identifiers**: Use standard gene/protein IDs
- **Missing Data**: Explicit NA values, never empty strings

### Format Selection

#### Choose Based on Analysis Type
- **H5AD**: Single-cell analysis workflows
- **CSV**: Simple bulk RNA-seq experiments
- **Excel**: Small datasets with multiple annotation sheets
- **HDF5**: Large datasets requiring compression

#### Consider Downstream Tools
- **scanpy**: H5AD format preferred
- **DESeq2**: Count matrices with integer values
- **Custom Analysis**: CSV for maximum compatibility

### Performance Considerations

#### Large Datasets
- **Compression**: Use compressed formats (H5AD, HDF5)
- **Sparse Matrices**: Appropriate for single-cell data
- **Chunked Loading**: For very large files
- **Memory Management**: Monitor memory usage during loading

#### Network Transfer
- **Compressed Files**: Reduce transfer time
- **Batch Loading**: Multiple small files vs. single large file
- **Cloud Storage**: Consider cloud-native formats

## Troubleshooting Common Issues

### Loading Problems

#### "File format not recognized"
**Cause**: Unsupported or malformed file format
**Solution**:
```bash
# Check file structure
"What format is this file and how can I load it?"

# Manual format specification
"Load this file treating it as a CSV with genes as rows"
```

#### "Inconsistent dimensions"
**Cause**: Matrix dimensions don't match metadata
**Solution**:
```bash
# Validate data structure
"Check if my expression matrix matches the sample metadata"

# Fix dimension mismatch
"Transpose this matrix to match the metadata"
```

### Data Quality Issues

#### "High percentage of missing values"
**Cause**: Poor data quality or incorrect format interpretation
**Solution**:
```bash
# Assess missing value patterns
"Analyze the missing value patterns in this proteomics data"

# Apply appropriate handling
"Handle missing values using MNAR imputation for this MS data"
```

#### "No variance in expression data"
**Cause**: Data may be pre-normalized or log-transformed
**Solution**:
```bash
# Check data distribution
"Examine the distribution of expression values"

# Apply appropriate preprocessing
"Skip normalization since this data appears pre-normalized"
```

### Format Compatibility

#### "Cannot convert between formats"
**Cause**: Incompatible data structures or missing information
**Solution**:
```bash
# Identify conversion requirements
"What information do I need to convert this data to AnnData format?"

# Provide missing metadata
"Use default gene symbols for missing gene annotations"
```

This comprehensive data formats guide covers all major biological data formats supported by Lobster AI, providing detailed specifications and best practices for effective data analysis.