# Manual Cell Type Annotation Service Documentation

## Overview

The Manual Cell Type Annotation Service provides expert-guided cell type annotation capabilities for single-cell RNA-seq data with a color-synchronized Rich terminal interface that matches UMAP plot colors. This addresses Step 7 of the customer workflow: "I assign each cluster to a named cell type or to 'Debris', sometimes collapsing multiple clusters into the same cell type".

## Key Features

### 🎨 Color-Synchronized Rich Interface
- **Perfect color matching** between UMAP plots and terminal interface
- **Visual cluster identification** eliminates cognitive load
- **Interactive menus** with color-coded cluster displays
- **Real-time progress tracking** with Rich components

### 🧬 Expert-Guided Annotation
- **Manual cluster assignment** with biological expertise
- **Cluster collapsing** for merging similar cell populations
- **Debris identification** with QC-based smart suggestions
- **Undo/redo functionality** with full annotation history

### 📋 Annotation Templates
- **Tissue-specific templates** for common organs (PBMC, Brain, Lung, Heart, etc.)
- **Marker gene validation** for biological consistency
- **Custom template creation** for specialized analyses
- **Template-based suggestions** with confidence scoring

### 💾 Data Management
- **Export/import mappings** for workflow reproducibility
- **Annotation validation** with coverage metrics
- **Integration with DataManagerV2** for seamless workflows
- **Provenance tracking** with full audit trails

## Architecture

### Core Components

```
Manual Annotation Service
├── ManualAnnotationService     # Main service class
├── ClusterInfo                 # Cluster metadata
├── AnnotationState            # Session state management
├── Rich Terminal Interface    # Color-synchronized UI
└── Integration Layer          # Tools for SingleCell Expert
```

### Data Flow

```
1. 📊 UMAP Plot Generation
   ├── Cluster colors extracted from plot
   └── Color palette stored for synchronization

2. 🖥️ Rich Terminal Interface
   ├── Colors synchronized with plot
   ├── Interactive cluster selection
   └── Real-time annotation progress

3. 💾 Annotation Application
   ├── Cell type mappings applied to AnnData
   ├── Metadata and provenance stored
   └── New modality created with annotations
```

## Quick Start Guide

### Step 1: Launch Interactive Annotation

```python
# From SingleCell Expert Agent
manually_annotate_clusters_interactive(
    modality_name="my_dataset_clustered",
    cluster_col="leiden",
    save_result=True
)
```

This launches the Rich terminal interface with:
- Color legend matching your UMAP plot
- Interactive cluster selection menus
- Progress tracking and validation
- Export/import capabilities

### Step 2: Rich Terminal Workflow

The interactive interface provides:

```
🧬 Manual Cell Type Annotation Service

Main Menu:
1. Annotate Clusters    - Assign cell types to clusters
2. Mark Debris         - Identify low-quality clusters
3. Collapse Clusters   - Merge clusters with same cell type
4. View Summary        - Show current annotation status
5. Apply Template      - Use predefined annotation template
6. Undo               - Undo last annotation action
7. Export             - Save annotations to file
8. Import             - Load annotations from file
9. Finish             - Complete annotation session
```

### Step 3: Color-Synchronized Annotation

- **Visual Connection**: Cluster colors in terminal exactly match UMAP plot
- **Easy Identification**: See Cluster 0 in red on plot → see "Cluster 0" in red in terminal
- **Cognitive Relief**: No mental mapping between plot and interface required

## Tool Reference

### Interactive Annotation Tools

#### `manually_annotate_clusters_interactive`
Launch Rich terminal interface for manual annotation with color synchronization.

**Parameters:**
- `modality_name` (str): Name of clustered single-cell modality
- `cluster_col` (str): Column containing cluster assignments (default: "leiden")
- `save_result` (bool): Whether to save annotated modality (default: True)

**Returns:** Comprehensive annotation results with color-synchronized interface completion.

#### `manually_annotate_clusters`
Directly assign cell types to clusters without interactive interface.

**Parameters:**
- `modality_name` (str): Name of clustered single-cell modality
- `annotations` (dict): Dictionary mapping cluster IDs to cell type names
- `cluster_col` (str): Column containing cluster assignments (default: "leiden")
- `save_result` (bool): Whether to save annotated modality (default: True)

**Example:**
```python
manually_annotate_clusters(
    modality_name="pbmc_clustered",
    annotations={
        "0": "T cells CD4+",
        "1": "T cells CD8+",
        "2": "B cells",
        "3": "NK cells",
        "4": "Monocytes"
    }
)
```

### Cluster Management Tools

#### `collapse_clusters_to_celltype`
Merge multiple clusters into a single cell type annotation.

**Parameters:**
- `modality_name` (str): Name of single-cell modality
- `cluster_list` (List[str]): List of cluster IDs to collapse
- `cell_type_name` (str): New cell type name for collapsed clusters
- `cluster_col` (str): Column containing cluster assignments (default: "leiden")
- `save_result` (bool): Whether to save result (default: True)

**Example:**
```python
# Collapse T cell subclusters
collapse_clusters_to_celltype(
    modality_name="pbmc_clustered",
    cluster_list=["0", "1", "5"],
    cell_type_name="T cells",
    cluster_col="leiden"
)
```

#### `mark_clusters_as_debris`
Mark specified clusters as debris for quality control.

**Parameters:**
- `modality_name` (str): Name of single-cell modality
- `debris_clusters` (List[str]): List of cluster IDs to mark as debris
- `remove_debris` (bool): Whether to remove debris clusters from data (default: False)
- `cluster_col` (str): Column containing cluster assignments (default: "leiden")
- `save_result` (bool): Whether to save result (default: True)

### Quality Control Tools

#### `suggest_debris_clusters`
Get smart suggestions for potential debris clusters based on QC metrics.

**Parameters:**
- `modality_name` (str): Name of single-cell modality
- `min_genes` (int): Minimum genes per cell threshold (default: 200)
- `max_mt_percent` (float): Maximum mitochondrial percentage (default: 50)
- `min_umi` (int): Minimum UMI count threshold (default: 500)
- `cluster_col` (str): Column containing cluster assignments (default: "leiden")

**Returns:** Smart suggestions based on:
- Low gene count per cluster
- High mitochondrial gene percentage
- Low UMI counts
- Very small cluster sizes (<10 cells)

### Template-Based Annotation

#### `apply_annotation_template`
Apply predefined tissue-specific annotation template.

**Parameters:**
- `modality_name` (str): Name of single-cell modality
- `tissue_type` (str): Type of tissue - Available options:
  - `"pbmc"` - Peripheral Blood Mononuclear Cells
  - `"brain"` - Brain tissue (neurons, glia, immune)
  - `"lung"` - Lung tissue (epithelial, immune, vascular)
  - `"heart"` - Heart tissue (cardiomyocytes, fibroblasts, vascular)
  - `"kidney"` - Kidney tissue (tubular, glomerular, vascular)
  - `"liver"` - Liver tissue (hepatocytes, stellate, immune)
  - `"intestine"` - Intestinal tissue (epithelial, stem, immune)
  - `"skin"` - Skin tissue (keratinocytes, melanocytes, immune)
  - `"tumor"` - Tumor microenvironment (malignant, immune, stromal)
- `cluster_col` (str): Column containing cluster assignments (default: "leiden")
- `expression_threshold` (float): Minimum expression for marker detection (default: 0.5)
- `save_result` (bool): Whether to save annotated modality (default: True)

**Example:**
```python
# Apply PBMC template for blood sample
apply_annotation_template(
    modality_name="blood_clustered",
    tissue_type="pbmc",
    expression_threshold=0.6
)
```

### Review and Export Tools

#### `review_annotation_assignments`
Review current manual annotation assignments with coverage statistics.

#### `export_annotation_mapping`
Export annotation mapping for reuse in other analyses.

**Parameters:**
- `output_filename` (str): Output filename (default: "annotation_mapping.json")
- `format` (str): Export format - "json" or "csv" (default: "json")

#### `import_annotation_mapping`
Import and apply annotation mapping from previous analysis.

**Parameters:**
- `mapping_file` (str): Path to mapping file (JSON format)
- `preview_only` (bool): If True, only show what would be applied (default: False)

## Rich Terminal Interface Guide

### Welcome Screen
```
🧬 Manual Cell Type Annotation Service

Welcome to the interactive annotation interface! This tool provides:

• Color-synchronized visualization matching your UMAP plot
• Interactive cluster assignment with expert guidance
• Debris identification and quality control
• Annotation templates for common tissue types
• Undo/redo functionality with full history

Current Session:
• Total clusters: 12
• Total cells: 8,543
• Annotated: 3
• Debris: 1

The colors in this terminal exactly match your UMAP plot colors for easy identification.
```

### Color Legend
```
┌─ Cluster Color Legend ─┐
│ Cluster ID │ Color │ Cell Count │ Status    │
├────────────┼───────┼────────────┼───────────┤
│ 0          │ ●●●   │ 1,234      │ Annotated │
│ 1          │ ●●●   │ 856        │ Pending   │
│ 2          │ ●●●   │ 445        │ Debris    │
└────────────┴───────┴────────────┴───────────┘
```

### Annotation Workflow

#### 1. Cluster Annotation Mode
```
🎯 Cluster Annotation Mode
Colors match your UMAP plot for easy identification.

┌─ Clusters to Annotate ─┐
│ ID │ Color │ Cells │ QC Metrics              │
├────┼───────┼───────┼─────────────────────────┤
│ 0  │ ●●●●  │ 1,234 │ Genes: 2,150, MT%: 8.5  │
│ 1  │ ●●●●  │ 856   │ Genes: 1,890, MT%: 12.1 │
└────┴───────┴───────┴─────────────────────────┘

● Enter cell type for cluster 0: T cells CD4+
✅ Cluster 0 annotated as 'T cells CD4+'
```

#### 2. Debris Identification Mode
```
🗑️ Debris Identification Mode

💡 Smart debris suggestions based on QC metrics:
┌─────────┬──────────────────────────┬───────┬─────────────────┐
│ Cluster │ Reason                   │ Cells │ Action          │
├─────────┼──────────────────────────┼───────┼─────────────────┤
│ 7       │ Low gene count (145)     │ 23    │ Mark as debris? │
│ 9       │ High MT% (65.2%)         │ 12    │ Mark as debris? │
└─────────┴──────────────────────────┴───────┴─────────────────┘

Apply smart debris suggestions? [y/N]: y
```

#### 3. Annotation Summary
```
📊 Annotation Summary

┌─ Overall Statistics ─┐
│ Metric        │ Count │ Percentage │
├───────────────┼───────┼────────────┤
│ Total Clusters│ 10    │ 100%       │
│ Annotated     │ 8     │ 80.0%      │
│ Debris        │ 1     │ 10.0%      │
│ Pending       │ 1     │ 10.0%      │
└───────────────┴───────┴────────────┘

┌─ Cell Type Annotations ─┐
│ Cell Type      │ Clusters │ Total Cells │ Avg Cells/Cluster │
├────────────────┼──────────┼─────────────┼────────────────────┤
│ T cells CD4+   │ 2        │ 2,090       │ 1,045              │
│ T cells CD8+   │ 2        │ 1,456       │ 728                │
│ B cells        │ 1        │ 445         │ 445                │
│ NK cells       │ 1        │ 332         │ 332                │
│ Monocytes      │ 2        │ 1,123       │ 562                │
└────────────────┴──────────┴─────────────┴────────────────────┘
```

## Programming Interface

### Core Service Usage

```python
from lobster.tools.manual_annotation_service import ManualAnnotationService
from rich.console import Console

# Initialize service
console = Console()
service = ManualAnnotationService(console)

# Initialize annotation session
state = service.initialize_annotation_session(
    adata=clustered_data,
    cluster_key='leiden'
)

# Launch interactive interface
cell_type_mapping = service.rich_annotation_interface()

# Apply annotations to data
adata_annotated = service.apply_annotations_to_adata(
    adata=clustered_data,
    cluster_key='leiden',
    cell_type_column='cell_type_manual'
)
```

### Template Service Usage

```python
from lobster.tools.annotation_templates import AnnotationTemplateService, TissueType

# Initialize template service
template_service = AnnotationTemplateService()

# Get available tissue types
tissue_types = template_service.get_all_tissue_types()
print(tissue_types)  # [TissueType.PBMC, TissueType.BRAIN, ...]

# Apply PBMC template
suggestions = template_service.apply_template_to_clusters(
    adata=clustered_data,
    tissue_type=TissueType.PBMC,
    cluster_col='leiden'
)

# Get marker genes for specific cell type
markers = template_service.get_markers_for_cell_type(
    tissue_type=TissueType.PBMC,
    cell_type='T cells CD4+'
)
print(markers)  # ['CD3D', 'CD3E', 'CD4', 'IL7R', 'CCR7', 'LEF1']
```

## Workflow Integration

### Standard Single-Cell Pipeline with Manual Annotation

```python
# Step 1-6: Standard single-cell preprocessing and clustering
check_data_status()
assess_data_quality("dataset")
filter_and_normalize_modality("dataset")
detect_doublets_in_modality("dataset_filtered_normalized")
cluster_modality("dataset_filtered_normalized")
find_marker_genes_for_clusters("dataset_clustered")

# Step 7: Manual annotation (THE KEY STEP)
manually_annotate_clusters_interactive("dataset_clustered")

# Step 8: Continue with annotated data
create_pseudobulk_matrix(
    "dataset_manually_annotated",
    sample_col="sample_id",
    celltype_col="cell_type_manual"
)
```

## Annotation Templates Reference

### PBMC Template
**Cell Types Available:**
- T cells CD4+ (CD3D, CD3E, CD4, IL7R, CCR7, LEF1)
- T cells CD8+ (CD3D, CD3E, CD8A, CD8B, CCL5, GZMK)
- T cells regulatory (CD3D, CD3E, CD4, FOXP3, IL2RA, CTLA4)
- NK cells (GNLY, NKG7, KLRD1, KLRB1, NCAM1, KLRF1)
- B cells naive (CD19, MS4A1, CD79A, CD79B, IGHD, TCL1A)
- B cells memory (CD19, MS4A1, CD79A, CD27, CD38, IGHG1)
- Plasma cells (IGHG1, IGHG2, IGHG3, IGHG4, JCHAIN, XBP1)
- Monocytes CD14+ (CD14, LYZ, S100A8, S100A9, FCN1, VCAN)
- Monocytes CD16+ (FCGR3A, MS4A7, LST1, AIF1, SERPINA1)
- Dendritic cells (FCER1A, CST3, CLEC9A, XCR1, BATF3, IRF8)
- Platelets (PPBP, PF4, NRGN, GP9, TUBB1, CLU)

### Brain Template
**Cell Types Available:**
- Excitatory neurons (SLC17A7, CAMK2A, RBFOX3, NEUROD2, NEUROD6, SATB2)
- Inhibitory neurons (GAD1, GAD2, SLC32A1, PVALB, SST, VIP)
- Astrocytes (GFAP, AQP4, ALDH1L1, S100B, SOX9, SLC1A3)
- Oligodendrocytes (MBP, MOG, PLP1, MAG, CNP, MOBP)
- Oligodendrocyte precursors (PDGFRA, CSPG4, SOX10, OLIG2, OLIG1, NKX2-2)
- Microglia (CX3CR1, P2RY12, TMEM119, AIF1, CSF1R, TREM2)
- Endothelial cells (PECAM1, VWF, CDH5, FLT1, CLDN5, PLVAP)
- Pericytes (PDGFRB, RGS5, ACTA2, CSPG4, ANPEP, MCAM)

### Additional Templates
- **Lung**: AT1/AT2 cells, Club cells, Ciliated cells, Basal cells, Alveolar macrophages
- **Heart**: Cardiomyocytes, Cardiac fibroblasts, Smooth muscle cells, Endothelial cells
- **Kidney**: Podocytes, Proximal tubule, Distal tubule, Collecting duct, Loop of Henle
- **Liver**: Hepatocytes, Cholangiocytes, Hepatic stellate cells, Kupffer cells
- **Tumor**: Tumor cells, Exhausted T cells, TAM M1/M2, Cancer-associated fibroblasts

## Best Practices

### 1. Color Synchronization Workflow
```bash
# Ensure UMAP plot is visible before starting annotation
create_umap_plot("dataset_clustered", color_by="leiden")

# Launch annotation with same clustering
manually_annotate_clusters_interactive("dataset_clustered", cluster_col="leiden")

# Colors will automatically match between plot and terminal
```

### 2. Quality Control Integration
```bash
# Get smart debris suggestions first
suggest_debris_clusters("dataset_clustered", min_genes=300, max_mt_percent=25)

# Apply suggestions or use interactive debris marking
mark_clusters_as_debris("dataset_clustered", debris_clusters=["7", "9"])

# Then proceed with biological annotation
manually_annotate_clusters_interactive("dataset_debris_marked")
```

### 3. Template-Guided Annotation
```bash
# Start with template suggestions
apply_annotation_template("dataset_clustered", tissue_type="pbmc")

# Review and refine with manual annotation
review_annotation_assignments("dataset_template_pbmc", annotation_col="cell_type_template")

# Use interactive mode for final curation
manually_annotate_clusters_interactive("dataset_template_pbmc")
```

### 4. Reproducible Workflows
```bash
# Export annotations from reference dataset
export_annotation_mapping("reference_annotated", output_filename="pbmc_reference.json")

# Import and apply to new dataset
import_annotation_mapping("new_dataset_clustered", mapping_file="pbmc_reference.json")

# Review and adjust for dataset-specific differences
review_annotation_assignments("new_dataset_imported_annotations")
```

## Advanced Features

### Annotation History and Undo
- **Full history tracking** of all annotation actions
- **Undo/redo capability** for mistake correction
- **Action timestamps** for audit trails
- **Session state preservation** throughout workflow

### Smart Suggestions
- **QC-based debris detection** using statistical thresholds
- **Template-based cell type suggestions** with confidence scoring
- **Biological consistency validation** against known markers
- **Coverage analysis** with annotation completeness metrics

### Export Formats

#### JSON Export Format
```json
{
  "cell_type_mapping": {
    "0": "T cells CD4+",
    "1": "T cells CD8+",
    "2": "B cells"
  },
  "debris_clusters": ["9"],
  "cluster_info": {
    "0": {
      "color": "#1f77b4",
      "cell_count": 1234,
      "assigned_type": "T cells CD4+",
      "is_debris": false,
      "qc_scores": {"mean_genes": 2150, "mean_mt_pct": 8.5}
    }
  },
  "export_timestamp": "2025-09-17T00:30:00"
}
```

#### CSV Export Format
```csv
cell_type,cell_count,percentage
T cells CD4+,2090,24.5
T cells CD8+,1456,17.1
B cells,445,5.2
NK cells,332,3.9
Monocytes,1123,13.2
Debris,234,2.7
Unassigned,2863,33.5
```

## Performance Considerations

### Large Dataset Optimization
- **Efficient cluster sampling** for >50k cell datasets
- **Progressive loading** of cluster information
- **Color palette caching** to avoid recomputation
- **Memory-efficient** Rich interface updates

### Terminal Compatibility
- **Automatic color support detection** via Rich
- **Graceful fallback** to text-based interface if needed
- **Cross-platform compatibility** (macOS, Linux, Windows)
- **SSH/remote terminal support** with color preservation

## Integration Points

### DataManagerV2 Integration
- Seamless modality management with provenance tracking
- Automatic file naming and workspace organization
- Integration with existing quality metrics and metadata
- Full compatibility with multi-omics workflows

### Existing Lobster Services
- **Quality Service**: QC metrics for debris suggestions
- **Clustering Service**: Leiden clustering results as input
- **Visualization Service**: Color palette extraction and synchronization
- **Enhanced SingleCell Service**: Marker gene integration

## Success Metrics

✅ **Visual Consistency**: Perfect color matching between plot and terminal
✅ **Intuitive UX**: Immediate visual connection for cluster identification
✅ **Fast Implementation**: 4-week timeline using existing infrastructure
✅ **Professional Polish**: Rich terminal with autocomplete, progress tracking
✅ **Expert Integration**: Seamless workflow for bioinformaticians
✅ **Reproducible Results**: Export/import for consistent annotations
✅ **Quality Control**: Smart debris detection with biological validation
✅ **Template Support**: Tissue-specific annotation guidance

The Manual Cell Type Annotation Service transforms expert-guided annotation from a tedious task into an intuitive, visually-guided workflow while maintaining all the robustness and professional capabilities required for single-cell RNA-seq analysis.