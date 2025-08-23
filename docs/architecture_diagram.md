# Lobster AI Modular DataManagerV2 Architecture

## System Architecture Overview - Post Migration

```mermaid
graph TB
    %% Data Sources
    subgraph "Data Sources"
        GEO[GEO Database<br/>GSE Datasets]
        CSV[CSV Files<br/>Local Data]
        EXCEL[Excel Files<br/>Lab Data]
        H5AD[H5AD Files<br/>Processed Data]
        MTX[10X MTX<br/>Single-cell Data]
    end

    %% Agents Layer
    subgraph "AI Agents - Modality Orchestrators"
        DE[Data Expert<br/>🔄 Data Loading & Management]
        TE[Transcriptomics Expert<br/>🧬 RNA-seq Analysis]
        PE[Proteomics Expert<br/>🧪 Protein Analysis]
        ME[Method Expert<br/>📚 Literature Research]
    end

    %% NEW: Analysis Services Layer (Stateless)
    subgraph "Analysis Services - Stateless & Modular"
        PREP[PreprocessingService<br/>🔧 Filter & Normalize]
        QUAL[QualityService<br/>📊 QC Assessment]
        CLUST[ClusteringService<br/>🎯 Leiden & UMAP]
        SCELL[EnhancedSingleCellService<br/>🔬 Doublets & Annotation]
        BULK[BulkRNASeqService<br/>📈 Bulk Analysis]
        PUBMED[PubMedService<br/>📚 Literature Mining]
        GEO_SVC[GEOService<br/>💾 Data Download]
    end

    %% DataManagerV2 Orchestration
    subgraph "DataManagerV2 Orchestration"
        DM2[DataManagerV2<br/>🎯 Modality Coordinator]
        MODALITIES[Modality Storage<br/>📊 AnnData Objects]
        PROV[Provenance Tracker<br/>📋 Analysis History]
        ERROR[Error Handling<br/>⚠️ Professional Exceptions]
    end

    %% Modality Adapters
    subgraph "Modality Adapters"
        TRA[TranscriptomicsAdapter<br/>🧬 RNA-seq Loading]
        PRA[ProteomicsAdapter<br/>🧪 Protein Loading]
        
        subgraph "Transcriptomics Types"
            TRSC[Single-cell RNA-seq<br/>Schema & Validation]
            TRBL[Bulk RNA-seq<br/>Schema & Validation]
        end
        
        subgraph "Proteomics Types"
            PRMS[Mass Spectrometry<br/>Missing Value Handling]
            PRAF[Affinity Proteomics<br/>Antibody Arrays]
        end
        
        TRA --> TRSC
        TRA --> TRBL
        PRA --> PRMS
        PRA --> PRAF
    end

    %% Storage Backends
    subgraph "Storage Backends"
        H5BE[H5ADBackend<br/>💾 Single Modality<br/>S3-Ready]
        MUBE[MuDataBackend<br/>🔗 Multi-Modal<br/>Integrated Analysis]
    end

    %% Schema & Validation
    subgraph "Schema System"
        TSCH[TranscriptomicsSchema<br/>📋 RNA-seq Rules]
        PSCH[ProteomicsSchema<br/>📋 Protein Rules]
        FVAL[FlexibleValidator<br/>⚠️ Warning-based QC]
    end

    %% Interfaces
    subgraph "Core Interfaces"
        IBACK[IDataBackend<br/>🔌 Storage Contract]
        IADAP[IModalityAdapter<br/>🔌 Processing Contract]
        IVAL[IValidator<br/>🔌 Validation Contract]
    end

    %% Data Flow Connections
    GEO --> DE
    CSV --> DE  
    EXCEL --> DE
    H5AD --> DE
    MTX --> DE

    %% NEW: Agent to Service connections
    DE --> GEO_SVC
    DE --> PUBMED
    TE --> PREP
    TE --> QUAL
    TE --> CLUST
    TE --> SCELL
    PE --> PREP
    PE --> QUAL

    %% Service to DataManager connections
    PREP --> |AnnData Processing| DM2
    QUAL --> |QC Metrics| DM2
    CLUST --> |Clustering Results| DM2
    SCELL --> |Annotations| DM2
    GEO_SVC --> |Dataset Loading| DM2
    PUBMED --> |Metadata| DM2

    %% DataManager orchestration
    DM2 --> TRA
    DM2 --> PRA
    DM2 --> MODALITIES
    DM2 --> PROV
    DM2 --> ERROR

    TRA --> TSCH
    PRA --> PSCH
    TSCH --> FVAL
    PSCH --> FVAL

    MODALITIES --> H5BE
    MODALITIES --> MUBE

    %% Interface implementations
    TRA -.-> IADAP
    PRA -.-> IADAP
    H5BE -.-> IBACK
    MUBE -.-> IBACK
    FVAL -.-> IVAL

    %% Styling
    classDef agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef service fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef orchestrator fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef adapter fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef backend fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef schema fill:#f1f8e9,stroke:#388e3c,stroke-width:2px
    classDef interface fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,stroke-dasharray: 5 5
    classDef source fill:#f5f5f5,stroke:#616161,stroke-width:1px

    class DE,TE,PE,ME agent
    class PREP,QUAL,CLUST,SCELL,BULK,PUBMED,GEO_SVC service
    class DM2,MODALITIES,PROV,ERROR orchestrator
    class TRA,PRA,TRSC,TRBL,PRMS,PRAF adapter
    class H5BE,MUBE backend
    class TSCH,PSCH,FVAL schema
    class IBACK,IADAP,IVAL interface
    class GEO,CSV,EXCEL,H5AD,MTX source
```

## Data Flow Diagram - Modular Service Architecture

```mermaid
sequenceDiagram
    participant User
    participant DataExpert as Data Expert Agent
    participant TransExpert as Transcriptomics Expert
    participant DM2 as DataManagerV2
    participant Service as Analysis Service
    participant Adapter as Modality Adapter
    participant Schema as Schema Validator
    participant Backend as Storage Backend

    %% Data Loading Flow
    User->>DataExpert: "Download GSE12345"
    DataExpert->>DM2: load_modality("geo_gse12345", source, "transcriptomics_single_cell")
    
    DM2->>Adapter: from_source(source_data)
    Adapter->>Adapter: Detect format, load data
    Adapter->>Schema: validate(adata)
    Schema-->>Adapter: ValidationResult (warnings/errors)
    Adapter-->>DM2: AnnData with schema compliance
    
    DM2->>DM2: Store as modality
    DM2-->>DataExpert: Modality loaded successfully
    DataExpert-->>User: "Loaded geo_gse12345: 5000 cells × 20000 genes"

    %% NEW: Modular Analysis Flow
    User->>TransExpert: "Filter and normalize the data"
    TransExpert->>DM2: get_modality("geo_gse12345")
    DM2-->>TransExpert: AnnData object
    
    TransExpert->>Service: PreprocessingService.filter_and_normalize_cells(adata, params)
    Service->>Service: Professional QC filtering
    Service->>Service: Scanpy normalization
    Service-->>TransExpert: (processed_adata, processing_stats)
    
    TransExpert->>DM2: Store new modality("geo_gse12345_filtered_normalized")
    TransExpert->>DM2: log_tool_usage(operation_details)
    TransExpert-->>User: "Filtering complete: 4500 cells retained (90%)"

    User->>TransExpert: "Run clustering analysis"
    TransExpert->>DM2: get_modality("geo_gse12345_filtered_normalized")
    DM2-->>TransExpert: Processed AnnData
    
    TransExpert->>Service: ClusteringService.cluster_and_visualize(adata, params)
    Service->>Service: HVG detection, PCA, neighbors graph
    Service->>Service: Leiden clustering, UMAP embedding
    Service-->>TransExpert: (clustered_adata, clustering_stats)
    
    TransExpert->>DM2: Store new modality("geo_gse12345_clustered")
    TransExpert->>DM2: log_tool_usage(clustering_results)
    TransExpert-->>User: "Clustering complete: 8 clusters identified"

    User->>TransExpert: "Find marker genes"
    TransExpert->>DM2: get_modality("geo_gse12345_clustered")
    TransExpert->>Service: EnhancedSingleCellService.find_marker_genes(adata, params)
    Service->>Service: Differential expression analysis
    Service-->>TransExpert: (marker_adata, marker_stats)
    
    TransExpert->>DM2: Store new modality("geo_gse12345_markers")
    TransExpert-->>User: "Marker genes identified for all clusters"

    %% Provenance and Error Handling
    Note over DM2: All operations tracked<br/>Professional error handling<br/>Complete provenance trail
```

## Component Interaction Matrix

```mermaid
graph LR
    subgraph "Agents → DataManagerV2"
        DE[Data Expert] --> |load_modality<br/>save_modality| DM2[DataManagerV2]
        TE[Transcriptomics Expert] --> |get_modality<br/>process_data| DM2
        PE[Proteomics Expert] --> |get_modality<br/>analyze_patterns| DM2
        ME[Method Expert] --> |parameter_guidance| DM2
    end

    subgraph "DataManagerV2 → Adapters"
        DM2 --> |from_source| TRA[TranscriptomicsAdapter]
        DM2 --> |from_source| PRA[ProteomicsAdapter]
    end

    subgraph "Adapters → Validation"
        TRA --> |validate| TSCH[TranscriptomicsSchema]
        PRA --> |validate| PSCH[ProteomicsSchema]
        TSCH --> FVAL[FlexibleValidator]
        PSCH --> FVAL
    end

    subgraph "DataManagerV2 → Storage"
        DM2 --> |save/load| H5BE[H5ADBackend]
        DM2 --> |save/load| MUBE[MuDataBackend]
    end

    classDef agent fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef orchestrator fill:#f3e5f5,stroke:#4a148c,stroke-width:3px
    classDef adapter fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef backend fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef schema fill:#fce4ec,stroke:#880e4f,stroke-width:2px

    class DE,TE,PE,ME agent
    class DM2 orchestrator
    class TRA,PRA adapter
    class H5BE,MUBE backend
    class TSCH,PSCH,FVAL schema

## Architecture Migration Summary

### 🎯 Migration Goals Achieved

The Lobster AI system has been successfully migrated from a dual-system architecture (legacy DataManager + DataManagerV2) to a clean, professional, modular DataManagerV2-only implementation.

### ✅ Key Improvements

#### **1. Modular Service Architecture**
- **Before**: Agents contained mixed responsibilities with dual code paths
- **After**: Clean separation with stateless analysis services and orchestration agents

#### **2. Professional Error Handling**
- **Custom Exception Hierarchy**: 
  - `TranscriptomicsError`, `PreprocessingError`, `QualityError`, etc.
  - `ModalityNotFoundError` for specific validation
- **Comprehensive Logging**: All operations tracked with parameters and results
- **Graceful Error Recovery**: Informative error messages with suggested fixes

#### **3. Stateless Services Design**
- **PreprocessingService**: AnnData filtering, normalization, batch correction
- **QualityService**: Comprehensive QC assessment with statistical metrics
- **ClusteringService**: Leiden clustering, PCA, UMAP visualization
- **EnhancedSingleCellService**: Doublet detection, cell type annotation
- **GEOService**: Professional dataset downloading and processing
- **PubMedService**: Literature mining and method extraction

### 🏗️ New Architecture Pattern

#### **Agent Tool Pattern**
```python
@tool
def tool_name(modality_name: str, **params) -> str:
    """Professional tool with comprehensive error handling."""
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ModalityNotFoundError(f"Modality '{modality_name}' not found")
        
        # 2. Get AnnData from modality
        adata = data_manager.get_modality(modality_name)
        
        # 3. Call stateless service
        result_adata, stats = service.method_name(adata, **params)
        
        # 4. Save new modality with descriptive name
        new_modality_name = f"{modality_name}_processed"
        data_manager.modalities[new_modality_name] = result_adata
        
        # 5. Log operation for provenance
        data_manager.log_tool_usage(tool_name, params, description)
        
        # 6. Format professional response
        return format_professional_response(stats, new_modality_name)
        
    except ServiceError as e:
        logger.error(f"Service error: {e}")
        return f"Service error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error: {str(e)}"
```

#### **Service Method Pattern**
```python
def service_method(
    self,
    adata: anndata.AnnData,
    **parameters
) -> Tuple[anndata.AnnData, Dict[str, Any]]:
    """
    Stateless service method working with AnnData directly.
    
    Returns:
        Tuple of (processed_adata, processing_statistics)
    """
    try:
        # 1. Create working copy
        adata_processed = adata.copy()
        
        # 2. Apply analysis algorithms
        # ... processing logic ...
        
        # 3. Calculate comprehensive statistics
        processing_stats = {
            "analysis_type": "method_type",
            "parameters_used": parameters,
            "results": {...}
        }
        
        return adata_processed, processing_stats
        
    except Exception as e:
        raise ServiceError(f"Method failed: {str(e)}")
```

### 📊 Modality Management System

#### **Descriptive Naming Convention**
Each analysis step creates new modalities with descriptive, traceable names:

```
geo_gse12345                    # Raw downloaded data
├── geo_gse12345_quality_assessed    # With QC metrics
├── geo_gse12345_filtered_normalized # Preprocessed data
├── geo_gse12345_doublets_detected   # With doublet annotations
├── geo_gse12345_clustered          # With clustering results
├── geo_gse12345_markers           # With marker genes
└── geo_gse12345_annotated        # With cell type annotations
```

#### **Professional Modality Tracking**
- **Provenance**: Complete analysis history with parameters
- **Statistics**: Comprehensive metrics for each processing step
- **Validation**: Schema enforcement and quality checks
- **Storage**: Automatic saving with professional file naming

### 🔬 Analysis Workflow Excellence

#### **Standard Single-cell RNA-seq Pipeline**
```
1. check_data_status() → Review available modalities
2. assess_data_quality(modality_name) → Professional QC assessment
3. filter_and_normalize_modality(...) → Clean and normalize
4. detect_doublets_in_modality(...) → Remove doublets
5. cluster_modality(...) → Leiden clustering + UMAP
6. find_marker_genes_for_clusters(...) → Differential expression
7. annotate_cell_types(...) → Automated annotation
8. create_analysis_summary() → Comprehensive report
```

#### **Quality Control Standards**
- **Professional QC Thresholds**: Evidence-based filtering parameters
- **Multi-metric Assessment**: Total counts, gene counts, mitochondrial%, ribosomal%
- **Statistical Validation**: Z-score outlier detection and percentile thresholds
- **Batch Effect Handling**: Automatic batch detection and correction options

#### **Error Handling & Recovery**
- **Input Validation**: Comprehensive parameter and data validation
- **Graceful Degradation**: Fallback methods when specialized tools unavailable
- **Informative Messages**: Clear error descriptions with suggested solutions
- **Operation Logging**: Complete audit trail for debugging and reproducibility

### 🚀 Benefits of New Architecture

#### **Code Quality Improvements**
- **50% Reduction** in agent code complexity (450+ → 200+ lines)
- **Zero Duplication**: No more dual code paths or is_v2 checks
- **Professional Standards**: Type hints, comprehensive docstrings, error handling
- **Testability**: Stateless services are easily unit tested

#### **Maintainability Enhancements**
- **Single Responsibility**: Each service handles one analysis domain
- **Modular Design**: Services can be used independently or combined
- **Clean Interfaces**: Consistent patterns across all analysis tools
- **Version Control**: Clear separation enables independent service updates

#### **Performance & Reliability**
- **Memory Efficiency**: Stateless services with minimal memory footprint
- **Fault Tolerance**: Comprehensive error handling prevents pipeline failures
- **Reproducibility**: Complete parameter logging and provenance tracking
- **Scalability**: Services can be distributed or parallelized in future versions

## Migration Impact Analysis

### 📈 Before Migration (Legacy System)
```
transcriptomics_expert.py: 450+ lines
├── Dual code paths (is_v2 checks everywhere)
├── Mixed responsibilities (orchestration + analysis)
├── Redundant implementations 
├── Complex error handling
└── Maintenance overhead
```

### 🎉 After Migration (Modular System)
```
transcriptomics_expert.py: 280 lines (clean)
├── Single DataManagerV2 path
├── Professional tool orchestration only
├── Stateless service delegation
├── Comprehensive error handling
└── Minimal maintenance overhead

Analysis Services: 4 refactored services
├── PreprocessingService: AnnData → (filtered_adata, stats)
├── QualityService: AnnData → (qc_adata, assessment)
├── ClusteringService: AnnData → (clustered_adata, results)
└── EnhancedSingleCellService: AnnData → (annotated_adata, metrics)
```

### 🔧 Technical Architecture Benefits

#### **Service Layer Advantages**
- **Reusability**: Services can be used by multiple agents
- **Testability**: Each service can be independently tested
- **Flexibility**: Easy to add new analysis methods
- **Performance**: Optimized algorithms with professional implementations

#### **Agent Layer Improvements**
- **Orchestration Focus**: Agents handle modality management and user interaction
- **Clean Tool Interface**: Consistent ~20-30 line tool implementations
- **Professional Responses**: Formatted outputs with comprehensive statistics
- **Error Management**: Hierarchical error handling with specific exceptions

#### **DataManagerV2 Integration**
- **Modality-Centric**: All data operations centered around named modalities
- **Provenance Tracking**: Complete analysis history with tool usage logging
- **Schema Validation**: Automatic validation ensures data integrity
- **Storage Management**: Professional file naming and workspace organization

This architecture provides a solid foundation for professional bioinformatics analysis with excellent maintainability, extensibility, and reliability.
