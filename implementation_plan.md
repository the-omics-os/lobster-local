# Implementation Plan: DataManager Modular Refactoring for Multi-Omics Support

## Overview
Transform the monolithic DataManager into a modular, extensible system with separate backends, adapters, and orchestration layers to support multi-omics data integration using MuData, with future-ready support for AWS S3 integration.

## 1. Architecture Design

### 1.1 Core Principles
- **Separation of Concerns**: Adapters handle modality-specific logic, backends handle storage, DataManager orchestrates
- **Per-Modality AnnData**: Each modality gets its own AnnData object with specific schema
- **MuData Integration**: Multi-modal data composed using MuData for cross-modality analysis
- **S3-Ready Design**: Local backends designed for easy S3 integration without code changes

### 1.2 Component Hierarchy
```
DataManager (Orchestrator)
    ├── Modality Adapters (Data Ingestion & Schema)
    │   ├── TranscriptomicsAdapter
    │   └── ProteomicsAdapter
    ├── Data Backends (Storage & Serialization)
    │   ├── H5ADBackend
    │   └── MuDataBackend
    └── Validation & Provenance
        ├── SchemaValidator
        └── ProvenanceTracker
```

## 2. Schema Specifications

### 2.1 Transcriptomics Schema

#### Single-Cell RNA-seq
```python
AnnData Structure:
- X: Raw UMI counts (sparse CSR matrix)
- layers:
    - 'normalized': Normalized counts (target sum normalization)
    - 'log1p': Log-transformed normalized counts
    - 'scaled': Z-scored expression values
- obs: Cell metadata
    - 'cell_id': Unique cell identifier
    - 'sample_id': Sample origin
    - 'batch': Batch identifier
    - 'condition': Experimental condition
    - 'cell_type': Annotated cell type
    - 'n_genes': Number of genes detected
    - 'total_counts': Total UMI counts
    - 'pct_counts_mt': Mitochondrial percentage
- var: Gene metadata
    - 'gene_id': Ensembl ID (primary key)
    - 'gene_symbol': HGNC symbol
    - 'chromosome': Genomic location
    - 'biotype': Gene biotype (protein_coding, lncRNA, etc.)
- obsm: Dimensionality reductions
    - 'X_pca': PCA coordinates
    - 'X_umap': UMAP embedding
    - 'X_tsne': t-SNE embedding (optional)
- uns: Unstructured metadata
    - 'provenance': Processing history
    - 'neighbors': Nearest neighbor graph
    - 'processing_params': Parameters used in processing
```

#### Bulk RNA-seq
```python
AnnData Structure:
- X: Expression matrix (counts or TPM)
- layers:
    - 'counts': Raw counts
    - 'tpm': TPM normalized
    - 'fpkm': FPKM normalized (optional)
- obs: Sample metadata
    - 'sample_id': Unique sample identifier
    - 'condition': Experimental condition
    - 'batch': Sequencing batch
    - 'treatment': Treatment information
- var: Gene metadata (same as single-cell)
```

### 2.2 Proteomics Schema

#### Mass Spectrometry & Affinity Proteomics
```python
AnnData Structure:
- X: Protein abundance matrix (with explicit NaN for missing values)
- layers:
    - 'raw_intensity': Raw intensities
    - 'normalized': Normalized abundances
    - 'imputed': Imputed values (optional)
- obs: Sample metadata
    - 'sample_id': Unique sample identifier
    - 'condition': Experimental condition
    - 'batch': MS run batch
    - 'instrument': MS instrument used
- var: Protein metadata
    - 'uniprot_id': UniProt accession (primary key)
    - 'gene_symbol': Gene symbol
    - 'protein_name': Full protein name
    - 'organism': Species
    - 'sequence_length': Protein length
- uns: Unstructured metadata
    - 'provenance': Processing history
    - 'peptide_to_protein': Peptide mapping table
    - 'ms_params': MS acquisition parameters
    - 'raw_data_uri': Links to raw mzML/mzTab files
```

## 3. File Structure

```
lobster/
├── core/
│   ├── interfaces/
│   │   ├── __init__.py
│   │   ├── backend.py          # IDataBackend interface
│   │   ├── adapter.py          # IModalityAdapter interface
│   │   └── validator.py        # IValidator interface
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseBackend with common functionality
│   │   ├── h5ad_backend.py    # H5AD file backend (S3-ready)
│   │   └── mudata_backend.py  # MuData backend for multi-modal
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py            # BaseAdapter with common functionality
│   │   ├── transcriptomics_adapter.py  # RNA-seq adapter
│   │   └── proteomics_adapter.py       # Proteomics adapter
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── transcriptomics.py  # Transcriptomics schema definitions
│   │   ├── proteomics.py       # Proteomics schema definitions
│   │   └── validation.py       # Flexible validation with warnings
│   ├── data_manager_v2.py      # New modular DataManager
│   └── provenance.py           # Provenance tracking utilities
```

## 4. Implementation Details

### 4.1 Interface Definitions

#### IDataBackend
```python
class IDataBackend(ABC):
    @abstractmethod
    def load(self, path: str, **kwargs) -> anndata.AnnData:
        """Load data from storage."""
        pass
    
    @abstractmethod
    def save(self, adata: anndata.AnnData, path: str, **kwargs) -> None:
        """Save data to storage."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if data exists at path."""
        pass
    
    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete data at path."""
        pass
```

#### IModalityAdapter
```python
class IModalityAdapter(ABC):
    @abstractmethod
    def from_source(self, source: Union[str, pd.DataFrame], **kwargs) -> anndata.AnnData:
        """Convert source data to AnnData with appropriate schema."""
        pass
    
    @abstractmethod
    def validate(self, adata: anndata.AnnData, strict: bool = False) -> ValidationResult:
        """Validate AnnData against modality schema."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return the expected schema for this modality."""
        pass
```

### 4.2 DataManager API

```python
class DataManagerV2:
    def __init__(self, default_backend: str = "h5ad"):
        self.backends = {}
        self.adapters = {}
        self.modalities = {}  # Store multiple modalities
        self.provenance = ProvenanceTracker()
    
    def register_backend(self, name: str, backend: IDataBackend):
        """Register a storage backend."""
        self.backends[name] = backend
    
    def register_adapter(self, name: str, adapter: IModalityAdapter):
        """Register a modality adapter."""
        self.adapters[name] = adapter
    
    def load_modality(self, 
                     name: str, 
                     source: Union[str, pd.DataFrame],
                     adapter: str,
                     validate: bool = True) -> anndata.AnnData:
        """Load data for a specific modality."""
        adapter_instance = self.adapters[adapter]
        adata = adapter_instance.from_source(source)
        
        if validate:
            validation_result = adapter_instance.validate(adata)
            if validation_result.has_errors:
                raise ValueError(f"Validation errors: {validation_result.errors}")
            if validation_result.has_warnings:
                logger.warning(f"Validation warnings: {validation_result.warnings}")
        
        self.modalities[name] = adata
        self.provenance.log_operation(name, "load", {"source": str(source)})
        return adata
    
    def save_modality(self, name: str, path: str, backend: str = None):
        """Save a modality using specified backend."""
        if name not in self.modalities:
            raise ValueError(f"Modality '{name}' not loaded")
        
        backend_name = backend or self.default_backend
        backend_instance = self.backends[backend_name]
        backend_instance.save(self.modalities[name], path)
        self.provenance.log_operation(name, "save", {"path": path, "backend": backend_name})
    
    def to_mudata(self) -> mudata.MuData:
        """Convert all modalities to MuData object."""
        import mudata
        return mudata.MuData(self.modalities)
    
    def get_modality(self, name: str) -> anndata.AnnData:
        """Get a specific modality."""
        return self.modalities.get(name)
```

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Days 1-2)
- [ ] Create interface definitions (backend.py, adapter.py, validator.py)
- [ ] Implement base classes (BaseBackend, BaseAdapter)
- [ ] Set up schema validation system with flexible warnings
- [ ] Create provenance tracking utilities

### Phase 2: Transcriptomics Pipeline (Days 3-4)
- [ ] Implement TranscriptomicsAdapter with schema enforcement
- [ ] Create TranscriptomicsSchema with single-cell and bulk variants
- [ ] Implement H5ADBackend with S3-ready path handling
- [ ] Add support for 10X, CSV, and MTX formats

### Phase 3: Proteomics Pipeline (Days 5-6)
- [ ] Implement ProteomicsAdapter starting with CSV support
- [ ] Create ProteomicsSchema with MS-specific fields
- [ ] Add peptide-to-protein mapping support
- [ ] Implement missingness handling for proteomics data

### Phase 4: MuData Integration (Day 7)
- [ ] Implement MuDataBackend for multi-modal storage
- [ ] Create multi-modal orchestration in DataManagerV2
- [ ] Add cross-modality utility functions
- [ ] Implement modality merging and splitting

### Phase 5: Agent Integration (Days 8-9)
- [ ] Update transcriptomics_expert.py to use DataManagerV2
- [ ] Create proteomics_expert.py agent
- [ ] Update preprocessing_service.py for new API
- [ ] Update quality_service.py for multi-modal support

### Phase 6: Testing & Documentation (Days 10-11)
- [ ] Unit tests for all backends and adapters
- [ ] Integration tests with sample multi-modal data
- [ ] API documentation and usage examples
- [ ] Performance benchmarking

## 6. AWS S3 Integration (Future-Ready)

The system is designed to support S3 with minimal changes:

### S3-Ready Design Patterns
```python
# Current (local) usage:
backend.save(adata, "/data/results/dataset.h5ad")

# Future S3 usage (no API change needed):
backend.save(adata, "s3://my-bucket/results/dataset.h5ad")
```

### Future S3Backend Implementation
```python
class S3Backend(BaseBackend):
    def __init__(self, aws_config: Dict[str, Any]):
        self.s3_client = boto3.client('s3', **aws_config)
        
    def save(self, adata: anndata.AnnData, path: str, **kwargs):
        if path.startswith("s3://"):
            # Parse S3 path and upload
            bucket, key = self._parse_s3_path(path)
            with tempfile.NamedTemporaryFile() as tmp:
                adata.write_h5ad(tmp.name)
                self.s3_client.upload_file(tmp.name, bucket, key)
        else:
            # Fallback to local storage
            super().save(adata, path, **kwargs)
```

## 7. Provenance Tracking

### W3C-PROV-like Structure
```python
provenance_record = {
    "activity_id": "uuid-12345",
    "activity_type": "data_loading",
    "timestamp": "2025-01-22T14:30:00Z",
    "agent": "TranscriptomicsAdapter",
    "inputs": [
        {
            "uri": "file:///data/GSE235449.csv",
            "checksum": "sha256:abc123...",
            "format": "csv"
        }
    ],
    "outputs": [
        {
            "uri": "file:///processed/GSE235449.h5ad",
            "checksum": "sha256:def456...",
            "format": "h5ad"
        }
    ],
    "parameters": {
        "normalization_method": "log1p",
        "target_sum": 10000
    },
    "software_versions": {
        "scanpy": "1.10.2",
        "anndata": "0.9.0",
        "lobster": "0.1.0"
    }
}
```

## 8. Benefits of This Architecture

1. **Modularity**: Easy to add new modalities (metabolomics, imaging) or backends
2. **S3-Ready**: Path-based design allows seamless S3 integration
3. **Standards Compliance**: Follows bioinformatics best practices
4. **Provenance**: Complete reproducibility and audit trail
5. **Flexibility**: Warning-based validation allows data exploration
6. **Scalability**: Backend abstraction enables cloud-native scaling
7. **Maintainability**: Clean separation of concerns

## 9. Key Design Decisions

1. **No Backward Compatibility**: Clean break allows optimal design
2. **MuData as Integration Layer**: Industry-standard for multi-omics
3. **Flexible Validation**: Warnings instead of hard failures
4. **Schema-First Design**: Clear contracts between components
5. **Local-First, Cloud-Ready**: Develop locally, deploy to cloud

## 10. Success Metrics

- All existing transcriptomics workflows function with new system
- Proteomics data can be loaded and processed
- Multi-modal analysis possible via MuData
- Clear provenance trail for all operations
- Performance comparable to current monolithic system
- Easy addition of new modalities without core changes

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Performance overhead from abstraction | Profile and optimize critical paths |
| Complex dependency management | Use dependency injection pattern |
| Schema evolution challenges | Version schemas with migration support |
| MuData limitations | Design adapters to work standalone |

## 12. Next Steps

1. Review and approve this implementation plan
2. Set up development branch
3. Begin Phase 1 implementation
4. Weekly progress reviews
5. Integration testing after each phase
6. Final review and merge

---

This implementation plan provides a comprehensive roadmap for transforming the DataManager into a production-ready, modular system capable of handling multiple omics data types with professional standards and future cloud scalability.
