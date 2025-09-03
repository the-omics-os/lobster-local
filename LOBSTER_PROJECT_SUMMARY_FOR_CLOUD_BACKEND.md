# Lobster AI Project Summary for Cloud Backend Development

## 🎯 Executive Summary

Lobster AI is a sophisticated multi-agent bioinformatics analysis platform with a complete open-source implementation and partial cloud integration framework. This document provides a comprehensive overview for developing the cloud backend that will seamlessly integrate with the existing local system.

## 📊 Current Project Status

### ✅ **What's Complete**
- **Full Local Implementation**: Complete bioinformatics platform with all analysis capabilities
- **Cloud Client Interface**: Client-side code ready to consume cloud APIs
- **Unified Interface Pattern**: BaseClient abstract class ensuring compatibility
- **Automatic Detection**: CLI automatically detects cloud keys and switches modes
- **Fallback Mechanism**: Graceful fallback from cloud to local mode

### 🔨 **What Needs Cloud Backend Implementation**
- **REST API Endpoints**: Server-side implementation of all client methods
- **Agent Orchestration**: Cloud-hosted LangGraph multi-agent system
- **Data Storage**: Cloud workspace and file management
- **Authentication**: API key validation and user management
- **Scalability**: Infrastructure for handling multiple concurrent sessions

## 🏗️ Architecture Overview

### Local System (Complete Implementation)
```
lobster/cli.py → lobster/core/client.py → lobster/agents/graph.py → lobster/core/data_manager_v2.py
     ↓                ↓                        ↓                           ↓
 User Input → AgentClient → Multi-Agent Graph → DataManagerV2 → Analysis Results
```

### Cloud System (Client Ready, Backend Needed)
```
lobster/cli.py → lobster_cloud/client.py → [CLOUD API NEEDED] → AWS Infrastructure
     ↓                ↓                         ↓                    ↓
 User Input → CloudLobsterClient → HTTP/REST Calls → Cloud Backend → Analysis Results
```

## 📁 Project Structure Analysis

### Core Open Source Components

#### **1. Entry Points**
- **`lobster/cli.py`** - Main CLI with cloud detection logic (✅ Complete)
- **`lobster/streamlit_app.py`** - Web UI with cloud support (✅ Complete)
- **`lobster/__main__.py`** - Module entry point (✅ Complete)

#### **2. Client Layer**
- **`lobster/core/client.py`** - Local AgentClient implementation (✅ Complete)
- **`lobster/core/interfaces/base_client.py`** - Unified client interface (✅ Complete)
- **`lobster_cloud/client.py`** - Cloud client consuming REST APIs (✅ Complete)

#### **3. Agent System (Local Implementation)**
- **`lobster/agents/graph.py`** - Multi-agent LangGraph coordination (✅ Complete)
- **`lobster/agents/supervisor.py`** - Supervisor agent orchestration (✅ Complete)
- **`lobster/agents/data_expert.py`** - Data loading and management expert (✅ Complete)
- **`lobster/agents/singlecell_expert.py`** - Single-cell RNA-seq specialist (✅ Complete)
- **`lobster/agents/bulk_rnaseq_expert.py`** - Bulk RNA-seq specialist (✅ Complete)
- **`lobster/agents/research_agent.py`** - Literature and dataset discovery (✅ Complete)
- **`lobster/agents/method_expert.py`** - Computational parameter extraction (✅ Complete)
- **`lobster/agents/proteomics_expert.py`** - Proteomics analysis specialist (✅ Complete)

#### **4. Data Management System**
- **`lobster/core/data_manager_v2.py`** - Modular multi-omics data orchestration (✅ Complete)
- **`lobster/core/adapters/`** - Modality-specific data loading (✅ Complete)
- **`lobster/core/backends/`** - Storage backend abstractions (✅ Complete)
- **`lobster/core/schemas/`** - Data validation and quality control (✅ Complete)

#### **5. Analysis Services**
- **`lobster/tools/preprocessing_service.py`** - Data filtering and normalization (✅ Complete)
- **`lobster/tools/quality_service.py`** - Quality control assessment (✅ Complete)
- **`lobster/tools/clustering_service.py`** - Clustering and dimensionality reduction (✅ Complete)
- **`lobster/tools/enhanced_singlecell_service.py`** - Advanced single-cell analysis (✅ Complete)
- **`lobster/tools/visualization_service.py`** - Publication-ready plotting (✅ Complete)
- **`lobster/tools/geo_service.py`** - GEO database integration (✅ Complete)
- **`lobster/tools/publication_service.py`** - Literature mining (✅ Complete)

#### **6. Configuration System**
- **`lobster/config/agent_config.py`** - Agent model configuration (✅ Complete)
- **`lobster/config/agent_registry.py`** - Centralized agent registry (✅ Complete)
- **`lobster/config/settings.py`** - Environment and settings management (✅ Complete)

## 🔌 Cloud Integration Interface

### BaseClient Interface Contract

The cloud backend must implement REST endpoints that correspond to these client methods:

```python
class BaseClient(ABC):
    # CORE METHODS - Must be implemented in cloud API
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]
    
    @abstractmethod
    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]
    
    @abstractmethod
    def read_file(self, filename: str) -> Optional[str]
    
    @abstractmethod
    def write_file(self, filename: str, content: str) -> bool
    
    @abstractmethod
    def get_conversation_history(self) -> List[Dict[str, str]]
    
    @abstractmethod
    def reset(self) -> None
    
    @abstractmethod
    def export_session(self, export_path: Optional[Path] = None) -> Path
    
    # OPTIONAL METHODS - Cloud-specific features
    def get_usage(self) -> Dict[str, Any]  # Usage tracking
    def list_models(self) -> Dict[str, Any]  # Available models
```

### Required Cloud API Endpoints

Based on the CloudLobsterClient implementation, the cloud backend needs these REST endpoints:

#### **1. Core Analysis Endpoint**
```
POST /query
Request Body:
{
  "query": "user input text",
  "session_id": "cloud_session_20250109_143022",
  "options": {
    "stream": false,
    "reasoning": true,
    "workspace": "/path/to/workspace"
  }
}

Response Format:
{
  "success": true,
  "response": "AI agent response text",
  "session_id": "cloud_session_20250109_143022",
  "has_data": true,
  "plots": [{"id": "plot_1", "title": "UMAP", "figure": {...}}],
  "duration": 15.2,
  "last_agent": "singlecell_expert_agent"
}
```

#### **2. Status and Health Endpoints**
```
GET /status
Response:
{
  "status": "online",
  "success": true,
  "version": "2.0.0",
  "session_id": "...",
  "message_count": 5,
  "has_data": true,
  "workspace": "/cloud/workspace/path"
}

GET /usage
Response:
{
  "api_key": "masked",
  "requests_today": 45,
  "requests_month": 1250,
  "compute_hours": 12.5,
  "data_processed_gb": 5.2
}

GET /models
Response:
{
  "available_models": ["claude-sonnet", "claude-opus", "gpt-4"],
  "current_model": "claude-sonnet",
  "region": "us-east-1"
}
```

#### **3. Workspace Management Endpoints**
```
GET /workspace/files?pattern=*
Response:
{
  "files": [
    {
      "name": "analysis_results.h5ad",
      "path": "/workspace/data/analysis_results.h5ad", 
      "size": 1048576,
      "modified": "2025-01-09T14:30:22Z"
    }
  ]
}

GET /workspace/files/{filename}
Response:
{
  "content": "file contents as string",
  "success": true
}

PUT /workspace/files/{filename}
Request Body:
{
  "content": "file contents to write"
}
Response:
{
  "success": true,
  "path": "/workspace/files/filename"
}
```

#### **4. Session Management Endpoints**
```
POST /session/reset
Request Body:
{
  "session_id": "cloud_session_20250109_143022"
}
Response:
{
  "success": true,
  "message": "Session reset successfully"
}

GET /session/export?session_id={session_id}
Response:
{
  "session_data": {...},
  "conversation_history": [...],
  "workspace_files": [...],
  "analysis_results": {...}
}
```

## 🤖 Agent System Architecture

### Multi-Agent Graph Structure

The cloud backend needs to replicate this local agent orchestration:

```python
# From lobster/agents/graph.py
def create_bioinformatics_graph(data_manager, checkpointer, callback_handler):
    """
    Creates a LangGraph multi-agent system with:
    - Supervisor agent (orchestrates workflow)
    - Data Expert (handles data loading/management) 
    - Research Agent (literature discovery)
    - Method Expert (parameter extraction)
    - Transcriptomics Expert (RNA-seq analysis)
    - Proteomics Expert (protein analysis)
    - Single-cell Expert (specialized single-cell workflows)
    - Bulk RNA-seq Expert (bulk analysis workflows)
    """
```

### Agent Registry System

The cloud backend should use the same agent registry pattern:

```python
# From lobster/config/agent_registry.py
AGENT_REGISTRY = {
    'data_expert_agent': AgentConfig(...),
    'singlecell_expert_agent': AgentConfig(...),
    'bulk_rnaseq_expert_agent': AgentConfig(...),
    'research_agent': AgentConfig(...),
    'method_expert_agent': AgentConfig(...),
    'proteomics_expert_agent': AgentConfig(...),
}
```

### Analysis Tools Integration

Each agent uses stateless analysis services that should be replicated in the cloud:

#### **Data Processing Services**
- **PreprocessingService**: Cell/sample filtering, normalization, batch correction
- **QualityService**: QC metrics, outlier detection, quality assessment
- **ClusteringService**: Leiden clustering, PCA, UMAP visualization
- **VisualizationService**: Publication-ready plots and figures

#### **Domain-Specific Services**
- **EnhancedSingleCellService**: Doublet detection, cell annotation, trajectory analysis
- **BulkRNASeqService**: Differential expression, pathway analysis
- **GEOService**: Dataset downloading, metadata parsing
- **PublicationService**: Literature search, method extraction

## 📊 Data Management Requirements

### DataManagerV2 Cloud Replication

The cloud backend needs to replicate DataManagerV2 functionality:

```python
# Core capabilities that must be available in cloud
class DataManagerV2:
    # Modality management
    def load_modality(name, source, adapter, **kwargs) -> AnnData
    def save_modality(name, path, backend, **kwargs) -> str
    def get_modality(name) -> AnnData
    def list_modalities() -> List[str]
    
    # Quality control
    def get_quality_metrics(modality) -> Dict[str, Any]
    def validate_modalities(strict=False) -> Dict[str, ValidationResult]
    
    # Machine learning preparation
    def prepare_ml_features(modality, **params) -> Dict[str, Any]
    def create_ml_splits(modality, target_column, **params) -> Dict[str, Any]
    
    # Visualization and export
    def add_plot(plot, title, source, **metadata) -> str
    def get_latest_plots(n=None) -> List[Dict[str, Any]]
    def create_data_package(output_dir) -> str
    
    # Workspace management
    def list_workspace_files() -> Dict[str, List[Dict[str, Any]]]
    def auto_save_state() -> List[str]
    def get_workspace_status() -> Dict[str, Any]
```

### Modality Adapters

The cloud needs to support these data loading adapters:

```python
# From lobster/core/adapters/
TranscriptomicsAdapter:
  - transcriptomics_single_cell (10X, H5AD, CSV)
  - transcriptomics_bulk (CSV, TSV, Excel)

ProteomicsAdapter:
  - proteomics_ms (mass spectrometry data)
  - proteomics_affinity (antibody arrays, immunoassays)
```

### Storage Backends

Cloud storage should replicate:

```python
# From lobster/core/backends/
H5ADBackend: Single-modality AnnData storage
MuDataBackend: Multi-modal integrated storage (if MuData available)
```

## 🔧 Analysis Capabilities to Replicate

### Single-Cell RNA-seq Pipeline
```
1. Quality Assessment → assess_data_quality()
2. Cell Filtering → filter_and_normalize_modality()
3. Doublet Detection → detect_doublets_in_modality()
4. Clustering → cluster_modality()
5. Marker Gene Discovery → find_marker_genes_for_clusters()
6. Cell Type Annotation → annotate_cell_types()
7. Trajectory Analysis → run_trajectory_analysis()
```

### Bulk RNA-seq Pipeline
```
1. Quality Assessment → assess_bulk_data_quality()
2. Normalization → normalize_bulk_counts()
3. Differential Expression → run_bulk_differential_expression()
4. Pathway Analysis → run_pathway_analysis()
5. Visualization → create_bulk_visualizations()
```

### Proteomics Pipeline
```
1. Missing Value Analysis → analyze_missing_values()
2. Normalization → normalize_proteomics_data()
3. Statistical Analysis → run_proteomics_statistical_tests()
4. Protein Network Analysis → analyze_protein_networks()
```

### Multi-Omics Integration
```
1. Data Alignment → align_multi_omics_samples()
2. Joint Analysis → run_multi_omics_integration()
3. Cross-Platform Validation → validate_multi_omics_results()
```

## 🛠️ Technical Implementation Details

### Required Cloud Infrastructure Components

#### **1. FastAPI Web Server**
```python
# Cloud backend should implement
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="Lobster Cloud API", version="2.0.0")

class QueryRequest(BaseModel):
    query: str
    session_id: str
    options: Dict[str, Any] = {}

@app.post("/query")
async def process_query(request: QueryRequest):
    # Implement multi-agent processing
    # Return standardized response format
```

#### **2. Session Management**
```python
# Cloud sessions should track:
{
  "session_id": "cloud_session_20250109_143022",
  "user_api_key": "masked_key",
  "conversation_history": [...],
  "workspace_path": "/cloud/workspaces/{session_id}/",
  "active_modalities": [...],
  "created_at": "2025-01-09T14:30:22Z",
  "last_activity": "2025-01-09T15:45:30Z"
}
```

#### **3. Data Storage Architecture**
```python
# Cloud workspace structure
/cloud/workspaces/{session_id}/
├── data/                    # User datasets and processed files
├── plots/                   # Generated visualizations
├── exports/                 # Analysis packages
├── cache/                   # Temporary processing files
└── metadata/                # Session and provenance data
```

### Authentication and Authorization

#### **API Key Management**
```python
# Cloud backend needs
class APIKeyValidator:
    def validate_key(self, api_key: str) -> Dict[str, Any]:
        # Validate API key
        # Return user info and permissions
        
    def check_usage_limits(self, api_key: str) -> bool:
        # Check if user has exceeded limits
        
    def log_api_usage(self, api_key: str, operation: str, compute_time: float):
        # Track usage for billing/monitoring
```

## 🔄 Cloud-Local Interface Specifications

### Request/Response Format Standardization

#### **Query Response Format (Critical for Compatibility)**
```python
# Both local and cloud must return this exact format
{
    "success": bool,                    # True if query processed successfully
    "response": str,                    # AI agent's response text
    "session_id": str,                  # Session identifier
    "has_data": bool,                   # Whether data is currently loaded
    "plots": List[Dict[str, Any]],      # Generated visualizations
    "duration": float,                  # Processing time in seconds
    "last_agent": Optional[str],        # Which agent provided response
    "error": Optional[str]              # Error message if success=False
}
```

#### **Status Response Format**
```python
# Local format (AgentClient.get_status())
{
    "session_id": str,
    "message_count": int,
    "has_data": bool,
    "data_summary": Optional[Dict],
    "workspace": str,
    "reasoning_enabled": bool,
    "callbacks_count": int
}

# Cloud format should be similar but adapted
{
    "session_id": str,
    "status": str,                      # "online", "processing", "error"
    "has_data": bool,
    "data_summary": Optional[Dict],
    "workspace": str,
    "message_count": int,
    "success": bool,
    "version": str
}
```

### File Operations Interface

#### **Workspace File Listing**
```python
# GET /workspace/files?pattern=*
# Must return same format as AgentClient.list_workspace_files()
[
    {
        "name": "filename.h5ad",
        "path": "/workspace/data/filename.h5ad",
        "size": 1048576,                # bytes
        "modified": "2025-01-09T14:30:22Z"  # ISO timestamp
    }
]
```

## 🧬 Bioinformatics Domain Knowledge

### Analysis Workflows Cloud Backend Must Support

#### **1. Single-Cell RNA-seq Analysis**
```python
# Typical user workflow the cloud must handle
"Download GSE109564 and perform single-cell analysis"
↓
1. Data Expert: Downloads GEO dataset using GEOService
2. Single-cell Expert: Assesses data quality using QualityService  
3. Single-cell Expert: Filters and normalizes using PreprocessingService
4. Single-cell Expert: Detects doublets using EnhancedSingleCellService
5. Single-cell Expert: Performs clustering using ClusteringService
6. Single-cell Expert: Finds marker genes using analysis tools
7. Single-cell Expert: Annotates cell types using reference databases
8. Supervisor: Coordinates results and generates comprehensive report
```

#### **2. Bulk RNA-seq Analysis**
```python
"Analyze differential expression between treatment groups"
↓
1. Data Expert: Loads user data using appropriate adapter
2. Bulk Expert: Quality control using QualityService
3. Bulk Expert: Normalization using PreprocessingService  
4. Bulk Expert: Differential expression using BulkRNASeqService
5. Bulk Expert: Pathway analysis using enrichment tools
6. Bulk Expert: Visualization using VisualizationService
7. Supervisor: Generates publication-ready report
```

#### **3. Literature-Guided Analysis**
```python
"Find optimal parameters for clustering analysis from recent papers"
↓
1. Research Agent: Literature search using PublicationService
2. Method Expert: Parameter extraction from papers
3. Method Expert: Recommends evidence-based parameters
4. Transcriptomics Expert: Applies parameters to analysis
5. Supervisor: Reports methodology and results
```

### Data Format Support

The cloud backend must handle these input formats:

#### **Transcriptomics Data**
- **10X Genomics**: MTX + barcodes + features files
- **H5AD**: AnnData format (scanpy standard)
- **CSV/TSV**: Gene expression matrices
- **Excel**: Multi-sheet expression data
- **GEO Datasets**: Direct GSE/GPL accession numbers

#### **Proteomics Data**
- **CSV/Excel**: Protein abundance matrices
- **MaxQuant Output**: proteinGroups.txt format
- **Spectronaut**: Protein abundance reports
- **Mass Spectrometry**: Various vendor formats

#### **Metadata Integration**
- **Sample Information**: Treatment, time-point, batch annotations
- **Experimental Design**: Multi-factor experimental setups
- **Clinical Data**: Patient/sample clinical parameters

## 🔒 Security and Privacy Requirements

### Data Handling Standards
- **Encryption**: All data encrypted in transit (HTTPS) and at rest
- **Session Isolation**: Each user session completely isolated
- **Temporary Storage**: Analysis files automatically cleaned up
- **API Key Security**: Keys never logged or exposed in responses
- **Workspace Isolation**: Users cannot access each other's workspaces

### Compliance Considerations
- **GDPR**: EU users should have data residency options
- **HIPAA**: Healthcare data should have appropriate safeguards
- **Audit Logging**: All operations logged for security monitoring
- **Access Controls**: Rate limiting and usage monitoring

## 📈 Performance and Scalability

### Expected Workloads
- **Typical Dataset**: 10,000-100,000 cells, 20,000-30,000 genes
- **Large Dataset**: 500,000+ cells, memory-efficient processing required
- **Analysis Time**: 5-30 minutes for standard workflows
- **Concurrent Users**: Design for 10-100 simultaneous sessions

### Resource Requirements
- **CPU**: Multi-core for scanpy/pandas operations
- **Memory**: 8-32GB for typical datasets, 64GB+ for large datasets
- **Storage**: Temporary (per-session), persistent (results), cache (reference data)
- **Network**: Low latency for interactive responses

## 🎨 Visualization and Plotting

### Plot Generation System

The cloud must replicate the sophisticated plotting capabilities:

```python
# From lobster/core/data_manager_v2.py
def add_plot(
    plot: go.Figure,
    title: str = None,
    source: str = None,
    dataset_info: Dict[str, Any] = None,
    analysis_params: Dict[str, Any] = None
) -> str:
    """
    Cloud backend must store plots with metadata:
    - Plotly Figure objects
    - Comprehensive metadata tracking
    - Professional naming conventions
    - Export to HTML/PNG formats
    """
```

### Plot Types to Support
- **UMAP/t-SNE**: Dimensionality reduction visualizations
- **Violin/Box Plots**: Gene expression distributions
- **Heatmaps**: Expression matrices and correlations  
- **Volcano Plots**: Differential expression results
- **Network Graphs**: Protein-protein interactions
- **Quality Control Plots**: Assessment visualizations

## 🧪 Testing and Validation

### Integration Testing Requirements

The cloud backend should pass these integration tests:

```python
# Test user workflows
def test_complete_single_cell_workflow():
    # Test full pipeline from data upload to results
    
def test_bulk_rnaseq_analysis():
    # Test bulk RNA-seq analysis pipeline
    
def test_literature_guided_analysis():
    # Test research agent + method expert coordination
    
def test_multi_modal_analysis():
    # Test integrated omics analysis

# Test technical requirements  
def test_session_isolation():
    # Ensure user sessions don't interfere
    
def test_concurrent_processing():
    # Multiple users processing simultaneously
    
def test_large_dataset_handling():
    # Memory-efficient processing of large datasets
    
def test_error_recovery():
    # Graceful handling of processing errors
```

## 📋 Implementation Priority Matrix

### **Phase 1: Core API (Essential)**
1. **POST /query** - Main analysis endpoint with agent orchestration
2. **GET /status** - Health check and session status
3. **Session Management** - Create, track, and manage user sessions
4. **Basic Authentication** - API key validation
5. **Error Handling** - Standardized error responses

### **Phase 2: Data Management (Critical)**
1. **Workspace Operations** - File upload, listing, reading
2. **DataManagerV2 Replication** - Modality management
3. **Analysis Services** - Core preprocessing and analysis tools
4. **Plot Generation** - Visualization pipeline
5. **Data Export** - Session and analysis exports

### **Phase 3: Advanced Features (Important)**
1. **Usage Tracking** - Monitor and bill API usage
2. **Model Selection** - Multiple LLM options
3. **Streaming Responses** - Real-time analysis feedback
4. **Advanced Analytics** - ML preparation and complex workflows
5. **Performance Optimization** - Caching and optimization

### **Phase 4: Enterprise Features (Optional)**
1. **Team Collaboration** - Shared workspaces
2. **Data Persistence** - Long-term storage options
3. **Custom Models** - User-specific model fine-tuning
4. **Advanced Security** - SSO, audit logs, compliance features
5. **Monitoring** - Comprehensive observability

## 🚀 Success Metrics

### Technical Success Criteria
- **Response Time**: <30 seconds for standard analyses
- **Uptime**: 99.9% availability
- **Compatibility**: 100% API compatibility with local client
- **Error Rate**: <1% of requests result in unrecoverable errors
- **Scalability**: Support 100+ concurrent users

### User Experience Criteria  
- **Seamless Switching**: Users can switch between local/cloud transparently
- **Feature Parity**: All local features available in cloud
- **Error Messages**: Clear, actionable error messages
- **Documentation**: Complete API documentation for integration
- **Monitoring**: Real-time status and usage visibility

## 📚 Reference Implementation

### Local System as Reference

The complete local implementation in this repository serves as the definitive reference for cloud backend development. Key files to study:

#### **Core Architecture**
- `lobster/agents/graph.py` - Multi-agent orchestration pattern
- `lobster/core/client.py` - Client interface implementation
- `lobster/core/data_manager_v2.py` - Data management patterns

#### **Analysis Implementation**
- `lobster/tools/preprocessing_service.py` - Data processing algorithms
- `lobster/tools/clustering_service.py` - Clustering and visualization
- `lobster/tools/quality_service.py` - Quality control standards

#### **Integration Patterns**
- `lobster/cli.py` - Cloud detection and client initialization
- `lobster_cloud/client.py` - Expected API consumption patterns
- `tests/test_cloud_switching.py` - Integration testing approach

## 🎯 Next Steps for Cloud Backend Development

1. **Set up FastAPI server** with the required endpoints
2. **Implement agent orchestration** using the same LangGraph patterns
3. **Create cloud data management** replicating DataManagerV2 functionality
4. **Add authentication and session management** for multi-user support
5. **Deploy analysis services** with all bioinformatics tools
6. **Test integration** with the existing CloudLobsterClient
7. **Scale infrastructure** for production workloads
8. **Monitor and optimize** for performance and reliability

The local system provides a complete, working reference implementation. The cloud backend should replicate this functionality while adding scalability, multi-user support, and cloud-native features.
