# API Reference Overview

## Introduction

The Lobster AI API provides a comprehensive set of interfaces for multi-omics bioinformatics analysis through a professional agent-based architecture. This reference documentation covers all public APIs, classes, and interfaces available in the Lobster AI system.

## API Organization

The Lobster AI API is organized into five main categories:

### 1. Core API (`lobster.core`)
The foundational layer providing data management, client interfaces, and system orchestration:
- **DataManagerV2**: Multi-modal data orchestration with provenance tracking
- **Client Interfaces**: BaseClient, AgentClient, APIAgentClient for local/cloud execution
- **Provenance System**: W3C-PROV compliant tracking of analysis operations
- **Schema Validators**: Data validation for transcriptomics and proteomics

### 2. Agent API (`lobster.agents`)
Specialized AI agents for different analytical domains:
- **SingleCell Expert**: Single-cell RNA-seq analysis with formula-guided differential expression
- **Bulk RNA-seq Expert**: Bulk RNA-seq analysis with pyDESeq2 integration
- **Proteomics Experts**: MS and affinity proteomics analysis (DDA/DIA workflows)
- **Data Expert**: Data loading, quality assessment, and GEO dataset management
- **Research Agent**: Literature mining and computational parameter extraction
- **ML Expert**: Machine learning transformations and model preparation

### 3. Services API (`lobster.tools`)
Stateless analysis services implementing scientific algorithms:
- **Transcriptomics Services**: Preprocessing, clustering, differential analysis
- **Proteomics Services**: Missing value handling, normalization, statistical testing
- **Utility Services**: GEO downloading, publication mining, visualization

### 4. Interface Definitions (`lobster.core.interfaces`)
Abstract base classes defining system contracts:
- **BaseClient**: Client interface for local/cloud consistency
- **IDataBackend**: Storage backend abstraction (H5AD, MuData, cloud)
- **IModalityAdapter**: Data format adapters for different modalities
- **IValidator**: Flexible validation with warnings instead of hard failures

### 5. Configuration API (`lobster.config`)
Agent registry and system configuration:
- **Agent Registry**: Centralized agent configuration and discovery
- **Model Configuration**: Per-agent LLM settings with fallback mechanisms
- **System Settings**: Environment-based configuration management

## Key Design Principles

### Agent-Based Architecture
- **Specialist Agents**: Each agent handles specific biological domains
- **Tool Pattern**: All agent tools follow validate → service → store → log pattern
- **Centralized Registry**: Single source of truth for agent configuration
- **Dynamic Handoffs**: Automatic agent-to-agent task routing

### Modular Data Management
- **Multi-Modal Support**: Unified handling of transcriptomics, proteomics, and future modalities
- **Professional Naming**: Consistent dataset naming conventions throughout pipeline
- **Provenance Tracking**: Complete audit trail of all processing operations
- **Schema Validation**: Type-safe data handling with modality-specific requirements

### Cloud/Local Hybrid Design
- **Interface Consistency**: Same API works for local and cloud execution
- **Graceful Fallback**: Automatic switching between execution modes
- **Unified CLI**: Single command-line interface for all deployment types
- **Session Management**: Consistent state handling across environments

### Scientific Rigor
- **Publication Quality**: All analyses meet scientific publication standards
- **Error Handling**: Comprehensive validation with actionable error messages
- **Reproducibility**: Complete provenance and parameter tracking
- **Best Practices**: Implementation of current bioinformatics standards

## API Conventions

### Method Signatures
All service methods follow the stateless pattern:
```python
def analyze_method(adata: anndata.AnnData, **params) -> Tuple[anndata.AnnData, Dict[str, Any]]
```

### Agent Tool Pattern
All agent tools follow the standard pattern:
```python
@tool
def agent_tool(modality_name: str, **params) -> str:
    # 1. Validate modality exists
    # 2. Call stateless service
    # 3. Store results with descriptive naming
    # 4. Log operation for provenance
    # 5. Return formatted response
```

### Error Handling
- **Specific Exceptions**: Custom exception hierarchy for different error types
- **Validation Results**: Flexible validation with errors, warnings, and info messages
- **Graceful Degradation**: Continue analysis when possible, warn about limitations

### Return Types
- **Services**: Return `Tuple[AnnData, Dict]` with processed data and statistics
- **Agent Tools**: Return formatted strings for LLM consumption
- **Clients**: Return structured dictionaries with success, response, and metadata

## Data Flow Architecture

```
User Input (CLI/API)
       ↓
BaseClient Implementation
       ↓
Agent System (LangGraph)
       ↓
Agent Tools (@tool decorated)
       ↓
Stateless Services
       ↓
DataManagerV2 (storage)
       ↓
Backends (H5AD/MuData/Cloud)
```

## Authentication & Configuration

### Environment Variables
```bash
# Required for LLM access
OPENAI_API_KEY=your-openai-key
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret

# Optional for enhanced features
NCBI_API_KEY=your-ncbi-key
LOBSTER_CLOUD_KEY=your-cloud-key  # Enables cloud mode
```

### Model Configuration
- **Per-Agent Settings**: Each agent can use different LLM configurations
- **Fallback Mechanisms**: Automatic fallback to alternative models
- **Thinking Mode**: Support for reasoning-capable models (Claude, GPT-4)
- **Temperature Control**: Fine-tuned parameters per agent type

## Getting Started

### Basic Client Usage
```python
from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize data manager
data_manager = DataManagerV2(workspace_path="./my_workspace")

# Create client
client = AgentClient(data_manager=data_manager)

# Query the system
result = client.query("Load GSE194247 and perform quality assessment")
```

### Cloud Client Usage
```python
# Set environment variable
os.environ['LOBSTER_CLOUD_KEY'] = 'your-key'

# Import cloud client (external package)
from lobster_cloud.client import CloudLobsterClient

client = CloudLobsterClient()
result = client.query("Analyze my single-cell data")
```

### CLI Usage
```bash
# Interactive mode with autocomplete
lobster chat

# Direct commands
lobster --help
```

## API Documentation Structure

This API reference is organized as follows:

- **[Core API Reference](14-core-api.md)**: DataManagerV2, clients, provenance
- **[Agents API Reference](15-agents-api.md)**: All agent tools and capabilities
- **[Services API Reference](16-services-api.md)**: Analysis services and algorithms
- **[Interfaces API Reference](17-interfaces-api.md)**: Abstract base classes and contracts

Each section provides:
- Class and method signatures with type hints
- Parameter descriptions and expected types
- Return value specifications
- Usage examples and common patterns
- Error conditions and exception handling
- Integration notes for cloud/local environments

## Version Compatibility

This API documentation reflects the current version of Lobster AI. The system maintains:
- **Backward Compatibility**: Existing agent tools remain functional
- **Interface Stability**: Core interfaces follow semantic versioning
- **Deprecation Warnings**: Clear migration paths for deprecated features
- **Cloud Synchronization**: API compatibility between local and cloud versions