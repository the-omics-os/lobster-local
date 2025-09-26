# Developer Overview - Lobster AI Architecture

## 🏗️ Overview

This guide provides a comprehensive introduction to developing within the Lobster AI codebase, covering architecture patterns, design principles, and development workflows. Lobster AI is a professional multi-agent bioinformatics analysis platform that combines specialized AI agents with proven scientific tools.

## 🎯 Core Design Principles

### 1. Agent-Based Architecture
- **Specialized Agents**: Each agent handles specific bioinformatics domains (single-cell, bulk RNA-seq, proteomics)
- **Centralized Registry**: Single source of truth for agent configuration via `AGENT_REGISTRY`
- **Natural Language Interface**: Users describe analyses in plain English

### 2. Modular Service Design
- **Stateless Services**: All analysis services are stateless and return `(processed_adata, statistics_dict)`
- **Separation of Concerns**: Agents coordinate workflows, services handle computation
- **Reusable Components**: Services can be used independently or composed in workflows

### 3. Multi-Modal Data Management
- **DataManagerV2**: Centralized orchestrator for multi-omics data with modality management
- **Professional Naming**: Consistent naming conventions for dataset versions and analysis stages
- **Provenance Tracking**: W3C-PROV compliant analysis history for reproducibility

### 4. Cloud/Local Hybrid Architecture
- **BaseClient Interface**: Consistent API for local and cloud execution
- **Seamless Switching**: Automatic detection and fallback between cloud and local modes
- **Unified CLI**: Single interface supporting both execution environments

## 🏛️ Architecture Components

### Core Directories

```
lobster/
├── agents/          # Specialized AI agents for bioinformatics domains
├── core/            # Data management, client infrastructure, interfaces
├── tools/           # Stateless analysis services
├── config/          # Configuration management and agent registry
├── cli.py           # Modern terminal interface with autocomplete
└── utils/           # Shared utilities and logging
```

### Key Architectural Patterns

#### 1. **Agent Registry Pattern**
```python
# lobster/config/agent_registry.py
@dataclass
class AgentRegistryConfig:
    name: str                          # Unique identifier
    display_name: str                  # Human-readable name
    description: str                   # Agent capabilities
    factory_function: str             # Module path to factory
    handoff_tool_name: Optional[str]  # Auto-generated tool name

AGENT_REGISTRY = {
    'data_expert_agent': AgentRegistryConfig(...),
    'singlecell_expert_agent': AgentRegistryConfig(...),
    # ... more agents
}
```

#### 2. **Service Pattern**
```python
class QualityService:
    """Stateless service for data quality assessment."""

    def assess_quality(self, adata: anndata.AnnData, **params) -> Tuple[anndata.AnnData, Dict]:
        """
        Returns:
            Tuple of (processed_adata, statistics_dict)
        """
        # Stateless processing logic
        return processed_adata, statistics
```

#### 3. **Agent Tool Pattern**
```python
@tool
def assess_data_quality(modality_name: str, **params) -> str:
    """Standard pattern for all agent tools."""
    # 1. Validate modality exists
    if modality_name not in data_manager.list_modalities():
        raise ModalityNotFoundError(f"Modality '{modality_name}' not found")

    # 2. Get data and call stateless service
    adata = data_manager.get_modality(modality_name)
    result_adata, stats = service.assess_quality(adata, **params)

    # 3. Store results with descriptive naming
    new_modality = f"{modality_name}_quality_assessed"
    data_manager.modalities[new_modality] = result_adata

    # 4. Log operation for provenance
    data_manager.log_tool_usage("assess_data_quality", params, stats)

    return formatted_response(stats, new_modality)
```

#### 4. **Client Adapter Pattern**
```python
# lobster/core/interfaces/base_client.py
class BaseClient(ABC):
    @abstractmethod
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass

# Implementations: AgentClient (local), CloudLobsterClient (cloud)
```

## 🔧 Development Setup

### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd lobster

# Install development dependencies
make dev-install

# Activate environment
source .venv/bin/activate

# Verify installation
python -m lobster --help
```

### 2. Required Environment Variables
```bash
# Required API Keys
export OPENAI_API_KEY="your-openai-api-key"
export AWS_BEDROCK_ACCESS_KEY="your-aws-access-key"
export AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key"

# Optional
export NCBI_API_KEY="your-ncbi-api-key"
export LOBSTER_CLOUD_KEY="your-cloud-api-key"  # Enables cloud mode
```

### 3. Development Commands
```bash
# Run all tests
make test

# Fast parallel testing
make test-fast

# Code formatting
make format

# Linting
make lint

# Type checking
make type-check

# Start CLI
lobster chat
```

## 🧪 Scientific Workflows

### Professional Naming Convention
```
geo_gse12345                          # Raw downloaded data
├── geo_gse12345_quality_assessed     # QC metrics added
├── geo_gse12345_filtered_normalized  # Preprocessed data
├── geo_gse12345_doublets_detected    # Doublet annotations
├── geo_gse12345_clustered           # Leiden clustering + UMAP
├── geo_gse12345_markers              # Differential expression
├── geo_gse12345_annotated           # Cell type annotations
└── geo_gse12345_pseudobulk          # Aggregated for DE analysis
```

### Data Flow Architecture
```
User Input (CLI)
    ↓
LobsterClientAdapter → BaseClient (AgentClient | CloudLobsterClient)
    ↓
Agent Registry → Specialized Agent (data_expert, singlecell_expert, etc.)
    ↓
Agent Tools → Stateless Services (QualityService, ClusteringService, etc.)
    ↓
DataManagerV2 → Modality Management → Storage Backends (H5AD, MuData)
    ↓
Results → CLI Response with Visualizations
```

## 🎨 Code Style Guidelines

### 1. Python Standards
- Follow PEP 8 style guidelines
- Use type hints for all functions and methods
- Line length: 88 characters (Black formatting)
- Comprehensive docstrings for all public functions

### 2. Scientific Accuracy
- Prioritize scientific accuracy over performance optimizations
- Include comprehensive QC metrics at each analysis step
- Support batch effect detection and correction
- Implement proper missing value handling strategies

### 3. Error Handling
```python
# Use specific exceptions
class ModalityNotFoundError(Exception):
    pass

class ServiceError(Exception):
    pass

# Proper error handling in tools
try:
    result = service.process(data)
except ServiceError as e:
    logger.error(f"Service error: {e}")
    return f"Analysis failed: {str(e)}"
```

## 🚀 Development Workflow

### 1. Adding New Features
1. **Design First**: Consider how the feature fits into existing patterns
2. **Use Registry**: For agents, add to `AGENT_REGISTRY` instead of manual graph edits
3. **Follow Patterns**: Use established service, tool, and adapter patterns
4. **Test Thoroughly**: Include unit, integration, and scientific validation tests
5. **Document**: Update relevant documentation files

### 2. Code Quality Checklist
- [ ] Type hints on all functions
- [ ] Comprehensive docstrings
- [ ] Error handling with specific exceptions
- [ ] Unit tests with 80%+ coverage
- [ ] Integration tests with real data
- [ ] Scientific validation where applicable
- [ ] CLI compatibility (local and cloud)

### 3. Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 📊 Performance Considerations

### 1. Memory Management
- Use memory-efficient data loading for large datasets
- Implement lazy loading where possible
- Monitor memory usage in long-running analyses

### 2. Computation Optimization
- Leverage GPU acceleration when available (ScVI, rapids)
- Use efficient algorithms for large-scale data
- Implement progress tracking for long operations

### 3. Caching Strategy
- File operations: 60s cache for cloud, 10s for local
- Intelligent caching for expensive computations
- Clear cache invalidation strategies

## 🔍 Debugging and Troubleshooting

### 1. Common Issues
- **Import Errors**: Check environment activation and dependencies
- **Agent Registry**: Verify factory function paths are correct
- **Data Loading**: Check file permissions and formats
- **Cloud Integration**: Verify API keys and network connectivity

### 2. Debugging Tools
```python
# Use structured logging
from lobster.utils.logger import get_logger
logger = get_logger(__name__)

# Enable debug mode
logger.setLevel(logging.DEBUG)

# Check system status
lobster chat
/status
```

### 3. Testing Connectivity
```bash
# Test agent registry
python -c "from lobster.config.agent_registry import AGENT_REGISTRY; print(list(AGENT_REGISTRY.keys()))"

# Test CLI with both clients
LOBSTER_CLOUD_KEY="" python -m lobster chat  # Local mode
LOBSTER_CLOUD_KEY="key" python -m lobster chat  # Cloud mode
```

## 📚 Further Reading

- **[Creating Agents Guide](09-creating-agents.md)** - Detailed agent development
- **[Creating Services Guide](10-creating-services.md)** - Service implementation patterns
- **[Creating Adapters Guide](11-creating-adapters.md)** - Data adapter development
- **[Testing Guide](12-testing-guide.md)** - Comprehensive testing framework
- **[CLAUDE.md](../../CLAUDE.md)** - Complete architectural documentation

## 🎯 Quick Reference

### Key Files to Know
- `lobster/config/agent_registry.py` - Agent configuration registry
- `lobster/core/interfaces/base_client.py` - Client interface definition
- `lobster/core/data_manager_v2.py` - Multi-modal data orchestrator
- `lobster/cli.py` - CLI implementation with autocomplete
- `tests/conftest.py` - Test configuration and fixtures

### Essential Commands
```bash
make dev-install    # Development setup
make test          # Run all tests
lobster chat       # Start interactive CLI
/help              # Show available commands
/status            # System status
/files             # List workspace files
```

This overview provides the foundation for contributing to Lobster AI. Each component follows established patterns that promote consistency, maintainability, and scientific rigor.