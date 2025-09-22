# Data Expert Agent Enhancements v2.2+

## Overview

The Data Expert Agent has been significantly enhanced to provide comprehensive workspace management capabilities alongside its core data acquisition and processing functions. This document outlines the new features, resolved duplications, and improved workflows available in v2.2+.

## Key Enhancements

### 🔄 New Workspace Restoration Tool

The data expert agent now includes a powerful `restore_workspace_datasets` tool that enables seamless session continuation and flexible dataset loading.

#### Tool Signature
```python
@tool
def restore_workspace_datasets(pattern: str = "recent") -> str:
    """
    Restore datasets from workspace based on pattern matching.

    This tool loads previously saved datasets back into memory from the workspace.
    Useful for continuing analysis sessions or loading specific datasets.
    """
```

#### Pattern Matching Options

| Pattern | Description | Example |
|---------|-------------|---------|
| `"recent"` | Load most recently used datasets (default) | `restore_workspace_datasets("recent")` |
| `"all"` | Load all available datasets | `restore_workspace_datasets("all")` |
| `"*"` | Load all datasets (same as "all") | `restore_workspace_datasets("*")` |
| `"<dataset_name>"` | Load specific dataset by name | `restore_workspace_datasets("geo_gse123456")` |
| `"<partial_name>*"` | Load datasets matching partial name | `restore_workspace_datasets("geo_*")` |

#### Features
- **Intelligent Loading**: Only loads datasets not already in memory
- **Memory Management**: Respects system memory constraints
- **Detailed Reporting**: Provides comprehensive summaries of loaded datasets
- **Error Handling**: Graceful handling of missing or corrupted datasets
- **Provenance Tracking**: Logs all restoration operations for audit trails

### 📋 Enhanced Function Documentation

The main `data_expert()` function now includes comprehensive documentation describing its role as a multi-omics data acquisition specialist:

```python
def data_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "data_expert_agent",
    handoff_tools: List = None
):
    """
    Create a multi-omics data acquisition, processing, and workspace management specialist agent.

    This expert agent serves as the primary interface for all data-related operations in the
    Lobster bioinformatics platform, specializing in:

    - **GEO Data Acquisition**: Fetching, validating, and downloading datasets from NCBI GEO
    - **Local File Processing**: Loading and validating custom data files with automatic format detection
    - **Workspace Management**: Restoring previous sessions and managing dataset persistence
    - **Multi-modal Integration**: Handling transcriptomics, proteomics, and other omics data types
    - **Quality Assurance**: Ensuring data integrity through schema validation and provenance tracking

    Built on the modular DataManagerV2 architecture, this agent provides seamless integration
    with downstream analysis workflows while maintaining professional scientific standards.
    """
```

### 🗂️ Resolved Command Duplication

**Problem**: Both `/workspace load` and `/restore` commands provided identical functionality, calling the same `client.data_manager.restore_session(pattern)` method. This created user confusion and maintenance overhead.

**Solution**:
- Removed `/workspace load <name>` from CLI help text
- Users now use the more flexible `/restore <pattern>` command
- Maintained all existing functionality while eliminating confusion

**Migration Guide**:
```bash
# Old (deprecated)
/workspace load gse123456_combined

# New (recommended)
/restore gse123456_combined
```

### 🎯 Updated System Prompt

The agent's system prompt has been enhanced to include workspace restoration as a core responsibility:

#### Core Tasks (Updated)
```
0. **Fetching metadata** and give a summary to the supervisor
1. **Download and load datasets** from various sources (GEO, local files, etc.)
2. **Process and validate data** using appropriate modality adapters
3. **Store data as modalities** with proper schema enforcement
4. **Restore workspace datasets** from previous sessions for continued analysis  ← NEW
5. **Provide data access** to other agents via modality names
6. **Maintain workspace** with proper organization and provenance tracking
```

## Usage Examples

### Session Continuation Workflow

```python
# Check what's currently loaded
list_available_modalities()

# Restore recent datasets for continued analysis
restore_workspace_datasets("recent")

# Load specific dataset by name
restore_workspace_datasets("geo_gse123456")

# Load all datasets matching pattern
restore_workspace_datasets("geo_*")

# Verify restored data and continue analysis
get_data_summary()
```

### Agent Integration Example

```python
# In your analysis workflow
user_request = "Continue analysis from yesterday's session"

# Data expert automatically handles restoration
response = data_expert_agent.invoke({
    "messages": [{"role": "user", "content": user_request}]
})

# Agent will:
# 1. Check current workspace state
# 2. Identify available datasets
# 3. Restore appropriate datasets based on context
# 4. Provide summary for continued analysis
```

### Advanced Pattern Matching

```python
# Load all single-cell datasets
restore_workspace_datasets("*single_cell*")

# Load all datasets from specific experiment
restore_workspace_datasets("experiment_batch_2*")

# Load all GEO datasets from specific series
restore_workspace_datasets("geo_gse*")
```

## Tool Response Format

The `restore_workspace_datasets` tool provides rich, structured responses:

```
Successfully restored 3 dataset(s) from workspace!

📊 **Loaded Datasets:**
  • **geo_gse123456**: 5,000 obs × 20,000 vars
  • **geo_gse123457**: 3,200 obs × 18,500 vars
  • **custom_liver_study**: 1,800 obs × 15,000 vars

💾 **Total Size**: 45.2 MB
⚡ **Pattern Used**: geo_*

✅ All restored datasets are now available as modalities for analysis.
```

## Error Handling

The tool provides helpful guidance when datasets are not found:

```
No datasets matched pattern 'nonexistent_dataset'.

Available datasets: 5 total
  • geo_gse123456 (12.3 MB)
  • geo_gse123457 (8.7 MB)
  • custom_liver_study (24.2 MB)

💡 **Try these patterns:**
  • "recent" - Load most recently used datasets
  • "all" - Load all available datasets
  • "<dataset_name>" - Load specific dataset
  • "geo_*" - Load all GEO datasets
```

## Integration Points

### DataManagerV2 Integration
- Uses `data_manager.restore_session(pattern)` for core functionality
- Integrates with `data_manager.available_datasets` for discovery
- Leverages `data_manager.log_tool_usage()` for provenance tracking

### Agent Ecosystem
- **Research Agent**: Can request data expert to restore specific datasets for analysis
- **Analysis Agents**: Automatically get access to restored modalities
- **Supervisor**: Coordinates workspace restoration based on user context

### CLI Integration
- Removed duplicate `/workspace load` command from help text
- Enhanced `/restore` command remains the primary interface
- Maintains autocomplete and progress indicators

## Migration from Previous Versions

### For Users
```bash
# v2.1 and earlier
/workspace load my_dataset

# v2.2+
/restore my_dataset
```

### For Developers
```python
# v2.1 and earlier - manual restoration
client.data_manager.restore_session("my_dataset")

# v2.2+ - agent-mediated restoration
data_expert_agent.invoke({
    "messages": [{"role": "user", "content": "Load my_dataset for analysis"}]
})
```

## Best Practices

### Pattern Selection
- Use `"recent"` for typical session continuation
- Use specific names when you know exactly what you need
- Use patterns (`geo_*`) for bulk operations
- Use `"all"` cautiously due to memory implications

### Memory Management
- Monitor system resources when using `"all"` pattern
- Consider loading datasets incrementally for large workspaces
- Use `list_available_modalities()` to check current memory usage

### Workflow Integration
- Always check current state with `get_data_summary()` before restoration
- Verify restored datasets with `list_available_modalities()`
- Use descriptive dataset names for easier pattern matching

## Troubleshooting

### Common Issues

**Dataset Not Found**
```
Problem: restore_workspace_datasets("my_dataset") returns "No datasets matched"
Solution: Check available datasets with list_available_modalities() and verify spelling
```

**Memory Issues**
```
Problem: System runs out of memory when loading all datasets
Solution: Use more specific patterns or load incrementally
```

**Permission Errors**
```
Problem: Cannot access workspace files
Solution: Verify workspace directory permissions and path configuration
```

## Future Enhancements

### Planned Features (v2.3+)
- **Selective Loading**: Load only specific components of large datasets
- **Smart Caching**: Intelligent memory management with LRU eviction
- **Cross-Session Analytics**: Track usage patterns for better defaults
- **Batch Operations**: Concurrent loading of multiple datasets

### API Extensions
- **Streaming Restoration**: Progressive loading with real-time feedback
- **Conditional Loading**: Load datasets based on analysis requirements
- **Workspace Analytics**: Detailed usage statistics and recommendations

## Summary

The Data Expert Agent enhancements in v2.2+ provide:

✅ **New Capabilities**: Flexible workspace restoration with pattern matching
✅ **Resolved Confusion**: Eliminated duplicate CLI commands
✅ **Enhanced Documentation**: Comprehensive agent role description
✅ **Improved Workflows**: Streamlined session continuation
✅ **Better Integration**: Seamless agent ecosystem coordination

These improvements make the Lobster AI platform more user-friendly and maintainable while preserving all existing functionality.