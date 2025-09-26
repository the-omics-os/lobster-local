# CLI Commands Reference

## Overview

Lobster AI provides a rich command-line interface with enhanced features including Tab completion, command history, and context-aware suggestions. The CLI supports both slash commands for system operations and natural language for analysis tasks.

## Getting Started

### Starting Lobster AI

```bash
# Start interactive chat mode
lobster chat

# Start with custom workspace
lobster chat --workspace /path/to/my/workspace

# Enable detailed agent reasoning
lobster chat --reasoning

# Enable verbose output for debugging
lobster chat --verbose

# Start with all debugging features
lobster chat --reasoning --verbose --debug
```

### Single Query Mode

```bash
# Execute a single query and exit
lobster query "Analyze my single-cell data"

# Save output to file
lobster query "Generate QC report" --output results.md

# Use custom workspace
lobster query "Load data.h5ad" --workspace /my/data
```

### API Server Mode

```bash
# Start API server for web interfaces
lobster serve

# Custom host and port
lobster serve --host 0.0.0.0 --port 8080
```

## Interactive Features

### Enhanced Input Capabilities

**Arrow Key Navigation** (requires `prompt-toolkit`):
- **←/→**: Navigate within your input text
- **↑/↓**: Browse command history
- **Ctrl+R**: Reverse search through history
- **Home/End**: Jump to beginning/end of line

**Tab Completion**:
- **Commands**: Type `/` and press Tab to see all commands
- **Files**: Tab completion after `/read`, `/plot`, `/open`
- **Context-Aware**: Smart suggestions based on current context
- **Cloud Integration**: Works with both local and cloud clients

**Command History**:
- **Persistent**: Commands saved between sessions
- **Search**: Use Ctrl+R to find previous commands
- **Edit**: Recall and modify previous commands

### Installation for Enhanced Features

```bash
# Install optional dependency for full features
pip install prompt-toolkit
```

## System Commands

### Help and Information

#### `/help`
Display comprehensive help with all available commands.

```
/help
```

Shows categorized list of commands with descriptions and examples.

#### `/status`
Show current system status including session info, loaded data, and agent configurations.

```
/status
```

**Output includes**:
- Session ID and mode
- Loaded data summary
- Memory usage
- Workspace location

#### `/input-features`
Display available input features and navigation capabilities.

```
/input-features
```

Shows status of Tab completion, arrow navigation, and command history.

### Workspace Management

#### `/workspace`
Show comprehensive workspace information.

```
/workspace
```

**Displays**:
- Workspace path and configuration
- Loaded modalities and backends
- Directory structure and usage

#### `/workspace list`
List all available datasets in workspace without loading them.

```
/workspace list
```

Shows datasets with status (loaded/available), size, shape, and modification date.

#### `/restore [pattern]`
Restore datasets from workspace based on pattern matching.

```
/restore                    # Restore recent datasets (default)
/restore recent            # Same as above
/restore all               # Restore all available datasets
/restore my_dataset        # Restore specific dataset by name
/restore *liver*           # Restore datasets matching pattern
/restore geo_*             # Restore all GEO datasets
```

**Features**:
- Tab completion for dataset names
- Flexible pattern matching support
- Shows loading progress with detailed summaries
- Intelligent memory management
- Session continuation support

**Pattern Options**:
- `recent` - Load most recently used datasets (default)
- `all` or `*` - Load all available datasets
- `<dataset_name>` - Load specific dataset by exact name
- `<partial_name>*` - Load datasets matching partial name pattern

> **Note**: This command replaces the previous `/workspace load` functionality with enhanced pattern matching and better integration with the data expert agent system.

**Parameters**:
- `recent`: Datasets from last session (default)
- `all`: All available datasets
- `pattern`: Glob pattern for selective restoration

### File Operations

#### `/files`
List all files in workspace organized by category.

```
/files
```

**Categories**:
- **Data**: Analysis datasets and input files
- **Exports**: Generated output files
- **Cache**: Temporary and cached files

#### `/tree`
Show directory tree view of current location and workspace.

```
/tree
```

Displays nested folder structure with file counts and sizes.

#### `/read <file>`
Load and analyze files from workspace or current directory.

```
/read data.h5ad                    # Load single file
/read *.h5ad                       # Load all H5AD files
/read data/*.csv                   # Load CSVs from data folder
/read sample_*.h5ad                # Pattern matching
```

**Supported Patterns**:
- `*`: Match any characters
- `?`: Match single character
- `[abc]`: Match any of a, b, or c
- `**`: Recursive directory matching

**Features**:
- Tab completion for file names
- Automatic format detection
- Batch loading with progress tracking
- Format conversion on-the-fly

#### `/open <file>`
Open file or folder in system default application.

```
/open results.pdf                  # Open in default PDF viewer
/open plots/                       # Open directory in file manager
/open .                            # Open current directory
```

Works with workspace files, absolute paths, and relative paths.

### Data Management

#### `/data`
Show comprehensive summary of currently loaded data.

```
/data
```

**For Single Modality**:
- Shape (observations × variables)
- Data type and memory usage
- Quality metrics
- Metadata columns
- Processing history

**For Multiple Modalities**:
- Individual modality summaries
- Combined statistics
- Cross-modality information

#### `/metadata`
Show detailed metadata information including cached GEO data.

```
/metadata
```

**Displays**:
- **Metadata Store**: Cached GEO and external datasets
- **Current Data Metadata**: Active dataset information
- **Validation Results**: Data quality assessments

#### `/modalities`
Show detailed information for each loaded modality.

```
/modalities
```

**For Each Modality**:
- Observation and variable columns
- Data layers and embeddings
- Unstructured annotations
- Shape and memory information

### Visualization

#### `/plots`
List all generated plots with metadata.

```
/plots
```

Shows plot ID, title, source, and creation time for all generated visualizations.

#### `/plot [ID]`
Open plots directory or specific plot.

```
/plot                              # Open plots directory
/plot plot_1                       # Open specific plot by ID
/plot "Quality Control"            # Open plot by title (partial match)
```

**Features**:
- Opens HTML version preferentially (interactive)
- Falls back to PNG if HTML unavailable
- Tab completion for plot IDs and titles

### Session Management

#### `/save`
Save current state including all loaded data and generated plots.

```
/save
```

**Saves**:
- All loaded modalities as H5AD files
- Generated plots in HTML and PNG formats
- Processing log and tool usage history
- Session metadata

#### `/export`
Export complete session data as a comprehensive package.

```
/export
```

Creates timestamped ZIP file with all data, plots, metadata, and analysis history.

#### `/reset`
Reset conversation and clear loaded data (with confirmation).

```
/reset
```

Prompts for confirmation before clearing:
- Conversation history
- Loaded modalities
- Generated plots
- Analysis state

### Configuration

#### `/modes`
List available operation modes with descriptions.

```
/modes
```

**Available Modes**:
- `development`: Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant - fast development
- `production`: Claude 4 Sonnet for all agents, 3.5 Sonnet v2 for assistant - production ready
- `cost-optimized`: Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant - cost optimized

#### `/mode <name>`
Change operation mode and agent configurations.

```
/mode production                   # Switch to production mode
/mode development                 # Use development models
/mode cost-optimized             # Switch to cost-optimized mode
```

**Effects**:
- Updates all agent model configurations
- Adjusts performance and cost parameters
- Maintains current data and session state

### Dashboard and Monitoring

#### `/dashboard`
Show comprehensive system health dashboard.

```
/dashboard
```

**Includes**:
- Core system status
- Resource utilization
- Agent status and health

#### `/workspace-info`
Show detailed workspace overview with recent activity.

```
/workspace-info
```

**Displays**:
- Workspace configuration and paths
- Recent files and data access
- Data loading statistics

#### `/analysis-dash`
Show analysis monitoring dashboard.

```
/analysis-dash
```

**Tracks**:
- Active analysis operations
- Generated visualizations
- Processing performance metrics

#### `/progress`
Show multi-task progress monitor for concurrent operations.

```
/progress
```

Displays active background operations with progress bars and status.

### Utility Commands

#### `/clear`
Clear the terminal screen.

```
/clear
```

#### `/exit`
Exit Lobster AI (with confirmation prompt).

```
/exit
```

## Shell Integration

Lobster AI supports common shell commands directly without the `/` prefix:

### Directory Navigation

```bash
cd /path/to/data                  # Change directory
pwd                               # Print working directory
ls                                # List directory contents with metadata
ls /path/to/folder               # List specific directory
```

### File Operations

```bash
mkdir new_folder                  # Create directory
touch new_file.txt               # Create file
cp source.txt dest.txt           # Copy file
mv old_name.txt new_name.txt     # Move/rename file
rm unwanted_file.txt             # Remove file
```

### File Viewing

```bash
cat data.csv                     # View file with syntax highlighting
open results/                    # Open in file manager (same as /open)
```

**Enhanced Features**:
- **Syntax Highlighting**: Automatic language detection
- **Structured Output**: Tables and formatted displays
- **Rich Metadata**: File sizes, modification dates, types

## Configuration Commands

### Agent Configuration

```bash
# List available model presets
lobster config list-models

# List available testing profiles
lobster config list-profiles

# Show current configuration
lobster config show-config

# Test specific configuration
lobster config test --profile production

# Test specific agent
lobster config test --profile production --agent singlecell_expert

# Create custom configuration interactively
lobster config create-custom

# Generate environment template
lobster config generate-env
```

## Usage Examples

### Common Workflows

#### Starting a New Analysis

```bash
# Start Lobster
lobster chat

# Check existing workspace
/workspace list

# Load previous work or start fresh
/restore recent

# Load new data
/read my_data.h5ad

# Check data status
/data

# Begin analysis
"Analyze this single-cell RNA-seq data and identify cell types"
```

#### Data Exploration

```bash
# Quick data overview
/data

# View metadata
/metadata

# Check file structure
/tree

# Explore analysis options
"What analysis can I do with this data?"
```

#### Visualization Management

```bash
# List all plots
/plots

# Open specific plot
/plot plot_3

# Open plots folder
/plot

# Save current state
/save
```

#### Session Management

```bash
# Check system status
/status

# View workspace info
/workspace

# Export everything
/export

# Clean restart if needed
/reset
```

### Advanced Usage

#### Batch Operations

```bash
# Load multiple files
/read *.h5ad

# Pattern-based restoration
/restore *experiment_2*

# Dataset loading operations
/restore batch_*
```

#### Configuration Switching

```bash
# Check available modes
/modes

# Switch for production analysis
/mode production

# Verify change
/status
```

#### Debugging and Monitoring

```bash
# Start with verbose debugging
lobster chat --verbose --debug

# Monitor system resources
/dashboard

# Track analysis progress
/progress

# View detailed workspace info
/workspace-info
```

## Troubleshooting Commands

### Diagnostic Information

```bash
# System status
/status

# Input capabilities
/input-features

# Workspace health
/workspace

# Data validation
/metadata
```

### Recovery Operations

```bash
# List available data
/workspace list

# Restore from backup
/restore all

# Clear and restart
/reset

# Export before major changes
/export
```

### Performance Optimization

```bash
# Check resource usage
/dashboard

# Switch to lighter mode
/mode cost-optimized

# Monitor active operations
/progress
```

This comprehensive CLI reference covers all available commands and their usage patterns. For analysis-specific workflows, see the Data Analysis Workflows section.