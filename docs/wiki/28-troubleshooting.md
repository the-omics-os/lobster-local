# Troubleshooting Guide

This comprehensive troubleshooting guide provides solutions to common issues encountered while using Lobster AI. Each problem includes symptoms, causes, and step-by-step solutions.

## Table of Contents

1. [Installation & Setup Issues](#installation--setup-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Analysis Failures](#analysis-failures)
4. [Performance Issues](#performance-issues)
5. [Visualization Problems](#visualization-problems)
6. [Cloud Integration Issues](#cloud-integration-issues)
7. [Agent & Tool Errors](#agent--tool-errors)
8. [Memory & Resource Problems](#memory--resource-problems)
9. [Output & Export Issues](#output--export-issues)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

---

## Installation & Setup Issues

### Issue: Cannot Install Lobster AI

**Symptoms:**
- `pip install` fails with dependency errors
- Python version compatibility issues
- Missing system dependencies

**Solutions:**

#### Check Python Version
```bash
# Verify Python version (requires 3.12+)
python --version

# If using wrong version, create conda environment
conda create -n lobster python=3.12
conda activate lobster
```

#### Clean Installation
```bash
# Remove existing installation
pip uninstall lobster-ai

# Clean install with development dependencies
git clone https://github.com/homara-ai/lobster.git
cd lobster
make clean-install
```

#### Resolve Dependency Conflicts
```bash
# Install with verbose output to see exact error
pip install -e . -v

# If conflicts occur, try constraint file
pip install -e . --constraint constraints.txt

# For conda users
conda env create -f environment.yml
```

### Issue: API Keys Not Working

**Symptoms:**
- "API key not found" errors
- Authentication failures
- Cannot access LLM models

**Solutions:**

#### Check Environment Variables
```bash
# Verify API keys are set
echo $OPENAI_API_KEY
echo $AWS_BEDROCK_ACCESS_KEY
echo $AWS_BEDROCK_SECRET_ACCESS_KEY

# Check .env file exists and is correctly formatted
cat .env
```

#### Fix .env File
```bash
# Create or update .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key-here
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key
NCBI_API_KEY=your-ncbi-api-key-optional
EOF

# Ensure no extra spaces or quotes
```

#### Test API Connection
```bash
# Test with minimal example
python -c "
from lobster.config.settings import get_settings
settings = get_settings()
print('Settings loaded successfully')
"
```

### Issue: CLI Interface Not Working

**Symptoms:**
- Plain text interface instead of Rich CLI
- Missing orange branding
- Arrow keys not working

**Solutions:**

#### Install Enhanced CLI Dependencies
```bash
# Install optional CLI enhancements
pip install prompt-toolkit

# Verify installation
python -c "import prompt_toolkit; print('Enhanced CLI available')"
```

#### Check Terminal Compatibility
```bash
# Test terminal capabilities
echo $TERM
python -c "
import sys
print(f'Terminal: {sys.stdout.isatty()}')
print(f'Colors: {hasattr(sys.stdout, \"isatty\")}')
"
```

#### Force Rich CLI Mode
```bash
# Start with explicit Rich mode
FORCE_RICH=1 lobster chat

# Or disable if causing issues
DISABLE_RICH=1 lobster chat
```

---

## Data Loading Problems

### Issue: Cannot Load Dataset from GEO

**Symptoms:**
- "Dataset not found" errors
- Download timeouts
- Corrupted downloads

**Solutions:**

#### Verify GEO Accession
```bash
🦞 You: "Search for GSE12345 in GEO database and verify it exists"
🦞 You: "Download GSE12345 with verbose output to see detailed progress"
```

#### Handle Network Issues
```bash
# Check internet connectivity
ping ncbi.nlm.nih.gov

# Use alternative download method
🦞 You: "Download GSE12345 using alternative mirror or cached version"

# Manual download and load
🦞 You: "Load local file that I downloaded manually from GEO"
```

#### Clear Cache and Retry
```bash
# Clear GEO cache
rm -rf .geo_cache/

# Retry download
🦞 You: "Download GSE12345 with fresh cache"
```

### Issue: File Format Not Recognized

**Symptoms:**
- "Unknown file format" errors
- Cannot parse file headers
- Encoding issues

**Solutions:**

#### Specify File Format Explicitly
```bash
🦞 You: "Load CSV file with genes as rows and samples as columns"
🦞 You: "Load TSV file with tab separators and first row as header"
🦞 You: "Load Excel file from sheet named 'RNAseq_data'"
```

#### Check File Encoding
```bash
# Check file encoding
file -i your_data.csv

# Convert if needed
iconv -f iso-8859-1 -t utf-8 your_data.csv > your_data_utf8.csv
```

#### Provide File Structure Information
```bash
🦞 You: "This is a count matrix with gene symbols in first column, sample IDs in header row"
🦞 You: "The file has metadata in the first 3 rows, data starts from row 4"
🦞 You: "File uses semicolon separators instead of commas"
```

### Issue: Large Files Won't Load

**Symptoms:**
- Memory errors during loading
- Loading process hangs
- "File too large" messages

**Solutions:**

#### Use Chunked Loading
```bash
🦞 You: "Load large file in chunks of 10000 rows to save memory"
🦞 You: "Subsample 50% of the data for initial exploration"
🦞 You: "Use sparse matrix format to reduce memory usage"
```

#### Optimize File Format
```bash
# Convert to more efficient format
🦞 You: "Convert CSV to H5AD format for faster loading"
🦞 You: "Compress data using sparse matrix representation"
```

#### Increase Available Memory
```bash
# Monitor memory usage
🦞 You: "/dashboard"  # Check system resources

# Use cloud processing for large files
export LOBSTER_CLOUD_KEY="your-api-key"
🦞 You: "Process this large dataset using cloud resources"
```

---

## Analysis Failures

### Issue: No Cells Pass Quality Control

**Symptoms:**
- "0 cells remaining after filtering"
- All cells filtered out
- QC thresholds too strict

**Solutions:**

#### Review QC Thresholds
```bash
🦞 You: "Show QC metric distributions before applying any filters"
🦞 You: "What are the recommended QC thresholds for this data type?"
```

#### Adjust Filtering Parameters
```bash
🦞 You: "Use more lenient QC thresholds: >500 genes per cell and <30% mitochondrial"
🦞 You: "Filter based on median absolute deviation instead of fixed thresholds"
🦞 You: "Show me the effect of different threshold combinations"
```

#### Check Data Quality
```bash
🦞 You: "Is this data extremely low quality or are the thresholds inappropriate?"
🦞 You: "Generate comprehensive QC report with recommendations"
🦞 You: "Compare QC metrics to typical ranges for this experiment type"
```

### Issue: Clustering Produces Poor Results

**Symptoms:**
- All cells in one cluster
- Too many small clusters
- Clusters don't make biological sense

**Solutions:**

#### Optimize Clustering Parameters
```bash
🦞 You: "Test clustering resolutions from 0.1 to 2.0 and show silhouette scores"
🦞 You: "Try different clustering algorithms: Leiden, Louvain, hierarchical"
🦞 You: "Adjust number of neighbors from 5 to 50 and compare results"
```

#### Check Preprocessing
```bash
🦞 You: "Verify that data normalization was applied correctly"
🦞 You: "Check if highly variable genes were identified properly"
🦞 You: "Ensure PCA was computed with appropriate number of components"
```

#### Evaluate Data Quality
```bash
🦞 You: "Generate PCA plot to check for obvious batch effects"
🦞 You: "Show UMAP plot to assess overall data structure"
🦞 You: "Calculate and plot variance explained by each PC"
```

### Issue: No Significantly Differentially Expressed Genes

**Symptoms:**
- All p-values > 0.05
- No genes pass FDR threshold
- Effect sizes very small

**Solutions:**

#### Check Sample Sizes and Power
```bash
🦞 You: "How many samples per group do I have? Is this sufficient for DE analysis?"
🦞 You: "Calculate power analysis for detecting 2-fold changes"
🦞 You: "Show distribution of fold changes even if not significant"
```

#### Adjust Statistical Parameters
```bash
🦞 You: "Use less stringent FDR threshold (0.1 instead of 0.05)"
🦞 You: "Try different statistical methods: DESeq2, edgeR, limma"
🦞 You: "Test for fold change thresholds: |log2FC| > 0.5"
```

#### Investigate Experimental Design
```bash
🦞 You: "Check if experimental conditions are properly balanced"
🦞 You: "Look for confounding factors in sample metadata"
🦞 You: "Generate PCA plot colored by treatment to see separation"
```

---

## Performance Issues

### Issue: Analysis Takes Too Long

**Symptoms:**
- Processes hang for hours
- No progress updates
- System becomes unresponsive

**Solutions:**

#### Monitor Progress
```bash
🦞 You: "/progress"  # Check current operations
🦞 You: "/dashboard"  # Monitor system resources
```

#### Optimize Analysis Parameters
```bash
🦞 You: "Use faster approximate methods for initial exploration"
🦞 You: "Reduce number of genes/cells for testing parameters"
🦞 You: "Enable parallel processing using multiple CPU cores"
```

#### Use Cloud Resources
```bash
# Switch to cloud for intensive analyses
export LOBSTER_CLOUD_KEY="your-api-key"
🦞 You: "Move this analysis to cloud infrastructure for faster processing"
```

### Issue: Memory Errors

**Symptoms:**
- "Out of memory" errors
- System crashes
- Killed processes

**Solutions:**

#### Reduce Memory Usage
```bash
🦞 You: "Convert to sparse matrix format to save memory"
🦞 You: "Process data in smaller chunks"
🦞 You: "Remove unnecessary variables from workspace"
```

#### Optimize Data Types
```bash
🦞 You: "Use int32 instead of int64 for count data"
🦞 You: "Apply gene filtering to reduce matrix size"
🦞 You: "Subsample cells for parameter testing"
```

#### Monitor Memory Usage
```bash
🦞 You: "/dashboard"  # Check memory consumption
🦞 You: "Show memory usage of current datasets"
```

---

## Visualization Problems

### Issue: Plots Not Displaying

**Symptoms:**
- Empty plot windows
- "No plots generated" messages
- Visualization errors

**Solutions:**

#### Check Plot Generation
```bash
🦞 You: "/plots"  # List available plots
🦞 You: "Generate simple scatter plot to test visualization system"
```

#### Verify Data Requirements
```bash
🦞 You: "Do I have the required data for this plot type?"
🦞 You: "Show me the data structure needed for UMAP visualization"
```

#### Regenerate Plots
```bash
🦞 You: "Create UMAP plot with different parameters"
🦞 You: "Generate static plot instead of interactive version"
🦞 You: "Export plot data for external visualization"
```

### Issue: Poor Quality Visualizations

**Symptoms:**
- Overlapping labels
- Unclear color schemes
- Low resolution images

**Solutions:**

#### Improve Plot Parameters
```bash
🦞 You: "Create high-resolution plot (300 DPI) suitable for publication"
🦞 You: "Use distinct colors for better cluster separation"
🦞 You: "Adjust point sizes and transparency for better visibility"
```

#### Customize Appearance
```bash
🦞 You: "Generate plot with custom color palette"
🦞 You: "Create plot with larger fonts for better readability"
🦞 You: "Export plot with editable text for post-processing"
```

---

## Cloud Integration Issues

### Issue: Cloud API Not Working

**Symptoms:**
- Authentication failures
- "Cloud service unavailable"
- Timeout errors

**Solutions:**

#### Verify Cloud Setup
```bash
# Check API key is set
echo $LOBSTER_CLOUD_KEY

# Test cloud connectivity
🦞 You: "/status"  # Should show cloud mode if working
```

#### Switch to Local Mode
```bash
# Temporarily disable cloud
unset LOBSTER_CLOUD_KEY
🦞 You: "Continue analysis in local mode"
```

#### Retry Cloud Connection
```bash
# Re-export API key
export LOBSTER_CLOUD_KEY="your-api-key"
🦞 You: "Test cloud connection and retry analysis"
```

### Issue: Slow Cloud Processing

**Symptoms:**
- Long wait times
- Frequent timeouts
- Poor responsiveness

**Solutions:**

#### Optimize for Cloud
```bash
🦞 You: "Use cloud-optimized analysis parameters"
🦞 You: "Split large analyses into smaller chunks"
```

#### Check Network
```bash
# Test network speed
speedtest-cli

# Use local processing for small analyses
🦞 You: "Process this small dataset locally to save time"
```

---

## Agent & Tool Errors

### Issue: Agent Handoffs Fail

**Symptoms:**
- "Agent not available" errors
- Wrong agent selected
- Tool execution failures

**Solutions:**

#### Check Agent Status
```bash
🦞 You: "/status"  # Check available agents
🦞 You: "List all available agents and their capabilities"
```

#### Explicit Agent Selection
```bash
🦞 You: "Use the single-cell expert to analyze this scRNA-seq data"
🦞 You: "Hand this proteomics task to the MS proteomics expert"
```

#### Restart Session
```bash
# Exit and restart Lobster
🦞 You: "/exit"
lobster chat  # Fresh session
```

### Issue: Tool Execution Errors

**Symptoms:**
- "Tool failed" messages
- Incomplete analysis results
- Error tracebacks

**Solutions:**

#### Check Input Requirements
```bash
🦞 You: "What data is required for this analysis?"
🦞 You: "Verify that my data meets the requirements"
```

#### Use Alternative Tools
```bash
🦞 You: "Try alternative method for this analysis"
🦞 You: "Use simpler version of this analysis"
```

#### Report Detailed Errors
```bash
🦞 You: "Show detailed error message and suggest solutions"
🦞 You: "Generate debug information for this failed analysis"
```

---

## Memory & Resource Problems

### Issue: System Becomes Unresponsive

**Symptoms:**
- High CPU usage
- System freezing
- Slow response times

**Solutions:**

#### Monitor Resources
```bash
🦞 You: "/dashboard"  # Check system status
htop  # Monitor processes externally
```

#### Optimize Resource Usage
```bash
🦞 You: "Kill any running background processes"
🦞 You: "Reduce analysis complexity to save resources"
🦞 You: "Clear workspace cache to free memory"
```

#### Adjust Analysis Settings
```bash
🦞 You: "Use single-threaded processing to reduce CPU load"
🦞 You: "Process data in smaller batches"
```

---

## Output & Export Issues

### Issue: Cannot Export Results

**Symptoms:**
- "Export failed" errors
- Missing output files
- Permission denied errors

**Solutions:**

#### Check File Permissions
```bash
# Verify write permissions
ls -la ./
🦞 You: "Export to a different directory with write permissions"
```

#### Specify Export Format
```bash
🦞 You: "Export results as CSV files"
🦞 You: "Save plots in PNG format instead of SVG"
🦞 You: "Export data in H5AD format for preservation"
```

#### Use Alternative Export Methods
```bash
🦞 You: "/export results"  # Use CLI export command
🦞 You: "Show me the data so I can copy it manually"
```

### Issue: Missing Analysis Results

**Symptoms:**
- "No results found"
- Empty output directories
- Lost analysis history

**Solutions:**

#### Check Analysis Status
```bash
🦞 You: "Show me all completed analyses in this session"
🦞 You: "/data"  # Check loaded datasets
🦞 You: "/files"  # List all available files
```

#### Regenerate Missing Results
```bash
🦞 You: "Re-run the differential expression analysis"
🦞 You: "Recreate the clustering analysis from preprocessed data"
```

#### Access Analysis History
```bash
🦞 You: "Show analysis history and provenance tracking"
🦞 You: "Export session log with all commands and results"
```

---

## Advanced Troubleshooting

### Debug Mode and Logging

#### Enable Verbose Output
```bash
# Start with debug mode
LOBSTER_DEBUG=1 lobster chat

# Check log files
tail -f ~/.lobster/logs/lobster.log
```

#### Capture Error Details
```bash
🦞 You: "Enable detailed error reporting for troubleshooting"
🦞 You: "Show me the complete error traceback"
🦞 You: "Generate diagnostic report for this issue"
```

### Manual Intervention

#### Direct Data Access
```python
# Access data manager directly
from lobster.core.data_manager_v2 import DataManagerV2
from pathlib import Path

dm = DataManagerV2(workspace_path=Path(".lobster_workspace"))
print(dm.list_modalities())

# Inspect specific dataset
adata = dm.get_modality("your_dataset_name")
print(adata.obs.head())
```

#### Service-Level Debugging
```python
# Test individual services
from lobster.tools.preprocessing_service import PreprocessingService

service = PreprocessingService()
# Test service methods directly
```

### Recovery Procedures

#### Workspace Recovery
```bash
# Backup current workspace
cp -r .lobster_workspace .lobster_workspace_backup

# Clean and reinitialize
🦞 You: "Initialize fresh workspace and reload data"
```

#### Session Recovery
```bash
# Save current state
🦞 You: "/export session-state"

# Restart and restore
lobster chat
🦞 You: "/import session-state"
```

---

## Getting Help

### When to Seek Support

Contact support if you encounter:
- Persistent crashes or system instability
- Data corruption or loss
- Reproducible bugs in core functionality
- Performance issues that can't be resolved

### Information to Include

When reporting issues, provide:
1. **System Information**: OS, Python version, Lobster version
2. **Error Messages**: Complete error text and tracebacks
3. **Data Description**: Dataset type, size, source
4. **Reproduction Steps**: Exact commands that trigger the issue
5. **Expected vs Actual Results**: What you expected vs what happened

### Community Resources

- **GitHub Issues**: [Report bugs and feature requests](https://github.com/homara-ai/lobster/issues)
- **Discord Community**: Real-time help and discussion
- **Documentation**: [Complete guide and tutorials](../README.md)
- **Example Notebooks**: Working examples and best practices

### Quick Diagnostic Command

```bash
🦞 You: "Run system diagnostics and generate troubleshooting report"
```

This command generates a comprehensive report including:
- System specifications
- Installation status
- Current workspace state
- Recent error logs
- Performance metrics

---

This troubleshooting guide covers the most common issues encountered in Lobster AI. For additional help, consult the [FAQ](29-faq.md) or reach out to the community through the support channels listed above.