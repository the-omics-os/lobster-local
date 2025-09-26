# User Guide Overview

## How Lobster AI Works

Lobster AI is a **multi-agent bioinformatics analysis platform** that combines specialized AI agents with proven scientific tools to analyze complex multi-omics data. Instead of requiring users to learn complex software interfaces or programming languages, Lobster AI allows researchers to interact with their data using **natural language**.

### Core Philosophy

**Natural Language Interface**: Simply describe what you want to do with your data:
- "Analyze the single-cell RNA-seq data and identify cell types"
- "Compare gene expression between treatment groups using DESeq2"
- "Generate a quality control report for my proteomics data"
- "Find datasets similar to mine in GEO database"

**Agent-Based Architecture**: Each agent specializes in specific analysis types:
- **Single-Cell Expert**: Handles scRNA-seq analysis, clustering, cell annotation
- **Bulk RNA-seq Expert**: Performs differential expression with pyDESeq2
- **MS Proteomics Expert**: Analyzes mass spectrometry data with database search
- **Affinity Proteomics Expert**: Processes Olink and antibody array data
- **Data Expert**: Manages file loading, format conversion, and GEO downloads
- **Research Agent**: Mines literature and identifies relevant datasets

### How It Works

1. **Load Your Data**: Use simple commands like `/read data.h5ad` or ask "Load my single-cell data"
2. **Natural Language Analysis**: Describe your analysis goals in plain English
3. **Agent Coordination**: The system routes your request to the appropriate specialist agent
4. **Scientific Processing**: Agents use established bioinformatics tools (scanpy, DESeq2, etc.)
5. **Interactive Results**: View results, generate plots, and iterate on analysis

### Key Features

#### Multi-Modal Data Support
- **Single-cell RNA-seq**: 10X, H5AD, CSV formats
- **Bulk RNA-seq**: Count matrices, normalized data
- **Mass Spectrometry Proteomics**: MaxQuant, Spectronaut output
- **Affinity Proteomics**: Olink NPX, antibody arrays
- **Multi-omics**: Integrated analysis across data types

#### Professional Analysis Workflows
- **Quality Control**: Automated QC metrics and visualizations
- **Normalization**: Method-appropriate normalization strategies
- **Statistical Analysis**: Proper statistical testing with FDR correction
- **Visualization**: Publication-quality interactive plots
- **Reproducibility**: Complete analysis provenance tracking

#### Advanced Capabilities
- **Literature Integration**: Automatic parameter extraction from publications
- **GEO Database Access**: Download and analyze public datasets
- **Cloud/Local Flexibility**: Seamless switching between execution modes
- **Formula-Guided Analysis**: R-style statistical formulas for complex designs

## Understanding Agent Responses

### Agent Communication Patterns

**Clarifying Questions**: Agents may ask for clarification:
```
"I see you have single-cell data. Would you like me to:
1. Perform quality control analysis
2. Identify cell clusters and types
3. Find differentially expressed genes
4. All of the above in a complete workflow?"
```

**Status Updates**: Agents provide progress information:
```
"Loading data... ✓
Calculating QC metrics... ✓
Filtering low-quality cells... ✓
Normalizing expression data... ✓"
```

**Recommendations**: Agents suggest next steps:
```
"Analysis complete! Based on your data characteristics, I recommend:
- Examining cluster markers for cell type annotation
- Running trajectory analysis for developmental processes
- Performing differential expression between conditions"
```

### Understanding Analysis Results

#### Data Summaries
Agents provide structured summaries of your data:
- **Shape**: Number of observations (cells/samples) × variables (genes/proteins)
- **Quality Metrics**: Missing values, outliers, batch effects
- **Processing Status**: What analysis steps have been completed

#### Statistical Results
Results include appropriate statistical context:
- **Significance Testing**: P-values with multiple testing correction
- **Effect Sizes**: Log fold changes, confidence intervals
- **Sample Sizes**: Power calculations and adequacy assessments

#### Visualizations
Plots are automatically generated with:
- **Scientific Accuracy**: Proper scaling, error bars, statistical annotations
- **Publication Quality**: High-resolution, well-labeled plots
- **Interactivity**: Zoom, pan, hover information in HTML plots

### Natural Language Interaction Patterns

#### Effective Communication

**Be Specific About Goals**:
- ✅ "Compare gene expression between control and treatment groups"
- ❌ "Analyze my data"

**Provide Context**:
- ✅ "I have single-cell RNA-seq data from mouse liver samples"
- ❌ "Here's my data file"

**Ask for Explanations**:
- ✅ "Why did you choose these normalization parameters?"
- ✅ "Can you explain the statistical test you used?"

#### Common Request Types

**Exploratory Analysis**:
- "Give me an overview of this dataset"
- "What does the data quality look like?"
- "Show me the main patterns in the data"

**Specific Analysis**:
- "Find differentially expressed genes between conditions"
- "Identify cell types in this single-cell data"
- "Perform pathway enrichment analysis"

**Comparative Analysis**:
- "Compare my results to similar studies"
- "Find public datasets like mine"
- "How do these results compare to the literature?"

**Method Guidance**:
- "What's the best normalization method for this data?"
- "How should I handle batch effects?"
- "What statistical test is appropriate here?"

## Working with Results

### Data Management
- **Modalities**: Data is organized by biological modality (transcriptomics, proteomics, etc.)
- **Provenance**: Complete history of analysis steps and parameters
- **Versioning**: Multiple processing stages saved with descriptive names

### Visualization System
- **Interactive Plots**: HTML plots with zoom, pan, hover information
- **Static Exports**: PNG versions for publications
- **Plot History**: All generated plots saved and accessible
- **Custom Styling**: Scientific color schemes and layouts

### Export and Sharing
- **Data Packages**: Complete analysis bundles with data, plots, and metadata
- **Session Export**: Save and restore analysis sessions
- **Publication Formats**: Export in formats suitable for papers and presentations

## Getting Started Tips

### First Steps
1. **Load Data**: Start with `/read filename` or describe your data
2. **Explore**: Ask "What does this data look like?" or use `/data` command
3. **Analyze**: Describe your research question in natural language
4. **Iterate**: Refine analysis based on results and agent suggestions

### Best Practices
- **Start Broad**: Begin with exploratory analysis before specific tests
- **Ask Questions**: Agents are designed to explain their methods and reasoning
- **Iterate Gradually**: Build analysis step-by-step rather than all at once
- **Save Progress**: Use `/save` to preserve important analysis states

### Common Workflows
1. **Data Loading** → **Quality Control** → **Normalization** → **Analysis** → **Visualization**
2. **Literature Review** → **Parameter Selection** → **Statistical Testing** → **Validation**
3. **Exploratory Analysis** → **Hypothesis Formation** → **Targeted Testing** → **Results Integration**

### When Things Go Wrong
- **Check Data Format**: Ensure files are in supported formats
- **Verify File Paths**: Use absolute paths or check current directory
- **Review Error Messages**: Agents provide detailed error explanations
- **Ask for Help**: Use `/help` or ask "How do I..." questions

## Advanced Features

### Multi-Agent Coordination
Agents automatically hand off tasks to specialists:
- Data loading requests go to the Data Expert
- Statistical analysis goes to appropriate domain expert
- Literature searches go to the Research Agent

### Cloud Integration
- **Seamless Switching**: Same interface for local and cloud execution
- **Scalability**: Handle larger datasets in cloud environment
- **Collaboration**: Share analyses across teams

### Extensibility
- **Custom Workflows**: Combine multiple analysis types
- **Parameter Optimization**: Agents suggest optimal settings
- **Method Comparison**: Evaluate different analytical approaches

This overview provides the conceptual foundation for using Lobster AI. For detailed command references and specific workflows, see the following sections of this user guide.