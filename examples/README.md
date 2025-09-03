# 🦞 Lobster AI - Examples

This directory contains example analyses to help you get started with Lobster AI's powerful bioinformatics capabilities.

## 📚 Available Examples

### 🧬 Single-Cell RNA-seq
- **basic_single_cell.py** - Complete pipeline from GEO download to clustering analysis

### 📊 Bulk RNA-seq (Coming Soon)
- **differential_expression.py** - Compare conditions with bulk RNA-seq
- **pathway_analysis.py** - Perform pathway enrichment analysis

### 🧪 Proteomics (Coming Soon)
- **mass_spec_analysis.py** - Analyze mass spectrometry data
- **protein_networks.py** - Build protein interaction networks

### 🔗 Multi-Omics (Coming Soon)
- **integrated_analysis.py** - Combine transcriptomics and proteomics

## 🚀 Quick Start

### **Option 1: Run Example Script**
```bash
# Make sure Lobster AI is installed
git clone https://github.com/homara-ai/lobster.git
cd lobster && make install

# Run the working example
python examples/basic_single_cell.py
```

### **Option 2: Interactive Analysis**
```bash
# Start the interactive interface
lobster chat

# Then try natural language commands like:
# "Download GSE109564 and perform single-cell clustering analysis"
# "Show me the quality metrics for my data"
# "Create a UMAP visualization with cell type annotations"
```

### **Option 3: Programmatic Usage**
```python
from lobster.core.client import AgentClient
from lobster.core.data_manager_v2 import DataManagerV2

# Initialize the system
data_manager = DataManagerV2()
client = AgentClient(data_manager=data_manager)

# Run analysis
result = client.query("Analyze my single-cell data for T cell markers")
print(result['response'])
```

## 🎯 **What Each Example Teaches**

### **basic_single_cell.py**
- GEO dataset downloading and parsing
- Quality control and filtering
- Normalization and preprocessing  
- Clustering with Leiden algorithm
- UMAP dimensionality reduction
- Marker gene identification
- Visualization generation

## 💡 **Getting Help**

- **Interactive Help**: Type `/help` in `lobster chat`
- **Full Documentation**: Check [INSTALLATION.md](../INSTALLATION.md)
- **Community Support**: [Discord](https://discord.gg/homaraai)
- **GitHub Issues**: [Report Problems](https://github.com/homara-ai/lobster/issues)

## 🔧 **Customizing Examples**

All examples are designed to be easily modified:

1. **Change Datasets**: Replace GEO accession numbers with your own
2. **Adjust Parameters**: Modify clustering resolution, QC thresholds
3. **Add Analysis Steps**: Extend with additional bioinformatics methods
4. **Custom Visualizations**: Add your own plotting preferences

## 🌟 **Ready to Analyze?**

Start with `basic_single_cell.py` to see Lobster AI in action, then explore the interactive `lobster chat` interface for your own analyses!
