# 🦞 Lobster AI - Streamlit Interface

## Overview
This is a comprehensive Streamlit web interface for the Lobster Multi-Agent Bioinformatics System that provides full feature parity with the CLI version.

## Features

### ✅ Complete CLI Feature Parity
- **Chat Interface**: Interactive chat with the multi-agent system
- **Terminal Output**: Real-time display of agent thinking and reasoning
- **File Management**: Browse, download, and upload workspace files
- **Mode Switching**: Switch between development, production, and other modes
- **Data Visualization**: Interactive Plotly charts embedded in responses
- **Command Panel**: All CLI commands available as buttons
- **Session Management**: Export, save, and reset conversations

### 🎯 Key Components

1. **Main Chat Area**
   - Ask questions about bioinformatics data
   - View AI responses with embedded visualizations
   - See agent reasoning in real-time

2. **Terminal Output Panel**
   - Shows agent thinking process
   - Displays tool usage logs
   - Processing step visualization
   - Toggle on/off as needed

3. **Sidebar Features**
   - System status display
   - Mode configuration
   - File browser (data/plots/exports)
   - Upload functionality
   - Download buttons for all files

4. **Quick Commands**
   - Show Status
   - List Files
   - Data Summary
   - Plot History
   - Save State
   - Export Session
   - Reset Conversation

## Installation

### Prerequisites
```bash
# Install required packages if not already installed
pip install streamlit pandas plotly rich
```

### Running the App

1. **Basic Launch**:
```bash
streamlit run lobster/streamlit_app.py
```

2. **With Custom Port**:
```bash
streamlit run lobster/streamlit_app.py --server.port 8080
```

3. **Network Access** (allows access from other machines):
```bash
streamlit run lobster/streamlit_app.py --server.address 0.0.0.0
```

## Usage Guide

### 1. Starting a Session
- The app automatically initializes a workspace in `.lobster_workspace/`
- A unique session ID is generated for tracking
- Default mode is "production" (can be changed in sidebar)

### 2. Loading Data
**Option A: Upload via Sidebar**
- Click "Browse files" in the Upload section
- Supports CSV, TSV, Excel, H5, H5AD formats
- Files are automatically saved to workspace

**Option B: Ask the AI**
- Type: "Load dataset GSE235449" (or any GEO ID)
- The AI will fetch and process the data

### 3. Analyzing Data
Simply chat with the AI:
- "What's in my dataset?"
- "Create a PCA plot"
- "Perform differential expression analysis"
- "Find top variable genes"

### 4. Viewing Agent Reasoning
- Toggle "Show Terminal Output" in sidebar
- Watch agents think through problems
- See which tools are being used

### 5. Managing Files
- All generated files appear in sidebar
- Click 📥 to download any file
- Files are organized by type (data/plots/exports)

### 6. Switching Modes
Available modes (in sidebar Configuration):
- **development**: Fast, lightweight models
- **production**: Balanced performance (default)
- **high-performance**: Enhanced for complex tasks
- **ultra-performance**: Maximum capability
- **cost-optimized**: Efficient models

### 7. Exporting Results
- Click "Export Session" to create a data package
- Includes all data, plots, and analysis history
- Downloads as a ZIP file

## Environment Variables

Create a `.env` file with your API keys:
```env
OPENAI_API_KEY=your-key-here
AWS_BEDROCK_ACCESS_KEY=your-key-here
AWS_BEDROCK_SECRET_ACCESS_KEY=your-key-here
NCBI_API_KEY=your-key-here
```

## Comparison with CLI

| Feature | CLI Command | Streamlit Location |
|---------|------------|-------------------|
| Chat | `lobster chat` | Main chat interface |
| Status | `/status` | Quick Commands → Show Status |
| Files | `/files` | Sidebar file browser |
| Data Info | `/data` | Data Panel (right side) |
| Plots | `/plots` | Plot History button |
| Save | `/save` | Commands → Save State |
| Export | `/export` | Commands → Export Session |
| Reset | `/reset` | Commands → Reset Conversation |
| Mode | `/mode <name>` | Configuration → Operation Mode |
| Upload | N/A | Sidebar → Upload File |
| Download | `/read <file>` | Sidebar → 📥 buttons |

## Troubleshooting

### Port Already in Use
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use a different port
streamlit run lobster/streamlit_app.py --server.port 8502
```

### Memory Issues
- The app caches data in session state
- Click "Reset Conversation" to clear memory
- Restart the app if needed

### Missing Dependencies
```bash
# Install all required packages
pip install -r requirements.txt
```

### API Key Errors
- Ensure `.env` file exists with valid keys
- Check AWS credentials are configured
- Verify NCBI API key is set

## Features in Detail

### Terminal Output Visualization
The terminal panel shows:
- 🤖 Agent activation and handoffs
- 💭 Agent reasoning (if enabled)
- 🔧 Tool usage and results
- ✓ Completion status with timing
- 🔄 Agent-to-agent communication

### File Management
- **Data Files**: Processed datasets, matrices
- **Plot Files**: HTML and PNG visualizations
- **Export Files**: Complete analysis packages
- All files downloadable with one click

### Real-time Updates
- Terminal output updates as agents work
- Plots appear immediately upon generation
- File browser refreshes automatically
- Status indicators show current state

## Advanced Features

### Custom Workspace
Set a custom workspace directory:
```python
# In streamlit_app.py, modify init_session_state():
workspace_path = Path("/your/custom/path") / ".lobster_workspace"
```

### Callback Customization
Adjust terminal output verbosity:
```python
# In init_client() function:
callback = StreamlitCallbackHandler(
    terminal_container=terminal_container,
    show_reasoning=True,  # Toggle reasoning display
    verbose=True,         # Toggle detailed output
    max_length=500       # Limit output length
)
```

## Support

For issues or questions:
1. Check the terminal output for detailed error messages
2. Review the sidebar status panel
3. Export session for debugging
4. Refer to main Lobster documentation

---

🦞 **Lobster AI** v2.0 | Multi-Agent Bioinformatics System | © 2025 homara AI
