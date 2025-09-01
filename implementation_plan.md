# Lobster Streamlit App - Feature Parity Implementation Plan

## Overview
Single sentence describing the overall goal.

Upgrade the Streamlit app to achieve complete feature parity with the CLI interface, enabling all command functionalities, proper mode switching, enhanced data visualization, and comprehensive file operations.

## Current State Analysis

### Working Features in Streamlit:
- Basic chat interface with agent queries
- File upload functionality
- Session management (save, export, reset)
- Basic configuration (mode selection, reasoning toggle)
- Plot display with download options
- Workspace file listing

### Missing/Limited Features:
1. **Shell Commands** - No support for cd, ls, cat, etc.
2. **Advanced Slash Commands** - Limited command set
3. **Proper Mode Switching** - Configuration changes don't reinitialize client
4. **Plot Management** - No plot history, search, or file manager integration
5. **Metadata Display** - Limited metadata and modality information
6. **Terminal Integration** - Import issue with TerminalCallbackHandler

## Types
Single sentence describing the type system changes.

No new types required, but will enhance existing dictionary structures for command processing, plot metadata, and session state management.

## Files
Single sentence describing file modifications.

Primary modifications to streamlit_app.py with potential creation of helper modules for command processing and UI components.

### Files to Modify:
- `lobster/streamlit_app.py` - Main application file requiring comprehensive updates
- `lobster/utils/__init__.py` - Already exports TerminalCallbackHandler correctly

### Potential New Files:
- `lobster/utils/streamlit_commands.py` - Command processor for slash and shell commands
- `lobster/utils/streamlit_ui.py` - Reusable UI components

## Functions
Single sentence describing function modifications.

Add new command processing functions, enhance existing display functions, and create proper mode switching logic.

### New Functions to Add:
- `process_slash_command(command: str, client: AgentClient, data_manager: DataManagerV2)` - Main command processor
- `execute_shell_command(command: str, current_dir: Path)` - Shell command executor
- `change_mode(new_mode: str, session_state: SessionState)` - Proper mode switching
- `display_metadata_details(data_manager: DataManagerV2)` - Enhanced metadata display
- `display_modalities_info(data_manager: DataManagerV2)` - Modality information display
- `display_plots_manager(data_manager: DataManagerV2)` - Plot history and management
- `display_workspace_status(data_manager: DataManagerV2)` - Comprehensive workspace info

### Functions to Modify:
- `init_session_state()` - Add current_directory tracking
- `init_client()` - Ensure proper callback initialization
- `display_sidebar()` - Add more command buttons and displays
- `display_chat_interface()` - Add command processing logic

## Classes
Single sentence describing class modifications.

Enhance StreamlitCallbackHandler to properly extend TerminalCallbackHandler and add command processor class.

### Classes to Modify:
- `StreamlitCallbackHandler` - Fix inheritance and imports

### New Classes:
- `CommandProcessor` - Centralized command processing logic
- `PlotManager` - Plot history and operations management

## Dependencies
Single sentence describing dependency modifications.

No new package dependencies required, only proper imports from existing modules.

## Testing
Single sentence describing testing approach.

Manual testing of each command type, mode switching scenarios, and file operations to ensure CLI parity.

### Test Scenarios:
1. Test all slash commands (/help, /status, /data, /metadata, etc.)
2. Test shell commands (cd, ls, cat, mkdir, etc.)
3. Test mode switching with client reinitialization
4. Test plot management and export
5. Test file operations and viewing

## Implementation Order
Single sentence describing the implementation sequence.

Phased implementation starting with core infrastructure fixes, then command processing, followed by UI enhancements.

1. **Phase 1: Fix Imports and Callbacks** (Immediate)
   - Fix TerminalCallbackHandler import
   - Ensure proper callback initialization

2. **Phase 2: Command Processing Infrastructure** (Priority High)
   - Add command processor
   - Implement slash command handler
   - Add shell command support

3. **Phase 3: Mode Switching** (Priority High)
   - Implement proper change_mode function
   - Ensure client reinitialization

4. **Phase 4: Enhanced Data Display** (Priority Medium)
   - Add metadata display
   - Add modality information
   - Enhance workspace status

5. **Phase 5: Plot Management** (Priority Medium)
   - Add plot history viewer
   - Implement plot search/filter
   - Add export options

6. **Phase 6: File Operations** (Priority Low)
   - Add file browser
   - Implement file operations
   - Add syntax highlighting for file viewing

## Implementation Details

### Command Structure
```python
SLASH_COMMANDS = {
    '/help': show_help,
    '/status': show_status,
    '/data': show_data_summary,
    '/metadata': show_metadata_details,
    '/workspace': show_workspace_info,
    '/modalities': show_modalities,
    '/plots': show_plot_history,
    '/plot': manage_plots,
    '/files': list_files,
    '/read': read_file,
    '/save': save_state,
    '/export': export_session,
    '/modes': list_modes,
    '/mode': change_mode,
    '/reset': reset_conversation,
    '/clear': clear_screen,
}

SHELL_COMMANDS = ['cd', 'pwd', 'ls', 'cat', 'mkdir', 'touch', 'cp', 'mv', 'rm']
```

### Session State Additions
```python
st.session_state.current_directory = Path.cwd()
st.session_state.command_history = []
st.session_state.plot_history = []
st.session_state.shell_enabled = False
```

## Success Criteria
- All CLI commands work in Streamlit
- Mode switching properly reinitializes client
- Plot management matches CLI functionality
- File operations work seamlessly
- Enhanced data/metadata display
- Consistent user experience between CLI and web interface
