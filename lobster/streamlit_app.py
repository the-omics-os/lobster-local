# streamlit_app.py
"""
Lobster AI - Streamlit Interface (Enhanced with CLI Feature Parity)
- Full command support (slash and shell commands)
- Proper mode switching with client reinitialization
- Comprehensive data and plot management
- Enhanced terminal integration
"""

import os
import sys
import json
import logging
import subprocess
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from io import StringIO

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# --- Your core stack ---
from lobster.core.client import AgentClient
# Updated to use DataManagerV2 - Modern modular data management
from lobster.core.data_manager_v2 import DataManagerV2
# Fixed import - TerminalCallbackHandler is properly exported from utils
from lobster.utils import TerminalCallbackHandler
from lobster.utils.auth import Auth
from lobster.config.settings import get_settings
from lobster.config.agent_config import get_agent_configurator, initialize_configurator

# -----------------------
# Basic login
# -----------------------
settings = get_settings()
# Initialise CognitoAuthenticator
authenticator = Auth.get_authenticator(settings.SECRETS_MANAGER_ID, settings.REGION)

# Authenticate user, and stop here if not logged in
is_logged_in = authenticator.login()
if not is_logged_in:
    st.stop()

def logout():
    authenticator.logout()

# -----------------------
# Basic setup & styling
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Lobster",
    page_icon="🦞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced UI Styling ---
st.markdown("""
<style>
    /* Enhanced styling for feature parity with CLI */
    .stChatMessage.user {
        background: rgba(37, 99, 235, 0.15);
        border-left: 3px solid #2563eb;
        border-radius: 8px;
        padding: 10px;
    }
    .stChatMessage.assistant {
        background: rgba(148, 163, 184, 0.15);
        border-left: 3px solid #64748b;
        border-radius: 8px;
        padding: 10px;
    }
    .stCodeBlock {
        font-size: 13px !important;
        border-radius: 6px !important;
    }
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
    }
    .command-palette {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }
    .status-item {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
    }
    .status-label {
        font-weight: 600;
        color: #666;
    }
    .status-value {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------
# Logging helpers
# -----------------------
class SimpleLogCollector:
    def __init__(self):
        self.logs = []
        self.step = 0

    def add_log(self, level: str, agent: str, message: str):
        self.logs.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "agent": agent,
            "message": message,
            "step": self.step
        })
        self.step += 1

    def get_formatted_logs(self) -> str:
        if not self.logs:
            return "No logs available"
        icon_map = {"START":"🤖","TOOL":"🔧","COMPLETE":"✅","ERROR":"❌","INFO":"ℹ️","REASONING":"💭","THINKING":"💭"}
        out = []
        for log in self.logs:
            agent = (log["agent"] or "System").replace("_"," ").title()
            icon = icon_map.get(log["level"], "•")
            out.append(f"{log['timestamp']} {icon} {agent}: {log['message']}")
        return "\n".join(out)

    def clear(self):
        self.logs.clear()
        self.step = 0


class StreamlitCallbackHandler(TerminalCallbackHandler):
    """Enhanced callback handler for Streamlit with proper inheritance."""
    
    def __init__(self, log_collector: SimpleLogCollector = None, *args, **kwargs):
        # Create a dummy console for the parent class
        dummy_console = Console(file=StringIO(), force_terminal=False)
        
        # Initialize parent with proper parameters
        super().__init__(
            console=dummy_console,
            verbose=kwargs.get('verbose', True),
            show_reasoning=kwargs.get('show_reasoning', True),
            show_tools=kwargs.get('show_tools', True),
            max_length=kwargs.get('max_length', 500),
            use_panels=False  # Disable panels for Streamlit
        )
        
        # Store log collector
        self.log_collector = log_collector or SimpleLogCollector()
    
    def _display_agent_event(self, event):
        """Override to use log collector instead of console."""
        agent_display = self._format_agent_name(event.agent_name)
        
        if event.type.name == "AGENT_START":
            self.log_collector.add_log("START", event.agent_name, "Starting analysis...")
        elif event.type.name == "AGENT_THINKING" and self.show_reasoning:
            if event.content:
                content = self._truncate_content(event.content)
                self.log_collector.add_log("REASONING", event.agent_name, content)
        elif event.type.name == "AGENT_ACTION":
            if event.content:
                self.log_collector.add_log("INFO", event.agent_name, event.content)
        elif event.type.name == "AGENT_COMPLETE":
            self.log_collector.add_log("COMPLETE", event.agent_name, "Analysis complete")
        elif event.type.name == "HANDOFF":
            from_agent = event.metadata.get("from", "Unknown")
            to_agent = event.metadata.get("to", "Unknown")
            handoff_msg = f"Handoff: {self._format_agent_name(from_agent)} → {self._format_agent_name(to_agent)}"
            self.log_collector.add_log("INFO", "system", handoff_msg)
            if event.content:
                self.log_collector.add_log("INFO", "system", f"Task: {self._truncate_content(event.content)}")
    
    def _display_tool_event(self, event):
        """Override to use log collector instead of console."""
        if not self.show_tools:
            return
        
        tool_name = event.metadata.get("tool_name", "Unknown Tool")
        
        if event.type.name == "TOOL_START":
            self.log_collector.add_log("TOOL", event.agent_name, f"Using tool: {tool_name}")
            if self.verbose and event.content:
                self.log_collector.add_log("INFO", event.agent_name, f"Input: {self._truncate_content(event.content)}")
        elif event.type.name == "TOOL_COMPLETE":
            self.log_collector.add_log("COMPLETE", event.agent_name, f"Tool {tool_name} complete")
            if self.verbose and event.content:
                self.log_collector.add_log("INFO", event.agent_name, f"Result: {self._truncate_content(event.content)}")
        elif event.type.name == "TOOL_ERROR":
            self.log_collector.add_log("ERROR", event.agent_name, f"Tool {tool_name} failed: {event.content}")


# -----------------------
# Command Processing
# -----------------------
class CommandProcessor:
    """Handles slash and shell command processing."""
    
    def __init__(self, client: AgentClient, data_manager: DataManagerV2):
        self.client = client
        self.data_manager = data_manager
        self.current_directory = st.session_state.get('current_directory', Path.cwd())
    
    def process_command(self, command: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """
        Process a command and return (is_command, response, data).
        
        Returns:
            Tuple of (is_command, response_text, additional_data)
        """
        command = command.strip()
        
        # Check for slash commands
        if command.startswith("/"):
            return self.process_slash_command(command)
        
        # Check for shell commands
        shell_commands = ['cd', 'pwd', 'ls', 'cat', 'mkdir', 'touch', 'cp', 'mv', 'rm']
        first_word = command.split()[0] if command else ""
        if first_word in shell_commands:
            return self.process_shell_command(command)
        
        return False, None, None
    
    def process_slash_command(self, command: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Process slash commands."""
        parts = command.lower().strip().split(maxsplit=1)
        cmd = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/help":
            return True, self.show_help(), None
        
        elif cmd == "/status":
            return True, self.show_status(), None
        
        elif cmd == "/data":
            return True, self.show_data_summary(), None
        
        elif cmd == "/metadata":
            return True, self.show_metadata_details(), None
        
        elif cmd == "/workspace":
            return True, self.show_workspace_info(), None
        
        elif cmd == "/modalities":
            return True, self.show_modalities(), None
        
        elif cmd == "/plots":
            return True, self.show_plot_history(), None
        
        elif cmd == "/plot":
            return True, self.manage_plots(args), None
        
        elif cmd == "/files":
            return True, self.list_files(), None
        
        elif cmd == "/read":
            if args:
                return True, self.read_file(args), None
            return True, "Usage: /read <filename>", None
        
        elif cmd == "/save":
            saved = self.data_manager.auto_save_state()
            msg = "✅ Saved: " + ", ".join(saved) if saved else "Nothing to save"
            return True, msg, None
        
        elif cmd == "/export":
            try:
                path = self.client.export_session()
                return True, f"✅ Session exported to: {path}", None
            except Exception as e:
                return True, f"❌ Export failed: {e}", None
        
        elif cmd == "/modes":
            return True, self.list_modes(), None
        
        elif cmd == "/mode":
            if args:
                return True, self.change_mode(args), None
            return True, "Usage: /mode <mode_name>", None
        
        elif cmd == "/reset":
            st.session_state.messages = []
            if self.client:
                self.client.reset()
            return True, "✅ Conversation reset", None
        
        elif cmd == "/clear":
            st.session_state.messages = []
            return True, "✅ Chat cleared", None
        
        else:
            return True, f"❌ Unknown command: {cmd}", None
    
    def process_shell_command(self, command: str) -> Tuple[bool, Optional[str], Optional[Any]]:
        """Process shell commands."""
        parts = command.strip().split()
        if not parts:
            return False, None, None
        
        cmd = parts[0].lower()
        
        try:
            if cmd == "cd":
                if len(parts) == 1:
                    new_dir = Path.home()
                else:
                    target = " ".join(parts[1:])
                    if target == "~":
                        new_dir = Path.home()
                    elif target.startswith("~/"):
                        new_dir = Path.home() / target[2:]
                    else:
                        new_dir = self.current_directory / target if not Path(target).is_absolute() else Path(target)
                    new_dir = new_dir.resolve()
                
                if new_dir.exists() and new_dir.is_dir():
                    self.current_directory = new_dir
                    st.session_state.current_directory = new_dir
                    os.chdir(new_dir)
                    return True, f"📁 {new_dir}", None
                else:
                    return True, f"❌ cd: no such directory: {target}", None
            
            elif cmd == "pwd":
                return True, f"📁 {self.current_directory}", None
            
            elif cmd == "ls":
                target_dir = self.current_directory
                if len(parts) > 1:
                    target_path = parts[1]
                    if target_path.startswith("~/"):
                        target_dir = Path.home() / target_path[2:]
                    else:
                        target_dir = self.current_directory / target_path if not Path(target_path).is_absolute() else Path(target_path)
                
                if target_dir.exists() and target_dir.is_dir():
                    items = list(target_dir.iterdir())
                    if not items:
                        return True, f"📁 Empty directory: {target_dir.name}", None
                    
                    # Format directory listing
                    dirs = sorted([f"📁 {item.name}/" for item in items if item.is_dir()])
                    files = sorted([f"📄 {item.name}" for item in items if item.is_file()])
                    
                    output = f"**Directory: {target_dir.name}**\n\n"
                    if dirs:
                        output += "**Directories:**\n" + "\n".join(dirs) + "\n\n"
                    if files:
                        output += "**Files:**\n" + "\n".join(files)
                    
                    return True, output, None
                else:
                    return True, f"❌ ls: cannot access '{target_dir}': No such directory", None
            
            elif cmd == "cat":
                if len(parts) < 2:
                    return True, "❌ cat: missing file argument", None
                
                file_path = " ".join(parts[1:])
                if not file_path.startswith("/") and not file_path.startswith("~/"):
                    file_path = self.current_directory / file_path
                else:
                    file_path = Path(file_path).expanduser()
                
                try:
                    if file_path.exists() and file_path.is_file():
                        content = file_path.read_text(encoding='utf-8', errors='replace')
                        
                        # Return formatted content
                        return True, f"**📄 {file_path.name}**\n```\n{content}\n```", None
                    else:
                        return True, f"❌ cat: {file_path}: No such file", None
                except Exception as e:
                    return True, f"❌ cat: {file_path}: {e}", None
            
            elif cmd in ["mkdir", "touch", "cp", "mv", "rm"]:
                # Execute the command
                result = subprocess.run(command, shell=True, cwd=self.current_directory,
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    success_msgs = {
                        "mkdir": f"✅ Created directory: {parts[1] if len(parts) > 1 else ''}",
                        "touch": f"✅ Created file: {parts[1] if len(parts) > 1 else ''}",
                        "cp": f"✅ Copied: {parts[1] if len(parts) > 1 else ''} → {parts[2] if len(parts) > 2 else ''}",
                        "mv": f"✅ Moved: {parts[1] if len(parts) > 1 else ''} → {parts[2] if len(parts) > 2 else ''}",
                        "rm": f"✅ Removed: {parts[1] if len(parts) > 1 else ''}"
                    }
                    return True, success_msgs.get(cmd, result.stdout or "✅ Command executed"), None
                else:
                    return True, f"❌ {result.stderr or 'Command failed'}", None
            
            else:
                return False, None, None
                
        except Exception as e:
            return True, f"❌ Error executing command: {e}", None
    
    def show_help(self) -> str:
        """Show help information."""
        return """
## 🦞 Available Commands

### Slash Commands:
- `/help` - Show this help message
- `/status` - Show system status
- `/data` - Show current data summary
- `/metadata` - Show detailed metadata information
- `/workspace` - Show workspace status and information
- `/modalities` - Show detailed modality information
- `/plots` - List all generated plots
- `/plot [ID]` - Open plots or specific plot by ID
- `/files` - List workspace files
- `/read <file>` - Read file from workspace
- `/save` - Save current state to workspace
- `/export` - Export session data
- `/modes` - List available operation modes
- `/mode <name>` - Change operation mode
- `/reset` - Reset conversation
- `/clear` - Clear screen

### Shell Commands:
- `cd <path>` - Change directory
- `pwd` - Print current directory
- `ls [path]` - List directory contents
- `cat <file>` - Display file contents
- `mkdir <dir>` - Create directory
- `touch <file>` - Create file
- `cp <src> <dst>` - Copy file/directory
- `mv <src> <dst>` - Move/rename file/directory
- `rm <file>` - Remove file
"""
    
    def show_status(self) -> str:
        """Show system status."""
        status = self.client.get_status()
        configurator = get_agent_configurator()
        current_mode = configurator.get_current_profile()
        
        output = "## 🦞 System Status\n\n"
        output += f"- **Session ID:** {status['session_id']}\n"
        output += f"- **Mode:** {current_mode}\n"
        output += f"- **Messages:** {status['message_count']}\n"
        output += f"- **Workspace:** {status['workspace']}\n"
        output += f"- **Data Loaded:** {'✓' if status['has_data'] else '✗'}\n"
        
        if status['has_data'] and status['data_summary']:
            summary = status['data_summary']
            output += f"- **Data Shape:** {summary.get('shape', 'N/A')}\n"
            output += f"- **Memory Usage:** {summary.get('memory_usage', 'N/A')}\n"
        
        return output
    
    def show_data_summary(self) -> str:
        """Show data summary."""
        if self.data_manager.has_data():
            summary = self.data_manager.get_data_summary()
            
            output = "## 📊 Current Data Summary\n\n"
            output += f"- **Status:** {summary['status']}\n"
            output += f"- **Shape:** {summary['shape'][0]} × {summary['shape'][1]}\n"
            output += f"- **Memory Usage:** {summary['memory_usage']}\n"
            
            if summary.get('columns'):
                cols_preview = ", ".join(summary['columns'][:5])
                if len(summary['columns']) > 5:
                    cols_preview += f" ... (+{len(summary['columns'])-5} more)"
                output += f"- **Columns:** {cols_preview}\n"
            
            if summary.get('processing_log'):
                output += "\n### Recent Processing Steps:\n"
                for step in summary['processing_log'][-5:]:
                    output += f"- {step}\n"
            
            return output
        else:
            return "No data currently loaded."
    
    def show_metadata_details(self) -> str:
        """Show detailed metadata."""
        output = "## 📋 Metadata Information\n\n"
        
        # Check metadata store
        if hasattr(self.data_manager, 'metadata_store') and self.data_manager.metadata_store:
            output += "### 🗄️ Metadata Store (Cached GEO/External Data):\n\n"
            for dataset_id, metadata_info in self.data_manager.metadata_store.items():
                metadata = metadata_info.get('metadata', {})
                validation = metadata_info.get('validation', {})
                
                output += f"**{dataset_id}**\n"
                output += f"- Title: {metadata.get('title', 'N/A')}\n"
                output += f"- Type: {validation.get('predicted_data_type', 'unknown')}\n"
                output += f"- Samples: {len(metadata.get('samples', {}))}\n"
                output += f"- Cached: {metadata_info.get('fetch_timestamp', 'N/A')}\n\n"
        
        # Show current metadata
        if hasattr(self.data_manager, 'current_metadata') and self.data_manager.current_metadata:
            output += "### 📊 Current Data Metadata:\n\n"
            for key, value in list(self.data_manager.current_metadata.items())[:10]:
                if isinstance(value, (list, dict)):
                    display_value = f"{type(value).__name__} with {len(value)} items"
                else:
                    display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                output += f"- **{key}:** {display_value}\n"
        
        return output if output != "## 📋 Metadata Information\n\n" else "No metadata available."
    
    def show_workspace_info(self) -> str:
        """Show workspace information."""
        if hasattr(self.data_manager, 'get_workspace_status'):
            status = self.data_manager.get_workspace_status()
            
            output = "## 🏗️ Workspace Information\n\n"
            output += f"- **Path:** {status.get('workspace_path', 'N/A')}\n"
            output += f"- **Modalities Loaded:** {status.get('modalities_loaded', 0)}\n"
            output += f"- **Backends:** {', '.join(status.get('registered_backends', []))}\n"
            output += f"- **Adapters:** {', '.join(status.get('registered_adapters', []))}\n"
            output += f"- **Default Backend:** {status.get('default_backend', 'N/A')}\n"
            output += f"- **Provenance:** {'✓' if status.get('provenance_enabled') else '✗'}\n"
            output += f"- **MuData:** {'✓' if status.get('mudata_available') else '✗'}\n"
            
            if status.get('modality_names'):
                output += f"\n### 🧬 Loaded Modalities:\n"
                for modality in status['modality_names']:
                    output += f"- {modality}\n"
            
            return output
        else:
            return "Workspace information not available."
    
    def show_modalities(self) -> str:
        """Show modality information."""
        if hasattr(self.data_manager, 'list_modalities'):
            modalities = self.data_manager.list_modalities()
            
            if modalities:
                output = "## 🧬 Modality Details\n\n"
                
                for modality_name in modalities:
                    try:
                        adata = self.data_manager.get_modality(modality_name)
                        output += f"### {modality_name}\n"
                        output += f"- **Shape:** {adata.n_obs} obs × {adata.n_vars} vars\n"
                        
                        if list(adata.obs.columns):
                            obs_cols = ", ".join(list(adata.obs.columns)[:5])
                            if len(adata.obs.columns) > 5:
                                obs_cols += f" ... (+{len(adata.obs.columns)-5} more)"
                            output += f"- **Obs Columns:** {obs_cols}\n"
                        
                        if list(adata.var.columns):
                            var_cols = ", ".join(list(adata.var.columns)[:5])
                            if len(adata.var.columns) > 5:
                                var_cols += f" ... (+{len(adata.var.columns)-5} more)"
                            output += f"- **Var Columns:** {var_cols}\n"
                        
                        if adata.layers:
                            output += f"- **Layers:** {', '.join(list(adata.layers.keys()))}\n"
                        
                        output += "\n"
                        
                    except Exception as e:
                        output += f"### {modality_name}\n"
                        output += f"- Error accessing modality: {e}\n\n"
                
                return output
            else:
                return "No modalities loaded."
        else:
            return "Modality information not available."
    
    def show_plot_history(self) -> str:
        """Show plot history."""
        plots = self.data_manager.get_plot_history()
        
        if plots:
            output = "## 📊 Generated Plots\n\n"
            for i, plot in enumerate(plots[-10:], 1):  # Show last 10 plots
                output += f"{i}. **{plot['title']}**\n"
                output += f"   - ID: {plot['id']}\n"
                output += f"   - Source: {plot['source']}\n"
                output += f"   - Created: {plot['timestamp']}\n\n"
            return output
        else:
            return "No plots generated yet."
    
    def manage_plots(self, args: str) -> str:
        """Manage plots - open or display specific plot."""
        if not args:
            # Just save plots to workspace
            saved_files = self.data_manager.save_plots_to_workspace()
            if saved_files:
                return f"✅ Saved {len(saved_files)} plot files to workspace"
            else:
                return "No plots to save"
        
        # Find plot by ID or name
        plot_id = args.strip()
        for plot_entry in self.data_manager.latest_plots:
            if plot_entry["id"] == plot_id or plot_id.lower() in plot_entry["title"].lower():
                return f"✅ Found plot: {plot_entry['title']}\nID: {plot_entry['id']}\nCreated: {plot_entry['timestamp']}"
        
        return f"❌ Plot not found: {plot_id}"
    
    def list_files(self) -> str:
        """List workspace files."""
        files = self.data_manager.list_workspace_files()
        
        output = "## 📁 Workspace Files\n\n"
        
        for category, items in files.items():
            if items:
                output += f"### {category.title()} ({len(items)})\n"
                for item in items[:10]:  # Show first 10
                    size_kb = item['size'] / 1024
                    output += f"- **{item['name']}** ({size_kb:.1f} KB)\n"
                if len(items) > 10:
                    output += f"- ... and {len(items)-10} more\n"
                output += "\n"
        
        return output if output != "## 📁 Workspace Files\n\n" else "No files in workspace."
    
    def read_file(self, filename: str) -> str:
        """Read a file from workspace."""
        try:
            content = self.client.read_file(filename)
            if content:
                # Determine file type for syntax highlighting hint
                ext = Path(filename).suffix
                lang_map = {'.py': 'python', '.js': 'javascript', '.json': 'json', 
                           '.md': 'markdown', '.txt': 'text', '.yaml': 'yaml', '.yml': 'yaml'}
                lang = lang_map.get(ext, 'text')
                
                return f"**📄 {filename}**\n```{lang}\n{content}\n```"
            else:
                return f"❌ Could not read file: {filename}"
        except Exception as e:
            return f"❌ Error reading file: {e}"
    
    def list_modes(self) -> str:
        """List available modes."""
        configurator = get_agent_configurator()
        current_mode = configurator.get_current_profile()
        profiles = configurator.list_available_profiles()
        
        output = "## 🎮 Available Modes\n\n"
        
        descriptions = {
            "development": "Fast, lightweight models for development",
            "production": "Balanced performance and cost",
            "high-performance": "Enhanced performance for complex tasks",
            "ultra-performance": "Maximum capability for demanding analyses",
            "cost-optimized": "Efficient models to minimize costs",
            "heavyweight": "Most capable models for all agents",
            "eu-compliant": "EU region models for compliance",
            "eu-high-performance": "High-performance EU region models"
        }
        
        for profile in sorted(profiles.keys()):
            status = " **[ACTIVE]**" if profile == current_mode else ""
            desc = descriptions.get(profile, "")
            output += f"- **{profile}**{status} - {desc}\n"
        
        return output
    
    def change_mode(self, new_mode: str) -> str:
        """Change operation mode."""
        configurator = get_agent_configurator()
        profiles = configurator.list_available_profiles()
        
        if new_mode not in profiles:
            return f"❌ Invalid mode: {new_mode}\n\nAvailable modes: {', '.join(sorted(profiles.keys()))}"
        
        try:
            # Store current settings
            current_workspace = Path(self.client.workspace_path)
            current_reasoning = self.client.enable_reasoning
            
            # Initialize new configurator with the specified profile
            initialize_configurator(profile=new_mode)
            
            # Reinitialize the client with new profile settings
            from rich.console import Console
            console = Console() if st.session_state.callbacks else None
            data_manager = DataManagerV2(
                workspace_path=current_workspace,
                console=console
            )
            
            # Create new client with updated configuration
            new_client = AgentClient(
                data_manager=data_manager,
                session_id=st.session_state.session_id,
                workspace_path=current_workspace,
                enable_reasoning=current_reasoning,
                custom_callbacks=st.session_state.callbacks
            )
            
            # Update session state
            st.session_state.client = new_client
            st.session_state.current_mode = new_mode
            
            return f"✅ Mode changed to: **{new_mode}**"
            
        except Exception as e:
            return f"❌ Failed to change mode: {e}"


# -----------------------
# Session Initialization
# -----------------------
def init_session_state():
    """Initialize session state with enhanced features."""
    if "initialized" in st.session_state:
        return

    # Workspace
    workspace_path = Path.cwd() / ".lobster_workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)

    st.session_state.initialized = True
    st.session_state.workspace_path = workspace_path
    st.session_state.data_manager = DataManagerV2(workspace_path=workspace_path)

    # Chat / agent state
    st.session_state.messages = []
    st.session_state.client = None
    st.session_state.current_mode = "production"
    st.session_state.enable_reasoning = True
    st.session_state.show_terminal = True
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.processing = False
    st.session_state.saved_uploads = {}
    
    # Enhanced features for CLI parity
    st.session_state.current_directory = Path.cwd()
    st.session_state.command_history = []
    st.session_state.plot_history = []

    # Logging callback
    st.session_state.log_collector = SimpleLogCollector()
    st.session_state.callbacks = [
        StreamlitCallbackHandler(
            log_collector=st.session_state.log_collector,
            show_reasoning=st.session_state.enable_reasoning,
            verbose=True
        )
    ]

    # Ensure a default configurator profile exists
    try:
        initialize_configurator(profile=st.session_state.current_mode)
    except Exception as e:
        logger.warning(f"Configurator init warning: {e}")


def init_client() -> AgentClient:
    """Initialize or get the agent client."""
    if st.session_state.client is None:
        st.session_state.client = AgentClient(
            data_manager=st.session_state.data_manager,
            session_id=st.session_state.session_id,
            workspace_path=st.session_state.workspace_path,
            enable_reasoning=st.session_state.enable_reasoning,
            custom_callbacks=st.session_state.callbacks
        )
    return st.session_state.client


def change_mode(new_mode: str, current_client: AgentClient) -> AgentClient:
    """
    Change the operation mode and reinitialize client with the new configuration.
    
    Args:
        new_mode: The new mode/profile to switch to
        current_client: The current AgentClient instance
        
    Returns:
        Updated AgentClient instance
    """
    # Store current settings before reinitializing
    current_workspace = Path(current_client.workspace_path)
    current_reasoning = current_client.enable_reasoning
    
    # Initialize a new configurator with the specified profile
    initialize_configurator(profile=new_mode)
    
    # Reinitialize the client with the new profile settings
    from rich.console import Console
    console = Console() if st.session_state.callbacks else None
    data_manager = DataManagerV2(
        workspace_path=current_workspace,
        console=console
    )
    
    client = AgentClient(
        data_manager=data_manager,
        session_id=st.session_state.session_id,
        workspace_path=current_workspace,
        enable_reasoning=current_reasoning,
        custom_callbacks=st.session_state.callbacks
    )
    
    return client


# -----------------------
# Terminal output
# -----------------------
def display_terminal_output():
    """Display terminal output if enabled."""
    if not st.session_state.show_terminal:
        return
    logs = st.session_state.log_collector.get_formatted_logs()
    if logs and logs != "No logs available":
        with st.expander("🖥️ **Agent Execution Log**", expanded=True):
            st.code(logs, language="text")


# -----------------------
# Enhanced Sidebar
# -----------------------
def display_sidebar():
    """Display enhanced sidebar with CLI features."""
    dm = st.session_state.data_manager
    st.sidebar.markdown("## 🦞 **Lobster AI**")
    st.sidebar.markdown("*Multi-Agent Bioinformatics System*")
    st.sidebar.markdown("---")
    st.button("Logout", "logout_btn", on_click=logout)
    st.sidebar.markdown("---")

    # Quick Commands
    with st.sidebar.expander("⚡ **Quick Commands**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Status", key="quick_status"):
                st.session_state.messages.append({"role": "user", "content": "/status"})
            if st.button("📊 Data", key="quick_data"):
                st.session_state.messages.append({"role": "user", "content": "/data"})
            if st.button("🧬 Modalities", key="quick_mod"):
                st.session_state.messages.append({"role": "user", "content": "/modalities"})
        with col2:
            if st.button("📈 Plots", key="quick_plots"):
                st.session_state.messages.append({"role": "user", "content": "/plots"})
            if st.button("📁 Files", key="quick_files"):
                st.session_state.messages.append({"role": "user", "content": "/files"})
            if st.button("❓ Help", key="quick_help"):
                st.session_state.messages.append({"role": "user", "content": "/help"})

    with st.sidebar.expander("📊 **System Status**", expanded=True):
        st.markdown(
            f"""
            <div class="status-box">
                <div class="status-item"><span class="status-label">Session:</span><span class="status-value">{st.session_state.session_id[:10]}…</span></div>
                <div class="status-item"><span class="status-label">Mode:</span><span class="status-value">{'🟢' if st.session_state.current_mode=='production' else '🔵'} {st.session_state.current_mode}</span></div>
                <div class="status-item"><span class="status-label">Directory:</span><span class="status-value">{st.session_state.current_directory.name}</span></div>
            """,
            unsafe_allow_html=True,
        )
        if dm.has_data():
            summary = dm.get_data_summary()
            shape = summary.get('shape', [0, 0])
            st.markdown(
                f"""<div class="status-item"><span class="status-label">Data:</span><span class="status-value">✓ {shape[0]} × {shape[1]}</span></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="status-item"><span class="status-label">Data:</span><span class="status-value">No data loaded</span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<div class="status-item"><span class="status-label">Plots:</span><span class="status-value">📈 {len(dm.latest_plots)}</span></div></div>""",
            unsafe_allow_html=True,
        )

    with st.sidebar.expander("⚙️ **Configuration**", expanded=False):
        configurator = get_agent_configurator()
        profiles = list(getattr(configurator, "list_available_profiles")().keys())
        
        # Ensure current mode is valid
        if st.session_state.current_mode not in profiles and profiles:
            st.session_state.current_mode = profiles[0]

        selected = st.selectbox("Operation Mode", options=profiles, index=profiles.index(st.session_state.current_mode))
        if selected != st.session_state.current_mode and st.button("🔄 Apply Mode"):
            # Use command processor to change mode
            st.session_state.messages.append({"role": "user", "content": f"/mode {selected}"})

        # Check if reasoning setting changed
        old_reasoning = st.session_state.enable_reasoning
        st.session_state.enable_reasoning = st.checkbox("Show Agent Reasoning", value=st.session_state.enable_reasoning)
        st.session_state.show_terminal = st.checkbox("Show Terminal Output", value=st.session_state.show_terminal)
        
        # If reasoning setting changed, update callback and recreate client
        if old_reasoning != st.session_state.enable_reasoning:
            # Update the callback handler with new reasoning setting
            st.session_state.callbacks = [
                StreamlitCallbackHandler(
                    log_collector=st.session_state.log_collector,
                    show_reasoning=st.session_state.enable_reasoning,
                    verbose=True
                )
            ]
            # Force client recreation with new callbacks
            st.session_state.client = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 **Upload File**")
    uploaded = st.sidebar.file_uploader("Choose a file", type=['csv', 'tsv', 'xlsx', 'xls', 'txt', 'json', 'h5', 'h5ad'])
    if uploaded is not None:
        # De-duplicate saves across reruns
        size = getattr(uploaded, "size", None)
        key = f"{uploaded.name}:{size}"
        if st.session_state.saved_uploads.get(uploaded.name) != size:
            data_dir = st.session_state.workspace_path / "data"
            data_dir.mkdir(exist_ok=True)
            file_path = data_dir / uploaded.name
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state.saved_uploads[uploaded.name] = size
            st.sidebar.success(f"✓ Uploaded: {uploaded.name}")

            # Try load basic tabular
            try:
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(file_path, index_col=0)
                elif uploaded.name.endswith(".tsv"):
                    df = pd.read_csv(file_path, sep="\t", index_col=0)
                elif uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(file_path, index_col=0)
                else:
                    df = None
                if df is not None:
                    st.session_state.data_manager.set_data(df)
                    st.sidebar.success(f"✓ Data loaded: {df.shape}")
            except Exception as e:
                st.sidebar.warning(f"Loaded to workspace (agent can parse). Preview failed: {e}")
        else:
            st.sidebar.info("File already uploaded (skipped re-save).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 **Workspace Files**")
    files = st.session_state.data_manager.list_workspace_files()

    def _render_file_group(label, group_key):
        items = files.get(group_key, [])
        if not items:
            return
        with st.sidebar.expander(f"{label} ({len(items)})", expanded=(group_key == "data")):
            for i, meta in enumerate(items[:5]):  # Show first 5
                name = meta["name"]
                path = Path(meta["path"])
                cols = st.columns([3, 1])
                with cols[0]:
                    st.text(name if len(name) <= 20 else name[:20] + "…")
                with cols[1]:
                    if path.exists():
                        with open(path, "rb") as fh:
                            st.download_button("📥", data=fh.read(), file_name=name, key=f"dl_{group_key}_{i}")

    _render_file_group("📊 Data Files", "data")
    _render_file_group("📈 Plots", "plots")
    _render_file_group("📦 Exports", "exports")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 **Session**")
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("💾 Save"):
            st.session_state.messages.append({"role": "user", "content": "/save"})
    with c2:
        if st.button("📦 Export"):
            st.session_state.messages.append({"role": "user", "content": "/export"})
    with c3:
        if st.button("🧹 Reset"):
            st.session_state.messages.append({"role": "user", "content": "/reset"})


# -----------------------
# Chat Interface
# -----------------------
def display_chat_interface():
    """Display enhanced chat interface with command support."""
    st.markdown("## 🦞 **Lobster AI** – Multi-Agent Bioinformatics System")
    st.caption("by Omics-OS | Type `/help` for commands")
    st.markdown("---")

    client = init_client()

    # Render history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for plot_entry in msg.get("plots", []) or []:
                if isinstance(plot_entry, dict) and "figure" in plot_entry:
                    st.plotly_chart(plot_entry["figure"], use_container_width=True)

    # Input
    new_prompt = st.chat_input("Ask about your data or use commands (/, cd, ls, cat, etc.)…", disabled=st.session_state.processing)

    if new_prompt:
        # Initialize command processor
        cmd_processor = CommandProcessor(client, st.session_state.data_manager)
        
        # Check if it's a command
        is_command, response, data = cmd_processor.process_command(new_prompt)
        
        if is_command:
            # Handle command response
            st.session_state.messages.append({"role": "user", "content": new_prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display immediately
            with st.chat_message("user"):
                st.markdown(new_prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
                if data:  # If there's additional data like plots
                    st.write(data)
            
            # Add to command history
            st.session_state.command_history.append(new_prompt)
            
        else:
            # Regular agent query
            st.session_state.messages.append({"role": "user", "content": new_prompt})
            with st.chat_message("user"):
                st.markdown(new_prompt)

            # Guard against double execution
            if st.session_state.processing:
                st.warning("Already processing a request, please wait…")
                return

            st.session_state.processing = True
            st.session_state.log_collector.clear()

            # Run the agent
            with st.chat_message("assistant"):
                with st.spinner("🦞 Analyzing…"):
                    try:
                        result = client.query(new_prompt, stream=False)

                        if result.get("success"):
                            response = result.get("response", "")
                            st.markdown(response)

                            # Show logs after completion
                            display_terminal_output()

                            # Show plots (if any)
                            plots = st.session_state.data_manager.latest_plots
                            if plots:
                                st.markdown("---")
                                st.markdown("### 📊 Generated Visualizations")
                                for plot_entry in plots:
                                    if isinstance(plot_entry, dict) and "figure" in plot_entry:
                                        if "title" in plot_entry:
                                            st.markdown(f"**{plot_entry['title']}**")
                                        st.plotly_chart(plot_entry["figure"], use_container_width=True)
                                        # Optional: download as HTML
                                        html_str = plot_entry["figure"].to_html()
                                        st.download_button(
                                            "📥 Download HTML",
                                            data=html_str,
                                            file_name=f"{plot_entry.get('id','plot')}.html",
                                            mime="text/html",
                                            key=f"dl_plot_{id(plot_entry)}",
                                        )

                            # Persist assistant message (+ plots) to history
                            save_msg = {"role": "assistant", "content": response}
                            if st.session_state.data_manager.latest_plots:
                                save_msg["plots"] = st.session_state.data_manager.latest_plots.copy()
                            st.session_state.messages.append(save_msg)

                        else:
                            err = f"❌ Error: {result.get('error','Unknown error')}"
                            st.error(err)
                            st.session_state.messages.append({"role": "assistant", "content": err})

                    except Exception as e:
                        err = f"❌ System error: {e}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                        logger.exception("Agent error")

                    finally:
                        st.session_state.processing = False


# -----------------------
# Main
# -----------------------
def main():
    """Main application entry point."""
    init_session_state()
    display_sidebar()
    display_chat_interface()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;color:#666'>
        🦞 <b>Lobster AI</b> v2.0 | Multi-Agent Bioinformatics System | © 2025 Omics-OS
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
