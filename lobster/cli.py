#!/usr/bin/env python3
"""
Modern, user-friendly CLI for the Multi-Agent Bioinformatics System.
Installable via pip or curl, with rich terminal interface.
"""

from pathlib import Path
from typing import Optional
import os
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = '900000'
import subprocess
import shutil
 
import typer
import tabulate
from tabulate import tabulate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich import box
from rich import console

from lobster.core.client import AgentClient
# Implobsterort the proper callback handler
from lobster.utils import TerminalCallbackHandler
from lobster.config.agent_config import get_agent_configurator, initialize_configurator, LobsterAgentConfigurator
import json


def change_mode(new_mode: str, current_client: AgentClient) -> AgentClient:
    """
    Change the operation mode and reinitialize client with the new configuration.
    
    Args:
        new_mode: The new mode/profile to switch to
        current_client: The current AgentClient instance
        
    Returns:
        Updated AgentClient instance
    """
    global client
    
    # Store current settings before reinitializing
    current_workspace = Path(current_client.workspace_path)
    current_reasoning = current_client.enable_reasoning
    
    # Initialize a new configurator with the specified profile
    initialize_configurator(profile=new_mode)
    
    # Reinitialize the client with the new profile settings
    client = init_client(
        workspace=current_workspace,
        reasoning=current_reasoning
    )
    
    return client


# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="lobster",
    help="🦞 Lobster by homara AI - Multi-Agent Bioinformatics Analysis System",
    add_completion=True,
    rich_markup_mode="rich"
)

# Create a subcommand for configuration management
config_app = typer.Typer(
    name="config",
    help="Configuration management for Lobster agents",
)
app.add_typer(config_app, name="config")

# Global client instance
client: Optional[AgentClient] = None

# Global current directory tracking
current_directory = Path.cwd()

def init_client(
    workspace: Optional[Path] = None,
    reasoning: bool = True,
    debug: bool = False
) -> AgentClient:
    """Initialize the agent client."""
    global client
    
    # Set workspace
    if workspace is None:
        workspace = Path.cwd() / ".lobster_workspace"
    
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Initialize DataManagerV2 with workspace support and console for progress tracking
    from lobster.core.data_manager_v2 import DataManagerV2
    data_manager = DataManagerV2(workspace_path=workspace, console=console)
    
    # Create reasoning callback using the terminal_callback_handler
    callbacks = []
    if reasoning:
        reasoning_callback = TerminalCallbackHandler(
            console=console, 
            show_reasoning=True
        )
        callbacks.append(reasoning_callback)
    
    # Initialize client with proper data_manager connection
    client = AgentClient(
        data_manager=data_manager,  # Pass the configured data_manager
        workspace_path=workspace,
        enable_reasoning=reasoning,
        # enable_langfuse=debug,
        custom_callbacks=callbacks  # Pass the proper callback
    )
    
    return client


def execute_shell_command(command: str) -> bool:
    """Execute shell commands and return True if successful."""
    global current_directory
    
    parts = command.strip().split()
    if not parts:
        return False
    
    cmd = parts[0].lower()
    
    try:
        if cmd == "cd":
            # Handle cd command
            if len(parts) == 1:
                # cd with no arguments goes to home
                new_dir = Path.home()
            else:
                target = " ".join(parts[1:])  # Handle paths with spaces
                if target == "~":
                    new_dir = Path.home()
                elif target.startswith("~/"):
                    new_dir = Path.home() / target[2:]
                else:
                    new_dir = current_directory / target if not Path(target).is_absolute() else Path(target)
                
                new_dir = new_dir.resolve()
            
            if new_dir.exists() and new_dir.is_dir():
                current_directory = new_dir
                os.chdir(current_directory)
                console.print(f"[grey74]{current_directory}[/grey74]")
                return True
            else:
                console.print(f"[red]cd: no such file or directory: {target}[/red]")
                return True  # We handled it, even if it failed
        
        elif cmd == "pwd":
            # Print working directory
            console.print(f"[grey74]{current_directory}[/grey74]")
            return True
        
        elif cmd == "ls":
            # List directory contents with structured output
            target_dir = current_directory
            show_path = ""
            if len(parts) > 1:
                target_path = parts[1]
                show_path = target_path
                if target_path.startswith("~/"):
                    target_dir = Path.home() / target_path[2:]
                else:
                    target_dir = current_directory / target_path if not Path(target_path).is_absolute() else Path(target_path)
            
            if target_dir.exists() and target_dir.is_dir():
                items = list(target_dir.iterdir())
                if not items:
                    console.print(f"[grey50]Empty directory: {show_path or str(target_dir)}[/grey50]")
                    return True
                
                # Create a structured table for ls output
                table = Table(
                    title=f"📁 Directory Contents: {show_path or target_dir.name}",
                    box=box.SIMPLE,
                    border_style="blue",
                    show_header=True,
                    title_style="bold blue"
                )
                table.add_column("Name", style="white", min_width=20)
                table.add_column("Type", style="cyan", width=10)
                table.add_column("Size", style="grey74", width=10)
                table.add_column("Modified", style="grey50", width=16)
                
                # Sort: directories first, then files
                dirs = [item for item in items if item.is_dir()]
                files = [item for item in items if item.is_file()]
                sorted_items = sorted(dirs, key=lambda x: x.name.lower()) + sorted(files, key=lambda x: x.name.lower())
                
                for item in sorted_items:
                    try:
                        stat = item.stat()
                        if item.is_dir():
                            name = f"[bold blue]{item.name}/[/bold blue]"
                            type_str = "📁 DIR"
                            size_str = "-"
                        else:
                            name = f"[white]{item.name}[/white]"
                            type_str = "📄 FILE"
                            size = stat.st_size
                            if size < 1024:
                                size_str = f"{size}B"
                            elif size < 1024**2:
                                size_str = f"{size/1024:.1f}KB"
                            elif size < 1024**3:
                                size_str = f"{size/1024**2:.1f}MB"
                            else:
                                size_str = f"{size/1024**3:.1f}GB"
                        
                        # Format modification time
                        from datetime import datetime
                        mod_time = datetime.fromtimestamp(stat.st_mtime)
                        mod_str = mod_time.strftime("%Y-%m-%d %H:%M")
                        
                        table.add_row(name, type_str, size_str, mod_str)
                    except (OSError, PermissionError):
                        # If we can't get stats, just show the name
                        name = f"[bold blue]{item.name}/[/bold blue]" if item.is_dir() else f"[white]{item.name}[/white]"
                        table.add_row(name, "?", "?", "?")
                
                console.print(table)
                console.print(f"\n[grey50]Total: {len(dirs)} directories, {len(files)} files[/grey50]")
                return True
            else:
                console.print(f"[red]ls: cannot access '{parts[1] if len(parts) > 1 else target_dir}': No such file or directory[/red]")
                return True
        
        elif cmd == "cat":
            # Enhanced cat command with syntax highlighting
            if len(parts) < 2:
                console.print("[red]cat: missing file argument[/red]")
                return True
            
            file_path = " ".join(parts[1:])  # Handle paths with spaces
            if not file_path.startswith("/") and not file_path.startswith("~/"):
                file_path = current_directory / file_path
            else:
                file_path = Path(file_path).expanduser()
            
            try:
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                    
                    # Try to guess syntax from extension for highlighting
                    import mimetypes
                    ext = file_path.suffix.lower()
                    
                    # Map common extensions to syntax highlighting
                    language_map = {
                        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                        '.html': 'html', '.css': 'css', '.json': 'json',
                        '.xml': 'xml', '.yaml': 'yaml', '.yml': 'yaml',
                        '.sh': 'bash', '.bash': 'bash', '.zsh': 'bash',
                        '.sql': 'sql', '.md': 'markdown', '.txt': 'text',
                        '.log': 'text', '.conf': 'text', '.cfg': 'text'
                    }
                    
                    language = language_map.get(ext, 'text')
                    
                    if content.strip():
                        syntax = Syntax(content, language, theme="monokai", line_numbers=True)
                        console.print(Panel(
                            syntax,
                            title=f"[bold blue]📄 {file_path.name}[/bold blue]",
                            border_style="blue",
                            box=box.ROUNDED
                        ))
                    else:
                        console.print(f"[grey50]📄 {file_path.name} (empty file)[/grey50]")
                else:
                    console.print(f"[red]cat: {file_path}: No such file or directory[/red]")
            except PermissionError:
                console.print(f"[red]cat: {file_path}: Permission denied[/red]")
            except UnicodeDecodeError:
                console.print(f"[red]cat: {file_path}: Binary file (cannot display)[/red]")
            except Exception as e:
                console.print(f"[red]cat: {file_path}: {e}[/red]")
            
            return True
        
        elif cmd in ["cp", "mv", "mkdir", "touch", "rm"]:
            # Execute other shell commands with improved output formatting
            result = subprocess.run(command, shell=True, cwd=current_directory, 
                                  capture_output=True, text=True)
            
            # Format success messages
            if result.returncode == 0:
                if cmd == "mkdir" and len(parts) > 1:
                    console.print(f"[green]📁 Created directory: {parts[1]}[/green]")
                elif cmd == "touch" and len(parts) > 1:
                    console.print(f"[green]📄 Created file: {parts[1]}[/green]")
                elif cmd == "cp" and len(parts) > 2:
                    console.print(f"[green]📋 Copied: {parts[1]} → {parts[2]}[/green]")
                elif cmd == "mv" and len(parts) > 2:
                    console.print(f"[green]📦 Moved: {parts[1]} → {parts[2]}[/green]")
                elif cmd == "rm" and len(parts) > 1:
                    console.print(f"[green]🗑️  Removed: {parts[1]}[/green]")
                elif result.stdout:
                    console.print(result.stdout.rstrip())
            
            # Always show errors
            if result.stderr:
                console.print(f"[red]{result.stderr.rstrip()}[/red]")
            
            return True
        
        else:
            # Not a recognized shell command
            return False
    
    except Exception as e:
        console.print(f"[red]Error executing command: {e}[/red]")
        return True  # We handled it, even if it failed


def get_current_agent_name() -> str:
    """Get the current active agent name for display."""
    global client
    if client and hasattr(client, 'callbacks') and client.callbacks:
        for callback in client.callbacks:
            if isinstance(callback, TerminalCallbackHandler):
                if hasattr(callback, 'current_agent') and callback.current_agent:
                    # Format the agent name properly
                    agent_name = callback.current_agent.replace('_', ' ').title()
                    return f"🦞 {agent_name}"
                # Check if there are any recent events that might indicate the active agent
                elif hasattr(callback, 'events') and callback.events:
                    # Get the most recent agent from events
                    for event in reversed(callback.events):
                        if event.agent_name and event.agent_name != "system" and event.agent_name != "unknown":
                            agent_name = event.agent_name.replace('_', ' ').title()
                            return f"🦞 {agent_name}"
                break
    return "🦞 Lobster"


def display_welcome():
    """Display welcome message with ASCII art."""
    welcome_text = """
    [bold black on white]                                                                      [/bold black on white]
    [bold black on white]  🦞  [bold red on white]LOBSTER[/bold red on white]  by  [bold black on white]homara AI[/bold black on white]  🦞  [/bold black on white]
    [bold black on white]                                                                      [/bold black on white]
    
    [bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]
    [grey50]         Multi-Agent Bioinformatics Analysis System v2.0         [/grey50]
    [bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]
    
    [bold white]Key Tasks:[/bold white]
    • Analyze RNA-seq & genomics data
    • Generate visualizations and plots
    • Extract insights from bioinformatics datasets
    • Access GEO & literature databases
    
    [bold white]Essential Commands:[/bold white]
    [red]/help[/red]         - Show all available commands
    [red]/status[/red]       - Show system status  
    [red]/files[/red]        - List all workspace files
    [red]/data[/red]         - Show current dataset information
    [red]/metadata[/red]     - Show detailed metadata information
    [red]/workspace[/red]    - Show workspace status and configuration
    [red]/plots[/red]        - List all generated visualizations
    [red]/plot[/red]         - Open plots directory in file manager
    [red]/plot[/red] <ID>    - Open a specific plot by ID or name
    [red]/read[/red] <file>  - Read file from workspace (supports subdirectories)
    [red]/modes[/red]        - List available operation modes
    
    [bold white]Additional Features:[/bold white]
    • Configuration management via [red]lobster config[/red] subcommands
    • Single query mode via [red]lobster query[/red] command  
    • API server mode via [red]lobster serve[/red] command
    
    [dim grey50]Powered by LangGraph | © 2025 homara AI[/dim grey50]
    """
    console.print(welcome_text)


def display_status(client: AgentClient):
    """Display current system status."""
    status = client.get_status()
    
    # Get current mode/profile
    configurator = get_agent_configurator()
    current_mode = configurator.get_current_profile()
    
    # Create status table
    table = Table(
        title="🦞 System Status", 
        box=box.ROUNDED,
        border_style="red",
        title_style="bold red on white"
    )
    table.add_column("Property", style="bold grey93")
    table.add_column("Value", style="white")
    
    table.add_row("Session ID", status["session_id"])
    table.add_row("Mode", current_mode)
    table.add_row("Messages", str(status["message_count"]))
    table.add_row("Workspace", status["workspace"])
    table.add_row("Data Loaded", "✓" if status["has_data"] else "✗")
    
    if status["has_data"] and status["data_summary"]:
        summary = status["data_summary"]
        table.add_row("Data Shape", str(summary.get("shape", "N/A")))
        table.add_row("Memory Usage", summary.get("memory_usage", "N/A"))
    
    console.print(table)


@app.command()
def chat(
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    reasoning: bool = typer.Option(True, "--reasoning/--no-reasoning", help="Show agent reasoning"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode with Langfuse")
):
    """
    Start an interactive chat session with the multi-agent system.
    """
    display_welcome()
    
    # Initialize client
    console.print("\n[red]🦞 Initializing Lobster agents...[/red]")
    client = init_client(workspace, reasoning, debug)
    console.print("[bold red]✓[/bold red] [white]System ready![/white]\n")
    
    # Show initial status
    display_status(client)
    
    # Chat loop
    console.print("\n[bold white on red] 💬 Chat Interface [/bold white on red] [grey50]Type your questions or use /help for commands[/grey50]\n")
    
    while True:
        try:
            # Get user input with rich prompt - show current directory
            current_path = str(current_directory.name) if current_directory != Path.home() else "~"
            if current_directory == Path.cwd():
                current_path = str(current_directory.name)
            user_input = Prompt.ask(f"\n[bold red]🦞 {current_path}[/bold red]")
            
            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input, client)
                continue
            
            # Check if it's a shell command first
            if execute_shell_command(user_input):
                continue
            
            # Process query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Processing...", total=None)
                
                # Run query
                result = client.query(user_input, stream=False)
                
                progress.stop()
            
            # Display response
            if result["success"]:
                # Show which agent provided the response if available
                agent_name = result.get("last_agent", "supervisor")
                if agent_name and agent_name != '__end__':
                    agent_display = agent_name.replace('_', ' ').title()
                    title = f"[bold white on red] 🦞 {agent_display} Response [/bold white on red]"
                else:
                    title = "[bold white on red] 🦞 Lobster Response [/bold white on red]"
                
                response_panel = Panel(
                    Markdown(result["response"]),
                    title=title,
                    border_style="red",
                    padding=(1, 2),
                    box=box.DOUBLE
                )
                console.print(response_panel)
                
                # Show any generated plots
                if result.get("plots"):
                    console.print(f"\n[red]📊 Generated {len(result['plots'])} visualization(s)[/red]")
            else:
                console.print(f"[red]Error: {result['error']}[/red]")
        
        except KeyboardInterrupt:
            if Confirm.ask("\n[red]🦞 Exit Lobster?[/red]"):
                console.print("\n[bold white on red] 👋 Thank you for using Lobster by homara AI [/bold white on red]\n")
                break
            continue
        except Exception as e:
            console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]{e}[/red]")


def handle_command(command: str, client: AgentClient):
    """Handle slash commands."""
    cmd = command.lower().strip()
    
    if cmd == "/help":
        help_text = """
        [bold white]Available Commands:[/bold white]
        
        [red]/help[/red]         [grey50]-[/grey50] Show this help message
        [red]/status[/red]       [grey50]-[/grey50] Show system status
        [red]/files[/red]        [grey50]-[/grey50] List workspace files
        [red]/data[/red]         [grey50]-[/grey50] Show current data summary
        [red]/metadata[/red]     [grey50]-[/grey50] Show detailed metadata information
        [red]/workspace[/red]    [grey50]-[/grey50] Show workspace status and information
        [red]/modalities[/red]   [grey50]-[/grey50] Show detailed modality information
        [red]/plots[/red]        [grey50]-[/grey50] List all generated plots
        [red]/plot[/red]         [grey50]-[/grey50] Open plots directory in file manager
        [red]/plot[/red] <ID>    [grey50]-[/grey50] Open specific plot by ID or name
        [red]/save[/red]         [grey50]-[/grey50] Save current state to workspace
        [red]/read[/red] <file>  [grey50]-[/grey50] Read a file from workspace
        [red]/export[/red]       [grey50]-[/grey50] Export session data
        [red]/reset[/red]        [grey50]-[/grey50] Reset conversation
        [red]/mode[/red] <name>  [grey50]-[/grey50] Change operation mode
        [red]/modes[/red]        [grey50]-[/grey50] List available modes
        [red]/clear[/red]        [grey50]-[/grey50] Clear screen
        [red]/exit[/red]         [grey50]-[/grey50] Exit the chat
        
        [bold white]Shell Commands:[/bold white] [grey50](execute directly without /)[/grey50]
        
        [yellow]cd[/yellow] <path>      [grey50]-[/grey50] Change directory
        [yellow]pwd[/yellow]            [grey50]-[/grey50] Print current directory
        [yellow]ls[/yellow] [path]      [grey50]-[/grey50] List directory contents
        [yellow]mkdir[/yellow] <dir>    [grey50]-[/grey50] Create directory
        [yellow]touch[/yellow] <file>   [grey50]-[/grey50] Create file
        [yellow]cp[/yellow] <src> <dst> [grey50]-[/grey50] Copy file/directory
        [yellow]mv[/yellow] <src> <dst> [grey50]-[/grey50] Move/rename file/directory
        [yellow]rm[/yellow] <file>      [grey50]-[/grey50] Remove file
        [yellow]cat[/yellow] <file>     [grey50]-[/grey50] Display file contents
        """
        console.print(Panel(
            help_text, 
            title="[bold white on red] 🦞 Help Menu [/bold white on red]", 
            border_style="red",
            box=box.DOUBLE
        ))
    
    elif cmd == "/status":
        display_status(client)
    
    elif cmd == "/files":
        # Get categorized workspace files from data_manager
        workspace_files = client.data_manager.list_workspace_files()
        
        if any(workspace_files.values()):
            for category, files in workspace_files.items():
                if files:
                    # Sort files by modified date (descending: newest first)
                    files_sorted = sorted(files, key=lambda f: f["modified"], reverse=True)
                    
                    table = Table(
                        title=f"🦞 {category.title()} Files",
                        box=box.ROUNDED,
                        border_style="red",
                        title_style="bold red on white"
                    )
                    table.add_column("Name", style="bold white")
                    table.add_column("Size", style="grey74")
                    table.add_column("Modified", style="grey50")
                    table.add_column("Path", style="dim grey50")
                    
                    for f in files_sorted:
                        from datetime import datetime
                        size_kb = f["size"] / 1024
                        mod_time = datetime.fromtimestamp(f["modified"]).strftime("%Y-%m-%d %H:%M")
                        table.add_row(f["name"], f"{size_kb:.1f} KB", mod_time, Path(f["path"]).parent.name)
                    
                    console.print(table)
                    console.print()  # Add spacing between categories
        else:
            console.print("[grey50]No files in workspace[/grey50]")
    
    elif cmd.startswith("/read "):
        filename = cmd[6:].strip()
        try:
            content = client.read_file(filename)
            if content:
                # Try to guess syntax from extension, fallback to plain text
                import mimetypes
                ext = Path(filename).suffix
                mime, _ = mimetypes.guess_type(filename)
                language = "python" if ext == ".py" else (mime.split("/")[1] if mime and "/" in mime else "text")
                syntax = Syntax(content, language, theme="monokai", line_numbers=True)
                console.print(Panel(
                    syntax, 
                    title=f"[bold white on red] 📄 {filename} [/bold white on red]",
                    border_style="red",
                    box=box.DOUBLE
                ))
            else:
                console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Could not read file: {filename}[/red]")
                console.log(f"File not found or empty: {filename}")
        except FileNotFoundError:
            console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]File not found: {filename}[/red]")
            console.log(f"FileNotFoundError: {filename}")
        except Exception as e:
            console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Could not read file: {filename}[/red]")
            console.log(f"Exception while reading file {filename}: {e}")
    
    elif cmd == "/export":
        export_path = client.export_session()
        console.print(f"[bold red]✓[/bold red] [white]Session exported to:[/white] [grey74]{export_path}[/grey74]")
    
    elif cmd == "/reset":
        if Confirm.ask("[red]🦞 Reset conversation?[/red]"):
            client.reset()
            console.print("[bold red]✓[/bold red] [white]Conversation reset[/white]")
    
    elif cmd == "/data":
        # Show current data summary with enhanced metadata display
        if client.data_manager.has_data():
            summary = client.data_manager.get_data_summary()
            
            table = Table(
                title="🦞 Current Data Summary",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white"
            )
            table.add_column("Property", style="bold grey93")
            table.add_column("Value", style="white")
            
            table.add_row("Status", summary["status"])
            table.add_row("Shape", f"{summary['shape'][0]} × {summary['shape'][1]}")
            table.add_row("Memory Usage", summary["memory_usage"])
            
            if summary.get("columns"):
                cols_preview = ", ".join(summary["columns"][:5])
                if len(summary["columns"]) > 5:
                    cols_preview += f" ... (+{len(summary['columns'])-5} more)"
                table.add_row("Columns", cols_preview)
            
            if summary.get("sample_names"):
                samples_preview = ", ".join(summary["sample_names"][:3])
                if len(summary.get("sample_names", [])) > 3:
                    samples_preview += f" ... (+{len(summary['sample_names'])-3} more)"
                table.add_row("Samples", samples_preview)
            
            if summary.get("metadata_keys"):
                meta_preview = ", ".join(summary["metadata_keys"][:3])
                if len(summary["metadata_keys"]) > 3:
                    meta_preview += f" ... (+{len(summary['metadata_keys'])-3} more)"
                table.add_row("Metadata Keys", meta_preview)
            
            if summary.get("processing_log"):
                recent_steps = summary["processing_log"][-2:] if len(summary["processing_log"]) > 2 else summary["processing_log"]
                table.add_row("Recent Steps", "; ".join(recent_steps))
            
            console.print(table)
            
            # Show detailed metadata if available
            if hasattr(client.data_manager, 'current_metadata') and client.data_manager.current_metadata:
                metadata = client.data_manager.current_metadata
                console.print("\n[bold red]📋 Detailed Metadata:[/bold red]")
                
                metadata_table = Table(
                    box=box.SIMPLE,
                    border_style="red",
                    show_header=True
                )
                metadata_table.add_column("Key", style="bold grey93")
                metadata_table.add_column("Value", style="white")
                
                for key, value in list(metadata.items())[:10]:  # Show first 10 items
                    # Format value for display
                    if isinstance(value, (list, dict)):
                        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    else:
                        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    metadata_table.add_row(key, display_value)
                
                if len(metadata) > 10:
                    metadata_table.add_row("...", f"(+{len(metadata)-10} more items)")
                
                console.print(metadata_table)
        else:
            console.print("[grey50]No data currently loaded[/grey50]")
    
    elif cmd == "/metadata":
        # Show metadata store contents (for DataManagerV2) and current metadata
        console.print("[bold red]📋 Metadata Information[/bold red]\n")
        
        # Check if using DataManagerV2 with metadata_store
        if hasattr(client.data_manager, 'metadata_store'):
            metadata_store = client.data_manager.metadata_store
            if metadata_store:
                console.print("[bold white]🗄️  Metadata Store (Cached GEO/External Data):[/bold white]")
                
                store_table = Table(
                    box=box.ROUNDED,
                    border_style="red",
                    title="🗄️ Metadata Store",
                    title_style="bold red on white"
                )
                store_table.add_column("Dataset ID", style="bold white")
                store_table.add_column("Type", style="cyan")
                store_table.add_column("Title", style="white")
                store_table.add_column("Samples", style="grey74")
                store_table.add_column("Cached", style="grey50")
                
                for dataset_id, metadata_info in metadata_store.items():
                    metadata = metadata_info.get('metadata', {})
                    validation = metadata_info.get('validation', {})
                    
                    # Extract key information
                    title = str(metadata.get('title', 'N/A'))[:40] + "..." if len(str(metadata.get('title', 'N/A'))) > 40 else str(metadata.get('title', 'N/A'))
                    data_type = validation.get('predicted_data_type', 'unknown').replace('_', ' ').title()
                    samples = len(metadata.get('samples', {})) if metadata.get('samples') else 'N/A'
                    
                    # Parse timestamp
                    timestamp = metadata_info.get('fetch_timestamp', '')
                    try:
                        from datetime import datetime
                        cached_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        cached_str = cached_time.strftime("%Y-%m-%d %H:%M")
                    except:
                        cached_str = timestamp[:16] if timestamp else "N/A"
                    
                    store_table.add_row(
                        dataset_id,
                        data_type,
                        title,
                        str(samples),
                        cached_str
                    )
                
                console.print(store_table)
                console.print()
            else:
                console.print("[grey50]No cached metadata in metadata store[/grey50]\n")
        
        # Show current data metadata
        if hasattr(client.data_manager, 'current_metadata') and client.data_manager.current_metadata:
            console.print("[bold white]📊 Current Data Metadata:[/bold white]")
            metadata = client.data_manager.current_metadata
            
            metadata_table = Table(
                box=box.SIMPLE,
                border_style="red",
                show_header=True
            )
            metadata_table.add_column("Key", style="bold grey93", width=25)
            metadata_table.add_column("Value", style="white", width=50)
            
            for key, value in metadata.items():
                # Format value for display
                if isinstance(value, dict):
                    display_value = f"Dict with {len(value)} keys: {', '.join(list(value.keys())[:3])}"
                    if len(value) > 3:
                        display_value += f" ... (+{len(value)-3} more)"
                elif isinstance(value, list):
                    display_value = f"List with {len(value)} items"
                    if len(value) > 0:
                        display_value += f": {', '.join(str(v) for v in value[:3])}"
                        if len(value) > 3:
                            display_value += f" ... (+{len(value)-3} more)"
                else:
                    display_value = str(value)
                    if len(display_value) > 60:
                        display_value = display_value[:60] + "..."
                
                metadata_table.add_row(key, display_value)
            
            console.print(metadata_table)
        else:
            console.print("[grey50]No current data metadata available[/grey50]")
    
    elif cmd == "/workspace":
        # Show workspace status and information
        console.print("[bold red]🏗️  Workspace Information[/bold red]\n")
        
        # Check if using DataManagerV2
        if hasattr(client.data_manager, 'get_workspace_status'):
            workspace_status = client.data_manager.get_workspace_status()
            
            # Main workspace info
            workspace_table = Table(
                title="🏗️ Workspace Status",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white"
            )
            workspace_table.add_column("Property", style="bold grey93")
            workspace_table.add_column("Value", style="white")
            
            workspace_table.add_row("Workspace Path", workspace_status.get("workspace_path", "N/A"))
            workspace_table.add_row("Modalities Loaded", str(workspace_status.get("modalities_loaded", 0)))
            workspace_table.add_row("Registered Backends", str(len(workspace_status.get("registered_backends", []))))
            workspace_table.add_row("Registered Adapters", str(len(workspace_status.get("registered_adapters", []))))
            workspace_table.add_row("Default Backend", workspace_status.get("default_backend", "N/A"))
            workspace_table.add_row("Provenance Enabled", "✓" if workspace_status.get("provenance_enabled") else "✗")
            workspace_table.add_row("MuData Available", "✓" if workspace_status.get("mudata_available") else "✗")
            
            console.print(workspace_table)
            
            # Show directories
            if workspace_status.get("directories"):
                dirs = workspace_status["directories"]
                console.print(f"\n[bold white]📁 Directories:[/bold white]")
                for dir_type, path in dirs.items():
                    console.print(f"  • {dir_type.title()}: [grey74]{path}[/grey74]")
            
            # Show loaded modalities
            if workspace_status.get("modality_names"):
                console.print(f"\n[bold white]🧬 Loaded Modalities:[/bold white]")
                for modality in workspace_status["modality_names"]:
                    console.print(f"  • {modality}")
            
            # Show available backends and adapters
            console.print(f"\n[bold white]🔧 Available Backends:[/bold white]")
            for backend in workspace_status.get("registered_backends", []):
                console.print(f"  • {backend}")
            
            console.print(f"\n[bold white]🔌 Available Adapters:[/bold white]")
            for adapter in workspace_status.get("registered_adapters", []):
                console.print(f"  • {adapter}")
        else:
            # Fallback for older DataManager
            console.print("[bold white]📁 Basic Workspace Info:[/bold white]")
            if hasattr(client.data_manager, 'workspace_path'):
                console.print(f"  • Path: [grey74]{client.data_manager.workspace_path}[/grey74]")
            if hasattr(client.data_manager, 'has_data'):
                console.print(f"  • Data Loaded: {'✓' if client.data_manager.has_data() else '✗'}")
    
    elif cmd == "/modalities":
        # Show detailed modality information (DataManagerV2 specific)
        if hasattr(client.data_manager, 'list_modalities'):
            modalities = client.data_manager.list_modalities()
            
            if modalities:
                console.print("[bold red]🧬 Modality Details[/bold red]\n")
                
                for modality_name in modalities:
                    try:
                        adata = client.data_manager.get_modality(modality_name)
                        
                        # Create modality table
                        mod_table = Table(
                            title=f"🧬 {modality_name}",
                            box=box.ROUNDED,
                            border_style="cyan",
                            title_style="bold cyan on white"
                        )
                        mod_table.add_column("Property", style="bold grey93")
                        mod_table.add_column("Value", style="white")
                        
                        mod_table.add_row("Shape", f"{adata.n_obs} obs × {adata.n_vars} vars")
                        
                        # Show obs columns
                        obs_cols = list(adata.obs.columns)
                        if obs_cols:
                            cols_preview = ", ".join(obs_cols[:5])
                            if len(obs_cols) > 5:
                                cols_preview += f" ... (+{len(obs_cols)-5} more)"
                            mod_table.add_row("Obs Columns", cols_preview)
                        
                        # Show var columns
                        var_cols = list(adata.var.columns)
                        if var_cols:
                            var_preview = ", ".join(var_cols[:5])
                            if len(var_cols) > 5:
                                var_preview += f" ... (+{len(var_cols)-5} more)"
                            mod_table.add_row("Var Columns", var_preview)
                        
                        # Show layers
                        if adata.layers:
                            layers_str = ", ".join(list(adata.layers.keys()))
                            mod_table.add_row("Layers", layers_str)
                        
                        # Show obsm
                        if adata.obsm:
                            obsm_str = ", ".join(list(adata.obsm.keys()))
                            mod_table.add_row("Obsm", obsm_str)
                        
                        # Show some uns info
                        if adata.uns:
                            uns_keys = list(adata.uns.keys())[:5]
                            uns_str = ", ".join(uns_keys)
                            if len(adata.uns) > 5:
                                uns_str += f" ... (+{len(adata.uns)-5} more)"
                            mod_table.add_row("Uns Keys", uns_str)
                        
                        console.print(mod_table)
                        console.print()
                        
                    except Exception as e:
                        console.print(f"[red]Error accessing modality {modality_name}: {e}[/red]")
            else:
                console.print("[grey50]No modalities loaded[/grey50]")
        else:
            console.print("[grey50]Modality information not available (using legacy DataManager)[/grey50]")
    
    elif cmd == "/plots":
        # Show generated plots
        plots = client.data_manager.get_plot_history()
        
        if plots:
            table = Table(
                title="🦞 Generated Plots",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white"
            )
            table.add_column("ID", style="bold white")
            table.add_column("Title", style="white")
            table.add_column("Source", style="grey74")
            table.add_column("Created", style="grey50")
            
            for plot in plots:
                from datetime import datetime
                try:
                    created = datetime.fromisoformat(plot["timestamp"].replace('Z', '+00:00'))
                    created_str = created.strftime("%Y-%m-%d %H:%M")
                except:
                    created_str = plot["timestamp"][:16] if plot["timestamp"] else "N/A"
                
                table.add_row(
                    plot["id"],
                    plot["title"],
                    plot["source"] or "N/A",
                    created_str
                )
            
            console.print(table)
        else:
            console.print("[grey50]No plots generated yet[/grey50]")
    
    elif cmd.startswith("/plot"):
        # Handle /plot command with optional plot ID/name
        parts = cmd.split(maxsplit=1)
        
        if len(parts) == 1:
            # /plot with no arguments - open plots directory
            plots_dir = client.data_manager.workspace_path / "plots"
            
            # Ensure plots directory exists
            if not plots_dir.exists():
                plots_dir.mkdir(parents=True, exist_ok=True)
                # Save any existing plots to the directory
                if client.data_manager.latest_plots:
                    saved_files = client.data_manager.save_plots_to_workspace()
                    if saved_files:
                        console.print(f"[bold red]✓[/bold red] [white]Saved {len(saved_files)} plot files to workspace[/white]")
            
            # Open the directory in file manager
            import platform
            import subprocess
            
            try:
                system = platform.system()
                if system == "Darwin":  # macOS
                    subprocess.run(["open", str(plots_dir)], check=True)
                    console.print(f"[bold red]✓[/bold red] [white]Opened plots directory in Finder:[/white] [grey74]{plots_dir}[/grey74]")
                elif system == "Linux":
                    # Try common file managers
                    file_managers = ["xdg-open", "nautilus", "dolphin", "thunar", "pcmanfm"]
                    opened = False
                    for fm in file_managers:
                        try:
                            subprocess.run([fm, str(plots_dir)], check=True, stderr=subprocess.DEVNULL)
                            console.print(f"[bold red]✓[/bold red] [white]Opened plots directory:[/white] [grey74]{plots_dir}[/grey74]")
                            opened = True
                            break
                        except (subprocess.CalledProcessError, FileNotFoundError):
                            continue
                    if not opened:
                        console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Could not open file manager. Directory: {plots_dir}[/red]")
                elif system == "Windows":
                    subprocess.run(["explorer", str(plots_dir)], check=True)
                    console.print(f"[bold red]✓[/bold red] [white]Opened plots directory in Explorer:[/white] [grey74]{plots_dir}[/grey74]")
                else:
                    console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Unsupported operating system: {system}[/red]")
                    console.print(f"[white]Plots directory:[/white] [grey74]{plots_dir}[/grey74]")
            except Exception as e:
                console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Failed to open plots directory: {e}[/red]")
                console.print(f"[white]Plots directory:[/white] [grey74]{plots_dir}[/grey74]")
        
        else:
            # /plot <ID or name> - open specific plot
            plot_identifier = parts[1].strip()
            
            # First, ensure plots are saved to workspace
            plots_dir = client.data_manager.workspace_path / "plots"
            if not plots_dir.exists():
                plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plots if needed
            if client.data_manager.latest_plots:
                saved_files = client.data_manager.save_plots_to_workspace()
            
            # Find the plot by ID or partial title match
            found_plot = None
            plot_info = None
            
            for plot_entry in client.data_manager.latest_plots:
                # Check ID match
                if plot_entry["id"] == plot_identifier:
                    found_plot = plot_entry
                    plot_info = plot_entry
                    break
                # Check title match (case-insensitive partial match)
                elif plot_identifier.lower() in plot_entry["title"].lower() or plot_identifier.lower() in plot_entry["original_title"].lower():
                    found_plot = plot_entry
                    plot_info = plot_entry
                    break
            
            if found_plot:
                # Construct the filename
                plot_id = plot_info["id"]
                plot_title = plot_info["title"]
                
                # Create sanitized filename (same logic as save_plots_to_workspace)
                safe_title = "".join(
                    c for c in plot_title if c.isalnum() or c in [" ", "_", "-"]
                ).rstrip()
                safe_title = safe_title.replace(" ", "_")
                filename_base = f"{plot_id}_{safe_title}" if safe_title else plot_id
                
                # Try to open HTML file first, then PNG
                html_path = plots_dir / f"{filename_base}.html"
                png_path = plots_dir / f"{filename_base}.png"
                
                file_to_open = html_path if html_path.exists() else png_path
                
                if file_to_open.exists():
                    import platform
                    import subprocess
                    
                    try:
                        system = platform.system()
                        if system == "Darwin":  # macOS
                            subprocess.run(["open", str(file_to_open)], check=True)
                            console.print(f"[bold red]✓[/bold red] [white]Opened plot:[/white] [grey74]{plot_info['original_title']}[/grey74]")
                        elif system == "Linux":
                            subprocess.run(["xdg-open", str(file_to_open)], check=True)
                            console.print(f"[bold red]✓[/bold red] [white]Opened plot:[/white] [grey74]{plot_info['original_title']}[/grey74]")
                        elif system == "Windows":
                            subprocess.run(["start", "", str(file_to_open)], shell=True, check=True)
                            console.print(f"[bold red]✓[/bold red] [white]Opened plot:[/white] [grey74]{plot_info['original_title']}[/grey74]")
                        else:
                            console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Unsupported operating system: {system}[/red]")
                    except Exception as e:
                        console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Failed to open plot: {e}[/red]")
                        console.print(f"[white]Plot file:[/white] [grey74]{file_to_open}[/grey74]")
                else:
                    console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Plot file not found. Try running /save first.[/red]")
            else:
                console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Plot not found: {plot_identifier}[/red]")
                console.print("[grey50]Use /plots to see available plot IDs and titles[/grey50]")
    
    elif cmd == "/save":
        # Auto-save current state
        saved_items = client.data_manager.auto_save_state()
        
        if saved_items:
            console.print(f"[bold red]✓[/bold red] [white]Saved to workspace:[/white]")
            for item in saved_items:
                console.print(f"  • {item}")
        else:
            console.print("[grey50]Nothing to save (no data or plots loaded)[/grey50]")
    
    elif cmd == "/modes":
        # List all available modes/profiles
        configurator = get_agent_configurator()
        current_mode = configurator.get_current_profile()
        available_profiles = configurator.list_available_profiles()
        
        # Create modes table
        table = Table(
            title="🦞 Available Modes", 
            box=box.ROUNDED,
            border_style="red",
            title_style="bold red on white"
        )
        table.add_column("Mode", style="bold white")
        table.add_column("Status", style="grey74")
        table.add_column("Description", style="grey50")
        
        for profile in sorted(available_profiles.keys()):
            # Add descriptions for each mode
            description = ""
            if profile == "development":
                description = "Fast, lightweight models for development"
            elif profile == "production":
                description = "Balanced performance and cost"
            elif profile == "high-performance":
                description = "Enhanced performance for complex tasks"
            elif profile == "ultra-performance":
                description = "Maximum capability for demanding analyses"
            elif profile == "cost-optimized":
                description = "Efficient models to minimize costs"
            elif profile == "heavyweight":
                description = "Most capable models for all agents"
            elif profile == "eu-compliant":
                description = "EU region models for compliance"
            elif profile == "eu-high-performance":
                description = "High-performance EU region models"
            
            status = "[bold green]ACTIVE[/bold green]" if profile == current_mode else ""
            table.add_row(profile, status, description)
            
        console.print(table)
    
    elif cmd.startswith("/mode "):
        # Get the new mode name from the command
        new_mode = cmd[6:].strip()
        
        # Get available profiles
        configurator = get_agent_configurator()
        available_profiles = configurator.list_available_profiles()
        
        if new_mode in available_profiles:
            # Change the mode and update the client
            change_mode(new_mode, client)
            console.print(f"[bold red]✓[/bold red] [white]Mode changed to:[/white] [bold red]{new_mode}[/bold red]")
            display_status(client)
        else:
            # Display available profilescan you
            console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Invalid mode: {new_mode}[/red]")
            console.print("[white]Available modes:[/white]")
            for profile in sorted(available_profiles.keys()):
                if profile == configurator.get_current_profile():
                    console.print(f"  • [bold red]{profile}[/bold red] (current)")
                else:
                    console.print(f"  • {profile}")
    
    elif cmd == "/clear":
        console.clear()
    
    elif cmd == "/exit":
        if Confirm.ask("[red]🦞 Exit Lobster?[/red]"):
            console.print("\n[bold white on red] 👋 Thank you for using Lobster by homara AI [/bold white on red]\n")
            raise KeyboardInterrupt
    
    else:
        console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Unknown command: {command}[/red]")


@app.command()
def query(
    question: str,
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w"),
    reasoning: bool = typer.Option(False, "--reasoning", "-r"),
    output: Optional[Path] = typer.Option(None, "--output", "-o")
):
    """
    Send a single query to the agent system.
    """
    # Initialize client
    client = init_client(workspace, reasoning)
    
    # Process query
    with console.status("[red]🦞 Processing query...[/red]"):
        result = client.query(question)
    
    # Display or save result
    if result["success"]:
        if output:
            output.write_text(result["response"])
            console.print(f"[bold red]✓[/bold red] [white]Response saved to:[/white] [grey74]{output}[/grey74]")
        else:
            console.print(Panel(
                Markdown(result["response"]),
                title="[bold white on red] 🦞 Lobster Response [/bold white on red]",
                border_style="red",
                box=box.DOUBLE
            ))
    else:
        console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]{result['error']}[/red]")

@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p"),
    host: str = typer.Option("0.0.0.0", "--host")
):
    """
    Start the agent system as an API server (for React UI).
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    
    # Create FastAPI app
    api = FastAPI(
        title="Lobster Agent API",
        description="🦞 Multi-Agent Bioinformatics System by homara AI",
        version="2.0"
    )
    
    class QueryRequest(BaseModel):
        question: str
        session_id: Optional[str] = None
        stream: bool = False
    
    @api.post("/query")
    async def query_endpoint(request: QueryRequest):
        try:
            client = init_client()
            result = client.query(request.question, stream=request.stream)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @api.get("/status")
    async def status_endpoint():
        client = init_client()
        return client.get_status()
    
    console.print(f"[red]🦞 Starting Lobster API server on {host}:{port}[/red]")
    uvicorn.run(api, host=host, port=port)


# Config subcommands
@config_app.command(name="list-models")
def list_models():
    """List all available model presets."""
    configurator = LobsterAgentConfigurator()
    models = configurator.list_available_models()
    
    console.print("\n[cyan]🤖 Available Model Presets[/cyan]")
    console.print("[cyan]" + "=" * 60 + "[/cyan]")
    
    table = Table(
        box=box.ROUNDED,
        border_style="cyan",
        title="🤖 Available Model Presets",
        title_style="bold cyan"
    )
    
    table.add_column("Preset Name", style="bold white")
    table.add_column("Tier", style="cyan")
    table.add_column("Region", style="white")
    table.add_column("Temperature", style="white")
    table.add_column("Description", style="white")
    
    for name, config in models.items():
        description = config.description[:40] + "..." if len(config.description) > 40 else config.description
        table.add_row(
            name,
            config.tier.value.title(),
            config.region,
            f"{config.temperature}",
            description
        )
    
    console.print(table)

@config_app.command(name="list-profiles")
def list_profiles():
    """List all available testing profiles."""
    configurator = LobsterAgentConfigurator()
    profiles = configurator.list_available_profiles()
    
    console.print("\n[cyan]⚙️  Available Testing Profiles[/cyan]")
    console.print("[cyan]" + "=" * 60 + "[/cyan]")
    
    for profile_name, config in profiles.items():
        console.print(f"\n[yellow]📋 {profile_name.title()}[/yellow]")
        for agent, model in config.items():
            console.print(f"   {agent}: {model}")

@config_app.command(name="show-config")
def show_config(profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Profile to show")):
    """Show current configuration."""
    configurator = initialize_configurator(profile=profile) if profile else LobsterAgentConfigurator()
    configurator.print_current_config()

@config_app.command(name="test")
def test(
    profile: str = typer.Option(..., "--profile", "-p", help="Profile to test"),
    agent: Optional[str] = typer.Option(None, "--agent", "-a", help="Specific agent to test")
):
    """Test a specific configuration."""
    try:
        configurator = initialize_configurator(profile=profile)
        
        if agent:
            # Test specific agent
            try:
                config = configurator.get_agent_config(agent)
                params = configurator.get_llm_params(agent)
                
                console.print(f"\n[green]✅ Agent '{agent}' configuration is valid[/green]")
                console.print(f"   Model: {config.model_config.model_id}")
                console.print(f"   Tier: {config.model_config.tier.value}")
                console.print(f"   Region: {config.model_config.region}")
                
            except KeyError:
                console.print(f"\n[red]❌ Agent '{agent}' not found in profile '{profile}'[/red]")
                return False
        else:
            # Test all agents dynamically
            console.print(f"\n[yellow]🧪 Testing Profile: {profile}[/yellow]")
            all_valid = True
            
            # Get all agents from the configurator's DEFAULT_AGENTS
            available_agents = configurator.DEFAULT_AGENTS
            
            for agent_name in available_agents:
                try:
                    config = configurator.get_agent_config(agent_name)
                    params = configurator.get_llm_params(agent_name)
                    console.print(f"   [green]✅ {agent_name}: {config.model_config.model_id}[/green]")
                except Exception as e:
                    console.print(f"   [red]❌ {agent_name}: {str(e)}[/red]")
                    all_valid = False
            
            if all_valid:
                console.print(f"\n[green]🎉 Profile '{profile}' is fully configured and valid![/green]")
            else:
                console.print(f"\n[yellow]⚠️  Profile '{profile}' has configuration issues[/yellow]")
        
        return True
        
    except Exception as e:
        console.print(f"\n[red]❌ Error testing configuration: {str(e)}[/red]")
        return False

@config_app.command(name="create-custom")
def create_custom():
    """Interactive creation of custom configuration."""
    console.print("\n[cyan]🛠️  Create Custom Configuration[/cyan]")
    console.print("[cyan]" + "=" * 50 + "[/cyan]")
    
    configurator = LobsterAgentConfigurator()
    available_models = configurator.list_available_models()
    
    # Show available models
    console.print("\n[yellow]Available models:[/yellow]")
    for i, (name, config) in enumerate(available_models.items(), 1):
        console.print(f"{i:2}. {name} ({config.tier.value}, {config.region})")
    
    config_data = {
        "profile": "custom",
        "agents": {}
    }
    
    # Use dynamic agent list
    agents = configurator.DEFAULT_AGENTS
    
    for agent in agents:
        console.print(f"\n[yellow]Configuring {agent}:[/yellow]")
        console.print("Choose a model preset (enter number or name):")
        
        choice = Prompt.ask(f"Model for {agent}")
        
        # Handle numeric choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                model_name = list(available_models.keys())[idx]
            else:
                console.print("[yellow]Invalid choice, using default (claude-sonnet)[/yellow]")
                model_name = "claude-sonnet"
        else:
            # Handle name choice
            if choice in available_models:
                model_name = choice
            else:
                console.print("[yellow]Invalid choice, using default (claude-sonnet)[/yellow]")
                model_name = "claude-sonnet"
        
        model_config = available_models[model_name]
        config_data["agents"][agent] = {
            "model_config": {
                "provider": model_config.provider.value,
                "model_id": model_config.model_id,
                "tier": model_config.tier.value,
                "temperature": model_config.temperature,
                "region": model_config.region,
                "description": model_config.description
            },
            "enabled": True,
            "custom_params": {}
        }
        
        console.print(f"   [green]Selected: {model_name}[/green]")
    
    # Save configuration
    config_file = "config/custom_agent_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    console.print(f"\n[green]✅ Custom configuration saved to: {config_file}[/green]")
    console.print("[yellow]To use this configuration, set:[/yellow]")
    console.print(f"   export GENIE_CONFIG_FILE={config_file}", style="yellow")

@config_app.command(name="generate-env")
def generate_env():
    """Generate .env template with all available options."""
    template = """# Genie AI Configuration Template
# Copy this file to .env and configure as needed

# =============================================================================
# API KEYS (Required)
# =============================================================================
OPENAI_API_KEY="your-openai-api-key-here"
AWS_BEDROCK_ACCESS_KEY="your-aws-access-key-here"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key-here"
NCBI_API_KEY="your-ncbi-api-key-here"

# =============================================================================
# AGENT CONFIGURATION (Professional System)
# =============================================================================

# Profile-based configuration (recommended)
# Available profiles: development, production, high-performance, cost-optimized, eu-compliant
GENIE_PROFILE=production

# OR use custom configuration file
# GENIE_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides (optional)
# Available models: claude-haiku, claude-sonnet, claude-sonnet-eu, claude-opus, claude-opus-eu, claude-3-7-sonnet, claude-3-7-sonnet-eu
# GENIE_SUPERVISOR_MODEL=claude-haiku
# GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
# GENIE_METHOD_AGENT_MODEL=claude-sonnet
# GENIE_GENERAL_CONVERSATION_MODEL=claude-haiku

# Global model override (overrides all agents)
# GENIE_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
# GENIE_SUPERVISOR_TEMPERATURE=0.5
# GENIE_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
# GENIE_METHOD_AGENT_TEMPERATURE=0.3

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server configuration
PORT=8501
HOST=0.0.0.0
DEBUG=False

# Data processing
GENIE_MAX_FILE_SIZE_MB=500
GENIE_CLUSTER_RESOLUTION=0.5
GENIE_CACHE_DIR=data/cache

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Lightweight development setup
# GENIE_PROFILE=development
# GENIE_SUPERVISOR_MODEL=claude-haiku
# GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-sonnet

# Example 2: High-performance research setup
# GENIE_PROFILE=high-performance
# GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-3-7-sonnet

# Example 3: EU compliance
# GENIE_PROFILE=eu-compliant
# AWS_REGION=eu-central-1

# Example 4: Cost-optimized setup
# GENIE_PROFILE=cost-optimized
# GENIE_GLOBAL_MODEL=claude-haiku
"""
    
    with open('.env.template', 'w') as f:
        f.write(template)
    
    console.print("[green]✅ Environment template saved to: .env.template[/green]")
    console.print("[yellow]Copy this file to .env and configure your API keys[/yellow]")

if __name__ == "__main__":
    app()
