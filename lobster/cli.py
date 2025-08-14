#!/usr/bin/env python3
"""
Modern, user-friendly CLI for the Multi-Agent Bioinformatics System.
Installable via pip or curl, with rich terminal interface.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm
from rich import box
from rich import console

from .core import AgentClient
# Import the proper callback handler
from .utils import TerminalCallbackHandler
from .config.agent_config import get_agent_configurator, initialize_configurator


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

# Global client instance
client: Optional[AgentClient] = None

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
    
    # Initialize DataManager with workspace support and console for progress tracking
    from .core.data_manager import DataManager
    data_manager = DataManager(workspace_path=workspace, console=console)
    
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
    [red]/read[/red] <file>  - Read file from workspace or absolute path (supports subdirectories)
    [red]/files[/red]        - List all workspace files
    [red]/data[/red]         - Show current dataset information  
    [red]/plots[/red]        - List all generated visualizations
    [red]/help[/red]         - Show all available commands
    
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
            # Get user input with rich prompt - always show Lobster
            user_input = Prompt.ask(f"\n[bold red]🦞 Lobster You[/bold red]")
            
            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input, client)
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
                # Response header always shows Lobster
                response_panel = Panel(
                    Markdown(result["response"]),
                    title="[bold white on red] 🦞 Lobster Response [/bold white on red]",
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
        
        [red]/help[/red]       [grey50]-[/grey50] Show this help message
        [red]/status[/red]     [grey50]-[/grey50] Show system status
        [red]/files[/red]      [grey50]-[/grey50] List workspace files
        [red]/data[/red]       [grey50]-[/grey50] Show current data summary
        [red]/plots[/red]      [grey50]-[/grey50] List generated plots
        [red]/save[/red]       [grey50]-[/grey50] Save current state to workspace
        [red]/read[/red] <file> [grey50]-[/grey50] Read a file from workspace
        [red]/export[/red]     [grey50]-[/grey50] Export session data
        [red]/reset[/red]      [grey50]-[/grey50] Reset conversation
        [red]/mode[/red] <name> [grey50]-[/grey50] Change operation mode
        [red]/modes[/red]      [grey50]-[/grey50] List available modes
        [red]/clear[/red]      [grey50]-[/grey50] Clear screen
        [red]/exit[/red]       [grey50]-[/grey50] Exit the chat
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
                    
                    for f in files:
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
        # Show current data summary
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
            
            if summary.get("metadata_keys"):
                meta_preview = ", ".join(summary["metadata_keys"][:3])
                if len(summary["metadata_keys"]) > 3:
                    meta_preview += f" ... (+{len(summary['metadata_keys'])-3} more)"
                table.add_row("Metadata", meta_preview)
            
            console.print(table)
        else:
            console.print("[grey50]No data currently loaded[/grey50]")
    
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
            # Display available profiles
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


if __name__ == "__main__":
    app()
