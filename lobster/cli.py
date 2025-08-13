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
    
    # Create reasoning callback using the terminal_callback_handler
    callbacks = []
    if reasoning:
        reasoning_callback = TerminalCallbackHandler(
            console=console, 
            show_reasoning=True
        )
        callbacks.append(reasoning_callback)
    
    # Initialize client
    client = AgentClient(
        workspace_path=workspace,
        enable_reasoning=reasoning,
        # enable_langfuse=debug,
        custom_callbacks=callbacks  # Pass the proper callback
    )
    
    return client


def display_welcome():
    """Display welcome message with ASCII art."""
    welcome_text = """
    [bold black on white]                                                                      [/bold black on white]
    [bold black on white]  🦞  [bold red on white]LOBSTER[/bold red on white]  by  [bold black on white]homara AI[/bold black on white]  🦞  [/bold black on white]
    [bold black on white]                                                                      [/bold black on white]
    
    [bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]
    [grey50]         Multi-Agent Bioinformatics Analysis System v2.0         [/grey50]
    [bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]
    
    [dim grey50]Powered by LangGraph | © 2025 homara AI[/dim grey50]
    """
    console.print(welcome_text)


def display_status(client: AgentClient):
    """Display current system status."""
    status = client.get_status()
    
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
            # Get user input with rich prompt
            user_input = Prompt.ask("\n[bold red]🦞 You[/bold red]")
            
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
        [red]/read[/red] <file> [grey50]-[/grey50] Read a file from workspace
        [red]/export[/red]     [grey50]-[/grey50] Export session data
        [red]/reset[/red]      [grey50]-[/grey50] Reset conversation
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
        files = client.list_workspace_files()
        if files:
            table = Table(
                title="🦞 Workspace Files",
                box=box.ROUNDED,
                border_style="red",
                title_style="bold red on white"
            )
            table.add_column("Name", style="bold white")
            table.add_column("Size", style="grey74")
            table.add_column("Modified", style="grey50")
            
            for f in files:
                size_kb = f["size"] / 1024
                table.add_row(f["name"], f"{size_kb:.1f} KB", f["modified"])
            
            console.print(table)
        else:
            console.print("[grey50]No files in workspace[/grey50]")
    
    elif cmd.startswith("/read "):
        filename = cmd[6:].strip()
        content = client.read_file(filename)
        if content:
            syntax = Syntax(content, "python", theme="monokai", line_numbers=True)
            console.print(Panel(
                syntax, 
                title=f"[bold white on red] 📄 {filename} [/bold white on red]",
                border_style="red",
                box=box.DOUBLE
            ))
        else:
            console.print(f"[bold red on white] ⚠️  Error [/bold red on white] [red]Could not read file: {filename}[/red]")
    
    elif cmd == "/export":
        export_path = client.export_session()
        console.print(f"[bold red]✓[/bold red] [white]Session exported to:[/white] [grey74]{export_path}[/grey74]")
    
    elif cmd == "/reset":
        if Confirm.ask("[red]🦞 Reset conversation?[/red]"):
            client.reset()
            console.print("[bold red]✓[/bold red] [white]Conversation reset[/white]")
    
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
def install():
    """
    Install the Lobster CLI system-wide.
    """
    console.print("[red]🦞 Installing Lobster CLI by homara AI...[/red]")
    
    # Create installation script
    install_script = """
#!/bin/bash
# Lobster CLI Installation Script

# Check Python version
python3 --version >/dev/null 2>&1 || { echo "Python 3 is required"; exit 1; }

# Install package
pip install --user lobster-cli

# Add to PATH if needed
if ! command -v lobster &> /dev/null; then
    echo 'export PATH="\$HOME/.local/bin:\$PATH"' >> ~/.bashrc
    echo "Please run: source ~/.bashrc"
fi

echo "🦞 Lobster CLI by homara AI installed successfully!"
echo "Run 'lobster chat' to start"
    """
    
    # Save installation script
    install_path = Path("/tmp/install_lobster.sh")
    install_path.write_text(install_script)
    install_path.chmod(0o755)
    
    console.print(f"[bold red]✓[/bold red] [white]Installation script created:[/white] [grey74]{install_path}[/grey74]")
    console.print("[red]Run:[/red] [white]curl -sSL https://homara.ai/lobster/install.sh | bash[/white]")


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
