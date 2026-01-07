"""
Shared configuration commands for CLI and Dashboard.

Extracted from cli.py to enable reuse across interfaces.
All commands accept OutputAdapter for UI-agnostic rendering.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lobster.core.client import AgentClient

from lobster.cli_internal.commands.output_adapter import OutputAdapter


def _build_agent_hierarchy(output: OutputAdapter, current_tier: str) -> None:
    """
    Display ASCII hierarchy of agent relationships.

    Shows supervisor at top, then worker agents with their child agents indented.
    """
    from lobster.config.agent_registry import AGENT_REGISTRY, get_valid_handoffs
    from lobster.config.subscription_tiers import is_agent_available

    valid_handoffs = get_valid_handoffs()
    supervisor_targets = valid_handoffs.get("supervisor", set())

    # Filter to available agents first
    available_agents = [
        name for name in sorted(supervisor_targets)
        if is_agent_available(name, current_tier) and AGENT_REGISTRY.get(name)
    ]

    if not available_agents:
        return

    output.print("\n[bold cyan]🔀 Agent Hierarchy[/bold cyan]")
    output.print("[dim]───────────────────────────────────────[/dim]")
    output.print("[bold white]supervisor[/bold white] [dim](orchestrator)[/dim]")

    for i, agent_name in enumerate(available_agents):
        config = AGENT_REGISTRY[agent_name]
        is_last = (i == len(available_agents) - 1)
        branch = "└── " if is_last else "├── "

        output.print(f"  {branch}[yellow]{config.display_name}[/yellow]")

        # Show child agents if any
        if config.child_agents:
            available_children = [
                c for c in config.child_agents
                if is_agent_available(c, current_tier) and AGENT_REGISTRY.get(c)
            ]
            child_prefix = "      " if is_last else "  │   "
            for j, child_name in enumerate(available_children):
                child_config = AGENT_REGISTRY[child_name]
                child_is_last = (j == len(available_children) - 1)
                child_branch = "└── " if child_is_last else "├── "
                output.print(f"{child_prefix}{child_branch}[dim]{child_config.display_name}[/dim]")

    output.print("")


def config_show(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    Show current configuration including provider, profile, config files, and agent models.

    Displays three tables:
    1. Current Configuration (provider, profile)
    2. Configuration Files (workspace, global)
    3. Agent Models (per-agent model configuration)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.llm_factory import LLMFactory
    from lobster.core.config_resolver import ConfigResolver
    from lobster.config.global_config import CONFIG_DIR as GLOBAL_CONFIG_DIR
    from lobster.config.workspace_config import WorkspaceProviderConfig
    from lobster.config.settings import get_settings
    from lobster.config.agent_registry import AGENT_REGISTRY
    from lobster.core.license_manager import get_current_tier
    from lobster.config.subscription_tiers import is_agent_available
    from lobster.config.providers import get_provider

    # Create resolver
    resolver = ConfigResolver(workspace_path=Path(client.workspace_path))

    # Resolve provider and profile
    provider, p_source = resolver.resolve_provider(
        runtime_override=client.provider_override
    )
    profile, pf_source = resolver.resolve_profile()

    # Check if config files exist
    workspace_config_exists = WorkspaceProviderConfig.exists(Path(client.workspace_path))
    global_config_path = GLOBAL_CONFIG_DIR / "providers.json"
    global_config_exists = global_config_path.exists()

    # ========================================================================
    # Table 1: Current Configuration
    # ========================================================================
    # Standard widths: 25 + 35 + 40 = 100 (consistent across all tables)
    config_table_data = {
        "title": "⚙️  Current Configuration",
        "columns": [
            {"name": "Setting", "style": "cyan", "width": 25},
            {"name": "Value", "style": "white", "width": 35},
            {"name": "Source", "style": "yellow", "width": 40},
        ],
        "rows": [
            ["Provider", f"[bold]{provider}[/bold]", p_source],
            ["Profile", f"[bold]{profile}[/bold]", pf_source],
        ]
    }

    output.print_table(config_table_data)

    # ========================================================================
    # Table 2: Config Files Status
    # ========================================================================
    workspace_status = "[green]✓ Exists[/green]" if workspace_config_exists else "[grey50]✗ Not found[/grey50]"
    workspace_path_str = str(Path(client.workspace_path) / "provider_config.json")

    global_status = "[green]✓ Exists[/green]" if global_config_exists else "[grey50]✗ Not found[/grey50]"
    global_path_str = str(global_config_path)

    status_table_data = {
        "title": "📁 Configuration Files",
        "columns": [
            {"name": "Location", "style": "cyan", "width": 25},
            {"name": "Status", "style": "white", "width": 15},
            {"name": "Path", "style": "dim", "width": 60},
        ],
        "rows": [
            ["Workspace Config", workspace_status, workspace_path_str],
            ["Global Config", global_status, global_path_str],
        ]
    }

    output.print_table(status_table_data)

    # ========================================================================
    # Table 3: Per-Agent Model Configuration
    # ========================================================================
    settings = get_settings()
    current_tier = get_current_tier()
    provider_obj = get_provider(provider)

    agent_table_data = {
        "title": "🤖 Agent Models",
        "columns": [
            {"name": "Agent", "style": "cyan", "width": 25},
            {"name": "Model", "style": "yellow", "width": 45},
            {"name": "Source", "style": "dim", "width": 30},
        ],
        "rows": []
    }

    # Show models for available agents
    for agent_name, agent_cfg in AGENT_REGISTRY.items():
        # Filter by license tier
        if not is_agent_available(agent_name, current_tier):
            continue

        try:
            # Get model parameters
            model_params = settings.get_agent_llm_params(agent_name)

            # Resolve model for this agent
            model_id, model_source = resolver.resolve_model(
                agent_name=agent_name,
                runtime_override=None,
                provider=provider
            )

            # If no model resolved, get provider's default model
            if not model_id:
                if provider_obj:
                    model_id = provider_obj.get_default_model()
                    model_source = "provider default"
                else:
                    model_id = model_params.get("model_id", "unknown")
                    model_source = "profile config"

            # Add row
            agent_table_data["rows"].append([
                agent_cfg.display_name,
                model_id,
                model_source
            ])
        except Exception:
            # Skip agents with config errors
            continue

    output.print_table(agent_table_data)

    # ========================================================================
    # Agent Hierarchy (ASCII tree)
    # ========================================================================
    _build_agent_hierarchy(output, current_tier)

    # ========================================================================
    # Usage hints
    # ========================================================================
    output.print("\n[cyan]💡 Available Config Commands:[/cyan]")
    output.print("\n[yellow]Provider Management:[/yellow]")
    output.print("  • [white]/config provider[/white] - List available providers")
    output.print("  • [white]/config provider <name>[/white] - Switch provider (runtime only)")
    output.print("  • [white]/config provider <name> --save[/white] - Switch and persist to workspace")

    output.print("\n[yellow]Model Management:[/yellow]")
    output.print("  • [white]/config model[/white] - List available models for current provider")
    output.print("  • [white]/config model <name>[/white] - Switch model (runtime only)")
    output.print("  • [white]/config model <name> --save[/white] - Switch model and persist to workspace")

    output.print("\n[yellow]Configuration Display:[/yellow]")
    output.print("  • [white]/config[/white] or [white]/config show[/white] - Show this configuration summary")

    return f"Displayed configuration (provider: {provider}, profile: {profile})"


def config_provider_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List all available LLM providers with their configuration status.

    Displays a table showing:
    - Provider name
    - Configuration status (configured or not)
    - Active indicator (● for currently active provider)

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.llm_factory import LLMFactory
    from lobster.config.providers import ProviderRegistry

    available_providers = LLMFactory.get_available_providers()
    current_provider = client.provider_override or LLMFactory.get_current_provider()

    # Build provider table
    provider_table_data = {
        "title": "🔌 LLM Providers",
        "columns": [
            {"name": "Provider", "style": "cyan"},
            {"name": "Status", "style": "white"},
            {"name": "Active", "style": "green"},
        ],
        "rows": []
    }

    # Dynamically fetch all registered providers from ProviderRegistry
    all_providers = ProviderRegistry.get_all()

    for provider_obj in all_providers:
        provider_name = provider_obj.name
        configured = "✓ Configured" if provider_name in available_providers else "✗ Not configured"
        active = "●" if provider_name == current_provider else ""

        status_style = "green" if provider_name in available_providers else "grey50"
        provider_table_data["rows"].append([
            provider_obj.display_name,
            f"[{status_style}]{configured}[/{status_style}]",
            f"[bold green]{active}[/bold green]" if active else ""
        ])

    output.print_table(provider_table_data)

    output.print(f"\n[cyan]💡 Usage:[/cyan]")
    output.print("  • [white]/config provider <name>[/white] - Switch to specified provider (runtime)")
    output.print("  • [white]/config provider <name> --save[/white] - Switch and persist to workspace")

    # Dynamically show available providers
    provider_names = ", ".join([p.name for p in all_providers])
    output.print(f"\n[cyan]Available providers:[/cyan] {provider_names}")

    if current_provider:
        output.print(f"\n[green]✓ Current provider: {current_provider}[/green]")

    return f"Listed providers (current: {current_provider})"


def config_provider_switch(
    client: "AgentClient",
    output: OutputAdapter,
    provider_name: str,
    save: bool = False
) -> Optional[str]:
    """
    Switch to a different LLM provider.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        provider_name: Name of provider to switch to (e.g., 'ollama', 'anthropic')
        save: If True, persist to workspace config; if False, runtime only

    Returns:
        Summary string for conversation history, or None
    """
    new_provider = provider_name.lower()

    if save:
        output.print(f"[yellow]Switching to {new_provider} provider and saving to workspace...[/yellow]")
    else:
        output.print(f"[yellow]Switching to {new_provider} provider (runtime only)...[/yellow]")

    # Switch runtime first
    result = client.switch_provider(new_provider)

    if not result["success"]:
        error_msg = result.get("error", "Unknown error")
        hint = result.get("hint", "")
        output.print(f"[red]✗ {error_msg}[/red]")
        if hint:
            output.print(f"[dim]{hint}[/dim]")
        return None

    # Success message
    output.print(
        f"[green]✓ Successfully switched to {result['provider']} provider[/green]"
    )

    # If not saving, show hint about persisting
    if not save:
        output.print(
            f"[dim]💡 Use [white]/config provider {new_provider} --save[/white] to persist this change[/dim]"
        )
        return f"Switched to {result['provider']} provider (runtime only)"

    # Persist to workspace config
    from lobster.config.workspace_config import WorkspaceProviderConfig

    try:
        workspace_path = Path(client.workspace_path)
        config = WorkspaceProviderConfig.load(workspace_path)
        config.global_provider = new_provider
        config.save(workspace_path)

        output.print(
            f"[green]✓ Saved to workspace config: {workspace_path / 'provider_config.json'}[/green]"
        )
        output.print(
            f"[dim]This provider will be used for all future sessions in this workspace.[/dim]"
        )

        return f"Switched to {result['provider']} provider and saved to workspace"

    except Exception as e:
        output.print(f"[red]✗ Failed to save workspace config: {str(e)}[/red]")
        output.print("[dim]Check workspace directory permissions[/dim]")
        return None


def config_model_list(client: "AgentClient", output: OutputAdapter) -> Optional[str]:
    """
    List available models for the current LLM provider.

    Displays a table with:
    - Model name
    - Display name
    - Description
    - Indicators for current and default models

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.llm_factory import LLMFactory
    from lobster.config.model_service import ModelServiceFactory
    from lobster.core.config_resolver import ConfigResolver
    from lobster.config.workspace_config import WorkspaceProviderConfig

    # Get current provider
    workspace_path = Path(client.workspace_path)
    resolver = ConfigResolver(workspace_path=workspace_path)
    current_provider, provider_source = resolver.resolve_provider(
        runtime_override=client.provider_override
    )

    try:
        service = ModelServiceFactory.get_service(current_provider)

        # For Ollama, check if server is available
        if current_provider == "ollama":
            from lobster.config.ollama_service import OllamaService
            if not OllamaService.is_available():
                output.print("[red]✗ Ollama server not accessible[/red]")
                output.print("[dim]Make sure Ollama is running: 'ollama serve'[/dim]")
                return None

        models = service.list_models()

        if not models:
            if current_provider == "ollama":
                output.print("[yellow]No Ollama models installed[/yellow]")
                output.print("\n[cyan]💡 Install a model:[/cyan]")
                output.print("  ollama pull llama3:8b-instruct")
            else:
                output.print(f"[yellow]No models available for {current_provider}[/yellow]")
            return None

        # Provider-specific table title
        provider_icons = {"anthropic": "🤖", "bedrock": "☁️", "ollama": "🦙", "gemini": "✨"}
        icon = provider_icons.get(current_provider, "🤖")
        title = f"{icon} Available {current_provider.capitalize()} Models"

        # Get current model from config
        config = WorkspaceProviderConfig.load(workspace_path)
        current_model = config.get_model_for_provider(current_provider) if WorkspaceProviderConfig.exists(workspace_path) else None

        model_table_data = {
            "title": title,
            "columns": [
                {"name": "Model", "style": "yellow"},
                {"name": "Display Name", "style": "cyan"},
                {"name": "Description", "style": "white", "max_width": 50},
            ],
            "rows": []
        }

        for model in models:
            is_current = "[green]●[/green]" if model.name == current_model else ""
            is_default = "[dim](default)[/dim]" if model.is_default else ""
            model_table_data["rows"].append([
                f"[bold]{model.name}[/bold] {is_current}",
                f"{model.display_name} {is_default}",
                model.description
            ])

        output.print_table(model_table_data)
        output.print(f"\n[cyan]Current provider:[/cyan] {current_provider} (from {provider_source})")
        output.print(f"\n[cyan]💡 Usage:[/cyan]")
        output.print("  • [white]/config model <name>[/white] - Switch model (runtime)")
        output.print("  • [white]/config model <name> --save[/white] - Switch + persist")
        output.print("  • [white]/config provider <name>[/white] - Change provider first")

        if current_model:
            output.print(f"\n[green]✓ Current model: {current_model}[/green]")

        return f"Listed models for {current_provider} provider"

    except Exception as e:
        output.print(f"[red]✗ Failed to list models for {current_provider}: {str(e)}[/red]")
        output.print("[dim]Check provider configuration[/dim]")
        return None


def config_model_switch(
    client: "AgentClient",
    output: OutputAdapter,
    model_name: str,
    save: bool = False
) -> Optional[str]:
    """
    Switch to a different model for the current provider.

    Args:
        client: AgentClient instance
        output: OutputAdapter for rendering
        model_name: Name of model to switch to
        save: If True, persist to workspace config; if False, runtime only

    Returns:
        Summary string for conversation history, or None
    """
    from lobster.config.llm_factory import LLMFactory
    from lobster.config.model_service import ModelServiceFactory
    from lobster.core.config_resolver import ConfigResolver
    from lobster.config.workspace_config import WorkspaceProviderConfig

    # Get current provider
    workspace_path = Path(client.workspace_path)
    resolver = ConfigResolver(workspace_path=workspace_path)
    current_provider, provider_source = resolver.resolve_provider(
        runtime_override=client.provider_override
    )

    try:
        service = ModelServiceFactory.get_service(current_provider)

        # For Ollama, check server availability
        if current_provider == "ollama":
            from lobster.config.ollama_service import OllamaService
            if not OllamaService.is_available():
                output.print("[red]✗ Ollama server not accessible[/red]")
                output.print("[dim]Make sure Ollama is running: 'ollama serve'[/dim]")
                return None

        # Validate model
        if not service.validate_model(model_name):
            available = ", ".join(service.get_model_names()[:5])
            if len(service.get_model_names()) > 5:
                available += ", ..."
            hint = f"Available models: {available}"
            if current_provider == "ollama":
                hint += f"\nInstall with: ollama pull {model_name}"
            output.print(f"[red]✗ Model '{model_name}' not valid for {current_provider}[/red]")
            output.print(f"[dim]{hint}[/dim]")
            return None

        # Store in environment for this session
        env_var_map = {
            "ollama": "OLLAMA_DEFAULT_MODEL",
            "anthropic": "ANTHROPIC_MODEL",
            "bedrock": "BEDROCK_MODEL",
        }
        env_var = env_var_map.get(current_provider)
        if env_var:
            os.environ[env_var] = model_name

        output.print(f"[green]✓ Switched to model: {model_name}[/green]")
        output.print(f"[dim]Provider: {current_provider}[/dim]")

        if not save:
            output.print("[dim]This change is temporary (session only)[/dim]")
            output.print(f"[dim]To persist: /config model {model_name} --save[/dim]")
            return f"Switched to model {model_name} (runtime only)"

        # Persist to workspace config
        try:
            config = WorkspaceProviderConfig.load(workspace_path)
            config.set_model_for_provider(current_provider, model_name)
            config.save(workspace_path)

            output.print(f"[green]✓ Saved to workspace config ({current_provider}_model)[/green]")
            output.print(f"[dim]Config file: {workspace_path}/provider_config.json[/dim]")
            output.print(f"\n[dim]This model will be used for {current_provider} in this workspace[/dim]")

            return f"Switched to model {model_name} and saved to workspace"

        except Exception as e:
            output.print(f"[red]✗ Failed to save configuration: {e}[/red]")
            output.print("[dim]Check file permissions[/dim]")
            return None

    except Exception as e:
        output.print(f"[red]✗ Failed to switch model: {str(e)}[/red]")
        output.print("[dim]Check provider configuration[/dim]")
        return None
