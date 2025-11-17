#!/usr/bin/env python3
"""
Configuration Manager CLI Tool for LOBSTER AI.

This tool provides command-line utilities to manage agent configurations,
view available models, switch profiles, and test different setups.
"""

import argparse
import json
import sys
from pathlib import Path

from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lobster.config.agent_config import (  # noqa: E402
    LobsterAgentConfigurator,
    initialize_configurator,
)


def print_colored(text: str, color: str = "white"):
    """Print colored text to terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }

    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def list_available_models():
    """List all available model presets."""
    configurator = LobsterAgentConfigurator()
    models = configurator.list_available_models()

    print_colored("\n🤖 Available Model Presets", "cyan")
    print_colored("=" * 60, "cyan")

    table_data = []
    for name, config in models.items():
        table_data.append(
            [
                name,
                config.tier.value.title(),
                config.region,
                f"{config.temperature}",
                (
                    config.description[:40] + "..."
                    if len(config.description) > 40
                    else config.description
                ),
            ]
        )

    headers = ["Preset Name", "Tier", "Region", "Temp", "Max Tokens", "Description"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


def list_available_profiles():
    """List all available testing profiles."""
    configurator = LobsterAgentConfigurator()
    profiles = configurator.list_available_profiles()

    print_colored("\n⚙️  Available Testing Profiles", "cyan")
    print_colored("=" * 60, "cyan")

    for profile_name, config in profiles.items():
        print_colored(f"\n📋 {profile_name.title()}", "yellow")
        for agent, model in config.items():
            print(f"   {agent}: {model}")


def show_current_config(profile: str = None):
    """Show current configuration."""
    configurator = (
        initialize_configurator(profile=profile)
        if profile
        else LobsterAgentConfigurator()
    )
    configurator.print_current_config()


def test_configuration(profile: str, agent: str = None):
    """Test a specific configuration."""
    try:
        configurator = initialize_configurator(profile=profile)

        if agent:
            # Test specific agent
            try:
                config = configurator.get_agent_model_config(agent)
                configurator.get_llm_params(agent)

                print_colored(f"\n✅ Agent '{agent}' configuration is valid", "green")
                print(f"   Model: {config.model_config.model_id}")
                print(f"   Tier: {config.model_config.tier.value}")
                print(f"   Region: {config.model_config.region}")

            except KeyError:
                print_colored(
                    f"\n❌ Agent '{agent}' not found in profile '{profile}'", "red"
                )
                return False
        else:
            # Test all agents dynamically
            print_colored(f"\n🧪 Testing Profile: {profile}", "yellow")
            all_valid = True

            # Get all agents from the configurator's DEFAULT_AGENTS
            available_agents = configurator.DEFAULT_AGENTS

            for agent_name in available_agents:
                try:
                    config = configurator.get_agent_model_config(agent_name)
                    configurator.get_llm_params(agent_name)
                    print_colored(
                        f"   ✅ {agent_name}: {config.model_config.model_id}", "green"
                    )
                except Exception as e:
                    print_colored(f"   ❌ {agent_name}: {str(e)}", "red")
                    all_valid = False

            if all_valid:
                print_colored(
                    f"\n🎉 Profile '{profile}' is fully configured and valid!", "green"
                )
            else:
                print_colored(
                    f"\n⚠️  Profile '{profile}' has configuration issues", "yellow"
                )

        return True

    except Exception as e:
        print_colored(f"\n❌ Error testing configuration: {str(e)}", "red")
        return False


def create_custom_config():
    """Interactive creation of custom configuration."""
    print_colored("\n🛠️  Create Custom Configuration", "cyan")
    print_colored("=" * 50, "cyan")

    configurator = LobsterAgentConfigurator()
    available_models = configurator.list_available_models()

    # Show available models
    print_colored("\nAvailable models:", "yellow")
    for i, (name, config) in enumerate(available_models.items(), 1):
        print(f"{i:2}. {name} ({config.tier.value}, {config.region})")

    config_data = {"profile": "custom", "agents": {}}

    # Use dynamic agent list
    agents = configurator.DEFAULT_AGENTS

    for agent in agents:
        print_colored(f"\nConfiguring {agent}:", "yellow")
        print("Choose a model preset (enter number or name):")

        choice = input(f"Model for {agent}: ").strip()

        # Handle numeric choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(available_models):
                model_name = list(available_models.keys())[idx]
            else:
                print_colored("Invalid choice, using default (claude-sonnet)", "yellow")
                model_name = "claude-sonnet"
        else:
            # Handle name choice
            if choice in available_models:
                model_name = choice
            else:
                print_colored("Invalid choice, using default (claude-sonnet)", "yellow")
                model_name = "claude-sonnet"

        model_config = available_models[model_name]
        config_data["agents"][agent] = {
            "model_config": {
                "provider": model_config.provider.value,
                "model_id": model_config.model_id,
                "tier": model_config.tier.value,
                "temperature": model_config.temperature,
                "region": model_config.region,
                "description": model_config.description,
            },
            "enabled": True,
            "custom_params": {},
        }

        print_colored(f"   Selected: {model_name}", "green")

    # Save configuration
    config_file = "config/custom_agent_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    print_colored(f"\n✅ Custom configuration saved to: {config_file}", "green")
    print_colored("To use this configuration, set:", "yellow")
    print_colored(f"   export LOBSTER_CONFIG_FILE={config_file}", "yellow")


def generate_env_template():
    """Generate .env template with all available options."""
    template = """# LOBSTER AI Configuration Template
# Copy this file to .env and configure as needed

# =============================================================================
# API KEYS (Required)
# =============================================================================
AWS_BEDROCK_ACCESS_KEY="your-aws-access-key-here"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key-here"
NCBI_API_KEY="your-ncbi-api-key-here"

# =============================================================================
# LOBSTER CLOUD CONFIGURATION (Optional)
# =============================================================================
# Set these to use Lobster Cloud instead of local processing
# Get your API key from https://cloud.lobster.ai or contact info@omics-os.com

# LOBSTER_CLOUD_KEY="your-cloud-api-key-here"
# LOBSTER_ENDPOINT="https://api.lobster.omics-os.com"  # Optional: defaults to production

# When LOBSTER_CLOUD_KEY is set, all processing will be done in the cloud
# When not set, Lobster will run locally with full functionality

# =============================================================================
# AGENT CONFIGURATION (Professional System)
# =============================================================================

# Profile-based configuration (recommended)
# Available profiles: development, production, cost-optimized
LOBSTER_PROFILE=production

# OR use custom configuration file
# LOBSTER_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides (optional)
# Available models: claude-haiku, claude-sonnet, claude-sonnet-eu, claude-opus, claude-opus-eu, claude-3-7-sonnet, claude-3-7-sonnet-eu
# LOBSTER_SUPERVISOR_MODEL=claude-haiku
# LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
# LOBSTER_METHOD_AGENT_MODEL=claude-sonnet
# LOBSTER_GENERAL_CONVERSATION_MODEL=claude-haiku

# Global model override (overrides all agents)
# LOBSTER_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
# LOBSTER_SUPERVISOR_TEMPERATURE=0.5
# LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
# LOBSTER_METHOD_AGENT_TEMPERATURE=0.3

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Server configuration
PORT=8501
HOST=0.0.0.0
DEBUG=False

# Data processing
LOBSTER_MAX_FILE_SIZE_MB=500
LOBSTER_CLUSTER_RESOLUTION=0.5
LOBSTER_CACHE_DIR=data/cache

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Development setup (Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant)
# LOBSTER_PROFILE=development

# Example 2: Production setup (Claude 4 Sonnet for all agents, 3.5 Sonnet v2 for assistant)
# LOBSTER_PROFILE=production

# Example 3: Cost-optimized setup (Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant)
# LOBSTER_PROFILE=cost-optimized
"""

    with open(".env.template", "w") as f:
        f.write(template)

    print_colored("✅ Environment template saved to: .env.template", "green")
    print_colored("Copy this file to .env and configure your API keys", "yellow")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LOBSTER AI Configuration Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list-models              # Show available models
  %(prog)s list-profiles            # Show available profiles
  %(prog)s show-config              # Show current configuration
  %(prog)s show-config -p development  # Show specific profile
  %(prog)s test -p production       # Test a profile
  %(prog)s test -p production -a supervisor  # Test specific agent
  %(prog)s create-custom           # Create custom configuration
  %(prog)s generate-env            # Generate .env template
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    subparsers.add_parser("list-models", help="List available model presets")

    # List profiles command
    subparsers.add_parser("list-profiles", help="List available testing profiles")

    # Show config command
    show_parser = subparsers.add_parser(
        "show-config", help="Show current configuration"
    )
    show_parser.add_argument("-p", "--profile", help="Profile to show")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test configuration")
    test_parser.add_argument("-p", "--profile", required=True, help="Profile to test")
    test_parser.add_argument("-a", "--agent", help="Specific agent to test")

    # Create custom command
    subparsers.add_parser(
        "create-custom", help="Create custom configuration interactively"
    )

    # Generate env command
    subparsers.add_parser("generate-env", help="Generate .env template file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "list-models":
            list_available_models()
        elif args.command == "list-profiles":
            list_available_profiles()
        elif args.command == "show-config":
            show_current_config(args.profile)
        elif args.command == "test":
            test_configuration(args.profile, args.agent)
        elif args.command == "create-custom":
            create_custom_config()
        elif args.command == "generate-env":
            generate_env_template()

    except KeyboardInterrupt:
        print_colored("\n\n👋 Goodbye!", "yellow")
    except Exception as e:
        print_colored(f"\n❌ Error: {str(e)}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()
