# Lobster AI Professional Configuration System

This document describes the professional configuration system for Lobster AI that provides centralized agent management and per-agent model configuration for easy testing and deployment.

## Overview

The new configuration system provides:
- **Centralized Agent Registry** - Single-source configuration for all system agents
- **Per-agent model configuration** - Different models for different agents
- **Testing profiles** - Pre-configured setups for different scenarios
- **Environment overrides** - Quick changes via environment variables
- **Type-safe configuration** - Prevents configuration errors
- **CLI management tools** - Easy configuration management

## 🔧 Centralized Agent Registry

The system now uses a centralized agent registry (`lobster/config/agent_registry.py`) that defines all agents in one place, eliminating the need to update multiple files when adding new agents.

### Benefits
- **Single Source of Truth**: All agent definitions in one location
- **Reduced Redundancy**: No more forgetting to update callbacks or graph configuration
- **Professional Error Prevention**: Eliminates common mistakes when adding agents
- **Dynamic Loading**: Agents are loaded dynamically from the registry
- **Easy Maintenance**: Add new agents by updating only the registry

### Current Registered Agents

| Agent Name | Display Name | Factory Function | Handoff Tool |
|------------|--------------|------------------|--------------|
| `data_expert_agent` | Data Expert | `lobster.agents.data_expert.data_expert` | `handoff_to_data_expert` |
| `singlecell_expert_agent` | Single-Cell Expert | `lobster.agents.singlecell_expert.singlecell_expert` | `handoff_to_singlecell_expert` |
| `bulk_rnaseq_expert_agent` | Bulk RNA-seq Expert | `lobster.agents.bulk_rnaseq_expert.bulk_rnaseq_expert` | `handoff_to_bulk_rnaseq_expert` |
| ~~`method_expert_agent`~~ | ~~Method Expert~~ | ~~`lobster.agents.method_expert.method_expert`~~ | **DEPRECATED v2.2+** - merged into `research_agent` |

### System Agents
These agents are tracked by the callback system but don't require factory functions:
- `supervisor` - Main coordination agent
- `transcriptomics_expert` - Legacy transcriptomics agent (if used)
- `method_agent` - Alternative method agent name
- `clarify_with_user` - User interaction agent

### Adding New Agents

To add a new agent to the system, simply update the `AGENT_REGISTRY` in `lobster/config/agent_registry.py`:

```python
AGENT_REGISTRY = {
    # ... existing agents ...
    'new_agent_name': AgentConfig(
        name='new_agent_name',
        display_name='New Agent',
        description='Handles new functionality',
        factory_function='lobster.agents.new_agent.new_agent',
        handoff_tool_name='handoff_to_new_agent',
        handoff_tool_description='Assign tasks to the new agent'
    ),
}
```

The system will automatically:
- ✅ Create the agent in the graph
- ✅ Generate handoff tools
- ✅ Update callback detection
- ✅ Include in agent lists

**No other files need to be modified!**

## Quick Start

### 1. Set Your Profile
```bash
# Set in your .env file
LOBSTER_PROFILE=production
```

Available profiles:
- `development` - Supervisor & experts: Claude 4 Sonnet, Assistant: Claude 3.7 Sonnet - consistent expert-tier development
- `production` - Supervisor: Claude 4.5 Sonnet, Experts: Claude 4 Sonnet, Assistant: Claude 3.7 Sonnet - optimal production (default)
- `godmode` - All agents: Claude 4.5 Sonnet - maximum performance

### 2. Override Specific Agents (Optional)
```bash
# Use different models for different agents
LOBSTER_SUPERVISOR_MODEL=claude-3-7-sonnet                 # Development model
LOBSTER_SINGLECELL_EXPERT_AGENT_MODEL=claude-4-5-sonnet   # Maximum capability
LOBSTER_METHOD_EXPERT_AGENT_MODEL=claude-4-sonnet         # Production model
```

### 3. Global Override (Optional)
```bash
# Override all agents with the same model
LOBSTER_GLOBAL_MODEL=claude-4-5-sonnet
```

## Available Models

The following models are available in the current configuration:

| Model Name | Tier | Description | Model ID |
|------------|------|-------------|----------|
| `claude-3-7-sonnet` | Ultra | Development and worker tier model | `us.anthropic.claude-3-7-sonnet-20250219-v1:0` |
| `claude-4-sonnet` | Ultra | Production tier model (balanced) | `us.anthropic.claude-sonnet-4-20250514-v1:0` |
| `claude-4-5-sonnet` | Ultra | Highest performance model | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |

All models support extended thinking/reasoning capabilities and are deployed in AWS Bedrock US East region by default.

## Configuration Profiles

### Development Profile
- **Supervisor**: `claude-4-sonnet` (consistent expert-tier coordination)
- **Expert Agents**: `claude-4-sonnet` (consistent expert-tier analysis)
- **Assistant**: `claude-3-7-sonnet` (cost-effective interface)
- **Use Case**: Fast development and testing with consistent expert-tier performance

### Production Profile (Default)
- **Supervisor**: `claude-4-5-sonnet` (optimal production coordination)
- **Expert Agents**: `claude-4-sonnet` (balanced expert analysis)
- **Assistant**: `claude-3-7-sonnet` (cost-effective interface)
- **Use Case**: Production deployments with optimal performance and cost balance

### Godmode Profile
- **Supervisor**: `claude-4-5-sonnet` (maximum capability)
- **Expert Agents**: `claude-4-5-sonnet` (maximum capability)
- **Assistant**: `claude-4-5-sonnet` (maximum capability)
- **Use Case**: Maximum performance for demanding analyses

## Environment Variables Reference

### New Configuration Keys

```bash
# Profile selection (recommended approach)
LOBSTER_PROFILE=production

# Custom configuration file
LOBSTER_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides
LOBSTER_SUPERVISOR_MODEL=claude-haiku
LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
LOBSTER_METHOD_AGENT_MODEL=claude-sonnet

# Global model override (overrides all agents)
LOBSTER_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
LOBSTER_SUPERVISOR_TEMPERATURE=0.5
LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
LOBSTER_METHOD_AGENT_TEMPERATURE=0.3
```

### Required API Keys (unchanged)
```bash
AWS_BEDROCK_ACCESS_KEY="your-aws-access-key"
AWS_BEDROCK_SECRET_ACCESS_KEY="your-aws-secret-key"
NCBI_API_KEY="your-ncbi-api-key"
```

## CLI Management Tool

Use the configuration manager CLI for easy setup:

```bash
# List available models
python config/config_manager.py list-models

# List available profiles
python config/config_manager.py list-profiles

# Show current configuration
python config/config_manager.py show-config

# Show specific profile configuration
python config/config_manager.py show-config -p development

# Test a profile
python config/config_manager.py test -p production

# Test specific agent in a profile
python config/config_manager.py test -p production -a supervisor

# Create custom configuration interactively
python config/config_manager.py create-custom

# Generate .env template
python config/config_manager.py generate-env
```

## Example Usage Scenarios

### Scenario 1: Development Testing
```bash
# In your .env file
LOBSTER_PROFILE=development
# Supervisor & experts: Claude 4 Sonnet, Assistant: Claude 3.7 Sonnet
```

### Scenario 2: Production Deployment
```bash
# In your .env file
LOBSTER_PROFILE=production
# Supervisor: Claude 4.5 Sonnet, Experts: Claude 4 Sonnet, Assistant: Claude 3.7 Sonnet (default)
```

### Scenario 3: Maximum Performance
```bash
# In your .env file
LOBSTER_PROFILE=godmode
# All agents: Claude 4.5 Sonnet
```

### Scenario 4: Custom Agent Override
```bash
# In your .env file
LOBSTER_PROFILE=production
LOBSTER_SINGLECELL_EXPERT_AGENT_MODEL=claude-4-5-sonnet  # Override specific agent to godmode
```

## Custom Configuration Files

Create custom configurations using JSON:

```bash
# Create interactively
python config/config_manager.py create-custom

# Use custom config
export LOBSTER_CONFIG_FILE=config/custom_agent_config.json
```

Example custom configuration file:
```json
{
  "profile": "custom",
  "agents": {
    "supervisor": {
      "model_config": {
        "provider": "bedrock_anthropic",
        "model_id": "us.anthropic.claude-3-haiku-20240307-v1:0",
        "tier": "lightweight",
        "temperature": 0.5,
        "region": "us-east-1"
      }
    },
    "transcriptomics_expert": {
      "model_config": {
        "provider": "bedrock_anthropic", 
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "tier": "heavy",
        "temperature": 0.8,
        "region": "us-east-1"
      }
    },
    "method_agent": {
      "model_config": {
        "provider": "bedrock_anthropic",
        "model_id": "us.anthropic.claude-3-5-sonnet-20241022-v2:0", 
        "tier": "standard",
        "temperature": 0.3,
        "region": "us-east-1"
      }
    }
  }
}
```

## Migration from Old System

The new system is backward compatible. Your existing `.env` configuration will continue to work as a fallback.

### New way (recommended):
```bash
LOBSTER_PROFILE=production
LOBSTER_SUPERVISOR_MODEL=claude-haiku
LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
```

## Troubleshooting

### Check Configuration
```bash
python config/config_manager.py show-config
```

### Test Configuration
```bash
python config/config_manager.py test -p your-profile
```

### Validate Specific Agent
```bash
python config/config_manager.py test -p production -a transcriptomics_expert
```

### Generate Template
```bash
python config/config_manager.py generate-env
cp .env.template .env
# Edit .env with your API keys
```

## Benefits

1. **Quick Testing** - Switch between configurations instantly
2. **Cost Control** - Use lightweight models for development  
3. **Performance Optimization** - Use heavy models only where needed
4. **Regional Compliance** - Easy EU/region-specific deployments
5. **Type Safety** - Prevent configuration errors
6. **Easy Management** - CLI tools for configuration management
7. **Backward Compatibility** - Existing setups continue to work

The new system makes it easy to experiment with different model combinations while maintaining production stability and cost control.
