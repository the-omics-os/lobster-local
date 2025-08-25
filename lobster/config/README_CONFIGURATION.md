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
| `method_expert_agent` | Method Expert | `lobster.agents.method_expert.method_expert` | `handoff_to_method_expert` |

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
GENIE_PROFILE=production
```

Available profiles:
- `development` - Lightweight models for development
- `production` - Balanced models for production use  
- `high-performance` - Heavy models for complex analysis
- `ultra-performance` - Ultra models for maximum capability
- `cost-optimized` - Lightweight models to minimize costs
- `heavyweight` - Heavy models across all agents
- `eu-compliant` - EU region models for compliance
- `eu-high-performance` - EU region high-performance models

### 2. Override Specific Agents (Optional)
```bash
# Use different models for different agents
GENIE_SUPERVISOR_MODEL=claude-haiku              # Lightweight supervisor
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus   # Heavy model for complex analysis  
GENIE_METHOD_AGENT_MODEL=claude-sonnet           # Balanced model for literature search
```

### 3. Global Override (Optional)
```bash
# Override all agents with the same model
GENIE_GLOBAL_MODEL=claude-sonnet
```

## Available Models

### US Region Models

| Model Name | Tier | Description |
|------------|------|-------------|
| `claude-3-haiku` | Lightweight | Fast, cost-effective Claude 3 Haiku |
| `claude-3-5-haiku` | Lightweight | Fast, cost-effective Claude 3.5 Haiku |
| `claude-3-sonnet` | Standard | Balanced Claude 3 Sonnet |
| `claude-3-5-sonnet` | Standard | Enhanced Claude 3.5 Sonnet |
| `claude-3-5-sonnet-v2` | Standard | Latest Claude 3.5 Sonnet v2 |
| `claude-4-sonnet` | Standard | Next-generation Claude 4 Sonnet |
| `claude-3-opus` | Heavy | Most capable Claude 3 Opus |
| `claude-4-opus` | Heavy | Advanced Claude 4 Opus |
| `claude-4-1-opus` | Heavy | Latest Claude 4.1 Opus |
| `claude-3-7-sonnet` | Ultra | Highest-performance Claude 3.7 Sonnet |

### EU Region Models (eu-central-1)

| Model Name | Tier | Description |
|------------|------|-------------|
| `claude-3-5-haiku-eu` | Lightweight | EU region Claude 3.5 Haiku |
| `claude-3-5-sonnet-eu` | Standard | EU region Claude 3.5 Sonnet |
| `claude-3-5-sonnet-v2-eu` | Standard | EU region Claude 3.5 Sonnet v2 |
| `claude-4-opus-eu` | Heavy | EU region Claude 4 Opus |
| `claude-4-1-opus-eu` | Heavy | EU region Claude 4.1 Opus |
| `claude-3-7-sonnet-eu` | Ultra | EU region Claude 3.7 Sonnet |

### Convenience Aliases

| Alias | Points To | Description |
|-------|-----------|-------------|
| `claude-haiku` | `claude-3-5-haiku` | Latest Claude 3.5 Haiku |
| `claude-sonnet` | `claude-3-5-sonnet-v2` | Latest Claude 3.5 Sonnet v2 |
| `claude-opus` | `claude-4-1-opus` | Latest Claude 4.1 Opus |

## Configuration Profiles

### Development Profile
- **Supervisor**: `claude-3-5-haiku` (lightweight, fast feedback)
- **Transcriptomics Expert**: `claude-3-5-sonnet` (balanced capability)
- **Method Agent**: `claude-3-5-haiku` (lightweight literature search)

### Production Profile  
- **Supervisor**: `claude-3-5-sonnet-v2` (reliable coordination)
- **Transcriptomics Expert**: `claude-4-opus` (maximum analysis capability)
- **Method Agent**: `claude-3-5-sonnet` (good literature understanding)

### High-Performance Profile
- **Supervisor**: `claude-4-sonnet` (efficient coordination)
- **Transcriptomics Expert**: `claude-3-7-sonnet` (cutting-edge analysis)
- **Method Agent**: `claude-4-opus` (thorough literature analysis)

### Ultra-Performance Profile
- **Supervisor**: `claude-3-7-sonnet` (ultra-high performance coordination)
- **Transcriptomics Expert**: `claude-3-7-sonnet` (maximum capability analysis)
- **Method Agent**: `claude-4-1-opus` (most advanced literature analysis)

### Cost-Optimized Profile
- **Supervisor**: `claude-3-haiku` (minimal overhead)
- **Transcriptomics Expert**: `claude-3-5-sonnet` (necessary capability)
- **Method Agent**: `claude-3-haiku` (basic literature search)

### Heavyweight Profile
- **Supervisor**: `claude-4-1-opus` (heavy coordination)
- **Transcriptomics Expert**: `claude-4-1-opus` (maximum analysis power)
- **Method Agent**: `claude-4-opus` (heavy literature analysis)

### EU-Compliant Profile
- **Supervisor**: `claude-3-5-sonnet-v2-eu` (EU region coordination)
- **Transcriptomics Expert**: `claude-4-1-opus-eu` (EU region maximum capability)
- **Method Agent**: `claude-3-5-sonnet-eu` (EU region literature search)

### EU High-Performance Profile
- **Supervisor**: `claude-3-7-sonnet-eu` (EU region ultra performance)
- **Transcriptomics Expert**: `claude-3-7-sonnet-eu` (EU region cutting-edge analysis)
- **Method Agent**: `claude-4-opus-eu` (EU region heavy literature analysis)

## Environment Variables Reference

### New Configuration Keys

```bash
# Profile selection (recommended approach)
GENIE_PROFILE=production

# Custom configuration file
GENIE_CONFIG_FILE=config/custom_agent_config.json

# Per-agent model overrides
GENIE_SUPERVISOR_MODEL=claude-haiku
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
GENIE_METHOD_AGENT_MODEL=claude-sonnet

# Global model override (overrides all agents)
GENIE_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature overrides
GENIE_SUPERVISOR_TEMPERATURE=0.5
GENIE_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
GENIE_METHOD_AGENT_TEMPERATURE=0.3
```

### Required API Keys (unchanged)
```bash
OPENAI_API_KEY="your-openai-api-key"
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
GENIE_PROFILE=development
GENIE_SUPERVISOR_MODEL=claude-haiku    # Fast supervisor for development
```

### Scenario 2: High-Performance Research
```bash
# In your .env file  
GENIE_PROFILE=high-performance
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-3-7-sonnet  # Latest model for analysis
```

### Scenario 3: Cost Optimization
```bash
# In your .env file
GENIE_PROFILE=cost-optimized
GENIE_GLOBAL_MODEL=claude-haiku  # All agents use lightweight model
```

### Scenario 4: EU Compliance
```bash
# In your .env file
GENIE_PROFILE=eu-compliant
AWS_REGION=eu-central-1
```

### Scenario 5: Custom Mix
```bash
# In your .env file
GENIE_SUPERVISOR_MODEL=claude-haiku              # Lightweight supervisor
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus   # Heavy analysis model
GENIE_METHOD_AGENT_MODEL=claude-sonnet           # Balanced literature search
```

## Custom Configuration Files

Create custom configurations using JSON:

```bash
# Create interactively
python config/config_manager.py create-custom

# Use custom config
export GENIE_CONFIG_FILE=config/custom_agent_config.json
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
GENIE_PROFILE=production
GENIE_SUPERVISOR_MODEL=claude-haiku
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
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
