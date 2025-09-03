# Thinking Configuration for Lobster Agents

This guide explains how to configure and use the thinking/reasoning feature for Lobster agents.

## Overview

The thinking feature allows compatible models (currently Claude 3.7 Sonnet) to output their step-by-step reasoning process before providing an answer. This can improve the quality of responses and provide transparency into the agent's decision-making process.

## Supported Models

Currently, only the following models support thinking:
- `claude-3-7-sonnet` (US region)
- `claude-3-7-sonnet-eu` (EU region)

## Configuration Methods

### 1. Profile-Based Configuration

The easiest way to enable thinking is through the testing profiles in `agent_config.py`. Several profiles already have thinking configured:

```python
# Production profile with thinking enabled
GENIE_PROFILE=production

# High-performance profile with extended thinking
GENIE_PROFILE=high-performance  

# EU high-performance with deep thinking
GENIE_PROFILE=eu-high-performance
```

### 2. Environment Variable Configuration

You can enable thinking for specific agents using environment variables:

```bash
# Enable thinking for the supervisor agent
export GENIE_SUPERVISOR_THINKING_ENABLED=true
export GENIE_SUPERVISOR_THINKING_BUDGET=2000

# Enable thinking for single-cell expert
export GENIE_SINGLECELL_EXPERT_THINKING_ENABLED=true
export GENIE_SINGLECELL_EXPERT_THINKING_BUDGET=5000

# Global thinking preset (applies to all agents with thinking-capable models)
export GENIE_GLOBAL_THINKING=extended  # Options: disabled, light, standard, extended, deep
```

### 3. Programmatic Configuration

You can also configure thinking when initializing the agent system:

```python
from lobster.config.agent_config import initialize_configurator

# Initialize with a profile that has thinking enabled
configurator = initialize_configurator(profile="production")

# Check thinking configuration for an agent
thinking_config = configurator.get_thinking_config("supervisor")
if thinking_config and thinking_config.enabled:
    print(f"Thinking enabled with {thinking_config.budget_tokens} token budget")
```

## Thinking Presets

The system includes several thinking presets:

| Preset | Enabled | Token Budget | Use Case |
|--------|---------|--------------|----------|
| disabled | No | - | No thinking output |
| light | Yes | 1,000 | Quick decisions with minimal reasoning |
| standard | Yes | 2,000 | Balanced reasoning for most tasks |
| extended | Yes | 5,000 | Complex analysis requiring deeper thought |
| deep | Yes | 10,000 | Very complex problems requiring extensive reasoning |

## How It Works

When thinking is enabled for an agent:

1. The agent's LLM parameters are automatically configured with the thinking settings
2. The model will output its reasoning process before providing the final answer
3. The thinking output is separate from the main response and doesn't count against the response token limit
4. The thinking process helps the model work through complex bioinformatics problems step-by-step

## Example Usage

### Running with Thinking Enabled

```bash
# Set profile with thinking configuration
export GENIE_PROFILE=production

# Or enable thinking for specific agents
export GENIE_SUPERVISOR_THINKING_ENABLED=true
export GENIE_SUPERVISOR_THINKING_BUDGET=3000

# Run the application
python -m lobster
```

### Checking Current Configuration

```python
from lobster.config.settings import get_settings

settings = get_settings()

# Print current agent configuration including thinking status
settings.print_agent_configuration()
```

## Important Notes

1. **Model Compatibility**: Thinking only works with models that support it (currently Claude 3.7 Sonnet variants)
2. **Token Budget**: The thinking budget is separate from the main response tokens
3. **Performance**: Thinking adds some latency but can significantly improve response quality
4. **Cost**: Thinking tokens are billed separately and may increase costs

## Troubleshooting

If thinking isn't working:

1. Verify the agent is using a thinking-capable model:
   ```python
   configurator = get_agent_configurator()
   model_config = configurator.get_model_config("supervisor")
   print(f"Model: {model_config.model_id}")
   print(f"Supports thinking: {model_config.supports_thinking}")
   ```

2. Check if thinking is enabled:
   ```python
   thinking_config = configurator.get_thinking_config("supervisor")
   print(f"Thinking enabled: {thinking_config.enabled if thinking_config else False}")
   ```

3. Look for thinking configuration in logs:
   - The graph creation will log when thinking is enabled
   - Example: "Supervisor thinking enabled with 2000 token budget"

## Best Practices

1. **Use thinking for complex tasks**: Enable thinking for agents handling complex reasoning tasks
2. **Adjust token budgets**: Start with standard (2000) and adjust based on task complexity
3. **Monitor performance**: Thinking adds latency, so balance quality vs speed
4. **Profile selection**: Choose profiles that match your use case (development, production, etc.)

## Example Configuration in Code

```python
# agent_config.py - Adding thinking to a custom profile
TESTING_PROFILES = {
    "my-custom-profile": {
        "supervisor": "claude-3-7-sonnet",
        "singlecell_expert": "claude-3-7-sonnet",
        "bulk_rnaseq_expert": "claude-4-sonnet",
        "method_agent": "claude-3-7-sonnet",
        "data_expert": "claude-3-5-haiku",
        "research_agent": "claude-3-5-haiku",
        "thinking": {
            "supervisor": "extended",      # 5000 tokens
            "singlecell_expert": "deep",   # 10000 tokens
            "method_agent": "standard"     # 2000 tokens
        }
    }
}
```

## Future Enhancements

As more models add thinking support, they can be configured by:
1. Setting `supports_thinking=True` in the model configuration
2. Adding the model to profiles with thinking settings
3. The system will automatically apply thinking when available
