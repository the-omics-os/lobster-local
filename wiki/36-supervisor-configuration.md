# Supervisor Configuration Guide

## Overview

The Lobster supervisor agent now features a dynamic, configuration-driven prompt system that automatically discovers agents and their capabilities, eliminating the need for manual updates when adding new agents.

## Key Features

- **Automatic Agent Discovery**: Supervisor automatically knows about all registered agents
- **Configurable Behavior**: Customize interaction style via environment variables
- **Dynamic Prompt Generation**: Prompt adapts based on active agents and configuration
- **Backward Compatible**: Default settings maintain existing behavior
- **Multiple Operation Modes**: Research, Production, and Development modes

## Configuration Options

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SUPERVISOR_ASK_QUESTIONS` | bool | true | Whether to ask clarification questions |
| `SUPERVISOR_MAX_QUESTIONS` | int | 3 | Maximum number of clarification questions |
| `SUPERVISOR_REQUIRE_CONFIRMATION` | bool | true | Require confirmation before downloads |
| `SUPERVISOR_REQUIRE_PREVIEW` | bool | true | Require metadata preview before downloads |
| `SUPERVISOR_AUTO_SUGGEST` | bool | true | Automatically suggest next steps |
| `SUPERVISOR_VERBOSE` | bool | false | Verbose delegation explanations |
| `SUPERVISOR_WORKFLOW_GUIDANCE` | str | standard | Guidance level: minimal, standard, detailed |
| `SUPERVISOR_DELEGATION_STRATEGY` | str | auto | Strategy: auto, conservative, aggressive |
| `SUPERVISOR_ERROR_HANDLING` | str | informative | Error mode: silent, informative, verbose |
| `SUPERVISOR_INCLUDE_DATA` | bool | true | Include data context in prompt |
| `SUPERVISOR_INCLUDE_WORKSPACE` | bool | true | Include workspace status |
| `SUPERVISOR_AUTO_DISCOVER` | bool | true | Auto-discover agents from registry |

## Configuration Modes

### Research Mode (Interactive)
Best for exploratory analysis and development.

```bash
export SUPERVISOR_ASK_QUESTIONS=true
export SUPERVISOR_WORKFLOW_GUIDANCE=detailed
export SUPERVISOR_REQUIRE_CONFIRMATION=true
```

**Characteristics:**
- Asks clarification questions (up to 3)
- Provides detailed workflow guidance
- Requires explicit confirmation for downloads
- Includes examples and detailed instructions

### Production Mode (Automated)
Optimized for automated workflows and pipelines.

```bash
export SUPERVISOR_ASK_QUESTIONS=false
export SUPERVISOR_WORKFLOW_GUIDANCE=minimal
export SUPERVISOR_REQUIRE_CONFIRMATION=false
```

**Characteristics:**
- No clarification questions
- Minimal workflow guidance
- Proceeds with downloads automatically
- Streamlined prompt for efficiency

### Development Mode (Verbose)
Useful for debugging and understanding system behavior.

```bash
export SUPERVISOR_VERBOSE=true
export SUPERVISOR_WORKFLOW_GUIDANCE=detailed
export SUPERVISOR_INCLUDE_SYSTEM=true
```

**Characteristics:**
- Detailed delegation explanations
- Full workflow documentation
- System information included
- Maximum transparency

## How It Works

### 1. Agent Discovery
The system automatically discovers all registered agents from the central registry:

```python
# lobster/config/agent_registry.py
AGENT_REGISTRY = {
    'new_agent': AgentConfig(
        name='new_agent',
        display_name='New Agent',
        description='Agent purpose',
        factory_function='lobster.agents.new_agent.new_agent'
    )
}
```

### 2. Dynamic Prompt Building
The supervisor prompt is built dynamically based on:
- Active agents in the system
- Configuration settings
- Current data context
- Workspace status

### 3. Capability Extraction
The system can optionally extract and display agent capabilities:
- Discovers @tool decorated functions
- Parses docstrings for descriptions
- Lists available tools per agent

## Adding New Agents

To add a new agent to the system:

1. **Create the agent module** in `lobster/agents/`
2. **Register in agent_registry.py**:
   ```python
   'new_agent': AgentConfig(
       name='new_agent',
       display_name='New Agent',
       description='What this agent does',
       factory_function='lobster.agents.new_agent.new_agent'
   )
   ```
3. **That's it!** The supervisor automatically discovers and includes the new agent

## Programmatic Usage

### Python API

```python
from lobster.config.supervisor_config import SupervisorConfig
from lobster.agents.supervisor import create_supervisor_prompt
from lobster.core.data_manager_v2 import DataManagerV2

# Create custom configuration
config = SupervisorConfig(
    ask_clarification_questions=False,
    workflow_guidance_level='minimal',
    verbose_delegation=True
)

# Generate prompt
data_manager = DataManagerV2()
prompt = create_supervisor_prompt(
    data_manager=data_manager,
    config=config,
    active_agents=['data_expert_agent', 'singlecell_expert_agent']
)
```

### Integration with Graph

```python
from lobster.agents.graph import create_bioinformatics_graph
from lobster.config.supervisor_config import SupervisorConfig

# Create custom supervisor configuration
supervisor_config = SupervisorConfig(
    ask_clarification_questions=True,
    max_clarification_questions=5
)

# Create graph with custom config
graph = create_bioinformatics_graph(
    data_manager=data_manager,
    supervisor_config=supervisor_config
)
```

## Configuration Validation

The system validates configuration values:
- `workflow_guidance_level`: Must be 'minimal', 'standard', or 'detailed'
- `delegation_strategy`: Must be 'auto', 'conservative', or 'aggressive'
- `error_handling`: Must be 'silent', 'informative', or 'verbose'
- `max_clarification_questions`: Clamped between 0 and 10
- `max_tools_per_agent`: Clamped between 0 and 20

Invalid values automatically fall back to defaults with a warning.

## Monitoring Configuration

### Check Current Configuration

```python
from lobster.config.settings import get_settings

settings = get_settings()
config = settings.get_supervisor_config()

print(f"Mode: {config.get_prompt_mode()}")
print(f"Ask questions: {config.ask_clarification_questions}")
print(f"Guidance level: {config.workflow_guidance_level}")
```

### View Configuration as Dictionary

```python
config_dict = config.to_dict()
print(config_dict)
```

## Performance Considerations

- **Prompt Size**: Dynamic generation keeps prompt size manageable
  - Production mode: ~8,000 characters
  - Standard mode: ~9,500 characters
  - Detailed mode: ~11,000 characters

- **Caching**: Agent capabilities are cached using LRU cache
  - Clear cache: `AgentCapabilityExtractor.clear_cache()`

- **Startup Time**: Minimal impact (<100ms) for prompt generation

## Troubleshooting

### Agents Not Appearing in Prompt
1. Verify agent is registered in `agent_registry.py`
2. Check factory function path is correct
3. Ensure agent module can be imported

### Configuration Not Taking Effect
1. Check environment variable names (must start with `SUPERVISOR_`)
2. Verify boolean values are 'true' or 'false' (lowercase)
3. Check for typos in variable names

### Performance Issues
1. Use minimal workflow guidance for production
2. Disable agent tool discovery if not needed
3. Consider caching prompt for repeated use

## Migration from Static Prompt

The system maintains backward compatibility:
- Default configuration matches previous behavior exactly
- `create_supervisor_prompt(data_manager)` works without changes
- Existing workflows continue functioning

To leverage new features, simply:
1. Set environment variables for desired behavior
2. Or pass a `SupervisorConfig` object to `create_supervisor_prompt()`

## Future Enhancements

Planned improvements include:
- Per-agent configuration overrides
- Dynamic capability extraction from running agents
- Prompt templates for specific domains
- Configuration profiles (save/load configurations)
- Web UI for configuration management