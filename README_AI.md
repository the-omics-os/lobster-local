# Genie AI - Project Architecture for AI Understanding

## Overview

Genie AI is a multi-agent bioinformatics analysis system built with LangGraph and Streamlit. It orchestrates specialized AI agents to perform complex bioinformatics analyses through a structured workflow.

## Core Architecture

```
┌─────────────────┐     ┌────────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│  LangGraph Agent   │────▶│  Data Manager   │
│    (app.py)     │     │     Service        │     │                 │
└─────────────────┘     └────────────────────┘     └─────────────────┘
                                │                           ▲
┌─────────────────┐             │                           │
│   CLI Interface │─────────────┤                           │
│  (agent_cli.py) │             ▼                           │
└─────────────────┘     ┌───────────────────┐               │
                        │     Supervisor     │               │
                        │      Agent        │               │
                        │  (langgraph-      │               │
                        │   supervisor)     │               │
                        └───────────────────┘               │
                                │                           │
                                │ Handoffs                  │
                                ▼                           │
        ┌───────────────────────┼───────────────────────┐   │
        ▼                       ▼                       ▼   │
┌──────────────┐      ┌──────────────────┐    ┌──────────────┐ │
│ Clarify with │      │  Transcriptomics │    │ Method Agent │ │
│    User      │      │     Expert       │    │              │ │
└──────────────┘      └──────────────────┘    └──────────────┘ │
                                │                       │     │
                                │                       │     │
                                ▼                       │     │
                    ┌─────────────────────────┐         │     │
                    │ Terminal Callback       │─────────┘     │
                    │ Handler (Thought        │               │
                    │ Process Display)        │───────────────┘
                    └─────────────────────────┘
```

## Key Components

### 1. Entry Point (`app.py`)
- **Purpose**: Main Streamlit application entry point
- **Responsibilities**:
  - Initialize UI components
  - Manage session state
  - Handle user interactions
  - Display results and plots
- **Key Objects**:
  - `StreamlitUI`: UI management class
  - `DataManager`: Central data handling
  - `LangGraphAgentService`: Multi-agent orchestrator

### 2. Multi-Agent System (`agents/graph.py`)

The system uses the langgraph-supervisor pattern for agent coordination:

#### Graph Structure with Supervisor:
```
START
  │
  ▼
supervisor ◄──────────────────────────────┐
  │                                       │
  │                                       │
  ├─[handoff]──▶ clarify_with_user ──────▶│
  │                                       │
  ├─[handoff]──▶ transcriptomics_expert ◄─┐
  │                     │                 │
  │                     │                 │
  │                     ├─[handoff]───────┼─▶ method_agent
  │                     │                 │       │
  │                     │                 │       │
  │                     ◄─────────────────┼───────┘
  │                     │                 │
  │                     └─[handoff]───────┘
  │                     
  └─[complete]──▶ compile_results ──▶ END
```

#### Node Functions:
- **supervisor_node**: Orchestrates the analysis, delegates to experts
- **clarify_with_user_node**: Handles user clarification requests
- **transcriptomics_expert_node**: Performs domain-specific analysis
- **method_agent_node**: Researches computational methods from literature
- **compile_results_node**: Aggregates and formats final results

### 3. State Management (`agents/state.py`)

#### State Hierarchy:
```
BioinformaticsAgentState (Main)
    ├── SupervisorState
    │   ├── supervisor_messages
    │   ├── research_brief
    │   └── domain_expert_results
    │
    ├── DomainExpertState
    │   ├── expert_messages
    │   ├── task_description
    │   └── analysis_results
    │
    └── MethodAgentState
        ├── method_messages
        ├── research_question
        └── extracted_parameters
```

#### Key State Attributes:
- `messages`: Conversation history (LangChain messages)
- `research_brief`: Parsed user intent
- `expert_results`: Results from domain experts
- `methodology_parameters`: Parameters from literature
- `analysis_complete`: Completion flag
- `final_report`: Compiled analysis report

### 4. Data Management (`core/data_manager.py`)

Central hub for all data operations:

#### Capabilities:
- **Data Storage**: Pandas DataFrames and AnnData objects
- **Plot Management**: Stores Plotly figures with metadata
- **Export Functions**: Creates downloadable data packages
- **Tool Tracking**: Logs all tool usage for reproducibility

#### Key Methods:
```python
set_data()          # Load and validate data
add_plot()          # Store visualization with metadata
get_data_summary()  # Generate data statistics
create_data_package() # Export all data/plots as ZIP
log_tool_usage()    # Track analysis steps
```

### 5. Service Layer (`services/langgraph_agent_service.py`)

Bridges the UI with the multi-agent system:

#### Responsibilities:
- Initialize and manage the agent graph
- Convert chat history to LangChain messages
- Stream agent responses
- Maintain conversation state
- Coordinate multiple callback handlers

#### Key Flow:
```python
run_agent(query) -> 
    Convert chat history -> 
    Create initial state -> 
    Stream graph execution -> 
    Extract response
```

### 6. Terminal Callback Handler (`utils/terminal_callback_handler.py`)

Provides comprehensive visibility into agent reasoning processes:

#### Capabilities:
- **Real-time Agent Monitoring**: Shows which agents are active and their transitions
- **LLM Operation Tracking**: Displays prompts, responses, and processing duration
- **Tool Usage Visualization**: Monitors when agents use tools and their results
- **Chain Execution Flow**: Tracks complex workflow execution
- **Error Handling Display**: Shows detailed error information and retry attempts

#### Output Features:
```python
# Color-coded terminal output with hierarchical indentation
🤖 CHAT MODEL START
  Model: claude-3-5-sonnet
  Human: Analyze my single cell data
🧠 LLM START  
  👤 Agent: Supervisor
  Duration: 1.23s
✓ LLM END
🔧 TOOL START: clustering_service
  Input: {"resolution": 0.5, "data_type": "single_cell"}
  Output: Analysis complete with 8 clusters identified
✓ TOOL END
```

#### Integration:
- Automatically captures all LangChain callback events
- Works with both Streamlit UI and CLI interfaces
- Provides verbose and quiet modes for different user needs

### 7. CLI Interface (`agent_cli.py`)

Terminal-based interface for direct agent system interaction:

#### Features:
- **Interactive Commands**: `!help`, `!status`, `!export`, `clear`, `exit`
- **Rich Terminal Output**: Color-coded responses with emoji indicators
- **Data Export**: Built-in functionality to export analysis results
- **Agent Monitoring**: Real-time display of agent reasoning processes
- **Error Recovery**: Graceful handling of interruptions and errors

#### Command Options:
```bash
python agent_cli.py --verbose    # Enable detailed agent output
python agent_cli.py --quiet      # Minimal output mode
python agent_cli.py --export-dir=/path  # Custom export directory
```

#### Usage Examples:
```bash
🔬 You: Download GSE109564 from GEO and analyze it
🔬 You: Cluster my single-cell data and create UMAP
🔬 You: !status  # Check system status
🔬 You: !export  # Export results
```

## Data Flow

1. **User Input** → Streamlit UI captures query
2. **Agent Service** → Converts to LangChain messages
3. **Graph Execution** → Supervisor analyzes intent
4. **Expert Delegation** → Task routed to appropriate expert
5. **Method Research** → Literature search if needed
6. **Analysis** → Expert performs computation
7. **Results** → Compiled and returned to UI
8. **Visualization** → Plots stored in DataManager
9. **Display** → Results shown in chat interface

## Agent Communication Protocol

Agents communicate through handoffs using the Command pattern from LangGraph:

```bash
# Supervisor handoff to transcriptomics expert
@ tool("transfer_to_transcriptomics_expert")
def handoff_tool(state, task_description, tool_call_id):
    return Command(
        goto="transcriptomics_expert",
        update={"messages": [...], "expert_task": task_description},
        graph=Command.PARENT,
    )

# Expert handoff to method agent
@ tool("request_method_agent")
def method_agent_handoff_tool(state, research_question, tool_call_id):
    return Command(
        goto="method_agent",
        update={"messages": [...], "method_research_question": research_question},
        graph=Command.PARENT,
    )

# Method agent returns to transcriptomics expert
@ tool("return_to_transcriptomics_expert")
def expert_handoff_tool(state, parameters, sources, tool_call_id):
    return Command(
        goto="transcriptomics_expert",
        update={
            "messages": [...],
            "methodology_parameters": parameters,
            "methodology_sources": sources
        },
        graph=Command.PARENT,
    )
```

## Professional Configuration System

### 8. Agent Configuration Management (`config/`)

Genie AI now features a professional configuration system that enables per-agent model configuration for easy testing and deployment:

#### Core Configuration Files:
- **`agent_config.py`**: Type-safe configuration system with model presets and testing profiles
- **`settings.py`**: Updated settings management with agent-specific configuration support
- **`config_manager.py`**: CLI tool for configuration management and testing

#### Configuration Features:
```python
# Per-agent model configuration
supervisor_config = get_agent_configurator().get_agent_config('supervisor')
transcriptomics_config = get_agent_configurator().get_agent_config('transcriptomics_expert')
method_agent_config = get_agent_configurator().get_agent_config('method_agent')

# Each agent can use different models
# Supervisor: claude-haiku (lightweight, fast coordination)
# Transcriptomics Expert: claude-opus (heavy, complex analysis)
# Method Agent: claude-sonnet (balanced literature research)
```

#### Available Model Presets:
- **Lightweight**: `claude-haiku` - Fast, cost-effective for coordination tasks
- **Standard**: `claude-sonnet` - Balanced performance for most tasks
- **Heavy**: `claude-opus` - Maximum capability for complex analysis
- **Ultra**: `claude-3-7-sonnet` - Latest high-performance models
- **Regional**: EU variants for compliance requirements

#### Testing Profiles:
- **Development**: Lightweight models for fast development cycles
- **Production**: Balanced models for production reliability
- **High-Performance**: Heavy models for complex research analysis
- **Cost-Optimized**: Lightweight models to minimize operational costs
- **EU-Compliant**: EU region models for data compliance

#### Environment Configuration:
```bash
# Profile-based configuration (recommended)
GENIE_PROFILE=production

# Per-agent model overrides
GENIE_SUPERVISOR_MODEL=claude-haiku
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-opus
GENIE_METHOD_AGENT_MODEL=claude-sonnet

# Global model override
GENIE_GLOBAL_MODEL=claude-sonnet

# Per-agent temperature control
GENIE_SUPERVISOR_TEMPERATURE=0.5
GENIE_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7
GENIE_METHOD_AGENT_TEMPERATURE=0.3
```

#### CLI Management Tools:
```bash
# View available models and profiles
python config/config_manager.py list-models
python config/config_manager.py list-profiles

# Show current configuration
python config/config_manager.py show-config

# Test configurations
python config/config_manager.py test -p production
python config/config_manager.py test -p development -a supervisor

# Create custom configurations
python config/config_manager.py create-custom
python config/config_manager.py generate-env
```

## File Organization

```
├── agents/              # Multi-agent system components
│   ├── graph.py        # Graph structure and node definitions
│   ├── state.py        # State definitions
│   ├── supervisor.py   # Orchestrator agent (uses configured model)
│   ├── transcriptomics_expert.py  # Domain expert (uses configured model)
│   └── method_agent.py # Literature research agent (uses configured model)
│
├── config/             # Professional configuration system
│   ├── agent_config.py # Type-safe agent configuration with presets
│   ├── settings.py     # Updated settings with per-agent support
│   └── config_manager.py # CLI configuration management tool
│
├── services/           # Business logic services
│   ├── langgraph_agent_service.py  # Agent orchestration
│   ├── clustering_service.py       # Analysis services
│   └── ...
│
├── core/               # Core utilities
│   └── data_manager.py # Central data handling
│
├── utils/              # Utility modules
│   ├── logger.py       # Logging configuration
│   ├── system_prompts.py # System prompts for agents
│   └── terminal_callback_handler.py # Agent reasoning display
│
├── app/                # UI components
│   └── ui.py          # Streamlit interface
│
├── app.py             # Main Streamlit entry point
├── agent_cli.py       # CLI interface for terminal interaction
└── README_CONFIGURATION.md # Comprehensive configuration documentation
```

## Key Design Patterns

1. **Supervisor Pattern**: Central supervisor coordinates agent interactions
2. **Handoff Pattern**: Agents communicate via handoffs with payloads
3. **Command Pattern**: Commands control agent transitions and state updates
4. **Repository Pattern**: DataManager centralizes data access
5. **Service Layer**: Separates business logic from UI
6. **Observer Pattern**: Callback handlers for streaming
7. **Chain of Responsibility**: Agents delegate tasks
8. **Configuration Pattern**: Professional configuration system with type safety
9. **Factory Pattern**: Model configuration factory for different agent types

## Extension Points

- **New Agents**: Add to `agents/` and register in graph, configure in agent configuration system
- **New Services**: Add to `services/` for specific analyses
- **New Tools**: Integrate with agent tools
- **New Data Types**: Extend DataManager
- **New Model Providers**: Extend configuration system with new providers (OpenAI, other Bedrock models, etc.)
- **New Testing Profiles**: Add custom profiles for specific deployment scenarios

## Runtime Behavior

1. Application starts with empty state
2. User provides analysis request
3. Supervisor agent receives the request
4. Supervisor may handoff to clarify_with_user if clarification needed
5. Supervisor hands off to domain experts via Command objects
6. Domain experts may hand off to other agents (e.g., method agent)
7. Agents return control to supervisor after completing their tasks
8. Supervisor compiles results when analysis is complete
9. State persisted for conversation continuity

## Critical Connections

- **DataManager ↔ All Agents**: Shared data access
- **UI ↔ Agent Service**: User interaction bridge
- **Graph ↔ State**: State drives graph execution
- **Experts ↔ Method Agent**: Parameter discovery
- **Services ↔ DataManager**: Analysis results storage

This architecture enables flexible, extensible bioinformatics analysis through coordinated AI agents working on shared data.
