# Creating Agents - Lobster AI Agent Development Guide

## 🎯 Overview

This guide covers how to create new specialized agents in the Lobster AI system. Agents are the core orchestrators that handle specific bioinformatics domains, coordinate workflows, and manage user interactions through natural language processing.

## 🏗️ Agent Architecture

### Agent Responsibilities
- **Domain Expertise**: Handle specific bioinformatics workflows (RNA-seq, proteomics, etc.)
- **Tool Coordination**: Orchestrate calls to stateless services
- **Data Management**: Interact with DataManagerV2 for modality handling
- **User Communication**: Provide formatted responses and guidance
- **Workflow Handoffs**: Transfer tasks between specialized agents

### Agent Components
```
Agent Implementation
├── Factory Function       # Entry point for agent creation
├── Agent Tools           # Domain-specific functionality (@tool decorators)
├── LLM Configuration     # Model parameters and callbacks
├── State Management      # Agent-specific state schema
├── Assistant Classes     # LLM-powered analysis helpers
└── Handoff Integration   # Inter-agent communication
```

## 📋 Agent Registry System

### 1. Registry Configuration
All agents must be registered in the centralized registry to be available in the system.

```python
# lobster/config/agent_registry.py
@dataclass
class AgentRegistryConfig:
    name: str                          # Unique identifier
    display_name: str                  # Human-readable name
    description: str                   # Agent capabilities description
    factory_function: str             # Module path to factory function
    handoff_tool_name: Optional[str]  # Auto-generated handoff tool name
    handoff_tool_description: Optional[str]  # Handoff tool description
```

### 2. Adding to Registry
```python
# Add to AGENT_REGISTRY dictionary
AGENT_REGISTRY = {
    'your_new_agent': AgentRegistryConfig(
        name='your_new_agent',
        display_name='Your New Agent',
        description='Handles your specific bioinformatics workflow',
        factory_function='lobster.agents.your_agent.your_agent_factory',
        handoff_tool_name='handoff_to_your_agent',
        handoff_tool_description='Assign specific workflow tasks to your agent'
    ),
}
```

### 3. Automatic Supervisor Discovery (v2.3+)

Once registered in `agent_registry.py`, your agent is **automatically discovered** by the supervisor:

- **No manual updates needed**: The supervisor dynamically discovers all registered agents
- **Automatic delegation rules**: The supervisor creates delegation rules based on your agent's description
- **Dynamic prompt generation**: Your agent appears in the supervisor's available experts list
- **Capability extraction**: The system can optionally extract your agent's @tool functions

This eliminates the need to manually update `supervisor.py` or `graph.py` when adding new agents.

## 🔨 Creating a New Agent

### Step 1: Create Agent Module

Create a new Python file in `lobster/agents/`:

```python
# lobster/agents/your_agent.py
"""
Your Agent for handling specific bioinformatics workflow.

This agent specializes in [describe the specific domain and capabilities].
"""

from typing import List, Dict, Optional
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrockConverse

from lobster.agents.state import YourAgentState  # Create if needed
from lobster.config.settings import get_settings
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def your_agent_factory(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "your_agent",
    handoff_tools: List = None
):
    """
    Create your specialized agent.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional callback for LLM monitoring
        agent_name: Agent identifier (should match registry)
        handoff_tools: List of handoff tools for agent communication

    Returns:
        LangGraph agent instance
    """

    # Initialize LLM with agent-specific configuration
    settings = get_settings()
    model_params = settings.get_agent_llm_params('your_agent')
    llm = ChatBedrockConverse(**model_params)

    if callback_handler and hasattr(llm, 'with_config'):
        llm = llm.with_config(callbacks=[callback_handler])

    # Define agent-specific tools
    tools = [
        check_available_modalities,
        perform_domain_analysis,
        # Add more tools as needed
    ]

    # Add handoff tools if provided
    if handoff_tools:
        tools.extend(handoff_tools)

    # Create agent with system prompt
    system_prompt = """You are a specialized bioinformatics agent for [domain].

    Your responsibilities include:
    - [Specific responsibility 1]
    - [Specific responsibility 2]
    - [Specific responsibility 3]

    Always use the available tools to:
    1. Check available data modalities first
    2. Validate data before analysis
    3. Provide detailed analysis results
    4. Suggest next steps or handoffs when appropriate

    When analysis is complete, provide a comprehensive summary including:
    - What analysis was performed
    - Key findings and statistics
    - Generated visualizations
    - Recommendations for next steps
    """

    return create_react_agent(
        llm,
        tools,
        state_schema=YourAgentState,  # Optional custom state
        system_prompt=system_prompt
    )
```

### Step 2: Define Agent Tools

Each tool should follow the standard pattern:

```python
@tool
def check_available_modalities() -> str:
    """Check what data modalities are currently available."""
    try:
        modalities = data_manager.list_modalities()
        if not modalities:
            return "No data modalities are currently loaded. Use data loading tools first."

        summary = "Available data modalities:\n"
        for modality in modalities:
            adata = data_manager.get_modality(modality)
            summary += f"- {modality}: {adata.n_obs} observations, {adata.n_vars} features\n"

        return summary

    except Exception as e:
        logger.error(f"Error checking modalities: {e}")
        return f"Error checking available data: {str(e)}"


@tool
def perform_domain_analysis(
    modality_name: str,
    parameter1: float = 1.0,
    parameter2: str = "default"
) -> str:
    """
    Perform domain-specific analysis on the specified modality.

    Args:
        modality_name: Name of the data modality to analyze
        parameter1: Analysis parameter with default value
        parameter2: Another analysis parameter

    Returns:
        Formatted analysis results
    """
    try:
        # 1. Validate modality exists
        if modality_name not in data_manager.list_modalities():
            raise ValueError(f"Modality '{modality_name}' not found")

        # 2. Get data and call stateless service
        adata = data_manager.get_modality(modality_name)

        from lobster.tools.your_service import YourService
        service = YourService()

        result_adata, statistics = service.perform_analysis(
            adata,
            parameter1=parameter1,
            parameter2=parameter2
        )

        # 3. Store results with descriptive naming
        result_modality = f"{modality_name}_analyzed"
        data_manager.modalities[result_modality] = result_adata

        # 4. Log operation for provenance
        data_manager.log_tool_usage(
            "perform_domain_analysis",
            {"parameter1": parameter1, "parameter2": parameter2},
            statistics
        )

        # 5. Format response
        response = f"""Analysis completed successfully!

**Analysis Parameters:**
- Parameter 1: {parameter1}
- Parameter 2: {parameter2}

**Key Results:**
{format_statistics(statistics)}

**New Dataset:** {result_modality}
- Observations: {result_adata.n_obs}
- Features: {result_adata.n_vars}

Use this dataset for downstream analysis or visualization.
"""
        return response

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return f"Analysis failed: {str(e)}"


def format_statistics(stats: Dict) -> str:
    """Helper function to format analysis statistics."""
    formatted = ""
    for key, value in stats.items():
        if isinstance(value, float):
            formatted += f"- {key}: {value:.3f}\n"
        else:
            formatted += f"- {key}: {value}\n"
    return formatted
```

### Step 3: Create Agent State (Optional)

If your agent needs custom state beyond the default:

```python
# lobster/agents/state.py (add to existing file)
from typing import TypedDict, List, Optional, Dict, Any

class YourAgentState(TypedDict):
    """State schema for your specialized agent."""
    messages: List[Any]
    analysis_parameters: Optional[Dict[str, Any]]
    current_modality: Optional[str]
    workflow_stage: str
    # Add other agent-specific state fields
```

### Step 4: Create Assistant Class (Optional)

For complex LLM-powered analysis:

```python
# lobster/agents/your_agent_assistant.py
"""Assistant class for LLM-powered analysis tasks."""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class YourAnalysisConfig:
    """Configuration for your analysis workflow."""
    parameter1: float
    parameter2: str
    advanced_options: Dict[str, Any]


class YourAgentAssistant:
    """LLM-powered assistant for your domain-specific tasks."""

    def __init__(self):
        self.llm = None  # Initialize if needed for complex tasks

    def extract_analysis_parameters(
        self,
        metadata: Dict[str, Any],
        user_input: str
    ) -> YourAnalysisConfig:
        """
        Extract analysis parameters from metadata and user input.

        Args:
            metadata: Dataset metadata
            user_input: User's analysis request

        Returns:
            Configuration object with extracted parameters
        """
        # Implement parameter extraction logic
        # This could use LLM for complex parameter inference

        return YourAnalysisConfig(
            parameter1=1.0,  # Extracted or default
            parameter2="inferred",
            advanced_options={}
        )
```

### Step 5: Register Agent

Add your agent to the registry:

```python
# lobster/config/agent_registry.py
AGENT_REGISTRY['your_new_agent'] = AgentRegistryConfig(
    name='your_new_agent',
    display_name='Your New Agent',
    description='Handles your specific bioinformatics workflow with advanced analysis capabilities',
    factory_function='lobster.agents.your_agent.your_agent_factory',
    handoff_tool_name='handoff_to_your_agent',
    handoff_tool_description='Assign workflow tasks to the specialized agent'
)
```

## 🔄 Agent Communication & Handoffs

### Handoff Tools
Handoff tools are automatically generated from registry configuration:

```python
# Handoff tools allow agents to transfer tasks
@tool
def handoff_to_your_agent(
    task_description: str,
    state: Dict,
    tool_call_id: str
) -> Command:
    """Transfer task to your specialized agent."""
    # Auto-generated from registry configuration
    pass
```

### Best Practices for Handoffs
1. **Clear Task Description**: Provide detailed context when handing off
2. **Data Validation**: Ensure required data is available before handoff
3. **State Preservation**: Important state should be maintained across handoffs
4. **Error Handling**: Graceful handling of handoff failures

```python
# Example handoff logic in agent tool
if requires_specialized_analysis:
    return f"""This task requires specialized analysis. I'll transfer you to the {target_agent} agent.

    **Transfer Reason:** {reason}
    **Current Data:** {current_modality}
    **Next Steps:** {recommended_steps}

    Please use the handoff tool to continue with specialized analysis.
    """
```

## 🧪 Testing Your Agent

### Unit Tests
```python
# tests/unit/agents/test_your_agent.py
import pytest
from unittest.mock import Mock

from lobster.agents.your_agent import your_agent_factory
from lobster.core.data_manager_v2 import DataManagerV2


class TestYourAgent:

    @pytest.fixture
    def mock_data_manager(self):
        return Mock(spec=DataManagerV2)

    def test_agent_creation(self, mock_data_manager):
        """Test agent factory creates agent successfully."""
        agent = your_agent_factory(mock_data_manager)
        assert agent is not None

    def test_check_modalities_tool(self, mock_data_manager):
        """Test modality checking tool."""
        mock_data_manager.list_modalities.return_value = ['test_data']

        # Test tool execution
        # Implementation depends on your testing framework
```

### Integration Tests
```python
# tests/integration/test_your_agent_integration.py
def test_your_agent_workflow(client, sample_data):
    """Test complete workflow with your agent."""

    # Load test data
    response = client.query("Load the sample dataset")
    assert response['success']

    # Test your agent's functionality
    response = client.query("Perform your specialized analysis")
    assert response['success']
    assert 'analyzed' in response['response']
```

## 📚 Best Practices

### 1. Tool Design
- **Single Responsibility**: Each tool should have a clear, focused purpose
- **Error Handling**: Comprehensive error handling with specific exceptions
- **Parameter Validation**: Validate all inputs before processing
- **Descriptive Returns**: Provide detailed, formatted responses

### 2. Data Management
- **Modality Validation**: Always check data exists before processing
- **Naming Conventions**: Follow professional naming patterns
- **Provenance Logging**: Log all operations for reproducibility
- **Result Storage**: Store intermediate and final results appropriately

### 3. Scientific Rigor
- **Parameter Documentation**: Document all analysis parameters
- **Statistical Validation**: Include proper statistical methods
- **Quality Control**: Implement appropriate QC checks
- **Reproducibility**: Ensure analyses can be reproduced

### 4. User Experience
- **Clear Communication**: Provide informative responses
- **Progress Indication**: Show progress for long-running operations
- **Error Messages**: User-friendly error descriptions
- **Next Steps**: Suggest logical next steps in workflows

## 🔍 Debugging Agents

### Common Issues
1. **Registry Not Found**: Check agent name matches registry exactly
2. **Import Errors**: Verify factory function path is correct
3. **Tool Failures**: Check data_manager access and service imports
4. **Handoff Issues**: Ensure handoff tools are properly configured

### Debugging Commands
```python
# Check registry
from lobster.config.agent_registry import AGENT_REGISTRY
print(AGENT_REGISTRY.keys())

# Test factory function
from lobster.agents.your_agent import your_agent_factory
agent = your_agent_factory(mock_data_manager)

# Test in CLI
lobster chat
/status  # Check system status
/help    # List available commands
```

## 📊 Agent Performance

### Optimization Tips
- **Lazy Loading**: Load services only when needed
- **Caching**: Cache expensive computations appropriately
- **Memory Management**: Handle large datasets efficiently
- **Progress Tracking**: Provide user feedback for long operations

### Monitoring
```python
# Use structured logging
logger.info(f"Starting analysis with parameters: {params}")
logger.debug(f"Processing {adata.n_obs} observations")

# Track performance metrics
import time
start_time = time.time()
# ... analysis ...
duration = time.time() - start_time
logger.info(f"Analysis completed in {duration:.2f} seconds")
```

## 🎯 Complete Example

See the complete agent implementation template in `lobster/agents/template_agent.py` for a full working example that demonstrates all the patterns and best practices described in this guide.

This guide provides everything needed to create sophisticated, domain-specific agents that integrate seamlessly with the Lobster AI platform while maintaining scientific rigor and user-friendly interactions.