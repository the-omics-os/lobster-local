"""
Test file to identify the AWS Bedrock Converse API thinking mode bug.

The error indicates that when 'thinking' is enabled, assistant messages must start 
with a thinking block. This test will help identify where in the chain this is failing.
"""

import os
import json
from unittest.mock import Mock, patch, MagicMock
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

# Set up basic test environment
os.environ.setdefault("AWS_REGION", "us-east-1")

def test_basic_bedrock_with_thinking():
    """Test basic Bedrock model with thinking enabled."""
    print("\n=== Test 1: Basic Bedrock with Thinking ===")
    
    # Configuration with thinking enabled
    model_params = {
        'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'temperature': 0.0,
        'max_tokens': 4096,
        'additional_model_request_fields': {
            'thinking': {
                'enabled': True,
                'budget_tokens': 10000
            }
        }
    }
    
    try:
        model = ChatBedrockConverse(**model_params)
        
        # Test with simple message
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is 2+2?")
        ]
        
        print(f"Input messages: {messages}")
        
        # This should work if the model handles thinking properly
        response = model.invoke(messages)
        print(f"Response type: {type(response)}")
        print(f"Response content: {response.content[:100] if hasattr(response, 'content') else response}")
        
    except Exception as e:
        print(f"ERROR in basic test: {e}")
        return False
    
    return True


def test_supervisor_message_flow():
    """Test the message flow through the supervisor setup."""
    print("\n=== Test 2: Supervisor Message Flow ===")
    
    from lobster.agents.langgraph_supervisor import create_supervisor
    from lobster.agents.state import OverallState
    from lobster.agents.supervisor import create_supervisor_prompt
    from lobster.tools.handoff_tool import create_custom_handoff_tool
    
    # Mock data manager
    mock_data_manager = Mock()
    mock_data_manager.get_state.return_value = {}
    
    # Create mock agent
    mock_agent = Mock()
    mock_agent.name = "test_agent"
    mock_agent.invoke = Mock(return_value={"messages": [AIMessage(content="Test response")]})
    
    # Model params with thinking
    model_params = {
        'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'temperature': 0.0,
        'max_tokens': 4096,
        'additional_model_request_fields': {
            'thinking': {
                'enabled': True,
                'budget_tokens': 10000
            }
        }
    }
    
    try:
        supervisor_model = ChatBedrockConverse(**model_params)
        
        # Create handoff tool
        handoff_tool = create_custom_handoff_tool(
            agent_name="test_agent",
            name="transfer_to_test_agent",
            description="Transfer to test agent"
        )
        
        # Create supervisor
        system_prompt = "You are a supervisor coordinating agents."
        
        workflow = create_supervisor(
            agents=[mock_agent],
            model=supervisor_model,
            prompt=system_prompt,
            supervisor_name="supervisor",
            state_schema=OverallState,
            add_handoff_messages=True,
            include_agent_name='inline',
            output_mode="last_message",
            tools=[handoff_tool]
        )
        
        # Compile the graph
        graph = workflow.compile()
        
        # Test invocation
        test_input = {
            "messages": [HumanMessage(content="Process this request")],
            "active_agent": "supervisor"
        }
        
        print(f"Input state: {test_input}")
        
        # This is where the error likely occurs
        result = graph.invoke(test_input, {"recursion_limit": 5})
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"ERROR in supervisor flow: {e}")
        # Parse the error to identify the problematic message structure
        if "ValidationException" in str(e):
            print("\n*** This is the thinking mode validation error! ***")
            print("The issue occurs when the supervisor tries to respond.")
            print("The model is expecting thinking blocks but getting regular text.")
        return False
    
    return True


def test_message_format_with_thinking():
    """Test different message formats to understand what Bedrock expects."""
    print("\n=== Test 3: Message Format Investigation ===")
    
    model_params = {
        'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'temperature': 0.0,
        'max_tokens': 4096,
        'additional_model_request_fields': {
            'thinking': {
                'enabled': True,
                'budget_tokens': 10000
            }
        }
    }
    
    model = ChatBedrockConverse(**model_params)
    
    # Test different message combinations
    test_cases = [
        {
            "name": "Simple conversation",
            "messages": [
                HumanMessage(content="Hello")
            ]
        },
        {
            "name": "With system message",
            "messages": [
                SystemMessage(content="You are an assistant"),
                HumanMessage(content="Hello")
            ]
        },
        {
            "name": "With AI response",
            "messages": [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!"),
                HumanMessage(content="How are you?")
            ]
        },
        {
            "name": "With tool use",
            "messages": [
                HumanMessage(content="Transfer to agent"),
                AIMessage(content="", tool_calls=[{"name": "transfer", "args": {}, "id": "1"}]),
                ToolMessage(content="Transferred", tool_call_id="1"),
            ]
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            response = model.invoke(test_case['messages'])
            print(f"  ✓ Success: {type(response)}")
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:200]}")
            if "thinking" in str(e).lower():
                print("  → Issue is related to thinking mode formatting")


def test_supervisor_without_thinking():
    """Test if the supervisor works without thinking mode."""
    print("\n=== Test 4: Supervisor Without Thinking ===")
    
    from lobster.agents.langgraph_supervisor import create_supervisor
    from lobster.agents.state import OverallState
    
    # Mock agent
    mock_agent = Mock()
    mock_agent.name = "test_agent"
    mock_agent.invoke = Mock(return_value={"messages": [AIMessage(content="Test response")]})
    
    # Model params WITHOUT thinking
    model_params = {
        'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'temperature': 0.0,
        'max_tokens': 4096,
        # No thinking configuration
    }
    
    try:
        supervisor_model = ChatBedrockConverse(**model_params)
        
        workflow = create_supervisor(
            agents=[mock_agent],
            model=supervisor_model,
            prompt="You are a supervisor.",
            supervisor_name="supervisor",
            state_schema=OverallState,
            output_mode="last_message",
        )
        
        graph = workflow.compile()
        
        test_input = {
            "messages": [HumanMessage(content="Process this")],
            "active_agent": "supervisor"
        }
        
        result = graph.invoke(test_input, {"recursion_limit": 5})
        print(f"✓ Works without thinking! Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        return True
        
    except Exception as e:
        print(f"✗ Failed even without thinking: {e}")
        return False


def diagnose_thinking_issue():
    """Main diagnostic function to identify the thinking mode bug."""
    print("=" * 60)
    print("BEDROCK THINKING MODE BUG DIAGNOSIS")
    print("=" * 60)
    
    results = []
    
    # Run tests
    # Note: These tests will likely fail without actual AWS credentials
    # But the error messages will help identify the issue
    
    print("\nNote: These tests require AWS credentials to fully execute.")
    print("Even mock tests can help identify the structural issue.\n")
    
    # Test 1: Basic Bedrock with thinking
    try:
        results.append(("Basic Bedrock with Thinking", test_basic_bedrock_with_thinking()))
    except Exception as e:
        results.append(("Basic Bedrock with Thinking", False))
        print(f"Test skipped due to: {e}")
    
    # Test 2: Supervisor message flow
    try:
        results.append(("Supervisor Message Flow", test_supervisor_message_flow()))
    except Exception as e:
        results.append(("Supervisor Message Flow", False))
        print(f"Test skipped due to: {e}")
    
    # Test 3: Message format investigation
    try:
        results.append(("Message Format Investigation", test_message_format_with_thinking()))
    except Exception as e:
        results.append(("Message Format Investigation", False))
        print(f"Test skipped due to: {e}")
    
    # Test 4: Supervisor without thinking
    try:
        results.append(("Supervisor Without Thinking", test_supervisor_without_thinking()))
    except Exception as e:
        results.append(("Supervisor Without Thinking", False))
        print(f"Test skipped due to: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print("\n" + "=" * 60)
    print("LIKELY ISSUE:")
    print("=" * 60)
    print("""
The error occurs because when AWS Bedrock's 'thinking' mode is enabled,
it expects assistant messages to have a specific format with thinking blocks.

The langgraph_supervisor creates messages that don't conform to this format,
particularly when:
1. The supervisor responds after agent handoffs
2. Tool messages are being processed
3. The supervisor adds its own messages to the conversation

POTENTIAL FIXES:
1. Disable thinking mode for the supervisor (simplest fix)
2. Modify message formatting in langgraph_supervisor to support thinking blocks
3. Add a message transformer that formats messages correctly for thinking mode
4. Use a different output_mode that's compatible with thinking

RECOMMENDED IMMEDIATE FIX:
Remove the 'thinking' configuration from the supervisor's model_params:
    # Remove or comment out:
    # 'additional_model_request_fields': {
    #     'thinking': {
    #         'enabled': True,
    #         'budget_tokens': 10000
    #     }
    # }
""")


if __name__ == "__main__":
    # Check if we can import required modules
    try:
        from lobster.agents.langgraph_supervisor import create_supervisor
        from lobster.agents.state import OverallState
        from lobster.tools.handoff_tool import create_custom_handoff_tool
        print("✓ All required modules can be imported")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you're running this from the lobster project directory")
        exit(1)
    
    # Run diagnosis
    diagnose_thinking_issue()
