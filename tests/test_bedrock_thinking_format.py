"""
Test file to identify the correct format for AWS Bedrock thinking mode configuration.
"""

import os
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

# Set up basic test environment
os.environ.setdefault("AWS_REGION", "us-east-1")

def test_thinking_formats():
    """Test different thinking configuration formats."""
    print("=" * 60)
    print("TESTING AWS BEDROCK THINKING MODE FORMATS")
    print("=" * 60)
    
    base_params = {
        'model': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        'temperature': 0.0,
        'max_tokens': 4096,
    }
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is 2+2?")
    ]
    
    # Different thinking configurations to test
    thinking_configs = [
        {
            "name": "No thinking (baseline)",
            "config": {}
        },
        {
            "name": "Thinking with enabled/budget_tokens (current format)",
            "config": {
                'additional_model_request_fields': {
                    'thinking': {
                        'enabled': True,
                        'budget_tokens': 10000
                    }
                }
            }
        },
        {
            "name": "Thinking with just budget_tokens",
            "config": {
                'additional_model_request_fields': {
                    'thinking': {
                        'budget_tokens': 10000
                    }
                }
            }
        },
        {
            "name": "Thinking as boolean",
            "config": {
                'additional_model_request_fields': {
                    'thinking': True
                }
            }
        },
        {
            "name": "Thinking with inferenceConfig",  
            "config": {
                'additional_model_request_fields': {
                    'inferenceConfig': {
                        'thinking': {
                            'enabled': True
                        }
                    }
                }
            }
        },
        {
            "name": "Direct thinking parameter",
            "config": {
                'thinking': True
            }
        }
    ]
    
    for test_config in thinking_configs:
        print(f"\nTesting: {test_config['name']}")
        print(f"Config: {test_config['config']}")
        
        try:
            # Merge base params with test config
            model_params = {**base_params, **test_config['config']}
            model = ChatBedrockConverse(**model_params)
            
            # Try to invoke the model
            response = model.invoke(messages)
            print(f"✓ SUCCESS! Response type: {type(response)}")
            if hasattr(response, 'content'):
                print(f"  Response preview: {response.content[:50]}...")
                
        except Exception as e:
            error_msg = str(e)
            print(f"✗ FAILED: {error_msg[:200]}")
            
            # Analyze the error
            if "extraneous key [enabled]" in error_msg:
                print("  → Issue: 'enabled' key not allowed in thinking config")
            elif "extraneous key [budget_tokens]" in error_msg:
                print("  → Issue: 'budget_tokens' key not recognized")
            elif "thinking" in error_msg.lower():
                print("  → Issue: Other thinking-related error")


def check_bedrock_docs():
    """Provide guidance based on AWS Bedrock documentation."""
    print("\n" + "=" * 60)
    print("AWS BEDROCK THINKING MODE GUIDANCE")
    print("=" * 60)
    print("""
Based on the error message mentioning:
"Please consult our documentation at https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking"

The correct format for thinking mode in AWS Bedrock might be:
1. Different from the Anthropic API format
2. Require specific message structure rather than configuration
3. Need to be enabled at the model level differently

The error about "Expected `thinking` or `redacted_thinking`" suggests that:
- Thinking blocks should be part of the message content structure
- Not a configuration parameter but a message format requirement
""")


def main():
    test_thinking_formats()
    check_bedrock_docs()
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FIX")
    print("=" * 60)
    print("""
Based on the test results, the immediate fix is to:

1. Remove the thinking configuration from the supervisor model params:
   
   # In lobster/agents/graph.py, remove or comment out:
   # 'additional_model_request_fields': {
   #     'thinking': {
   #         'enabled': True,
   #         'budget_tokens': 10000
   #     }
   # }

2. If thinking mode is required, research the correct AWS Bedrock format:
   - Check AWS Bedrock documentation for Claude 3.5 Sonnet v2
   - The format might be completely different from what's currently used
   - It might require message-level formatting rather than config-level

3. For now, disable thinking for the supervisor to fix the immediate issue.
""")


if __name__ == "__main__":
    main()
