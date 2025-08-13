"""
Simple test for LangChain's ChatBedrockConverse.

This script attempts to use the ChatBedrockConverse class
with minimal configuration to avoid validation errors.
"""

from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

graph = create_react_agent(
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    tools=[check_weather],
    prompt="You are a helpful assistant",
)
inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)

def test_chat_bedrock_converse():
    """Test ChatBedrockConverse with minimal parameters."""
    print("Testing ChatBedrockConverse...")
    
    # Load environment variables (AWS credentials)
    load_dotenv()
    
    try:
        # Create the LLM with minimal parameters
        # model_id should be a valid AWS Bedrock model ID
        # For Claude: anthropic.claude-3-sonnet-20240229-v1:0
        llm = ChatBedrockConverse(
            model_id="eu.anthropic.claude-sonnet-4-20250514-v1:0",
            temperature=0.7
        )
        print("✓ Successfully created ChatBedrockConverse instance")

        create_react_agent(
                    model=self.llm,
                    tools=self.tools,
                    prompt=system_message,
                    name="bioinformatics_supervisor"
                )        
        
        # Try to generate a simple response
        print("Sending a test query...")
        response = llm.predict("Hello, who are you?")
        
        print("\nResponse from model:")
        print(response)
        print("\n✓ Test successful!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nPossible issues:")
        print("1. Check AWS credentials are correctly configured")
        print("2. Verify the model ID is correct and available in your AWS region")
        print("3. Check that you have proper permissions to access the model")

if __name__ == "__main__":
    test_chat_bedrock_converse()
