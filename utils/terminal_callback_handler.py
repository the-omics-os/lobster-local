"""
Simplified Terminal Callback Handler for Multi-Agent Reasoning Display.

Clean implementation that shows agent transitions and reasoning steps
in a clear, easy-to-follow format matching the new simplified architecture.
"""

import sys
from typing import Any, Dict, List
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from utils.logger import get_logger

logger = get_logger(__name__)


class TerminalCallbackHandler(BaseCallbackHandler):
    """
    Simplified callback handler for displaying agent thought processes.
    
    Shows:
    - Agent transitions (supervisor -> worker -> supervisor)
    - Agent reasoning and decisions
    - Tool usage and results
    - Clear, clean output without excessive complexity
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the terminal callback handler.
        
        Args:
            verbose: Whether to show detailed reasoning or just key events
        """
        self.verbose = verbose
        self.current_agent = None
        self.start_time = None
        
        # Simple color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'blue': '\033[34m',     # Agent names
            'green': '\033[32m',    # Success/completion
            'yellow': '\033[33m',   # Tools
            'red': '\033[31m',      # Errors
            'gray': '\033[90m',     # Details
            'cyan': '\033[36m',     # System messages
        }
    
    def _print(self, message: str, color: str = None):
        """Print a message with optional color."""
        if color and color in self.colors:
            message = f"{self.colors[color]}{message}{self.colors['reset']}"
        print(message, file=sys.stderr)
        sys.stderr.flush()
    
    def _print_agent_header(self, agent_name: str):
        """Print a clear header when switching to a new agent."""
        if agent_name != self.current_agent:
            self.current_agent = agent_name
            
            # Clean agent name for display
            display_name = agent_name.replace('_', ' ').title()
            
            self._print("\n" + "=" * 60)
            self._print(f"🤖 {display_name} Agent", color='blue')
            self._print("=" * 60)
    
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        """Called when a chat model starts."""
        if not self.verbose:
            return
            
        self.start_time = datetime.now()
        
        # Handle None serialized parameter
        if not serialized or not isinstance(serialized, dict):
            return
            
        # Check if this is from a specific agent
        agent_name = serialized.get('name', 'unknown')
        if agent_name and agent_name != 'unknown':
            self._print_agent_header(agent_name)
            self._print("💭 Thinking...", color='cyan')

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Called when an LLM starts - shows agent reasoning."""
        if not self.verbose:
            return
            
        self.start_time = datetime.now()
        
        # Handle None serialized parameter
        if not serialized or not isinstance(serialized, dict):
            return
            
        # Check if this is from a specific agent
        agent_name = serialized.get('name', 'unknown')
        if agent_name and agent_name != 'unknown':
            self._print_agent_header(agent_name)
            self._print("💭 Thinking...", color='cyan')
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when an LLM finishes - shows the response."""
        if not self.verbose or not response.generations:
            return
            
        # Calculate duration
        duration_str = ""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            duration_str = f" ({duration:.1f}s)"
        
        # Show the agent's response/decision
        for generation in response.generations[0]:
            if generation.text and generation.text.strip():
                self._print(f"💬 Response{duration_str}:", color='green')
                
                # Clean up the response for display
                response_text = generation.text.strip()
                
                # If it's a long response, show preview
                if len(response_text) > 300:
                    response_text = response_text[:300] + "..."
                
                # Print the response with proper formatting
                lines = response_text.split('\n')
                for line in lines:
                    if line.strip():
                        self._print(f"  {line.strip()}")
                
                self._print("")  # Add spacing
    
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts - shows tool usage."""
        # Handle None serialized parameter
        if not serialized or not isinstance(serialized, dict):
            return
            
        tool_name = serialized.get('name', 'unknown_tool')
        
        # Clean up tool name for display
        display_tool = tool_name.replace('_', ' ').title()
        
        self._print(f"🔧 Using Tool: {display_tool}", color='yellow')
        
        if self.verbose and input_str:
            # Show tool input if it's reasonable length
            input_display = str(input_str)  # Ensure it's a string
            if len(input_display) > 150:
                input_display = input_display[:150] + "..."
            self._print(f"   Input: {input_display}", color='gray')
    
    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes - shows tool result."""
        if self.verbose and output:
            # Show tool output
            output_display = str(output)
            if len(output_display) > 200:
                output_display = output_display[:200] + "..."
            
            self._print(f"   Result: {output_display}", color='gray')
        
        self._print(f"✅ Tool completed", color='green')
        self._print("")  # Add spacing
    
    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool errors."""
        self._print(f"❌ Tool error: {str(error)}", color='red')
        self._print("")  # Add spacing
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called when an LLM errors."""
        self._print(f"❌ Agent error: {str(error)}", color='red')
        self._print("")  # Add spacing
    
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Called when a chain starts - detect agent transitions."""
        # Handle None serialized parameter
        if not serialized or not isinstance(serialized, dict):
            return
            
        chain_name = serialized.get('name', '')
        if not chain_name:
            return
        
        # Detect agent handoffs by looking for agent names in the chain
        agent_names = ['supervisor', 'transcriptomics_expert', 'method_agent', 'clarify_with_user']
        
        for agent_name in agent_names:
            if agent_name in chain_name.lower():
                self._print_agent_header(agent_name)
                break
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain finishes."""
        # Check if this indicates completion
        if isinstance(outputs, dict) and outputs.get('analysis_complete'):
            self._print("\n🎉 Analysis Complete!", color='green')
            self._print("=" * 60)
    
    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        """Called when an agent takes an action."""
        if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
            # This is handled by on_tool_start, so we don't need to duplicate
            pass
    
    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Called when an agent finishes."""
        if hasattr(finish, 'return_values') and finish.return_values:
            self._print("✅ Agent task completed", color='green')
    
    # New methods specific to our multi-agent system
    
    def on_agent_handoff(self, from_agent: str, to_agent: str, task: str = None):
        """Show agent handoff clearly."""
        from_display = from_agent.replace('_', ' ').title()
        to_display = to_agent.replace('_', ' ').title()
        
        self._print(f"\n🔄 Handoff: {from_display} → {to_display}", color='cyan')
        
        if task:
            task_display = task
            if len(task_display) > 100:
                task_display = task_display[:100] + "..."
            self._print(f"   Task: {task_display}", color='gray')
        
        self._print("")
    
    def on_graph_step(self, node_name: str, step_info: str = None):
        """Show graph execution steps."""
        if node_name != self.current_agent:
            self._print_agent_header(node_name)
        
        if step_info:
            self._print(f"📍 {step_info}", color='cyan')
    
    def show_final_result(self, result: str):
        """Show the final result clearly."""
        self._print("\n" + "🎯 FINAL RESULT " + "=" * 45, color='green')
        self._print("")
        
        # Format the result nicely
        lines = result.split('\n')
        for line in lines:
            if line.strip():
                self._print(line.strip())
        
        self._print("\n" + "=" * 60)


# Compatibility class for existing code
class StreamlitCallbackHandler(TerminalCallbackHandler):
    """Streamlit-compatible version of the terminal callback handler."""
    
    def __init__(self, container=None, verbose: bool = True):
        """
        Initialize with optional Streamlit container.
        
        Args:
            container: Streamlit container for output (optional)
            verbose: Whether to show detailed output
        """
        super().__init__(verbose=verbose)
        self.container = container
        self.messages = []  # Store messages for Streamlit display
    
    def _print(self, message: str, color: str = None):
        """Override to store messages for Streamlit."""
        # Still print to terminal
        super()._print(message, color)
        
        # Store for Streamlit
        self.messages.append({
            'message': message,
            'color': color,
            'timestamp': datetime.now()
        })
        
        # Update Streamlit container if provided
        if self.container:
            try:
                # Simple text display for now
                self.container.text(message)
            except Exception:
                pass  # Fail silently if container issues
