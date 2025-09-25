"""
Clean Agent Client Interface for LangGraph Multi-Agent System.
Provides a simple, extensible interface for both CLI and future UI implementations.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import json

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langfuse.langchain import CallbackHandler as LangfuseCallback

from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.base_client import BaseClient
from lobster.agents.graph import create_bioinformatics_graph

# Configure logging
logger = logging.getLogger(__name__)


class AgentClient(BaseClient):
    def __init__(
        self,
        data_manager: Optional[DataManagerV2] = None,
        session_id: str = None,
        enable_reasoning: bool = True,
        enable_langfuse: bool = False,
        workspace_path: Optional[Path] = None,
        custom_callbacks: Optional[List] = None,
        manual_model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent client with DataManagerV2.
        
        Args:
            data_manager: DataManagerV2 instance (creates new if None)
            session_id: Unique session identifier
            enable_reasoning: Show agent reasoning/thinking process
            enable_langfuse: Enable Langfuse debugging callback
            workspace_path: Path to workspace for file operations
            custom_callbacks: Additional callback handlers
            manual_model_params: Manual model parameter overrides
        """
        # Set up session
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.enable_reasoning = enable_reasoning
        
        # Set up workspace
        self.workspace_path = workspace_path or Path.cwd()
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize DataManagerV2
        if data_manager is None:
            from rich.console import Console
            console = Console() if custom_callbacks else None
            self.data_manager = DataManagerV2(
                workspace_path=self.workspace_path,
                console=console
            )
            logger.info("Initialized with DataManagerV2 (modular multi-omics)")
        else:
            self.data_manager = data_manager
        
        # Set up callbacks
        self.callbacks = []
        if enable_langfuse and os.getenv("LANGFUSE_PUBLIC_KEY"):
            self.callbacks.append(LangfuseCallback())
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)
        
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        # Initialize graph - pass all callbacks
        self.graph = create_bioinformatics_graph(
            data_manager=self.data_manager,
            checkpointer=self.checkpointer,
            store=self.store,
            callback_handler=self.callbacks,  # Pass the list of callbacks
            manual_model_params=manual_model_params  # Placeholder for future manual model params
        )        
        
        # Conversation state
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "session_id": self.session_id,
            "workspace": str(self.workspace_path)
        }
    
    def query(self, user_input: str, stream: bool = False) -> Dict[str, Any]:
        """
        Process a user query through the agent system.
        
        Args:
            user_input: User's input text
            stream: Whether to stream the response
            
        Returns:
            Dictionary with response and metadata
        """
        # Add user message
        self.messages.append(HumanMessage(content=user_input))
        
        # Prepare graph input
        graph_input = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        config = {
            "configurable": {"thread_id": self.session_id},
            "callbacks": self.callbacks,
            "recursion_limit": 100  # Prevent hitting default limit of 25
        }
        
        if stream:
            return self._stream_query(graph_input, config)
        else:
            return self._run_query(graph_input, config)
    
    def _run_query(self, graph_input: Dict, config: Dict) -> Dict[str, Any]:
        """Run a query and return the complete response."""
        try:
            # Track execution
            start_time = datetime.now()
            events = []
            last_agent = None
            
            # Execute graph
            for event in self.graph.stream(
                input=graph_input, 
                config=config,
                stream_mode='updates'
                ):
                events.append(event)
                
                # Track which agent is responding
                if event:
                    for node_name in event.keys():
                        if node_name and node_name != '__end__':
                            last_agent = node_name
            
            # Extract final response from the last event
            final_response = self._extract_response(events)
            
            # Update messages with the final response (not the raw events)
            if final_response:
                self.messages.append(AIMessage(content=final_response))
            
            return {
                "success": True,
                "response": final_response,
                "duration": (datetime.now() - start_time).total_seconds(),
                "events_count": len(events),
                "session_id": self.session_id,
                "has_data": self.data_manager.has_data(),
                "plots": self.data_manager.get_latest_plots(5) if self.data_manager.has_data() else [],
                "last_agent": last_agent  # Include which agent provided the response
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error: {str(e)}",
                "session_id": self.session_id
            }
    
    def _stream_query(self, graph_input: Dict, config: Dict) -> Generator[Dict[str, Any], None, None]:
        """Stream query execution with intermediate results."""
        try:
            start_time = datetime.now()
            
            for event in self.graph.stream(graph_input, config):
                # Process each event
                for node_name, node_output in event.items():
                    # Extract meaningful content
                    content = self._extract_event_content(node_output)
                    
                    if content:
                        yield {
                            "type": "stream",
                            "node": node_name,
                            "content": content,
                            "timestamp": datetime.now().isoformat()
                        }
            
            # Final response
            yield {
                "type": "complete",
                "duration": (datetime.now() - start_time).total_seconds(),
                "session_id": self.session_id
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "session_id": self.session_id
            }
    
    def _extract_response(self, events: List[Dict]) -> str:
        """Extract the final response from events, expecting supervisor responses."""
        if not events:
            return "No response generated."
        
        # Process events in reverse chronological order to find the last supervisor response
        for event in reversed(events):
            # Check for supervisor key first
            if 'supervisor' not in event:
                # Log any unexpected keys
                unexpected_keys = [key for key in event.keys() if key != 'supervisor']
                if unexpected_keys:
                    logger.warning(f"Unexpected event keys found (expected 'supervisor'): {unexpected_keys}")
                continue
            
            supervisor_data = event['supervisor']
            if not isinstance(supervisor_data, dict) or 'messages' not in supervisor_data:
                continue
                
            messages = supervisor_data['messages']
            if not isinstance(messages, list):
                continue
            
            # Find the last AIMessage in the supervisor's messages
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    content = self._extract_content_from_message(msg.content)
                    if content:
                        return content
        
        return "No response generated."
    
    def _extract_content_from_message(self, content) -> str:
        """Extract text content from a message, handling both string and list formats."""
        # Handle backward compatibility - if content is still a string
        if isinstance(content, str):
            return content.strip()
        
        # Handle new list format with content blocks
        if isinstance(content, list):
            text_parts = []
            reasoning_parts = []
            
            for block in content:
                if isinstance(block, dict):
                    # Extract text content
                    if block.get('type') == 'text' and 'text' in block:
                        text_parts.append(block['text'])
                    
                    # Extract reasoning content if enabled
                    elif block.get('type') == 'reasoning_content' and self.enable_reasoning:
                        if 'reasoning_content' in block and isinstance(block['reasoning_content'], dict):
                            reasoning_text = block['reasoning_content'].get('text', '')
                            if reasoning_text:
                                reasoning_parts.append(f"[Thinking: {reasoning_text}]")
            
            # Combine parts - show reasoning first if enabled, then the main text
            result_parts = []
            if reasoning_parts and self.enable_reasoning:
                result_parts.extend(reasoning_parts)
            if text_parts:
                result_parts.extend(text_parts)
            
            if result_parts:
                return '\n\n'.join(result_parts).strip()
        
        # Fallback for any other format
        return str(content).strip() if content else ""
    
    def _extract_event_content(self, node_output: Dict) -> Optional[str]:
        """Extract displayable content from a node output."""
        if not isinstance(node_output, dict):
            return None
        
        # Check for messages - only return content from AI messages
        if "messages" in node_output and node_output["messages"]:
            # Look for the last AI message in this event
            for msg in reversed(node_output["messages"]):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    return self._extract_content_from_message(msg.content)
        
        # Check for other relevant fields
        for key in ["analysis_results", "next", "data_context"]:
            if key in node_output and node_output[key]:
                return f"{key}: {node_output[key]}"
        
        return None
    
    # Enhanced file operations
    def detect_file_type(self, file_path: Path) -> Dict[str, Any]:
        """
        Detect file type with comprehensive format identification.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file type information
        """
        import mimetypes
        
        # File extension mapping for bioinformatics and common formats
        extension_map = {
            # Bioinformatics data formats
            '.h5ad': {'category': 'bioinformatics', 'type': 'single_cell_data', 'description': 'Single-cell RNA-seq data (H5AD format)', 'binary': True},
            '.h5mu': {'category': 'bioinformatics', 'type': 'multimodal_data', 'description': 'Multi-modal omics data (H5MU format)', 'binary': True},
            '.loom': {'category': 'bioinformatics', 'type': 'genomics_data', 'description': 'Genomics data (Loom format)', 'binary': True},
            '.h5': {'category': 'bioinformatics', 'type': 'hdf5_data', 'description': 'HDF5 data file', 'binary': True},
            '.mtx': {'category': 'bioinformatics', 'type': 'matrix_data', 'description': 'Matrix Market sparse matrix', 'binary': False},
            '.mex': {'category': 'bioinformatics', 'type': 'matrix_data', 'description': 'Matrix Exchange format', 'binary': False},
            
            # Tabular data formats
            '.csv': {'category': 'tabular', 'type': 'delimited_data', 'description': 'Comma-separated values', 'binary': False},
            '.tsv': {'category': 'tabular', 'type': 'delimited_data', 'description': 'Tab-separated values', 'binary': False},
            '.txt': {'category': 'tabular', 'type': 'delimited_data', 'description': 'Plain text data', 'binary': False},
            '.xlsx': {'category': 'tabular', 'type': 'spreadsheet_data', 'description': 'Excel spreadsheet', 'binary': True},
            '.xls': {'category': 'tabular', 'type': 'spreadsheet_data', 'description': 'Excel spreadsheet (legacy)', 'binary': True},
            
            # Configuration and metadata
            '.json': {'category': 'metadata', 'type': 'structured_data', 'description': 'JSON metadata', 'binary': False},
            '.yaml': {'category': 'metadata', 'type': 'structured_data', 'description': 'YAML configuration', 'binary': False},
            '.yml': {'category': 'metadata', 'type': 'structured_data', 'description': 'YAML configuration', 'binary': False},
            '.xml': {'category': 'metadata', 'type': 'structured_data', 'description': 'XML data', 'binary': False},
            
            # Code and scripts  
            '.py': {'category': 'code', 'type': 'python_script', 'description': 'Python script', 'binary': False},
            '.r': {'category': 'code', 'type': 'r_script', 'description': 'R script', 'binary': False},
            '.sh': {'category': 'code', 'type': 'shell_script', 'description': 'Shell script', 'binary': False},
            '.bash': {'category': 'code', 'type': 'shell_script', 'description': 'Bash script', 'binary': False},
            
            # Documentation
            '.md': {'category': 'documentation', 'type': 'markdown', 'description': 'Markdown document', 'binary': False},
            '.rst': {'category': 'documentation', 'type': 'restructured_text', 'description': 'reStructuredText document', 'binary': False},
            
            # Archives
            '.gz': {'category': 'archive', 'type': 'compressed', 'description': 'Gzip compressed file', 'binary': True},
            '.zip': {'category': 'archive', 'type': 'compressed', 'description': 'ZIP archive', 'binary': True},
            '.tar': {'category': 'archive', 'type': 'compressed', 'description': 'TAR archive', 'binary': True},
        }
        
        ext = file_path.suffix.lower()
        
        # Handle compound extensions like .csv.gz
        if file_path.name.endswith('.gz'):
            # Check the extension before .gz
            name_without_gz = file_path.name[:-3]  # Remove .gz
            inner_ext = Path(name_without_gz).suffix.lower()
            if inner_ext in extension_map:
                info = extension_map[inner_ext].copy()
                info['compressed'] = True
                info['description'] += ' (gzip compressed)'
                return info
            ext = '.gz'
        
        # Direct extension match
        if ext in extension_map:
            return extension_map[ext].copy()
        
        # Fallback to MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('text/'):
                return {
                    'category': 'text',
                    'type': 'plain_text',
                    'description': f'Text file ({mime_type})',
                    'binary': False
                }
            elif mime_type.startswith('image/'):
                return {
                    'category': 'image', 
                    'type': 'image_file',
                    'description': f'Image file ({mime_type})',
                    'binary': True
                }
        
        # Unknown file type
        return {
            'category': 'unknown',
            'type': 'unknown',
            'description': f'Unknown file type ({ext or "no extension"})',
            'binary': True  # Assume binary for safety
        }
    
    def locate_file(self, filename: str) -> Dict[str, Any]:
        """
        Locate file with comprehensive search and validation.
        
        Args:
            filename: Filename or path to search for
            
        Returns:
            Dictionary with file location and metadata
        """
        file_path = Path(filename)
        
        # If it's an absolute path, check directly
        if file_path.is_absolute():
            if file_path.exists():
                if file_path.is_file():
                    file_info = self.detect_file_type(file_path)
                    return {
                        'found': True,
                        'path': file_path,
                        'relative_to_workspace': file_path.relative_to(self.workspace_path) if file_path.is_relative_to(self.workspace_path) else None,
                        'size_bytes': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                        'readable': os.access(file_path, os.R_OK),
                        **file_info
                    }
                else:
                    return {'found': False, 'error': f"Path exists but is not a file: {file_path}"}
            else:
                return {'found': False, 'error': f"File not found: {file_path}"}
        
        # For relative paths, search in workspace directories
        search_paths = [
            self.workspace_path / filename,
            self.workspace_path / "data" / filename,
            self.data_manager.data_dir / filename,
            self.data_manager.workspace_path / "plots" / filename,
            self.data_manager.exports_dir / filename,
            self.data_manager.cache_dir / filename,
            Path.cwd() / filename  # Current working directory
        ]
        
        # Remove duplicates while preserving order
        unique_search_paths = []
        seen = set()
        for path in search_paths:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_search_paths.append(path)
        
        for search_path in unique_search_paths:
            if search_path.exists() and search_path.is_file():
                try:
                    file_info = self.detect_file_type(search_path)
                    return {
                        'found': True,
                        'path': search_path.resolve(),
                        'relative_to_workspace': search_path.relative_to(self.workspace_path) if search_path.is_relative_to(self.workspace_path) else None,
                        'size_bytes': search_path.stat().st_size,
                        'modified': datetime.fromtimestamp(search_path.stat().st_mtime),
                        'readable': os.access(search_path, os.R_OK),
                        'searched_paths': [str(p) for p in unique_search_paths],
                        **file_info
                    }
                except (OSError, PermissionError):
                    continue
        
        return {
            'found': False,
            'error': f"File '{filename}' not found in any search location",
            'searched_paths': [str(p) for p in unique_search_paths]
        }
    
    def load_data_file(self, filename: str) -> Dict[str, Any]:
        """
        Smart data loading into DataManagerV2 based on file type.
        
        Args:
            filename: File to load
            
        Returns:
            Dictionary with loading results and metadata
        """
        # First, locate the file
        file_info = self.locate_file(filename)
        
        if not file_info['found']:
            return {
                'success': False,
                'error': file_info['error'],
                'searched_paths': file_info.get('searched_paths', [])
            }
        
        file_path = file_info['path']
        file_type = file_info['type']
        
        # Check if file is readable
        if not file_info.get('readable', True):
            return {
                'success': False,
                'error': f"Permission denied: Cannot read {file_path}"
            }
        
        # Generate modality name from filename
        modality_name = file_path.stem  # Filename without extension
        
        try:
            # Check if this modality already exists
            if modality_name in self.data_manager.list_modalities():
                # Generate unique name
                counter = 1
                original_name = modality_name
                while modality_name in self.data_manager.list_modalities():
                    modality_name = f"{original_name}_{counter}"
                    counter += 1
            
            # Load based on file type
            if file_type in ['single_cell_data', 'multimodal_data', 'genomics_data', 'hdf5_data']:
                # Use DataManager's load_modality method for bioinformatics formats
                # Try to auto-detect if it's single-cell or bulk based on file
                adapter_name = "transcriptomics_single_cell"  # Default assumption
                
                adata = self.data_manager.load_modality(
                    name=modality_name,
                    source=str(file_path),
                    adapter=adapter_name,
                    validate=False  # Skip validation for now to be more permissive
                )
                
                return {
                    'success': True,
                    'modality_name': modality_name,
                    'file_path': str(file_path),
                    'file_type': file_info['description'],
                    'data_shape': (adata.n_obs, adata.n_vars),
                    'size_bytes': file_info['size_bytes'],
                    'message': f"Data loaded successfully as modality '{modality_name}'"
                }
                
            elif file_type in ['delimited_data', 'spreadsheet_data']:
                # For tabular data, load as DataFrame and convert using transcriptomics adapter
                try:
                    if file_path.suffix.lower() in ['.csv']:
                        import pandas as pd
                        df = pd.read_csv(file_path)
                    elif file_path.suffix.lower() in ['.tsv', '.txt']:
                        import pandas as pd  
                        df = pd.read_csv(file_path, sep='\t')
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        import pandas as pd
                        df = pd.read_excel(file_path)
                    else:
                        return {
                            'success': False,
                            'error': f"Unsupported tabular format: {file_path.suffix}"
                        }
                    
                    # Use transcriptomics adapter for tabular data (genes x samples or samples x genes)
                    adapter_name = "transcriptomics_bulk"  # Bulk is more generic for tabular data
                    
                    adata = self.data_manager.load_modality(
                        name=modality_name,
                        source=df,
                        adapter=adapter_name,
                        validate=False  # Skip validation to be more permissive
                    )
                    
                    return {
                        'success': True,
                        'modality_name': modality_name,
                        'file_path': str(file_path),
                        'file_type': file_info['description'],
                        'data_shape': (adata.n_obs, adata.n_vars),
                        'size_bytes': file_info['size_bytes'],
                        'message': f"Tabular data loaded successfully as modality '{modality_name}'"
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'error': f"Failed to load tabular data: {str(e)}"
                    }
            
            else:
                return {
                    'success': False,
                    'error': f"File type '{file_info['description']}' is not a supported data format for loading into workspace",
                    'suggestion': "Use '/read' for text files or ensure file is in a supported bioinformatics format (.h5ad, .csv, .tsv, etc.)"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to load file: {str(e)}",
                'file_path': str(file_path),
                'file_type': file_info['description']
            }

    # Workspace operations
    def list_workspace_files(self, pattern: str = "*") -> List[Dict[str, Any]]:
        """List files in the workspace."""
        files = []
        for path in self.workspace_path.glob(pattern):
            if path.is_file():
                files.append({
                    "name": path.name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                })
        return files
    
    def read_file(self, filename: str) -> Optional[str]:
        """
        Read a file from the workspace or absolute path.
        
        Args:
            filename: Either a relative filename (searched in workspace) or absolute path
            
        Returns:
            File content as string, or None if not found
        """
        file_path = Path(filename)
        
        # If it's an absolute path, try to read directly
        if file_path.is_absolute():
            if file_path.exists() and file_path.is_file():
                try:
                    return file_path.read_text()
                except Exception as e:
                    return f"Error reading file {file_path}: {e}"
            else:
                return f"File not found: {file_path}"
        
        # For relative paths, search in workspace and data directories
        search_paths = [
            self.workspace_path / filename,
            self.data_manager.data_dir / filename,
            self.data_manager.workspace_path / "plots" / filename,
            self.data_manager.exports_dir / filename,
            self.data_manager.cache_dir / filename
        ]
        
        for search_path in search_paths:
            if search_path.exists() and search_path.is_file():
                try:
                    return search_path.read_text()
                except Exception as e:
                    return f"Error reading file: {e}"
        
        return f"File not found in workspace: {filename}"
    
    def write_file(self, filename: str, content: str) -> bool:
        """Write a file to the workspace."""
        try:
            file_path = self.workspace_path / filename
            file_path.write_text(content)
            return True
        except Exception:
            return False
    
    # State management
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get formatted conversation history."""
        history = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "system"
            
            history.append({
                "role": role,
                "content": msg.content if hasattr(msg, 'content') else str(msg)
            })
        return history
    
    def get_status(self) -> Dict[str, Any]:
        """Get current client status."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "has_data": self.data_manager.has_data(),
            "data_summary": self.data_manager.get_data_summary() if self.data_manager.has_data() else None,
            "workspace": str(self.workspace_path),
            "reasoning_enabled": self.enable_reasoning,
            "callbacks_count": len(self.callbacks)
        }
    
    def reset(self):
        """Reset the conversation state."""
        self.messages = []
        self.metadata["reset_at"] = datetime.now().isoformat()
    
    def export_session(self, export_path: Optional[Path] = None) -> Path:
        """Export the current session data."""
        if self.data_manager.has_data():
            export_path = self.data_manager.create_data_package(
                output_dir=str(self.data_manager.exports_dir)
            )
            return Path(export_path)
        
        export_path = export_path or self.workspace_path / f"session_{self.session_id}.json"
        
        session_data = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "conversation": self.get_conversation_history(),
            "status": self.get_status(),
            "workspace_status": self.data_manager.get_workspace_status(),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        return export_path
