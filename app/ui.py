"""
Streamlit UI module.

This module provides the Streamlit user interface for the application,
keeping UI code separate from core business logic.
"""

import streamlit as st
from typing import Any, Optional, List

from langchain.callbacks.base import BaseCallbackHandler
from utils.logger import get_logger
from core.data_manager import DataManager

import logging

logger = get_logger(__name__)
logger.setLevel(logging.CRITICAL)

class StreamingCallbackHandler(BaseCallbackHandler):
    """
    Minimal callback handler - only implements required methods.
    No thinking buffer is maintained.
    """
    
    def __init__(self):
        """Initialize the callback handler."""
        pass
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Handle LLM start event."""
        pass
    
    def on_llm_end(self, response, **kwargs):
        """Handle LLM end event."""
        pass
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Handle tool start event."""
        pass
    
    def on_tool_end(self, output, **kwargs):
        """Handle tool end event."""
        pass
    
    def on_agent_action(self, action, **kwargs):
        """Handle agent action event."""
        pass
    
    def on_agent_finish(self, finish, **kwargs):
        """Handle agent finish event."""
        pass
    
    def get_and_clear_thinking(self):
        """
        Empty placeholder function to maintain compatibility.
        
        Returns:
            str: Empty string
        """
        return ""

class StreamlitUI:
    """
    Streamlit UI manager.
    
    This class handles all Streamlit UI components and interactions,
    keeping the UI code separate from core application logic.
    """
    
    def __init__(self, title: str = "Genie 2.0", subtitle: str = "AI Bioinformatician"):
        """
        Initialize the Streamlit UI.
        
        Args:
            title: Application title
            subtitle: Application subtitle
        """
        self.title = title
        self.subtitle = subtitle
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=f"{self.title} - {self.subtitle}",
            page_icon="🧬",
            layout="wide"
        )
    
    def setup_css(self):
        """Add custom CSS to the Streamlit app (removed as per requirements)."""
        pass
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
        if 'callback_handler' not in st.session_state:
            st.session_state.callback_handler = StreamingCallbackHandler()
    
    def show_sidebar(self, data_manager: DataManager):
        """Show sidebar with data info and upload options."""
        st.sidebar.header("🧬 Genie 2.0")
        
        # Show file upload section
        self.show_file_upload_section()
        
        st.sidebar.markdown("---")
        
        # Existing data summary section
        if data_manager.has_data():
            st.sidebar.success("✅ Data Loaded")
            summary = data_manager.get_data_summary()
            
            # Create two columns for the data summary and download button
            col1, col2 = st.sidebar.columns([3, 1])
            
            with col1:
                with st.expander("📊 Data Summary", expanded=False):
                    st.write(f"**Shape:** {summary['shape']}")
                    st.write(f"**Memory:** {summary['memory_usage']}")
                    if summary.get('metadata_keys'):
                        st.write(f"**Metadata:** {', '.join(summary['metadata_keys'][:3])}")
            
            # Add download button
            with col2:
                if st.button("💾", help="Download all data, plots, and technical summary", key="download_button"):
                    try:
                        with st.spinner("Preparing download package..."):
                            # Create export directory if it doesn't exist
                            import os
                            os.makedirs("data/exports", exist_ok=True)
                            
                            # Create data package
                            zip_file = data_manager.create_data_package()
                            
                            # Provide download link
                            with open(zip_file, "rb") as fp:
                                btn = st.download_button(
                                    label="Download ZIP",
                                    data=fp,
                                    file_name=os.path.basename(zip_file),
                                    mime="application/zip",
                                    key="download_data_package"
                                )
                    except Exception as e:
                        st.error(f"Error creating download package: {str(e)}")
        else:
            st.sidebar.info("No data loaded")
            
        # Show analysis suggestions
        st.sidebar.markdown("---")
        st.sidebar.header("💡 Quick Start")
        
        if not data_manager.has_data():
            st.sidebar.markdown("""
            **Get Started:**
            1. Upload your data files above, or
            2. Ask me to download from GEO
            
            **Example queries:**
            - "Download GSE109564 from GEO"
            - "Upload and analyze my expression data"
            """)
        else:
            # Show context-aware suggestions
            data_type = data_manager.current_metadata.get('data_type', 'unknown')
            
            if data_type == 'single_cell':
                st.sidebar.markdown("""
                **Single-cell Analysis:**
                - "Assess data quality"
                - "Detect doublets"
                - "Cluster cells and create UMAP"
                - "Annotate cell types"
                """)
            elif data_type == 'bulk_rnaseq':
                st.sidebar.markdown("""
                **Bulk RNA-seq Analysis:**
                - "Run quality control"
                - "Perform differential expression"
                - "Run pathway enrichment"
                """)
            else:
                st.sidebar.markdown("""
                **Available Analyses:**
                - "Assess data quality"
                - "Cluster and visualize"
                - "Find marker genes"
                """)
    
    def show_main_ui(self):
        """Display the main UI elements."""
        st.title(f"🧬 {self.title}")
        st.markdown(f"Chat with your {self.subtitle}")
    
    def display_chat_history(self):
        """Display chat message history."""
        if 'messages' not in st.session_state:
            return
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "plots" in message:
                    for plot_item in message["plots"]:
                        # Handle both old format (just figure) and new format (dict with id and figure)
                        if isinstance(plot_item, dict) and "id" in plot_item and "figure" in plot_item:
                            plot_id = plot_item["id"]
                            plot = plot_item["figure"]
                        else:
                            # For backward compatibility with old format
                            plot = plot_item
                            plot_id = f"legacy_plot_{id(plot)}_{id(message)}"  # Make more unique by including message id
                            
                        st.plotly_chart(
                            plot, 
                            use_container_width=True, 
                            key=plot_id  # Use key parameter for unique identification, not id
                        )
    
    def get_chat_input(self) -> Optional[str]:
        """
        Get user input from chat interface.
        
        Returns:
            str or None: User input or None if no input
        """
        return st.chat_input("Ask me about bioinformatics analysis...")
    
    def start_thinking_refresh_thread(self):
        """Simplified version - no refresh thread needed."""
        pass
    
    def stop_thinking_refresh_thread(self):
        """Simplified version - no refresh thread to stop."""
        pass
    
    def add_user_message(self, message: str):
        """
        Add a user message to the chat history.
        
        Args:
            message: User message text
        """
        st.session_state.messages.append({"role": "user", "content": message})
        with st.chat_message("user"):
            st.markdown(message)
    
    def add_assistant_message(self, message: str, plots: List[Any] = None):
        """
        Add an assistant message to the chat history.
        
        Args:
            message: Assistant message text
            plots: Optional list of plotly figures or plot entries
        """
        # Process plots if provided
        plot_entries = []
        if plots:
            for plot_item in plots:
                # Check if it's already a plot entry dictionary or just a figure
                if isinstance(plot_item, dict) and "id" in plot_item and "figure" in plot_item:
                    # It's already a plot entry dictionary
                    plot_entries.append(plot_item)
                else:
                    # It's a raw figure, create a plot entry with a generated ID
                    plot_id = f"msg_plot_{id(plot_item)}"
                    plot_entries.append({
                        "id": plot_id,
                        "figure": plot_item,
                        "title": "Plot"
                    })
        
        # Add to session state
        if plot_entries:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": message,
                "plots": plot_entries
            })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": message
            })
        
        with st.chat_message("assistant"):
            st.markdown(message)
            if plot_entries:
                for plot_entry in plot_entries:
                    plot = plot_entry["figure"]
                    plot_id = plot_entry["id"]
                    
                    st.plotly_chart(
                        plot, 
                        use_container_width=True, 
                        key=plot_id
                    )
    
    def clear_thinking(self):
        """Empty placeholder method for compatibility."""
        pass

    def show_file_upload_section(self):
        """Show file upload section in sidebar."""
        st.sidebar.header("📁 Data Upload")
        
        upload_type = st.sidebar.selectbox(
            "Upload Type",
            ["Expression Matrix", "Sample Metadata", "FASTQ Files (Bulk RNA-seq)"]
        )
        
        if upload_type == "Expression Matrix":
            uploaded_file = st.sidebar.file_uploader(
                "Choose expression matrix file",
                type=['csv', 'tsv', 'txt', 'xlsx', 'h5', 'h5ad', 'mtx'],
                help="Upload your gene expression matrix (samples × genes or genes × samples)"
            )
            
            data_type = st.sidebar.selectbox(
                "Data Type",
                ["auto", "bulk_rnaseq", "single_cell"],
                help="Select data type or let the system auto-detect"
            )
            
            if uploaded_file and st.sidebar.button("Upload Expression Data"):
                with st.spinner("Processing uploaded file..."):
                    from services.file_upload_service import FileUploadService
                    upload_service = FileUploadService(st.session_state.data_manager)
                    result = upload_service.upload_expression_matrix(uploaded_file, data_type)
                    st.sidebar.success("Upload completed!")
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })
                    st.rerun()
        
        elif upload_type == "Sample Metadata":
            uploaded_file = st.sidebar.file_uploader(
                "Choose metadata file",
                type=['csv', 'tsv', 'txt', 'xlsx'],
                help="Upload sample metadata with experimental conditions"
            )
            
            if uploaded_file and st.sidebar.button("Upload Metadata"):
                with st.spinner("Processing metadata..."):
                    from services.file_upload_service import FileUploadService
                    upload_service = FileUploadService(st.session_state.data_manager)
                    result = upload_service.upload_sample_metadata(uploaded_file)
                    st.sidebar.success("Metadata uploaded!")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })
                    st.rerun()
        
        elif upload_type == "FASTQ Files (Bulk RNA-seq)":
            uploaded_files = st.sidebar.file_uploader(
                "Choose FASTQ files",
                type=['fastq', 'fq'],
                accept_multiple_files=True,
                help="Upload FASTQ files for bulk RNA-seq analysis"
            )
            
            if uploaded_files and st.sidebar.button("Upload FASTQ Files"):
                with st.spinner("Processing FASTQ files..."):
                    from services.file_upload_service import FileUploadService
                    upload_service = FileUploadService(st.session_state.data_manager)
                    result = upload_service.upload_fastq_files(uploaded_files)
                    st.sidebar.success("FASTQ files uploaded!")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result
                    })
                    st.rerun()
