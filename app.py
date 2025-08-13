"""
Main application entry point.

This is the main entry point for the application, bringing together all
components and services to run the bioinformatics AI assistant.
"""

import os
import streamlit as st

from utils.logger import setup_logger
from utils.terminal_callback_handler import TerminalCallbackHandler
from config.settings import get_settings
from services.langgraph_agent_service_OLD import LangGraphAgentService
from app.ui import StreamlitUI

# Configure logging
logger = setup_logger(__name__)
settings = get_settings()

def main():
    """Main application function."""
    # Ensure export directory exists
    os.makedirs("data/exports", exist_ok=True)
    
    # Initialize UI
    ui = StreamlitUI("Genie 2.0", "AI Bioinformatician")
    ui.setup_page_config()
    ui.setup_css()
    ui.initialize_session_state()
    
    # Display UI components
    ui.show_main_ui()
    ui.show_sidebar(st.session_state.data_manager)
    ui.display_chat_history()
    
    # Initialize terminal callback handler if not already done
    if 'terminal_callback' not in st.session_state:
        st.session_state.terminal_callback = TerminalCallbackHandler(verbose=True)
    
    # Initialize agent if needed
    if 'agent_service' not in st.session_state:
        with st.spinner("Initializing AI agent..."):
            # Create langgraph agent service with terminal callback for thought processes
            agent_service = LangGraphAgentService(
                data_manager=st.session_state.data_manager,
                callback_handler=st.session_state.terminal_callback,
                chat_history=st.session_state.messages  # Pass existing chat history
            )
            st.session_state.agent_service = agent_service
    
    # Get user input
    prompt = ui.get_chat_input()
    
    # Process user input
    if prompt:
        # Clear previous thinking content
        ui.clear_thinking()
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process response outside of chat_message context
        try:
            # Run the agent with streaming, passing the current chat history
            with st.spinner("Thinking..."):
                response = st.session_state.agent_service.run_agent(
                    query=prompt, 
                    chat_history=st.session_state.messages
                )
            
            # Display the response using chat message
            with st.chat_message("assistant"):
                st.markdown(response)
                
                # Display plots if available
                if st.session_state.data_manager.latest_plots:
                    for i, plot_entry in enumerate(st.session_state.data_manager.latest_plots):
                        # Handle both dictionary format (with id and figure) and direct figure objects
                        if isinstance(plot_entry, dict) and "id" in plot_entry and "figure" in plot_entry:
                            plot_id = plot_entry["id"]
                            plot = plot_entry["figure"]
                        else:
                            # For backward compatibility with old format
                            plot = plot_entry
                            plot_id = f"main_plot_{id(plot)}"
                        
                        st.plotly_chart(
                            plot, 
                            use_container_width=True, 
                            key=plot_id,  # Add unique key for each plot
                            config={
                                'displayModeBar': True,
                                'toImageButtonOptions': {
                                    'format': 'png',
                                    'filename': 'genie_plot',
                                    'height': 800,
                                    'width': 1200,
                                    'scale': 2
                                },
                                'displaylogo': False,
                                'modeBarButtonsToAdd': ['downloadCsv', 'downloadXlsx']
                            }
                        )
            
            # Update the chat history
            if st.session_state.data_manager.latest_plots:
                # Process plots before adding to messages
                processed_plots = []
                for plot_entry in st.session_state.data_manager.latest_plots:
                    # Check if it's already a plot entry dictionary or just a figure
                    if isinstance(plot_entry, dict) and "id" in plot_entry and "figure" in plot_entry:
                        # It's already a plot entry dictionary
                        processed_plots.append(plot_entry)
                    else:
                        # It's a raw figure, create a plot entry with a generated ID
                        plot_id = f"hist_plot_{id(plot_entry)}"
                        processed_plots.append({
                            "id": plot_id,
                            "figure": plot_entry,
                            "title": "Plot"
                        })
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "plots": processed_plots
                })
                # Clear plots after displaying
                st.session_state.data_manager.latest_plots = []
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_msg
            })

if __name__ == "__main__":
    main()
