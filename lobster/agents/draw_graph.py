#!/usr/bin/env python
"""
Simple script to generate a flow diagram of the Lobster bioinformatics agent graph.
"""

import os
import sys
import argparse
from langgraph.checkpoint.memory import InMemorySaver

# Add the current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
##########################################
##########################################
##########################################
## NEEDS Migration to DATAMANGER 2
##########################################
##########################################
##########################################
# Import required components
from lobster.core.data_manager import DataManager
from lobster.agents.graph import create_bioinformatics_graph


def generate_graph_image(output_path=None, display_image=False):
    """
    Generate a Mermaid diagram of the bioinformatics agent graph.
    
    Args:
        output_path: Path where to save the image. If None, uses 'lobster_graph.png'
        display_image: Whether to attempt displaying the image (works in IPython/Jupyter)
    
    Returns:
        Path to the generated image file
    """
    print("Initializing data manager...")
    data_manager = DataManager()
    
    print("Creating bioinformatics graph...")
    # Create graph with minimal dependencies
    checkpointer = InMemorySaver()
    supervisor = create_bioinformatics_graph(
        data_manager=data_manager,
        checkpointer=checkpointer
    )
    
    # Always save output in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_path is None:
        output_path = os.path.join(script_dir, "lobster_graph.png")
    else:
        output_path = os.path.join(script_dir, os.path.basename(output_path))
    
    # Ensure output directory exists (should always exist since it's script_dir)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generating Mermaid diagram and saving to {output_path}...")
    # Generate the PNG and save it
    png_data = supervisor.get_graph().draw_mermaid_png()
    
    with open(output_path, "wb") as f:
        f.write(png_data)
    
    print(f"Graph diagram saved to: {output_path}")
    
    # Display the image if requested and running in IPython
    if display_image:
        try:
            from IPython.display import display, Image
            print("Displaying image...")
            display(Image(png_data))
        except ImportError:
            print("IPython not available. Image saved but cannot be displayed.")
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Mermaid diagram of the Lobster bioinformatics graph")
    parser.add_argument("--output", "-o", help="Output file path for the diagram (default: lobster_graph.png)")
    parser.add_argument("--display", "-d", action="store_true", help="Attempt to display the image (works in IPython/Jupyter)")
    
    args = parser.parse_args()
    generate_graph_image(args.output, args.display)
