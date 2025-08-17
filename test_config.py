#!/usr/bin/env python3
"""
Test script for config functionality in cli.py
"""
import sys
from pathlib import Path
from rich.console import Console

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from our implementation - import the functions as they were defined
from lobster.cli import console
# These are the function names we used in the cli.py file
from lobster.cli import list_models, list_profiles, show_config, test, create_custom, generate_env

# Run a test of each command
def test_config_commands():
    print("\n===== Testing Configuration Commands =====")
    
    print("\n1. Testing list-models command:")
    try:
        list_models()
    except Exception as e:
        print(f"Error in list-models command: {e}")
    
    print("\n2. Testing list-profiles command:")
    try:
        list_profiles()
    except Exception as e:
        print(f"Error in list-profiles command: {e}")
    
    print("\n3. Testing show-config command (current profile):")
    try:
        show_config(profile=None)
    except Exception as e:
        print(f"Error in show-config command: {e}")
    
    # Test complete
    print("\n===== Testing complete! =====")

if __name__ == "__main__":
    test_config_commands()
