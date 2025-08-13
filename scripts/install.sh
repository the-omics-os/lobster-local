#!/bin/bash
# Lobster AI Installation Script
# https://get.lobster-ai.com

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Lobster logo
print_logo() {
    echo -e "${RED}"
    echo "    🦞 LOBSTER AI"
    echo "    by Homara AI"
    echo -e "${NC}"
}

# Print colored message
print_message() {
    echo -e "${2}${1}${NC}"
}

# Check if command exists
command_exists() {
    command -v "$@" > /dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_message "❌ Unsupported operating system: $OSTYPE" "$RED"
        exit 1
    fi
}

# Check Python version
check_python() {
    if command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        print_message "❌ Python is not installed. Please install Python 3.9 or higher." "$RED"
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.9"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_message "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION" "$RED"
        exit 1
    fi
    
    print_message "✓ Python $PYTHON_VERSION detected" "$GREEN"
}

# Install using pip
install_with_pip() {
    print_message "📦 Installing Lobster AI via pip..." "$BLUE"
    
    # Upgrade pip first
    $PYTHON_CMD -m pip install --upgrade pip
    
    # Install lobster-ai
    if $PYTHON_CMD -m pip install lobster-ai; then
        print_message "✓ Lobster AI installed successfully!" "$GREEN"
    else
        print_message "❌ Failed to install Lobster AI" "$RED"
        exit 1
    fi
}

# Setup configuration
setup_config() {
    print_message "\n🔧 Configuration Setup" "$BLUE"
    
    # Create config directory
    CONFIG_DIR="$HOME/.lobster"
    mkdir -p "$CONFIG_DIR"
    
    # Check if .env exists
    if [ -f "$CONFIG_DIR/.env" ]; then
        print_message "✓ Configuration file already exists at $CONFIG_DIR/.env" "$GREEN"
    else
        print_message "📝 Creating configuration file..." "$YELLOW"
        
        # Create .env template
        cat > "$CONFIG_DIR/.env" << EOF
# Lobster AI Configuration
# Please add your API keys below

# Required
OPENAI_API_KEY=
AWS_BEDROCK_ACCESS_KEY=
AWS_BEDROCK_SECRET_ACCESS_KEY=

# Optional
NCBI_API_KEY=
GENIE_PROFILE=production
EOF
        
        print_message "✓ Configuration file created at $CONFIG_DIR/.env" "$GREEN"
        print_message "⚠️  Please edit $CONFIG_DIR/.env and add your API keys" "$YELLOW"
    fi
}

# Add to PATH if needed
check_path() {
    # Check if lobster is in PATH
    if command_exists lobster; then
        print_message "✓ 'lobster' command is available in PATH" "$GREEN"
    else
        # Find where pip installed the script
        LOBSTER_PATH=$($PYTHON_CMD -m site --user-base)/bin
        
        if [ -f "$LOBSTER_PATH/lobster" ]; then
            print_message "📝 Adding Lobster to PATH..." "$YELLOW"
            
            # Detect shell and add to appropriate config
            if [ -n "$ZSH_VERSION" ]; then
                SHELL_CONFIG="$HOME/.zshrc"
            elif [ -n "$BASH_VERSION" ]; then
                SHELL_CONFIG="$HOME/.bashrc"
            else
                SHELL_CONFIG="$HOME/.profile"
            fi
            
            echo "export PATH=\"$LOBSTER_PATH:\$PATH\"" >> "$SHELL_CONFIG"
            print_message "✓ Added to $SHELL_CONFIG" "$GREEN"
            print_message "⚠️  Please run: source $SHELL_CONFIG" "$YELLOW"
        fi
    fi
}

# Verify installation
verify_installation() {
    print_message "\n🔍 Verifying installation..." "$BLUE"
    
    if $PYTHON_CMD -c "import lobster" 2>/dev/null; then
        print_message "✓ Lobster AI Python package verified" "$GREEN"
        
        # Get version
        VERSION=$($PYTHON_CMD -c "from lobster.version import __version__; print(__version__)")
        print_message "✓ Version: $VERSION" "$GREEN"
    else
        print_message "❌ Failed to import Lobster AI package" "$RED"
        exit 1
    fi
}

# Main installation flow
main() {
    clear
    print_logo
    
    print_message "🦞 Installing Lobster AI - Multi-Agent Bioinformatics Analysis System\n" "$BLUE"
    
    # Check prerequisites
    detect_os
    check_python
    
    # Install
    install_with_pip
    
    # Setup
    setup_config
    check_path
    
    # Verify
    verify_installation
    
    # Success message
    echo
    print_message "🎉 Installation complete!" "$GREEN"
    echo
    print_message "Next steps:" "$BLUE"
    print_message "1. Add your API keys to ~/.lobster/.env" "$NC"
    print_message "2. Run 'lobster chat' to start" "$NC"
    print_message "3. Visit https://docs.lobster-ai.com for documentation" "$NC"
    echo
    print_message "Need help? Join our Discord: https://discord.gg/homaraai" "$BLUE"
    echo
}

# Run main function
main
