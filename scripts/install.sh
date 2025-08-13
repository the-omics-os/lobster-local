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

# Determine package manager to use (uv > pip3 > pip)
determine_pkg_manager() {
    # Check if uv is available (preferred)
    if command_exists uv; then
        echo "uv"
    elif command_exists pip3; then
        echo "pip3"
    elif command_exists pip; then
        echo "pip"
    else
        echo ""
    fi
}

# Determine pip command to use (for backward compatibility)
determine_pip_cmd() {
    if command_exists pip3; then
        echo "pip3"
    elif command_exists pip; then
        echo "pip"
    else
        echo ""
    fi
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
    
    # Determine package manager
    PKG_MGR=$(determine_pkg_manager)
    if [ -z "$PKG_MGR" ]; then
        print_message "❌ No package manager found. Please install pip, pip3, or uv." "$RED"
        exit 1
    fi
    print_message "✓ Using package manager: $PKG_MGR" "$GREEN"
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.9"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        print_message "❌ Python $REQUIRED_VERSION or higher is required. Found: $PYTHON_VERSION" "$RED"
        exit 1
    fi
    
    print_message "✓ Python $PYTHON_VERSION detected" "$GREEN"
    
    # Check for venv module
    if ! $PYTHON_CMD -c "import venv" 2>/dev/null; then
        print_message "❌ Python venv module not found. Your Python installation may be incomplete." "$RED"
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            print_message "Try: sudo apt install python3-venv (Ubuntu/Debian)" "$YELLOW"
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            print_message "Try: brew reinstall python3 (macOS)" "$YELLOW"
        fi
        exit 1
    fi
    
    print_message "✓ Python venv module available" "$GREEN"
}

# Create virtual environment with robust error handling
create_venv() {
    VENV_PATH="$1"
    print_message "🐍 Creating virtual environment at $VENV_PATH..." "$BLUE"
    
    # Remove existing venv if it exists
    if [ -d "$VENV_PATH" ]; then
        print_message "Found existing virtual environment, removing..." "$YELLOW"
        rm -rf "$VENV_PATH"
    fi
    
    # Try to create venv with normal method
    if $PYTHON_CMD -m venv "$VENV_PATH"; then
        print_message "✓ Virtual environment created successfully" "$GREEN"
    else
        # If failed, try without pip
        print_message "⚠️ Failed to create virtual environment. Trying alternative method..." "$YELLOW"
        if $PYTHON_CMD -m venv "$VENV_PATH" --without-pip; then
            print_message "✓ Created environment without pip, installing pip manually..." "$YELLOW"
            
            # Get pip installer
            curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
            "$VENV_PATH/bin/python" /tmp/get-pip.py
            rm /tmp/get-pip.py
            
            print_message "✓ Pip installed in virtual environment" "$GREEN"
        else
            print_message "❌ Failed to create virtual environment. Please check your Python installation." "$RED"
            exit 1
        fi
    fi
    
    # Verify pip exists
    if [ ! -f "$VENV_PATH/bin/pip" ]; then
        print_message "❌ Virtual environment created but pip is not available." "$RED"
        exit 1
    fi
}

# Install using package manager
install_with_pkg_manager() {
    PKG_MGR=$(determine_pkg_manager)
    print_message "📦 Installing Lobster AI via $PKG_MGR..." "$BLUE"
    
    if [ "$PKG_MGR" = "uv" ]; then
        # Using uv
        print_message "✓ Using uv for faster installation" "$GREEN"
        
        # Install lobster-ai with uv
        if uv pip install lobster-ai; then
            print_message "✓ Lobster AI installed successfully with uv!" "$GREEN"
        else
            print_message "❌ Failed to install Lobster AI with uv" "$RED"
            exit 1
        fi
    else
        # Using pip or pip3
        # Upgrade pip and install build tools first
        $PYTHON_CMD -m $PKG_MGR install --upgrade pip build wheel
        
        # Install lobster-ai
        if $PYTHON_CMD -m $PKG_MGR install lobster-ai; then
            print_message "✓ Lobster AI installed successfully!" "$GREEN"
        else
            print_message "❌ Failed to install Lobster AI" "$RED"
            exit 1
        fi
    fi
}

# Install from local directory
install_from_local() {
    VENV_PATH="$1"
    print_message "📦 Installing Lobster AI from local directory..." "$BLUE"
    
    # Check if uv is available outside venv
    if command_exists uv; then
        print_message "✓ Using uv for faster installation" "$GREEN"
        
        # Install with uv
        if uv pip install -e .; then
            print_message "✓ Lobster AI core installed successfully with uv!" "$GREEN"
        else
            print_message "❌ Failed to install with uv, falling back to pip" "$YELLOW"
            # Fall back to pip
            # Determine which pip executable exists in the venv
            if [ -f "$VENV_PATH/bin/pip3" ]; then
                VENV_PIP="$VENV_PATH/bin/pip3"
            else
                VENV_PIP="$VENV_PATH/bin/pip"
            fi
            
            # Upgrade pip and install build tools first
            "$VENV_PIP" install --upgrade pip build wheel
            
            # Install Lobster
            if "$VENV_PIP" install -e .; then
                print_message "✓ Lobster AI core installed successfully!" "$GREEN"
            else
                print_message "❌ Failed to install Lobster AI" "$RED"
                exit 1
            fi
        fi
    else
        # No uv, use pip
        # Determine which pip executable exists in the venv
        if [ -f "$VENV_PATH/bin/pip3" ]; then
            VENV_PIP="$VENV_PATH/bin/pip3"
        else
            VENV_PIP="$VENV_PATH/bin/pip"
        fi
        
        # Upgrade pip and install build tools first
        "$VENV_PIP" install --upgrade pip build wheel
        
        # Install Lobster
        print_message "📦 Installing Lobster AI and dependencies..." "$BLUE"
        
        if "$VENV_PIP" install -e .; then
            print_message "✓ Lobster AI core installed successfully!" "$GREEN"
        else
            print_message "❌ Failed to install Lobster AI" "$RED"
            exit 1
        fi
    fi
}

# Install dev dependencies - handles zsh square bracket escaping
install_dev_deps() {
    VENV_PATH="$1"
    print_message "📦 Installing development dependencies..." "$BLUE"
    
    # Check if uv is available
    if command_exists uv; then
        print_message "✓ Using uv for faster installation" "$GREEN"
        
        # Check if we're running in zsh
        if [ -n "$ZSH_VERSION" ]; then
            print_message "✓ Detected ZSH shell, escaping square brackets..." "$GREEN"
            uv pip install -e ".[dev]"
        else
            uv pip install -e .[dev]
        fi
    else
        # No uv, use pip
        # Determine which pip executable exists in the venv
        if [ -f "$VENV_PATH/bin/pip3" ]; then
            VENV_PIP="$VENV_PATH/bin/pip3"
        else
            VENV_PIP="$VENV_PATH/bin/pip"
        fi
        
        print_message "✓ Using pip: $VENV_PIP" "$GREEN"
        
        # Check if we're running in zsh
        if [ -n "$ZSH_VERSION" ]; then
            print_message "✓ Detected ZSH shell, escaping square brackets..." "$GREEN"
            "$VENV_PIP" install -e ".[dev]"
        else
            "$VENV_PIP" install -e .[dev]
        fi
    fi
    
    print_message "✓ Development dependencies installed!" "$GREEN"
}

# Create and configure local installation
local_install() {
    # Create .venv directory in current project
    VENV_PATH=".venv"
    create_venv "$VENV_PATH"
    
    # Install core dependencies
    install_from_local "$VENV_PATH"
    
    # Ask if dev dependencies should be installed
    print_message "\n📋 Would you like to install development dependencies? (y/n)" "$YELLOW"
    read -r install_dev
    if [[ "$install_dev" == "y" ]] || [[ "$install_dev" == "Y" ]]; then
        install_dev_deps "$VENV_PATH"
    fi
    
    print_message "\n✅ Local installation complete! To activate:\n" "$GREEN"
    print_message "   source $VENV_PATH/bin/activate" "$YELLOW"
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
    
    print_message "🦞 Lobster AI - Multi-Agent Bioinformatics Analysis System\n" "$BLUE"
    
    # Check prerequisites
    detect_os
    check_python
    
    # Determine installation type
    print_message "📋 Installation options:" "$BLUE"
    print_message "  1) Install to current directory (with virtual environment)" "$NC"
    print_message "  2) Install globally via pip" "$NC"
    print_message "\nChoose installation method (1-2):" "$YELLOW"
    read -r install_choice
    
    case $install_choice in
        1)
            # Local installation with venv
            local_install
            setup_config
            ;;
        2)
            # Global installation via package manager
            install_with_pkg_manager
            setup_config
            check_path
            ;;
        *)
            print_message "❌ Invalid choice. Exiting." "$RED"
            exit 1
            ;;
    esac
    
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
    
    # ZSH warning
    if [ -n "$ZSH_VERSION" ]; then
        # Get the correct pip command
        PIP_CMD=$(determine_pip_cmd)
        print_message "⚠️  ZSH USERS: When installing packages with square brackets, use:" "$YELLOW"
        print_message "   $PIP_CMD install -e \".[dev]\"  # Note the quotes" "$NC"
        print_message "   or" "$NC"
        print_message "   noglob $PIP_CMD install -e .[dev]" "$NC"
        echo
    fi
}

# Run main function
main
