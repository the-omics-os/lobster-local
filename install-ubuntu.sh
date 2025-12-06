#!/usr/bin/env bash
# Lobster AI - Ubuntu/Debian Installation Helper
# This script checks for and installs system dependencies before running make install

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Helper functions
print_header() {
    echo ""
    echo -e "${BLUE}${BOLD}============================================${NC}"
    echo -e "${BLUE}${BOLD}   $1${NC}"
    echo -e "${BLUE}${BOLD}============================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Check if running on Debian/Ubuntu
check_platform() {
    if [ ! -f /etc/os-release ]; then
        print_error "Cannot detect OS. This script is for Ubuntu/Debian systems."
        exit 1
    fi

    . /etc/os-release

    if [[ "$ID" != "ubuntu" && "$ID" != "debian" && "$ID_LIKE" != *"ubuntu"* && "$ID_LIKE" != *"debian"* ]]; then
        print_error "This script is designed for Ubuntu/Debian systems."
        print_info "Detected: $PRETTY_NAME"
        print_info "You may need to manually install dependencies for your distribution."
        exit 1
    fi

    print_success "Detected: $PRETTY_NAME"
}

# Check for Python 3.11+
check_python() {
    print_info "Checking for Python 3.11+..."

    # Try different Python commands
    for cmd in python3.13 python3.12 python3.11 python3 python; do
        if command -v "$cmd" &> /dev/null; then
            version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
            major=$(echo "$version" | cut -d. -f1)
            minor=$(echo "$version" | cut -d. -f2)

            if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                PYTHON_CMD="$cmd"
                print_success "Found Python $version at $cmd"
                return 0
            fi
        fi
    done

    print_error "Python 3.11+ not found!"
    return 1
}

# Check if system dependencies are installed
check_dependencies() {
    local missing_packages=()

    print_info "Checking system dependencies..."

    # Required packages for Ubuntu/Debian
    local required_packages=(
        "build-essential"
        "pkg-config"
        "libhdf5-dev"
        "libxml2-dev"
        "libxslt1-dev"
        "libffi-dev"
        "libssl-dev"
        "libblas-dev"
        "liblapack-dev"
    )

    for package in "${required_packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package"; then
            missing_packages+=("$package")
        fi
    done

    # Check for Python dev package
    python_version=$(echo "$PYTHON_CMD" | grep -oP '\d+\.\d+')
    if [ -z "$python_version" ]; then
        python_version="3.11"
    fi

    local python_dev_pkg="python${python_version}-dev"
    local python_venv_pkg="python${python_version}-venv"

    if ! dpkg -l | grep -q "^ii  $python_dev_pkg"; then
        missing_packages+=("$python_dev_pkg")
    fi

    if ! dpkg -l | grep -q "^ii  $python_venv_pkg"; then
        missing_packages+=("$python_venv_pkg")
    fi

    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_success "All system dependencies are installed"
        return 0
    else
        echo ""
        print_warning "Missing system packages:"
        for package in "${missing_packages[@]}"; do
            echo "  • $package"
        done
        echo ""
        return 1
    fi
}

# Install missing dependencies
install_dependencies() {
    print_info "Installing system dependencies..."
    print_info "This requires sudo privileges."
    echo ""

    # Determine Python version for dev packages
    python_version=$(echo "$PYTHON_CMD" | grep -oP '\d+\.\d+')
    if [ -z "$python_version" ]; then
        python_version="3.11"
    fi

    # Update package lists
    sudo apt-get update

    # Install packages
    sudo apt-get install -y \
        build-essential \
        pkg-config \
        python${python_version}-dev \
        python${python_version}-venv \
        libhdf5-dev \
        libxml2-dev \
        libxslt1-dev \
        libffi-dev \
        libssl-dev \
        libblas-dev \
        liblapack-dev \
        git

    if [ $? -eq 0 ]; then
        print_success "System dependencies installed successfully"
        return 0
    else
        print_error "Failed to install dependencies"
        return 1
    fi
}

# Main installation
main() {
    print_header "🦞 Lobster AI - Ubuntu Installation"

    # Check platform
    check_platform
    echo ""

    # Check Python
    if ! check_python; then
        echo ""
        print_info "Python 3.11+ is required. Install with:"
        echo ""
        echo -e "  ${CYAN}sudo add-apt-repository ppa:deadsnakes/ppa${NC}"
        echo -e "  ${CYAN}sudo apt-get update${NC}"
        echo -e "  ${CYAN}sudo apt-get install python3.11 python3.11-venv python3.11-dev${NC}"
        echo ""
        exit 1
    fi
    echo ""

    # Check dependencies
    if ! check_dependencies; then
        echo ""
        read -p "Install missing dependencies? (y/N): " -n 1 -r
        echo ""

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if ! install_dependencies; then
                print_error "Installation failed"
                exit 1
            fi
        else
            print_warning "Cannot proceed without system dependencies"
            print_info "Install manually with:"
            echo ""
            echo -e "  ${CYAN}sudo apt-get install -y build-essential python3.11-dev libhdf5-dev libblas-dev${NC}"
            echo ""
            exit 1
        fi
    fi

    echo ""
    print_header "Installing Lobster AI"

    # Check if Makefile exists
    if [ ! -f "Makefile" ]; then
        print_error "Makefile not found. Are you in the lobster directory?"
        exit 1
    fi

    # Run make install
    print_info "Running: make install"
    echo ""

    if make install; then
        echo ""
        print_header "✅ Installation Complete!"
        echo ""
        print_success "Lobster AI is now installed!"
        echo ""
        print_info "Next steps:"
        echo ""
        echo -e "  ${CYAN}1. Configure your API key:${NC}"
        echo -e "     ${YELLOW}nano .env${NC}"
        echo ""
        echo -e "  ${CYAN}2. Activate the virtual environment:${NC}"
        echo -e "     ${YELLOW}source .venv/bin/activate${NC}"
        echo ""
        echo -e "  ${CYAN}3. Start using Lobster AI:${NC}"
        echo -e "     ${YELLOW}lobster chat${NC}"
        echo ""
        echo -e "  ${CYAN}For help:${NC}"
        echo -e "     ${YELLOW}lobster --help${NC}"
        echo ""
    else
        echo ""
        print_error "Installation failed!"
        print_info "Check the error messages above for details."
        exit 1
    fi
}

# Run main function
main "$@"
