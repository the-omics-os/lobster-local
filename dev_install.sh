#!/bin/bash
# Development installation script for Lobster AI (Open Source)
# Installs core and local packages only

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦞 Lobster AI Open Source Installation Script${NC}"
echo "=============================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓ Virtual environment detected: $VIRTUAL_ENV${NC}"
elif [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment found but not activated${NC}"
    echo "   Activating .venv..."
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠️  No virtual environment detected${NC}"
    echo "   It's recommended to run this in a virtual environment"
    echo "   Run 'make install' for automatic setup"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

echo ""
echo -e "${YELLOW}📦 Installing open source packages...${NC}"

# Install core package first
if [ -d "lobster-core" ]; then
    echo "1/2 Installing lobster-core..."
    pip install -e ./lobster-core/ || {
        echo -e "${RED}❌ Failed to install lobster-core${NC}"
        exit 1
    }
else
    echo -e "${YELLOW}⚠️  lobster-core directory not found, skipping...${NC}"
fi

# Install local package (includes CLI)
if [ -d "lobster-local" ]; then
    echo "2/2 Installing lobster-local..."
    pip install -e ./lobster-local/ || {
        echo -e "${RED}❌ Failed to install lobster-local${NC}"
        exit 1
    }
else
    echo -e "${YELLOW}⚠️  lobster-local directory not found, using main package...${NC}"
    echo "2/2 Installing main lobster package..."
    pip install -e . || {
        echo -e "${RED}❌ Failed to install main package${NC}"
        exit 1
    }
fi

echo ""
echo -e "${GREEN}🎉 Open source installation complete!${NC}"
echo ""
echo -e "${BLUE}📋 What you have installed:${NC}"
echo "   • 🔗 lobster-core: Shared interfaces and utilities"
echo "   • 💻 lobster-local: Complete local bioinformatics implementation"
echo "   • 🦞 Full CLI with all features"
echo ""
echo -e "${BLUE}🚀 Getting started:${NC}"
echo "1. Configure your API keys:"
echo "   ${YELLOW}cp .env.example .env && nano .env${NC}"
echo ""
echo "2. Test the installation:"
echo "   ${YELLOW}lobster --help${NC}"
echo ""
echo "3. Start analyzing:"
echo "   ${YELLOW}lobster chat${NC}"
echo ""
echo -e "${BLUE}☁️  Interested in Lobster Cloud?${NC}"
echo "   Visit: ${YELLOW}https://cloud.lobster.ai${NC}"
echo "   Email: ${YELLOW}cloud@homara.ai${NC}"
echo ""
echo -e "${GREEN}✅ Ready to transform your bioinformatics research!${NC}"
