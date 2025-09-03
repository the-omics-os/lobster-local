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
echo -e "${YELLOW}📦 Installing Lobster AI (single package)...${NC}"

# Install the main lobster package
echo "Installing lobster package..."
pip install -e . || {
    echo -e "${RED}❌ Failed to install lobster package${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}🎉 Open source installation complete!${NC}"
echo ""
echo -e "${BLUE}📋 What you have installed:${NC}"
echo "   • 🦞 Complete Lobster AI bioinformatics platform"
echo "   • 🤖 All AI agents (Data, Research, Transcriptomics, Proteomics)"
echo "   • 🔧 Full analysis capabilities and visualization tools"
echo "   • 💻 Interactive CLI with natural language interface"
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
