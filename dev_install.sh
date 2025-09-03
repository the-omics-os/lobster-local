#!/bin/bash
# Development installation script for Lobster package split

set -e  # Exit on any error

echo "🦞 Lobster Development Installation Script"
echo "==========================================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected"
    echo "   It's recommended to run this in a virtual environment"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
fi

echo
echo "📦 Installing packages in development mode..."

# Install core package first
echo "1/3 Installing lobster-core..."
cd lobster-core
pip install -e .
cd ..

# Install local package (includes CLI)
echo "2/3 Installing lobster-local (includes CLI)..."
cd lobster-local
pip install -e .
cd ..

# Install cloud client
echo "3/3 Installing lobster-cloud..."
cd lobster-cloud
pip install -e .
cd ..

echo
echo "✅ Development installation complete!"
echo
echo "📋 Next steps:"
echo "   • Test local mode: lobster query 'What is RNA-seq?'"
echo "   • Test cloud mode: export LOBSTER_CLOUD_KEY=your-key && lobster query 'What is RNA-seq?'"
echo "   • Run full test: python test_cloud_local.py"
echo
echo "🔧 Package structure:"
echo "   • lobster-core: Shared interfaces and utilities"
echo "   • lobster (from lobster-local): Full local implementation with CLI"
echo "   • lobster-cloud: Cloud client library"
echo
echo "🌟 The CLI will automatically detect LOBSTER_CLOUD_KEY and switch modes!"
