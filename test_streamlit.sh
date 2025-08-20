#!/bin/bash

# Test script for Lobster Streamlit app
echo "🦞 Testing Lobster Streamlit App..."
echo "================================================"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed"
    echo "Installing streamlit..."
    pip install streamlit
else
    echo "✅ Streamlit is installed"
fi

# Check for other required dependencies
echo ""
echo "Checking dependencies..."
python -c "import pandas; print('✅ pandas installed')" 2>/dev/null || echo "❌ pandas missing"
python -c "import plotly; print('✅ plotly installed')" 2>/dev/null || echo "❌ plotly missing"
python -c "import rich; print('✅ rich installed')" 2>/dev/null || echo "❌ rich missing"
python -c "import langchain_core; print('✅ langchain_core installed')" 2>/dev/null || echo "❌ langchain_core missing"
python -c "import langchain_aws; print('✅ langchain_aws installed')" 2>/dev/null || echo "❌ langchain_aws missing"

echo ""
echo "================================================"
echo "To run the Streamlit app, use:"
echo "  streamlit run lobster/streamlit_app.py"
echo ""
echo "Or with custom settings:"
echo "  streamlit run lobster/streamlit_app.py --server.port 8080 --server.address 0.0.0.0"
echo "================================================"
