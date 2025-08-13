#!/bin/bash

# Lobster AI Integration Test Runner
# ================================
# 
# Simple wrapper script for running Lobster AI integration tests
#
# Usage:
#   ./run_tests.sh                    # Run tests sequentially
#   ./run_tests.sh --parallel         # Run tests in parallel
#   ./run_tests.sh --parallel -w 8    # Run tests in parallel with 8 workers
#   ./run_tests.sh --help             # Show help

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
INPUT_FILE="${SCRIPT_DIR}/test_cases.json"
OUTPUT_FILE="${SCRIPT_DIR}/results.json"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_integration_tests.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🦞 Lobster AI Integration Test Runner${NC}"
echo "=================================================="

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo -e "${RED}Error: $PYTHON_SCRIPT not found!${NC}"
    exit 1
fi

# Check if test cases file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo -e "${RED}Error: Test cases file $INPUT_FILE not found!${NC}"
    exit 1
fi

# Check if we're in a virtual environment or have the required packages
echo -e "${YELLOW}Checking Python environment...${NC}"
cd "$PROJECT_ROOT"

# Try to import the lobster package
if ! python -c "import lobster.core.client" 2>/dev/null; then
    echo -e "${RED}Error: Lobster package not found or not properly installed!${NC}"
    echo "Please make sure you have:"
    echo "  1. Installed the lobster package: pip install -e ."
    echo "  2. Set up your environment variables (OPENAI_API_KEY, etc.)"
    echo "  3. Are running from the project root directory"
    exit 1
fi

echo -e "${GREEN}✓ Python environment looks good${NC}"

# Count test cases
TEST_COUNT=$(python -c "import json; print(len(json.load(open('$INPUT_FILE'))))")
echo -e "${BLUE}Found $TEST_COUNT test cases${NC}"

# Run the tests
echo -e "${YELLOW}Starting tests...${NC}"
echo ""

# Change to project root and run the Python script
cd "$PROJECT_ROOT"
python "$PYTHON_SCRIPT" --input "$INPUT_FILE" --output "$OUTPUT_FILE" "$@"

# Check if results file was created
if [[ -f "$OUTPUT_FILE" ]]; then
    echo ""
    echo -e "${GREEN}✓ Test results saved to: $OUTPUT_FILE${NC}"
    
    # Show summary if possible
    if command -v jq >/dev/null 2>&1; then
        echo ""
        echo -e "${BLUE}Test Summary:${NC}"
        jq '.summary' "$OUTPUT_FILE"
    else
        echo ""
        echo -e "${YELLOW}Install 'jq' to see formatted test summary${NC}"
        echo "Raw summary:"
        python -c "import json; data=json.load(open('$OUTPUT_FILE')); print(f\"Total: {data['summary']['total_tests']}, Passed: {data['summary']['passed_tests']}, Failed: {data['summary']['failed_tests']}, Duration: {data['summary']['total_duration']:.2f}s\")"
    fi
else
    echo -e "${RED}Error: Results file was not created${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}🦞 Test run completed!${NC}"
