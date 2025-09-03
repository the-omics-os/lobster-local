#!/bin/bash
# Automated AWS Deployment Script for Lobster Cloud
# This script builds and deploys the Lambda function

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LAMBDA_FUNCTION_NAME="lobster-api"
DEPLOYMENT_DIR="lambda-deployment"
DEPLOYMENT_ZIP="lambda-deployment.zip"
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Parse command line arguments
BUILD_ONLY=false
DEPLOY_ONLY=false
TEST_AFTER_DEPLOY=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --deploy-only)
            DEPLOY_ONLY=true
            shift
            ;;
        --no-test)
            TEST_AFTER_DEPLOY=false
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift
            shift
            ;;
        --function-name)
            LAMBDA_FUNCTION_NAME="$2"
            shift
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --build-only       Only build the deployment package"
            echo "  --deploy-only      Only deploy (skip build)"
            echo "  --no-test          Skip testing after deployment"
            echo "  --region REGION    AWS region (default: us-east-1)"
            echo "  --function-name NAME  Lambda function name (default: lobster-api)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🚀 Lobster Cloud Deployment Script${NC}"
echo "=================================="

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo -e "${RED}❌ AWS credentials not configured. Run 'aws configure' first.${NC}"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found.${NC}"
        exit 1
    fi
    
    # Check if we have the required files
    if [ ! -f "lobster-server/lambda_function.py" ]; then
        echo -e "${RED}❌ Lambda function file not found: lobster-server/lambda_function.py${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Build the deployment package
build_deployment_package() {
    echo -e "${YELLOW}Building deployment package...${NC}"
    
    # Clean up previous builds
    rm -rf "$DEPLOYMENT_DIR"
    rm -f "$DEPLOYMENT_ZIP"
    
    # Create deployment directory
    mkdir -p "$DEPLOYMENT_DIR"
    
    # Copy Lambda function
    cp lobster-server/lambda_function.py "$DEPLOYMENT_DIR/"
    
    # Install packages in virtual environment to avoid conflicts
    echo "Installing dependencies..."
    python3 -m venv "$DEPLOYMENT_DIR/venv"
    source "$DEPLOYMENT_DIR/venv/bin/activate"
    
    # Install lobster packages first
    echo "Installing lobster-core..."
    pip install -e ./lobster-core
    
    echo "Installing lobster-local..."
    pip install -e ./lobster-local
    
    # Install other dependencies (minimal set for Lambda)
    echo "Installing minimal dependencies..."
    pip install \
        requests>=2.31.0 \
        typing-extensions>=4.5.0 \
        langchain>=0.1.0,\<0.2.0 \
        langgraph>=0.0.20,\<0.1.0 \
        pandas>=1.5.0,\<2.1.0 \
        numpy>=1.23.0,\<1.25.0 \
        biopython>=1.81,\<1.84 \
        matplotlib>=3.6.0,\<3.8.0 \
        seaborn>=0.12.0,\<0.13.0
    
    deactivate
    
    # Copy installed packages to deployment directory
    echo "Copying packages..."
    cp -r "$DEPLOYMENT_DIR/venv/lib/python"*"/site-packages/"* "$DEPLOYMENT_DIR/"
    
    # Copy lobster packages manually to ensure they're included
    if [ -d "lobster-core/lobster_core" ]; then
        cp -r lobster-core/lobster_core "$DEPLOYMENT_DIR/"
    fi
    
    if [ -d "lobster-local" ]; then
        # Copy the entire lobster-local package structure
        mkdir -p "$DEPLOYMENT_DIR/lobster_local"
        
        # Copy all necessary directories
        for dir in agents config core tools utils; do
            if [ -d "lobster-local/$dir" ]; then
                cp -r "lobster-local/$dir" "$DEPLOYMENT_DIR/lobster_local/"
            fi
        done
        
        # Copy individual files
        for file in __init__.py version.py; do
            if [ -f "lobster-local/$file" ]; then
                cp "lobster-local/$file" "$DEPLOYMENT_DIR/lobster_local/"
            fi
        done
    fi
    
    # Remove virtual environment directory
    rm -rf "$DEPLOYMENT_DIR/venv"
    
    # Remove unnecessary files to reduce package size
    echo "Optimizing package size..."
    find "$DEPLOYMENT_DIR" -name "*.pyc" -delete
    find "$DEPLOYMENT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$DEPLOYMENT_DIR" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$DEPLOYMENT_DIR" -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$DEPLOYMENT_DIR" -name "test_*" -delete 2>/dev/null || true
    
    # Remove large ML libraries that might cause size issues
    rm -rf "$DEPLOYMENT_DIR"/tensorflow* 2>/dev/null || true
    rm -rf "$DEPLOYMENT_DIR"/torch* 2>/dev/null || true
    rm -rf "$DEPLOYMENT_DIR"/sklearn* 2>/dev/null || true
    rm -rf "$DEPLOYMENT_DIR"/scipy* 2>/dev/null || true
    rm -rf "$DEPLOYMENT_DIR"/plotly* 2>/dev/null || true
    
    # Create the deployment zip
    echo "Creating deployment package..."
    cd "$DEPLOYMENT_DIR"
    zip -r "../$DEPLOYMENT_ZIP" . -q
    cd ..
    
    # Check package size
    PACKAGE_SIZE=$(du -h "$DEPLOYMENT_ZIP" | cut -f1)
    echo -e "${GREEN}✅ Deployment package created: $DEPLOYMENT_ZIP ($PACKAGE_SIZE)${NC}"
    
    # Warn if package is too large
    PACKAGE_SIZE_BYTES=$(wc -c < "$DEPLOYMENT_ZIP")
    MAX_SIZE=$((250 * 1024 * 1024))  # 250MB
    if [ "$PACKAGE_SIZE_BYTES" -gt "$MAX_SIZE" ]; then
        echo -e "${YELLOW}⚠️ Warning: Package size ($PACKAGE_SIZE) is close to Lambda limit (250MB)${NC}"
    fi
    
    # Clean up deployment directory
    rm -rf "$DEPLOYMENT_DIR"
}

# Deploy to AWS Lambda
deploy_to_lambda() {
    echo -e "${YELLOW}Deploying to AWS Lambda...${NC}"
    
    if [ ! -f "$DEPLOYMENT_ZIP" ]; then
        echo -e "${RED}❌ Deployment package not found: $DEPLOYMENT_ZIP${NC}"
        echo "Run with --build-only first, or without --deploy-only"
        exit 1
    fi
    
    # Check if function exists
    if aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" &> /dev/null; then
        echo "Updating existing Lambda function..."
        aws lambda update-function-code \
            --function-name "$LAMBDA_FUNCTION_NAME" \
            --zip-file "fileb://$DEPLOYMENT_ZIP" \
            --region "$AWS_REGION" \
            --no-cli-pager
    else
        echo "Creating new Lambda function..."
        aws lambda create-function \
            --function-name "$LAMBDA_FUNCTION_NAME" \
            --runtime python3.11 \
            --role "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/lambda-execution-role" \
            --handler lambda_function.lambda_handler \
            --zip-file "fileb://$DEPLOYMENT_ZIP" \
            --timeout 300 \
            --memory-size 512 \
            --region "$AWS_REGION" \
            --no-cli-pager
        
        echo -e "${YELLOW}Note: You may need to create the IAM role 'lambda-execution-role' if it doesn't exist${NC}"
    fi
    
    echo -e "${GREEN}✅ Lambda function deployed successfully${NC}"
}

# Test the deployment
test_deployment() {
    echo -e "${YELLOW}Testing deployment...${NC}"
    
    # Create test payload
    cat > test_payload.json << EOF
{
    "httpMethod": "POST",
    "path": "/status",
    "headers": {
        "Authorization": "Bearer test-enterprise-001",
        "Content-Type": "application/json"
    },
    "body": "{}"
}
EOF
    
    # Invoke Lambda function
    echo "Testing Lambda function directly..."
    RESULT=$(aws lambda invoke \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --region "$AWS_REGION" \
        --payload file://test_payload.json \
        --no-cli-pager \
        response.json)
    
    # Check if invocation was successful
    if [ $? -eq 0 ]; then
        echo "Lambda invocation result:"
        cat response.json | python3 -m json.tool 2>/dev/null || cat response.json
        echo -e "${GREEN}✅ Lambda function test passed${NC}"
    else
        echo -e "${RED}❌ Lambda function test failed${NC}"
        exit 1
    fi
    
    # Clean up test files
    rm -f test_payload.json response.json
}

# Main execution
main() {
    check_prerequisites
    
    if [ "$DEPLOY_ONLY" = false ]; then
        build_deployment_package
    fi
    
    if [ "$BUILD_ONLY" = false ]; then
        deploy_to_lambda
        
        if [ "$TEST_AFTER_DEPLOY" = true ]; then
            test_deployment
        fi
    fi
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    
    if [ "$BUILD_ONLY" = false ]; then
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. Set up API Gateway (see aws_setup_guide.md)"
        echo "2. Configure API keys"
        echo "3. Test with: python test_aws_deployment.py"
        echo ""
        echo -e "${BLUE}Function ARN:${NC}"
        aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" --query 'Configuration.FunctionArn' --output text 2>/dev/null || echo "Run 'aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME' to get ARN"
    fi
}

# Run main function
main
