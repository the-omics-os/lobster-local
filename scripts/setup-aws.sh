#!/bin/bash

# 🦞 Lobster AI - AWS Setup Automation Script
# This script automates the AWS infrastructure setup for deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="us-east-2"
ECR_REPO_NAME="homara"
IAM_USER_NAME="github-actions-homara-lobster"
IAM_POLICY_NAME="GitHubActionsLobsterPolicy"
IAM_ROLE_NAME="AppRunnerECRAccessRole"

echo -e "${BLUE}🦞 Lobster AI - AWS Infrastructure Setup${NC}"
echo "=============================================="

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
    echo "Installation: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not configured. Please run 'aws configure' first.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AWS CLI is installed and configured${NC}"

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${BLUE}🔍 AWS Account ID: ${ACCOUNT_ID}${NC}"

# Function to check if resource exists
resource_exists() {
    local resource_type=$1
    local resource_name=$2
    case $resource_type in
        "user")
            aws iam get-user --user-name "$resource_name" &> /dev/null
            ;;
        "policy")
            aws iam get-policy --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$resource_name" &> /dev/null
            ;;
        "role")
            aws iam get-role --role-name "$resource_name" &> /dev/null
            ;;
        "ecr")
            aws ecr describe-repositories --repository-names "$resource_name" --region "$AWS_REGION" &> /dev/null
            ;;
    esac
}

# 1. Create IAM User for GitHub Actions
echo -e "\n${YELLOW}📝 Step 1: Creating IAM User for GitHub Actions${NC}"
if resource_exists "user" "$IAM_USER_NAME"; then
    echo -e "${YELLOW}⚠️  IAM user '$IAM_USER_NAME' already exists${NC}"
else
    aws iam create-user --user-name "$IAM_USER_NAME"
    echo -e "${GREEN}✅ Created IAM user: $IAM_USER_NAME${NC}"
fi

# 2. Create IAM Policy
echo -e "\n${YELLOW}📝 Step 2: Creating IAM Policy${NC}"
if resource_exists "policy" "$IAM_POLICY_NAME"; then
    echo -e "${YELLOW}⚠️  IAM policy '$IAM_POLICY_NAME' already exists${NC}"
else
    # Check if policy file exists
    if [ ! -f "github-actions-policy.json" ]; then
        echo -e "${RED}❌ Policy file 'github-actions-policy.json' not found!${NC}"
        echo -e "${BLUE}ℹ️  Please run this script from the project root directory${NC}"
        exit 1
    fi

    echo -e "${BLUE}📄 Using policy from github-actions-policy.json${NC}"
    aws iam create-policy \
        --policy-name "$IAM_POLICY_NAME" \
        --policy-document file://github-actions-policy.json

    # Attach policy to user
    aws iam attach-user-policy \
        --user-name "$IAM_USER_NAME" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$IAM_POLICY_NAME"

    echo -e "${GREEN}✅ Created and attached IAM policy: $IAM_POLICY_NAME${NC}"
fi

# 3. Create Access Key
echo -e "\n${YELLOW}📝 Step 3: Creating Access Key${NC}"
if aws iam list-access-keys --user-name "$IAM_USER_NAME" --query 'AccessKeyMetadata[0].AccessKeyId' --output text | grep -q "AKIA"; then
    echo -e "${YELLOW}⚠️  Access key already exists for user '$IAM_USER_NAME'${NC}"
    echo -e "${BLUE}ℹ️  If you need new keys, delete the old ones first${NC}"
else
    echo -e "${BLUE}🔑 Creating access key for GitHub Actions...${NC}"
    ACCESS_KEY_OUTPUT=$(aws iam create-access-key --user-name "$IAM_USER_NAME")
    
    ACCESS_KEY_ID=$(echo "$ACCESS_KEY_OUTPUT" | grep -o '"AccessKeyId": "[^"]*"' | cut -d'"' -f4)
    SECRET_ACCESS_KEY=$(echo "$ACCESS_KEY_OUTPUT" | grep -o '"SecretAccessKey": "[^"]*"' | cut -d'"' -f4)
    
    echo -e "${GREEN}✅ Access key created successfully!${NC}"
    echo -e "\n${RED}🚨 IMPORTANT: Save these credentials for GitHub Secrets:${NC}"
    echo -e "${YELLOW}AWS_ACCESS_KEY_ID: ${ACCESS_KEY_ID}${NC}"
    echo -e "${YELLOW}AWS_SECRET_ACCESS_KEY: ${SECRET_ACCESS_KEY}${NC}"
    echo -e "\n${RED}⚠️  These will not be shown again!${NC}"
fi

# 4. Create ECR Repository
echo -e "\n${YELLOW}📝 Step 4: Creating ECR Repository${NC}"
if resource_exists "ecr" "$ECR_REPO_NAME"; then
    echo -e "${YELLOW}⚠️  ECR repository '$ECR_REPO_NAME' already exists${NC}"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$AWS_REGION"
    echo -e "${GREEN}✅ Created ECR repository: $ECR_REPO_NAME${NC}"
fi

# Get ECR repository URI
ECR_URI=$(aws ecr describe-repositories \
    --repository-names "$ECR_REPO_NAME" \
    --region "$AWS_REGION" \
    --query 'repositories[0].repositoryUri' \
    --output text)
echo -e "${BLUE}📦 ECR Repository URI: ${ECR_URI}${NC}"

# 5. Create App Runner Service Role
echo -e "\n${YELLOW}📝 Step 5: Creating App Runner Service Role${NC}"
if resource_exists "role" "$IAM_ROLE_NAME"; then
    echo -e "${YELLOW}⚠️  IAM role '$IAM_ROLE_NAME' already exists${NC}"
else
    # Check if trust policy file exists
    if [ ! -f "apprunner-trust-policy.json" ]; then
        echo -e "${RED}❌ Trust policy file 'apprunner-trust-policy.json' not found!${NC}"
        echo -e "${BLUE}ℹ️  Please run this script from the project root directory${NC}"
        exit 1
    fi

    echo -e "${BLUE}📄 Using trust policy from apprunner-trust-policy.json${NC}"
    aws iam create-role \
        --role-name "$IAM_ROLE_NAME" \
        --assume-role-policy-document file://apprunner-trust-policy.json

    aws iam attach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess

    echo -e "${GREEN}✅ Created App Runner service role: $IAM_ROLE_NAME${NC}"
fi

# 6. Create App Runner Service-Linked Role
echo -e "\n${YELLOW}📝 Step 6: Creating App Runner Service-Linked Role${NC}"
if aws iam get-role --role-name AWSServiceRoleForAppRunner &> /dev/null; then
    echo -e "${GREEN}✅ App Runner service-linked role already exists${NC}"
else
    echo -e "${BLUE}Creating App Runner service-linked role...${NC}"
    aws iam create-service-linked-role --aws-service-name apprunner.amazonaws.com || true
    echo -e "${GREEN}✅ Service-linked role created${NC}"
fi

# 7. Set up billing alert (optional)
echo -e "\n${YELLOW}📝 Step 7: Setting up billing alert (optional)${NC}"
read -p "Do you want to set up a $20/month billing alert? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    aws budgets create-budget \
        --account-id "$ACCOUNT_ID" \
        --budget '{
            "BudgetName": "LobsterAI-Monthly",
            "BudgetLimit": {
                "Amount": "20",
                "Unit": "USD"
            },
            "TimeUnit": "MONTHLY",
            "BudgetType": "COST"
        }' 2>/dev/null && echo -e "${GREEN}✅ Billing alert created${NC}" || echo -e "${YELLOW}⚠️  Billing alert may already exist${NC}"
fi

# Clean up temporary files
rm -f /tmp/github-actions-policy.json /tmp/apprunner-trust-policy.json

echo -e "\n${GREEN}🎉 AWS Infrastructure Setup Complete!${NC}"
echo "=============================================="
echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. Add the AWS credentials to GitHub Secrets:"
echo -e "   - Go to: https://github.com/YOUR_USERNAME/lobster/settings/secrets/actions"
echo -e "   - Add: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
echo -e "2. Add your other environment variables as GitHub Secrets"
echo -e "3. Push to main branch to trigger deployment"
echo -e "4. Monitor deployment at: https://github.com/YOUR_USERNAME/lobster/actions"
echo -e "\n${GREEN}🦞 Your Lobster AI app will be live at: https://[random-id].us-east-2.awsapprunner.com${NC}"

# Summary
echo -e "\n${BLUE}📋 Resources Created:${NC}"
echo -e "• IAM User: ${IAM_USER_NAME}"
echo -e "• IAM Policy: ${IAM_POLICY_NAME}"
echo -e "• IAM Role: ${IAM_ROLE_NAME}"
echo -e "• ECR Repository: ${ECR_URI}"
echo -e "• AWS Region: ${AWS_REGION}"
