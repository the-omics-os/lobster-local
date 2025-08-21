#!/bin/bash

# 🦞 Lobster AI - AWS Fargate Setup Automation Script
# This script automates the AWS ECS Fargate infrastructure setup for deployment

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
IAM_POLICY_NAME="GitHubActionsLobsterFargatePolicy"
ECS_TASK_EXECUTION_ROLE="ecsTaskExecutionRole"
ECS_TASK_ROLE="ecsTaskRole"
ECS_CLUSTER_NAME="lobster-cluster"
ECS_SERVICE_NAME="lobster-streamlit-service"
ALB_NAME="lobster-alb"
TARGET_GROUP_NAME="lobster-tg"
SECURITY_GROUP_ALB="lobster-alb-sg"
SECURITY_GROUP_ECS="lobster-ecs-sg"

echo -e "${BLUE}🦞 Lobster AI - AWS Fargate Infrastructure Setup${NC}"
echo "=================================================="

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
        "cluster")
            aws ecs describe-clusters --clusters "$resource_name" --region "$AWS_REGION" --query 'clusters[?status==`ACTIVE`]' --output text | grep -q "$resource_name"
            ;;
        "log-group")
            aws logs describe-log-groups --log-group-name-prefix "$resource_name" --region "$AWS_REGION" --query 'logGroups[?logGroupName==`'$resource_name'`]' --output text | grep -q "$resource_name"
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

# 2. Create IAM Policy for Fargate
echo -e "\n${YELLOW}📝 Step 2: Creating IAM Policy for Fargate${NC}"
if resource_exists "policy" "$IAM_POLICY_NAME"; then
    echo -e "${YELLOW}⚠️  IAM policy '$IAM_POLICY_NAME' already exists${NC}"
    # Ensure policy is attached to user even if it already exists
    aws iam attach-user-policy \
        --user-name "$IAM_USER_NAME" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$IAM_POLICY_NAME" 2>/dev/null || true
else
    # Check if policy file exists
    if [ ! -f "scripts/fargate-iam-policies.json" ]; then
        echo -e "${RED}❌ Policy file 'scripts/fargate-iam-policies.json' not found!${NC}"
        echo -e "${BLUE}ℹ️  Please run this script from the project root directory${NC}"
        exit 1
    fi

    echo -e "${BLUE}📄 Using policy from scripts/fargate-iam-policies.json${NC}"
    aws iam create-policy \
        --policy-name "$IAM_POLICY_NAME" \
        --policy-document file://scripts/fargate-iam-policies.json

    # Attach policy to user
    aws iam attach-user-policy \
        --user-name "$IAM_USER_NAME" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/$IAM_POLICY_NAME"

    echo -e "${GREEN}✅ Created and attached IAM policy: $IAM_POLICY_NAME${NC}"
fi

# 3. Create Access Key (if needed)
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

# 5. Create ECS Task Execution Role
echo -e "\n${YELLOW}📝 Step 5: Creating ECS Task Execution Role${NC}"
if resource_exists "role" "$ECS_TASK_EXECUTION_ROLE"; then
    echo -e "${YELLOW}⚠️  IAM role '$ECS_TASK_EXECUTION_ROLE' already exists${NC}"
else
    echo -e "${BLUE}📄 Using trust policy from scripts/ecs-trust-policy.json${NC}"
    aws iam create-role \
        --role-name "$ECS_TASK_EXECUTION_ROLE" \
        --assume-role-policy-document file://scripts/ecs-trust-policy.json

    aws iam attach-role-policy \
        --role-name "$ECS_TASK_EXECUTION_ROLE" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

    echo -e "${GREEN}✅ Created ECS task execution role: $ECS_TASK_EXECUTION_ROLE${NC}"
fi

# 6. Create ECS Task Role
echo -e "\n${YELLOW}📝 Step 6: Creating ECS Task Role${NC}"
if resource_exists "role" "$ECS_TASK_ROLE"; then
    echo -e "${YELLOW}⚠️  IAM role '$ECS_TASK_ROLE' already exists${NC}"
else
    aws iam create-role \
        --role-name "$ECS_TASK_ROLE" \
        --assume-role-policy-document file://scripts/ecs-trust-policy.json

    echo -e "${GREEN}✅ Created ECS task role: $ECS_TASK_ROLE${NC}"
fi

# 7. Create CloudWatch Log Group
echo -e "\n${YELLOW}📝 Step 7: Creating CloudWatch Log Group${NC}"
LOG_GROUP_NAME="/ecs/lobster-streamlit"
if resource_exists "log-group" "$LOG_GROUP_NAME"; then
    echo -e "${YELLOW}⚠️  Log group '$LOG_GROUP_NAME' already exists${NC}"
else
    aws logs create-log-group \
        --log-group-name "$LOG_GROUP_NAME" \
        --region "$AWS_REGION"
    echo -e "${GREEN}✅ Created CloudWatch log group: $LOG_GROUP_NAME${NC}"
fi

# 8. Create ECS Cluster
echo -e "\n${YELLOW}📝 Step 8: Creating ECS Cluster${NC}"
if resource_exists "cluster" "$ECS_CLUSTER_NAME"; then
    echo -e "${YELLOW}⚠️  ECS cluster '$ECS_CLUSTER_NAME' already exists${NC}"
else
    aws ecs create-cluster \
        --cluster-name "$ECS_CLUSTER_NAME" \
        --region "$AWS_REGION"
    echo -e "${GREEN}✅ Created ECS cluster: $ECS_CLUSTER_NAME${NC}"
fi

# 9. Get Default VPC and Subnets
echo -e "\n${YELLOW}📝 Step 9: Getting VPC and Subnet Information${NC}"
DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=is-default,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text \
    --region "$AWS_REGION")

if [ "$DEFAULT_VPC" = "None" ] || [ -z "$DEFAULT_VPC" ]; then
    echo -e "${RED}❌ No default VPC found. Please create a VPC first.${NC}"
    exit 1
fi

SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query 'Subnets[*].SubnetId' \
    --output text \
    --region "$AWS_REGION")

echo -e "${BLUE}🌐 Default VPC: ${DEFAULT_VPC}${NC}"
echo -e "${BLUE}🌐 Subnets: ${SUBNETS}${NC}"

# 10. Create Security Groups
echo -e "\n${YELLOW}📝 Step 10: Creating Security Groups${NC}"

# ALB Security Group
ALB_SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_ALB" "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || echo "None")

if [ "$ALB_SG_ID" = "None" ] || [ -z "$ALB_SG_ID" ]; then
    ALB_SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_ALB" \
        --description "Security group for Lobster ALB" \
        --vpc-id "$DEFAULT_VPC" \
        --region "$AWS_REGION" \
        --query 'GroupId' \
        --output text)

    # Allow HTTP traffic from anywhere
    aws ec2 authorize-security-group-ingress \
        --group-id "$ALB_SG_ID" \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region "$AWS_REGION"

    echo -e "${GREEN}✅ Created ALB security group: $ALB_SG_ID${NC}"
else
    echo -e "${YELLOW}⚠️  ALB security group already exists: $ALB_SG_ID${NC}"
fi

# ECS Security Group
ECS_SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_ECS" "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query 'SecurityGroups[0].GroupId' \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || echo "None")

if [ "$ECS_SG_ID" = "None" ] || [ -z "$ECS_SG_ID" ]; then
    ECS_SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_ECS" \
        --description "Security group for Lobster ECS tasks" \
        --vpc-id "$DEFAULT_VPC" \
        --region "$AWS_REGION" \
        --query 'GroupId' \
        --output text)

    # Allow traffic from ALB on port 8501
    aws ec2 authorize-security-group-ingress \
        --group-id "$ECS_SG_ID" \
        --protocol tcp \
        --port 8501 \
        --source-group "$ALB_SG_ID" \
        --region "$AWS_REGION"

    echo -e "${GREEN}✅ Created ECS security group: $ECS_SG_ID${NC}"
else
    echo -e "${YELLOW}⚠️  ECS security group already exists: $ECS_SG_ID${NC}"
fi

echo -e "\n${GREEN}🎉 AWS Fargate Infrastructure Setup Complete!${NC}"
echo "=============================================="
echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. Add the following secrets to GitHub:"
echo -e "   - Go to: https://github.com/YOUR_USERNAME/lobster/settings/secrets/actions"
echo -e "   - Add AWS_ACCESS_KEY_ID (from above output if new)"
echo -e "   - Add AWS_SECRET_ACCESS_KEY (from above output if new)"
echo -e "   - Add AWS_REGION: ${AWS_REGION}"
echo -e "   - Add ECS_CLUSTER_NAME: ${ECS_CLUSTER_NAME}"
echo -e "   - Add ECS_SERVICE_NAME: ${ECS_SERVICE_NAME}"
echo -e "2. Update your GitHub Actions workflow to use Fargate deployment"
echo -e "3. Push to main branch to trigger deployment"
echo -e "4. Monitor deployment at: https://github.com/YOUR_USERNAME/lobster/actions"

# Summary
echo -e "\n${BLUE}📋 Resources Created:${NC}"
echo -e "• IAM User: ${IAM_USER_NAME}"
echo -e "• IAM Policy: ${IAM_POLICY_NAME}"
echo -e "• ECS Task Execution Role: ${ECS_TASK_EXECUTION_ROLE}"
echo -e "• ECS Task Role: ${ECS_TASK_ROLE}"
echo -e "• ECR Repository: ${ECR_URI}"
echo -e "• ECS Cluster: ${ECS_CLUSTER_NAME}"
echo -e "• CloudWatch Log Group: ${LOG_GROUP_NAME}"
echo -e "• VPC: ${DEFAULT_VPC}"
echo -e "• ALB Security Group: ${ALB_SG_ID}"
echo -e "• ECS Security Group: ${ECS_SG_ID}"
echo -e "• AWS Region: ${AWS_REGION}"

echo -e "\n${RED}🔑 CRITICAL: Add these GitHub Secrets:${NC}"
echo -e "${YELLOW}AWS_ACCESS_KEY_ID: [from above output if new]${NC}"
echo -e "${YELLOW}AWS_SECRET_ACCESS_KEY: [from above output if new]${NC}"
echo -e "${YELLOW}AWS_REGION: ${AWS_REGION}${NC}"
echo -e "${YELLOW}ECS_CLUSTER_NAME: ${ECS_CLUSTER_NAME}${NC}"
echo -e "${YELLOW}ECS_SERVICE_NAME: ${ECS_SERVICE_NAME}${NC}"

echo -e "\n${GREEN}🦞 Ready for Fargate deployment! Update your GitHub workflow and push to main.${NC}"
