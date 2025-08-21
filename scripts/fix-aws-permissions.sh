#!/bin/bash

# 🦞 Lobster AI - AWS Permissions Fix Script
# This script updates existing AWS resources with corrected permissions

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="us-east-2"
IAM_USER_NAME="github-actions-lobster"
IAM_POLICY_NAME="GitHubActionsLobsterPolicy"
IAM_ROLE_NAME="AppRunnerECRAccessRole"

echo -e "${BLUE}🔧 Fixing AWS Permissions for Lobster AI${NC}"
echo "=============================================="

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${BLUE}🔍 AWS Account ID: ${ACCOUNT_ID}${NC}"

# 1. Update IAM Policy
echo -e "\n${YELLOW}📝 Step 1: Updating IAM Policy${NC}"
POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${IAM_POLICY_NAME}"

# Create the updated policy document
cat > /tmp/updated-github-actions-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage",
                "ecr:CreateRepository",
                "ecr:DescribeRepositories"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "apprunner:CreateService",
                "apprunner:UpdateService",
                "apprunner:DescribeService",
                "apprunner:ListServices",
                "apprunner:DeleteService",
                "apprunner:TagResource",
                "apprunner:UntagResource",
                "apprunner:ListTagsForResource"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreateRole",
                "iam:GetRole",
                "iam:AttachRolePolicy",
                "iam:PassRole",
                "iam:CreateServiceLinkedRole"
            ],
            "Resource": [
                "arn:aws:iam::${ACCOUNT_ID}:role/AppRunnerECRAccessRole",
                "arn:aws:iam::*:role/aws-service-role/apprunner.amazonaws.com/AWSServiceRoleForAppRunner"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole"
            ],
            "Resource": "arn:aws:iam::${ACCOUNT_ID}:role/AppRunnerECRAccessRole",
            "Condition": {
                "StringEquals": {
                    "iam:PassedToService": "apprunner.amazonaws.com"
                }
            }
        }
    ]
}
EOF

# Substitute the account ID in the policy
sed -i.bak "s/\${ACCOUNT_ID}/${ACCOUNT_ID}/g" /tmp/updated-github-actions-policy.json

# Create a new policy version
echo -e "${BLUE}Updating IAM policy...${NC}"
aws iam create-policy-version \
    --policy-arn "$POLICY_ARN" \
    --policy-document file:///tmp/updated-github-actions-policy.json \
    --set-as-default

echo -e "${GREEN}✅ IAM policy updated successfully${NC}"

# 2. Update App Runner Role Trust Policy
echo -e "\n${YELLOW}📝 Step 2: Updating App Runner Role Trust Policy${NC}"

cat > /tmp/updated-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "tasks.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

echo -e "${BLUE}Updating role trust policy...${NC}"
aws iam update-assume-role-policy \
    --role-name "$IAM_ROLE_NAME" \
    --policy-document file:///tmp/updated-trust-policy.json

echo -e "${GREEN}✅ Role trust policy updated successfully${NC}"

# 3. Verify the role has the correct policy attached
echo -e "\n${YELLOW}📝 Step 3: Verifying Role Policies${NC}"
ATTACHED_POLICIES=$(aws iam list-attached-role-policies --role-name "$IAM_ROLE_NAME" --query 'AttachedPolicies[].PolicyArn' --output text)

if echo "$ATTACHED_POLICIES" | grep -q "AWSAppRunnerServicePolicyForECRAccess"; then
    echo -e "${GREEN}✅ Role has correct policy attached${NC}"
else
    echo -e "${YELLOW}⚠️  Attaching AWSAppRunnerServicePolicyForECRAccess policy...${NC}"
    aws iam attach-role-policy \
        --role-name "$IAM_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
    echo -e "${GREEN}✅ Policy attached successfully${NC}"
fi

# 4. Create Service-Linked Role for App Runner (if it doesn't exist)
echo -e "\n${YELLOW}📝 Step 4: Ensuring App Runner Service-Linked Role exists${NC}"
if aws iam get-role --role-name AWSServiceRoleForAppRunner &> /dev/null; then
    echo -e "${GREEN}✅ App Runner service-linked role already exists${NC}"
else
    echo -e "${BLUE}Creating App Runner service-linked role...${NC}"
    aws iam create-service-linked-role --aws-service-name apprunner.amazonaws.com || true
    echo -e "${GREEN}✅ Service-linked role created${NC}"
fi

# Clean up
rm -f /tmp/updated-github-actions-policy.json /tmp/updated-github-actions-policy.json.bak /tmp/updated-trust-policy.json

echo -e "\n${GREEN}🎉 Permissions Fixed!${NC}"
echo "=============================================="
echo -e "${BLUE}Next Steps:${NC}"
echo -e "1. Push your changes to trigger the GitHub Actions workflow"
echo -e "2. The deployment should now work without permission errors"
echo -e "\n${YELLOW}If you still encounter issues, you may need to:${NC}"
echo -e "- Delete old policy versions if you've hit the version limit (5 versions max)"
echo -e "- Ensure your GitHub Secrets are correctly set"
echo -e "\n${GREEN}Good luck with your deployment! 🦞${NC}"
