# 🦞 Lobster AI - AWS Fargate Deployment Guide

Complete step-by-step guide to deploy your Streamlit app to AWS ECS Fargate with automatic CI/CD from GitHub.

## 📋 Prerequisites

- **AWS Account** with billing enabled
- **GitHub repository** with your code
- **AWS CLI** installed locally
- **Docker** installed locally (optional, for testing)

---

## 🚀 Part 1: AWS Infrastructure Setup

### 1.1 Install and Configure AWS CLI

```bash
# Install AWS CLI (if not already installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
```

Enter your credentials:
- **AWS Access Key ID**: `<YOUR_ACCESS_KEY>`
- **AWS Secret Access Key**: `<YOUR_SECRET_KEY>`
- **Default region**: `us-east-2`
- **Default output format**: `json`

### 1.2 Automated Setup (Recommended)

🚀 **Easy Option:** Use the automated Fargate setup script:

```bash
# Run the automated setup script from project root
bash scripts/setup-fargate.sh
```

This script will:
- ✅ Create IAM user and policies for GitHub Actions
- ✅ Generate access keys  
- ✅ Create ECR repository
- ✅ Set up ECS task execution and task roles
- ✅ Create ECS cluster
- ✅ Create CloudWatch log group
- ✅ Set up VPC security groups for ALB and ECS
- ✅ Display all necessary GitHub secrets

**Note:** Make sure to save the displayed credentials for GitHub Secrets!

---

## 🔐 Part 2: GitHub Secrets Configuration

### 2.1 Add AWS Secrets to GitHub

Go to your GitHub repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Add these secrets:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | `AKIA...` | From GitHub Actions IAM user |
| `AWS_SECRET_ACCESS_KEY` | `wJal...` | From GitHub Actions IAM user |
| `AWS_REGION` | `us-east-2` | AWS region for deployment |
| `ECS_CLUSTER_NAME` | `lobster-cluster` | ECS cluster name |
| `ECS_SERVICE_NAME` | `lobster-streamlit-service` | ECS service name |
| `OPENAI_API_KEY` | `sk-proj-...` | Your OpenAI API key |
| `AWS_BEDROCK_ACCESS_KEY` | `AKIA...` | Your AWS Bedrock access key |
| `AWS_BEDROCK_SECRET_ACCESS_KEY` | `EF/a...` | Your AWS Bedrock secret key |
| `NCBI_API_KEY` | `b01b...` | Your NCBI API key |
| `GENIE_PROFILE` | `ultra-performance` | Your Genie profile |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` | Your Langfuse public key |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` | Your Langfuse secret key |
| `LANGFUSE_HOST` | `https://us.cloud.langfuse.com` | Your Langfuse host |
| `STRIPE_PUBLISHABLE_KEY` | `pk_live_...` | Your Stripe publishable key |
| `STRIPE_SECRET_KEY` | `sk_live_...` | Your Stripe secret key |
| `STRIPE_WEBHOOK_SECRET` | `whsec_...` | Your Stripe webhook secret |

### 2.2 Verify Secrets

Go to **Actions** tab in your GitHub repository and check that all secrets are listed (but values are hidden).

---

## 🏗️ Part 3: Initial Deployment

### 3.1 Test Docker Build Locally (Optional)

```bash
# Build the Docker image locally to test
docker build -f Dockerfile.streamlit -t lobster-ai:test .

# Test run locally
docker run -p 8501:8501 --env-file .env lobster-ai:test

# Visit http://localhost:8501 to test
```

### 3.2 Deploy to AWS Fargate

1. **Push to main branch:**
```bash
git add .
git commit -m "Add AWS Fargate deployment configuration"
git push origin main
```

2. **Monitor deployment:**
- Go to GitHub **Actions** tab
- Watch the "Deploy to AWS Fargate - ECS" workflow
- Check for any errors in the logs

3. **Get your app URL:**
There are several ways to find your deployed app URL:

**Option A: AWS Console (Most Reliable)**
1. Go to AWS Console: https://console.aws.amazon.com/ec2/
2. Navigate to **Load Balancers** under **Load Balancing**
3. Look for your load balancer named `lobster-alb`
4. The **DNS name** will be displayed

**Option B: GitHub Actions Output**
- Check the workflow logs in the "Create or Update ECS Service" step
- Look for the ALB DNS name output

**Option C: AWS CLI**
```bash
aws elbv2 describe-load-balancers \
    --names lobster-alb \
    --query 'LoadBalancers[0].DNSName' \
    --output text \
    --region us-east-2
```

Your app will be available at: `http://[alb-dns-name].us-east-2.elb.amazonaws.com`

---

## 💰 Part 4: Cost Management

### 4.1 Expected Costs

| Service | Configuration | Monthly Cost (Estimate) |
|---------|---------------|------------------------|
| **ECR** | < 1GB storage | ~$0.10 |
| **ECS Fargate** | 0.5 vCPU, 1GB RAM, minimal traffic | ~$15-20 |
| **Application Load Balancer** | Basic usage | ~$18 |
| **Data Transfer** | Minimal usage | ~$1 |
| **CloudWatch Logs** | Basic logging | ~$2 |
| **Total** | | **~$36-41/month** |

### 4.2 Cost Optimization Tips

```bash
# Set up billing alerts
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget '{
        "BudgetName": "LobsterAI-Fargate-Monthly",
        "BudgetLimit": {
            "Amount": "50",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }'

# Monitor costs
aws ce get-cost-and-usage \
    --time-period Start=2024-08-01,End=2024-08-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

---

## 🔧 Part 5: Management & Monitoring

### 5.1 ECS Fargate Management

```bash
# Check service status
aws ecs describe-services \
    --cluster lobster-cluster \
    --services lobster-streamlit-service \
    --region us-east-2

# View service logs
aws logs get-log-events \
    --log-group-name /ecs/lobster-streamlit \
    --log-stream-name ecs/lobster-streamlit/$(date +%Y/%m/%d) \
    --region us-east-2

# Scale service (if needed)
aws ecs update-service \
    --cluster lobster-cluster \
    --service lobster-streamlit-service \
    --desired-count 2 \
    --region us-east-2
```

### 5.2 Application Load Balancer Management

```bash
# Check ALB status
aws elbv2 describe-load-balancers \
    --names lobster-alb \
    --region us-east-2

# Check target group health
aws elbv2 describe-target-health \
    --target-group-arn $(aws elbv2 describe-target-groups --names lobster-tg --query 'TargetGroups[0].TargetGroupArn' --output text --region us-east-2) \
    --region us-east-2
```

---

## 🚨 Part 6: Troubleshooting

### 6.1 Common Issues

**❌ Build Fails - "Permission Denied"**
```bash
# Check IAM permissions
aws iam list-attached-user-policies --user-name github-actions-homara-lobster
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::ACCOUNT:user/github-actions-homara-lobster \
    --action-names ecs:RegisterTaskDefinition
```

**❌ ECS Service Won't Start**
```bash
# Check service events
aws ecs describe-services \
    --cluster lobster-cluster \
    --services lobster-streamlit-service \
    --region us-east-2 \
    --query 'services[0].events'

# Check task definition
aws ecs describe-task-definition \
    --task-definition lobster-streamlit-task \
    --region us-east-2
```

**❌ Application Load Balancer Health Checks Failing**
```bash
# Check target group health
aws elbv2 describe-target-health \
    --target-group-arn $(aws elbv2 describe-target-groups --names lobster-tg --query 'TargetGroups[0].TargetGroupArn' --output text --region us-east-2) \
    --region us-east-2

# Check security group rules
aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=lobster-ecs-sg" \
    --region us-east-2
```

### 6.2 Debug Commands

```bash
# Test ECR login
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-2.amazonaws.com

# List ECR images
aws ecr list-images --repository-name homara --region us-east-2

# Check ECS service logs
aws logs tail /ecs/lobster-streamlit --follow --region us-east-2

# Check running tasks
aws ecs list-tasks --cluster lobster-cluster --service-name lobster-streamlit-service --region us-east-2
```

---

## 🎯 Part 7: Advanced Configuration

### 7.1 Custom Domain (Optional)

```bash
# Create Route 53 hosted zone (if you have a domain)
aws route53 create-hosted-zone \
    --name yourdomain.com \
    --caller-reference $(date +%s)

# Add CNAME record pointing to ALB
aws route53 change-resource-record-sets \
    --hosted-zone-id YOUR_ZONE_ID \
    --change-batch '{
        "Changes": [{
            "Action": "CREATE",
            "ResourceRecordSet": {
                "Name": "lobster.yourdomain.com",
                "Type": "CNAME",
                "TTL": 300,
                "ResourceRecords": [{"Value": "YOUR_ALB_DNS_NAME"}]
            }
        }]
    }'
```

### 7.2 Auto Scaling Configuration

```bash
# Create auto scaling target
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --resource-id service/lobster-cluster/lobster-streamlit-service \
    --scalable-dimension ecs:service:DesiredCount \
    --min-capacity 1 \
    --max-capacity 3 \
    --region us-east-2

# Create scaling policy
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --resource-id service/lobster-cluster/lobster-streamlit-service \
    --scalable-dimension ecs:service:DesiredCount \
    --policy-name lobster-scaling-policy \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy '{
        "TargetValue": 70.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        }
    }' \
    --region us-east-2
```

---

## ✅ Part 8: Verification Checklist

Before going live, verify:

- [ ] ECR repository created and accessible
- [ ] GitHub Actions workflow runs successfully
- [ ] ECS cluster and service are running
- [ ] All environment variables are set correctly
- [ ] Application Load Balancer is healthy
- [ ] App loads at the ALB DNS name
- [ ] Health check endpoint responds at `/healthz`
- [ ] API keys work correctly
- [ ] Streamlit interface is functional
- [ ] File upload/download works
- [ ] Plots generate correctly
- [ ] Billing alerts are configured

---

## 📞 Support

If you encounter issues:

1. **Check GitHub Actions logs** for build errors
2. **Check ECS service events** for runtime errors
3. **Check CloudWatch logs** for application logs
4. **Verify all secrets** are correctly set
5. **Test locally** with Docker first
6. **Check AWS service limits** in your region

## 🎉 Success!

Once everything is set up, you'll have:
- ✅ Automatic deployments on every push to main
- ✅ Scalable, containerized hosting on AWS Fargate
- ✅ Load balanced with Application Load Balancer
- ✅ Secure environment variable management
- ✅ Professional CI/CD pipeline
- ✅ CloudWatch monitoring and logging
- ✅ Cost-effective hosting for low traffic applications

Your Lobster AI Streamlit app is now ready for production on AWS Fargate! 🦞

---

## 🆚 Migration from App Runner

If you're migrating from App Runner:

1. **Run the setup script:** `bash scripts/setup-fargate.sh`
2. **Update GitHub secrets** with the new values (ECS_CLUSTER_NAME, ECS_SERVICE_NAME)
3. **Push to main** to trigger the new Fargate deployment
4. **Verify the new deployment** works correctly
5. **Clean up App Runner resources** once migration is complete:
   ```bash
   # Delete App Runner service
   aws apprunner delete-service --service-arn YOUR_APPRUNNER_SERVICE_ARN
   
   # Delete old IAM role
   aws iam detach-role-policy --role-name AppRunnerECRAccessRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
   aws iam delete-role --role-name AppRunnerECRAccessRole
   ```

*Need help? Check the AWS ECS documentation or create an issue in the GitHub repository.*
