# 🦞 Lobster AI - AWS Deployment Guide

Complete step-by-step guide to deploy your Streamlit app to AWS App Runner with automatic CI/CD from GitHub.

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
- **AWS Access Key ID**: `<>`
- **AWS Secret Access Key**: `<>`
- **Default region**: `us-east-2`
- **Default output format**: `json`

### 1.1.1 Activate STS for Required Regions

⚠️ **Important:** You need to activate AWS STS (Security Token Service) for the region you're deploying to. By default, only `us-east-1` is activated.

1. Go to AWS Console → **IAM** → **Account settings**
2. Scroll down to **Security Token Service (STS)**
3. Find **us-east-2 (US East - Ohio)** and click **Activate**
4. Wait a few minutes for activation to complete

Without this, you may encounter "STS is not activated in this region" errors during deployment.

### 1.2 Create IAM User for GitHub Actions

```bash
# Create IAM user for GitHub Actions
aws iam create-user --user-name github-actions-lobster

# Create access key
aws iam create-access-key --user-name github-actions-lobster
```

**📝 Save the Access Key ID and Secret Access Key - you'll need them for GitHub Secrets!**

### 1.6 Automated Setup (Recommended)

🚀 **Easy Option:** Use the automated setup script instead of manual steps:

```bash
# Run the automated setup script from project root
bash scripts/setup-aws.sh
```

This script will:
- ✅ Create IAM user and policies
- ✅ Generate access keys  
- ✅ Create ECR repository
- ✅ Set up App Runner role with correct trust policy
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
# Test run locally
docker run -p 8501:8501 --env-file .env lobster-ai:test

# Visit http://localhost:8501 to test
```

### 3.2 Deploy to AWS

1. **Push to main branch:**
```bash
git add .
git commit -m "Add AWS deployment configuration"
git push origin main
```

2. **Monitor deployment:**
- Go to GitHub **Actions** tab
- Watch the "Deploy Lobster AI to AWS App Runner" workflow
- Check for any errors in the logs

3. **Get your app URL:**
There are several ways to find your deployed app URL:

**Option A: AWS Console (Most Reliable)**
1. Go to AWS Console: https://console.aws.amazon.com/apprunner/
2. Make sure you're in the `us-east-2` region (Ohio)
3. Look for your service named `lobster-streamlit`
4. Click on the service name
5. The **Service URL** will be displayed at the top

**Option B: GitHub Actions Output**
- Check the workflow logs in the "App Runner output" step
- Look for the service URL (if displayed)

**Option C: AWS CLI**
```bash
aws apprunner describe-service \
    --service-arn $(aws apprunner list-services --query 'ServiceSummaryList[?ServiceName==`lobster-streamlit`].ServiceArn' --output text --region us-east-2) \
    --query 'Service.ServiceUrl' \
    --output text \
    --region us-east-2
```

Your app will be available at: `https://[random-id].us-east-2.awsapprunner.com`

---

## 💰 Part 4: Cost Management

### 4.1 Expected Costs

| Service | Configuration | Monthly Cost (Estimate) |
|---------|---------------|------------------------|
| **ECR** | < 1GB storage | ~$0.10 |
| **App Runner** | 1 vCPU, 2GB RAM, minimal traffic | ~$5-10 |
| **Data Transfer** | Minimal usage | ~$1 |
| **Total** | | **~$6-11/month** |

### 4.2 Cost Optimization Tips

```bash
# Set up billing alerts
aws budgets create-budget \
    --account-id $(aws sts get-caller-identity --query Account --output text) \
    --budget '{
        "BudgetName": "LobsterAI-Monthly",
        "BudgetLimit": {
            "Amount": "20",
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

### 5.1 App Runner Management

```bash
# Check service status
aws apprunner describe-service \
    --service-arn arn:aws:apprunner:us-east-2:ACCOUNT:service/lobster-ai-streamlit

# View service logs
aws logs get-log-events \
    --log-group-name /aws/apprunner/lobster-ai-streamlit \
    --log-stream-name application

# Scale service (if needed)
aws apprunner update-service \
    --service-arn arn:aws:apprunner:us-east-2:ACCOUNT:service/lobster-ai-streamlit \
    --instance-configuration Cpu="2 vCPU",Memory="4 GB"
```

### 5.2 Monitoring Setup

```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "LobsterAI-Monitoring" \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/AppRunner", "RequestCount", "ServiceName", "lobster-ai-streamlit"],
                        ["AWS/AppRunner", "ResponseTime", "ServiceName", "lobster-ai-streamlit"]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-2",
                    "title": "App Runner Metrics"
                }
            }
        ]
    }'
```

---

## 🚨 Part 6: Troubleshooting

### 6.1 Common Issues

**❌ Build Fails - "Permission Denied"**
```bash
# Check IAM permissions
aws iam list-attached-user-policies --user-name github-actions-lobster
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::ACCOUNT:user/github-actions-lobster \
    --action-names ecr:GetAuthorizationToken
```

**❌ App Runner Service Won't Start**
```bash
# Check service events
aws apprunner describe-service \
    --service-arn arn:aws:apprunner:us-east-2:ACCOUNT:service/lobster-ai-streamlit \
    --query 'Service.ServiceStatus'

# Check logs
aws logs describe-log-groups \
    --log-group-name-prefix /aws/apprunner/lobster-ai-streamlit
```

**❌ Environment Variables Not Working**
```bash
# Verify secrets in GitHub
curl -H "Authorization: token YOUR_GITHUB_TOKEN" \
    https://api.github.com/repos/YOUR_USERNAME/lobster/actions/secrets

# Check App Runner environment variables
aws apprunner describe-service \
    --service-arn arn:aws:apprunner:us-east-2:ACCOUNT:service/lobster-ai-streamlit \
    --query 'Service.SourceConfiguration.ImageRepository.ImageConfiguration.RuntimeEnvironmentVariables'
```

### 6.2 Debug Commands

```bash
# Test ECR login
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-2.amazonaws.com

# List ECR images
aws ecr list-images --repository-name lobster-ai --region us-east-2

# Check App Runner service logs
aws logs tail /aws/apprunner/lobster-ai-streamlit/application --follow
```

---

## 🎯 Part 7: Advanced Configuration

### 7.1 Custom Domain (Optional)

```bash
# Create custom domain
aws apprunner associate-custom-domain \
    --service-arn arn:aws:apprunner:us-east-2:ACCOUNT:service/lobster-ai-streamlit \
    --domain-name lobster.yourdomain.com \
    --enable-www-subdomain
```

### 7.2 Auto Scaling Configuration

```bash
# Update auto scaling
aws apprunner update-service \
    --service-arn arn:aws:apprunner:us-east-2:ACCOUNT:service/lobster-ai-streamlit \
    --auto-scaling-configuration-arn arn:aws:apprunner:us-east-2:ACCOUNT:autoscalingconfiguration/high-availability
```

---

## ✅ Part 8: Verification Checklist

Before going live, verify:

- [ ] ECR repository created and accessible
- [ ] GitHub Actions workflow runs successfully
- [ ] App Runner service is running
- [ ] All environment variables are set correctly
- [ ] App loads at the provided URL
- [ ] Health check endpoint responds
- [ ] API keys work correctly
- [ ] Streamlit interface is functional
- [ ] File upload/download works
- [ ] Plots generate correctly
- [ ] Billing alerts are configured

---

## 📞 Support

If you encounter issues:

1. **Check GitHub Actions logs** for build errors
2. **Check App Runner logs** for runtime errors
3. **Verify all secrets** are correctly set
4. **Test locally** with Docker first
5. **Check AWS service limits** in your region

## 🎉 Success!

Once everything is set up, you'll have:
- ✅ Automatic deployments on every push to main
- ✅ Scalable, managed hosting on AWS
- ✅ Secure environment variable management
- ✅ Professional CI/CD pipeline
- ✅ Cost-effective hosting (~$6-11/month)

Your Lobster AI Streamlit app is now ready for production! 🦞

---

*Need help? Check the AWS App Runner documentation or create an issue in the GitHub repository.*
