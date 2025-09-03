# Lobster Cloud - Complete User Guide

A comprehensive guide for using Lobster's new Cloud/Local architecture for business validation testing.

## 🚀 Quick Start (5 Minutes)

### 1. Deploy to AWS

```bash
# Make scripts executable (if not already done)
chmod +x deploy_to_aws.sh
chmod +x test_aws_deployment.py

# Deploy Lambda function (requires AWS CLI configured)
./deploy_to_aws.sh

# The script will:
# - Build deployment package with all dependencies
# - Deploy Lambda function to AWS
# - Test the deployment automatically
```

### 2. Set Up API Gateway

Follow the detailed steps in [aws_setup_guide.md](aws_setup_guide.md):

1. Create API Gateway REST API
2. Configure `/query` POST endpoint
3. Set up API keys (test-enterprise-001, test-enterprise-002, demo-user-001)
4. Deploy to production stage

### 3. Test Your Deployment

```bash
# Set your endpoint (get this from API Gateway)
export LOBSTER_ENDPOINT="https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod"
export LOBSTER_CLOUD_KEY="test-enterprise-001"

# Run comprehensive tests
python test_aws_deployment.py --verbose

# Test with CLI
lobster query "What is RNA-seq?"
```

## 🌩️ Cloud vs Local Usage

### Automatic Mode Detection

Lobster automatically detects whether to use Cloud or Local mode based on environment variables:

```bash
# LOCAL MODE (default)
lobster query "What is RNA-seq?"
# Output: 💻 Using Lobster Local

# CLOUD MODE (when API key is set)
LOBSTER_CLOUD_KEY="test-enterprise-001" lobster query "What is RNA-seq?"
# Output: 🌩️ Using Lobster Cloud
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `LOBSTER_CLOUD_KEY` | API key for cloud access | `test-enterprise-001` |
| `LOBSTER_ENDPOINT` | Custom cloud endpoint | `https://api.yourcompany.com/prod` |

## 🔑 API Keys for Testing

The system comes with pre-configured test keys for immediate business validation:

| API Key | Tier | Max Queries/Hour | Purpose |
|---------|------|------------------|---------|
| `test-enterprise-001` | Enterprise | 100 | Primary testing |
| `test-enterprise-002` | Enterprise | 100 | Load testing |
| `demo-user-001` | Demo | 10 | Demo/trial users |

## 🛠️ Advanced Usage

### CLI Enhancements

The CLI now includes enhanced cloud detection with retry logic:

```bash
# Automatic retry on connection failures
LOBSTER_CLOUD_KEY="test-key" lobster chat

# Enhanced error messages with troubleshooting tips
# - Connection timeout suggestions
# - Authentication failure guidance
# - Endpoint validation help
```

### Cloud Client Features

```python
from lobster_cloud.client import CloudLobsterClient

# Initialize with endpoint support
client = CloudLobsterClient(
    api_key="test-enterprise-001",
    endpoint="https://your-api.amazonaws.com/prod"
)

# Test connection
status = client.get_status()
if status.get("success"):
    print("✅ Connected to cloud")

# Process queries
result = client.query("What is single-cell RNA-seq?")
```

### Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/query` | POST | Process bioinformatics queries |
| `/status` | POST | Health check and version info |
| `/usage` | POST | Get API key usage statistics |
| `/models` | POST | List available models |

## 📊 Testing & Validation

### Comprehensive Test Suite

```bash
# Run all tests with detailed output
python test_aws_deployment.py --verbose

# Test with custom endpoint
python test_aws_deployment.py --endpoint https://your-api.com/prod

# Test with multiple API keys
python test_aws_deployment.py --api-key test-enterprise-001 --api-key test-enterprise-002
```

### Test Categories

The test suite covers:

1. **Status Endpoint** - Health checks and connectivity
2. **API Key Validation** - Authentication testing
3. **Query Processing** - Real bioinformatics queries
4. **Usage Tracking** - API usage monitoring
5. **CORS Support** - Browser compatibility
6. **Error Handling** - Edge case management
7. **Performance** - Response time validation

### Business Validation Scenarios

```bash
# Test 1: Basic functionality
LOBSTER_CLOUD_KEY="test-enterprise-001" lobster query "What is RNA-seq?"

# Test 2: Complex analysis
LOBSTER_CLOUD_KEY="test-enterprise-001" lobster query "How do I analyze single-cell data?"

# Test 3: Data upload workflow
LOBSTER_CLOUD_KEY="test-enterprise-001" lobster chat
# Then upload a dataset and analyze

# Test 4: Performance comparison
time lobster query "Explain differential gene expression"  # Local
time LOBSTER_CLOUD_KEY="test-enterprise-001" lobster query "Explain differential gene expression"  # Cloud
```

## 🔧 Troubleshooting

### Common Issues

**1. Cloud Connection Fails**
```bash
# Check your API key
echo $LOBSTER_CLOUD_KEY

# Test endpoint manually
curl -X POST -H "Authorization: Bearer test-enterprise-001" \
  -H "Content-Type: application/json" \
  "$LOBSTER_ENDPOINT/status"
```

**2. Lambda Deployment Issues**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Rebuild deployment package
./deploy_to_aws.sh --build-only

# Check Lambda logs
aws logs tail /aws/lambda/lobster-api
```

**3. API Gateway Problems**
- Verify API key is associated with usage plan
- Check CORS is enabled for all methods
- Ensure deployment was successful

### Debug Commands

```bash
# Test Lambda function directly
aws lambda invoke --function-name lobster-api \
  --payload '{"httpMethod":"POST","path":"/status","headers":{"Authorization":"Bearer test-enterprise-001"},"body":"{}"}' \
  response.json

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix "API-Gateway-Execution-Logs"
```

## 💰 Cost Estimation

### AWS Costs (Monthly)

For **10,000 queries/month** (typical validation scale):

- **Lambda (512MB, 30s avg)**: ~$25/month
- **API Gateway**: ~$35/month  
- **CloudWatch Logs**: ~$2/month
- **Total**: **~$62/month**

### Scaling Costs

| Monthly Queries | Estimated Cost |
|----------------|---------------|
| 1,000 | $8 |
| 10,000 | $62 |
| 100,000 | $580 |
| 1,000,000 | $5,800 |

*Costs scale linearly with usage*

## 🎯 Business Model Validation

### Key Metrics to Track

1. **Usage Patterns**
   - Queries per user per day
   - Peak usage times
   - Query complexity distribution

2. **Performance Metrics**
   - Average response time
   - Success rate
   - Error types and frequency

3. **User Behavior**
   - Feature adoption
   - Session duration
   - Retention rates

### Alpha Testing Checklist

- [ ] Deploy Lambda function successfully
- [ ] Configure API Gateway with test keys
- [ ] Verify all endpoints working
- [ ] Test CLI cloud mode switching
- [ ] Run comprehensive test suite
- [ ] Measure performance vs local mode
- [ ] Set up usage monitoring
- [ ] Document user workflows
- [ ] Prepare feedback collection
- [ ] Plan scaling strategy

## 📈 Next Steps

### Phase 4: Production Hardening

1. **Security Enhancements**
   - Replace hardcoded API keys with AWS Secrets Manager
   - Implement proper user authentication
   - Add rate limiting per user

2. **Monitoring & Analytics**
   - Set up CloudWatch dashboards
   - Implement usage tracking with DynamoDB
   - Add cost monitoring alerts

3. **Performance Optimization**
   - Optimize Lambda cold starts
   - Implement response caching
   - Add Lambda provisioned concurrency

4. **Business Features**
   - User account management
   - Billing integration
   - Usage tier management

### Production Deployment

Once validation is complete:

1. **Replace Test Infrastructure**
   - Production API keys
   - Proper authentication system
   - User management

2. **Scale Infrastructure**
   - Multi-region deployment
   - Load balancing
   - Database integration

3. **Business Operations**
   - Payment processing
   - Customer support
   - Usage analytics

## 📞 Support

### Getting Help

1. **Check the guides**:
   - `aws_setup_guide.md` - Detailed AWS setup
   - `TROUBLESHOOTING.md` - Common issues (if created)

2. **Run diagnostics**:
   ```bash
   python test_aws_deployment.py --verbose
   ```

3. **Check logs**:
   ```bash
   aws logs tail /aws/lambda/lobster-api
   ```

### Files Overview

| File | Purpose |
|------|---------|
| `lobster-server/lambda_function.py` | AWS Lambda backend |
| `deploy_to_aws.sh` | Automated deployment |
| `test_aws_deployment.py` | Comprehensive testing |
| `aws_setup_guide.md` | Step-by-step AWS setup |
| `CLOUD_USAGE_GUIDE.md` | This guide |

---

## 🎉 Success Criteria

You've successfully implemented Lobster Cloud when:

- ✅ CLI automatically switches between local/cloud modes
- ✅ AWS Lambda responds to all endpoints
- ✅ API Gateway authentication works
- ✅ Test suite passes completely
- ✅ Real bioinformatics queries process in cloud
- ✅ Performance is acceptable (< 60s per query)
- ✅ Ready for alpha user testing

**The system is now ready for business model validation!**

Focus on getting real users testing the system rather than perfecting the architecture. Speed to market is critical for business validation.
