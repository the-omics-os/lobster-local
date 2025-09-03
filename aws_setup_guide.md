# AWS Setup Guide - Lobster Cloud Backend

This guide provides step-by-step instructions for deploying the Lobster Cloud backend to AWS for business validation testing.

## Prerequisites

- AWS CLI installed and configured
- AWS account with Lambda and API Gateway permissions
- Python 3.11 installed locally
- `zip` command available

## Quick Start (5-minute setup)

```bash
# 1. Clone and prepare the deployment
git clone <your-repo>
cd lobster

# 2. Run the automated deployment script
./deploy_to_aws.sh

# 3. Test the deployment
python test_aws_deployment.py
```

## Manual Setup Instructions

### Step 1: Create Lambda Function

1. **Go to AWS Lambda Console**
   - Open [AWS Lambda Console](https://console.aws.amazon.com/lambda/)
   - Click "Create function"

2. **Function Configuration**
   - Choose "Author from scratch"
   - Function name: `lobster-api`
   - Runtime: `Python 3.11`
   - Architecture: `x86_64`
   - Click "Create function"

3. **Basic Settings**
   - Timeout: `5 minutes` (300 seconds)
   - Memory: `512 MB` (start with this, can increase if needed)
   - Ephemeral storage: `512 MB`

### Step 2: Upload Lambda Code

1. **Prepare the deployment package**
   ```bash
   # Run the deployment script (it creates lambda-deployment.zip)
   ./deploy_to_aws.sh --build-only
   ```

2. **Upload to Lambda**
   - In Lambda console, go to "Code" tab
   - Click "Upload from" → ".zip file"
   - Upload `lambda-deployment.zip`
   - Click "Save"

3. **Set Handler**
   - Handler: `lambda_function.lambda_handler`

### Step 3: Create API Gateway

1. **Go to API Gateway Console**
   - Open [API Gateway Console](https://console.aws.amazon.com/apigateway/)
   - Click "Create API"
   - Choose "REST API" → "Build"

2. **API Configuration**
   - API name: `lobster-cloud-api`
   - Description: `Lobster AI Cloud Backend`
   - Endpoint Type: `Regional`
   - Click "Create API"

### Step 4: Configure API Resources

1. **Create Query Resource**
   - Click "Actions" → "Create Resource"
   - Resource Name: `query`
   - Resource Path: `/query`
   - Enable CORS: ✅
   - Click "Create Resource"

2. **Create POST Method**
   - Select `/query` resource
   - Click "Actions" → "Create Method"
   - Choose `POST` → Click checkmark
   - Integration type: `Lambda Function`
   - Lambda Region: Choose your region
   - Lambda Function: `lobster-api`
   - Click "Save"
   - Click "OK" to give API Gateway permission

3. **Enable CORS**
   - Select `/query` resource
   - Click "Actions" → "Enable CORS"
   - Access-Control-Allow-Origin: `*`
   - Access-Control-Allow-Headers: `Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token`
   - Access-Control-Allow-Methods: `GET,POST,OPTIONS`
   - Click "Enable CORS and replace existing CORS headers"

### Step 5: Setup API Keys

1. **Create Usage Plan**
   - Go to "Usage Plans" in left sidebar
   - Click "Create"
   - Name: `lobster-test-plan`
   - Description: `Test usage plan for Lobster Cloud`
   - Throttle: 100 requests per second
   - Burst: 200
   - Quota: 10,000 requests per month
   - Click "Next"

2. **Associate API Stage**
   - Add API Stage
   - API: `lobster-cloud-api`
   - Stage: `prod` (we'll create this next)
   - Click "Next" → "Done"

3. **Create API Keys**
   - Go to "API Keys" in left sidebar
   - Click "Create API Key"
   
   **Create these test keys:**
   
   **Enterprise Test Key 1:**
   - Name: `test-enterprise-001`
   - Auto Generate: ✅
   - Click "Save"
   - Copy the generated key

   **Enterprise Test Key 2:**
   - Name: `test-enterprise-002`
   - Auto Generate: ✅
   - Click "Save"
   - Copy the generated key

   **Demo Key:**
   - Name: `demo-user-001`
   - Auto Generate: ✅
   - Click "Save"
   - Copy the generated key

4. **Associate Keys with Usage Plan**
   - Go back to "Usage Plans"
   - Select `lobster-test-plan`
   - Go to "API Keys" tab
   - Click "Add API Key to Usage Plan"
   - Add all three keys created above

### Step 6: Deploy API

1. **Deploy to Stage**
   - Go back to your API resources
   - Click "Actions" → "Deploy API"
   - Deployment stage: `[New Stage]`
   - Stage name: `prod`
   - Stage description: `Production stage for Lobster Cloud`
   - Click "Deploy"

2. **Get API Endpoint**
   - Note the "Invoke URL" (e.g., `https://abc123.execute-api.us-east-1.amazonaws.com/prod`)
   - This is your `LOBSTER_ENDPOINT`

## Testing the Deployment

### Test 1: Status Check

```bash
# Replace with your actual API endpoint
export LOBSTER_ENDPOINT="https://your-api-id.execute-api.us-east-1.amazonaws.com/prod"
export LOBSTER_CLOUD_KEY="test-enterprise-001"

# Test status endpoint
curl -X POST \
  -H "Authorization: Bearer $LOBSTER_CLOUD_KEY" \
  -H "Content-Type: application/json" \
  "$LOBSTER_ENDPOINT/status"
```

Expected response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "environment": "aws-lambda",
  "success": true
}
```

### Test 2: Query Processing

```bash
# Test a simple query
curl -X POST \
  -H "Authorization: Bearer $LOBSTER_CLOUD_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RNA-seq?"}' \
  "$LOBSTER_ENDPOINT/query"
```

### Test 3: CLI Integration

```bash
# Test with the Lobster CLI
LOBSTER_CLOUD_KEY="test-enterprise-001" \
LOBSTER_ENDPOINT="https://your-api-id.execute-api.us-east-1.amazonaws.com/prod" \
lobster query "What is RNA-seq?"
```

Expected output:
```
🌩️ Using Lobster Cloud
[Response from cloud processing...]
```

## Troubleshooting

### Common Issues

1. **Lambda Timeout**
   - Increase timeout to 5 minutes in Lambda settings
   - Consider increasing memory if processing is slow

2. **CORS Errors**
   - Ensure CORS is enabled on API Gateway
   - Check that all required headers are allowed

3. **API Key Issues**
   - Verify API key is correctly associated with usage plan
   - Check that usage plan is associated with API stage

4. **Lambda Errors**
   - Check CloudWatch logs: AWS Console → CloudWatch → Log groups → `/aws/lambda/lobster-api`
   - Look for import errors or missing dependencies

5. **Package Size Issues**
   - Lambda has a 250MB unzipped limit
   - Remove unnecessary dependencies
   - Use Lambda layers for large dependencies

### Debugging Commands

```bash
# Check Lambda logs
aws logs describe-log-streams --log-group-name "/aws/lambda/lobster-api"

# Get recent Lambda errors
aws logs filter-log-events \
  --log-group-name "/aws/lambda/lobster-api" \
  --start-time $(date -d "1 hour ago" +%s)000 \
  --filter-pattern "ERROR"

# Test API Gateway directly
aws apigateway test-invoke-method \
  --rest-api-id YOUR_API_ID \
  --resource-id YOUR_RESOURCE_ID \
  --http-method POST \
  --body '{"query": "test"}'
```

## Production Considerations

### Security
- Replace hardcoded API keys with AWS Secrets Manager
- Enable AWS WAF for DDoS protection
- Set up VPC for Lambda if needed
- Enable API Gateway access logs

### Monitoring
- Set up CloudWatch alarms for error rates
- Monitor Lambda duration and memory usage
- Track API Gateway 4xx/5xx errors
- Set up billing alerts

### Performance
- Consider using Lambda Provisioned Concurrency for consistent performance
- Implement caching with ElastiCache if needed
- Use Lambda layers for common dependencies
- Optimize cold start times

### Scaling
- API Gateway handles auto-scaling
- Lambda concurrent execution limit: 1000 (default)
- Monitor and adjust based on usage patterns

## API Reference

### Endpoints

- `POST /query` - Process a query
- `POST /status` - Health check
- `POST /usage` - Get usage statistics
- `POST /models` - List available models

### Authentication

All requests require an `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY
```

### Request Format

```json
{
  "query": "Your question here",
  "options": {
    "workspace": "optional",
    "reasoning": true
  }
}
```

### Response Format

```json
{
  "response": "Generated response",
  "success": true,
  "cloud_processed": true,
  "user_tier": "enterprise"
}
```

## Cost Estimation

### AWS Costs (Monthly, approximate)

- **Lambda**: $0.20 per 1M requests + $0.0000166667 per GB-second
- **API Gateway**: $3.50 per 1M requests
- **CloudWatch Logs**: $0.50 per GB stored

**Example for 10,000 queries/month:**
- Lambda (512MB, 30s avg): ~$25
- API Gateway: ~$0.035
- CloudWatch: ~$2
- **Total**: ~$27/month

Scale linearly with usage. Monitor actual costs in AWS Cost Explorer.

## Next Steps

1. **Deploy and test** - Get the basic system working
2. **Add monitoring** - Set up CloudWatch dashboards
3. **Implement usage tracking** - Add DynamoDB for analytics
4. **Security hardening** - Replace hardcoded keys
5. **Performance optimization** - Based on real usage patterns

For immediate business validation, the current setup is sufficient. Focus on getting users testing the system rather than perfecting the architecture.
