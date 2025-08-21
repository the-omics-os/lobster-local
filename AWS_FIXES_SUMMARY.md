# AWS App Runner Deployment Fixes Summary

## Issues Fixed

### 1. **Trust Policy Error (Critical Fix)**
**File:** `scripts/apprunner-trust-policy.json`
- **Issue:** Trust policy had wrong service principal `build.apprunner.amazonaws.com`
- **Fix:** Changed to `tasks.apprunner.amazonaws.com` for ECR-based deployments
- **Why:** App Runner needs the correct service principal to assume the role for accessing ECR images

### 2. **GitHub Actions Workflow Issues**
**File:** `.github/workflows/deploy-to-aws.yml`

#### Fixed deprecated syntax:
- **Old:** `uses: aws-actions/configure-aws-credentials@v1`
- **New:** `uses: aws-actions/configure-aws-credentials@v4`

#### Fixed deprecated output syntax:
- **Old:** `echo "::set-output name=image::$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG"`
- **New:** `echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT`

#### Fixed App Runner deploy parameters:
- **Removed:** `runtime: PYTHON_3` (not needed for image-based deployments)
- **Changed:** `wait-for-service-stability: true` to `wait-for-service-stability-seconds: 1200`

### 3. **Setup Script Path Issues**
**File:** `scripts/setup-aws.sh`
- **Fixed:** File paths to reference `scripts/` directory correctly
- **Changed:** `file://github-actions-policy.json` to `file://scripts/github-actions-policy.json`
- **Changed:** `file://apprunner-trust-policy.json` to `file://scripts/apprunner-trust-policy.json`

### 4. **IAM Policy Configuration**
**File:** `scripts/github-actions-policy.json`
- **Status:** Already correctly configured with all necessary permissions
- **Includes:** PassRole permission with correct condition for App Runner service

## Required Actions

### 1. Re-run Setup Script
If you've already created AWS resources, you need to update the IAM role with the new trust policy:
```bash
# From project root directory
bash scripts/setup-aws.sh
```

### 2. Verify GitHub Secrets
Ensure these secrets are set in your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (should be `us-east-2`)
- `ROLE_ARN` (format: `arn:aws:iam::ACCOUNT_ID:role/AppRunnerECRAccessRole`)

### 3. Manual Role Update (Alternative)
If the setup script doesn't update the role, manually update it:
```bash
# Delete old role
aws iam detach-role-policy --role-name AppRunnerECRAccessRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
aws iam delete-role --role-name AppRunnerECRAccessRole

# Create new role with correct trust policy
aws iam create-role --role-name AppRunnerECRAccessRole --assume-role-policy-document file://scripts/apprunner-trust-policy.json
aws iam attach-role-policy --role-name AppRunnerECRAccessRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
```

## Testing Deployment

After applying fixes:
1. Commit and push changes to trigger workflow:
   ```bash
   git add .
   git commit -m "Fix AWS App Runner deployment configuration"
   git push origin main
   ```

2. Monitor the GitHub Actions workflow at:
   `https://github.com/YOUR_USERNAME/lobster/actions`

## Common Error Resolutions

### If "iam:PassRole" error persists:
1. Verify the IAM user has the updated policy attached
2. Ensure ROLE_ARN secret matches the actual role ARN
3. Check that the role's trust policy allows `tasks.apprunner.amazonaws.com`

### If ECR push fails:
1. Verify ECR repository exists in the correct region (us-east-2)
2. Check IAM user has ECR permissions

### If App Runner service creation fails:
1. Ensure service-linked role exists: `AWSServiceRoleForAppRunner`
2. Verify the image URL is correct in the workflow output

## Key Changes Summary

| Component | Before | After |
|-----------|--------|-------|
| Trust Policy Service | `build.apprunner.amazonaws.com` | `tasks.apprunner.amazonaws.com` |
| AWS CLI Action | v1 | v4 |
| Output Syntax | `::set-output` | `$GITHUB_OUTPUT` |
| Wait Parameter | `wait-for-service-stability: true` | `wait-for-service-stability-seconds: 1200` |
| Runtime Parameter | `runtime: PYTHON_3` | (removed - not needed for images) |
