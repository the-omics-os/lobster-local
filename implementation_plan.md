# Implementation Plan

Migrate Streamlit application from AWS App Runner to AWS Fargate with public access via Application Load Balancer.

This implementation transforms the current App Runner setup into a scalable, containerized solution using AWS ECS Fargate. The migration will maintain automatic deployments via GitHub Actions while providing better control over networking, scaling, and infrastructure configuration. Given the low traffic requirements (10-50 users/day), the setup prioritizes simplicity and cost-effectiveness over complex scaling strategies.

## [Types]

Define data structures for ECS task definitions and service configurations.

**ECS Task Definition Structure:**
```json
{
  "family": "lobster-streamlit-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [{
    "name": "lobster-streamlit",
    "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/homara:TAG",
    "portMappings": [{"containerPort": 8501, "protocol": "tcp"}],
    "environment": [...],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/lobster-streamlit",
        "awslogs-region": "us-east-2",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

**ALB Target Group Configuration:**
```json
{
  "Protocol": "HTTP",
  "Port": 8501,
  "VpcId": "vpc-default",
  "TargetType": "ip",
  "HealthCheckPath": "/healthz",
  "HealthCheckProtocol": "HTTP",
  "HealthCheckIntervalSeconds": 30,
  "HealthyThresholdCount": 2,
  "UnhealthyThresholdCount": 3
}
```

## [Files]

Modify existing files and create new infrastructure configuration files.

**New Files to Create:**
- `scripts/ecs-task-definition.json` - ECS task definition template
- `scripts/fargate-iam-policies.json` - Updated IAM policies for ECS/Fargate
- `scripts/ecs-trust-policy.json` - Trust policies for ECS task and execution roles
- `scripts/setup-fargate.sh` - Automated Fargate infrastructure setup script
- `AWS_FARGATE_DEPLOYMENT_GUIDE.md` - Updated deployment documentation

**Existing Files to Modify:**
- `.github/workflows/deploy-to-aws.yml` - Replace App Runner deployment with ECS Fargate deployment
- `scripts/github-actions-policy.json` - Add ECS permissions, remove App Runner permissions
- `scripts/setup-aws.sh` - Update to use Fargate setup script

**Files to Keep Unchanged:**
- `Dockerfile.streamlit` - Container definition remains the same
- `scripts/apprunner-trust-policy.json` - Keep for reference but won't be used

## [Functions]

Create new deployment functions and modify existing automation scripts.

**New Functions:**
- `create_ecs_cluster()` in setup-fargate.sh - Creates ECS cluster named "lobster-cluster"
- `create_task_definition()` in setup-fargate.sh - Registers ECS task definition from template
- `create_fargate_service()` in setup-fargate.sh - Creates Fargate service with ALB integration
- `setup_application_load_balancer()` in setup-fargate.sh - Creates ALB, target group, and listeners
- `create_security_groups()` in setup-fargate.sh - Creates security groups for ALB and ECS tasks
- `deploy_to_fargate()` in GitHub Actions - Updates ECS service with new task definition

**Modified Functions:**
- `create_iam_policies()` in setup-fargate.sh - Update to include ECS/Fargate permissions instead of App Runner
- `main()` in setup-fargate.sh - Replace App Runner setup flow with ECS/Fargate setup flow

**Removed Functions:**
- `create_app_runner_service()` - No longer needed
- `deploy_to_app_runner()` - Replaced with Fargate deployment

## [Classes]

No new classes required for this infrastructure migration.

The implementation uses AWS CLI commands and GitHub Actions workflows rather than object-oriented programming patterns. All infrastructure is managed through JSON configuration files and shell scripts.

## [Dependencies]

Update AWS service dependencies and GitHub Actions.

**New AWS Services:**
- Amazon ECS (Elastic Container Service)
- Application Load Balancer (ALB)
- VPC Security Groups
- CloudWatch Logs (for container logging)

**GitHub Actions Dependencies:**
- `aws-actions/amazon-ecs-deploy-task-definition@v1` - Deploy task definitions to ECS
- `aws-actions/amazon-ecs-render-task-definition@v1` - Render task definition templates

**AWS CLI Permissions Required:**
```json
{
  "ecs:CreateCluster", "ecs:DescribeClusters", "ecs:CreateService", 
  "ecs:UpdateService", "ecs:DescribeServices", "ecs:RegisterTaskDefinition",
  "elbv2:CreateLoadBalancer", "elbv2:CreateTargetGroup", "elbv2:CreateListener",
  "ec2:CreateSecurityGroup", "ec2:AuthorizeSecurityGroupIngress",
  "iam:CreateRole", "iam:AttachRolePolicy", "iam:PassRole"
}
```

**Removed Dependencies:**
- App Runner service APIs
- App Runner IAM policies

## [Testing]

Create validation strategy for Fargate deployment and service health.

**New Test Files:**
- `tests/test_fargate_deployment.py` - Validate ECS service health and ALB connectivity
- `scripts/validate-fargate-setup.sh` - Infrastructure validation script

**Testing Strategy:**
1. **Infrastructure Tests:** Verify ECS cluster, service, and ALB creation
2. **Health Check Tests:** Confirm container startup and health endpoint response
3. **Load Balancer Tests:** Validate ALB routing to healthy targets
4. **Environment Variable Tests:** Ensure all secrets are properly injected
5. **Deployment Pipeline Tests:** Test full CI/CD workflow from git push to live service

**Modified Test Approach:**
- Update existing integration tests to target ALB endpoint instead of App Runner URL
- Add ECS service status checks to deployment validation
- Include ALB health check validation in test suite

## [Implementation Order]

Sequential steps to minimize deployment conflicts and ensure successful migration.

1. **Create ECS Infrastructure Setup Script** - Build setup-fargate.sh with all AWS resource creation logic
2. **Update IAM Policies and Roles** - Create ECS-specific policies and trust relationships
3. **Create Task Definition Template** - Define containerized application configuration
4. **Set Up Network Infrastructure** - Create security groups and load balancer configuration
5. **Update GitHub Actions Workflow** - Replace App Runner deployment with ECS deployment steps
6. **Create Infrastructure Provisioning** - Run setup script to create all AWS resources
7. **Test Deployment Pipeline** - Validate end-to-end deployment from GitHub to live service
8. **Update Documentation** - Create new deployment guide and update existing docs
9. **Cleanup Old Resources** - Remove App Runner service and related resources after successful migration
