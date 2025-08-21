# 🚀 Deploying Streamlit with AWS Lightsail Containers

This document explains how to deploy our Streamlit app to **Amazon Lightsail Containers** using Docker and GitHub Actions.

---

## 1. Prerequisites

- AWS account with Lightsail enabled
- GitHub repository containing the project (with `Dockerfile`)
- AWS CLI installed locally (for initial setup)
- Docker installed locally (if you want to test builds)

---

## 2. One-Time AWS Setup

### 2.1 Authenticate Docker with Amazon ECR

Before pushing images, you must authenticate Docker with ECR:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

Replace `<account-id>` with your AWS account ID.

### 2.2 Create an ECR Repository (only once)

```bash
aws ecr create-repository --repository-name homara
```

### 2.3 Create a Lightsail Container Service (only once)

```bash
aws lightsail create-container-service \
  --service-name streamlit-service \
  --power small \
  --scale 1 \
  --region us-east-1
```

This creates a managed container service that will run our Streamlit app.

---

## 3. GitHub Actions Workflow

We use GitHub Actions to automate:

1. Build Docker image
2. Push image to ECR
3. Deploy to Lightsail

The workflow file is `.github/workflows/deploy.yml`:

```yaml
name: Deploy to AWS Lightsail

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & Push Docker Image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: streamlit-app
          IMAGE_TAG: latest
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Deploy to Lightsail Container Service
        run: |
          aws lightsail create-container-service-deployment \
            --service-name streamlit-service \
            --containers "{\
              \"streamlit\": {\
                \"image\": \"${{ steps.login-ecr.outputs.registry }}/streamlit-app:latest\",\
                \"ports\": {\"8501\": \"HTTP\"}\
              }\
            }" \
            --public-endpoint "{\
              \"containerName\": \"streamlit\",\
              \"containerPort\": 8501,\
              \"healthCheck\": {\"path\": \"/\"}\
            }"
```

---

## 4. Secrets in GitHub

Add these secrets under **Repo → Settings → Secrets and variables → Actions**:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (example: `us-east-1`)

---

## 5. Deployment URL

After deployment, AWS will provide a public URL:

```
https://streamlit-service.<region>.lightsail.aws.amazonaws.com
```

This is the endpoint where the Streamlit app will be available.

---

## 6. What We Removed

Originally, the project used **ECS/Fargate** (with load balancers, task definitions, custom IAM roles, and shell scripts). That was overkill for our use case.

Now, with Lightsail:

- **Removed**: `setup-fargate.sh`, `fargate-iam-policies.json`, ECS task definitions, and ECS GitHub Actions steps.
- **Kept**: `Dockerfile` and GitHub Actions workflow.
- **Simplified**: One container service, automatically managed by AWS.

---

✅ With this setup, every push to `main` automatically builds, pushes, and deploys the Streamlit app with the least AWS complexity.

