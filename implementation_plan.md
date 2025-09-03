# Lobster Cloud/Local Split - Fast Implementation Plan

## Overview
Minimal viable cloud split for validation. Focus on speed to market with API key authentication (no user registration).

## Phase 1: Package Structure (Day 1-2)

### 1.1 Create Three Packages

```
lobster/                    # Current monorepo
├── lobster-core/          # NEW: Shared interfaces
├── lobster-local/         # NEW: Local implementation  
├── lobster-cloud/         # NEW: Cloud client
└── lobster-server/        # NEW: AWS backend (private)
```

### 1.2 Move Files to lobster-core/

Create `lobster-core/` with minimal shared code:

```bash
mkdir -p lobster-core/lobster_core/{interfaces,schemas,utils}

# Move these files:
cp lobster/core/interfaces/*.py lobster-core/lobster_core/interfaces/
cp lobster/core/schemas/*.py lobster-core/lobster_core/schemas/
cp lobster/utils/logger.py lobster-core/lobster_core/utils/
cp lobster/agents/state.py lobster-core/lobster_core/interfaces/
```

### 1.3 Create Abstract Base Classes

Create `lobster-core/lobster_core/interfaces/base_client.py`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseLobsterClient(ABC):
    @abstractmethod
    def query(self, user_input: str, **kwargs) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass

class BaseDataManager(ABC):
    @abstractmethod
    def has_data(self) -> bool:
        pass
    
    @abstractmethod
    def load_modality(self, name: str, source: Any, adapter: str, **kwargs) -> Any:
        pass
```

### 1.4 Create lobster-local Package

Copy current implementation with minimal changes:

```bash
mkdir -p lobster-local/lobster_local
cp -r lobster/* lobster-local/lobster_local/

# Update imports in all files:
# sed -i 's/from lobster\./from lobster_local\./g' lobster-local/lobster_local/**/*.py
```

### 1.5 Create lobster-cloud Package

Minimal cloud client:

```bash
mkdir -p lobster-cloud/lobster_cloud
```

Create `lobster-cloud/lobster_cloud/client.py`:

```python
import requests
from typing import Dict, Any
from lobster_core.interfaces.base_client import BaseLobsterClient

class CloudLobsterClient(BaseLobsterClient):
    def __init__(self, api_key: str, endpoint: str = "https://api.lobster.homara.ai"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def query(self, user_input: str, **kwargs) -> Dict[str, Any]:
        response = self.session.post(
            f"{self.endpoint}/query",
            json={"query": user_input, "options": kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.endpoint}/status")
        response.raise_for_status()
        return response.json()
```

## Phase 2: Unified CLI Entry Point (Day 3)

### 2.1 Create Smart CLI Router

Update `lobster/cli.py` to detect mode:

```python
def init_client(workspace: Optional[Path] = None, reasoning: bool = False, debug: bool = False) -> AgentClient:
    """Initialize either local or cloud client based on environment."""
    
    # Check for cloud API key
    cloud_key = os.environ.get('LOBSTER_CLOUD_KEY')
    
    if cloud_key:
        # Use cloud client
        from lobster_cloud.client import CloudLobsterClient
        console.print("[bold red]🌩️  Using Lobster Cloud[/bold red]")
        return CloudLobsterClient(api_key=cloud_key)
    else:
        # Use local client (existing code)
        from lobster_local.core.client import AgentClient
        console.print("[bold red]💻 Using Lobster Local[/bold red]")
        # ... existing initialization code ...
```

### 2.2 Package Setup Files

Create `lobster-core/setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="lobster-core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "anndata>=0.8.0",
    ]
)
```

Create `lobster-local/setup.py`:

```python
setup(
    name="lobster",  # Keep same name for compatibility
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "lobster-core>=0.1.0",
        "langchain>=0.1.0",
        "langgraph>=0.0.20",
        # ... all current dependencies ...
    ],
    entry_points={
        "console_scripts": [
            "lobster=lobster_local.cli:app",
        ],
    },
)
```

Create `lobster-cloud/setup.py`:

```python
setup(
    name="lobster-cloud",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "lobster-core>=0.1.0",
        "requests>=2.28.0",
    ]
)
```

## Phase 3: Minimal AWS Setup (Day 4-5)

### 3.1 Simple Lambda + API Gateway

**AWS Setup Steps:**

1. **Create Lambda Function**
```bash
# In AWS Console:
1. Go to Lambda > Create function
2. Name: lobster-api
3. Runtime: Python 3.11
4. Architecture: x86_64
5. Create function
```

2. **Create API Gateway**
```bash
# In AWS Console:
1. Go to API Gateway > Create API
2. Choose: REST API
3. Name: lobster-api
4. Create API
5. Create Resource: /query
6. Create Method: POST
7. Integration: Lambda Function > lobster-api
8. Deploy API > New Stage > "prod"
```

3. **Set up API Keys (Quick hack for testing)**
```bash
# In API Gateway:
1. API Keys > Create API Key
2. Name: test-enterprise-001
3. Copy the key
4. Usage Plans > Create
5. Name: enterprise-plan
6. Add API Stage > lobster-api/prod
7. Add API Key > test-enterprise-001
```

### 3.2 Simple Lambda Code

Create `lobster-server/lambda_function.py`:

```python
import json
import os
from lobster_local.core.client import AgentClient
from lobster_local.core.data_manager_v2 import DataManagerV2

# Initialize once per container (cold start optimization)
data_manager = DataManagerV2(workspace_path="/tmp/workspace")
client = AgentClient(data_manager=data_manager)

def lambda_handler(event, context):
    try:
        # Simple API key validation (hack for testing)
        api_key = event['headers'].get('Authorization', '').replace('Bearer ', '')
        
        # Hardcoded valid keys for testing
        VALID_KEYS = {
            'test-enterprise-001': {'tier': 'enterprise', 'name': 'Test User 1'},
            'test-enterprise-002': {'tier': 'enterprise', 'name': 'Test User 2'},
        }
        
        if api_key not in VALID_KEYS:
            return {
                'statusCode': 401,
                'body': json.dumps({'error': 'Invalid API key'})
            }
        
        # Parse request
        body = json.loads(event['body'])
        query = body.get('query', '')
        
        # Process query
        result = client.query(query)
        
        return {
            'statusCode': 200,
            'body': json.dumps(result),
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            }
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### 3.3 Deploy Lambda

Create deployment package:

```bash
# Create deployment directory
mkdir lambda-deploy
cd lambda-deploy

# Install dependencies
pip install -t . lobster-local langchain langgraph pandas numpy

# Copy lambda function
cp ../lobster-server/lambda_function.py .

# Create zip
zip -r9 ../lambda-deploy.zip .

# Upload to Lambda (in AWS Console)
# Function code > Upload from > .zip file
```

## Phase 4: Testing Setup (Day 6)

### 4.1 Local Testing Script

Create `test_cloud_local.py`:

```python
#!/usr/bin/env python3
"""Test both local and cloud versions"""

import os
import sys

def test_local():
    print("=== Testing Local Version ===")
    os.environ.pop('LOBSTER_CLOUD_KEY', None)
    os.system('lobster query "What is RNA-seq?"')

def test_cloud():
    print("\n=== Testing Cloud Version ===")
    os.environ['LOBSTER_CLOUD_KEY'] = 'test-enterprise-001'
    os.system('lobster query "What is RNA-seq?"')

if __name__ == "__main__":
    test_local()
    test_cloud()
```

### 4.2 Development Install Script

Create `dev_install.sh`:

```bash
#!/bin/bash
# Install all packages in development mode

# Install core
cd lobster-core
pip install -e .
cd ..

# Install local (includes CLI)
cd lobster-local
pip install -e .
cd ..

# Install cloud client
cd lobster-cloud
pip install -e .
cd ..

echo "✅ Development installation complete!"
echo "Test with: python test_cloud_local.py"
```

## Phase 5: Quick Hacks for Fast Testing

### 5.1 Bypass Heavy Dependencies in Cloud

For Lambda, create `lobster-server/mock_agents.py`:

```python
"""Mock agents for cloud testing - replace with real implementation later"""

class MockSupervisor:
    def query(self, text):
        return {"response": f"Cloud processed: {text}", "success": True}

# In lambda_function.py, use mock for now:
# from mock_agents import MockSupervisor as Supervisor
```

### 5.2 Simple Usage Tracking

Add to `lambda_function.py`:

```python
# Simple DynamoDB usage tracking
import boto3
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('lobster-usage')

def track_usage(api_key, query_length):
    table.put_item(Item={
        'api_key': api_key,
        'timestamp': int(time.time()),
        'query_length': query_length,
        'cost': query_length * 0.0001  # Simple cost model
    })
```

### 5.3 Environment Variables for Testing

Create `.env.cloud`:

```bash
# For cloud testing
LOBSTER_CLOUD_KEY=test-enterprise-001
LOBSTER_ENDPOINT=https://your-api-id.execute-api.us-east-1.amazonaws.com/prod

# For local testing (comment out for local)
# LOBSTER_CLOUD_KEY=
```

## Phase 6: Deployment Commands

### 6.1 Quick Deploy Script

Create `deploy_to_aws.sh`:

```bash
#!/bin/bash
# Quick and dirty deployment

# Build lambda package
cd lobster-server
rm -rf package lambda-deploy.zip
pip install -t package lobster-local
cd package && zip -r9 ../lambda-deploy.zip . && cd ..
zip -g lambda-deploy.zip lambda_function.py

# Upload to Lambda (requires AWS CLI)
aws lambda update-function-code \
    --function-name lobster-api \
    --zip-file fileb://lambda-deploy.zip

echo "✅ Deployed to AWS Lambda!"
```

## Testing Checklist

1. **Local Installation Test**
   ```bash
   ./dev_install.sh
   lobster query "What is RNA-seq?"  # Should work locally
   ```

2. **Cloud Mode Test**
   ```bash
   export LOBSTER_CLOUD_KEY=test-enterprise-001
   lobster query "What is RNA-seq?"  # Should hit AWS
   ```

3. **Package Import Test**
   ```python
   # Should work after install
   from lobster_core.interfaces import BaseLobsterClient
   from lobster_local.core.client import AgentClient
   from lobster_cloud.client import CloudLobsterClient
   ```

## Next Steps for Production

1. **Add S3 for Large Data** - Store analysis results
2. **Add DynamoDB for State** - Track user sessions
3. **Add CloudWatch Logs** - Monitor usage
4. **Add Stripe Integration** - For real billing
5. **Add Docker for Lambda** - Better dependency management

## Estimated Timeline

- Day 1-2: Package separation
- Day 3: CLI integration  
- Day 4-5: AWS setup
- Day 6: Testing
- **Total: 6 days to MVP**

## Notes

- This plan prioritizes speed over perfection
- Security is minimal (just API keys)
- No user management system
- Perfect for validating demand before building full system
