# Configuration Guide

This guide covers all aspects of configuring Lobster AI, from basic API key setup to advanced model customization and cloud integration.

## Table of Contents

- [Environment Variables](#environment-variables)
- [API Key Management](#api-key-management)
- [Model Profiles](#model-profiles)
- [Cloud vs Local Configuration](#cloud-vs-local-configuration)
- [Advanced Settings](#advanced-settings)
- [Configuration Management](#configuration-management)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting Configuration](#troubleshooting-configuration)

## Environment Variables

Lobster AI uses environment variables for configuration. These can be set in a `.env` file or as system environment variables.

### Required Variables

```env
# Essential API Keys (Required for core functionality)
OPENAI_API_KEY=your-openai-api-key-here
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key-here
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key-here
```

### Optional Variables

```env
# Enhanced Literature Search
NCBI_API_KEY=your-ncbi-api-key-here

# Model Configuration
GENIE_PROFILE=production                    # Model preset configuration
GENIE_SUPERVISOR_TEMPERATURE=0.5            # Supervisor agent temperature
GENIE_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7 # Analysis agent temperature
GENIE_METHOD_AGENT_TEMPERATURE=0.3          # Method agent temperature

# Application Settings
GENIE_MAX_FILE_SIZE_MB=500                  # Maximum file size for uploads
GENIE_CLUSTER_RESOLUTION=0.5                # Default clustering resolution
GENIE_CACHE_DIR=data/cache                  # Cache directory location

# Server Configuration (for web interface)
PORT=8501                                   # Web interface port
HOST=0.0.0.0                              # Web interface host
DEBUG=False                                 # Debug mode

# Observability (Optional)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com    # Optional: custom instance
```

### Cloud Configuration

```env
# Lobster Cloud Integration (Optional)
LOBSTER_CLOUD_KEY=your-cloud-api-key-here
LOBSTER_ENDPOINT=https://api.lobster.homara.ai  # Optional: defaults to production
```

### Supervisor Configuration (v2.3+)

The supervisor agent now features dynamic configuration for customizing interaction behavior:

```env
# Interaction Settings
SUPERVISOR_ASK_QUESTIONS=true               # Ask clarification questions (default: true)
SUPERVISOR_MAX_QUESTIONS=3                  # Max clarification questions (default: 3)
SUPERVISOR_REQUIRE_CONFIRMATION=true        # Require download confirmation (default: true)
SUPERVISOR_REQUIRE_PREVIEW=true             # Preview metadata before download (default: true)

# Response Settings
SUPERVISOR_AUTO_SUGGEST=true               # Suggest next steps (default: true)
SUPERVISOR_VERBOSE=false                   # Verbose delegation explanations (default: false)
SUPERVISOR_INCLUDE_EXPERT_OUTPUT=true      # Include full expert output (default: true)
SUPERVISOR_SUMMARIZE_OUTPUT=false          # Summarize expert output (default: false)

# Workflow Guidance
SUPERVISOR_WORKFLOW_GUIDANCE=standard      # minimal, standard, detailed (default: standard)
SUPERVISOR_DELEGATION_STRATEGY=auto        # auto, conservative, aggressive (default: auto)
SUPERVISOR_ERROR_HANDLING=informative      # silent, informative, verbose (default: informative)

# Context Settings
SUPERVISOR_INCLUDE_DATA=true              # Include data context (default: true)
SUPERVISOR_INCLUDE_WORKSPACE=true         # Include workspace status (default: true)
SUPERVISOR_INCLUDE_SYSTEM=false           # Include system info (default: false)
SUPERVISOR_INCLUDE_MEMORY=false           # Include memory stats (default: false)

# Agent Discovery
SUPERVISOR_AUTO_DISCOVER=true             # Auto-discover agents (default: true)
SUPERVISOR_INCLUDE_AGENT_TOOLS=false      # List agent tools (default: false)
SUPERVISOR_MAX_TOOLS_PER_AGENT=5          # Tools shown per agent (default: 5)
```

## API Key Management

### OpenAI API Key

**Required for**: Core AI functionality, natural language processing

**How to get it:**
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create new secret key
5. Copy the key (it won't be shown again)

**Configuration:**
```env
OPENAI_API_KEY=sk-proj-abcd1234...
```

**Usage considerations:**
- Pay-per-use billing
- Monitor usage in OpenAI dashboard
- Set usage limits if needed
- Keep the key secure and private

### AWS Bedrock Access

**Required for**: Advanced AI models, specific analysis features

**How to get it:**
1. Create [AWS account](https://aws.amazon.com/console/)
2. Enable AWS Bedrock in your region
3. Create IAM user with Bedrock permissions
4. Generate access key and secret key

**Required IAM permissions:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    }
  ]
}
```

**Configuration:**
```env
AWS_BEDROCK_ACCESS_KEY=AKIA...
AWS_BEDROCK_SECRET_ACCESS_KEY=abc123...
AWS_REGION=us-east-1  # Optional: defaults to us-east-1
```

### NCBI API Key (Optional)

**Benefits**: Enhanced literature search, higher rate limits, priority access

**How to get it:**
1. Visit [NCBI E-utilities](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/)
2. Follow registration instructions
3. Receive API key via email

**Configuration:**
```env
NCBI_API_KEY=your-ncbi-api-key-here
```

## Model Profiles

Lobster AI uses predefined model profiles to optimize performance, cost, and compliance for different use cases.

### Available Profiles

#### `development` Profile
**Use case**: Testing, development, lightweight analysis

```env
GENIE_PROFILE=development
```

**Characteristics:**
- Lightweight models (Claude Haiku)
- Faster response times
- Lower costs
- Suitable for prototyping

#### `production` Profile (Default)
**Use case**: Standard research and analysis

```env
GENIE_PROFILE=production
```

**Characteristics:**
- Balanced performance and cost
- Standard Claude Sonnet models
- Good for most research tasks
- Recommended for general use

#### `high-performance` Profile
**Use case**: Complex analyses, large datasets, research publications

```env
GENIE_PROFILE=high-performance
```

**Characteristics:**
- Premium models (Claude Sonnet 4)
- Highest accuracy and reasoning
- Better handling of complex queries
- Higher costs but best results

#### `cost-optimized` Profile
**Use case**: Budget-conscious usage, routine analyses

```env
GENIE_PROFILE=cost-optimized
```

**Characteristics:**
- Lightweight models across all agents
- Minimal API usage
- Good for batch processing
- Budget-friendly option

#### `eu-compliant` Profile
**Use case**: European compliance, data residency requirements

```env
GENIE_PROFILE=eu-compliant
AWS_REGION=eu-central-1
```

**Characteristics:**
- EU region models only
- GDPR-compliant processing
- European data residency
- Suitable for sensitive data

### Custom Model Configuration

For fine-tuned control, override individual agent settings:

```env
# Base profile
GENIE_PROFILE=production

# Custom overrides
GENIE_SUPERVISOR_MODEL=claude-sonnet-4
GENIE_TRANSCRIPTOMICS_EXPERT_MODEL=claude-3-7-sonnet
GENIE_METHOD_AGENT_MODEL=claude-haiku

# Temperature controls (0.0 = deterministic, 1.0 = creative)
GENIE_SUPERVISOR_TEMPERATURE=0.3        # Conservative coordination
GENIE_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7  # Balanced analysis
GENIE_METHOD_AGENT_TEMPERATURE=0.1      # Precise parameter extraction
```

### Supervisor Operation Modes (v2.3+)

The supervisor agent supports three pre-configured operation modes:

#### Research Mode (Interactive)
**Use case**: Exploratory analysis, method development, teaching

```env
SUPERVISOR_ASK_QUESTIONS=true
SUPERVISOR_WORKFLOW_GUIDANCE=detailed
SUPERVISOR_REQUIRE_CONFIRMATION=true
```

**Characteristics:**
- Asks clarification questions to ensure understanding
- Provides detailed workflow guidance
- Requires explicit confirmation for downloads
- Includes examples and detailed instructions
- Best for new users and complex exploratory work

#### Production Mode (Automated)
**Use case**: Automated pipelines, batch processing, routine analyses

```env
SUPERVISOR_ASK_QUESTIONS=false
SUPERVISOR_WORKFLOW_GUIDANCE=minimal
SUPERVISOR_REQUIRE_CONFIRMATION=false
```

**Characteristics:**
- No clarification questions (proceeds with best interpretation)
- Minimal workflow guidance
- Automatic downloads when context is clear
- Streamlined prompt (~1,400 characters smaller)
- Optimized for speed and efficiency

#### Development Mode (Verbose)
**Use case**: Debugging, system understanding, agent development

```env
SUPERVISOR_VERBOSE=true
SUPERVISOR_WORKFLOW_GUIDANCE=detailed
SUPERVISOR_INCLUDE_SYSTEM=true
SUPERVISOR_INCLUDE_MEMORY=true
```

**Characteristics:**
- Detailed delegation explanations
- Full workflow documentation
- System and memory information included
- Maximum transparency for debugging
- Helpful when developing new agents or workflows

## Cloud vs Local Configuration

Lobster AI automatically detects whether to run locally or in the cloud based on configuration.

### Local Mode (Default)

**When**: No `LOBSTER_CLOUD_KEY` is set

**Configuration**: Standard environment variables
```env
OPENAI_API_KEY=your-key
AWS_BEDROCK_ACCESS_KEY=your-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret
```

**Characteristics:**
- Full local processing
- Complete data control
- Requires local compute resources
- All features available

### Cloud Mode

**When**: `LOBSTER_CLOUD_KEY` is set

**Configuration:**
```env
# Cloud API key enables cloud mode
LOBSTER_CLOUD_KEY=your-cloud-api-key-here

# Optional: custom endpoint
LOBSTER_ENDPOINT=https://api.lobster.homara.ai

# Local API keys not required in cloud mode
```

**Characteristics:**
- Processing in Lobster Cloud
- Managed infrastructure
- No local compute requirements
- Pay-per-use cloud billing

### Hybrid Configuration

You can maintain both configurations and switch between them:

```env
# Local mode keys
OPENAI_API_KEY=your-openai-key
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret

# Cloud mode key (commented out)
# LOBSTER_CLOUD_KEY=your-cloud-key

# Shared settings
GENIE_PROFILE=production
NCBI_API_KEY=your-ncbi-key
```

**Switch to cloud mode:**
```bash
export LOBSTER_CLOUD_KEY=your-cloud-key
lobster chat  # Will use cloud mode
```

## Advanced Settings

### Data Processing Configuration

```env
# File handling
GENIE_MAX_FILE_SIZE_MB=500              # Maximum upload size
GENIE_CLUSTER_RESOLUTION=0.5            # Default clustering resolution
GENIE_CACHE_DIR=data/cache              # Cache location

# Memory management
GENIE_MAX_MEMORY_GB=8                   # Memory limit for processing
GENIE_PARALLEL_JOBS=4                   # Number of parallel jobs

# Quality thresholds
GENIE_MIN_CELLS_PER_GENE=3              # Gene filtering threshold
GENIE_MIN_GENES_PER_CELL=200            # Cell filtering threshold
GENIE_MAX_MITO_PERCENT=20               # Mitochondrial content limit
```

### Web Interface Configuration

```env
# Server settings
PORT=8501                               # Streamlit port
HOST=0.0.0.0                           # Bind address
DEBUG=False                             # Debug mode

# Authentication (when enabled)
STREAMLIT_AUTH_ENABLED=True
COGNITO_USER_POOL_ID=your-pool-id
COGNITO_CLIENT_ID=your-client-id
```

### Observability and Monitoring

```env
# Langfuse integration for LLM observability
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com

# Custom logging
LOBSTER_LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
LOBSTER_LOG_FILE=lobster.log            # Log file location
```

## Configuration Management

### Using Configuration Commands

Lobster AI includes built-in configuration management commands:

```bash
# Generate .env file interactively
lobster config generate-env

# Show current configuration
lobster config show-config

# Test API connectivity
lobster config test

# Test specific profile
lobster config test --profile high-performance

# List available models
lobster config list-models
```

### Environment File Locations

Lobster AI looks for configuration in this order:

1. **Current directory**: `./.env`
2. **Home directory**: `~/.lobster/.env`
3. **System environment variables**

### Configuration Validation

Test your configuration:

```bash
# Basic connectivity test
lobster config test

# Comprehensive health check
lobster chat
/dashboard
```

Expected output:
```
🦞 Configuration Status
├── ✅ Environment: .env loaded from ./
├── ✅ OpenAI API: Connected
├── ✅ AWS Bedrock: Connected
├── ✅ NCBI API: Connected (optional)
├── ✅ Profile: production
└── ✅ Mode: Local processing
```

## Security Best Practices

### API Key Security

**DO:**
- Use `.env` files for local development
- Set environment variables in production
- Rotate API keys regularly
- Monitor API usage and costs
- Use least-privilege IAM policies

**DON'T:**
- Commit API keys to version control
- Share API keys in plain text
- Use personal API keys in production
- Leave unused API keys active

### Environment File Security

```bash
# Secure .env file permissions
chmod 600 .env

# Add to .gitignore
echo ".env" >> .gitignore

# Use environment-specific files
.env.development
.env.production
.env.local
```

### Production Security

```bash
# Use system environment variables in production
export OPENAI_API_KEY="$VAULT_OPENAI_KEY"
export AWS_BEDROCK_ACCESS_KEY="$VAULT_AWS_KEY"

# Or use secrets management
kubectl create secret generic lobster-secrets \
  --from-literal=openai-key="$OPENAI_API_KEY" \
  --from-literal=aws-key="$AWS_BEDROCK_ACCESS_KEY"
```

### Cloud Security

```env
# Cloud mode automatically handles API key security
LOBSTER_CLOUD_KEY=your-secure-cloud-key

# No need to expose other API keys locally
# All processing happens in secure cloud environment
```

## Troubleshooting Configuration

### Common Configuration Issues

#### API Key Errors

**Symptom**: `Invalid API key` or `Authentication failed`

**Solutions:**
```bash
# Check environment loading
source .env
echo $OPENAI_API_KEY

# Verify key format
# OpenAI: sk-proj-... or sk-...
# AWS: AKIA... (20 chars)

# Test connectivity
lobster config test

# Check usage limits
# Visit OpenAI/AWS dashboards
```

#### Profile Issues

**Symptom**: `Unknown profile` or model errors

**Solutions:**
```bash
# Check available profiles
lobster config list-profiles

# Reset to default
export GENIE_PROFILE=production

# Clear profile cache
rm -rf .lobster_workspace/cache/profiles/
```

#### Environment Loading Issues

**Symptom**: Settings not taking effect

**Solutions:**
```bash
# Check file location
ls -la .env

# Verify file format (no spaces around =)
cat .env

# Check for BOM or special characters
file .env

# Load manually
set -a; source .env; set +a
```

#### Model Access Issues

**Symptom**: `Model not available` errors

**Solutions:**
```bash
# Check AWS region
echo $AWS_REGION

# Verify Bedrock model access
aws bedrock list-foundation-models --region us-east-1

# Test with different profile
export GENIE_PROFILE=cost-optimized
```

### Configuration Debugging

Enable debug mode for detailed configuration info:

```bash
# Debug mode
lobster chat --debug

# Show all environment variables
lobster config show-config --verbose

# Test each component
lobster config test --component openai
lobster config test --component aws
lobster config test --component ncbi
```

### Reset Configuration

If configuration becomes corrupted:

```bash
# Backup current config
cp .env .env.backup

# Regenerate default configuration
lobster config generate-env --force

# Or start fresh
rm .env
make setup-env
```

### Configuration Templates

For different environments:

**Development template:**
```env
# Development configuration
GENIE_PROFILE=development
OPENAI_API_KEY=your-dev-key
AWS_BEDROCK_ACCESS_KEY=your-dev-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-dev-aws-secret
DEBUG=True
LOBSTER_LOG_LEVEL=DEBUG
```

**Production template:**
```env
# Production configuration
GENIE_PROFILE=production
OPENAI_API_KEY=${VAULT_OPENAI_KEY}
AWS_BEDROCK_ACCESS_KEY=${VAULT_AWS_KEY}
AWS_BEDROCK_SECRET_ACCESS_KEY=${VAULT_AWS_SECRET}
DEBUG=False
LOBSTER_LOG_LEVEL=INFO
GENIE_MAX_FILE_SIZE_MB=1000
```

**Cloud template:**
```env
# Cloud configuration
LOBSTER_CLOUD_KEY=your-cloud-key
GENIE_PROFILE=high-performance
NCBI_API_KEY=your-ncbi-key
```

---

**Next Steps**: With configuration complete, you're ready to start analyzing data! Try the [Getting Started Guide](01-getting-started.md) for example workflows, or jump into `lobster chat` to begin your first analysis.