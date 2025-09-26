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

Choose ONE of the following LLM providers:

```env
# Option 1: Claude API (Recommended for simplicity)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Option 2: AWS Bedrock (For AWS users)
AWS_BEDROCK_ACCESS_KEY=your-aws-access-key-here
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret-key-here

# Optional: Force a specific provider (auto-detected by default)
# LOBSTER_LLM_PROVIDER=anthropic  # or "bedrock"
```

### Optional Variables

```env
# Enhanced Literature Search
NCBI_API_KEY=your-ncbi-api-key-here

# Model Configuration
LOBSTER_PROFILE=production                    # Model preset configuration
LOBSTER_SUPERVISOR_TEMPERATURE=0.5            # Supervisor agent temperature
LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7 # Analysis agent temperature
LOBSTER_METHOD_AGENT_TEMPERATURE=0.3          # Method agent temperature

# Application Settings
LOBSTER_MAX_FILE_SIZE_MB=500                  # Maximum file size for uploads
LOBSTER_CLUSTER_RESOLUTION=0.5                # Default clustering resolution
LOBSTER_CACHE_DIR=data/cache                  # Cache directory location

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
LOBSTER_ENDPOINT=https://api.lobster.omics-os.com  # Optional: defaults to production
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

## LLM Provider Management

Lobster AI supports multiple LLM providers with automatic detection. Choose ONE of the following providers:

### Claude API (Recommended)

**Best for**: Simple setup, direct billing, most users

**How to get it:**
1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create new API key
5. Copy the key (it won't be shown again)

**Configuration:**
```env
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
```

**Usage considerations:**
- Pay-per-use billing directly with Anthropic
- Simple setup with just one API key
- Global availability
- Automatic model updates

### AWS Bedrock Access

**Best for**: AWS users, enterprise compliance, specific regions

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

### Provider Auto-Detection

Lobster AI automatically detects which provider to use based on available environment variables:

**Priority Order:**
1. **Manual Override**: `LOBSTER_LLM_PROVIDER=anthropic` or `=bedrock`
2. **Claude API**: If `ANTHROPIC_API_KEY` is set
3. **AWS Bedrock**: If `AWS_BEDROCK_ACCESS_KEY` and `AWS_BEDROCK_SECRET_ACCESS_KEY` are set

**Manual Provider Selection:**
```env
# Force specific provider (overrides auto-detection)
LOBSTER_LLM_PROVIDER=anthropic  # or "bedrock"
```

**Check Current Provider:**
```bash
# Lobster will show detected provider on startup
lobster chat

# Or check configuration
lobster config show-config
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
LOBSTER_PROFILE=development
```

**Characteristics:**
- Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant
- Fast development cycle
- Balanced performance and cost
- Suitable for prototyping

#### `production` Profile (Default)
**Use case**: Standard research and analysis

```env
LOBSTER_PROFILE=production
```

**Characteristics:**
- Claude 4 Sonnet for all agents, 3.5 Sonnet v2 for assistant
- Production-ready quality
- Best performance for analysis
- Recommended for research use

#### `cost-optimized` Profile
**Use case**: Budget-conscious usage, routine analyses

```env
LOBSTER_PROFILE=cost-optimized
```

**Characteristics:**
- Claude 3.7 Sonnet for all agents, 3.5 Sonnet v2 for assistant
- Minimal API usage costs
- Good for batch processing
- Budget-friendly option

### Custom Model Configuration

For fine-tuned control, override individual agent settings:

```env
# Base profile
LOBSTER_PROFILE=production

# Custom overrides
LOBSTER_SUPERVISOR_MODEL=claude-sonnet-4
LOBSTER_TRANSCRIPTOMICS_EXPERT_MODEL=claude-3-7-sonnet
LOBSTER_METHOD_AGENT_MODEL=claude-haiku

# Temperature controls (0.0 = deterministic, 1.0 = creative)
LOBSTER_SUPERVISOR_TEMPERATURE=0.3        # Conservative coordination
LOBSTER_TRANSCRIPTOMICS_EXPERT_TEMPERATURE=0.7  # Balanced analysis
LOBSTER_METHOD_AGENT_TEMPERATURE=0.1      # Precise parameter extraction
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

**Configuration**: Choose ONE LLM provider
```env
# Option 1: Claude API
ANTHROPIC_API_KEY=your-claude-key

# Option 2: AWS Bedrock
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret
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
LOBSTER_ENDPOINT=https://api.lobster.omics-os.com

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
# Local mode keys (choose ONE provider)
ANTHROPIC_API_KEY=your-claude-key
# OR
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret

# Cloud mode key (commented out)
# LOBSTER_CLOUD_KEY=your-cloud-key

# Shared settings
LOBSTER_PROFILE=production
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
LOBSTER_MAX_FILE_SIZE_MB=500              # Maximum upload size
LOBSTER_CLUSTER_RESOLUTION=0.5            # Default clustering resolution
LOBSTER_CACHE_DIR=data/cache              # Cache location

# Memory management
LOBSTER_MAX_MEMORY_GB=8                   # Memory limit for processing
LOBSTER_PARALLEL_JOBS=4                   # Number of parallel jobs

# Quality thresholds
LOBSTER_MIN_CELLS_PER_GENE=3              # Gene filtering threshold
LOBSTER_MIN_GENES_PER_CELL=200            # Cell filtering threshold
LOBSTER_MAX_MITO_PERCENT=20               # Mitochondrial content limit
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
lobster config test --profile production

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
├── ✅ LLM Provider: Claude API (or AWS Bedrock)
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
export ANTHROPIC_API_KEY="$VAULT_ANTHROPIC_KEY"
# OR
export AWS_BEDROCK_ACCESS_KEY="$VAULT_AWS_KEY"

# Or use secrets management (choose ONE provider)
kubectl create secret generic lobster-secrets \
  --from-literal=anthropic-key="$ANTHROPIC_API_KEY"

# OR for AWS Bedrock
kubectl create secret generic lobster-secrets \
  --from-literal=aws-key="$AWS_BEDROCK_ACCESS_KEY" \
  --from-literal=aws-secret="$AWS_BEDROCK_SECRET_ACCESS_KEY"
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
echo $ANTHROPIC_API_KEY    # For Claude API
echo $AWS_BEDROCK_ACCESS_KEY  # For AWS Bedrock

# Verify key format
# Claude API: sk-ant-...
# AWS: AKIA... (20 chars)

# Test connectivity
lobster config test

# Check usage limits
# Visit Anthropic Console or AWS dashboards
```

#### Profile Issues

**Symptom**: `Unknown profile` or model errors

**Solutions:**
```bash
# Check available profiles
lobster config list-profiles

# Reset to default
export LOBSTER_PROFILE=production

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
export LOBSTER_PROFILE=cost-optimized
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
# Development configuration (choose ONE provider)
LOBSTER_PROFILE=development
ANTHROPIC_API_KEY=your-claude-key
# OR
AWS_BEDROCK_ACCESS_KEY=your-aws-key
AWS_BEDROCK_SECRET_ACCESS_KEY=your-aws-secret
DEBUG=True
LOBSTER_LOG_LEVEL=DEBUG
```

**Production template:**
```env
# Production configuration (choose ONE provider)
LOBSTER_PROFILE=production
ANTHROPIC_API_KEY=${VAULT_ANTHROPIC_KEY}
# OR
AWS_BEDROCK_ACCESS_KEY=${VAULT_AWS_KEY}
AWS_BEDROCK_SECRET_ACCESS_KEY=${VAULT_AWS_SECRET}
DEBUG=False
LOBSTER_LOG_LEVEL=INFO
LOBSTER_MAX_FILE_SIZE_MB=1000
```

**Cloud template:**
```env
# Cloud configuration
LOBSTER_CLOUD_KEY=your-cloud-key
LOBSTER_PROFILE=production
NCBI_API_KEY=your-ncbi-key
```

---

**Next Steps**: With configuration complete, you're ready to start analyzing data! Try the [Getting Started Guide](01-getting-started.md) for example workflows, or jump into `lobster chat` to begin your first analysis.