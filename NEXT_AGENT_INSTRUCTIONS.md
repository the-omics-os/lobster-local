# Lobster Cloud/Local Split - Phase 2 & 3 Instructions

## Phase 1 Status: ✅ COMPLETED

**What was accomplished:**
- ✅ Package structure created (lobster-core, lobster-local, lobster-cloud, lobster-server)
- ✅ Abstract base classes implemented (BaseLobsterClient, BaseDataManager)
- ✅ Smart CLI router implemented with automatic cloud/local detection
- ✅ Development installation workflow created (dev_install.sh)
- ✅ All packages installed and verified working
- ✅ Cloud client with error handling implemented
- ✅ Test scripts created (test_cloud_local.py, verify_installation.py)

**Current Status:**
```bash
# Local mode works
lobster query "What is RNA-seq?"  # Shows: 💻 Using Lobster Local

# Cloud mode detection works (but no backend yet)
LOBSTER_CLOUD_KEY=test-key lobster query "What is RNA-seq?"  # Shows: 🌩️ Using Lobster Cloud
```

---

## Phase 2: Enhanced CLI Integration & Local Testing (Priority: HIGH)

### Objective
Complete the CLI integration, ensure robust testing, and prepare for AWS deployment.

### Tasks

#### 2.1 Enhance CLI Cloud Detection
**File to modify**: `lobster/cli.py`

Current implementation works but needs enhancement:
```python
# Current: Basic cloud detection
# Needed: Better error messages, fallback handling, endpoint configuration
```

**Required improvements:**
1. Add support for `LOBSTER_ENDPOINT` environment variable
2. Improve error messages when cloud fails
3. Add retry logic for cloud connections
4. Better logging of mode selection

#### 2.2 Create Comprehensive Test Suite
**Files to create:**
- `test_phase2_integration.py` - End-to-end CLI testing
- `test_cloud_client_advanced.py` - Advanced cloud client tests
- `test_package_compatibility.py` - Cross-package compatibility tests

**Test coverage needed:**
- CLI mode switching with various environment combinations
- Error handling scenarios (network failures, invalid keys, etc.)
- Package import verification across different environments
- Memory usage and performance comparisons

#### 2.3 Create Production-Ready Installation
**Files to create:**
- `install_production.sh` - Production installation script
- `requirements_cloud.txt` - Minimal cloud client requirements
- `requirements_local.txt` - Full local requirements

**Requirements:**
- Support for both pip and conda environments
- Version pinning for production stability
- Optional dependencies handling
- Cross-platform compatibility (macOS, Linux, Windows)

### Success Criteria for Phase 2:
- [ ] CLI seamlessly switches between modes with clear feedback
- [ ] All test suites pass (unit, integration, end-to-end)
- [ ] Installation works on clean environments
- [ ] Documentation updated with usage examples
- [ ] Performance benchmarks established

---

## Phase 3: AWS Infrastructure Deployment (Priority: CRITICAL)

### Objective
Deploy a minimal but production-ready AWS backend for immediate business validation.

### 3.1 AWS Lambda Setup

**File to create**: `lobster-server/lambda_function.py`
```python
# Minimal Lambda function that:
# 1. Accepts POST requests to /query
# 2. Validates API keys (hardcoded for testing)
# 3. Processes queries using lobster-local logic
# 4. Returns JSON responses
# 5. Handles errors gracefully
```

**Key requirements:**
- Use `lobster-local` package for actual processing
- Implement simple API key validation (hardcoded test keys)
- Add request logging for usage tracking
- Optimize cold start performance
- Handle timeout scenarios

**Dependencies to bundle:**
```bash
# Core dependencies only (keep Lambda package small)
lobster-local
langchain>=0.1.0
langgraph>=0.0.20
pandas>=1.5.0
numpy>=1.23.0
# Exclude heavy ML dependencies initially
```

### 3.2 API Gateway Configuration

**File to create**: `aws_setup_guide.md`

**Manual AWS setup steps** (next agent should create detailed guide):
1. Create Lambda function (`lobster-api`)
2. Set up API Gateway REST API
3. Configure `/query` POST endpoint
4. Set up API keys and usage plans
5. Configure CORS
6. Set up monitoring and logging

**Test API keys to configure:**
- `test-enterprise-001` - For initial testing
- `test-enterprise-002` - For load testing
- `demo-user-001` - For demonstrations

### 3.3 Deployment Automation

**Files to create:**
- `deploy_to_aws.sh` - Automated deployment script
- `create_lambda_package.py` - Lambda package builder
- `test_aws_deployment.py` - AWS endpoint testing

**Deployment script should:**
1. Build Lambda deployment package
2. Update Lambda function code
3. Test the deployed endpoint
4. Validate API key authentication
5. Run smoke tests

### 3.4 Cloud Client Enhancement

**File to modify**: `lobster-cloud/lobster_cloud/client.py`

**Add features:**
- Automatic endpoint discovery
- Request retries with exponential backoff
- Response caching for repeated queries
- Usage statistics tracking
- Better error categorization

### Success Criteria for Phase 3:
- [ ] Lambda function deployed and responding
- [ ] API Gateway configured with authentication
- [ ] Test API keys working
- [ ] Cloud client successfully connecting to AWS
- [ ] Basic usage monitoring in place

---

## Phase 4: Testing & Business Validation (Priority: HIGH)

### Objective
Ensure the system is ready for real user testing and business model validation.

### 4.1 End-to-End Validation

**File to create**: `test_production_readiness.py`

**Test scenarios:**
1. **Local Mode Performance**: Benchmark current vs. new architecture
2. **Cloud Mode Functionality**: Full query processing via AWS
3. **Failover Testing**: Cloud-to-local fallback scenarios
4. **Load Testing**: Multiple concurrent cloud requests
5. **Cost Analysis**: AWS usage cost estimation

### 4.2 User Experience Testing

**Files to create:**
- `user_experience_test.py` - UX validation script
- `CLOUD_USAGE_GUIDE.md` - User documentation
- `TROUBLESHOOTING.md` - Common issues and solutions

**UX requirements:**
- Seamless mode switching (users shouldn't need to think about it)
- Clear error messages with actionable advice
- Performance feedback (local vs cloud speed comparison)
- Usage tracking and billing transparency

### 4.3 Business Model Validation Setup

**Files to create:**
- `usage_analytics.py` - Track user patterns
- `cost_calculation.py` - Calculate per-query costs
- `user_feedback_collector.py` - Gather validation data

**Analytics to track:**
- Query complexity distribution
- Local vs cloud usage patterns
- User retention and engagement
- Cost per query actual vs estimated

### Success Criteria for Phase 4:
- [ ] All systems passing production readiness tests
- [ ] User documentation complete and tested
- [ ] Analytics and cost tracking operational
- [ ] Ready for alpha user testing
- [ ] Business model validation data collection ready

---

## Critical Implementation Notes

### Security Considerations
1. **API Key Management**: Currently using hardcoded keys - acceptable for validation phase
2. **Input Validation**: Ensure Lambda validates all inputs
3. **Rate Limiting**: Implement basic rate limiting per API key
4. **Logging**: Log usage but not sensitive user data

### Performance Optimization
1. **Lambda Cold Start**: Minimize by keeping packages lightweight
2. **Response Caching**: Cache common query results
3. **Async Processing**: For long-running queries, consider async patterns
4. **Database**: Use DynamoDB for user sessions and usage tracking

### Cost Management
1. **Resource Limits**: Set sensible Lambda timeout and memory limits
2. **Usage Monitoring**: Track costs per API key
3. **Auto-scaling**: Configure API Gateway throttling
4. **Budget Alerts**: Set up AWS cost monitoring

---

## Files Structure After Completion

```
lobster/
├── lobster-core/           # ✅ Completed
├── lobster-local/          # ✅ Completed  
├── lobster-cloud/          # ✅ Basic, needs enhancement
├── lobster-server/         # 🔄 Needs Lambda implementation
│   ├── lambda_function.py  # 📝 CREATE
│   ├── requirements.txt    # 📝 CREATE
│   └── mock_users.py      # 📝 CREATE
├── aws_setup_guide.md      # 📝 CREATE
├── deploy_to_aws.sh        # 📝 CREATE
├── test_production_readiness.py # 📝 CREATE
├── CLOUD_USAGE_GUIDE.md    # 📝 CREATE
└── TROUBLESHOOTING.md      # 📝 CREATE
```

## Immediate Next Steps for Agent

1. **Start with Phase 2 CLI enhancements** - ensure local functionality is rock-solid
2. **Create the Lambda function** - this is the critical path for cloud functionality
3. **Set up AWS infrastructure** - manual setup first, automation second
4. **Test end-to-end** - ensure the full flow works
5. **Document everything** - for user adoption and troubleshooting

## Success Metrics

**Technical Success:**
- ✅ CLI works seamlessly in both modes
- ✅ AWS Lambda responding to requests
- ✅ API authentication functional
- ✅ Error handling robust
- ✅ Performance acceptable (< 30s for complex queries)

**Business Success:**
- ✅ Ready for alpha user testing
- ✅ Cost tracking operational
- ✅ User feedback collection ready
- ✅ Freemium value proposition clear

## Emergency Fallbacks

If AWS setup proves complex:
1. **Fallback 1**: Deploy to Railway/Heroku with Docker
2. **Fallback 2**: Create local mock server for testing
3. **Fallback 3**: Focus on perfecting local mode first

**The goal is business model validation, not perfect architecture. Speed to market is critical.**

---

*This task represents the critical path to a functioning freemium Lobster AI system. The next agent should prioritize getting the AWS backend working over architectural perfection.*
