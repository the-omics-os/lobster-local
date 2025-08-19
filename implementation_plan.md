# Implementation Plan

Transform Lobster CLI application into a FastAPI-based system accessible via TypeScript/React frontend with AWS ECS Fargate deployment.

The current Lobster application is a sophisticated multi-agent bioinformatics system using LangGraph with a terminal-based interface. This implementation will restructure the core components to expose all functionality through RESTful APIs and WebSocket connections, enabling a modern web-based user interface while preserving the powerful multi-agent architecture. The system will support persistent user sessions, file uploads, public repository access, and real-time analysis streaming. The deployment strategy focuses on AWS ECS Fargate for simplified container management suitable for small team testing environments.

## [Types]

Define API models and data structures for the web interface.

### Request/Response Models
```python
# Session Management
class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = None
    workspace_name: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_active: datetime
    user_id: Optional[str]
    workspace_path: str
    status: str

# Chat Interface
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    plots: Optional[List[PlotInfo]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    plots: Optional[List[PlotInfo]] = None
    data_updated: bool = False

# Data Management
class DataUploadResponse(BaseModel):
    success: bool
    filename: str
    file_id: str
    rows: int
    columns: int
    message: str

class DatasetInfo(BaseModel):
    source: str
    dataset_id: str
    shape: Tuple[int, int]
    processing_step: str
    created_at: datetime

# Plot Management
class PlotInfo(BaseModel):
    id: str
    title: str
    timestamp: datetime
    source: str
    format: str

# System Status
class SystemStatus(BaseModel):
    status: str
    active_sessions: int
    total_sessions: int
    system_health: Dict[str, Any]
```

### WebSocket Events
```python
class WSEventType(Enum):
    CHAT_STREAM = "chat_stream"
    AGENT_THINKING = "agent_thinking"
    ANALYSIS_PROGRESS = "analysis_progress"
    DATA_UPDATED = "data_updated"
    PLOT_GENERATED = "plot_generated"
    ERROR = "error"

class WSMessage(BaseModel):
    event_type: WSEventType
    session_id: str
    data: Dict[str, Any]
    timestamp: datetime
```

## [Files]

Comprehensive file structure modifications and additions.

### New API Files
- `lobster/api/__init__.py` - API package initialization
- `lobster/api/main.py` - FastAPI application setup and configuration
- `lobster/api/models.py` - Pydantic models for API requests/responses
- `lobster/api/dependencies.py` - FastAPI dependency injection functions
- `lobster/api/middleware.py` - Custom middleware for CORS, logging, error handling
- `lobster/api/websocket.py` - WebSocket connection management and handlers

### API Route Modules
- `lobster/api/routes/__init__.py` - Routes package initialization
- `lobster/api/routes/sessions.py` - Session management endpoints
- `lobster/api/routes/chat.py` - Chat and conversation endpoints
- `lobster/api/routes/data.py` - Data upload and management endpoints
- `lobster/api/routes/plots.py` - Plot retrieval and export endpoints
- `lobster/api/routes/health.py` - System health and status endpoints
- `lobster/api/routes/files.py` - File upload/download endpoints

### Session Management
- `lobster/api/session_manager.py` - Session lifecycle management
- `lobster/api/session_store.py` - Session persistence and retrieval

### Enhanced Core Files
- `lobster/core/api_client.py` - Adapted AgentClient for API usage
- `lobster/core/websocket_callback.py` - WebSocket-aware callback handler
- `lobster/utils/api_helpers.py` - API utility functions

### Configuration and Deployment
- `docker/Dockerfile.api` - Optimized Docker image for API service
- `docker/docker-compose.api.yml` - Docker Compose for local API development
- `aws/ecs-task-definition.json` - ECS task definition for Fargate
- `aws/cloudformation-stack.yml` - CloudFormation template for AWS resources
- `nginx/nginx.conf` - Nginx configuration for reverse proxy
- `.env.production` - Production environment variables template

### Documentation
- `README_UI.md` - Comprehensive UI deployment and usage guide
- `docs/api/README.md` - API documentation and usage examples
- `docs/api/endpoints.md` - Detailed endpoint specifications
- `docs/deployment/aws-setup.md` - AWS deployment step-by-step guide

## [Functions]

API endpoint functions and enhanced core functionality.

### Session Management Functions
- `create_session(request: SessionCreateRequest) -> SessionResponse` - Create new user session
- `get_session(session_id: str) -> SessionResponse` - Retrieve session information
- `list_sessions(user_id: Optional[str]) -> List[SessionResponse]` - List user sessions
- `delete_session(session_id: str) -> Dict[str, str]` - Delete session and cleanup resources
- `extend_session(session_id: str) -> SessionResponse` - Extend session timeout

### Chat Interface Functions
- `send_message(request: ChatRequest) -> ChatResponse` - Process user message through agents
- `get_chat_history(session_id: str, limit: int) -> List[ChatMessage]` - Retrieve conversation history
- `stream_chat(request: ChatRequest) -> AsyncIterator[str]` - Stream agent responses via WebSocket
- `clear_conversation(session_id: str) -> Dict[str, str]` - Clear conversation history

### Data Management Functions
- `upload_file(file: UploadFile, session_id: str) -> DataUploadResponse` - Handle file uploads
- `download_geo_dataset(geo_id: str, session_id: str) -> DataUploadResponse` - Download from GEO
- `get_dataset_info(session_id: str) -> Optional[DatasetInfo]` - Get current dataset information
- `list_workspace_files(session_id: str) -> Dict[str, List[Dict]]` - List session workspace files
- `export_session_data(session_id: str) -> str` - Create downloadable data package

### Plot Management Functions
- `get_plots(session_id: str) -> List[PlotInfo]` - List generated plots
- `get_plot_data(plot_id: str, session_id: str) -> Dict[str, Any]` - Retrieve plot data
- `download_plot(plot_id: str, format: str, session_id: str) -> FileResponse` - Download plot file
- `clear_plots(session_id: str) -> Dict[str, str]` - Clear session plots

### WebSocket Functions
- `handle_websocket_connection(websocket: WebSocket, session_id: str)` - Manage WebSocket connections
- `broadcast_to_session(session_id: str, message: WSMessage)` - Send message to session WebSockets
- `send_agent_progress(session_id: str, agent_name: str, progress: Dict)` - Stream agent progress

### Enhanced Core Functions (Modified)
- `AgentClient.__init__()` - Add WebSocket callback support and session awareness
- `DataManager.set_data()` - Add session-based workspace isolation
- `create_bioinformatics_graph()` - Add WebSocket-aware callback configuration

## [Classes]

API service classes and enhanced existing classes.

### New API Classes
- `FastAPIApp(FastAPI)` - Main application class with middleware and route configuration
- `SessionManager` - Manages session lifecycle, cleanup, and persistence
- `SessionStore` - Handles session data storage and retrieval
- `WebSocketManager` - Manages WebSocket connections and broadcasting
- `FileUploadHandler` - Processes multipart file uploads and validation
- `APICallbackHandler(BaseCallbackHandler)` - WebSocket-aware callback for agent streaming

### Enhanced Existing Classes
- `AgentClient` - Modified for session-based operation and WebSocket integration
- `DataManager` - Enhanced with session workspace isolation and API-friendly methods
- `TerminalCallbackHandler` - Extended to support WebSocket broadcasting

### Session Management Classes
```python
class SessionManager:
    def __init__(self, storage_backend: str = "memory")
    def create_session(self, user_id: Optional[str] = None) -> str
    def get_session(self, session_id: str) -> Optional[Session]
    def delete_session(self, session_id: str) -> bool
    def cleanup_expired_sessions(self) -> int

class Session:
    session_id: str
    user_id: Optional[str]
    agent_client: AgentClient
    created_at: datetime
    last_active: datetime
    workspace_path: Path
    websocket_connections: Set[WebSocket]
```

### WebSocket Management Classes
```python
class WebSocketManager:
    def __init__(self)
    async def connect(self, websocket: WebSocket, session_id: str)
    async def disconnect(self, websocket: WebSocket, session_id: str)
    async def broadcast_to_session(self, session_id: str, message: dict)
    async def send_personal_message(self, message: dict, websocket: WebSocket)
```

## [Dependencies]

Additional packages and version updates.

### New Dependencies
```toml
# API Framework
"fastapi>=0.104.0"
"uvicorn[standard]>=0.24.0"
"websockets>=12.0"
"python-multipart>=0.0.6"  # File upload support

# Session Management
"redis>=5.0.0"  # Optional session storage backend
"aiofiles>=23.0"  # Async file operations

# AWS Integration
"boto3>=1.34.0"
"botocore>=1.34.0"

# Enhanced Production Support
"gunicorn>=21.0.0"  # WSGI server for production
"prometheus-client>=0.19.0"  # Metrics collection
```

### Updated Dependencies
- `uvicorn` updated to latest with WebSocket support
- `fastapi` updated for latest features and security patches
- `pydantic` v2 compatibility ensured for all models

## [Testing]

Comprehensive testing strategy for API functionality.

### Test Files Structure
- `tests/api/__init__.py` - API test package
- `tests/api/test_session_management.py` - Session creation, retrieval, deletion tests
- `tests/api/test_chat_endpoints.py` - Chat functionality and streaming tests
- `tests/api/test_file_uploads.py` - File upload validation and processing tests
- `tests/api/test_websocket.py` - WebSocket connection and messaging tests
- `tests/api/test_data_management.py` - Data operations and workspace isolation tests
- `tests/integration/test_api_agent_integration.py` - End-to-end agent workflow tests

### Testing Approach
- **Unit Tests**: Individual endpoint functionality, input validation, error handling
- **Integration Tests**: Multi-agent workflows through API, session persistence, WebSocket communication
- **Load Tests**: Session capacity, concurrent user handling, memory usage under load
- **Security Tests**: Input sanitization, file upload safety, session hijacking prevention

### Test Fixtures and Utilities
- `pytest-asyncio>=0.21.0` for async endpoint testing
- Test client fixtures for FastAPI testing
- Mock WebSocket connections for streaming tests
- Sample data files for upload testing
- Session cleanup utilities for test isolation

## [Implementation Order]

Logical sequence for building the API system.

### Phase 1: Core API Infrastructure (Priority 1)
1. **API Foundation Setup**
   - Create FastAPI application structure (`lobster/api/main.py`)
   - Implement basic middleware for CORS and logging (`lobster/api/middleware.py`)
   - Set up Pydantic models for core data structures (`lobster/api/models.py`)
   - Create health check endpoint (`lobster/api/routes/health.py`)

2. **Session Management Core**
   - Implement `SessionManager` class with in-memory storage
   - Create session CRUD endpoints (`lobster/api/routes/sessions.py`)
   - Add session-aware `AgentClient` modifications (`lobster/core/api_client.py`)
   - Implement session timeout and cleanup mechanisms

### Phase 2: Agent Integration (Priority 2)
3. **Chat API Implementation**
   - Adapt existing AgentClient for API usage (`lobster/core/api_client.py`)
   - Create chat endpoints with synchronous responses (`lobster/api/routes/chat.py`)
   - Implement conversation history persistence per session
   - Add error handling and response formatting

4. **WebSocket Integration**
   - Implement WebSocket connection management (`lobster/api/websocket.py`)
   - Create WebSocket-aware callback handler (`lobster/core/websocket_callback.py`)
   - Add real-time agent streaming capability
   - Implement progress broadcasting for long-running operations

### Phase 3: Data Management (Priority 3)
5. **File Upload System**
   - Create file upload endpoints (`lobster/api/routes/files.py`)
   - Implement file validation and processing
   - Add session-based workspace isolation
   - Create secure file storage with cleanup

6. **Data Operations API**
   - Implement data management endpoints (`lobster/api/routes/data.py`)
   - Add GEO dataset download through API
   - Create dataset information and workspace file listing
   - Implement data export and download functionality

### Phase 4: Visualization and Export (Priority 4)
7. **Plot Management API**
   - Create plot retrieval endpoints (`lobster/api/routes/plots.py`)
   - Implement plot format conversion and download
   - Add plot metadata and history management
   - Create plot export in multiple formats (PNG, HTML, SVG)

8. **Export and Download Features**
   - Implement comprehensive data package creation
   - Add session export with all data, plots, and metadata
   - Create downloadable ZIP files with proper cleanup
   - Add export scheduling for large datasets

### Phase 5: Production Deployment (Priority 5)
9. **Docker and AWS Preparation**
   - Create production Dockerfile (`docker/Dockerfile.api`)
   - Set up environment variable management
   - Create AWS ECS task definition (`aws/ecs-task-definition.json`)
   - Implement health checks and logging configuration

10. **AWS Infrastructure Deployment**
    - Create CloudFormation template for ECS Fargate setup
    - Set up Application Load Balancer with HTTPS
    - Configure EFS for persistent workspace storage
    - Implement S3 integration for file uploads and exports
    - Set up CloudWatch logging and monitoring

### Phase 6: Documentation and Testing (Priority 6)
11. **Comprehensive Testing Suite**
    - Implement unit tests for all API endpoints
    - Create integration tests for multi-agent workflows
    - Add WebSocket connection and streaming tests
    - Perform load testing and session capacity validation

12. **Documentation and Deployment Guide**
    - Create comprehensive README_UI.md with deployment instructions
    - Write API documentation with example requests/responses
    - Create step-by-step AWS deployment guide
    - Document TypeScript/React integration patterns
