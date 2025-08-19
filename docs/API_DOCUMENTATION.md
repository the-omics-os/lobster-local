# Lobster AI - API Documentation for Frontend Integration

## Overview

The Lobster AI API provides a comprehensive bioinformatics analysis system with multi-agent capabilities, real-time streaming, and session-based isolation. This documentation is designed for frontend developers and AI systems to understand backend communication patterns.

## Base Configuration

**Base URL:** `http://localhost:8000` (development) or your production URL
**API Version:** `v1`
**All API endpoints are prefixed with:** `/api/v1`

## Authentication & Sessions

The API uses session-based authentication with UUID identification. No user authentication is required for development.

### Session Workflow
1. **Create Session** → Get session_id
2. **Use session_id** in all subsequent requests
3. **Optional:** Connect to WebSocket for real-time updates
4. **Cleanup:** Delete session when done

## Core API Endpoints

### 1. Session Management

#### Create Session
```http
POST /api/v1/sessions
Content-Type: application/json

{
  "name": "My Analysis Session",
  "description": "Single-cell RNA analysis",
  "user_id": "user123",
  "timeout_minutes": 60
}
```

**Response:**
```json
{
  "success": true,
  "message": "Session created successfully",
  "timestamp": "2025-08-18T18:00:00Z",
  "session": {
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "name": "My Analysis Session",
    "status": "active",
    "created_at": "2025-08-18T18:00:00Z",
    "workspace_path": "workspaces/123e4567-e89b-12d3-a456-426614174000",
    "datasets": [],
    "message_count": 0
  }
}
```

#### Get Session Info
```http
GET /api/v1/sessions/{session_id}
```

#### List All Sessions
```http
GET /api/v1/sessions
```

#### Delete Session
```http
DELETE /api/v1/sessions/{session_id}
```

### 2. Chat Interface

#### Send Message to Agents
```http
POST /api/v1/sessions/{session_id}/chat
Content-Type: application/json

{
  "message": "Please analyze the GSE123456 dataset",
  "stream": true,
  "include_history": true,
  "max_history": 10
}
```

**Response:**
```json
{
  "success": true,
  "message": "Message processed successfully",
  "timestamp": "2025-08-18T18:01:00Z",
  "chat_message": {
    "id": "msg-uuid",
    "role": "assistant",
    "content": "I'll analyze the GSE123456 dataset for you...",
    "timestamp": "2025-08-18T18:01:00Z",
    "metadata": {
      "duration": 5.2,
      "has_data": true,
      "session_id": "session-uuid"
    },
    "plots": ["plot1.png", "plot2.svg"]
  },
  "conversation_id": "session-uuid",
  "data_updated": true
}
```

#### Get Conversation History
```http
GET /api/v1/sessions/{session_id}/chat/history?limit=50
```

#### Clear Conversation
```http
DELETE /api/v1/sessions/{session_id}/chat/history
```

### 3. File Management

#### Upload File
```http
POST /api/v1/sessions/{session_id}/files/upload
Content-Type: multipart/form-data

form-data:
- file: [binary file data]
- description: "Single-cell expression matrix"
```

**Response:**
```json
{
  "success": true,
  "message": "File uploaded and loaded successfully",
  "file_path": "workspaces/session-uuid/data/expression_matrix.csv",
  "file_size": 1048576,
  "dataset": {
    "name": "expression_matrix.csv",
    "file_type": "csv",
    "status": "ready",
    "rows": 1000,
    "columns": 50
  }
}
```

#### List Files
```http
GET /api/v1/sessions/{session_id}/files?directory=data
```

#### Download File
```http
GET /api/v1/sessions/{session_id}/files/{file_path}/download
```

#### Delete File
```http
DELETE /api/v1/sessions/{session_id}/files/{file_path}
```

### 4. Data Management

#### Download GEO Dataset
```http
POST /api/v1/sessions/{session_id}/data/geo-download
Content-Type: application/json

{
  "geo_id": "GSE123456",
  "destination_path": "data/geo_data"
}
```

#### List Datasets
```http
GET /api/v1/sessions/{session_id}/data/datasets
```

#### Get Data Summary
```http
GET /api/v1/sessions/{session_id}/data/summary
```

#### Perform Data Operation
```http
POST /api/v1/sessions/{session_id}/data/operations
Content-Type: application/json

{
  "dataset_name": "current_dataset",
  "operation": "clustering",
  "parameters": {
    "method": "leiden",
    "resolution": 0.5
  }
}
```

#### Load Data File
```http
POST /api/v1/sessions/{session_id}/data/load
Content-Type: application/json

{
  "file_path": "data/expression_matrix.csv"
}
```

#### Get Workspace Info
```http
GET /api/v1/sessions/{session_id}/data/workspace
```

### 5. Plot Management

#### List Plots
```http
GET /api/v1/sessions/{session_id}/plots?limit=20
```

**Response:**
```json
{
  "success": true,
  "message": "Retrieved 3 plots",
  "plots": [
    {
      "id": "umap_plot.png",
      "title": "umap_plot",
      "timestamp": "2025-08-18T18:05:00Z",
      "source": "agent_generated",
      "format": "png",
      "path": "plots/umap_plot.png",
      "size_bytes": 256000
    }
  ],
  "total": 3
}
```

#### Download Plot
```http
GET /api/v1/sessions/{session_id}/plots/{plot_id}/download
```

#### Generate Plot
```http
POST /api/v1/sessions/{session_id}/plots/generate
Content-Type: application/json

{
  "plot_type": "umap",
  "description": "UMAP visualization with cell type annotations",
  "parameters": {
    "color_by": "cell_type",
    "size": "800x600"
  }
}
```

#### Get Latest Plots
```http
GET /api/v1/sessions/{session_id}/plots/latest?count=5
```

#### Delete Plot
```http
DELETE /api/v1/sessions/{session_id}/plots/{plot_id}
```

### 6. Export System

#### Create Export Package
```http
POST /api/v1/sessions/{session_id}/export
Content-Type: application/json

{
  "include_data": true,
  "include_plots": true,
  "include_logs": false,
  "format": "zip"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Export package created successfully",
  "export_id": "export-uuid",
  "download_url": "/api/v1/exports/export-uuid/download",
  "expires_at": "2025-08-19T18:00:00Z",
  "size_bytes": 5242880
}
```

#### Download Export
```http
GET /api/v1/exports/{export_id}/download
```

#### Get Export Info
```http
GET /api/v1/exports/{export_id}/info
```

### 7. System Health

#### Health Check
```http
GET /api/v1/health
```

**Response:**
```json
{
  "success": true,
  "message": "System is healthy",
  "system": {
    "status": "healthy",
    "active_sessions": 5,
    "total_sessions": 12,
    "uptime_seconds": 3600,
    "memory_usage": {
      "total_gb": 16.0,
      "used_gb": 4.2,
      "percent_used": 26.3
    }
  },
  "version": "1.0.0",
  "environment": "development"
}
```

## WebSocket Real-time Communication

### Connection
```javascript
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/${session_id}`);
```

### Message Types Received

#### Connection Established
```json
{
  "event_type": "connection",
  "session_id": "session-uuid",
  "data": {
    "status": "connected",
    "message": "WebSocket connected successfully"
  }
}
```

#### Chat Streaming
```json
{
  "event_type": "chat_stream",
  "session_id": "session-uuid",
  "data": {
    "token": "Hello",
    "agent": "transcriptomics_expert",
    "type": "token"
  }
}
```

#### Agent Progress
```json
{
  "event_type": "analysis_progress",
  "session_id": "session-uuid",
  "data": {
    "status": "tool_start",
    "agent": "data_expert",
    "tool": "clustering_service",
    "message": "Using tool: clustering_service"
  }
}
```

#### Data Updated
```json
{
  "event_type": "data_updated",
  "session_id": "session-uuid",
  "data": {
    "status": "data_loaded",
    "dataset": {
      "has_data": true,
      "summary": "1000 cells, 2000 genes"
    }
  }
}
```

#### Plot Generated
```json
{
  "event_type": "plot_generated",
  "session_id": "session-uuid", 
  "data": {
    "status": "plot_created",
    "plot": {
      "path": "plots/umap_visualization.png",
      "name": "umap_visualization.png"
    }
  }
}
```

#### Error Notification
```json
{
  "event_type": "error",
  "session_id": "session-uuid",
  "data": {
    "status": "error",
    "agent": "data_expert",
    "error": "File format not supported",
    "message": "Error in data_expert: File format not supported"
  }
}
```

### Messages You Can Send

#### Ping for Connection Health
```json
{
  "type": "ping",
  "timestamp": "2025-08-18T18:00:00Z"
}
```

#### Subscribe to Event Types
```json
{
  "type": "subscribe",
  "event_types": ["chat_stream", "analysis_progress", "plot_generated"]
}
```

## Common Frontend Integration Patterns

### 1. Session Initialization
```typescript
// Create new session
const sessionResponse = await fetch('/api/v1/sessions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Analysis Session',
    timeout_minutes: 60
  })
});

const session = await sessionResponse.json();
const sessionId = session.session.session_id;

// Connect WebSocket for real-time updates
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/${sessionId}`);
```

### 2. File Upload with Progress
```typescript
const formData = new FormData();
formData.append('file', file);
formData.append('description', 'RNA expression data');

const uploadResponse = await fetch(`/api/v1/sessions/${sessionId}/files/upload`, {
  method: 'POST',
  body: formData
});
```

### 3. Chat with Agents
```typescript
const chatResponse = await fetch(`/api/v1/sessions/${sessionId}/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Perform quality control analysis',
    stream: true
  })
});
```

### 4. Real-time Updates Handling
```typescript
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.event_type) {
    case 'chat_stream':
      // Update chat interface with streaming response
      appendToChatStream(message.data.token);
      break;
      
    case 'analysis_progress':
      // Update progress bar or status indicator
      updateProgress(message.data.message, message.data.agent);
      break;
      
    case 'plot_generated':
      // Refresh plot list or display new plot
      refreshPlots();
      break;
      
    case 'data_updated':
      // Update dataset information
      refreshDataStatus();
      break;
      
    case 'error':
      // Display error notification
      showError(message.data.message);
      break;
  }
};
```

## Error Handling

All API responses follow this structure:

**Success Response:**
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "timestamp": "2025-08-18T18:00:00Z",
  // ... additional response data
}
```

**Error Response:**
```json
{
  "success": false,
  "message": "Error description",
  "timestamp": "2025-08-18T18:00:00Z",
  "error_code": "VALIDATION_ERROR",
  "details": {}
}
```

**HTTP Status Codes:**
- `200` - Success
- `201` - Created (for POST operations)
- `400` - Bad Request (invalid input)
- `404` - Not Found (session/resource not found)
- `413` - Payload Too Large (file size exceeded)
- `422` - Unprocessable Content (validation error)
- `500` - Internal Server Error

## Data Models

### Session Model
```typescript
interface SessionInfo {
  session_id: string;
  name?: string;
  description?: string;
  user_id?: string;
  status: 'active' | 'inactive' | 'error' | 'expired';
  created_at: string;
  last_activity: string;
  timeout_minutes: number;
  workspace_path: string;
  datasets: string[];
  message_count: number;
}
```

### Chat Message Model
```typescript
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  metadata?: Record<string, any>;
  plots?: string[];
}
```

### File Info Model
```typescript
interface FileInfo {
  name: string;
  path: string;
  size_bytes: number;
  created_at: string;
  modified_at: string;
  file_type: string;
}
```

### Plot Info Model
```typescript
interface PlotInfo {
  id: string;
  title: string;
  timestamp: string;
  source: string;
  format: 'png' | 'jpg' | 'svg' | 'html' | 'pdf';
  path: string;
  size_bytes?: number;
}
```

### WebSocket Event Model
```typescript
interface WebSocketMessage {
  event_type: 'chat_stream' | 'agent_thinking' | 'analysis_progress' | 
               'data_updated' | 'plot_generated' | 'error';
  session_id: string;
  data: Record<string, any>;
  timestamp: string;
}
```

## Supported File Types

**Upload Formats:**
- `.csv` - Comma-separated values
- `.tsv` - Tab-separated values  
- `.h5` - HDF5 format
- `.h5ad` - AnnData HDF5 format
- `.mtx` - Matrix Market format

**Plot Formats:**
- `.png` - PNG images
- `.jpg/.jpeg` - JPEG images
- `.svg` - Scalable Vector Graphics
- `.html` - Interactive HTML plots
- `.pdf` - PDF documents

## Common Usage Workflows

### 1. Basic Analysis Workflow
```javascript
// 1. Create session
const session = await createSession("RNA Analysis");
const sessionId = session.session.session_id;

// 2. Connect WebSocket
const ws = connectWebSocket(sessionId);

// 3. Upload data
await uploadFile(sessionId, dataFile);

// 4. Start analysis
await sendChatMessage(sessionId, "Perform clustering analysis");

// 5. Monitor progress via WebSocket
// 6. Download results when complete
```

### 2. GEO Dataset Analysis
```javascript
// 1. Create session
const sessionId = await createSession("GEO Analysis");

// 2. Download GEO dataset
await downloadGEODataset(sessionId, "GSE123456");

// 3. Start analysis
await sendChatMessage(sessionId, "Analyze this dataset and create UMAP visualization");

// 4. Get generated plots
const plots = await getPlots(sessionId);
```

### 3. Real-time Progress Monitoring
```javascript
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  
  // Update UI based on event type
  if (msg.event_type === 'analysis_progress') {
    setProgress(msg.data.message, msg.data.agent);
  } else if (msg.event_type === 'plot_generated') {
    refreshPlotsList();
  }
};
```

## Rate Limits & Constraints

- **File Upload**: 500MB maximum per file
- **Session Timeout**: 5-1440 minutes (configurable)
- **Export Expiry**: 24 hours
- **WebSocket**: No explicit rate limits
- **Concurrent Sessions**: No hard limit (resource dependent)

## Environment Variables

For frontend configuration:

```env
# API Base URL
REACT_APP_API_URL=http://localhost:8000

# WebSocket URL  
REACT_APP_WS_URL=ws://localhost:8000

# Upload limits
REACT_APP_MAX_FILE_SIZE=524288000  # 500MB in bytes
```

## Error Recovery Patterns

### Session Recovery
```javascript
// Check if session still exists
const sessionCheck = await fetch(`/api/v1/sessions/${sessionId}`);
if (!sessionCheck.ok) {
  // Session expired, create new one
  const newSession = await createSession();
  sessionId = newSession.session.session_id;
}
```

### WebSocket Reconnection
```javascript
ws.onclose = () => {
  // Attempt reconnection after delay
  setTimeout(() => {
    connectWebSocket(sessionId);
  }, 1000);
};
```

### File Upload Error Handling
```javascript
try {
  const result = await uploadFile(sessionId, file);
  if (!result.success) {
    showError(`Upload failed: ${result.message}`);
  }
} catch (error) {
  if (error.status === 413) {
    showError('File too large. Maximum size is 500MB');
  } else if (error.status === 422) {
    showError('Invalid file format');
  }
}
```

## Agent Capabilities

The multi-agent system includes:

1. **Supervisor Agent** - Coordinates other agents and routing
2. **Data Expert** - Data loading, preprocessing, quality control
3. **Transcriptomics Expert** - RNA-seq analysis, single-cell analysis
4. **Method Expert** - Statistical methods, clustering, visualization

**Agent Communication Examples:**
- "Load the GSE123456 dataset and perform quality control"
- "Create a UMAP visualization colored by cell types"
- "Perform differential expression analysis between conditions"
- "Generate a clustering analysis with Leiden algorithm"
- "Show me the data summary and basic statistics"

## Interactive API Documentation

**Access full interactive documentation at:**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc` 
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

These provide complete endpoint documentation with example requests/responses and the ability to test endpoints directly in the browser.
