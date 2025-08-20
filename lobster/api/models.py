"""
Lobster AI - API Data Models
Pydantic models for request/response validation and serialization.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    EXPIRED = "expired"


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"


class DatasetStatus(str, Enum):
    """Dataset status enumeration."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    PROCESSING = "processing"


class FileType(str, Enum):
    """Supported file types."""
    CSV = "csv"
    TSV = "tsv"
    H5 = "h5"
    H5AD = "h5ad"
    MTX = "mtx"
    GEO = "geo"


class WSEventType(str, Enum):
    """WebSocket event types."""
    CHAT_STREAM = "chat_stream"
    AGENT_THINKING = "agent_thinking"
    ANALYSIS_PROGRESS = "analysis_progress"
    DATA_UPDATED = "data_updated"
    PLOT_GENERATED = "plot_generated"
    ERROR = "error"


# Base Models
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Session Models
class SessionCreate(BaseModel):
    """Session creation request."""
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: Optional[str] = None
    timeout_minutes: Optional[int] = Field(default=30, ge=5, le=1440)


class SessionUpdate(BaseModel):
    """Session update request."""
    name: Optional[str] = None
    description: Optional[str] = None
    timeout_minutes: Optional[int] = Field(None, ge=5, le=1440)


class SessionInfo(BaseModel):
    """Session information response."""
    session_id: UUID
    name: Optional[str]
    description: Optional[str]
    user_id: Optional[str]
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    timeout_minutes: int
    workspace_path: str
    datasets: List[str] = Field(default_factory=list)
    message_count: int = 0
    
    class Config:
        use_enum_values = True


class SessionResponse(BaseResponse):
    """Session operation response."""
    session: Optional[SessionInfo] = None


class SessionListResponse(BaseResponse):
    """Session list response."""
    sessions: List[SessionInfo]
    total: int


# Message Models
class ChatMessage(BaseModel):
    """Chat message model."""
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
    plots: Optional[List[str]] = None  # List of plot IDs
    
    class Config:
        use_enum_values = True


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str
    stream: bool = Field(default=True)
    include_history: bool = Field(default=True)
    max_history: int = Field(default=10, ge=1, le=100)


class ChatResponse(BaseResponse):
    """Chat response model."""
    chat_message: Optional[ChatMessage] = None
    conversation_id: Optional[UUID] = None
    plots: Optional[List[str]] = None
    data_updated: bool = False


# WebSocket Models
class WSMessage(BaseModel):
    """WebSocket message model."""
    event_type: WSEventType
    session_id: UUID
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


# Data Models
class DatasetInfo(BaseModel):
    """Dataset information model."""
    name: str
    path: str
    file_type: FileType
    size_bytes: int
    status: DatasetStatus
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    
    class Config:
        use_enum_values = True


class FileUploadResponse(BaseResponse):
    """File upload response."""
    file_path: str
    file_size: int
    dataset: Optional[DatasetInfo] = None


class DataOperationRequest(BaseModel):
    """Data operation request."""
    dataset_name: str
    operation: str
    parameters: Optional[Dict[str, Any]] = None


class DataOperationResponse(BaseResponse):
    """Data operation response."""
    result: Optional[Dict[str, Any]] = None
    output_files: Optional[List[str]] = None


# GEO Data Models
class GEODownloadRequest(BaseModel):
    """GEO dataset download request."""
    geo_id: str = Field(..., pattern=r'^GSE\d+$')
    destination_path: Optional[str] = None
    
    @validator('geo_id')
    def validate_geo_id(cls, v):
        if not v.startswith('GSE'):
            raise ValueError('GEO ID must start with "GSE"')
        return v.upper()


class GEODownloadResponse(BaseResponse):
    """GEO download response."""
    geo_id: str
    download_path: str
    dataset: Optional[DatasetInfo] = None


# Analysis Models
class AnalysisRequest(BaseModel):
    """Analysis request model."""
    dataset_name: str
    analysis_type: str
    parameters: Dict[str, Any]


class AnalysisResponse(BaseResponse):
    """Analysis response model."""
    analysis_id: UUID
    status: str
    result: Optional[Dict[str, Any]] = None
    plots: Optional[List[str]] = None


# Plot Models
class PlotInfo(BaseModel):
    """Plot information model."""
    id: str
    title: str
    timestamp: datetime
    source: str
    format: str = "png"
    path: str
    size_bytes: Optional[int] = None


class PlotListResponse(BaseResponse):
    """Plot list response."""
    plots: List[PlotInfo]
    total: int


# System Models
class SystemHealth(BaseModel):
    """System health status."""
    status: str
    active_sessions: int
    total_sessions: int
    uptime_seconds: float
    memory_usage: Dict[str, Any]
    disk_usage: Dict[str, Any]


class HealthResponse(BaseResponse):
    """Health check response."""
    system: SystemHealth
    version: str
    environment: str


# Workspace Models
class WorkspaceInfo(BaseModel):
    """Workspace information."""
    path: str
    total_files: int
    total_size_bytes: int
    data_files: List[Dict[str, Any]]
    plot_files: List[Dict[str, Any]]
    export_files: List[Dict[str, Any]]


class WorkspaceResponse(BaseResponse):
    """Workspace information response."""
    workspace: WorkspaceInfo


# File Models
class FileInfo(BaseModel):
    """File information model."""
    name: str
    path: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    file_type: str


class FileListResponse(BaseResponse):
    """File list response."""
    files: List[FileInfo]
    total: int
    total_size_bytes: int


# Export Models
class ExportRequest(BaseModel):
    """Data export request."""
    include_data: bool = True
    include_plots: bool = True
    include_logs: bool = False
    format: str = Field(default="zip", pattern=r'^(zip|tar)$')


class ExportResponse(BaseResponse):
    """Export response."""
    export_id: UUID
    download_url: str
    expires_at: datetime
    size_bytes: int


# File Metadata and Preview Models
class WorkspaceFileMetadata(BaseModel):
    """Workspace file metadata model for performance-optimized file listing."""
    name: str
    path: str
    relative_path: str
    directory: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    file_type: str
    file_format: str
    is_data_file: bool
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    session_id: Optional[str] = None
    
    # Additional metadata for specific formats
    has_header: Optional[bool] = None
    delimiter: Optional[str] = None
    compressed: Optional[bool] = None
    format_info: Optional[Dict[str, Any]] = None
    analysis_error: Optional[str] = None


class FilePreviewRequest(BaseModel):
    """File preview request model."""
    file_path: str = Field(..., description="Relative path to the file within the session workspace")
    max_rows: int = Field(default=10, ge=1, le=100, description="Maximum number of rows to include in preview")
    max_columns: int = Field(default=20, ge=1, le=50, description="Maximum number of columns to include in preview")


class FilePreviewData(BaseModel):
    """File preview data structure."""
    headers: List[str]
    rows: List[List[str]]
    total_rows: Union[int, str]  # Can be 'unknown' for compressed files
    total_columns: int
    is_truncated: bool
    preview_rows: int
    preview_columns: int
    delimiter: Optional[str] = None
    format_info: Optional[Dict[str, Any]] = None


class FilePreviewResponse(BaseResponse):
    """File preview response model."""
    preview_data: Optional[FilePreviewData] = None
    file_info: Optional[WorkspaceFileMetadata] = None


class WorkspaceFileListResponse(BaseResponse):
    """Workspace file list response with metadata-only approach."""
    files: List[WorkspaceFileMetadata]
    total_files: int
    total_size_bytes: int
    data_files_count: int = 0
    plot_files_count: int = 0
    other_files_count: int = 0
