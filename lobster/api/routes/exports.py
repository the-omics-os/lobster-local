"""
Lobster AI - Export and Download Routes
Data export, session packaging, and download endpoints.
"""

import os
import zipfile
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4
from pathlib import Path
import tempfile
import shutil
import json

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import FileResponse

from lobster.api.models import (
    ExportRequest,
    ExportResponse,
    BaseResponse
)
from lobster.api.session_manager import SessionManager
from lobster.core.api_client import APIAgentClient
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()

# Export storage directory
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

# Export expiration time (24 hours)
EXPORT_EXPIRY_HOURS = 24


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get the session manager from app state."""
    return request.app.state.session_manager


async def cleanup_expired_exports():
    """Background task to clean up expired export files."""
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=EXPORT_EXPIRY_HOURS)
        
        for export_file in EXPORT_DIR.glob("*.zip"):
            try:
                # Check file modification time
                file_time = datetime.fromtimestamp(export_file.stat().st_mtime)
                if file_time < cutoff_time:
                    export_file.unlink()
                    logger.info(f"Cleaned up expired export: {export_file.name}")
            except Exception as e:
                logger.warning(f"Error cleaning up export file {export_file}: {e}")
    
    except Exception as e:
        logger.error(f"Error in export cleanup task: {e}")


@router.post("/sessions/{session_id}/export", response_model=ExportResponse)
async def create_export(
    session_id: UUID,
    export_request: ExportRequest,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Create a downloadable export package of session data.
    
    Creates a ZIP or TAR archive containing:
    - Session data files (if requested)
    - Generated plots (if requested)  
    - Analysis logs (if requested)
    - Session metadata and conversation history
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Generate unique export ID
        export_id = uuid4()
        export_filename = f"lobster_export_{session_id.hex[:8]}_{export_id.hex[:8]}.{export_request.format}"
        export_path = EXPORT_DIR / export_filename
        
        # Get or create API agent client to access export functionality
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                workspace_path=session.workspace_path
            )
        
        # Create export package
        try:
            if export_request.format == "zip":
                await _create_zip_export(
                    session, 
                    export_path, 
                    export_request,
                    session._api_client
                )
            elif export_request.format == "tar":
                await _create_tar_export(
                    session,
                    export_path,
                    export_request,
                    session._api_client
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported export format"
                )
        
        except Exception as export_error:
            logger.error(f"Error creating export package: {export_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create export package: {str(export_error)}"
            )
        
        # Schedule cleanup task
        background_tasks.add_task(cleanup_expired_exports)
        
        # Update session activity
        session.update_activity()
        
        # Calculate expiry time
        expires_at = datetime.utcnow() + timedelta(hours=EXPORT_EXPIRY_HOURS)
        
        # Create response
        response = ExportResponse(
            success=True,
            message="Export package created successfully",
            export_id=export_id,
            download_url=f"/api/v1/exports/{export_id}/download",
            expires_at=expires_at,
            size_bytes=export_path.stat().st_size if export_path.exists() else 0
        )
        
        logger.info(f"Created export package for session {session_id}: {export_filename}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating export for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create export package: {str(e)}"
        )


async def _create_zip_export(
    session, 
    export_path: Path, 
    export_request: ExportRequest,
    api_client: APIAgentClient
):
    """Create a ZIP export package."""
    with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add session metadata
        session_info = session.to_session_info()
        metadata = {
            "session": session_info.dict(),
            "export_created": datetime.utcnow().isoformat(),
            "export_request": export_request.dict()
        }
        
        # Write metadata to zip
        zipf.writestr("session_metadata.json", 
                     json.dumps(metadata, indent=2, default=str))
        
        # Add conversation history
        history = api_client.get_conversation_history()
        zipf.writestr("conversation_history.json",
                     json.dumps(history, indent=2, default=str))
        
        # Add data files if requested
        if export_request.include_data:
            data_dir = session.workspace_path / "data"
            if data_dir.exists():
                for file_path in data_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = f"data/{file_path.relative_to(data_dir)}"
                        zipf.write(file_path, arcname)
        
        # Add plots if requested
        if export_request.include_plots:
            plots_dir = session.workspace_path / "plots"
            if plots_dir.exists():
                for file_path in plots_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = f"plots/{file_path.relative_to(plots_dir)}"
                        zipf.write(file_path, arcname)
        
        # Add logs if requested
        if export_request.include_logs:
            # Add basic session log info
            log_info = {
                "session_id": str(session.session_id),
                "created_at": session.created_at.isoformat(),
                "message_count": session.message_count,
                "datasets": session.datasets,
                "workspace_path": str(session.workspace_path)
            }
            zipf.writestr("session_log.json",
                         json.dumps(log_info, indent=2, default=str))


async def _create_tar_export(
    session,
    export_path: Path, 
    export_request: ExportRequest,
    api_client: APIAgentClient
):
    """Create a TAR export package."""
    import tarfile
    
    with tarfile.open(export_path, 'w:gz') as tarf:
        # Create temporary directory for metadata files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create session metadata
            session_info = session.to_session_info()
            metadata = {
                "session": session_info.dict(),
                "export_created": datetime.utcnow().isoformat(),
                "export_request": export_request.dict()
            }
            
            # Write metadata file
            metadata_file = temp_path / "session_metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, default=str))
            tarf.add(metadata_file, arcname="session_metadata.json")
            
            # Add conversation history
            history = api_client.get_conversation_history()
            history_file = temp_path / "conversation_history.json"
            history_file.write_text(json.dumps(history, indent=2, default=str))
            tarf.add(history_file, arcname="conversation_history.json")
            
            # Add data files if requested
            if export_request.include_data:
                data_dir = session.workspace_path / "data"
                if data_dir.exists():
                    tarf.add(data_dir, arcname="data")
            
            # Add plots if requested
            if export_request.include_plots:
                plots_dir = session.workspace_path / "plots"
                if plots_dir.exists():
                    tarf.add(plots_dir, arcname="plots")


@router.get("/exports/{export_id}/download")
async def download_export(
    export_id: UUID,
    background_tasks: BackgroundTasks
):
    """
    Download an export package.
    
    Downloads the previously created export package by ID.
    """
    try:
        # Find the export file
        export_files = list(EXPORT_DIR.glob(f"*{export_id.hex[:8]}*"))
        
        if not export_files:
            raise HTTPException(
                status_code=404,
                detail="Export package not found or expired"
            )
        
        export_file = export_files[0]  # Take the first match
        
        # Check if export has expired
        file_time = datetime.fromtimestamp(export_file.stat().st_mtime)
        if datetime.utcnow() - file_time > timedelta(hours=EXPORT_EXPIRY_HOURS):
            # Clean up expired file
            export_file.unlink()
            raise HTTPException(
                status_code=410,
                detail="Export package has expired"
            )
        
        # Schedule cleanup after download
        background_tasks.add_task(cleanup_expired_exports)
        
        # Return file response
        return FileResponse(
            path=str(export_file),
            filename=export_file.name,
            media_type='application/octet-stream'
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading export {export_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download export: {str(e)}"
        )


@router.get("/exports/{export_id}/info")
async def get_export_info(export_id: UUID):
    """
    Get information about an export package.
    
    Returns metadata about the export without downloading it.
    """
    try:
        # Find the export file
        export_files = list(EXPORT_DIR.glob(f"*{export_id.hex[:8]}*"))
        
        if not export_files:
            raise HTTPException(
                status_code=404,
                detail="Export package not found"
            )
        
        export_file = export_files[0]
        
        # Get file stats
        stat = export_file.stat()
        file_time = datetime.fromtimestamp(stat.st_mtime)
        expires_at = file_time + timedelta(hours=EXPORT_EXPIRY_HOURS)
        
        # Check if expired
        is_expired = datetime.utcnow() > expires_at
        
        return {
            "success": True,
            "message": "Export information retrieved",
            "export_id": str(export_id),
            "filename": export_file.name,
            "size_bytes": stat.st_size,
            "created_at": file_time.isoformat(),
            "expires_at": expires_at.isoformat(),
            "is_expired": is_expired,
            "download_url": f"/api/v1/exports/{export_id}/download"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting export info {export_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get export information: {str(e)}"
        )


@router.delete("/exports/{export_id}", response_model=BaseResponse)
async def delete_export(export_id: UUID):
    """
    Delete an export package.
    
    Removes the export file from storage.
    """
    try:
        # Find the export file
        export_files = list(EXPORT_DIR.glob(f"*{export_id.hex[:8]}*"))
        
        if not export_files:
            raise HTTPException(
                status_code=404,
                detail="Export package not found"
            )
        
        export_file = export_files[0]
        
        # Delete the file
        export_file.unlink()
        
        logger.info(f"Deleted export package: {export_file.name}")
        
        return BaseResponse(
            success=True,
            message="Export package deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting export {export_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete export: {str(e)}"
        )
