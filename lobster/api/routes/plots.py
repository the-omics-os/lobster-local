"""
Lobster AI - Plot Management Routes
Plot retrieval, visualization, and export endpoints.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import FileResponse, JSONResponse

from lobster.api.models import (
    PlotInfo,
    PlotListResponse,
    BaseResponse
)
from lobster.api.session_manager import SessionManager
from lobster.core.api_client import APIAgentClient
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()

# Supported plot formats
SUPPORTED_PLOT_FORMATS = ['png', 'jpg', 'jpeg', 'svg', 'html', 'pdf']


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get the session manager from app state."""
    return request.app.state.session_manager


@router.get("/sessions/{session_id}/plots", response_model=PlotListResponse)
async def get_plots(
    session_id: UUID,
    limit: Optional[int] = Query(None, ge=1, le=100, description="Maximum number of plots to return"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get list of generated plots for the session.
    
    Returns information about all plots generated during the session,
    including their paths, timestamps, and metadata.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get or create API agent client
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                workspace_path=session.workspace_path
            )
        
        # Get plot files from workspace
        plot_files = session._api_client.list_workspace_files(directory="plots")
        
        # Convert to PlotInfo objects
        plots = []
        for file_data in plot_files:
            # Extract plot format from file extension
            file_path = Path(file_data["name"])
            plot_format = file_path.suffix.lower().lstrip('.')
            if plot_format not in SUPPORTED_PLOT_FORMATS:
                plot_format = "png"  # Default format
            
            plot_info = PlotInfo(
                id=file_data["name"],  # Use filename as ID
                title=file_path.stem,  # Use filename without extension as title
                timestamp=datetime.fromisoformat(file_data["modified"]),
                source="agent_generated",
                format=plot_format,
                path=file_data["path"],
                size_bytes=file_data["size"]
            )
            plots.append(plot_info)
        
        # Sort by timestamp (most recent first)
        plots.sort(key=lambda p: p.timestamp, reverse=True)
        
        # Apply limit if specified
        if limit:
            plots = plots[:limit]
        
        # Update session activity
        session.update_activity()
        
        # Create response
        response = PlotListResponse(
            success=True,
            message=f"Retrieved {len(plots)} plots",
            plots=plots,
            total=len(plots)
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plots for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get plots: {str(e)}"
        )


@router.get("/sessions/{session_id}/plots/{plot_id}/download")
async def download_plot(
    session_id: UUID,
    plot_id: str,
    format: Optional[str] = Query(None, description="Output format (png, jpg, svg, html, pdf)"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Download a specific plot file.
    
    Args:
        session_id: Session UUID
        plot_id: Plot identifier (filename)
        format: Optional format conversion (if supported)
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Construct plot file path
        plot_path = session.workspace_path / "plots" / plot_id
        
        # Security check: ensure the file is within the plots directory
        try:
            plot_path.resolve().relative_to((session.workspace_path / "plots").resolve())
        except ValueError:
            raise HTTPException(
                status_code=403,
                detail="Access denied: Plot path outside workspace"
            )
        
        # Check if plot file exists
        if not plot_path.exists() or not plot_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Plot not found"
            )
        
        # Update session activity
        session.update_activity()
        
        # Determine media type based on file extension
        file_extension = plot_path.suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.html': 'text/html',
            '.pdf': 'application/pdf'
        }
        
        media_type = media_type_map.get(file_extension, 'application/octet-stream')
        
        # Return file response
        return FileResponse(
            path=str(plot_path),
            filename=plot_path.name,
            media_type=media_type
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading plot for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download plot: {str(e)}"
        )


@router.get("/sessions/{session_id}/plots/{plot_id}/info")
async def get_plot_info(
    session_id: UUID,
    plot_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get detailed information about a specific plot.
    
    Args:
        session_id: Session UUID
        plot_id: Plot identifier (filename)
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Construct plot file path
        plot_path = session.workspace_path / "plots" / plot_id
        
        # Security check: ensure the file is within the plots directory
        try:
            plot_path.resolve().relative_to((session.workspace_path / "plots").resolve())
        except ValueError:
            raise HTTPException(
                status_code=403,
                detail="Access denied: Plot path outside workspace"
            )
        
        # Check if plot file exists
        if not plot_path.exists() or not plot_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Plot not found"
            )
        
        # Get file stats
        stat = plot_path.stat()
        file_extension = plot_path.suffix.lower().lstrip('.')
        
        # Create plot info
        plot_info = PlotInfo(
            id=plot_id,
            title=plot_path.stem,
            timestamp=datetime.fromtimestamp(stat.st_mtime),
            source="agent_generated",
            format=file_extension if file_extension in SUPPORTED_PLOT_FORMATS else "png",
            path=f"plots/{plot_id}",
            size_bytes=stat.st_size
        )
        
        # Update session activity
        session.update_activity()
        
        return {
            "success": True,
            "message": "Plot information retrieved successfully",
            "plot": plot_info.dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plot info for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get plot information: {str(e)}"
        )


@router.delete("/sessions/{session_id}/plots/{plot_id}", response_model=BaseResponse)
async def delete_plot(
    session_id: UUID,
    plot_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Delete a specific plot file.
    
    Args:
        session_id: Session UUID
        plot_id: Plot identifier (filename)
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Construct plot file path
        plot_path = session.workspace_path / "plots" / plot_id
        
        # Security check: ensure the file is within the plots directory
        try:
            plot_path.resolve().relative_to((session.workspace_path / "plots").resolve())
        except ValueError:
            raise HTTPException(
                status_code=403,
                detail="Access denied: Plot path outside workspace"
            )
        
        # Check if plot file exists
        if not plot_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Plot not found"
            )
        
        # Delete the plot file
        plot_path.unlink()
        
        # Update session activity
        session.update_activity()
        
        logger.info(f"Deleted plot {plot_id} from session {session_id}")
        
        return BaseResponse(
            success=True,
            message=f"Plot {plot_id} deleted successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting plot for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete plot: {str(e)}"
        )


@router.delete("/sessions/{session_id}/plots", response_model=BaseResponse)
async def clear_all_plots(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Clear all plots from the session.
    
    Removes all generated plot files from the session workspace.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get plots directory
        plots_dir = session.workspace_path / "plots"
        
        if not plots_dir.exists():
            return BaseResponse(
                success=True,
                message="No plots directory found"
            )
        
        # Count and delete all plot files
        deleted_count = 0
        for plot_file in plots_dir.glob("*"):
            if plot_file.is_file():
                try:
                    plot_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Error deleting plot file {plot_file}: {e}")
        
        # Update session activity
        session.update_activity()
        
        logger.info(f"Cleared {deleted_count} plots from session {session_id}")
        
        return BaseResponse(
            success=True,
            message=f"Cleared {deleted_count} plots successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing plots for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear plots: {str(e)}"
        )


@router.post("/sessions/{session_id}/plots/generate")
async def generate_plot(
    session_id: UUID,
    plot_request: Dict[str, Any],  # Flexible plot generation request
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Generate a plot through the agent system.
    
    This endpoint triggers plot generation using the visualization capabilities
    of the multi-agent system.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get or create API agent client
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                workspace_path=session.workspace_path
            )
        
        # Extract plot parameters
        plot_type = plot_request.get("plot_type", "visualization")
        plot_description = plot_request.get("description", "")
        parameters = plot_request.get("parameters", {})
        
        # Construct query for plot generation
        plot_query = f"Please create a {plot_type}"
        if plot_description:
            plot_query += f" showing {plot_description}"
        
        # Add parameters to query if provided
        if parameters:
            params_str = ", ".join([f"{k}={v}" for k, v in parameters.items()])
            plot_query += f" with parameters: {params_str}"
        
        # Execute plot generation through the agent system
        result = await session._api_client.query(
            user_input=plot_query,
            stream=True
        )
        
        # Update session activity
        session.update_activity()
        
        return {
            "success": result.get("success", True),
            "message": f"Plot generation request processed: {plot_type}",
            "session_id": str(session_id),
            "plot_type": plot_type,
            "agent_response": result.get("response", ""),
            "plots_generated": result.get("plots", []),
            "duration": result.get("duration", 0)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating plot for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate plot: {str(e)}"
        )


@router.get("/sessions/{session_id}/plots/latest")
async def get_latest_plots(
    session_id: UUID,
    count: int = Query(5, ge=1, le=20, description="Number of latest plots to return"),
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get the most recently generated plots for the session.
    
    Returns the latest plots sorted by creation time.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get or create API agent client
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                workspace_path=session.workspace_path
            )
        
        # Try to get latest plots from data manager
        latest_plots = []
        try:
            if session._api_client.data_manager.has_data():
                plots_from_dm = session._api_client.data_manager.get_latest_plots(count)
                
                # Convert to API format
                for plot_path in plots_from_dm:
                    plot_file = Path(plot_path)
                    if plot_file.exists():
                        stat = plot_file.stat()
                        plot_info = {
                            "id": plot_file.name,
                            "title": plot_file.stem,
                            "path": str(plot_file.relative_to(session.workspace_path)),
                            "format": plot_file.suffix.lower().lstrip('.'),
                            "size_bytes": stat.st_size,
                            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                        latest_plots.append(plot_info)
        
        except Exception as dm_error:
            logger.warning(f"Error getting plots from data manager: {dm_error}")
            
            # Fallback: get plots from file system
            plot_files = session._api_client.list_workspace_files(directory="plots")
            plot_files.sort(key=lambda f: f["modified"], reverse=True)
            
            for file_data in plot_files[:count]:
                latest_plots.append({
                    "id": file_data["name"],
                    "title": Path(file_data["name"]).stem,
                    "path": file_data["path"],
                    "format": Path(file_data["name"]).suffix.lower().lstrip('.'),
                    "size_bytes": file_data["size"],
                    "created_at": file_data["modified"],
                    "modified_at": file_data["modified"]
                })
        
        # Update session activity
        session.update_activity()
        
        return {
            "success": True,
            "message": f"Retrieved {len(latest_plots)} latest plots",
            "session_id": str(session_id),
            "plots": latest_plots,
            "count": len(latest_plots)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest plots for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest plots: {str(e)}"
        )
