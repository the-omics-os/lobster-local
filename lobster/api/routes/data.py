"""
Lobster AI - Data Management Routes
Data operations, GEO downloads, and dataset management endpoints.
"""

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Request, Query
from fastapi.responses import JSONResponse

from lobster.api.models import (
    GEODownloadRequest,
    GEODownloadResponse,
    DataOperationRequest,
    DataOperationResponse,
    DatasetInfo,
    DatasetStatus,
    FileType,
    WorkspaceInfo,
    WorkspaceResponse,
    BaseResponse
)
from lobster.api.session_manager import SessionManager
from lobster.core.api_client import APIAgentClient
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get the session manager from app state."""
    return request.app.state.session_manager


@router.post("/sessions/{session_id}/data/geo-download", response_model=GEODownloadResponse)
async def download_geo_dataset(
    session_id: UUID,
    geo_request: GEODownloadRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Download a GEO dataset for analysis in the session.
    
    This endpoint downloads and processes GEO datasets using the specialized
    GEO service tools integrated with the multi-agent system.
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
        
        # Initiate GEO download through the agent system
        result = await session._api_client.download_geo_dataset(geo_request.geo_id)
        
        # Update session activity
        session.update_activity()
        
        # Add GEO dataset to session datasets list
        if result.get("success") and geo_request.geo_id not in session.datasets:
            session.datasets.append(geo_request.geo_id)
        
        # Create dataset info if successful
        dataset = None
        if result.get("success"):
            dataset = DatasetInfo(
                name=geo_request.geo_id,
                path=geo_request.destination_path or f"data/{geo_request.geo_id}",
                file_type=FileType.GEO,
                size_bytes=0,  # Will be updated when download completes
                status=DatasetStatus.LOADING,
                created_at=session.created_at,
                metadata={
                    "geo_id": geo_request.geo_id,
                    "source": "GEO",
                    "download_initiated": True
                }
            )
        
        # Create response
        response = GEODownloadResponse(
            success=result.get("success", True),
            message=result.get("message", f"GEO dataset {geo_request.geo_id} download initiated"),
            geo_id=geo_request.geo_id,
            download_path=geo_request.destination_path or f"data/{geo_request.geo_id}",
            dataset=dataset
        )
        
        logger.info(f"GEO download initiated for session {session_id}: {geo_request.geo_id}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading GEO dataset for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download GEO dataset: {str(e)}"
        )


@router.get("/sessions/{session_id}/data/datasets")
async def list_datasets(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    List all datasets available in the session.
    
    Returns information about loaded datasets, their status, and metadata.
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
        
        # Get data status from agent client
        status = session._api_client.get_status()
        data_manager_status = status.get("data_manager_status", {})
        has_data = data_manager_status.get("has_data", False)
        
        # Build dataset list
        datasets = []
        if has_data and session._api_client.data_manager.has_data():
            # Get data summary from the data manager
            try:
                data_summary = session._api_client.data_manager.get_data_summary()
                
                # Create dataset info from current loaded data
                dataset_info = DatasetInfo(
                    name="current_dataset",
                    path=data_manager_status.get("data_dir", ""),
                    file_type=FileType.CSV,  # Default, could be enhanced
                    size_bytes=0,  # Could be calculated
                    status=DatasetStatus.READY,
                    created_at=session.created_at,
                    metadata={
                        "summary": data_summary,
                        "session_id": str(session_id)
                    }
                )
                datasets.append(dataset_info)
            except Exception as e:
                logger.warning(f"Error getting data summary: {e}")
        
        # Update session activity
        session.update_activity()
        
        return {
            "success": True,
            "message": f"Retrieved {len(datasets)} datasets",
            "session_id": str(session_id),
            "datasets": [dataset.dict() for dataset in datasets],
            "total": len(datasets),
            "has_data": has_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing datasets for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/sessions/{session_id}/data/workspace", response_model=WorkspaceResponse)
async def get_workspace_info(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get comprehensive workspace information for the session.
    
    Returns details about all files, directories, and organization within the workspace.
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
        
        # Get workspace file listings by directory
        data_files = session._api_client.list_workspace_files(directory="data")
        plot_files = session._api_client.list_workspace_files(directory="plots")
        export_files = session._api_client.list_workspace_files(directory="exports")
        
        # Calculate totals
        all_files = data_files + plot_files + export_files
        total_files = len(all_files)
        total_size = sum(file_data["size"] for file_data in all_files)
        
        # Create workspace info
        workspace_info = WorkspaceInfo(
            path=str(session.workspace_path),
            total_files=total_files,
            total_size_bytes=total_size,
            data_files=data_files,
            plot_files=plot_files,
            export_files=export_files
        )
        
        # Update session activity
        session.update_activity()
        
        # Create response
        response = WorkspaceResponse(
            success=True,
            message="Workspace information retrieved successfully",
            workspace=workspace_info
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace info for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get workspace information: {str(e)}"
        )


@router.post("/sessions/{session_id}/data/operations", response_model=DataOperationResponse)
async def perform_data_operation(
    session_id: UUID,
    operation_request: DataOperationRequest,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Perform a data operation through the agent system.
    
    This endpoint allows triggering specific data operations like:
    - Quality control analysis
    - Clustering analysis
    - Differential expression
    - Data preprocessing
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
        
        # Construct query for the operation
        operation_query = f"Please perform {operation_request.operation} analysis on dataset {operation_request.dataset_name}"
        
        # Add parameters to query if provided
        if operation_request.parameters:
            params_str = ", ".join([f"{k}={v}" for k, v in operation_request.parameters.items()])
            operation_query += f" with parameters: {params_str}"
        
        # Execute the operation through the agent system
        result = await session._api_client.query(
            user_input=operation_query,
            stream=True
        )
        
        # Update session activity
        session.update_activity()
        
        # Create response
        response = DataOperationResponse(
            success=result.get("success", True),
            message=f"Data operation '{operation_request.operation}' completed",
            result={
                "operation": operation_request.operation,
                "dataset": operation_request.dataset_name,
                "parameters": operation_request.parameters,
                "agent_response": result.get("response", ""),
                "duration": result.get("duration", 0),
                "has_data": result.get("has_data", False)
            },
            output_files=result.get("plots", [])
        )
        
        logger.info(f"Data operation completed for session {session_id}: {operation_request.operation}")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing data operation for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform data operation: {str(e)}"
        )


@router.get("/sessions/{session_id}/data/summary")
async def get_data_summary(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get a summary of the current loaded dataset in the session.
    
    Returns information about the dataset structure, dimensions, and basic statistics.
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
        
        # Check if data is loaded
        if not session._api_client.data_manager.has_data():
            return {
                "success": True,
                "message": "No dataset currently loaded",
                "session_id": str(session_id),
                "has_data": False,
                "summary": None
            }
        
        # Get data summary
        try:
            data_summary = session._api_client.data_manager.get_data_summary()
            
            # Update session activity
            session.update_activity()
            
            return {
                "success": True,
                "message": "Data summary retrieved successfully",
                "session_id": str(session_id),
                "has_data": True,
                "summary": data_summary
            }
        
        except Exception as summary_error:
            logger.warning(f"Error getting data summary: {summary_error}")
            return {
                "success": True,
                "message": "Data is loaded but summary unavailable",
                "session_id": str(session_id),
                "has_data": True,
                "summary": None,
                "error": str(summary_error)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data summary for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get data summary: {str(e)}"
        )


@router.delete("/sessions/{session_id}/data/current", response_model=BaseResponse)
async def clear_current_data(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Clear the currently loaded dataset from the session.
    
    This removes the dataset from memory but keeps files on disk.
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
        
        # Check if data is loaded
        if not session._api_client.data_manager.has_data():
            return BaseResponse(
                success=True,
                message="No dataset currently loaded to clear"
            )
        
        # Clear the data from the data manager
        try:
            # Reset the data manager to clear loaded data
            session._api_client.data_manager.reset()
            
            # Update session activity
            session.update_activity()
            
            logger.info(f"Cleared current dataset for session {session_id}")
            
            return BaseResponse(
                success=True,
                message="Current dataset cleared successfully"
            )
        
        except Exception as clear_error:
            logger.error(f"Error clearing data: {clear_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear data: {str(clear_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing data for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear current data: {str(e)}"
        )


@router.post("/sessions/{session_id}/data/load")
async def load_data_file(
    session_id: UUID,
    file_request: Dict[str, str],  # Simple dict with file_path
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Load a specific file as the active dataset.
    
    This endpoint loads an uploaded file into the data manager for analysis.
    """
    try:
        # Get the session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Session not found or expired"
            )
        
        # Get file path from request
        file_path = file_request.get("file_path")
        if not file_path:
            raise HTTPException(
                status_code=400,
                detail="file_path is required"
            )
        
        # Get or create API agent client
        if not hasattr(session, '_api_client') or session._api_client is None:
            session._api_client = APIAgentClient(
                session_id=session_id,
                session_manager=session_manager,
                workspace_path=session.workspace_path
            )
        
        # Construct full file path
        full_path = session.workspace_path / file_path
        
        # Security check: ensure the file is within the workspace
        try:
            full_path.resolve().relative_to(session.workspace_path.resolve())
        except ValueError:
            raise HTTPException(
                status_code=403,
                detail="Access denied: File path outside workspace"
            )
        
        # Check if file exists
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        
        # Load the data
        try:
            session._api_client.data_manager.set_data(str(full_path))
            
            # Notify about data update
            await session._api_client._notify_data_updates()
            
            # Update session activity
            session.update_activity()
            
            # Add to datasets list if not already there
            filename = full_path.name
            if filename not in session.datasets:
                session.datasets.append(filename)
            
            logger.info(f"Loaded data file for session {session_id}: {file_path}")
            
            return {
                "success": True,
                "message": f"Data file {file_path} loaded successfully",
                "session_id": str(session_id),
                "file_path": file_path,
                "has_data": True
            }
        
        except Exception as load_error:
            logger.error(f"Error loading data file: {load_error}")
            raise HTTPException(
                status_code=422,
                detail=f"Failed to load data file: {str(load_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading data file for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load data file: {str(e)}"
        )


@router.get("/sessions/{session_id}/data/status")
async def get_data_status(
    session_id: UUID,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get the current data loading status for the session.
    
    Returns information about loaded datasets, processing status, and capabilities.
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
        
        # Get comprehensive status
        client_status = session._api_client.get_status()
        data_manager_status = client_status.get("data_manager_status", {})
        
        # Update session activity
        session.update_activity()
        
        return {
            "success": True,
            "message": "Data status retrieved successfully",
            "session_id": str(session_id),
            "has_data": data_manager_status.get("has_data", False),
            "data_manager": data_manager_status,
            "datasets": session.datasets,
            "workspace_path": str(session.workspace_path),
            "message_count": session.message_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data status for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get data status: {str(e)}"
        )
