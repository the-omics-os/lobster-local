"""
Lobster AI - Health Check Routes
System health monitoring and status endpoints.
"""

import os
import psutil
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from lobster.api.models import HealthResponse, SystemHealth
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Create router instance
router = APIRouter()

# Track application startup time
_startup_time = time.time()


def get_system_metrics():
    """Get system resource usage metrics."""
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent,
            "used_gb": round(memory.used / (1024**3), 2)
        }
        
        # Disk usage for current directory
        disk = psutil.disk_usage(Path.cwd())
        disk_info = {
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 1),
            "used_gb": round(disk.used / (1024**3), 2)
        }
        
        return memory_info, disk_info
        
    except Exception as e:
        logger.warning(f"Could not get system metrics: {e}")
        return {}, {}


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns system status, resource usage, and application metrics.
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - _startup_time
        
        # Get system metrics
        memory_usage, disk_usage = get_system_metrics()
        
        # For now, we'll get session count from the session manager if available
        # This will be updated when session manager is integrated
        active_sessions = 0
        total_sessions = 0
        
        # Create system health object
        system_health = SystemHealth(
            status="healthy",
            active_sessions=active_sessions,
            total_sessions=total_sessions,
            uptime_seconds=uptime_seconds,
            memory_usage=memory_usage,
            disk_usage=disk_usage
        )
        
        # Create response
        response = HealthResponse(
            success=True,
            message="System is healthy",
            system=system_health,
            version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        
        # Return degraded health status
        system_health = SystemHealth(
            status="degraded",
            active_sessions=0,
            total_sessions=0,
            uptime_seconds=0,
            memory_usage={},
            disk_usage={}
        )
        
        response = HealthResponse(
            success=False,
            message=f"Health check failed: {str(e)}",
            system=system_health,
            version="1.0.0",
            environment=os.getenv("ENVIRONMENT", "development")
        )
        
        return JSONResponse(
            status_code=503,
            content=response.dict()
        )


@router.get("/health/simple")
async def simple_health_check():
    """
    Simple health check endpoint for load balancers.
    
    Returns basic OK status without detailed metrics.
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health/readiness")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration.
    
    Checks if the application is ready to serve requests.
    """
    try:
        # Check if essential directories exist
        workspaces_dir = Path("workspaces")
        if not workspaces_dir.exists():
            workspaces_dir.mkdir(exist_ok=True)
        
        # Add any other readiness checks here
        # e.g., database connectivity, external service availability
        
        return {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "workspaces_directory": "ok",
                "file_system": "ok"
            }
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@router.get("/health/liveness")
async def liveness_check():
    """
    Liveness check endpoint for Kubernetes/container orchestration.
    
    Checks if the application is alive and should not be restarted.
    """
    return {
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time() - _startup_time
    }
