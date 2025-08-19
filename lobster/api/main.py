"""
Lobster AI - FastAPI Main Application
Multi-agent bioinformatics system web service.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from lobster.api.middleware import setup_middleware
from lobster.api.session_manager import SessionManager
from lobster.api.routes import health, sessions, chat, files, data, plots, exports, payments
from lobster.api import websocket
from lobster.config.settings import get_settings
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Global session manager instance
session_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global session_manager
    
    # Startup
    logger.info("Starting Lobster AI FastAPI service...")
    
    # Initialize session manager
    session_manager = SessionManager()
    app.state.session_manager = session_manager
    
    # Create necessary directories
    base_dir = Path("workspaces")
    base_dir.mkdir(exist_ok=True)
    logger.info(f"Created workspaces directory: {base_dir}")
    
    logger.info("Lobster AI FastAPI service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Lobster AI FastAPI service...")
    
    # Clean up sessions
    if session_manager:
        await session_manager.cleanup_all_sessions()
    
    logger.info("Lobster AI FastAPI service shut down complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    settings = get_settings()
    
    app = FastAPI(
        title="Lobster AI API",
        description="Multi-agent bioinformatics analysis system with real-time capabilities",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app, settings)
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(sessions.router, prefix="/api/v1", tags=["sessions"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(files.router, prefix="/api/v1", tags=["files"])
    app.include_router(data.router, prefix="/api/v1", tags=["data"])
    app.include_router(plots.router, prefix="/api/v1", tags=["plots"])
    app.include_router(exports.router, prefix="/api/v1", tags=["exports"])
    app.include_router(payments.router, prefix="/api/v1", tags=["payments"])
    # WebSocket router without prefix to match frontend expectation
    app.include_router(websocket.router, tags=["websocket"])
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "lobster.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our custom logging
    )
