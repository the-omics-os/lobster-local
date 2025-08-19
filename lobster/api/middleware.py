"""
Lobster AI - API Middleware
FastAPI middleware for CORS, logging, error handling, and security.
"""

import time
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from lobster.utils.logger import get_logger
from lobster.config.settings import get_settings

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging API requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable):
        # Generate request ID for tracing
        request_id = str(uuid4())[:8]
        
        # Log incoming request
        start_time = time.time()
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"[{request_id}] {response.status_code} - "
                f"Processed in {process_time:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as exc:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] Request failed after {process_time:.3f}s: {exc}",
                exc_info=True
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "request_id": request_id
                }
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add CSP for API responses (relaxed for development)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' ws: wss:;"
        )
        
        return response


def setup_middleware(app: FastAPI, settings) -> None:
    """Configure all middleware for the FastAPI application."""
    
    # Trusted hosts middleware (must be first)
    if hasattr(settings, 'ALLOWED_HOSTS') and settings.ALLOWED_HOSTS:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS
        )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=getattr(settings, 'ALLOWED_ORIGINS', ["*"]),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-Request-ID",
        ],
        expose_headers=["X-Request-ID"],
    )
    
    # Security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("FastAPI middleware setup complete")


# Error handlers that can be imported and used
async def validation_exception_handler(request: Request, exc):
    """Handle validation errors with detailed messages."""
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": exc.errors() if hasattr(exc, 'errors') else str(exc)
        }
    )


async def http_exception_handler(request: Request, exc):
    """Handle HTTP exceptions with consistent format."""
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"Unhandled exception on {request.url.path}: {exc}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )
