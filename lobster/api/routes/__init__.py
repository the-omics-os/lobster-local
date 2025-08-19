"""
Lobster AI - API Routes
FastAPI route modules for different API endpoints.
"""

# Route modules will be imported here for easy access
from . import health, sessions, chat, files, data, plots, exports

__all__ = ['health', 'sessions', 'chat', 'files', 'data', 'plots', 'exports']
