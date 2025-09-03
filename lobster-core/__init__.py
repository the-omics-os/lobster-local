"""
Lobster Core - Shared interfaces and utilities for Lobster AI system
"""

__version__ = "0.1.0"

from .interfaces.base_client import BaseLobsterClient, BaseDataManager

__all__ = [
    "BaseLobsterClient", 
    "BaseDataManager"
]
