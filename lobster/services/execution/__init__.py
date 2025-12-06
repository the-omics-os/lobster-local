"""
Custom code execution services for Lobster AI agents.

This module provides services for executing arbitrary Python code with
workspace context injection, safety validation, and W3C-PROV compliance.
It also provides SDK delegation for complex reasoning tasks.
"""

from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
)
from lobster.services.execution.execution_context_builder import (
    ExecutionContextBuilder,
)

# SDK Delegation Service (premium feature - graceful fallback if unavailable)
try:
    from lobster.services.execution.sdk_delegation_service import (
        SDKDelegationError,
        SDKDelegationService,
    )
    HAS_SDK_DELEGATION = True
except ImportError:
    # Premium feature not available in open-core distribution
    SDKDelegationService = None
    SDKDelegationError = Exception  # Fallback base class
    HAS_SDK_DELEGATION = False

__all__ = [
    "CustomCodeExecutionService",
    "CodeExecutionError",
    "CodeValidationError",
    "ExecutionContextBuilder",
    "SDKDelegationService",
    "SDKDelegationError",
    "HAS_SDK_DELEGATION",
]
