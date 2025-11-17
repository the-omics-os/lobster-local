"""
Utilities for H5AD serialization and sanitization.

This module provides functions to ensure data can be safely saved to H5AD format
by converting non-serializable types (Path objects, tuples, numpy types, etc.)
to H5AD-compatible types.
"""

import collections
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def sanitize_key(key: Any, slash_replacement: str = "__") -> str:
    """
    Sanitize dictionary keys for HDF5 compatibility.

    HDF5 uses '/' as a path separator, so we replace it with '__'.

    Args:
        key: Dictionary key to sanitize
        slash_replacement: String to replace '/' with (default: '__')

    Returns:
        Sanitized key as string
    """
    if isinstance(key, str) and "/" in key:
        return key.replace("/", slash_replacement)
    return str(key) if not isinstance(key, str) else key


def sanitize_value(
    obj: Any,
    slash_replacement: str = "__",
    strict: bool = False
) -> Any:
    """
    Sanitize a value for H5AD/HDF5 serialization.

    Converts problematic types to H5AD-compatible equivalents:
    - OrderedDict → dict
    - tuple → list
    - numpy scalars → Python scalars
    - bool → str (HDF5 requirement)
    - None → "" (empty string)
    - Path → str
    - Non-serializable objects → str (as last resort)

    Args:
        obj: Object to sanitize
        slash_replacement: String to replace '/' in dict keys (default: '__')
        strict: If True, raise ValueError for non-serializable types instead
                of converting to string (default: False)

    Returns:
        Sanitized object safe for H5AD serialization

    Raises:
        ValueError: If strict=True and obj cannot be serialized

    Examples:
        >>> sanitize_value(Path("/tmp/data.csv"))
        '/tmp/data.csv'

        >>> sanitize_value({"a": (1, 2, 3)})
        {'a': [1, 2, 3]}

        >>> sanitize_value(np.int64(42))
        42
    """
    # Handle None early
    if obj is None:
        return ""

    # OrderedDict → dict (with recursive sanitization)
    if isinstance(obj, collections.OrderedDict):
        return {
            sanitize_key(k, slash_replacement): sanitize_value(v, slash_replacement, strict)
            for k, v in obj.items()
        }

    # tuple → list (with recursive sanitization)
    if isinstance(obj, tuple):
        return [sanitize_value(v, slash_replacement, strict) for v in obj]

    # numpy scalars → Python scalars
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # bool → string (HDF5 requirement - bool handling can be tricky)
    if isinstance(obj, bool):
        return str(obj)

    # dict → dict (with recursive sanitization)
    if isinstance(obj, dict):
        return {
            sanitize_key(k, slash_replacement): sanitize_value(v, slash_replacement, strict)
            for k, v in obj.items()
        }

    # list → list (with recursive sanitization)
    if isinstance(obj, list):
        return [sanitize_value(v, slash_replacement, strict) for v in obj]

    # numpy arrays
    if isinstance(obj, np.ndarray):
        try:
            # Numeric arrays can be preserved
            if np.issubdtype(obj.dtype, np.number):
                return obj
            # Non-numeric or object arrays → convert to string array
            return np.array([str(x) if x is not None else "" for x in obj])
        except (ValueError, TypeError):
            return np.array([str(x) if x is not None else "" for x in obj])

    # Path objects → string (CRITICAL for GEO metadata)
    if isinstance(obj, Path):
        return str(obj)

    # Catch-all: Test JSON serializability
    # If object is JSON-serializable, it's likely H5AD-safe
    try:
        json.dumps(obj)
        return obj  # Safe to pass through
    except (TypeError, ValueError):
        if strict:
            raise ValueError(
                f"Cannot sanitize non-serializable object of type {type(obj).__name__}. "
                f"Value: {repr(obj)[:100]}"
            )
        else:
            # Last resort: stringify
            logger.debug(
                f"Converting non-serializable object of type {type(obj).__name__} "
                f"to string for H5AD compatibility"
            )
            return str(obj)


def sanitize_dict(
    data: Dict[str, Any],
    slash_replacement: str = "__",
    strict: bool = False
) -> Dict[str, Any]:
    """
    Sanitize all keys and values in a dictionary for H5AD serialization.

    Convenience function that applies sanitization to both keys and values
    recursively throughout the dictionary structure.

    Args:
        data: Dictionary to sanitize
        slash_replacement: String to replace '/' in keys (default: '__')
        strict: If True, raise errors for non-serializable types

    Returns:
        Sanitized dictionary

    Examples:
        >>> sanitize_dict({"file/path": Path("/tmp/data.csv"), "count": (1, 2, 3)})
        {'file__path': '/tmp/data.csv', 'count': [1, 2, 3]}
    """
    return {
        sanitize_key(k, slash_replacement): sanitize_value(v, slash_replacement, strict)
        for k, v in data.items()
    }


def validate_for_h5ad(obj: Any, path: str = "root") -> List[str]:
    """
    Validate an object for H5AD serialization and return list of issues.

    This function checks for known problematic types without modifying the object.
    Useful for pre-save validation and debugging.

    Args:
        obj: Object to validate
        path: Current path in object tree (for error messages)

    Returns:
        List of validation warnings (empty if all OK)

    Examples:
        >>> issues = validate_for_h5ad({"data": Path("/tmp/file.csv")})
        >>> print(issues)
        ['Path object at root.data: /tmp/file.csv']
    """
    issues = []

    # Check for Path objects
    if isinstance(obj, Path):
        issues.append(f"Path object at {path}: {obj}")
        return issues

    # Check tuples (should be lists)
    if isinstance(obj, tuple):
        issues.append(f"Tuple at {path} (should be list): {obj[:3]}...")
        for i, item in enumerate(obj):
            issues.extend(validate_for_h5ad(item, f"{path}[{i}]"))
        return issues

    # Recursively check dicts
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Check for slash in keys
            if isinstance(key, str) and "/" in key:
                issues.append(f"Key with '/' at {path}: {key}")
            issues.extend(validate_for_h5ad(value, f"{path}.{key}"))
        return issues

    # Recursively check lists
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            issues.extend(validate_for_h5ad(item, f"{path}[{i}]"))
        return issues

    # Test JSON serializability
    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        issues.append(
            f"Non-serializable {type(obj).__name__} at {path}: {repr(obj)[:50]}"
        )

    return issues
