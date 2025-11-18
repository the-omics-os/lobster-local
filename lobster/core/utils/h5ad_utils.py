"""
Utilities for H5AD serialization and sanitization.

This module provides functions to ensure data can be safely saved to H5AD format
by converting non-serializable types (Path objects, tuples, numpy types, etc.)
to H5AD-compatible types.
"""

import collections
import datetime
import json
import logging
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Optional imports for comprehensive type support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

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
    - tuple → numpy string array
    - numpy scalars → Python scalars → string (for nested dicts)
    - pandas types → appropriate string representation
    - datetime types → ISO format strings
    - set/frozenset → numpy string array (via list)
    - bool → str (HDF5 requirement)
    - int/float in nested dicts → str (HDF5 requirement for nested structures)
    - None → "" (empty string)
    - Path → str
    - list → recursively sanitized (lists in nested dicts must have string values)
    - Non-serializable objects → str (as last resort)

    CRITICAL FIX: HDF5 cannot handle scalar int/float values OR Python lists
    in nested dictionary structures. All such values must be converted to strings
    or numpy arrays.

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
        {'a': array(['1', '2', '3'], dtype='<U...')}

        >>> sanitize_value(np.int64(42))
        '42'

        >>> sanitize_value({"nested": {"int_val": 42}})
        {'nested': {'int_val': '42'}}
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

    # tuple → numpy string array (with recursive sanitization)
    # HDF5 cannot handle Python tuples in nested structures
    if isinstance(obj, tuple):
        sanitized_items = [sanitize_value(v, slash_replacement, strict) for v in obj]
        # Convert to numpy string array for H5AD compatibility
        return np.array([str(item) for item in sanitized_items], dtype=str)

    # numpy scalars → Python scalars (DON'T RETURN - let fall through!)
    # CRITICAL BUG FIX: Previously returned immediately, bypassing stringification
    if isinstance(obj, (np.generic,)):
        obj = obj.item()  # Convert to Python primitive but continue processing
        # Fall through to subsequent type checks

    # Pandas types → string representation
    # Handle pandas-specific types if pandas is available
    if HAS_PANDAS:
        # pd.NA/NaT → empty string
        # Skip numpy arrays and collections to avoid ambiguous truth value warnings
        if not isinstance(obj, (np.ndarray, list, dict, tuple)):
            try:
                if pd.isna(obj):
                    return ""
            except (TypeError, ValueError):
                # pd.isna() can raise exceptions for some types
                pass
        # pd.Timestamp → ISO format string
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        # pd.Timedelta → string representation
        if isinstance(obj, pd.Timedelta):
            return str(obj)
        # pd.Period → string representation
        if hasattr(pd, 'Period') and isinstance(obj, pd.Period):
            return str(obj)

    # Datetime types → ISO format strings
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)

    # Decimal/Fraction → string representation
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, Fraction):
        return str(obj)

    # set/frozenset → numpy string array (via list conversion)
    # HDF5 cannot handle Python sets
    if isinstance(obj, (set, frozenset)):
        sanitized_items = [sanitize_value(v, slash_replacement, strict) for v in sorted(obj, key=str)]
        return np.array([str(item) for item in sanitized_items], dtype=str)

    # bool → string (HDF5 requirement - bool handling can be tricky)
    # MUST come after pandas/numpy checks as they may have bool subclasses
    if isinstance(obj, bool):
        return str(obj)

    # int/float → string (CRITICAL FIX for nested dicts)
    # HDF5 cannot handle scalar int/float in nested dictionaries
    # This MUST come before dict/list handling to ensure values are converted
    if isinstance(obj, (int, float)):
        return str(obj)

    # dict → dict (with recursive sanitization)
    if isinstance(obj, dict):
        return {
            sanitize_key(k, slash_replacement): sanitize_value(v, slash_replacement, strict)
            for k, v in obj.items()
        }

    # list → recursively sanitized list
    # CRITICAL FIX: Lists in nested dicts must contain only strings
    if isinstance(obj, list):
        # Empty lists are OK
        if not obj:
            return []

        # Check if list contains dicts - if so, stringify the entire dict
        # HDF5 cannot handle lists of dicts (anndata converts lists to arrays)
        contains_dict = any(isinstance(item, dict) for item in obj)

        if contains_dict:
            # Stringify dict elements completely
            sanitized_items = []
            for item in obj:
                if isinstance(item, dict):
                    # Recursively sanitize the dict first, then stringify
                    sanitized_dict = sanitize_value(item, slash_replacement, strict)
                    sanitized_items.append(str(sanitized_dict))
                else:
                    sanitized_items.append(str(sanitize_value(item, slash_replacement, strict)))
        else:
            # No dicts - recursively sanitize items
            sanitized_items = [sanitize_value(v, slash_replacement, strict) for v in obj]

        # Convert to numpy string array for H5AD compatibility
        return np.array([str(item) for item in sanitized_items], dtype=str)

    # numpy arrays
    if isinstance(obj, np.ndarray):
        try:
            # Numeric arrays can be preserved
            if np.issubdtype(obj.dtype, np.number):
                return obj
            # Non-numeric or object arrays → convert to string array
            return np.array([str(x) if x is not None else "" for x in obj], dtype=str)
        except (ValueError, TypeError):
            return np.array([str(x) if x is not None else "" for x in obj], dtype=str)

    # Path objects → string (CRITICAL for GEO metadata)
    if isinstance(obj, Path):
        return str(obj)

    # str → pass through (already safe)
    if isinstance(obj, str):
        return obj

    # Catch-all: Stringify any remaining types
    # REMOVED JSON serializability test - it was allowing int/float to pass through
    # which causes "Can't implicitly convert non-string objects to strings" error
    if strict:
        raise ValueError(
            f"Cannot sanitize non-serializable object of type {type(obj).__name__}. "
            f"Value: {repr(obj)[:100]}"
        )
    else:
        # Last resort: stringify
        logger.debug(
            f"Converting object of type {type(obj).__name__} "
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

    # Check tuples (should be converted to numpy arrays or lists)
    if isinstance(obj, tuple):
        issues.append(f"Tuple at {path} (should be numpy array or list): {obj[:3]}...")
        for i, item in enumerate(obj):
            issues.extend(validate_for_h5ad(item, f"{path}[{i}]"))
        return issues

    # numpy arrays of strings are OK (used for sanitized lists)
    if isinstance(obj, np.ndarray):
        # String arrays are valid for H5AD
        if obj.dtype.kind in ('U', 'S', 'O'):  # Unicode, bytes, or object string arrays
            return issues
        # Numeric arrays are valid
        if np.issubdtype(obj.dtype, np.number):
            return issues
        # Other numpy types might be problematic
        issues.append(
            f"Potentially problematic ndarray at {path}: dtype={obj.dtype}, shape={obj.shape}"
        )
        return issues

    # Recursively check dicts
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Check for slash in keys
            if isinstance(key, str) and "/" in key:
                issues.append(f"Key with '/' at {path}: {key}")
            issues.extend(validate_for_h5ad(value, f"{path}.{key}"))
        return issues

    # Recursively check lists (should be converted to numpy arrays)
    if isinstance(obj, list):
        # Empty lists are OK
        if not obj:
            return issues
        # Lists should ideally be numpy arrays for H5AD, but we'll check contents
        for i, item in enumerate(obj):
            issues.extend(validate_for_h5ad(item, f"{path}[{i}]"))
        return issues

    # Integers and floats in nested structures should be strings
    if isinstance(obj, (int, float)):
        # Note: This is context-dependent - top-level ints/floats are OK,
        # but nested ones in dicts should be strings
        # We'll report this as a warning
        issues.append(
            f"Numeric value at {path}: {obj} (should be string in nested dicts)"
        )
        return issues

    # Strings are always OK
    if isinstance(obj, str):
        return issues

    # Test JSON serializability for other types
    try:
        json.dumps(obj)
    except (TypeError, ValueError):
        issues.append(
            f"Non-serializable {type(obj).__name__} at {path}: {repr(obj)[:50]}"
        )

    return issues
