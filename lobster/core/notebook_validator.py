"""
Jupyter Notebook Validator for Syntax and Import Validation.

This module provides comprehensive validation of Jupyter notebooks before execution,
including Python syntax checking, import resolution, and code structure analysis.
"""

import ast
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nbformat

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """
    Represents a validation issue found in a notebook.

    Attributes:
        severity: 'error' (blocking) or 'warning' (non-blocking)
        cell_index: Index of the cell with the issue (0-based)
        line_number: Line number within the cell (if applicable)
        message: Description of the issue
        code_snippet: Relevant code causing the issue
    """

    severity: str  # 'error' or 'warning'
    cell_index: int
    line_number: Optional[int] = None
    message: str = ""
    code_snippet: Optional[str] = None

    def __str__(self) -> str:
        """Format issue for display."""
        location = f"Cell {self.cell_index}"
        if self.line_number is not None:
            location += f", Line {self.line_number}"

        result = f"[{self.severity.upper()}] {location}: {self.message}"
        if self.code_snippet:
            result += f"\n  Code: {self.code_snippet}"

        return result


@dataclass
class NotebookValidationResult:
    """
    Result of comprehensive notebook validation.

    Attributes:
        is_valid: True if notebook passes all checks (no errors)
        issues: List of validation issues found
        imports_found: Set of import statements discovered
        missing_imports: Set of imports that cannot be resolved
        syntax_errors: List of syntax error details
    """

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    imports_found: Set[str] = field(default_factory=set)
    missing_imports: Set[str] = field(default_factory=set)
    syntax_errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if validation has blocking errors."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def error_count(self) -> int:
        """Count of blocking errors."""
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of non-blocking warnings."""
        return sum(1 for issue in self.issues if issue.severity == "warning")

    def add_error(
        self,
        message: str,
        cell_index: int,
        line_number: Optional[int] = None,
        code_snippet: Optional[str] = None,
    ) -> None:
        """Add an error issue."""
        self.is_valid = False
        self.issues.append(
            ValidationIssue(
                severity="error",
                cell_index=cell_index,
                line_number=line_number,
                message=message,
                code_snippet=code_snippet,
            )
        )

    def add_warning(
        self,
        message: str,
        cell_index: int,
        line_number: Optional[int] = None,
        code_snippet: Optional[str] = None,
    ) -> None:
        """Add a warning issue."""
        self.issues.append(
            ValidationIssue(
                severity="warning",
                cell_index=cell_index,
                line_number=line_number,
                message=message,
                code_snippet=code_snippet,
            )
        )

    def __str__(self) -> str:
        """String representation of validation result."""
        if self.is_valid and not self.has_warnings:
            return "✓ Notebook validation passed with no issues"

        parts = []
        if self.error_count > 0:
            parts.append(f"✗ {self.error_count} error(s)")
        if self.warning_count > 0:
            parts.append(f"⚠ {self.warning_count} warning(s)")

        summary = ", ".join(parts)

        # Add detailed issues
        if self.issues:
            issue_details = "\n\n".join(str(issue) for issue in self.issues[:5])
            if len(self.issues) > 5:
                issue_details += f"\n\n... and {len(self.issues) - 5} more issues"
            summary += f"\n\n{issue_details}"

        return summary


class NotebookValidator:
    """
    Comprehensive validator for Jupyter notebooks.

    Validates notebooks for:
    - Python syntax correctness
    - Import resolution
    - Code structure issues
    - Common anti-patterns
    """

    def __init__(self, strict_imports: bool = True) -> None:
        """
        Initialize notebook validator.

        Args:
            strict_imports: If True, treat unresolved imports as errors.
                          If False, treat them as warnings.
        """
        self.strict_imports = strict_imports
        logger.debug(f"Initialized NotebookValidator (strict_imports={strict_imports})")

    def validate(self, notebook_path: Path) -> NotebookValidationResult:
        """
        Perform comprehensive validation of a notebook.

        Args:
            notebook_path: Path to .ipynb file

        Returns:
            NotebookValidationResult with detailed validation results
        """
        result = NotebookValidationResult()

        # Load notebook
        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            result.add_error(message=f"Cannot read notebook file: {e}", cell_index=-1)
            return result

        # Validate each code cell
        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue

            # Skip empty cells
            if not cell.source.strip():
                continue

            # Validate syntax
            syntax_issues = self._validate_syntax(cell.source, cell_idx)
            result.issues.extend(syntax_issues)

            # Extract imports
            imports = self._extract_imports(cell.source, cell_idx)
            result.imports_found.update(imports)

            # Check for common issues
            common_issues = self._check_common_issues(cell.source, cell_idx)
            result.issues.extend(common_issues)

        # Validate imports can be resolved
        import_issues = self._validate_imports(result.imports_found)
        result.missing_imports = {issue[0] for issue in import_issues}

        for module_name, error_msg in import_issues:
            if self.strict_imports:
                result.add_error(
                    message=f"Cannot import '{module_name}': {error_msg}",
                    cell_index=-1,  # Import check is global
                )
            else:
                result.add_warning(
                    message=f"Cannot import '{module_name}': {error_msg}", cell_index=-1
                )

        # Update is_valid based on errors
        result.is_valid = not result.has_errors

        logger.info(
            f"Validation complete: {result.error_count} errors, {result.warning_count} warnings"
        )

        return result

    def _validate_syntax(self, source: str, cell_idx: int) -> List[ValidationIssue]:
        """
        Validate Python syntax in cell source.

        Args:
            source: Cell source code
            cell_idx: Cell index for error reporting

        Returns:
            List of syntax validation issues
        """
        issues = []

        try:
            ast.parse(source)
        except SyntaxError as e:
            # Extract relevant code snippet
            lines = source.split("\n")
            if e.lineno and 0 < e.lineno <= len(lines):
                snippet = lines[e.lineno - 1].strip()
            else:
                snippet = None

            issues.append(
                ValidationIssue(
                    severity="error",
                    cell_index=cell_idx,
                    line_number=e.lineno,
                    message=f"Syntax error: {e.msg}",
                    code_snippet=snippet,
                )
            )
        except Exception as e:
            issues.append(
                ValidationIssue(
                    severity="error",
                    cell_index=cell_idx,
                    message=f"Failed to parse cell: {e}",
                    code_snippet=None,
                )
            )

        return issues

    def _extract_imports(self, source: str, cell_idx: int) -> Set[str]:
        """
        Extract import statements from cell source.

        Args:
            source: Cell source code
            cell_idx: Cell index (for logging)

        Returns:
            Set of module names imported
        """
        imports = set()

        try:
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get base module name (e.g., 'numpy' from 'numpy.linalg')
                        base_module = alias.name.split(".")[0]
                        imports.add(base_module)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get base module name
                        base_module = node.module.split(".")[0]
                        imports.add(base_module)

        except Exception as e:
            logger.debug(f"Failed to extract imports from cell {cell_idx}: {e}")

        return imports

    def _validate_imports(self, imports: Set[str]) -> List[Tuple[str, str]]:
        """
        Validate that imports can be resolved.

        Args:
            imports: Set of module names to validate

        Returns:
            List of (module_name, error_message) for unresolved imports
        """
        missing = []

        for module_name in imports:
            try:
                # Try to find the module spec
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    missing.append((module_name, "Module not found"))
            except ModuleNotFoundError:
                missing.append((module_name, "Module not installed"))
            except Exception as e:
                missing.append((module_name, str(e)))

        return missing

    def _check_common_issues(self, source: str, cell_idx: int) -> List[ValidationIssue]:
        """
        Check for common code issues and anti-patterns.

        Args:
            source: Cell source code
            cell_idx: Cell index for error reporting

        Returns:
            List of issues found
        """
        issues = []

        # Check for undefined variables (basic check)
        # This is a simplified check - full analysis would require execution context

        # Check for potentially problematic patterns
        if "exec(" in source or "eval(" in source:  # noqa: S102
            issues.append(
                ValidationIssue(
                    severity="warning",
                    cell_index=cell_idx,
                    message="Use of exec() or eval() detected - potential security risk",
                    code_snippet=None,
                )
            )

        # Check for bare except clauses
        if "except:" in source and "except Exception" not in source:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    cell_index=cell_idx,
                    message="Bare except clause detected - consider catching specific exceptions",
                    code_snippet=None,
                )
            )

        # Check for print statements in production code (optional warning)
        if source.count("print(") > 5:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    cell_index=cell_idx,
                    message=f"Many print statements ({source.count('print(')}) - consider using logging",
                    code_snippet=None,
                )
            )

        return issues

    def validate_quick(self, notebook_path: Path) -> bool:
        """
        Quick validation check (syntax only, no import resolution).

        Args:
            notebook_path: Path to .ipynb file

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            for cell in nb.cells:
                if cell.cell_type == "code" and cell.source.strip():
                    try:
                        ast.parse(cell.source)
                    except SyntaxError:
                        return False

            return True

        except Exception:
            return False
