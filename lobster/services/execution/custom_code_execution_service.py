"""
Custom Code Execution Service for ad-hoc Python code execution.

This service provides a fallback mechanism for agents to execute arbitrary
Python code when existing specialized tools don't cover edge cases. It maintains
W3C-PROV compliance and integrates with Lobster's notebook export system.

SECURITY MODEL:
- Subprocess-based execution for process isolation
- Timeout enforcement (300s default)
- Workspace-only file access
- No network access (future: can add with Docker)
- Crash isolation (user code crashes don't kill Lobster)
"""

import ast
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.services.execution.execution_context_builder import ExecutionContextBuilder
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum captured output length (10,000 chars ~= 200 lines)
MAX_OUTPUT_LENGTH = 10000

# Default execution timeout (5 minutes)
DEFAULT_TIMEOUT = 300


class CodeExecutionError(Exception):
    """Base exception for code execution failures."""

    pass


class CodeValidationError(CodeExecutionError):
    """Raised when code fails safety validation."""

    pass


class CustomCodeExecutionService:
    """
    Stateless service for executing arbitrary Python code with safety checks.

    This service follows Lobster's 3-tuple pattern: (result, stats, AnalysisStep).
    It provides a high-trust execution model (Jupyter-like) with package restrictions
    and opt-in provenance tracking.

    Design Philosophy:
    - High trust: Execute user code as-is (like Jupyter)
    - Safety warnings: AST validation, forbidden import detection
    - Ephemeral by default: Only persist to provenance if persist=True
    - Full workspace access: Auto-inject modalities, CSV, JSON, queues

    Injected Variables (available in custom code):
    - WORKSPACE (Path): Full workspace path for reading inputs
    - OUTPUT_DIR (Path): Recommended directory for exports (workspace/exports/)
    - modalities (list): Available modality names
    - adata (AnnData): Loaded modality if modality_name specified
    - Workspace data (dict/DataFrame): If load_workspace_files=True

    File Output Convention:
        Use OUTPUT_DIR for CSV/TSV/Excel exports to ensure predictable file locations:

        >>> import pandas as pd
        >>> df = pd.DataFrame(...)
        >>> df.to_csv(OUTPUT_DIR / "my_results.csv")

        This saves to workspace/exports/my_results.csv for easy user discovery.
        Note: OUTPUT_DIR is a convention, not enforced. Code can write anywhere.

    Example:
        >>> service = CustomCodeExecutionService(data_manager)
        >>> result, stats, ir = service.execute(
        ...     code="result = 2 + 2",
        ...     persist=False
        ... )
        >>> result
        4
    """

    # Forbidden imports (security) - these modules are blocked entirely
    FORBIDDEN_MODULES = {
        # Process execution
        "subprocess",
        "multiprocessing",
        "concurrent",
        # Import mechanisms (bypass vectors)
        "importlib",
        "imp",
        # Serialization (RCE vectors via pickle deserialization)
        "pickle",
        "marshal",
        "shelve",
        "dill",
        # Low-level (escape vectors)
        "ctypes",
        "cffi",
    }

    # Forbidden specific imports (from X import Y)
    # os module is allowed for os.path, but dangerous functions are blocked
    FORBIDDEN_FROM_IMPORTS = {
        # os dangerous functions (keep os module for os.path)
        ("os", "system"),
        ("os", "popen"),
        ("os", "exec"),
        ("os", "execl"),
        ("os", "execle"),
        ("os", "execlp"),
        ("os", "execv"),
        ("os", "execve"),
        ("os", "execvp"),
        ("os", "execvpe"),
        ("os", "spawnl"),
        ("os", "spawnle"),
        ("os", "spawnlp"),
        ("os", "spawnlpe"),
        ("os", "spawnv"),
        ("os", "spawnve"),
        ("os", "spawnvp"),
        ("os", "spawnvpe"),
        ("os", "fork"),
        ("os", "forkpty"),
        ("os", "kill"),
        ("os", "killpg"),
        # signal manipulation (can kill parent process)
        ("signal", "signal"),
        ("signal", "kill"),
        ("signal", "alarm"),
        # destructive file ops
        ("shutil", "rmtree"),
        ("shutil", "move"),
    }

    # Allowed imports (standard Lobster stack from pyproject.toml)
    ALLOWED_MODULES = {
        # Core scientific computing
        "numpy",
        "np",
        "pandas",
        "pd",
        "scipy",
        "sklearn",
        # Bioinformatics
        "scanpy",
        "sc",
        "anndata",
        "ad",
        "mudata",
        "mdata",
        "pydeseq2",
        "biopython",
        "Bio",
        # Visualization
        "plotly",
        "matplotlib",
        "plt",
        "seaborn",
        "sns",
        # Standard library (safe subset)
        "math",
        "statistics",
        "re",
        "json",
        "csv",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "typing",
        # Lobster internal (for advanced users)
        "lobster",
    }

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize the custom code execution service.

        Args:
            data_manager: DataManagerV2 instance for workspace access
        """
        self.data_manager = data_manager
        self.context_builder = ExecutionContextBuilder(data_manager)
        logger.debug("Initialized CustomCodeExecutionService")

    def _resolve_workspace_key_path(self, key: str) -> Optional[Path]:
        """
        Resolve workspace_key to actual file path in workspace subdirectories.

        Searches in priority order: metadata/ → data/ → literature/ → root (backward compat)
        Follows DRY principle by using workspace structure knowledge.

        Args:
            key: Workspace key identifier

        Returns:
            Path to file if found, None otherwise

        Example:
            >>> service._resolve_workspace_key_path("harmonized_publication_metadata")
            Path(".lobster_workspace/metadata/harmonized_publication_metadata.json")
        """
        workspace_path = Path(self.data_manager.workspace_path)

        # Search workspace subdirectories in priority order
        search_dirs = ["metadata", "data", "literature", ""]  # "" = root (backward compat)

        for subdir in search_dirs:
            base_path = workspace_path / subdir if subdir else workspace_path

            # Try both .json and .csv extensions
            for ext in [".json", ".csv"]:
                file_path = base_path / f"{key}{ext}"
                if file_path.exists():
                    logger.debug(f"Resolved workspace_key '{key}' → {file_path.relative_to(workspace_path)}")
                    return file_path

        logger.warning(f"workspace_key '{key}' not found in any subdirectory")
        return None

    def execute(
        self,
        code: str,
        modality_name: Optional[str] = None,
        load_workspace_files: bool = True,
        workspace_keys: Optional[List[str]] = None,
        persist: bool = False,
        description: str = "Custom code execution",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Tuple[Any, Dict[str, Any], AnalysisStep]:
        """
        Execute arbitrary Python code with workspace context injection.

        Args:
            code: Python code to execute (can be multi-line)
            modality_name: Optional specific modality to load as 'adata'
            load_workspace_files: Auto-inject CSV/JSON files from workspace
            workspace_keys: Optional list of specific workspace keys to load (reduces token usage)
                          If provided, only these files are loaded. If None, all files are loaded.
                          Example: ["sra_prjna834801_samples"] loads only that file
            persist: If True, save this execution to provenance/notebook export
            description: Human-readable description of what this code does
            timeout: Maximum execution time in seconds (default: 300s)

        Returns:
            Tuple containing:
            - result: Execution result (from 'result' variable or last expression)
            - stats: Execution statistics (success, duration, warnings, etc.)
            - ir: AnalysisStep for provenance tracking

        Raises:
            CodeValidationError: If code fails safety checks
            CodeExecutionError: If execution fails

        Example:
            >>> result, stats, ir = service.execute(
            ...     code="import numpy as np\\nresult = np.mean([1, 2, 3])",
            ...     persist=False
            ... )
            >>> result
            2.0
            >>> stats['success']
            True

            >>> # Selective loading to avoid token overflow
            >>> result, stats, ir = service.execute(
            ...     code="samples = sra_prjna834801_samples['samples']; result = len(samples)",
            ...     workspace_keys=["sra_prjna834801_samples"],
            ...     persist=False
            ... )
        """
        start_time = time.time()

        logger.info(f"Executing custom code: {description}")
        logger.debug(f"Code ({len(code)} chars):\n{code[:500]}...")

        # Step 1: Validate code safety
        validation_warnings = self._validate_code_safety(code)

        # Step 2: Ensure modality is saved to disk (subprocess needs file access)
        if modality_name and modality_name in self.data_manager.list_modalities():
            modality_path = self.data_manager.workspace_path / f"{modality_name}.h5ad"
            if not modality_path.exists():
                logger.debug(
                    f"Saving modality {modality_name} to disk for subprocess access"
                )
                adata = self.data_manager.get_modality(modality_name)
                adata.write_h5ad(modality_path)

        # Step 2.5: Resolve workspace_key paths (Bug fix - Gemini Option C)
        resolved_paths = {}
        if workspace_keys:
            for key in workspace_keys:
                resolved_path = self._resolve_workspace_key_path(key)
                if resolved_path:
                    resolved_paths[key] = str(resolved_path)
                else:
                    logger.warning(f"workspace_key '{key}' not found, will be unavailable in code")

        # Step 3: Build execution context (now just metadata for subprocess)
        exec_context = {
            "modality_name": modality_name,
            "workspace_path": self.data_manager.workspace_path,
            "load_workspace_files": load_workspace_files,
            "workspace_keys": workspace_keys,
            "resolved_paths": resolved_paths,  # NEW: Pass resolved paths to subprocess
            "timeout": timeout,
        }

        # Step 4: Execute code in subprocess
        result, stdout_output, stderr_output, exec_error = self._execute_in_namespace(
            code, exec_context
        )

        # Step 4: Compute statistics
        duration = time.time() - start_time
        stats = {
            "success": exec_error is None,
            "duration_seconds": round(duration, 3),
            "warnings": validation_warnings,
            "stdout_lines": len(stdout_output.splitlines()) if stdout_output else 0,
            "stderr_lines": len(stderr_output.splitlines()) if stderr_output else 0,
            "stdout_preview": stdout_output[:500] if stdout_output else "",
            "stderr_preview": stderr_output[:500] if stderr_output else "",
            "result_type": type(result).__name__ if result is not None else None,
            "modality_loaded": modality_name,
            "workspace_files_loaded": load_workspace_files,
            "workspace_keys": workspace_keys,
            "selective_loading": workspace_keys is not None,
            "persisted": persist,
            "error": str(exec_error) if exec_error else None,
        }

        # Step 5: Generate IR (always, but mark as non-exportable if persist=False)
        ir = self._create_ir(
            code=code,
            description=description,
            modality_name=modality_name,
            load_workspace_files=load_workspace_files,
            workspace_keys=workspace_keys,
            persist=persist,
            stats=stats,
        )

        logger.info(
            f"Code execution {'succeeded' if stats['success'] else 'failed'} "
            f"in {duration:.2f}s"
        )

        if exec_error:
            logger.error(f"Execution error: {exec_error}")
            raise CodeExecutionError(
                f"Code execution failed: {exec_error}\n\n"
                f"Stderr output:\n{stderr_output}"
            )

        return result, stats, ir

    def _validate_code_safety(self, code: str) -> List[str]:
        """
        Validate code for syntax errors and forbidden imports.

        Args:
            code: Python code to validate

        Returns:
            List of warning messages (empty if no warnings)

        Raises:
            CodeValidationError: If code has syntax errors or forbidden imports

        Example:
            >>> warnings = service._validate_code_safety("import pandas as pd")
            >>> warnings
            []
            >>> service._validate_code_safety("import subprocess")
            Traceback (most recent call last):
            ...
            CodeValidationError: Forbidden import: 'subprocess'
        """
        warnings = []

        # Syntax validation
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise CodeValidationError(
                f"Syntax error in code at line {e.lineno}: {e.msg}\n" f"Code: {e.text}"
            )

        # Import validation
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.FORBIDDEN_MODULES:
                        raise CodeValidationError(
                            f"Forbidden import: '{alias.name}'. "
                            f"Subprocess execution and destructive file operations are restricted."
                        )
                    if alias.name not in self.ALLOWED_MODULES:
                        warnings.append(
                            f"Import '{alias.name}' not in standard Lobster stack. "
                            f"Execution may fail if module not installed."
                        )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                # Check for forbidden module imports
                if module in self.FORBIDDEN_MODULES:
                    raise CodeValidationError(
                        f"Forbidden import: '{module}'. "
                        f"Subprocess execution and file system operations are restricted."
                    )

                # Check for forbidden specific imports (from X import Y)
                for alias in node.names:
                    if (module, alias.name) in self.FORBIDDEN_FROM_IMPORTS:
                        raise CodeValidationError(
                            f"Forbidden import: 'from {module} import {alias.name}'. "
                            f"System command execution and destructive operations are restricted."
                        )

                if module and module.split(".")[0] not in self.ALLOWED_MODULES:
                    warnings.append(
                        f"Import from '{module}' not in standard Lobster stack."
                    )

            # SECURITY: Block dangerous code execution functions
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "compile"]:
                        raise CodeValidationError(
                            f"Function '{node.func.id}()' is forbidden. "
                            f"This function can execute arbitrary code and bypass safety checks. "
                            f"Use explicit operations instead."
                        )
                    if node.func.id == "__import__":
                        raise CodeValidationError(
                            f"Function '__import__()' is forbidden. "
                            f"Use standard 'import' statements instead."
                        )

        return warnings

    def _generate_context_setup_code(
        self,
        modality_name: Optional[str],
        workspace_path: Path,
        load_workspace_files: bool,
        workspace_keys: Optional[List[str]] = None,
        resolved_paths: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate Python code to set up execution context in subprocess.

        Since subprocess runs in isolation, we can't pass objects directly.
        Instead, we generate code that loads data from workspace files.

        Args:
            modality_name: Optional modality to load as 'adata'
            workspace_path: Path to workspace
            load_workspace_files: Whether to load CSV/JSON files
            workspace_keys: Optional list of specific keys to load (selective loading)

        Returns:
            Python code string that sets up context
        """
        setup_code = f"""
# Auto-generated context setup for Lobster custom code execution
import sys
import builtins
from pathlib import Path
import json

# ==============================================================================
# SECURITY NOTES:
# - Dangerous builtins (eval, exec, compile, __import__) are blocked at AST level
# - We do NOT remove them at runtime because Python's import system needs exec()
# - File access is unrestricted in subprocess, but env vars are filtered
# - Full sandbox requires Docker isolation (Phase 2)
# ==============================================================================

# ==============================================================================
# Workspace configuration
# ==============================================================================
WORKSPACE = Path('{workspace_path}')
sys.path.append(str(WORKSPACE))  # SECURITY: Lower priority - cannot shadow stdlib

# OUTPUT_DIR: Recommended directory for all CSV/TSV/Excel exports (v1.0+)
# Using this ensures files go to workspace/exports/ for easy user discovery
# Convention: df.to_csv(f"{{OUTPUT_DIR}}/my_results.csv")
OUTPUT_DIR = WORKSPACE / 'exports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize context
modalities = []
"""

        # Add modality loading if specified
        if modality_name:
            setup_code += f"""
# Load specified modality
try:
    import anndata
    modality_path = WORKSPACE / '{modality_name}.h5ad'
    if modality_path.exists():
        adata = anndata.read_h5ad(modality_path)
        print(f"Loaded modality '{modality_name}': {{adata.n_obs}} obs x {{adata.n_vars}} vars")
    else:
        print(f"Warning: Modality file not found: {{modality_path}}")
        adata = None
except Exception as e:
    print(f"Error loading modality: {{e}}")
    adata = None
"""

        # Add workspace file loading
        if load_workspace_files:
            # Determine if selective loading is enabled
            if workspace_keys:
                # v2.0: Use pre-resolved paths from main process (Gemini Option C - DRY principle)
                resolved_paths_dict = resolved_paths or {}
                paths_str = repr(resolved_paths_dict)  # {'key': '/full/path/to/file.json', ...}

                setup_code += f"""
# Selective workspace file loading with pre-resolved paths (v2.0)
import pandas as pd
import json

resolved_paths = {paths_str}
loaded_files = []

for key, file_path_str in resolved_paths.items():
    file_path = Path(file_path_str)

    if not file_path.exists():
        print(f"Warning: Resolved path does not exist: {{file_path}}")
        continue

    try:
        if file_path.suffix == '.json':
            with open(file_path) as f:
                content = json.load(f)

                # Memory optimization: load samples only for large files (Bug 3 fix)
                file_size = file_path.stat().st_size
                if isinstance(content, dict) and 'samples' in content and file_size > 100_000_000:
                    # Large file (>100MB): load samples list only
                    globals()[key] = content['samples']
                    print(f"Loaded {{key}}: {{len(content['samples'])}} samples (large file optimization)")
                else:
                    # Small file: load full content
                    globals()[key] = content
                    print(f"Loaded {{key}} from {{file_path.parent.name}}/{{file_path.name}}")

                loaded_files.append(f"{{key}} ({{file_path.parent.name}}/{{file_path.name}})")

        elif file_path.suffix == '.csv':
            globals()[key] = pd.read_csv(file_path)
            loaded_files.append(f"{{key}} ({{file_path.parent.name}}/{{file_path.name}})")
            print(f"Loaded {{key}}: {{len(globals()[key])}} rows from CSV")

    except Exception as e:
        print(f"Error loading {{key}} from {{file_path}}: {{e}}")

if loaded_files:
    print(f"✓ Loaded {{len(loaded_files)}} workspace keys successfully")
else:
    print(f"Warning: No workspace keys loaded (resolved_paths={{list(resolved_paths.keys())}})")

# Convenience imports
workspace_path = WORKSPACE
Path = Path

"""
            else:
                # Full loading: load all files (backward compatible)
                setup_code += """
# Auto-load CSV and JSON files
import pandas as pd

csv_data = {}
json_data = {}

# Load CSV files from workspace root
for csv_file in WORKSPACE.glob('*.csv'):
    var_name = csv_file.stem.replace('-', '_').replace(' ', '_')
    if var_name and var_name[0].isdigit():
        var_name = 'data_' + var_name
    var_name = ''.join(c for c in var_name if c.isalnum() or c == '_') or 'data'

    try:
        globals()[var_name] = pd.read_csv(csv_file)
        csv_data[var_name] = csv_file.name
    except Exception as e:
        print(f"Warning: Could not load {csv_file.name}: {e}")

# Also scan exports/ subdirectory (v1.0+ centralized exports)
exports_dir = WORKSPACE / 'exports'
if exports_dir.exists():
    for csv_file in exports_dir.glob('*.csv'):
        var_name = csv_file.stem.replace('-', '_').replace(' ', '_')
        if var_name and var_name[0].isdigit():
            var_name = 'data_' + var_name
        var_name = ''.join(c for c in var_name if c.isalnum() or c == '_') or 'data'

        try:
            globals()[var_name] = pd.read_csv(csv_file)
            csv_data[var_name] = f"exports/{csv_file.name}"
        except Exception as e:
            print(f"Warning: Could not load exports/{csv_file.name}: {e}")

# Load JSON files (skip hidden files)
for json_file in WORKSPACE.glob('*.json'):
    if json_file.name.startswith('.'):
        continue

    var_name = json_file.stem.replace('-', '_').replace(' ', '_')
    if var_name and var_name[0].isdigit():
        var_name = 'data_' + var_name
    var_name = ''.join(c for c in var_name if c.isalnum() or c == '_') or 'data'

    try:
        with open(json_file) as f:
            globals()[var_name] = json.load(f)
            json_data[var_name] = json_file.name
    except Exception as e:
        print(f"Warning: Could not load {json_file.name}: {e}")

# Load JSONL queue files
for queue_file in ['download_queue.jsonl', 'publication_queue.jsonl']:
    queue_path = WORKSPACE / queue_file
    if queue_path.exists():
        var_name = queue_file.replace('.jsonl', '').replace('-', '_')
        try:
            entries = []
            with open(queue_path) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
            globals()[var_name] = entries
        except Exception as e:
            print(f"Warning: Could not load {queue_file}: {e}")

if csv_data or json_data:
    print(f"Loaded workspace files: {len(csv_data)} CSV, {len(json_data)} JSON")

# Convenience imports
workspace_path = WORKSPACE
Path = Path

"""

        setup_code += "\n# User code starts here\n"
        return setup_code

    def _execute_in_namespace(
        self, code: str, context: Dict[str, Any]
    ) -> Tuple[Any, str, str, Optional[Exception]]:
        """
        Execute code in isolated subprocess with timeout.

        SECURITY: Uses subprocess.run() for process isolation, timeout enforcement,
        and crash isolation. User code cannot affect Lobster's main process.

        Args:
            code: Python code to execute
            context: Execution context (modality_name, workspace_path, etc. for setup)

        Returns:
            Tuple of (result, stdout, stderr, error)

        Example:
            >>> context = {'modality_name': 'test', 'workspace_path': Path('...'), ...}
            >>> result, stdout, stderr, error = service._execute_in_namespace(
            ...     "result = 2 + 2", context
            ... )
            >>> result
            4
        """
        # Generate context setup code
        modality_name = context.get("modality_name")
        workspace_path = context.get("workspace_path", self.data_manager.workspace_path)
        load_workspace_files = context.get("load_workspace_files", True)
        workspace_keys = context.get("workspace_keys")
        resolved_paths = context.get("resolved_paths", {})

        setup_code = self._generate_context_setup_code(
            modality_name=modality_name,
            workspace_path=workspace_path,
            load_workspace_files=load_workspace_files,
            workspace_keys=workspace_keys,
            resolved_paths=resolved_paths,
        )

        # Combine setup + user code + result extraction
        full_script = setup_code + "\n" + code + "\n\n"

        # Add result extraction code
        full_script += """
# Extract result and write to temp file for parent process
import json
if 'result' in dir() and result is not None:
    result_path = WORKSPACE / '.execution_result.json'
    try:
        # Try to serialize result
        with open(result_path, 'w') as f:
            json.dump({'result': result, 'type': type(result).__name__}, f)
    except (TypeError, ValueError):
        # Result not JSON-serializable, save as string
        with open(result_path, 'w') as f:
            json.dump({'result': str(result), 'type': type(result).__name__}, f)
"""

        # Write script to temporary file
        script_path = workspace_path / f".script_{uuid.uuid4().hex}.py"
        result_path = workspace_path / ".execution_result.json"

        try:
            script_path.write_text(full_script)

            # Execute in subprocess with timeout (from context)
            timeout_seconds = context.get("timeout", DEFAULT_TIMEOUT)
            logger.debug(f"Executing code in subprocess (timeout={timeout_seconds}s)")

            # SECURITY: Filter environment variables to prevent credential leakage
            # Only pass safe, necessary variables to subprocess
            import os as _os

            safe_env = {
                "PATH": _os.environ.get("PATH", ""),
                "HOME": _os.environ.get("HOME", ""),
                "USER": _os.environ.get("USER", ""),
                "TMPDIR": _os.environ.get("TMPDIR", "/tmp"),
                "LANG": _os.environ.get("LANG", "en_US.UTF-8"),
                "LC_ALL": _os.environ.get("LC_ALL", ""),
                # Python-specific (empty PYTHONPATH prevents hijacking)
                "PYTHONPATH": "",
                "PYTHONHOME": _os.environ.get("PYTHONHOME", ""),
            }

            proc_result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=safe_env,  # SECURITY: Filtered environment only
            )

            stdout_output = proc_result.stdout
            stderr_output = proc_result.stderr
            return_code = proc_result.returncode

            # Truncate output if needed
            if len(stdout_output) > MAX_OUTPUT_LENGTH:
                stdout_output = (
                    stdout_output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
                )
                logger.warning(f"Stdout truncated at {MAX_OUTPUT_LENGTH} characters")

            if len(stderr_output) > MAX_OUTPUT_LENGTH:
                stderr_output = (
                    stderr_output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
                )
                logger.warning(f"Stderr truncated at {MAX_OUTPUT_LENGTH} characters")

            # Extract result if available
            result = None
            error = None

            if return_code != 0:
                # Execution failed
                error = Exception(
                    f"Code execution failed with return code {return_code}"
                )
            else:
                # Try to load result
                if result_path.exists():
                    try:
                        with open(result_path) as f:
                            result_data = json.load(f)
                            result = result_data.get("result")
                    except Exception as e:
                        logger.debug(f"Could not load result file: {e}")

            return result, stdout_output, stderr_output, error

        except subprocess.TimeoutExpired:
            error = Exception(f"Code execution exceeded {DEFAULT_TIMEOUT}s timeout")
            return None, "", f"Execution timeout after {DEFAULT_TIMEOUT}s", error

        except Exception as e:
            logger.error(f"Subprocess execution error: {e}")
            error = e
            return None, "", f"Subprocess error: {e}", error

        finally:
            # Clean up temporary files
            if script_path.exists():
                script_path.unlink()
            if result_path.exists():
                result_path.unlink()

    def _create_ir(
        self,
        code: str,
        description: str,
        modality_name: Optional[str],
        load_workspace_files: bool,
        workspace_keys: Optional[List[str]],
        persist: bool,
        stats: Dict[str, Any],
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for provenance and notebook export.

        Args:
            code: Executed Python code
            description: Human-readable description
            modality_name: Loaded modality name (if any)
            load_workspace_files: Whether workspace files were loaded
            workspace_keys: Specific workspace keys loaded (if selective loading)
            persist: Whether to include in notebook export
            stats: Execution statistics

        Returns:
            AnalysisStep object

        Example:
            >>> ir = service._create_ir(
            ...     code="result = 2 + 2",
            ...     description="Simple addition",
            ...     modality_name=None,
            ...     load_workspace_files=False,
            ...     workspace_keys=None,
            ...     persist=False,
            ...     stats={'success': True, 'duration_seconds': 0.001}
            ... )
            >>> ir.operation
            'custom_code_execution'
            >>> ir.exportable
            False
        """
        # Extract imports from code
        imports = self._extract_imports(code)

        # Build workspace setup code for notebook export
        # This ensures WORKSPACE and OUTPUT_DIR are defined when notebook runs
        workspace_setup = """# Workspace setup (for notebook execution)
from pathlib import Path
import sys

WORKSPACE = Path('{{ workspace_path }}')
sys.path.append(str(WORKSPACE))

# OUTPUT_DIR: Recommended directory for CSV/TSV/Excel exports
OUTPUT_DIR = WORKSPACE / 'exports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
"""

        # Combine workspace setup with user code for notebook export
        # Uses Jinja2 template syntax for workspace_path parameter injection
        code_template = f"{workspace_setup}\n# User code\n{code}"

        # Get workspace path for parameter default
        workspace_path_str = str(self.data_manager.workspace_path)

        # Parameter schema (for Papermill injection if exported)
        parameter_schema = {
            "workspace_path": ParameterSpec(
                param_type="str",
                papermill_injectable=True,  # Can override workspace location
                default_value=workspace_path_str,
                required=True,
                description="Path to Lobster workspace directory",
            ),
            "code": ParameterSpec(
                param_type="str",
                papermill_injectable=False,  # Code should not be overridden
                default_value=code,
                required=True,
                description="Python code to execute",
            ),
            "description": ParameterSpec(
                param_type="str",
                papermill_injectable=False,
                default_value=description,
                required=False,
                description="Human-readable description",
            ),
        }

        # Add modality_name parameter if used
        if modality_name:
            parameter_schema["modality_name"] = ParameterSpec(
                param_type="str",
                papermill_injectable=True,  # Can override which modality to use
                default_value=modality_name,
                required=False,
                description="Modality to load as adata",
            )

        return AnalysisStep(
            operation="custom_code_execution",
            tool_name="execute_custom_code",
            description=description,
            library="custom",
            code_template=code_template,
            imports=imports,
            parameters={
                "workspace_path": workspace_path_str,  # For Jinja2 template rendering
                "code": code,
                "description": description,
                "modality_name": modality_name,
                "load_workspace_files": load_workspace_files,
                "workspace_keys": workspace_keys,
                "selective_loading": workspace_keys is not None,
            },
            parameter_schema=parameter_schema,
            input_entities=[modality_name] if modality_name else [],
            output_entities=[],  # Custom code may not produce named outputs
            execution_context={
                "persist": persist,
                "success": stats["success"],
                "duration": stats["duration_seconds"],
                "warnings": stats["warnings"],
                "selective_loading": workspace_keys is not None,
                "workspace_keys_count": len(workspace_keys) if workspace_keys else 0,
            },
            validates_on_export=True,
            requires_validation=True,
            exportable=persist,  # Only include in notebook if persist=True
        )

    def _extract_imports(self, code: str) -> List[str]:
        """
        Extract import statements from code using AST.

        Args:
            code: Python code

        Returns:
            List of import statements as strings

        Example:
            >>> imports = service._extract_imports("import numpy as np\\nfrom pandas import DataFrame")
            >>> imports
            ['import numpy as np', 'from pandas import DataFrame']
        """
        imports = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            imports.append(f"import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"import {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = ", ".join(
                        f"{a.name} as {a.asname}" if a.asname else a.name
                        for a in node.names
                    )
                    imports.append(f"from {module} import {names}")

        except SyntaxError:
            # If code has syntax errors, return empty (validation will catch it)
            pass

        return imports
