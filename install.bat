@echo off
REM Lobster AI - Windows Installation Script (Batch)
REM This is a fallback for systems where PowerShell execution is restricted

echo.
echo ========================================
echo    Lobster AI - Windows Installation
echo ========================================
echo.

REM Try to run PowerShell script first
echo Attempting to run PowerShell installer...
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1"

if %ERRORLEVEL% EQU 0 (
    goto :EOF
)

echo.
echo PowerShell script failed or blocked. Using fallback installation...
echo.

REM Find Python
set PYTHON_CMD=
for %%P in (python python3 python3.11 python3.12 python3.13 py) do (
    %%P --version >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=%%P
        goto :found_python
    )
)

:found_python
if "%PYTHON_CMD%"=="" (
    echo ERROR: Python 3.11+ not found!
    echo.
    echo Please install Python 3.11 or higher from:
    echo   https://www.python.org/downloads/
    echo.
    echo Make sure to check 'Add Python to PATH' during installation!
    pause
    exit /b 1
)

echo Found Python: %PYTHON_CMD%
echo.

REM Check Python version
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python 3.11 or higher is required!
    echo Your version:
    %PYTHON_CMD% --version
    echo.
    echo Please upgrade Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Create virtual environment
if exist .venv (
    echo Virtual environment already exists.
    set /p recreate="Recreate it? (y/N): "
    if /i "%recreate%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q .venv
    )
)

if not exist .venv (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

echo.
echo Installing Lobster AI and dependencies...
echo (This may take 3-5 minutes on first install)
echo.

REM Install
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -e .

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Common issues:
    echo   - If you see compiler errors, you may need Visual Studio Build Tools
    echo   - Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
    echo   - Or use Docker instead: docker-compose run --rm lobster-cli
    echo.
    pause
    exit /b 1
)

echo.
echo Installation complete!
echo.

REM Create .env file
if not exist .env (
    echo Creating .env configuration file...
    if exist .env.example (
        copy .env.example .env >nul
    ) else (
        (
            echo # Lobster AI Configuration
            echo # Add your API key below (required^)
            echo.
            echo # Option 1: Claude API (recommended for quick start^)
            echo ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
            echo.
            echo # Option 2: AWS Bedrock (recommended for production^)
            echo # AWS_BEDROCK_ACCESS_KEY=your-key
            echo # AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret
            echo.
            echo # Optional: NCBI API key for enhanced literature search
            echo # NCBI_API_KEY=your-ncbi-key
        ) > .env
    )
    echo .env file created.
    echo.
    echo IMPORTANT: Edit .env and add your API key!
    echo   Run: notepad .env
)

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Configure your API key:
echo    notepad .env
echo.
echo 2. Activate the virtual environment:
echo    .venv\Scripts\activate.bat
echo.
echo 3. Start using Lobster AI:
echo    lobster chat
echo.
echo For help: lobster --help
echo.
echo Note: Native Windows installation is experimental.
echo For the most reliable experience, consider Docker Desktop:
echo   docker-compose run --rm lobster-cli
echo.
pause
