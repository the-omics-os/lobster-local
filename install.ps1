# Lobster AI - Windows Installation Script
# This script sets up Lobster AI on Windows systems

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$ESC = [char]27
$RED = "$ESC[31m"
$GREEN = "$ESC[32m"
$YELLOW = "$ESC[33m"
$BLUE = "$ESC[34m"
$RESET = "$ESC[0m"

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = $RESET
    )
    Write-Host "${Color}${Message}${RESET}"
}

function Test-PythonVersion {
    param([string]$PythonCmd)

    try {
        $version = & $PythonCmd --version 2>&1 | Out-String
        if ($version -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]

            if ($major -ge 3 -and $minor -ge 12) {
                return $true
            }
        }
    }
    catch {
        return $false
    }
    return $false
}

function Find-Python {
    $pythonCandidates = @("python", "python3", "python3.12", "python3.13", "py")

    foreach ($cmd in $pythonCandidates) {
        if (Test-PythonVersion -PythonCmd $cmd) {
            return $cmd
        }
    }

    return $null
}

# Header
Write-Host ""
Write-ColorOutput "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" $BLUE
Write-ColorOutput "   🦞 Lobster AI - Windows Installation" $BLUE
Write-ColorOutput "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" $BLUE
Write-Host ""

# Check Python
Write-ColorOutput "🔍 Checking for Python 3.12+..." $BLUE
$python = Find-Python

if ($null -eq $python) {
    Write-ColorOutput "❌ Python 3.12 or higher not found!" $RED
    Write-Host ""
    Write-ColorOutput "Please install Python 3.12+ from:" $YELLOW
    Write-ColorOutput "  https://www.python.org/downloads/" $YELLOW
    Write-Host ""
    Write-ColorOutput "Make sure to check 'Add Python to PATH' during installation!" $YELLOW
    exit 1
}

$pythonVersion = & $python --version
Write-ColorOutput "✅ Found: $pythonVersion" $GREEN
Write-Host ""

# Check if virtual environment already exists
$venvPath = ".venv"
if (Test-Path $venvPath) {
    Write-ColorOutput "⚠️  Virtual environment already exists at $venvPath" $YELLOW
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Write-ColorOutput "🗑️  Removing existing virtual environment..." $BLUE
        Remove-Item -Recurse -Force $venvPath
    }
    else {
        Write-ColorOutput "Using existing virtual environment..." $GREEN
    }
}

# Create virtual environment
if (-not (Test-Path $venvPath)) {
    Write-ColorOutput "🐍 Creating virtual environment..." $BLUE
    & $python -m venv $venvPath

    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "❌ Failed to create virtual environment!" $RED
        Write-ColorOutput "Try: $python -m pip install --upgrade pip" $YELLOW
        exit 1
    }
    Write-ColorOutput "✅ Virtual environment created" $GREEN
    Write-Host ""
}

# Activate virtual environment and install
$venvPython = ".\$venvPath\Scripts\python.exe"
$activateScript = ".\$venvPath\Scripts\Activate.ps1"

Write-ColorOutput "📦 Installing Lobster AI and dependencies..." $BLUE
Write-ColorOutput "(This may take 3-5 minutes on first install)" $YELLOW
Write-Host ""

# Upgrade pip first
& $venvPython -m pip install --quiet --upgrade pip setuptools wheel

# Install Lobster
& $venvPython -m pip install -e .

if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "❌ Installation failed!" $RED
    Write-Host ""
    Write-ColorOutput "Common issues:" $YELLOW
    Write-ColorOutput "  • If you see compiler errors, you may need Visual Studio Build Tools" $YELLOW
    Write-ColorOutput "  • Download from: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" $YELLOW
    Write-ColorOutput "  • Or use Docker instead: docker-compose run --rm lobster-cli" $YELLOW
    exit 1
}

Write-ColorOutput "✅ Installation complete!" $GREEN
Write-Host ""

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-ColorOutput "📝 Creating .env configuration file..." $BLUE

    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
    }
    else {
        # Create basic .env file
        @"
# Lobster AI Configuration
# Add your API key below (required)

# Option 1: Claude API (recommended for quick start)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Option 2: AWS Bedrock (recommended for production)
# AWS_BEDROCK_ACCESS_KEY=your-key
# AWS_BEDROCK_SECRET_ACCESS_KEY=your-secret

# Optional: NCBI API key for enhanced literature search
# NCBI_API_KEY=your-ncbi-key
"@ | Out-File -FilePath ".env" -Encoding UTF8
    }

    Write-ColorOutput "✅ Created .env file" $GREEN
    Write-Host ""
    Write-ColorOutput "⚠️  IMPORTANT: Edit .env and add your API key!" $YELLOW
    Write-ColorOutput "   Run: notepad .env" $YELLOW
    Write-Host ""
}

# Success message
Write-ColorOutput "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" $GREEN
Write-ColorOutput "   ✅ Installation Complete!" $GREEN
Write-ColorOutput "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" $GREEN
Write-Host ""

Write-ColorOutput "Next steps:" $BLUE
Write-Host ""
Write-ColorOutput "1. Configure your API key:" $BLUE
Write-ColorOutput "   notepad .env" $YELLOW
Write-Host ""
Write-ColorOutput "2. Activate the virtual environment:" $BLUE
Write-ColorOutput "   $activateScript" $YELLOW
Write-Host ""
Write-ColorOutput "3. Start using Lobster AI:" $BLUE
Write-ColorOutput "   lobster chat" $YELLOW
Write-Host ""

Write-ColorOutput "For help and documentation:" $BLUE
Write-ColorOutput "   lobster --help" $YELLOW
Write-ColorOutput "   https://github.com/the-omics-os/lobster-local/wiki" $YELLOW
Write-Host ""

Write-ColorOutput "⚠️  Note: Native Windows installation is experimental." $YELLOW
Write-ColorOutput "   For the most reliable experience, consider using Docker Desktop:" $YELLOW
Write-ColorOutput "   docker-compose run --rm lobster-cli" $YELLOW
Write-Host ""
