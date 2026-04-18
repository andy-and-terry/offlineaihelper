<#
.SYNOPSIS
    Installs offlineaihelper and its dependencies.
.DESCRIPTION
    Checks that Python >= 3.11 and Ollama are available, installs the Python
    package in editable mode with dev extras, and pulls required Ollama models.
#>

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Success { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Failure { param([string]$Msg) Write-Host "  [FAIL] $Msg" -ForegroundColor Red }
function Write-Step    { param([string]$Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------
# 1. Python >= 3.11 check
# ---------------------------------------------------------------------------
Write-Step "Checking Python version"

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
}
if (-not $pythonCmd) {
    Write-Failure "Python not found. Install Python 3.11+ from https://python.org and re-run."
    exit 1
}

$versionOutput = & $pythonCmd.Source --version 2>&1
if ($versionOutput -match 'Python (\d+)\.(\d+)') {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
        Write-Failure "Python $major.$minor detected. Python 3.11+ is required."
        exit 1
    }
    Write-Success "Python $major.$minor found at $($pythonCmd.Source)"
} else {
    Write-Failure "Could not determine Python version from: $versionOutput"
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Ollama installed & running
# ---------------------------------------------------------------------------
Write-Step "Checking Ollama"

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    Write-Failure "Ollama not found. Install from https://ollama.com and re-run."
    exit 1
}
Write-Success "Ollama binary found at $($ollamaCmd.Source)"

try {
    $ollamaVersion = & ollama --version 2>&1
    Write-Success "Ollama version: $ollamaVersion"
} catch {
    Write-Failure "Could not run 'ollama --version': $_"
    exit 1
}

# Check if server is reachable
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    Write-Success "Ollama server is running at http://localhost:11434"
} catch {
    Write-Failure "Ollama server not reachable at http://localhost:11434. Start it with 'ollama serve' and re-run."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Install Python package
# ---------------------------------------------------------------------------
Write-Step "Installing offlineaihelper[dev]"

try {
    & $pythonCmd.Source -m pip install --quiet -e ".[dev]"
    Write-Success "Package installed successfully"
} catch {
    Write-Failure "pip install failed: $_"
    exit 1
}

# ---------------------------------------------------------------------------
# 3b. Install Node.js dependencies
# ---------------------------------------------------------------------------
Write-Step "Installing Node.js dependencies"

$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeCmd) {
    Write-Failure "Node.js not found. Install Node.js 18+ from https://nodejs.org and re-run."
    exit 1
}
$nodeVersion = & node --version 2>&1
Write-Success "Node.js $nodeVersion found"

$npmCmd = Get-Command npm -ErrorAction SilentlyContinue
if (-not $npmCmd) {
    Write-Failure "npm not found. Install Node.js 18+ from https://nodejs.org and re-run."
    exit 1
}

$nodeDir = Join-Path $PSScriptRoot "node"
if (Test-Path $nodeDir) {
    Push-Location $nodeDir
    try {
        & npm install --silent
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "npm install failed"
            exit 1
        }
        Write-Success "Node.js dependencies installed"
    } finally {
        Pop-Location
    }
} else {
    Write-Failure "node/ directory not found at $nodeDir"
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Pull Ollama models
# ---------------------------------------------------------------------------
Write-Step "Pulling Ollama models"

$pullScript = Join-Path $PSScriptRoot "scripts\pull-models.ps1"
if (Test-Path $pullScript) {
    & $pullScript
    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Model pull script failed (exit code $LASTEXITCODE)"
        exit 1
    }
} else {
    Write-Failure "pull-models.ps1 not found at $pullScript"
    exit 1
}

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
Write-Host "`n================================================" -ForegroundColor Green
Write-Host "  offlineaihelper installed successfully!" -ForegroundColor Green
Write-Host "  Run (Python CLI):  offlineaihelper ask --prompt 'Hello!'" -ForegroundColor Green
Write-Host "  Run (Node.js CLI): node node/src/cli.js ask --prompt 'Hello!'" -ForegroundColor Green
Write-Host "================================================`n" -ForegroundColor Green
