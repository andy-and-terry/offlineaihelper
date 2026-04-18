<#
.SYNOPSIS
    Pulls all Ollama models defined in config/models.json.
.DESCRIPTION
    Reads assistant and moderator model names from the config, runs
    'ollama pull' for each, then verifies they appear in 'ollama list'.
    Exits with code 1 on failure when running in strict mode.
#>

[CmdletBinding()]
param(
    [switch]$NoStrict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Success { param([string]$Msg) Write-Host "  [OK] $Msg" -ForegroundColor Green }
function Write-Failure { param([string]$Msg) Write-Host "  [FAIL] $Msg" -ForegroundColor Red }
function Write-Step    { param([string]$Msg) Write-Host "`n==> $Msg" -ForegroundColor Cyan }

$StrictMode = -not $NoStrict.IsPresent

# ---------------------------------------------------------------------------
# Load config/models.json
# ---------------------------------------------------------------------------
$scriptDir = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $scriptDir "config\models.json"

if (-not (Test-Path $configPath)) {
    Write-Failure "models.json not found at $configPath"
    exit 1
}

$config = Get-Content $configPath -Raw | ConvertFrom-Json

$models = @(
    $config.assistant.ollama_model,
    $config.moderator.ollama_model
)

Write-Step "Models to pull"
foreach ($m in $models) { Write-Host "  • $m" }

# ---------------------------------------------------------------------------
# Pull each model
# ---------------------------------------------------------------------------
foreach ($model in $models) {
    Write-Step "Pulling model: $model"
    try {
        & ollama pull $model
        if ($LASTEXITCODE -ne 0) {
            Write-Failure "ollama pull $model exited with code $LASTEXITCODE"
            if ($StrictMode) { exit 1 }
        }
    } catch {
        Write-Failure "Error pulling ${model}: $_"
        if ($StrictMode) { exit 1 }
    }
}

# ---------------------------------------------------------------------------
# Verify models appear in ollama list
# ---------------------------------------------------------------------------
Write-Step "Verifying pulled models"

$listOutput = & ollama list 2>&1
$listText = $listOutput | Out-String

foreach ($model in $models) {
    # Strip tag if present for a loose check; Ollama may normalise names.
    $baseName = ($model -split ':')[0]
    if ($listText -match [regex]::Escape($baseName)) {
        Write-Success "$model is present"
    } else {
        Write-Failure "$model NOT found in 'ollama list' output"
        if ($StrictMode) {
            Write-Host "ollama list output:`n$listText" -ForegroundColor Yellow
            exit 1
        }
    }
}

Write-Host "`nAll models pulled successfully.`n" -ForegroundColor Green
