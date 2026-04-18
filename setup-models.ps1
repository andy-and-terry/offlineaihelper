param(
    [string]$ConfigPath = ".\config\models.json"
)

$ErrorActionPreference = "Stop"

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw "Ollama CLI is not available. Run .\install.ps1 first."
}

if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$models = @(
    $env:OAH_MODERATION_MODEL,
    $env:OAH_CHAT_MODEL,
    $env:OAH_CODING_MODEL,
    $env:OAH_QA_MODEL,
    $env:OAH_IMAGE_MODEL,
    $env:OAH_LOW_END_MODEL,
    $config.models.moderation,
    $config.models.chat,
    $config.models.coding,
    $config.models.qa,
    $config.models.image,
    $config.models.low_end
) | Where-Object { $_ -ne $null -and $_.Trim().Length -gt 0 } | Select-Object -Unique
Write-Host "Pulling configured Ollama models..."
foreach ($model in $models) {
    Write-Host "  ollama pull $model"
    ollama pull $model
}

Write-Host "Model setup complete."
