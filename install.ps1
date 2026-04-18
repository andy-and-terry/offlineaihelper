$ErrorActionPreference = "Stop"

Write-Host "Checking for Ollama installation..."
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "Ollama is already installed."
    exit 0
}

$installerPath = Join-Path $env:TEMP "OllamaSetup.exe"
Write-Host "Downloading Ollama installer to $installerPath"
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installerPath

Write-Host "Running Ollama installer..."
Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    throw "Ollama installation did not complete correctly. Install manually from https://ollama.com/download/windows"
}

Write-Host "Ollama installed successfully."
