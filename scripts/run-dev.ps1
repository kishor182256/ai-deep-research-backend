param(
    [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    python -m venv (Join-Path $ProjectRoot ".venv")
}

$Listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Listener) {
    Write-Host "FastAPI already appears to be running on http://127.0.0.1:$Port"
    Write-Host "Open docs: http://127.0.0.1:$Port/docs"
    Write-Host "To use another port, run: .\scripts\run-dev.ps1 -Port 8010"
    exit 0
}

Write-Host "Installing backend packages..."
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host "Starting FastAPI on http://127.0.0.1:$Port"
Set-Location $ProjectRoot
& $Python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload
