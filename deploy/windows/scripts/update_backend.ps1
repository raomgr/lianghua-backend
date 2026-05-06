param(
  [switch]$InstallDeps,
  [switch]$RunRefresh,
  [switch]$RestartService = $true
)

$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $backendRoot

Write-Host "Updating backend repository in $backendRoot"
git pull

if ($InstallDeps) {
  Write-Host "Installing backend dependencies"
  & .\.venv\Scripts\pip install -r requirements.txt
}

if ($RunRefresh) {
  Write-Host "Running backend refresh (sync + train)"
  & .\.venv\Scripts\python manage.py sync
  & .\.venv\Scripts\python manage.py train
}

if ($RestartService) {
  Write-Host "Restarting backend service"
  & (Join-Path $PSScriptRoot "restart_backend.ps1")
}

Write-Host "Backend update completed."
