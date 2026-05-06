param(
  [string]$PublishRoot = "D:\wwwroot\lianghua-web",
  [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$appRoot = Split-Path $backendRoot -Parent
$frontendRoot = Join-Path $appRoot "frontend"

if (-not (Test-Path $frontendRoot)) {
  throw "Frontend source directory not found: $frontendRoot"
}

Set-Location $frontendRoot

Write-Host "Updating frontend repository in $frontendRoot"
git pull

if ($InstallDeps) {
  Write-Host "Installing frontend dependencies"
  npm install
}

Write-Host "Building and publishing frontend"
& (Join-Path $backendRoot "deploy\windows\scripts\build_frontend.ps1") -PublishRoot $PublishRoot

Write-Host "Frontend update completed."
