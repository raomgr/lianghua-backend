param(
  [string]$PublishRoot = "D:\wwwroot\lianghua-web"
)

$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$appRoot = Split-Path $backendRoot -Parent
$frontendRoot = Join-Path $appRoot "frontend"
$iisConfigPath = Join-Path $backendRoot "deploy\windows\iis\web.config"

Set-Location $frontendRoot

if (-not (Test-Path ".env.production")) {
  Set-Content -Path ".env.production" -Value "VITE_API_BASE_URL=http://116.62.21.194:8000"
}

npm install
npm run build

New-Item -ItemType Directory -Force -Path $publishRoot | Out-Null
Copy-Item ".\dist\*" $publishRoot -Recurse -Force

if (Test-Path $iisConfigPath) {
  Copy-Item $iisConfigPath (Join-Path $publishRoot "web.config") -Force
}

Write-Host "前端静态文件已发布到: $publishRoot"
