$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $backendRoot

$logRoot = "D:\ashare-quant\logs"
$stdoutLog = Join-Path $logRoot "refresh.out.log"
$stderrLog = Join-Path $logRoot "refresh.err.log"

if (-not (Test-Path $logRoot)) {
  New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
}

if (-not (Test-Path ".env")) {
  throw "backend\\.env 不存在。请先创建正式配置文件，再执行每日刷新任务。"
}

if (-not (Test-Path ".venv")) {
  throw "backend\\.venv 不存在。请先执行部署初始化脚本。"
}

function Invoke-BackendCommand([string[]]$Arguments) {
  Add-Content -Path $stdoutLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] running: python $($Arguments -join ' ')"
  & .\.venv\Scripts\python @Arguments 1>> $stdoutLog 2>> $stderrLog
  if ($LASTEXITCODE -ne 0) {
    throw "命令执行失败: python $($Arguments -join ' ')"
  }
}

Invoke-BackendCommand @("manage.py", "sync")
Invoke-BackendCommand @("manage.py", "train")

Add-Content -Path $stdoutLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] daily refresh completed"
