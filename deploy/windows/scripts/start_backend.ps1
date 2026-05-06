$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $backendRoot

$logRoot = "D:\ashare-quant\logs"
$stdoutLog = Join-Path $logRoot "backend.out.log"
$stderrLog = Join-Path $logRoot "backend.err.log"

if (-not (Test-Path $logRoot)) {
  New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
}

if (-not (Test-Path ".env")) {
  throw "backend/.env is missing. Create the production env file before starting the backend."
}

if (-not (Test-Path ".venv")) {
  throw "backend/.venv is missing. Run the backend preparation script first."
}

Add-Content -Path $stdoutLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] starting backend service"
$pythonExe = Join-Path $backendRoot ".venv\Scripts\python.exe"
$arguments = @("manage.py", "run", "--host", "0.0.0.0", "--port", "8000")

$process = Start-Process `
  -FilePath $pythonExe `
  -ArgumentList $arguments `
  -WorkingDirectory $backendRoot `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -PassThru `
  -Wait `
  -NoNewWindow

exit $process.ExitCode
