$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $backendRoot

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

& .\.venv\Scripts\pip install --upgrade pip
& .\.venv\Scripts\pip install -r requirements.txt

if (-not (Test-Path ".env")) {
  throw "backend\.env 不存在。请先创建正式配置文件。"
}

& .\.venv\Scripts\python manage.py sync
& .\.venv\Scripts\python manage.py train
