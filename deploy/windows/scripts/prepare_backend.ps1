$ErrorActionPreference = "Stop"

$backendRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $backendRoot

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

& .\.venv\Scripts\pip install --upgrade pip
& .\.venv\Scripts\pip install -r requirements.txt

if (-not (Test-Path ".env")) {
  throw "backend/.env is missing. Create the production env file first."
}

& .\.venv\Scripts\python manage.py sync
& .\.venv\Scripts\python manage.py train
