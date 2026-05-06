param(
  [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$taskName = "AshareQuantBackend"
$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "start_backend.ps1")).Path

if (-not (Test-Path $scriptPath)) {
  throw "Startup script not found: $scriptPath"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest -LogonType ServiceAccount
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
if ($StartNow) {
  Start-ScheduledTask -TaskName $taskName
}

Write-Host "Scheduled task registered: $taskName"
if ($StartNow) {
  Write-Host "Scheduled task started: $taskName"
}
