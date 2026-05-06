param(
  [string]$DailyTime = "18:00",
  [switch]$RunNow
)

$ErrorActionPreference = "Stop"

$taskName = "AshareQuantDailyRefresh"
$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "run_refresh.ps1")).Path

if (-not (Test-Path $scriptPath)) {
  throw "Refresh script not found: $scriptPath"
}

try {
  $atTime = [DateTime]::ParseExact($DailyTime, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
} catch {
  throw "DailyTime format is invalid. Use HH:mm, for example 18:00."
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Daily -At $atTime
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest -LogonType ServiceAccount
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

if ($RunNow) {
  Start-ScheduledTask -TaskName $taskName
}

Write-Host "Daily refresh task registered: $taskName ($DailyTime)"
if ($RunNow) {
  Write-Host "Daily refresh task started: $taskName"
}
