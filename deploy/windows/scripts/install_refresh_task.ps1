param(
  [string]$DailyTime = "18:00",
  [switch]$RunNow
)

$ErrorActionPreference = "Stop"

$taskName = "AshareQuantDailyRefresh"
$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "run_refresh.ps1")).Path

if (-not (Test-Path $scriptPath)) {
  throw "未找到刷新脚本: $scriptPath"
}

try {
  $atTime = [DateTime]::ParseExact($DailyTime, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
} catch {
  throw "DailyTime 格式不正确，请使用 HH:mm，例如 18:00"
}

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Daily -At $atTime
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest -LogonType ServiceAccount
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null

if ($RunNow) {
  Start-ScheduledTask -TaskName $taskName
}

Write-Host "已注册每日刷新任务: $taskName ($DailyTime)"
if ($RunNow) {
  Write-Host "已立即启动每日刷新任务: $taskName"
}
