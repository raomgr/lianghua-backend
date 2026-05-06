param(
  [switch]$SkipStop
)

$ErrorActionPreference = "Stop"

$taskName = "AshareQuantBackend"
$port = 8000

function Stop-BackendProcessByPort {
  $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
  if (-not $connections) {
    Write-Host "No listening process found on port $port."
    return
  }

  $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($processId in $processIds) {
    if ($processId -and $processId -gt 0) {
      try {
        Stop-Process -Id $processId -Force -ErrorAction Stop
        Write-Host "Stopped process on port ${port}: PID=$processId"
      } catch {
        Write-Warning "Failed to stop PID ${processId}: $($_.Exception.Message)"
      }
    }
  }
}

if (-not $SkipStop) {
  Stop-BackendProcessByPort
  Start-Sleep -Seconds 2
}

Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3

$connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($connections) {
  Write-Host "Backend restarted successfully on port $port."
} else {
  Write-Warning "Backend task started, but port $port is not listening yet. Check logs in D:\ashare-quant\logs."
}
