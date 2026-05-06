param(
  [string]$SiteName = "Default Web Site",
  [string]$SitePath = "D:\wwwroot\lianghua-web",
  [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

Import-Module WebAdministration

$ipAddress = "*"

if (-not (Test-Path $SitePath)) {
  New-Item -ItemType Directory -Force -Path $SitePath | Out-Null
}

if (-not (Get-Website -Name $SiteName -ErrorAction SilentlyContinue)) {
  New-Website -Name $SiteName -PhysicalPath $SitePath -Port $Port -IPAddress $ipAddress | Out-Null
} else {
  Set-ItemProperty "IIS:\Sites\$SiteName" -Name physicalPath -Value $SitePath
  if (-not (Get-WebBinding -Name $SiteName -Protocol "http" | Where-Object { $_.bindingInformation -eq "*:$Port:" })) {
    New-WebBinding -Name $SiteName -Protocol "http" -Port $Port -IPAddress $ipAddress | Out-Null
  }
  Start-Website -Name $SiteName
}

Write-Host "IIS site ready: $SiteName -> $SitePath (http://$ipAddress:$Port/)"
