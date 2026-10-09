<#
Serve the Movie Recommendation API so phones, tablets and other computers on the
same Wi-Fi can open it.

Run from the project folder with the venv active:
    powershell -ExecutionPolicy Bypass -File .\serve_lan.ps1
    powershell -ExecutionPolicy Bypass -File .\serve_lan.ps1 -Port 8001

Run once as Administrator to let the script open the Windows Firewall port.
#>
param([int]$Port = 8001)

Set-Location $PSScriptRoot

# LAN address of the adapter that has a default gateway (your real Wi-Fi/Ethernet).
$cfg = Get-NetIPConfiguration |
    Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq 'Up' } |
    Select-Object -First 1
$ip = $cfg.IPv4Address.IPAddress

# Firewall rule (needs admin the first time).
$ruleName = "Movie API $Port"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    $isAdmin = ([Security.Principal.WindowsPrincipal] `
        [Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if ($isAdmin) {
        New-NetFirewallRule -DisplayName $ruleName -Direction Inbound `
            -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
        Write-Host "Firewall: opened port $Port." -ForegroundColor Green
    } else {
        Write-Host "Firewall rule missing. If other devices cannot connect," -ForegroundColor Yellow
        Write-Host "re-run this script once as Administrator." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Open on any device on the same Wi-Fi:" -ForegroundColor Cyan
if ($ip) {
    Write-Host "  Demo : http://${ip}:$Port/"
    Write-Host "  Docs : http://${ip}:$Port/docs#/Recommendations/popular_recommendations"
    Write-Host "  Health: http://${ip}:$Port/health"
} else {
    Write-Host "  Could not detect your LAN IP. Run 'ipconfig' and use the IPv4 address." -ForegroundColor Yellow
}
Write-Host "On this PC: http://127.0.0.1:$Port/docs"
Write-Host ""

python -m uvicorn api:app --host 0.0.0.0 --port $Port
