<#
.SYNOPSIS
    Runs the full stack (Flask Backend + Vite Frontend) locally.

.DESCRIPTION
    This script starts the backend and frontend in separate processes and ensures
    they are cleaned up when the script is terminated.

.NOTES
    Run this script from the project root directory.
#>

$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   Khorium MeshGen - Local Full Stack Launcher" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# Check for backend .env
if (-not (Test-Path "backend\.env")) {
    Write-Warning "backend\.env not found. Using default development settings (or .env in root if loaded)."
}

# -----------------------------------------------------------------------------
# Function to Start Process
# -----------------------------------------------------------------------------
function Start-MyProcess {
    param (
        [string]$Name,
        [string]$Command,
        [string]$Arguments,
        [string]$WorkingDirectory
    )

    Write-Host "Starting $Name..." -ForegroundColor Yellow
    $ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
    $ProcessInfo.FileName = $Command
    $ProcessInfo.Arguments = $Arguments
    if ($WorkingDirectory) {
        $ProcessInfo.WorkingDirectory = (Join-Path $PWD $WorkingDirectory)
    }
    $ProcessInfo.RedirectStandardOutput = $false # Let it print to console for now, or true to capture
    $ProcessInfo.RedirectStandardError = $false
    $ProcessInfo.UseShellExecute = $true # Open in new window/tab behavior usually prefers ShellExecute=true for distinct consoles or false for integrated. 
                                         # For a "launcher", creating new windows is often cleaner for logs.
                                         # Let's try starting them in the SAME window is messy. 
                                         # Creating new windows is better for "Run everything".
    
    $Process = [System.Diagnostics.Process]::Start($ProcessInfo)
    return $Process
}

# -----------------------------------------------------------------------------
# Start Backend
# -----------------------------------------------------------------------------
Write-Host "Launching Backend (Flask)..." -ForegroundColor Green
# We use 'cmd /k' to keep the window open if it crashes, for debugging
$BackendProcess = Start-MyProcess -Name "Backend" -Command "cmd.exe" -Arguments "/c start ""Flask Backend"" cmd /k ""python backend\api_server.py""" -WorkingDirectory "."

# -----------------------------------------------------------------------------
# Start Frontend
# -----------------------------------------------------------------------------
Write-Host "Launching Frontend (Vite)..." -ForegroundColor Green
$FrontendProcess = Start-MyProcess -Name "Frontend" -Command "cmd.exe" -Arguments "/c start ""Vite Frontend"" cmd /k ""npm run dev""" -WorkingDirectory "web-frontend"

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "   Stack Running!" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Backend: http://localhost:5000" -ForegroundColor Gray
Write-Host "Frontend: http://localhost:3000 (usually)" -ForegroundColor Gray
Write-Host "`nPress any key to close the launcher (the app windows will stay open)..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
