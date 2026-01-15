<#
.SYNOPSIS
    Sets up the local development environment for the MeshPackageLean project.

.DESCRIPTION
    This script performs the following setup steps:
    1. Checks for backend .env file and creates it from .env.example if missing.
    2. Installs Python dependencies from requirements.txt.
    3. Installs Node.js dependencies for the web-frontend.

.NOTES
    Run this script from the project root directory.
#>

$ErrorActionPreference = "Stop"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   Khorium MeshGen - Local Development Setup" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

# 1. Check Backend .env
Write-Host "`n[1/3] Checking Backend Configuration..." -ForegroundColor Yellow
if (-not (Test-Path "backend\.env")) {
    Write-Host "    backend\.env not found. Creating from .env.example..." -ForegroundColor Gray
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" "backend\.env"
        Write-Host "    [SUCCESS] Created backend\.env. Please edit it with your secrets if needed." -ForegroundColor Green
    } elseif (Test-Path "backend\.env.example") {
         Copy-Item "backend\.env.example" "backend\.env"
         Write-Host "    [SUCCESS] Created backend\.env. Please edit it with your secrets if needed." -ForegroundColor Green
    } else {
        Write-Warning "    Could not find .env.example to copy. Please create backend\.env manually."
    }
} else {
    Write-Host "    [OK] backend\.env exists." -ForegroundColor Green
}

# 2. Python Dependencies
Write-Host "`n[2/3] Installing Python Dependencies..." -ForegroundColor Yellow
# Check if virtual environment is strongly recommended? For now, we assume user is in their desired env.
try {
    pip install -r requirements.txt
    Write-Host "    [SUCCESS] Python dependencies installed." -ForegroundColor Green
} catch {
    Write-Error "    Failed to install Python dependencies. Ensure pip is in your PATH."
}

# 3. Node Dependencies
Write-Host "`n[3/3] Installing Frontend Dependencies..." -ForegroundColor Yellow
if (Test-Path "web-frontend") {
    Push-Location "web-frontend"
    try {
        if (Get-Command "npm" -ErrorAction SilentlyContinue) {
            npm install
            Write-Host "    [SUCCESS] Node dependencies installed." -ForegroundColor Green
        } else {
            Write-Warning "    'npm' command not found. Skipping frontend setup. Please install Node.js."
        }
    } catch {
        Write-Error "    Failed to run npm install."
    } finally {
        Pop-Location
    }
} else {
    Write-Warning "    'web-frontend' directory not found."
}

Write-Host "`n=======================================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "To run the app, use: scripts\run_local_stack.ps1" -ForegroundColor White
