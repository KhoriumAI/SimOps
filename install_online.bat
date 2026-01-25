@echo off
setlocal enabledelayedexpansion
title SimOps Online Installer

echo ===============================================================================
echo SimOps Online Installer
echo ===============================================================================
echo.
echo This script will install SimOps using the latest images from GitHub Registry.
echo.

REM Check for Docker
docker --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop for Windows and try again.
    pause
    exit /b 1
)

echo [1/5] Finding available ports...
set FRONTEND_PORT=3000
set API_PORT=8000
set DASHBOARD_PORT=9181

:check_frontend
netstat -ano | findstr /R /C:":!FRONTEND_PORT! " >nul
if !errorlevel! equ 0 (
    set /a FRONTEND_PORT+=1
    goto check_frontend
)

:check_api
netstat -ano | findstr /R /C:":!API_PORT! " >nul
if !errorlevel! equ 0 (
    set /a API_PORT+=1
    goto check_api
)

:check_dashboard
netstat -ano | findstr /R /C:":!DASHBOARD_PORT! " >nul
if !errorlevel! equ 0 (
    set /a DASHBOARD_PORT+=1
    goto check_dashboard
)

echo - Frontend Port:  !FRONTEND_PORT!
echo - API Port:       !API_PORT!
echo - Dashboard Port: !DASHBOARD_PORT!

echo.
echo [2/5] Creating local logs directory...
if not exist "logs" mkdir logs

echo.
echo [3/5] Checking for updates...
set NEED_PULL=0
docker images -q ghcr.io/khoriumai/simops-frontend:latest >nul 2>&1
if !errorlevel! neq 0 (
    echo - Fresh installation detected.
    set NEED_PULL=1
) else (
    echo - Comparing local version with GitHub...
    REM Try to get remote digest. This might fail if not logged in or no internet.
    powershell -Command "$local = (docker images ghcr.io/khoriumai/simops-frontend:latest --format '{{.Digest}}'); if (-not $local) { exit 0 }; try { $manifest = (docker manifest inspect ghcr.io/khoriumai/simops-frontend:latest | ConvertFrom-Json); if ($manifest.manifests) { $remote = $manifest.manifests[0].digest } else { $remote = $manifest.config.digest }; if ($remote -and $local -ne $remote) { exit 1 } else { exit 0 } } catch { exit 0 }" >nul 2>&1
    
    if !errorlevel! equ 1 (
        echo.
        echo ****************************************
        echo    UPDATE AVAILABLE ON GITHUB!
        echo ****************************************
        echo.
        set /p UPDATE_CHOICE="Download latest version? (y/N): "
        if /i "!UPDATE_CHOICE!"=="y" set NEED_PULL=1
    ) else (
        echo - System is up to date.
    )
)

echo.
if !NEED_PULL! equ 1 (
    echo [4/5] Pulling latest images from GitHub...
    docker-compose -f docker-compose-online.yml pull
    if !errorlevel! neq 0 (
        echo [WARNING] Failed to pull images. Will try to use local versions.
        pause
    )
) else (
    echo [4/5] Using local images (up to date).
)

echo.
echo [5/5] Starting SimOps services...
docker-compose -f docker-compose-online.yml up -d
if !errorlevel! neq 0 (
    echo [ERROR] Failed to start services.
    pause
    exit /b 1
)

echo.
echo ===============================================================================
echo Installation Complete! opening SimOps...
echo ===============================================================================
echo.
echo - Workbench:     http://localhost:!FRONTEND_PORT!
echo - API Server:    http://localhost:!API_PORT!
echo.
echo ===============================================================================
echo.
timeout /t 3 >nul
start http://localhost:!FRONTEND_PORT!
pause
exit /b 0
