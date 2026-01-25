@echo off
setlocal
title SimOps Online Installer

echo ===============================================================================
echo SimOps Online Installer
echo ===============================================================================
echo.
echo This script will install SimOps using the latest images from GitHub Registry.
echo.

REM Check for Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed or not in PATH.
    echo Please install Docker Desktop for Windows and try again.
    pause
    exit /b 1
)

echo [1/5] Finding available ports...
set FRONTEND_PORT=5173
set API_PORT=3001
set DASHBOARD_PORT=9181

:check_frontend
netstat -ano | findstr /R /C:":%FRONTEND_PORT% " >nul
if %errorlevel% equ 0 (
    set /a FRONTEND_PORT+=1
    goto check_frontend
)

:check_api
netstat -ano | findstr /R /C:":%API_PORT% " >nul
if %errorlevel% equ 0 (
    set /a API_PORT+=1
    goto check_api
)

:check_dashboard
netstat -ano | findstr /R /C:":%DASHBOARD_PORT% " >nul
if %errorlevel% equ 0 (
    set /a DASHBOARD_PORT+=1
    goto check_dashboard
)

echo - Frontend Port:  %FRONTEND_PORT%
echo - API Port:       %API_PORT%
echo - Dashboard Port: %DASHBOARD_PORT%

echo [2/5] Creating local logs directory...
if not exist "logs" mkdir logs

echo [3/5] Checking image status...
set NEED_PULL=1
docker images -q ghcr.io/khoriumai/simops-frontend:latest >nul 2>&1
if %errorlevel% equ 0 (
    echo - Images found locally.
    set /p UPDATE_CHOICE="Check for updates? (y/N): "
    if /i "%UPDATE_CHOICE%" neq "y" set NEED_PULL=0
)

if %NEED_PULL% equ 1 (
    echo.
    echo [4/5] Authenticating and pulling latest images...
    echo Username: Your GitHub Username
    echo Password: Your Personal Access Token (PAT)
    docker login ghcr.io
    if %errorlevel% neq 0 (
        echo [ERROR] Login failed.
        pause
        exit /b 1
    )
    docker-compose -f docker-compose-online.yml pull
) else (
    echo [4/5] Skipping download (using local images).
)

echo.
echo [5/5] Starting SimOps services...
docker-compose -f docker-compose-online.yml up -d

echo.
echo ===============================================================================
echo Installation Complete! Opening SimOps...
echo ===============================================================================
echo.
timeout /t 3 >nul
start http://localhost:%FRONTEND_PORT%
echo.
echo - Job Dashboard: http://localhost:%DASHBOARD_PORT%
echo.
echo The auto-updater (Watchtower) is running in the background.
echo.
timeout /t 5
exit /b 0
