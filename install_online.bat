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

echo [3/5] Authenticating with GitHub Container Registry...
echo.
echo You need to log in to download the private SimOps images.
echo Username: Your GitHub Username
echo Password: Your Personal Access Token (PAT) with 'read:packages' scope.
echo.
docker login ghcr.io

if %errorlevel% neq 0 (
    echo [ERROR] Login failed. You cannot pull images without authentication.
    pause
    exit /b 1
)

echo.
echo [4/5] Pulling latest images...
docker-compose -f docker-compose-online.yml pull

echo.
echo [5/5] Starting SimOps services...
REM Clean up any previous instance in this folder to avoid conflicts
docker-compose -f docker-compose-online.yml down --remove-orphans >nul 2>&1
docker-compose -f docker-compose-online.yml up -d

echo.
echo ===============================================================================
echo Installation Complete!
echo ===============================================================================
echo.
echo 1. Access SimOps UI at:  http://localhost:%FRONTEND_PORT%
echo.
echo [DEBUG/Technical]
echo - Backend API:           http://localhost:%API_PORT%
echo - Job Dashboard:         http://localhost:%DASHBOARD_PORT%
echo.
echo The auto-updater (Watchtower) is running and will check for updates hourly.
echo.
pause
exit /b 0
