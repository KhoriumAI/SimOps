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

echo [1/4] Creating local logs directory...
if not exist "logs" mkdir logs

echo [2/4] Authenticating with GitHub Container Registry...
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
echo [3/4] Pulling latest images...
docker-compose -f docker-compose-online.yml pull

echo.
echo [4/4] Starting SimOps services...
REM Clean up any previous instance in this folder to avoid conflicts
docker-compose -f docker-compose-online.yml down --remove-orphans >nul 2>&1
docker-compose -f docker-compose-online.yml up -d

echo.
echo ===============================================================================
echo Installation Complete!
echo ===============================================================================
echo.
echo 1. Access SimOps UI at:  http://localhost:5173
echo.
echo [DEBUG/Technical]
echo - Backend API:           http://localhost:3001
echo - Job Dashboard:         http://localhost:9181
echo.
echo The auto-updater (Watchtower) is running and will check for updates hourly.
echo.
pause
