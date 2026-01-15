@echo off
REM ============================================================================
REM SimOps Offline Update Script
REM ============================================================================
REM Use this to apply updates to an existing SimOps installation
REM Just replace the image tar files and run this script
REM ============================================================================

setlocal EnableDelayedExpansion
title SimOps Offline Updater

echo.
echo  _____ _           ___              
echo /  ___(_)         / _ \             
echo \ `--. _ _ __ ___/ /_\ \_ __  _ __  
echo  `--. \ ^| '_ ` _ \  _  ^| '_ \^| '_ \ 
echo /\__/ / ^| ^| ^| ^| ^| ^| ^| ^| ^|_) ^| ^|_) ^|
echo \____/^|_^|_^| ^|_^| ^|_\_^| ^|_/ .__/^| .__/ 
echo                         ^| ^|   ^| ^|    
echo     OFFLINE UPDATER     ^|_^|   ^|_^|    
echo.
echo ============================================================================
echo.

REM Check Docker
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running!
    pause
    exit /b 1
)

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo [STEP 1/4] Stopping existing services...
echo ============================================================================
docker-compose -f docker-compose-offline.yml down
echo [OK] Services stopped

echo.
echo [STEP 2/4] Loading updated Docker images...
echo ============================================================================

if not exist "images" (
    echo [ERROR] No images folder found. Nothing to update.
    pause
    exit /b 1
)

for %%f in (images\*.tar) do (
    echo Loading: %%~nxf
    docker load -i "%%f"
    if !errorlevel! neq 0 (
        echo [WARN] Failed to load %%~nxf
    ) else (
        echo   Done.
    )
)

echo.
echo [STEP 3/4] Restarting services with new images...
echo ============================================================================
docker-compose -f docker-compose-offline.yml up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services
    pause
    exit /b 1
)

echo.
echo [STEP 4/4] Cleaning up old images...
echo ============================================================================
docker image prune -f
echo [OK] Cleanup complete

echo.
echo ============================================================================
echo                       UPDATE COMPLETE!
echo ============================================================================
echo.
echo SimOps has been updated. Access it at:
echo   http://localhost:8080
echo.
echo Note: Your data (uploads, configurations) has been preserved.
echo ============================================================================
echo.

pause
