@echo off
REM ============================================================================
REM SimOps Offline Installer
REM ============================================================================
REM This script loads Docker images and starts SimOps services
REM Run this on an air-gapped machine with Docker Desktop installed
REM ============================================================================

setlocal EnableDelayedExpansion
title SimOps Offline Installer

echo.
echo  _____ _           ___              
echo /  ___(_)         / _ \             
echo \ `--. _ _ __ ___/ /_\ \_ __  _ __  
echo  `--. \ ^| '_ ` _ \  _  ^| '_ \^| '_ \ 
echo /\__/ / ^| ^| ^| ^| ^| ^| ^| ^|_) ^| ^|_) ^|
echo \____/^|_^|_^| ^|_^| ^|_\_^| ^|_/ .__/^| .__/ 
echo                         ^| ^|   ^| ^|    
echo     OFFLINE INSTALLER   ^|_^|   ^|_^|    
echo.
echo ============================================================================
echo.

REM ============================================================================
REM OPENFOAM PREREQUISITE CHECK
REM ============================================================================
echo ============================================================================
echo                     PREREQUISITE: OpenFOAM Installation
echo ============================================================================
echo.
echo SimOps requires OpenFOAM for CFD simulations. If you only need meshing and
echo CalculiX thermal analysis, you can skip OpenFOAM.
echo.
echo  +------------------------------------------------------------------------+
echo  ^|  REQUIRED VERSION: ESI OpenFOAM v2312 (from openfoam.com)             ^|
echo  ^|                                                                        ^|
echo  ^|  WARNING: DO NOT install OpenFOAM Foundation (cfd.direct) or other    ^|
echo  ^|           versions. Schema formats differ and WILL cause errors!      ^|
echo  +------------------------------------------------------------------------+
echo.
echo  INSTALLATION OPTIONS:
echo.
echo  [OPTION A - Docker (Recommended)]
echo    Docker Desktop handles OpenFOAM automatically via the worker container.
echo    No additional installation required if proceeding with this installer.
echo.
echo  [OPTION B - WSL2 (Windows)]
echo    1. Enable WSL2:  wsl --install
echo    2. Install Ubuntu from Microsoft Store
echo    3. In Ubuntu terminal, run:
echo       curl -s https://dl.openfoam.com/add-debian-repo.sh ^| sudo bash
echo       sudo apt install openfoam2312-default
echo       echo 'source /usr/lib/openfoam/openfoam2312/etc/bashrc' ^>^> ~/.bashrc
echo.
echo  [OPTION C - macOS (Docker)]
echo    Install Docker Desktop for Mac, then this installer handles the rest.
echo.
echo  Download Link: https://www.openfoam.com/download/install-windows
echo.
echo ============================================================================
echo.

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running!
    echo.
    echo Please start Docker Desktop and try again.
    echo.
    pause
    exit /b 1
)
echo [OK] Docker is running

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check for images folder
if not exist "images" (
    echo [ERROR] Images folder not found!
    echo.
    echo Expected folder structure:
    echo   %SCRIPT_DIR%
    echo   +-- images/
    echo   ^|   +-- backend.tar
    echo   ^|   +-- frontend.tar
    echo   ^|   +-- worker.tar
    echo   ^|   +-- redis.tar
    echo   ^|   +-- dashboard.tar
    echo   +-- docker-compose-offline.yml
    echo   +-- install.bat
    echo.
    pause
    exit /b 1
)

echo.
echo [STEP 1/3] Loading Docker Images...
echo ============================================================================

for %%f in (images\*.tar) do (
    echo Loading: %%~nxf
    docker load -i "%%f"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to load %%~nxf
        pause
        exit /b 1
    )
    echo   Done.
)

echo.
echo [OK] All images loaded successfully!
echo.

REM Create .env from template if it doesn't exist
if not exist ".env" (
    if exist ".env.template" (
        echo Creating .env from template...
        copy ".env.template" ".env" >nul
        echo [OK] Created .env configuration file
        echo.
        echo [NOTE] You can edit .env to customize ports and settings
        echo.
    )
)

echo [STEP 2/3] Starting SimOps Services...
echo ============================================================================

docker-compose -f docker-compose-offline.yml up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services!
    pause
    exit /b 1
)

echo.
echo [OK] Services started!
echo.

echo [STEP 3/3] Waiting for services to be ready...
echo ============================================================================

REM Wait for backend health check
set /a attempts=0
:healthcheck
set /a attempts+=1
if %attempts% gtr 30 (
    echo [WARNING] Services are taking longer than expected to start.
    echo           They may still be initializing. Check Docker Dashboard.
    goto :done
)

timeout /t 2 /nobreak >nul
curl -s http://localhost:5000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo   Waiting for backend... (attempt %attempts%/30)
    goto :healthcheck
)

echo [OK] Backend is healthy!

:done
echo.
echo ============================================================================
echo                     INSTALLATION COMPLETE!
echo ============================================================================
echo.
echo SimOps is now running! Access it at:
echo.
echo   Web Application:  http://localhost:8080
echo   API Server:       http://localhost:5000
echo   Job Dashboard:    http://localhost:9181
echo.
echo Useful commands:
echo   View logs:        docker-compose -f docker-compose-offline.yml logs -f
echo   Stop services:    docker-compose -f docker-compose-offline.yml down
echo   Restart:          docker-compose -f docker-compose-offline.yml restart
echo.
echo ============================================================================
echo.

REM Open browser
echo Opening SimOps in your default browser...
start http://localhost:8080

pause
