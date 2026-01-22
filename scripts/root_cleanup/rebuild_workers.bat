@echo off
REM SimOps Docker Worker Rebuild Script (with Cached Base Image Support)
REM =====================================================================

echo ============================================================
echo SimOps Worker Rebuild Script
echo ============================================================

REM Auto-cleanup temp files
echo.
echo [CLEANUP] Removing temp files from project root...
del /Q "temp_*.msh" 2>nul
del /Q "temp_*.json" 2>nul
del /Q "temp_*.vtk" 2>nul
del /Q "debug_*.msh" 2>nul
del /Q "debug_*.vtk" 2>nul
del /Q "*.frd" 2>nul
del /Q "*.inp" 2>nul
del /Q "SingleTet.msh" 2>nul
del /Q "cube.msh" 2>nul
del /Q "*_surface.msh" 2>nul
del /Q "--version" 2>nul
rmdir /S /Q "temp_test" 2>nul
rmdir /S /Q "temp_test_viz" 2>nul
rmdir /S /Q "tmp_test" 2>nul
rmdir /S /Q "__pycache__" 2>nul
echo   [OK] Cleanup complete

echo.
echo Choose rebuild mode:
echo   [1] FAST Rebuild (uses cached base image, ~5 seconds)
echo   [2] FULL Rebuild (from scratch, ~3 minutes)
echo   [3] Build BASE image only (one-time setup, ~3 minutes)
echo   [4] Cancel
echo.

set /p choice="Enter choice (1-4): "

if "%choice%"=="4" goto :cancel
if "%choice%"=="3" goto :buildbase
if "%choice%"=="2" goto :fullrebuild
if "%choice%"=="1" goto :fastrebuild

echo Invalid choice.
pause
exit /b 1

:cancel
echo Cancelled.
exit /b 0

:buildbase
echo.
echo [BASE] Building base image with OpenFOAM and dependencies...
echo        This will take ~3 minutes but only needs to be done once.
echo.
docker build -t simops-worker-base:latest -f Dockerfile.worker-base .
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Base image build failed
    pause
    exit /b 1
)
echo   [OK] Base image built: simops-worker-base:latest
echo.
echo You can now use option [1] for fast rebuilds!
pause
exit /b 0

:fastrebuild
echo.
echo [1/3] Stopping containers...
docker-compose kill 2>nul
docker-compose down --remove-orphans 2>nul
echo   [OK] Containers stopped

echo.
echo [2/3] FAST Rebuild using cached base image...

REM Check if base image exists
docker image inspect simops-worker-base:latest >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [WARN] Base image not found. Building it first...
    echo.
    docker build -t simops-worker-base:latest -f Dockerfile.worker-base .
    if %ERRORLEVEL% NEQ 0 (
        echo   [ERROR] Base image build failed
        pause
        exit /b 1
    )
    echo   [OK] Base image built
    echo.
)

REM Build worker using the fast Dockerfile
echo   Building worker with cached dependencies...
docker build -t simops-worker:latest -f Dockerfile.worker-fast .
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Fast build failed
    pause
    exit /b 1
)
echo   [OK] Worker image rebuilt

goto :startcontainers

:fullrebuild
echo.
echo [1/3] Stopping containers...
docker-compose kill 2>nul
docker-compose down --remove-orphans 2>nul
echo   [OK] Containers stopped

echo.
echo [2/3] FULL Rebuild from scratch (no cache)...
docker-compose build --no-cache worker
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Build failed
    pause
    exit /b 1
)
echo   [OK] Images rebuilt

goto :startcontainers

:startcontainers
echo.
echo [3/3] Starting containers...
docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Failed to start containers
    pause
    exit /b 1
)
echo   [OK] Containers started

timeout /t 2 /nobreak >nul

echo.
echo ============================================================
echo Rebuild complete! Showing logs (Ctrl+C to exit)...
echo ============================================================
docker-compose logs -f
