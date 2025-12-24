@echo off
REM SimOps Docker Worker Rebuild Script (Batch version)
REM Kills existing workers, rebuilds containers, and shows logs

echo ============================================================
echo SimOps Worker Rebuild Script
echo ============================================================

REM Step 1: Force stop all containers
echo.
echo [1/3] Force stopping all containers...
docker-compose kill
docker-compose down --remove-orphans
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Failed to stop containers
    exit /b 1
)
echo   [OK] Containers stopped

REM Step 2: Rebuild Docker images
echo.
echo [2/3] Rebuilding Docker images...
docker-compose build --no-cache worker
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Build failed
    exit /b 1
)
echo   [OK] Images rebuilt

echo.
echo   Starting containers...
docker-compose up -d
if %ERRORLEVEL% NEQ 0 (
    echo   [ERROR] Failed to start containers
    exit /b 1
)
echo   [OK] Containers started

REM Wait for containers to initialize
timeout /t 2 /nobreak >nul

REM Step 3: Show logs
echo.
echo [3/3] Showing Docker logs (Ctrl+C to exit)...
echo ============================================================
docker-compose logs -f
