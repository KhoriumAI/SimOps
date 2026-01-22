@echo off
setlocal

echo.
echo ========================================================
echo   SIMOPS ENGINEERING WORKBENCH - INSTALLER / RUNNER
echo ========================================================
echo.

:: Check for Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found. Please install Docker Desktop for Windows.
    echo Visit: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo [1/4] Pulling latest images (optional)...
docker-compose pull

echo [2/4] Starting SimOps services in background...
docker-compose up -d

echo [3/4] Waiting for services to initialize...
echo (This may take 15-30 seconds on first run)
timeout /t 15 /nobreak

echo [4/4] Launching SimOps Workbench in your browser...
start http://localhost:3000

echo.
echo ========================================================
echo   SIMOPS IS RUNNING
echo   API: http://localhost:8000
echo   Frontend: http://localhost:3000
echo ========================================================
echo.
echo To stop SimOps, close this window and run 'docker-compose down'.
echo.
pause
