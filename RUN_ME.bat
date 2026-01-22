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

echo Select startup mode:
echo   [R] Rebuild Docker images (backend/frontend) and run
echo   [S] Start with existing images
echo   [Q] Quit
echo.

:prompt
set /p choice=Choose [R/S/Q]: 
if /I "%choice%"=="R" goto rebuild
if /I "%choice%"=="S" goto run
if /I "%choice%"=="Q" exit /b 0
echo Invalid choice. Please enter R, S, or Q.
echo.
goto prompt

:rebuild
echo Rebuilding Docker images...
docker-compose build
goto run

:run
echo Starting SimOps services in background...
docker-compose up -d

echo Waiting for services to initialize...
echo (This may take 15-30 seconds on first run)
timeout /t 15 /nobreak

echo Launching SimOps Workbench in your browser...
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
