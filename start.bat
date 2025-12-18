@echo off
REM =================================================================
REM SimOps Startup Script (Windows)
REM =================================================================
REM Starts the SimOps Docker appliance with all services
REM
REM Usage:
REM   start.bat              - Start with defaults
REM   start.bat --build      - Rebuild images
REM =================================================================

echo.
echo ============================================================
echo    SIMOPS - Thermal Analysis Vending Machine
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Create directories
if not exist "input" mkdir input
if not exist "output" mkdir output
if not exist "logs" mkdir logs

REM Copy .env if not exists
if not exist ".env" (
    echo Creating .env from .env.example...
    copy .env.example .env
)

REM Check Docker
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

REM Start services
echo.
echo Starting SimOps services...

if "%1"=="--build" (
    docker-compose up -d --build
) else (
    docker-compose up -d
)

REM Wait for services
echo.
echo Waiting for services to be healthy...
timeout /t 5 /nobreak >nul

REM Show status
echo.
docker-compose ps

echo.
echo ============================================================
echo    SimOps is running!
echo ============================================================
echo.
echo    Dashboard:  http://localhost:9181
echo    Input:      %cd%\input\
echo    Output:     %cd%\output\
echo.
echo    Drop STEP files into the input folder to start simulations.
echo.
echo    Commands:
echo      docker-compose logs -f          View logs
echo      docker-compose down             Stop services
echo      docker-compose ps               Check status
echo.
pause
