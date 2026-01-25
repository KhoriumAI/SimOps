@echo off
setlocal enabledelayedexpansion
title SimOps Engineering Workbench

echo ========================================================
echo   SIMOPS ENGINEERING WORKBENCH - INSTALLER / RUNNER
echo ========================================================
echo.

:MENU
echo Select startup mode:
echo   [R] Rebuild Docker images (backend/frontend) and run
echo   [S] Start with existing images
echo   [U] Update images from GitHub (Online Mode)
echo   [D] Stop and clean up (Docker Down)
echo   [Q] Quit
echo.
set /p CHOICE="Choose [R/S/U/D/Q]: "

if /i "%CHOICE%"=="r" goto REBUILD
if /i "%CHOICE%"=="s" goto START
if /i "%CHOICE%"=="u" goto UPDATE
if /i "%CHOICE%"=="d" goto DOWN
if /i "%CHOICE%"=="q" exit /b 0

echo Invalid choice.
goto MENU

:REBUILD
echo Rebuilding Docker images...
docker-compose up -d --build
goto FINISH

:START
echo Starting SimOps services...
docker-compose up -d
goto FINISH

:UPDATE
echo Checking for updates on GitHub...
call install_online.bat
exit /b 0

:DOWN
echo Stopping SimOps services...
docker-compose down
echo Done.
pause
goto MENU

:FINISH
if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Something went wrong while starting services.
    pause
    goto MENU
)

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
exit /b 0
