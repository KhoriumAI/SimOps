@echo off
setlocal enabledelayedexpansion
title SimOps Online Installer
cd /d "%~dp0"

where python >nul 2>&1
if !errorlevel! equ 0 (
    python install_online.py
    exit /b !errorlevel!
)

where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3 install_online.py
    exit /b !errorlevel!
)

echo [ERROR] Python not found. Install Python 3.8+ from https://www.python.org/downloads/
echo.
pause
exit /b 1
