@echo off
REM =================================================================
REM SimOps Stop Script (Windows)
REM =================================================================

echo.
echo Stopping SimOps services...
echo.

cd /d "%~dp0"

docker-compose down

echo.
echo SimOps stopped.
echo.
pause
