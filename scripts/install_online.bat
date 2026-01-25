@echo off
setlocal

echo ================================================================================
echo                        SIMOPS ONLINE LAUNCHER
echo ================================================================================

echo.
echo [1/2] Creating/Checking local configuration...
if not exist .env (
    if exist .env.template (
        copy .env.template .env
        echo      Created .env from template.
    ) else (
        echo      Warning: No .env or .env.template found. Using defaults.
    )
) else (
    echo      Found existing .env.
)

echo.
echo [2/2] Pulling latest updates and starting services...
echo      (This may take a few minutes if there are new updates)
echo.

docker-compose -f docker-compose-online.yml pull
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to pull images.
    echo         Please ensure you are logged in if the images are private:
    echo         "docker login ghcr.io"
    pause
    exit /b %ERRORLEVEL%
)

docker-compose -f docker-compose-online.yml up -d --remove-orphans
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Failed to start services.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ================================================================================
echo                         SIMOPS IS RUNNING
echo ================================================================================
echo.
echo   Web App:    http://localhost:8080
echo   Dashboard:  http://localhost:9181
echo.
echo   Auto-Update: ENABLED (Checking every hour)
echo.
echo   Press any key to close this window...
pause >nul
