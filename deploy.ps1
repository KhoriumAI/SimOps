# Frontend Deployment Script for Khorium MeshGen
# Usage: .\deploy.ps1

# Configuration
$S3_BUCKET = "muaz-mesh-web-dev"

Write-Host "--- Starting Frontend Deployment ---" -ForegroundColor Cyan

# 1. Frontend Build
Write-Host "[1/2] Building Frontend..." -ForegroundColor Yellow
$frontendPath = Join-Path $PSScriptRoot "web-frontend"
if (Test-Path $frontendPath) {
    Push-Location $frontendPath
    npm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed!"
        Pop-Location
        exit $LASTEXITCODE
    }
    Pop-Location
} else {
    Write-Error "web-frontend directory not found at $frontendPath!"
    exit 1
}

# 2. Sync to S3
Write-Host "[2/2] Syncing to S3 bucket: $S3_BUCKET..." -ForegroundColor Yellow
$distPath = Join-Path $frontendPath "dist"
if (Test-Path $distPath) {
    aws s3 sync $distPath s3://$S3_BUCKET --delete
    if ($LASTEXITCODE -ne 0) {
        Write-Error "S3 sync failed! Make sure AWS CLI is configured."
        exit $LASTEXITCODE
    }
} else {
    Write-Error "Build output (dist) not found!"
    exit 1
}

Write-Host "--- Frontend Deployment Successful! ---" -ForegroundColor Green
Write-Host "Note: Remember to manually restart the backend on EC2 after pulling latest changes." -ForegroundColor Cyan
