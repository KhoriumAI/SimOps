#!/usr/bin/env pwsh
# SimOps Docker Worker Rebuild Script
# Kills existing workers, rebuilds containers, and shows logs

Write-Host "🔄 SimOps Worker Rebuild Script" -ForegroundColor Cyan
Write-Host "=" * 50

# Step 1: Force stop all containers
Write-Host "`n[1/3] Force stopping all containers..." -ForegroundColor Yellow
docker-compose kill
docker-compose down --remove-orphans
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Containers stopped" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to stop containers" -ForegroundColor Red
    exit 1
}

# Step 2: Rebuild Docker images
Write-Host "`n[2/3] Rebuilding Docker images..." -ForegroundColor Yellow
docker-compose build --no-cache worker
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Images rebuilt" -ForegroundColor Green
} else {
    Write-Host "  ✗ Build failed" -ForegroundColor Red
    exit 1
}

# Start containers in detached mode
Write-Host "`n  Starting containers..." -ForegroundColor Cyan
docker-compose up -d
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Containers started" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to start containers" -ForegroundColor Red
    exit 1
}

# Wait a moment for containers to initialize
Start-Sleep -Seconds 2

# Step 3: Show logs
Write-Host "`n[3/3] Showing Docker logs (Ctrl+C to exit)..." -ForegroundColor Yellow
Write-Host "=" * 50 -ForegroundColor Cyan
docker-compose logs -f
