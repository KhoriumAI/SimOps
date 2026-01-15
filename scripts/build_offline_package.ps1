<# 
.SYNOPSIS
    Build SimOps Offline Installer Package
    
.DESCRIPTION
    This script builds all Docker images for SimOps and exports them
    to portable .tar files for air-gap installation.
    
.NOTES
    Run this script on a machine with internet access.
    The output 'installer' folder can then be copied to offline machines.
    
.EXAMPLE
    .\build_offline_package.ps1
#>

param(
    [string]$OutputDir = "installer",
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-Step { param($msg) Write-Host "`n===> $msg" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Banner
Write-Host @"

 _____ _           ___              
/  ___(_)         / _ \             
\ `--. _ _ __ ___/ /_\ \_ __  _ __  
 `--. \ | '_ ` _ \  _  | '_ \| '_ \ 
/\__/ / | | | | | | | | | |_) | |_) |
\____/|_|_| |_| |_\_| |_/ .__/| .__/ 
                        | |   | |    
    OFFLINE INSTALLER   |_|   |_|    BUILD SCRIPT v$Version

"@ -ForegroundColor Magenta

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
    # Check Docker is running
    Write-Step "Checking Docker..."
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not running. Please start Docker Desktop."
    }
    Write-Success "Docker version: $dockerVersion"

    # Create output directory
    Write-Step "Creating installer directory..."
    $InstallerPath = Join-Path $ProjectRoot $OutputDir
    $ImagesPath = Join-Path $InstallerPath "images"
    
    if (Test-Path $InstallerPath) {
        Remove-Item -Recurse -Force $InstallerPath
    }
    New-Item -ItemType Directory -Force -Path $ImagesPath | Out-Null
    Write-Success "Created: $InstallerPath"

    # Build Backend Image
    Write-Step "Building Backend Image..."
    docker build -t simops-backend:offline -f backend/Dockerfile .
    if ($LASTEXITCODE -ne 0) { throw "Failed to build backend image" }
    Write-Success "Backend image built"

    # Build Frontend Image
    Write-Step "Building Frontend Image..."
    docker build -t simops-frontend:offline -f web-frontend/Dockerfile ./web-frontend
    if ($LASTEXITCODE -ne 0) { throw "Failed to build frontend image" }
    Write-Success "Frontend image built"

    # Build Worker Image
    Write-Step "Building Worker Image (this may take a while - installing OpenFOAM/CalculiX)..."
    docker build -t simops-worker:offline -f Dockerfile.worker .
    if ($LASTEXITCODE -ne 0) { throw "Failed to build worker image" }
    Write-Success "Worker image built"

    # Pull Redis image
    Write-Step "Pulling Redis Image..."
    docker pull redis:7-alpine
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull redis image" }
    docker tag redis:7-alpine simops-redis:offline
    Write-Success "Redis image ready"

    # Pull RQ Dashboard image
    Write-Step "Pulling RQ Dashboard Image..."
    docker pull cjlapao/rq-dashboard:latest
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull dashboard image" }
    docker tag cjlapao/rq-dashboard:latest simops-dashboard:offline
    Write-Success "Dashboard image ready"

    # Export images
    Write-Step "Exporting Docker Images to tar files..."
    
    $images = @(
        @{ Name = "simops-backend:offline"; File = "backend.tar" },
        @{ Name = "simops-frontend:offline"; File = "frontend.tar" },
        @{ Name = "simops-worker:offline"; File = "worker.tar" },
        @{ Name = "redis:7-alpine"; File = "redis.tar" },
        @{ Name = "cjlapao/rq-dashboard:latest"; File = "dashboard.tar" }
    )

    foreach ($img in $images) {
        $tarPath = Join-Path $ImagesPath $img.File
        Write-Host "  Saving $($img.Name) -> $($img.File)..."
        docker save -o $tarPath $img.Name
        if ($LASTEXITCODE -ne 0) { throw "Failed to save $($img.Name)" }
        
        $size = [math]::Round((Get-Item $tarPath).Length / 1MB, 1)
        Write-Host "    Saved: $size MB" -ForegroundColor DarkGray
    }
    Write-Success "All images exported"

    # Copy docker-compose and install script
    Write-Step "Copying installer files..."
    Copy-Item "docker-compose-offline.yml" -Destination $InstallerPath
    Copy-Item "scripts/install.bat" -Destination $InstallerPath -ErrorAction SilentlyContinue
    Copy-Item "scripts/install.ps1" -Destination $InstallerPath -ErrorAction SilentlyContinue
    
    # Create .env template
    @"
# SimOps Offline Configuration
# ============================
# Edit these values as needed before running install script

# Ports
FRONTEND_PORT=8080
BACKEND_PORT=5000
REDIS_PORT=6379
DASHBOARD_PORT=9181

# Workers
WORKER_REPLICAS=2
WORKER_CPU_LIMIT=2
WORKER_MEM_LIMIT=4G
WORKER_TIMEOUT=1800

# Security (CHANGE THIS!)
JWT_SECRET_KEY=simops-offline-$(Get-Random)-$(Get-Date -Format 'yyyyMMdd')

# Timezone
TZ=America/Los_Angeles

# Logging
LOG_LEVEL=INFO
"@ | Out-File -FilePath (Join-Path $InstallerPath ".env.template") -Encoding UTF8

    Write-Success "Installer files copied"

    # Calculate total size
    $totalSize = (Get-ChildItem -Path $InstallerPath -Recurse | Measure-Object -Property Length -Sum).Sum
    $totalSizeGB = [math]::Round($totalSize / 1GB, 2)

    # Summary
    Write-Host @"

================================================================================
                        BUILD COMPLETE!
================================================================================

Installer Package: $InstallerPath
Total Size: $totalSizeGB GB

Contents:
  - images/          Docker image tar files
  - docker-compose-offline.yml
  - .env.template
  - install.bat / install.ps1

NEXT STEPS:
  1. Copy the entire '$OutputDir' folder to a USB drive or network share
  2. On the target offline machine:
     a. Install Docker Desktop
     b. Copy the installer folder
     c. Run install.bat (or install.ps1 for PowerShell)
  3. Access SimOps at http://localhost:8080

================================================================================
"@ -ForegroundColor Green

} catch {
    Write-Error $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}
