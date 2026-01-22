<#
.SYNOPSIS
    SimOps Offline Installer (PowerShell Version)
    
.DESCRIPTION
    Loads Docker images and starts SimOps services on an air-gapped machine.
    
.EXAMPLE
    .\install.ps1
    
.EXAMPLE
    .\install.ps1 -SkipBrowserOpen
#>

param(
    [switch]$SkipBrowserOpen,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Step { param($msg) Write-Host "`n===> $msg" -ForegroundColor Cyan }
function Write-OK { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Banner
Write-Host @"

 _____ _           ___              
/  ___(_)         / _ \             
\ `--. _ _ __ ___/ /_\ \_ __  _ __  
 `--. \ | '_ ` _ \  _  | '_ \| '_ \ 
/\__/ / | | | | | | | | | |_) | |_) |
\____/|_|_| |_| |_\_| |_/ .__/| .__/ 
                        | |   | |    
    OFFLINE INSTALLER   |_|   |_|    

"@ -ForegroundColor Magenta

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

try {
    # ==========================================================================
    # OPENFOAM PREREQUISITE CHECK
    # ==========================================================================
    Write-Host @"

================================================================================
                    PREREQUISITE: OpenFOAM Installation
================================================================================

SimOps requires OpenFOAM for CFD simulations. If you only need meshing and
CalculiX thermal analysis, you can skip OpenFOAM.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  REQUIRED VERSION: ESI OpenFOAM v2312 (from openfoam.com)                ║
  ║                                                                          ║
  ║  ⚠️  DO NOT install OpenFOAM Foundation (cfd.direct) or other versions.  ║
  ║      Schema formats differ between versions and WILL cause errors!      ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  INSTALLATION OPTIONS:
  
  [OPTION A - Docker (Recommended)] ───────────────────────────────────────────
    Docker Desktop handles OpenFOAM automatically via the worker container.
    No additional installation required if proceeding with this installer.
    
  [OPTION B - WSL2 (Windows)] ─────────────────────────────────────────────────
    1. Enable WSL2:  wsl --install
    2. Install Ubuntu from Microsoft Store  
    3. In Ubuntu terminal, run:
       curl -s https://dl.openfoam.com/add-debian-repo.sh | sudo bash
       sudo apt install openfoam2312-default
       echo 'source /usr/lib/openfoam/openfoam2312/etc/bashrc' >> ~/.bashrc
       
  [OPTION C - macOS (Docker)] ─────────────────────────────────────────────────
    Install Docker Desktop for Mac, then this installer handles the rest.

  Download Link: https://www.openfoam.com/download/install-windows

================================================================================
"@ -ForegroundColor Yellow

    # Check Docker
    Write-Step "Checking Docker..."

    $dockerCheck = docker version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not running. Please start Docker Desktop and try again."
    }
    Write-OK "Docker is running"

    # Check images folder
    $imagesPath = Join-Path $ScriptDir "images"
    if (-not (Test-Path $imagesPath)) {
        throw "Images folder not found at: $imagesPath"
    }

    # Load images
    Write-Step "Loading Docker Images..."
    $tarFiles = Get-ChildItem -Path $imagesPath -Filter "*.tar"
    
    if ($tarFiles.Count -eq 0) {
        throw "No .tar files found in images folder"
    }

    foreach ($tar in $tarFiles) {
        Write-Host "  Loading: $($tar.Name)..."
        docker load -i $tar.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to load: $($tar.Name)"
        }
    }
    Write-OK "All images loaded ($($tarFiles.Count) images)"

    # Create .env if needed
    $envFile = Join-Path $ScriptDir ".env"
    $envTemplate = Join-Path $ScriptDir ".env.template"
    
    if (-not (Test-Path $envFile) -and (Test-Path $envTemplate)) {
        Write-Host "  Creating .env from template..."
        Copy-Item $envTemplate $envFile
        Write-OK "Created .env configuration"
    }

    # Start services
    Write-Step "Starting SimOps Services..."
    docker-compose -f docker-compose-offline.yml up -d
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start services"
    }
    Write-OK "Services started"

    # Wait for health
    Write-Step "Waiting for services to be ready..."
    $maxAttempts = 30
    $attempt = 0
    $healthy = $false

    while ($attempt -lt $maxAttempts -and -not $healthy) {
        $attempt++
        Start-Sleep -Seconds 2
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000/api/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $healthy = $true
            }
        } catch {
            Write-Host "  Waiting for backend... (attempt $attempt/$maxAttempts)" -ForegroundColor DarkGray
        }
    }

    if ($healthy) {
        Write-OK "Backend is healthy!"
    } else {
        Write-Warn "Services may still be initializing. Check Docker Dashboard."
    }

    # Done
    Write-Host @"

================================================================================
                     INSTALLATION COMPLETE!
================================================================================

SimOps is now running! Access it at:

  Web Application:  http://localhost:8080
  API Server:       http://localhost:5000  
  Job Dashboard:    http://localhost:9181

Useful commands:
  View logs:    docker-compose -f docker-compose-offline.yml logs -f
  Stop:         docker-compose -f docker-compose-offline.yml down
  Restart:      docker-compose -f docker-compose-offline.yml restart

================================================================================
"@ -ForegroundColor Green

    # Open browser
    if (-not $SkipBrowserOpen) {
        Write-Host "Opening SimOps in your browser..."
        Start-Process "http://localhost:8080"
    }

} catch {
    Write-Err $_.Exception.Message
    Write-Host "`nInstallation failed. Please check the error above." -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}
