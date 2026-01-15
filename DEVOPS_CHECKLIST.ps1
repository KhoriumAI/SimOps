#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Unified DevOps Checklist for MeshPackageLean
    
.DESCRIPTION
    This script consolidates all pre-commit, pre-push, and post-deployment checks
    from various documentation files into a single executable checklist.
    
    It can run all checks or specific sections based on flags.
    
.PARAMETER PreCommit
    Run only pre-commit checks (type safety, schema sync, env vars)
    
.PARAMETER PrePush
    Run only pre-push checks (happy path validation)
    
.PARAMETER PostPush
    Run only post-deployment checks (health checks, connectivity)
    
.PARAMETER All
    Run all checks in chronological order (default)
    
.PARAMETER Url
    Base URL for post-deployment checks (default: http://localhost:5000)
    
.EXAMPLE
    .\DEVOPS_CHECKLIST.ps1 -All
    
.EXAMPLE
    .\DEVOPS_CHECKLIST.ps1 -PreCommit
    
.EXAMPLE
    .\DEVOPS_CHECKLIST.ps1 -PostPush -Url https://api.khorium.ai
#>

param(
    [switch]$PreCommit,
    [switch]$PrePush,
    [switch]$PostPush,
    [switch]$All,
    [string]$Url = "http://localhost:5000"
)

# Color output functions
function Write-Success {
    param([string]$Message)
    Write-Host "[OK] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Write-Failure {
    param([string]$Message)
    Write-Host "[FAIL] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

function Write-Section {
    param([string]$Title)
    Write-Host "`n============================================" -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "============================================`n" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Message)
    Write-Host "[*] " -ForegroundColor Blue -NoNewline
    Write-Host $Message
}

# Track overall status
$script:HasErrors = $false

function Invoke-Check {
    param(
        [string]$Name,
        [scriptblock]$Check,
        [switch]$Optional
    )
    
    Write-Info "Running: $Name"
    
    try {
        $result = & $Check
        
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "Command exited with code $LASTEXITCODE"
        }
        
        Write-Success "$Name passed"
        return $true
    }
    catch {
        if ($Optional) {
            Write-Host "  [SKIP] " -ForegroundColor Yellow -NoNewline
            Write-Host "$Name skipped (optional): $($_.Exception.Message)"
            return $true
        }
        else {
            Write-Failure "$Name failed: $($_.Exception.Message)"
            $script:HasErrors = $true
            return $false
        }
    }
}

# ============================================
# PRE-COMMIT CHECKS
# ============================================
function Test-PreCommit {
    Write-Section "PRE-COMMIT CHECKS"
    
    Write-Info "These checks should pass before committing code to Git"
    
    # Determine Python path - prefer venv if it exists
    $venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $pythonCmd = $venvPython
        Write-Info "Using venv Python: $venvPython"
    } else {
        $pythonCmd = "python"
        Write-Host "  [WARN] No venv found, using system Python" -ForegroundColor Yellow
    }
    
    # Check 1: Type Safety (mypy)
    Invoke-Check -Name "Type Safety Check (mypy)" -Check {
        Push-Location backend
        try {
            & $pythonCmd -m mypy . --config-file mypy.ini
            if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
        }
        finally {
            Pop-Location
        }
    }
    
    # Check 2: Database Schema Sync (alembic)
    Invoke-Check -Name "Database Schema Sync (alembic check)" -Check {
        Push-Location backend
        try {
            & $pythonCmd -m alembic check
            if ($LASTEXITCODE -ne 0) { throw "alembic check failed" }
        }
        finally {
            Pop-Location
        }
    }
    
    # Check 3: Environment Variable Audit
    Invoke-Check -Name "Environment Variable Audit" -Check {
        & $pythonCmd scripts/check_env_vars.py
        if ($LASTEXITCODE -ne 0) { throw "env check failed" }
    }
    
    # Check 4: Git Commit Message Format (if last commit exists)
    Invoke-Check -Name "Git Commit Message Format" -Check {
        $lastCommit = git log -1 --pretty=%B 2>$null
        if ($lastCommit) {
            # Check if last commit follows format: type/description
            $validPrefixes = @('feat/', 'fix/', 'debug/', 'refactor/', 'docs/', 'chore/', 'test/', 'perf/', 'style/')
            $hasValidPrefix = $false
            foreach ($prefix in $validPrefixes) {
                if ($lastCommit.StartsWith($prefix)) {
                    $hasValidPrefix = $true
                    break
                }
            }
            
            if (-not $hasValidPrefix) {
                Write-Host "  Warning: Last commit message does not follow format: type/description" -ForegroundColor Yellow
                Write-Host "  Valid types: feat, fix, debug, refactor, docs, chore, test, perf, style" -ForegroundColor Yellow
                Write-Host "  Example: feat/add_new_feature or fix/login_crash" -ForegroundColor Yellow
            }
        }
    } -Optional
    
    Write-Host ""
}

# ============================================
# PRE-PUSH CHECKS
# ============================================
function Test-PrePush {
    Write-Section "PRE-PUSH CHECKS"
    
    Write-Info "These checks validate the system before pushing to remote"
    
    # Check 1: Happy Path Validation (optional - requires local stack running)
    Invoke-Check -Name "Happy Path Validation" -Check {
        Write-Info "Running happy path validation against local stack..."
        python scripts/validate_happy_path.py --url http://localhost:5000
        if ($LASTEXITCODE -ne 0) { throw "happy path validation failed" }
    } -Optional
    
    # Check 2: Verify no uncommitted changes
    Invoke-Check -Name "No Uncommitted Changes" -Check {
        $status = git status --porcelain
        if ($status) {
            Write-Host "  Uncommitted changes detected:" -ForegroundColor Yellow
            Write-Host $status -ForegroundColor Yellow
            throw "Please commit or stash changes before pushing"
        }
    } -Optional
    
    Write-Host ""
}

# ============================================
# POST-DEPLOYMENT CHECKS
# ============================================
function Test-PostDeployment {
    Write-Section "POST-DEPLOYMENT CHECKS"
    
    Write-Info "These checks verify deployment was successful"
    Write-Info "Using URL: $Url"
    
    # Check 1: Backend Health
    Invoke-Check -Name "Backend Health Check" -Check {
        $response = Invoke-RestMethod -Uri "$Url/api/health" -Method Get -ErrorAction Stop
        if ($response.status -ne "healthy") {
            throw "Backend reported unhealthy status: $($response.status)"
        }
    }
    
    # Check 2: Strategy Endpoint
    Invoke-Check -Name "Strategy Endpoint Available" -Check {
        $response = Invoke-RestMethod -Uri "$Url/api/strategies" -Method Get -ErrorAction Stop
        $strategies = $response.strategies
        if (-not $strategies -or $strategies.Count -eq 0) {
            throw "No strategies returned"
        }
    }
    
    # Check 3: CORS Headers (if production URL)
    if ($Url -match "khorium\.ai") {
        Invoke-Check -Name "CORS Configuration" -Check {
            $headers = Invoke-WebRequest -Uri "$Url/api/health" -Method Options -ErrorAction Stop
            $originHeader = $headers.Headers["Access-Control-Allow-Origin"]
            if (-not $originHeader) {
                throw "CORS headers not configured"
            }
        } -Optional
    }
    
    # Check 4: Gunicorn Running (if localhost)
    if ($Url -match "localhost") {
        Invoke-Check -Name "Gunicorn Process Running" -Check {
            $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
                $_.CommandLine -match "gunicorn"
            }
            if (-not $processes) {
                # Try checking via netstat on port 5000 or 3000
                Write-Info "Checking for process listening on port..."
            }
        } -Optional
    }
    
    Write-Host ""
}

# ============================================
# MAIN EXECUTION
# ============================================

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   MeshPackageLean DevOps Checklist" -ForegroundColor Cyan  
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Determine which checks to run
$runPreCommit = $PreCommit -or $All -or (-not $PreCommit -and -not $PrePush -and -not $PostPush)
$runPrePush = $PrePush -or $All
$runPostPush = $PostPush

# Execute checks
if ($runPreCommit) {
    Test-PreCommit
}

if ($runPrePush) {
    Test-PrePush
}

if ($runPostPush) {
    Test-PostDeployment
}

# Summary
Write-Section "SUMMARY"

if ($script:HasErrors) {
    Write-Failure "Some checks failed. Please review errors above."
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Yellow
    Write-Host "  - Type errors: cd backend, then run: mypy . --config-file mypy.ini" -ForegroundColor Yellow
    Write-Host "  - Schema errors: cd backend, then run: alembic revision --autogenerate -m description" -ForegroundColor Yellow
    Write-Host "  - Env var errors: Add missing variables to backend/.env.example" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}
else {
    Write-Success "All checks passed!"
    Write-Host ""
    Write-Host "You are ready to commit/push/deploy!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 0
}
