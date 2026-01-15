# GPU Mesher Build Script for Windows
# Automatically builds _gpumesher.pyd for the current CUDA installation

param(
    [string]$PythonEnv = "mesh310",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "GPU Mesher Build Script" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan

# Navigate to gpu_experiments
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $scriptDir

# Verify CUDA installation
Write-Host "`n[1/6] Checking CUDA installation..." -ForegroundColor Yellow

# Try nvcc first
$nvccFound = $false
try {
    $nvccOutput = & nvcc --version 2>&1
    $nvccFound = $true
    Write-Host "CUDA found in PATH: $($nvccOutput | Select-String 'release')" -ForegroundColor Green
} catch {
    Write-Host "nvcc not in PATH. Searching for CUDA installation..." -ForegroundColor Yellow
    
    # Search for CUDA in common locations
    $cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaRoot) {
        $versions = Get-ChildItem $cudaRoot -Directory | Where-Object { $_.Name -like "v*" } | Sort-Object Name -Descending
        if ($versions.Count -gt 0) {
            $cudaPath = Join-Path $versions[0].FullName "bin"
            if (Test-Path $cudaPath) {
                Write-Host "Found CUDA at: $cudaPath" -ForegroundColor Green
                $env:PATH = "$cudaPath;$env:PATH"
                
                # Verify it works now
                try {
                    $nvccOutput = & nvcc --version 2>&1
                    $nvccFound = $true
                    Write-Host "CUDA added to PATH: $($nvccOutput | Select-String 'release')" -ForegroundColor Green
                } catch {
                    Write-Host "ERROR: Found CUDA but nvcc still fails. Check installation." -ForegroundColor Red
                    exit 1
                }
            }
        }
    }
}

if (-not $nvccFound) {
    Write-Host "ERROR: CUDA not found. Please install NVIDIA CUDA Toolkit 12.x" -ForegroundColor Red
    Write-Host "Download: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Red
    Write-Host "`nOr add CUDA bin to your PATH manually:" -ForegroundColor Yellow
    Write-Host '  $env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;$env:PATH"' -ForegroundColor Yellow
    exit 1
}

# Activate Python environment
Write-Host "`n[2/6] Activating Python environment: $PythonEnv..." -ForegroundColor Yellow
try {
    & conda activate $PythonEnv
    $pythonVersion = & python --version
    Write-Host "Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Could not activate conda environment. Using system Python." -ForegroundColor Yellow
}

# Clean build directory if requested
if ($Clean -and (Test-Path "build")) {
    Write-Host "`n[3/6] Cleaning old build..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force build
    Write-Host "Cleaned." -ForegroundColor Green
} else {
    Write-Host "`n[3/6] Skipping clean (use -Clean to rebuild from scratch)" -ForegroundColor Yellow
}

# Create build directory
Write-Host "`n[4/6] Setting up build directory..." -ForegroundColor Yellow
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}
Set-Location build

# Run CMake configure
Write-Host "`n[5/6] Configuring with CMake..." -ForegroundColor Yellow

# Check for Visual Studio
$vsPath = $null
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
)

foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        $vsPath = $path
        Write-Host "Found Visual Studio at: $vsPath" -ForegroundColor Green
        break
    }
}

if (-not $vsPath) {
    Write-Host "WARNING: Visual Studio not detected in standard locations." -ForegroundColor Yellow
    Write-Host "Attempting build anyway - CMake may find it through vcvarsall.bat" -ForegroundColor Yellow
}

try {
    # Run CMake with output capture
    $cmakeOutput = & cmake .. -G "Visual Studio 17 2022" -A x64 2>&1
    $cmakeExitCode = $LASTEXITCODE
    
    # Display output
    Write-Host $cmakeOutput
    
    if ($cmakeExitCode -ne 0) { 
        Write-Host "`nERROR: CMake configuration failed with exit code $cmakeExitCode" -ForegroundColor Red
        Write-Host "`nCommon fixes:" -ForegroundColor Yellow
        Write-Host "1. Install Visual Studio 2022 with 'Desktop development with C++'" -ForegroundColor Yellow
        Write-Host "   Download: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
        Write-Host "2. Or try Visual Studio 2019: cmake .. -G 'Visual Studio 16 2019' -A x64" -ForegroundColor Yellow
        Write-Host "3. If VS is installed, run from 'Developer Command Prompt for VS 2022'" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Configuration successful!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: CMake execution failed: $_" -ForegroundColor Red
    Write-Host "`nIs CMake installed? Check with: cmake --version" -ForegroundColor Yellow
    exit 1
}

# Build the Python module
Write-Host "`n[6/6] Building _gpumesher.pyd..." -ForegroundColor Yellow
try {
    & cmake --build . --config Release --target _gpumesher
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    Write-Host "Build successful!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Build failed. Check compiler errors above." -ForegroundColor Red
    exit 1
}

# Copy to core/
Write-Host "`n[FINAL] Copying _gpumesher.pyd to core/..." -ForegroundColor Yellow
$pydSource = "Release\_gpumesher.pyd"
$pydDest = "..\..\core\_gpumesher.pyd"

if (Test-Path $pydSource) {
    Copy-Item $pydSource $pydDest -Force
    Write-Host "SUCCESS: _gpumesher.pyd deployed to core/" -ForegroundColor Green
    Write-Host "`nYou can now use the 'HighSpeed GPU' strategy!" -ForegroundColor Cyan
} else {
    Write-Host "ERROR: Build output not found at $pydSource" -ForegroundColor Red
    exit 1
}

Write-Host "`n==================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
