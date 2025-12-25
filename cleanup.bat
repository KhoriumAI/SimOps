@echo off
REM SimOps Cleanup Script - Remove temp/test files from project root
REM ==================================================================

echo ============================================================
echo SimOps Cleanup Script
echo ============================================================
echo.
echo This will remove temporary mesh files, test outputs, and debug files
echo from the SimOps project root directory.
echo.

set /p confirm="Continue? (Y/N): "
if /I not "%confirm%"=="Y" (
    echo Cancelled.
    exit /b 0
)

echo.
echo Cleaning up...

REM Remove temp mesh files
echo   Removing temp_*.msh files...
del /Q "temp_*.msh" 2>nul
del /Q "temp_*.json" 2>nul
del /Q "temp_*.vtk" 2>nul

REM Remove debug files
echo   Removing debug_*.msh and debug_*.vtk files...
del /Q "debug_*.msh" 2>nul
del /Q "debug_*.vtk" 2>nul

REM Remove test meshes in root
echo   Removing test mesh files in root...
del /Q "cube.msh" 2>nul
del /Q "SingleTet.msh" 2>nul
del /Q "*_surface.msh" 2>nul

REM Remove test .frd/.inp in root
echo   Removing test .frd/.inp files in root...
del /Q "*.frd" 2>nul
del /Q "*.inp" 2>nul

REM Remove stray version file
del /Q "--version" 2>nul

REM Remove empty temp directories
echo   Removing temp directories...
rmdir /S /Q "temp_test" 2>nul
rmdir /S /Q "temp_test_viz" 2>nul
rmdir /S /Q "tmp_test" 2>nul
rmdir /S /Q "__pycache__" 2>nul
rmdir /S /Q ".pytest_cache" 2>nul

REM Remove test output directories (but keep main output)
echo   Removing test_output* directories...
rmdir /S /Q "test_output" 2>nul
rmdir /S /Q "test_output_aero" 2>nul
rmdir /S /Q "test_output_dispatch" 2>nul
rmdir /S /Q "test_output_report" 2>nul
rmdir /S /Q "output_loft_run" 2>nul

echo.
echo ============================================================
echo Cleanup complete!
echo ============================================================
pause
