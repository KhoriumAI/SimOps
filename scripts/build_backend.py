import PyInstaller.__main__
import os
import shutil
import platform
from pathlib import Path

# Configuration
BACKEND_DIR = Path('simops-backend')
SCRIPT_NAME = 'api_server.py'
OUTPUT_NAME = 'api_server' # .exe added automatically on Windows
TAURI_BIN_DIR = Path('desktop-app/src-tauri/binaries')
ICON_PATH = Path('desktop-app/src-tauri/icons/icon.ico') # Optional

def build_backend():
    print("🚀 Starting PyInstaller Build for SimOps Backend...")
    
    # Ensure clean slate
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
        
    # PyInstaller Args
    args = [
        str(BACKEND_DIR / SCRIPT_NAME),
        '--name=' + OUTPUT_NAME,
        '--onefile',
        '--clean',
        '--noconfirm',
        # Add hidden imports for dynamic libraries
        '--hidden-import=gmsh',
        '--hidden-import=engineio.async_drivers.threading', # Common flask-socketio issue
        '--hidden-import=flask_cors',
        # Include entire SimOps core/pipeline/templates logic if reused
        '--add-data=simops_pipeline.py:.',
        '--add-data=core;core',
        '--add-data=tools;tools',
        '--add-data=simops;simops',
        '--add-data=simops-backend/simops_output;simops_output', # Ensure folder exists? No, mkdir
    ]
    
    # Run
    PyInstaller.__main__.run(args)
    
    # Move to Tauri Sidecar Directory
    # Sidecar naming convention: name-target-triple(.exe)
    # e.g. api_server-x86_64-pc-windows-msvc.exe
    
    target_triple = "x86_64-pc-windows-msvc" # Assuming Windows 64-bit for this user
    src_exe = Path('dist') / (OUTPUT_NAME + '.exe')
    
    TAURI_BIN_DIR.mkdir(parents=True, exist_ok=True)
    dst_exe = TAURI_BIN_DIR / f"{OUTPUT_NAME}-{target_triple}.exe"
    
    if src_exe.exists():
        shutil.copy2(src_exe, dst_exe)
        print(f"✅ Success! Copied sidecar binary to: {dst_exe}")
        print(f"   PLEASE UPDATE tauri.conf.json: 'externalBin': ['binaries/{OUTPUT_NAME}']")
    else:
        print(f"❌ Error: Build failed, {src_exe} not found.")

if __name__ == "__main__":
    build_backend()
