#!/usr/bin/env python
"""
SimOps Test Cleanup Script
===========================
Removes temporary files generated during testing and verification.
Safe to run at any time - only removes known temporary test files.
"""

import os
import shutil
from pathlib import Path
from typing import List

def cleanup_test_files():
    """Remove all temporary test files from SI physics verification and other tests."""
    
    root = Path(__file__).parent
    
    # Patterns for temporary test files (in root directory only)
    temp_patterns = [
        # SI Physics verification tests
        "si_*.inp", "si_*.frd", "si_*.dat", "si_*.sta", "si_*.cvg",
        "benchmark.inp", "benchmark.frd", "benchmark.dat", "benchmark.sta", "benchmark.cvg",
        "repro_*.inp", "repro_*.frd", "repro_*.dat", "repro_*.sta", "repro_*.cvg",
        
        # Debug directories (safe to remove)
        "debug_structural_cube",
        "structural_test_env",
    ]
    
    removed_files = []
    removed_dirs = []
    
    # Clean root directory temp files
    for pattern in temp_patterns:
        if "*" in pattern:
            # Glob pattern
            for file_path in root.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_files.append(str(file_path.relative_to(root)))
                    except Exception as e:
                        print(f"Warning: Could not remove {file_path}: {e}")
        else:
            # Directory or exact file name
            target = root / pattern
            if target.is_dir():
                try:
                    shutil.rmtree(target)
                    removed_dirs.append(str(target.relative_to(root)))
                except Exception as e:
                    print(f"Warning: Could not remove directory {target}: {e}")
            elif target.is_file():
                try:
                    target.unlink()
                    removed_files.append(str(target.relative_to(root)))
                except Exception as e:
                    print(f"Warning: Could not remove {target}: {e}")
    
    # Report
    if removed_files or removed_dirs:
        print(f"Cleaned up {len(removed_files)} temporary files")
        if removed_files and len(removed_files) <= 20:
            for f in removed_files:
                print(f"  - {f}")
        if removed_dirs:
            print(f"Removed {len(removed_dirs)} temporary directories:")
            for d in removed_dirs:
                print(f"  - {d}/")
    else:
        print("No temporary test files found to clean.")
    
    return len(removed_files) + len(removed_dirs)

if __name__ == "__main__":
    print("="*60)
    print("SimOps Test Cleanup")
    print("="*60)
    count = cleanup_test_files()
    print("="*60)
    print(f"Cleanup complete. Removed {count} items.")
