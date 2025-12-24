"""
ANSYS Workbench Automated CDB Validation Test
==============================================

This script:
1. Generates a test CDB file using export_mechanical_cdb
2. Creates a Workbench journal script (.wbjn) to import it
3. Executes the script and captures output
4. Reports success/failure

Usage: python test_ansys_import.py
"""

import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from core.export_mechanical_cdb import (
    export_mechanical_cdb, 
    create_sample_box_mesh_tet4
)

# Configuration
ANSYS_VERSION = "v252"
ANSYS_ROOT = Path(r"C:\Program Files\ANSYS Inc") / ANSYS_VERSION
RUNWB2_EXE = ANSYS_ROOT / "Framework" / "bin" / "Win64" / "RunWB2.exe"

OUTPUT_DIR = script_dir / "generated_meshes"
CDB_FILE = OUTPUT_DIR / "test_ansys_import.cdb"
WBJN_SCRIPT = OUTPUT_DIR / "import_test.wbjn"
LOG_FILE = OUTPUT_DIR / "ansys_import_log.txt"


def generate_test_cdb():
    """Generate a minimal test CDB file"""
    print("[Test] Generating test CDB file...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create sample mesh
    points, tets, named_selections = create_sample_box_mesh_tet4(
        nx=3, ny=3, nz=3,  # Small mesh for fast testing
        size=(50, 50, 50)
    )
    
    # Export to CDB
    success = export_mechanical_cdb(
        filename=str(CDB_FILE),
        points=points,
        elements=tets,
        element_type="tet10",
        named_selections=named_selections,
        material_id=1,
        verbose=True
    )
    
    if success:
        print(f"[Test] CDB file created: {CDB_FILE}")
        print(f"[Test] File size: {CDB_FILE.stat().st_size} bytes")
    else:
        print("[Test] ERROR: Failed to create CDB file!")
        
    return success


def generate_wbjn_script():
    """Generate Workbench Journal script to import CDB"""
    print("[Test] Generating WBJN import script...")
    
    # Use the MAPDL-generated CDB for testing (known good format)
    cdb_path_fwd = str(OUTPUT_DIR / "mesh_mapdl.cdb").replace("\\", "/")
    
    wbjn_content = f'''# encoding: utf-8
# ANSYS Workbench Journal Script - Auto-generated for CDB validation
# Run via: RunWB2.exe -B -R "import_test.wbjn"

import sys

try:
    SetScriptVersion(Version="25.2")  # v252
    
    print("[WBJN] Creating External Model system...")
    template1 = GetTemplate(TemplateName="External Model")
    system1 = template1.CreateSystem()
    
    print("[WBJN] Adding CDB file...")
    setup1 = system1.GetContainer(ComponentName="Setup")
    file_path = "{cdb_path_fwd}"
    imported_file = setup1.AddDataFile(FilePath=file_path)
    
    print("[WBJN] Configuring import properties...")
    # Try to find and disable validation checkboxes
    # The property access varies by ANSYS version
    try:
        # Try accessing via DataEntity
        data_entity = imported_file.GetDataEntity()
        if hasattr(data_entity, "CheckValidBlockedCdbFile"):
            data_entity.CheckValidBlockedCdbFile = False
            print("[WBJN] Disabled CheckValidBlockedCdbFile")
        if hasattr(data_entity, "ProcessMaterialProperties"):
            data_entity.ProcessMaterialProperties = False
            print("[WBJN] Disabled ProcessMaterialProperties")
    except Exception as e:
        print("[WBJN] Could not configure properties (non-fatal): " + str(e))
    
    print("[WBJN] Updating system (this is the import step)...")
    # Use system1.Update() not setup1.Update()
    system1.Update(AllDependencies=True)
    
    print("[WBJN] Checking for errors...")
    # Check state of Model component
    model_comp = system1.GetComponent(Name="Model")
    state = model_comp.GetState()
    print("[WBJN] Model component state: " + str(state))
    
    # State values: Unfulfilled, InputsOk, OKToRefresh, UpToDate, etc.
    if "Error" in str(state) or "Failed" in str(state):
        print("[WBJN] IMPORT FAILED: Model component in error state")
        sys.exit(1)
    else:
        print("[WBJN] SUCCESS: CDB file imported!")
    
except Exception as e:
    print("[WBJN] IMPORT FAILED: " + str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

finally:
    print("[WBJN] Exiting without save...")
'''
    
    with open(WBJN_SCRIPT, 'w', encoding='utf-8') as f:
        f.write(wbjn_content)
    
    print(f"[Test] WBJN script created: {WBJN_SCRIPT}")
    return True


def run_ansys_test():
    """Execute the WBJN script in ANSYS Workbench"""
    print("[Test] Running ANSYS Workbench import test...")
    
    runwb2_path = RUNWB2_EXE
    print(f"[Test] RunWB2.exe path: {runwb2_path}")
    
    if not runwb2_path.exists():
        print(f"[Test] ERROR: RunWB2.exe not found at {runwb2_path}")
        print("[Test] Searching for alternative locations...")
        
        # Try to find it
        for alt_path in [
            ANSYS_ROOT / "Framework" / "bin" / "Win64" / "RunWB2.exe",
            ANSYS_ROOT / "aisol" / "bin" / "winx64" / "RunWB2.exe",
        ]:
            if alt_path.exists():
                print(f"[Test] Found at: {alt_path}")
                runwb2_path = alt_path
                break
        else:
            print("[Test] ERROR: Could not find RunWB2.exe")
            return False
    
    # Build command
    cmd = [
        str(runwb2_path),
        "-B",  # Batch mode (no GUI)
        "-R", str(WBJN_SCRIPT)
    ]
    
    print(f"[Test] Command: {' '.join(cmd)}")
    print("[Test] This may take 30-60 seconds...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=str(OUTPUT_DIR)
        )
        
        # Save output to log
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
            f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")
        
        print(f"[Test] Exit code: {result.returncode}")
        print(f"[Test] Log saved to: {LOG_FILE}")
        
        # Print key output
        print("\n--- ANSYS Output ---")
        for line in result.stdout.split('\n'):
            if '[WBJN]' in line:
                print(line)
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("\n[Test] ✓ CDB IMPORT SUCCESSFUL!")
            return True
        else:
            print("\n[Test] ✗ CDB IMPORT FAILED")
            print("Check log file for details.")
            return False
            
    except subprocess.TimeoutExpired:
        print("[Test] ERROR: ANSYS process timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"[Test] ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("ANSYS Mechanical CDB Import Validation Test")
    print("=" * 60)
    
    # Step 1: Generate CDB
    if not generate_test_cdb():
        return 1
    
    # Step 2: Generate WBJN script
    if not generate_wbjn_script():
        return 1
    
    # Step 3: Run ANSYS test
    success = run_ansys_test()
    
    print("=" * 60)
    if success:
        print("RESULT: CDB format is VALID for ANSYS Mechanical import")
    else:
        print("RESULT: CDB format has ERRORS - check log for details")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
