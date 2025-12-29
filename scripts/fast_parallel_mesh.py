"""
Fast Parallel Mesh Generator with Watchdog Timeout

STRATEGY: "Persistent Workers with Timeouts"
- Each worker loads CAD, isolates ONE volume, meshes, exports
- Strict 30s timeout per volume - hangs get killed and logged
- Failed geometry dumped to failures/ for manual inspection

PREREQUISITES: pip install func_timeout
"""

import gmsh
import multiprocessing
import time
import os
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from func_timeout import func_timeout, FunctionTimedOut
except ImportError:
    print("ERROR: func_timeout not installed!")
    print("Run: pip install func_timeout")
    sys.exit(1)

# CONFIGURATION
STEP_FILE = str(PROJECT_ROOT / "cad_files" / "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
OUTPUT_DIR = str(PROJECT_ROOT / "temp_stls" / "fast_output")
FAIL_DIR = str(PROJECT_ROOT / "temp_stls" / "failures")
FINAL_MERGED = str(PROJECT_ROOT / "generated_meshes" / "heater_board_MERGED.msh")
TIMEOUT_SEC = 30  # Max time allowed per volume before killing it


def robust_worker(vol_tag):
    """
    Worker that loads CAD, isolates one volume, meshes, and exports.
    Wrapped in a timeout watchdog.
    """
    output_path = os.path.join(OUTPUT_DIR, f"vol_{vol_tag}.msh")
    fail_path = os.path.join(FAIL_DIR, f"fail_vol_{vol_tag}.brep")
    
    # Skip if already done
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return (vol_tag, "SKIPPED", 0)

    start_t = time.time()
    
    try:
        def _do_work():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("General.Verbosity", 0)
            
            # Fast Import
            gmsh.model.occ.importShapes(STEP_FILE)
            gmsh.model.occ.synchronize()
            
            # Delete everything EXCEPT target
            all_vols = gmsh.model.getEntities(dim=3)
            to_delete = [v for v in all_vols if v[1] != vol_tag]
            gmsh.model.occ.remove(to_delete, recursive=True)
            gmsh.model.occ.synchronize()
            
            # Mesh parameters - balanced for speed/quality
            gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay (Robust)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
            
            # Generate 3D mesh
            gmsh.model.mesh.generate(3)
            
            # Check Quality
            nodes = gmsh.model.mesh.getNodes()[0]
            if nodes.size == 0:
                raise Exception("No nodes generated")
            
            # CRITICAL: Assign unique physical group for this volume
            vols = gmsh.model.getEntities(3)
            if vols:
                vol_tag_new = vols[0][1]
                pg = gmsh.model.addPhysicalGroup(3, [vol_tag_new])
                gmsh.model.setPhysicalName(3, pg, f"Volume_{vol_tag}")
                
            gmsh.write(output_path)
            gmsh.finalize()

        # THE WATCHDOG - kills hangs at TIMEOUT_SEC
        func_timeout(TIMEOUT_SEC, _do_work)
        return (vol_tag, "SUCCESS", time.time() - start_t)

    except FunctionTimedOut:
        # DUMP THE BODY FOR INSPECTION
        try:
            if gmsh.isInitialized():
                # Try to save the isolated bad geometry
                try:
                    gmsh.write(fail_path)
                except:
                    pass
                gmsh.finalize()
        except: 
            pass
        return (vol_tag, "TIMEOUT", TIMEOUT_SEC)
        
    except Exception as e:
        try:
            if gmsh.isInitialized(): 
                gmsh.finalize()
        except: 
            pass
        return (vol_tag, f"ERROR: {str(e)[:100]}", time.time() - start_t)


def merge_all_meshes():
    """Merge all successful meshes into one file, preserving physical groups."""
    print("\n" + "=" * 50)
    print("MERGING ALL VOLUMES...")
    print("=" * 50)
    
    gmsh.initialize()
    gmsh.model.add("Merged_Assembly")
    
    msh_files = sorted(Path(OUTPUT_DIR).glob("vol_*.msh"))
    print(f"Found {len(msh_files)} mesh files to merge")
    
    for msh_file in msh_files:
        try:
            gmsh.merge(str(msh_file))
        except Exception as e:
            print(f"  [!] Failed to merge {msh_file.name}: {e}")
    
    # Verify physical groups
    phys_groups = gmsh.model.getPhysicalGroups(3)
    print(f"[OK] Merged mesh contains {len(phys_groups)} physical volume groups")
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(FINAL_MERGED), exist_ok=True)
    
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(FINAL_MERGED)
    gmsh.finalize()
    
    print(f"[OK] Final mesh saved: {FINAL_MERGED}")


def main():
    print(f"=" * 60)
    print(f"FAST PARALLEL MESH (Timeout: {TIMEOUT_SEC}s per volume)")
    print(f"=" * 60)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAIL_DIR, exist_ok=True)

    # 1. Scan for volume IDs (Quick Load)
    print(f"\nLoading: {os.path.basename(STEP_FILE)}")
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.occ.importShapes(STEP_FILE)
    gmsh.model.occ.synchronize()
    all_vols = gmsh.model.getEntities(dim=3)
    all_tags = [tag for dim, tag in all_vols]
    gmsh.finalize()
    
    print(f"Found {len(all_tags)} volumes. Starting workers...\n")
    
    # 2. Parallel Pool
    # Use cores - 2 to keep system responsive
    cpu_count = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {cpu_count} parallel workers")
    print("-" * 60)
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        # Map tasks with unordered results for faster completion visibility
        results = pool.imap_unordered(robust_worker, all_tags)
        
        # Monitor Progress
        success = 0
        skipped = 0
        fails = 0
        timeouts = 0
        total = len(all_tags)
        
        for i, res in enumerate(results):
            tag, status, duration = res
            progress = f"[{i+1:3d}/{total}]"
            
            if status == "SUCCESS":
                success += 1
                print(f"{progress} Vol {tag:3d}: [OK] {duration:.1f}s")
            elif status == "SKIPPED":
                skipped += 1
                # Silent for cached
            elif status == "TIMEOUT":
                timeouts += 1
                print(f"{progress} Vol {tag:3d}: [TIMEOUT] ({duration:.0f}s) - saved to failures/")
            else:
                fails += 1
                print(f"{progress} Vol {tag:3d}: [FAIL] {status}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Success:  {success}")
    print(f"  Skipped:  {skipped}")
    print(f"  Timeouts: {timeouts}")
    print(f"  Errors:   {fails}")
    print(f"  Total:    {total}")
    
    if timeouts > 0 or fails > 0:
        print(f"\nWARNING: Check {FAIL_DIR} for problematic geometry files.")
        print("         Drag+drop into Gmsh GUI to inspect the broken parts.")
    
    # 3. Merge all successful meshes
    if success + skipped > 0:
        merge_all_meshes()
    else:
        print("\n[!] No successful meshes to merge!")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
