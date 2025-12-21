import os
import shutil
from pathlib import Path
import sys

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from simops_pipeline import cleanup_output_dir

def test_cleanup():
    test_dir = Path("tests/temp_cleanup_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    
    # Files to KEEP
    keep_files = ["result.png", "report.pdf", "data.json", "mesh.msh", "model.vtk", "run.log", "solver.inp"]
    for f in keep_files:
        (test_dir / f).write_text("should keep")
        
    # Files to DELETE
    delete_files = ["job.frd", "job.spool", "job.cvg", "job.rout", "job.equ", "job.f", "job.dat", "temp.12d", "run.sta"]
    for f in delete_files:
        (test_dir / f).write_text("should delete")
        
    # Dirs to DELETE
    delete_dirs = ["0.1", "1.0", "processor0", "processor1", "polyMesh", "postProcessing", "VTK"]
    for d in delete_dirs:
        (test_dir / d).mkdir()
        (test_dir / d / "temp.txt").write_text("bin")
        
    # Dirs to KEEP
    keep_dirs = ["0", "images", "logs"]
    for d in keep_dirs:
        (test_dir / d).mkdir()
        (test_dir / d / "essential.txt").write_text("keep")

    print(f"Running cleanup on {test_dir}...")
    cleanup_output_dir(str(test_dir))
    
    # Verify
    errors = []
    for f in keep_files:
        if not (test_dir / f).exists():
            errors.append(f"Expected to keep file: {f}")
            
    for f in delete_files:
        if (test_dir / f).exists():
            errors.append(f"Expected to delete file: {f}")
            
    for d in delete_dirs:
        if (test_dir / d).exists():
            errors.append(f"Expected to delete dir: {d}")
            
    for d in keep_dirs:
        if not (test_dir / d).exists():
            errors.append(f"Expected to keep dir: {d}")

    if errors:
        print("[FAIL] Cleanup verification failed:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("[PASS] Cleanup verification successful!")
        return True

if __name__ == "__main__":
    success = test_cleanup()
    if success:
        # Cleanup test dir
        shutil.rmtree("tests/temp_cleanup_test")
        sys.exit(0)
    else:
        sys.exit(1)
