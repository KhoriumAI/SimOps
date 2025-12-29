import os
import glob
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS_TO_CLEAN = [
    # Temp Meshes
    "apps/desktop/temp_mesh_*.msh",
    "apps/desktop/temp_*.brep",
    "apps/desktop/*.tmp",
    
    # Large Binaries in Root
    "*.cgns",
    "*.bdf",
    "*.msh",  # Be careful with this one, generally we don't want msh in root
    "*.vtk",
    "*_soup.stl",
    
    # Logs
    "jobs_log/*.log*",
    "*.log.processing",
    "*.log"
]

def cleanup_project():
    logger.info(f"Starting cleanup in: {PROJECT_ROOT}")
    count = 0
    size_freed = 0

    for pattern in PATTERNS_TO_CLEAN:
        full_pattern = os.path.join(PROJECT_ROOT, pattern)
        files = glob.glob(full_pattern)
        
        for file_path in files:
            try:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                logger.info(f"Deleted: {os.path.relpath(file_path, PROJECT_ROOT)} ({size/1024:.2f} KB)")
                count += 1
                size_freed += size
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")

    logger.info(f"Cleanup complete. Removed {count} files. Freed {size_freed/1024/1024:.2f} MB.")

if __name__ == "__main__":
    confirm = input("This will delete temporary files and logs. Type 'yes' to proceed: ")
    if confirm.lower() == "yes":
        cleanup_project()
    else:
        print("Cleanup cancelled.")
