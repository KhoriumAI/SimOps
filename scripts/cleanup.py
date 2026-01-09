import os
import sys
import glob
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATTERNS_TO_CLEAN = [
    # Temp Meshes and leftovers in root
    "temp_mesh_*.msh",
    "bbox_temp_*.msh",
    "last_ditch_bbox_*.msh",
    "test_modal_result.msh",
    "temp_key.pem",
    
    # Modal/Log leftovers
    "modal_cube_*.txt",
    "modal_final_run*.txt",
    "modal_poc_output.txt",
    "modal_run_output.txt",
    "promotion_log_*.log",
    "staging_dist_info.txt",
    
    # Infrastructure configs in root
    "cf_dev_distconfig*.json",
    "cf_staging_config*.json",
    "cf_staging_distconfig*.json",
    "ssm_*.json",
    "ssm_params.json",
    "ssm_update.json",
    "ssm_discovery*.json",
    
    # Other leftovers
    "*.tmp",
    "*.log.processing",
    "*_soup.stl",
    "fused_*.stl",
    "dirty_*.stl",

    # Subfolder temp items
    "apps/desktop/temp_mesh_*.msh",
    "apps/desktop/temp_*.brep",
    "apps/desktop/*.tmp",
    "temp_meshes/*.msh",
    "temp_stls/*.stl",
]

DIRECTORIES_TO_CLEAN = [
    "temp_defeatured",
    "temp_defeatured_core",
    "temp_failures_v3",
    "temp_geometry",
    "temp_parts"
]

def cleanup_project():
    logger.info(f"Starting cleanup in: {PROJECT_ROOT}")
    count = 0
    size_freed = 0

    # Clean files
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

    # Clean directories
    for dir_name in DIRECTORIES_TO_CLEAN:
        dir_path = os.path.join(PROJECT_ROOT, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                # Calculate size before deleting
                for root, dirs, files in os.walk(dir_path):
                    for f in files:
                        size_freed += os.path.getsize(os.path.join(root, f))
                
                shutil.rmtree(dir_path)
                logger.info(f"Deleted directory: {dir_name}")
                count += 1
            except Exception as e:
                logger.error(f"Failed to delete directory {dir_path}: {e}")

    logger.info(f"Cleanup complete. Removed {count} items. Freed {size_freed/1024/1024:.2f} MB.")

if __name__ == "__main__":
    # If run with --auto, don't ask for confirmation
    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        cleanup_project()
    else:
        confirm = input("This will delete temporary files, directories, and logs. Type 'yes' to proceed: ")
        if confirm.lower() == "yes":
            cleanup_project()
        else:
            print("Cleanup cancelled.")
