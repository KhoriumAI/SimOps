import glob
import os
import pyvista as pv
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")
OUTPUT_STL = os.path.join(PROJECT_ROOT, "robust_soup.stl")

def main():
    print("Merging available STLs for visualization...")
    stls = glob.glob(os.path.join(TEMP_DIR, "*.stl"))
    print(f"Found {len(stls)} files.")
    
    if not stls:
        print("No files found.")
        sys.exit(1)
        
    blocks = pv.MultiBlock([pv.read(s) for s in stls])
    merged = blocks.combine().extract_surface()
    merged.save(OUTPUT_STL)
    print(f"Saved {OUTPUT_STL}")
    
    # Now run shell preserver logic? 
    # Or just tell user to run shell preserver?
    # I'll create a job to run shell_preserver on this file.
    
if __name__ == "__main__":
    main()
