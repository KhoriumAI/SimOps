import gmsh
import glob
import os
import sys

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")
OUTPUT_STL = os.path.join(PROJECT_ROOT, "robust_soup.stl")

def main():
    print("Merging STLs using GMSH...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    stls = sorted(glob.glob(os.path.join(TEMP_DIR, "vol_*.stl")))
    if not stls:
        print("No files found.")
        sys.exit(1)
        
    print(f"Merging {len(stls)} files...")
    
    # Merge all
    for p in stls:
        try:
            gmsh.merge(p)
        except Exception as e:
            print(f"Error merging {p}: {e}")
            
    # Save as single STL
    # GMSH saves visible entities.
    print(f"Writing to {OUTPUT_STL}...")
    gmsh.write(OUTPUT_STL)
    gmsh.finalize()
    print("Done.")

if __name__ == "__main__":
    main()
