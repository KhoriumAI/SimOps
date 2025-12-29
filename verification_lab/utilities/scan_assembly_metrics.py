import gmsh
import os
import math
import csv

INPUT_STEP = "cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
OUTPUT_CSV = "assembly_metrics.csv"

def get_diagonal(bbox):
    dx = bbox[3] - bbox[0]
    dy = bbox[4] - bbox[1]
    dz = bbox[5] - bbox[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def scan():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # Try current directory first, then absolute
    if not os.path.exists(INPUT_STEP):
        # Allow user to change this easily
        print(f"Error: Could not find {INPUT_STEP}")
        gmsh.finalize()
        return

    print(f"Loading {INPUT_STEP}...")
    gmsh.open(INPUT_STEP)
    
    volumes = gmsh.model.getEntities(3)
    
    print(f"Scanning {len(volumes)} volumes...")
    
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # HEADER
        writer.writerow(['VolID', 'Diagonal_mm', 'Density_FeatsPerMM', 'NumCurves', 'NumSurfaces', 'BoundingBoxVolume', 'Heuristic_Decision'])
        
        for dim, tag in volumes:
            # 1. Bounding Box
            bbox = gmsh.model.getBoundingBox(dim, tag)
            diag = get_diagonal(bbox)
            dx = bbox[3] - bbox[0]
            dy = bbox[4] - bbox[1]
            dz = bbox[5] - bbox[2]
            bb_vol = dx * dy * dz
            
            # 2. Topology Counts
            surfaces = gmsh.model.getBoundary([(dim, tag)], recursive=False)
            curves = gmsh.model.getBoundary([(dim, tag)], recursive=True)
            n_surf = len(surfaces)
            n_curv = len(curves)
            
            # 3. Density Score
            density = (n_curv + n_surf) / (diag + 1e-9)
            
            # 4. Current Heuristic Test
            decision = "UNKNOWN"
            if diag > 50.0:
                decision = "KEEP (Large)"
            elif density > 12.0:
                decision = "BOX (Screw?)"
            else:
                decision = "KEEP (Detail)"

            writer.writerow([tag, f"{diag:.2f}", f"{density:.2f}", n_curv, n_surf, f"{bb_vol:.2f}", decision])
            
    gmsh.finalize()
    print(f"\nDone! Open '{OUTPUT_CSV}' in Excel/Sheets to find your threshold.")

if __name__ == "__main__":
    scan()
