import glob
import os
import pyvista as pv
import numpy as np
import sys

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")
SOUP_FILE = os.path.join(PROJECT_ROOT, "robust_soup.stl")

def log(msg):
    print(msg)

def check_file(path):
    try:
        mesh = pv.read(path)
        bounds = mesh.bounds
        n_pts = mesh.n_points
        n_cells = mesh.n_cells
        # Check manifold (expensive? No, just edges)
        # PyVista/VTK filter 'FeatureEdges' with BoundaryEdges=True
        edges = mesh.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        n_open = edges.n_cells
        
        return {
            "name": os.path.basename(path),
            "bounds": bounds,
            "n_pts": n_pts,
            "n_cells": n_cells,
            "n_open_edges": n_open,
            "is_watertight": n_open == 0
        }
    except Exception as e:
        return {"name": os.path.basename(path), "error": str(e)}

def main():
    log("--- INSPECTING ROBUST SOUP ---")
    if os.path.exists(SOUP_FILE):
        info = check_file(SOUP_FILE)
        log(f"Soup: {info['name']}")
        log(f"  Points: {info.get('n_pts')}, Cells: {info.get('n_cells')}")
        log(f"  Bounds: {info.get('bounds')}")
        log(f"  Open Edges: {info.get('n_open_edges')} (Watertight: {info.get('is_watertight')})")
        
        # Check Connectivity of Soup
        mesh = pv.read(SOUP_FILE)
        conn = mesh.connectivity(extraction_mode='all')
        n_regions = conn.n_arrays  # Wait, connectivity filter usually separates regions.
        # extraction_mode='all' creates a RegionId scaler?
        # Let's count region IDs.
        regions = conn["RegionId"]
        n_distinct = len(np.unique(regions))
        log(f"  Distinct Regions (Connectivity): {n_distinct}")
        
    else:
        log("Soup file not found.")

    log("\n--- SCANNING ALL STLS FOR OPEN EDGES ---")
    stls = sorted(glob.glob(os.path.join(TEMP_DIR, "vol_*.stl")))
    bad_files = []
    
    for p in stls:
        # Fast check?
        # Reading 150 files might take 30s. Acceptable.
        try:
            m = pv.read(p)
            n_open = m.n_open_edges
            if n_open > 0:
                log(f"  [!] {os.path.basename(p)}: {n_open} open edges")
                bad_files.append(os.path.basename(p))
        except Exception as e:
            log(f"  [Error] {os.path.basename(p)}: {e}")
            
    log(f"\nFound {len(bad_files)} non-watertight volumes.")
    log(f"Bad Volumes: {bad_files}")

if __name__ == "__main__":
    main()
