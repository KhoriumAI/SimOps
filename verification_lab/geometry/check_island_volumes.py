import pyvista as pv
import numpy as np

def check_islands(path):
    print(f"Loading {path}...")
    mesh = pv.read(path)
    print("Computing connectivity...")
    conn = mesh.connectivity(extraction_mode='all')
    ids = conn.cell_data["RegionId"]
    n = len(np.unique(ids))
    print(f"Found {n} islands.")
    
    total_vol = 0
    for i in range(n):
        island = conn.threshold([i, i], scalars="RegionId")
        vol = island.volume
        area = island.area
        if i < 20 or vol > 10000:
            print(f"  Island {i}: Vol={vol:.2f}, Area={area:.2f}")
        total_vol += vol
    
    print(f"Total calculated volume: {total_vol:.2f}")

if __name__ == "__main__":
    check_islands("fixed_soup.stl")
