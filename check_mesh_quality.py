import meshio
import numpy as np

def compute_tet_volume(p0, p1, p2, p3):
    """Compute volume of tetrahedron. Positive = valid, Negative = inverted."""
    # Volume = 1/6 * det([p1-p0, p2-p0, p3-p0])
    return np.linalg.det([p1-p0, p2-p0, p3-p0]) / 6.0

def check_quality(filename):
    print(f"Checking {filename}...")
    try:
        mesh = meshio.read(filename)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    points = mesh.points
    tets = None
    
    # Extract tets
    for c in mesh.cells:
        if c.type == 'tetra':
            tets = c.data
            break
        elif c.type == 'tetra10':
            tets = c.data[:, :4]
            break
    
    if tets is None:
        print("No tetrahedra found.")
        return

    volumes = []
    inverted_count = 0
    
    print(f"Analyzing {len(tets)} tetrahedra...")
    
    for i, tet in enumerate(tets):
        p0 = points[tet[0]]
        p1 = points[tet[1]]
        p2 = points[tet[2]]
        p3 = points[tet[3]]
        
        vol = compute_tet_volume(p0, p1, p2, p3)
        volumes.append(vol)
        
        if vol <= 0:
            inverted_count += 1
            if inverted_count <= 5: # Print first few
                print(f"  Tet {i} is INVERTED/ZERO! Vol: {vol}")
                
    min_vol = min(volumes)
    max_vol = max(volumes)
    avg_vol = sum(volumes) / len(volumes)
    
    print("-" * 30)
    print(f"Total Tets: {len(tets)}")
    print(f"Inverted Tets: {inverted_count}")
    print(f"Min Volume: {min_vol:.6e}")
    print(f"Max Volume: {max_vol:.6e}")
    print(f"Avg Volume: {avg_vol:.6e}")
    
    if inverted_count > 0:
        print("\nCRITICAL FAILURE: Mesh contains inverted elements!")
        print("Fluent will CRASH on these.")
    else:
        print("\nSUCCESS: No inverted elements found.")

if __name__ == "__main__":
    # Check our latest export
    check_quality("C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_retopo.msh")
