import numpy as np

def compute_volume_5tet(hex_verts):
    # Current indices
    tet_splits = [
        [0, 1, 3, 5],
        [0, 3, 2, 6],
        [0, 5, 4, 7],
        [3, 6, 7, 5],
        [0, 3, 5, 7]
    ]
    vol = 0
    for tet in tet_splits:
        p = hex_verts[tet]
        v = abs(np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))) / 6.0
        vol += v
    return vol

def run_test():
    # Unit cube vertices (0-7 standard ordering)
    # Bottom: 0=(000), 1=(100), 2=(110), 3=(010)
    # Top:    4=(001), 5=(101), 6=(111), 7=(011)
    verts = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=float)
    
    vol = compute_volume_5tet(verts)
    print(f"Computed Volume: {vol}")
    print(f"Expected Volume: 1.0")
    
    if abs(vol - 1.0) < 1e-6:
        print("PASS")
    else:
        print("FAIL - Trying to find correct decomposition...")
        
        # known good 5-tet decomposition?
        # Ref: https://gitlab.kitware.com/vtk/vtk/-/blob/master/Common/DataModel/vtkTetra.cxx
        # But indices depend on vertex numbering.
        
        # Let's try to construct one.
        # Corners: 
        # C1: 0,1,3,4 -> Vol 1/6 (Confirmed)
        # C2: 1,2,0,5 ? No. 2 neighbors 1,3,6. 5 is above 1.
        # C2 (at 2): 2,1,3,6?
        # C3 (at 5): 5,1,4,6?
        # C4 (at 7): 7,3,4,6?
        # Center: 1,3,4,6 (If these are the inner faces of corners)
        
        candidates = [
            [0,1,3,4], # Corner at 0
            [2,1,3,6], # Corner at 2 (1,3,6 neighbors)
            [5,1,4,6], # Corner at 5 (1,4,6 neighbors)
            [7,3,4,6], # Corner at 7 (3,4,6 neighbors)
            [1,3,4,6]  # Center
        ]
        
        vol2 = 0
        for tet in candidates:
            p = verts[tet]
            v = abs(np.dot(p[1]-p[0], np.cross(p[2]-p[0], p[3]-p[0]))) / 6.0
            print(f"Tet {tet}: {v}")
            vol2 += v
            
        print(f"New Volume: {vol2}")

if __name__ == "__main__":
    run_test()
