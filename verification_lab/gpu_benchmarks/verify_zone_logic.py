
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from core.zone_manager import FluentZoneManager

def create_dummy_box_mesh():
    """
    Create a simple 2-element quad mesh (2 squares sharing an edge)
    Points:
    0: 0,0,0
    1: 1,0,0
    2: 1,1,0
    3: 0,1,0
    4: 2,0,0
    5: 2,1,0
    
    Face 1: [0, 1, 2, 3] (Normal +Z)
    Face 2: [1, 4, 5, 2] (Normal +Z) -> Planar neighbor
    """
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], # 6
        [1.0, 0.0, 1.0], # 7
        [1.0, 1.0, 1.0], # 8
        [0.0, 1.0, 1.0], # 9
    ])
    
    elements = [
        {"id": 101, "type": "quadrilateral", "nodes": [0, 1, 2, 3]}, # Left square (Z=0)
        {"id": 102, "type": "quadrilateral", "nodes": [1, 4, 5, 2]}, # Right square (Z=0)
        {"id": 103, "type": "quadrilateral", "nodes": [6, 7, 8, 9]}, # Top square (Z=1)
    ]
    
    return points, elements

def test_manager():
    print("="*60)
    print("TESTING FLUENT ZONE MANAGER")
    print("="*60)
    
    manager = FluentZoneManager()
    
    # 1. Load Mesh
    points, elements = create_dummy_box_mesh()
    manager.set_mesh_data(points, elements)
    
    # Check adjacency
    print("\n[Test 1] Adjacency Check")
    adj_101 = manager.face_adjacency.get(101, [])
    print(f"Neighbors of 101 (expected [102]): {adj_101}")
    if 102 in adj_101:
        print("PASS")
    else:
        print("FAIL: 101 should be adjacent to 102")

    # 2. Select Single
    print("\n[Test 2] Single Selection")
    manager.select_face(101)
    print(f"Selected: {manager.selected_faces} (Expected {{101}})")
    assert len(manager.selected_faces) == 1 and 101 in manager.selected_faces
    
    # 3. Create Zone
    print("\n[Test 3] Create Zone")
    manager.create_zone("bottom_wall", "wall")
    print(f"Zones: {manager.zone_registry.keys()}")
    print(f"Face 101 Zone: {manager.face_to_zone.get(101)}")
    assert "bottom_wall" in manager.zone_registry
    assert manager.face_to_zone[101] == "bottom_wall"
    assert len(manager.selected_faces) == 0 # Should clear
    
    # 4. Spill Select
    print("\n[Test 4] Spill Select")
    # 101 and 102 are coplanar (Z=0). 103 is parallel but not connected.
    manager.spill_select(101)
    print(f"Selected after spill 101: {manager.selected_faces}")
    # Should get 101 and 102. 103 is not connected.
    assert 101 in manager.selected_faces
    assert 102 in manager.selected_faces
    assert 103 not in manager.selected_faces
    
    # 5. Overwrite Zone
    print("\n[Test 5] Overwrite Zone")
    # currently 101 is in 'bottom_wall'. 
    # Select 101 and 102 and make new zone
    manager.create_zone("main_floor", "wall")
    
    z_bottom = manager.zone_registry["bottom_wall"]
    z_floor = manager.zone_registry["main_floor"]
    
    print(f"bottom_wall faces: {z_bottom.face_ids}")
    print(f"main_floor faces: {z_floor.face_ids}")
    
    # 101 should move from bottom to floor
    assert 101 not in z_bottom.face_ids
    assert 101 in z_floor.face_ids
    
    print("\nALL TESTS PASSED")

if __name__ == "__main__":
    test_manager()
