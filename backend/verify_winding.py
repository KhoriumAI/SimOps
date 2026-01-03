import numpy as np

def compute_normal(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return np.cross(v1, v2)

def verify_tet_faces(faces, nodes, centroid):
    print(f"Verifying {len(faces)} faces...")
    correct_count = 0
    for i, face in enumerate(faces):
        p1 = nodes[face[0]]
        p2 = nodes[face[1]]
        p3 = nodes[face[2]]
        
        normal = compute_normal(p1, p2, p3)
        # Vector from centroid to face center
        face_center = (np.array(p1) + np.array(p2) + np.array(p3)) / 3.0
        outward_vec = face_center - centroid
        
        dot = np.dot(normal, outward_vec)
        
        status = "CORRECT" if dot > 0 else "INVERTED"
        if dot > 0: correct_count += 1
        
        print(f"Face {i}: {face} -> Dot: {dot:.4f} [{status}]")
        
    return correct_count == len(faces)

def main():
    # Define a simple regular tetrahedron centered at origin
    nodes = {
        0: [1, 1, 1],
        1: [1, -1, -1],
        2: [-1, 1, -1],
        3: [-1, -1, 1]
    }
    centroid = np.array([0, 0, 0])
    
    print("--- Testing Applied Fix Logic ---")
    # The logic implemented in api_server.py:
    # ((n1, n2, n3), 
    # ((n1, n4, n2), 
    # ((n1, n3, n4), 
    # ((n2, n4, n3), 
    
    # Mapping: n1=0, n2=1, n3=2, n4=3
    applied_faces = [
        (0, 1, 2), # n1, n2, n3
        (0, 3, 1), # n1, n4, n2
        (0, 2, 3), # n1, n3, n4
        (1, 3, 2)  # n2, n4, n3
    ]
    
    success = verify_tet_faces(applied_faces, nodes, centroid)
    
    if success:
        print("\n[SUCCESS] All faces have outward pointing normals!")
    else:
        print("\n[FAILURE] Some faces are still inverted.")

if __name__ == "__main__":
    main()
