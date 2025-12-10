import gmsh
import sys
import os
import json
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from strategies.polyhedral_strategy import PolyhedralMeshGenerator

def check_poly_mesh():
    print("Running Polyhedral Mesh Repro...")
    
    # Create a simple cube geometry
    gmsh.initialize()
    gmsh.model.add("cube_test")
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    
    # Generate Tet Mesh
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.5)
    gmsh.model.mesh.generate(3)
    
    gmsh.write("temp_cube_base.msh")
    gmsh.finalize()
    
    # Run Poly Strategy
    generator = PolyhedralMeshGenerator()
    input_file = str(Path("temp_cube_base.msh").resolve())
    output_file = str(Path("temp_cube_poly.msh").resolve())
    
    print(f"Input: {input_file}")
    
    try:
        success = generator.run_meshing_strategy(input_file, output_file)
        print(f"Strategy Success: {success}")
        
        if success:
            json_file = output_file.replace('.msh', '.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                nodes = np.array([data['nodes'][str(i)] for i in range(len(data['nodes']))])
                print(f"Dual Nodes (Centroids): {len(nodes)}")
                
                # Check bounding box of dual nodes
                mins = np.min(nodes, axis=0)
                maxs = np.max(nodes, axis=0)
                print(f"Dual Mesh BBox: {mins} to {maxs}")
                print(f"Expected BBox: [0,0,0] to [1,1,1]")
                
                if mins[0] > 0.1 or maxs[0] < 0.9:
                    print("ISSUE CONFIRMED: Dual Mesh is shrunk/dented (Nodes are internal).")
                else:
                    print("Dual Mesh extends to boundary (Unexpected for standard Dual).")
                    
    except Exception as e:
        print(f"CRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_poly_mesh()
