import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.hex_dominant_strategy import HighFidelityDiscretization, ConvexDecomposition
import gmsh
import trimesh

def test_step4_discrete():
    """Test with discrete mesh approach (no OCC conversion)"""
    print("=== Testing Hex Dominant Meshing - Step 4 (Discrete Mesh Approach) ===")
    
    cad_file = "C:/Users/Owner/Downloads/MeshTest/cad_files/cube.step"
    
    if not os.path.exists(cad_file):
        print(f"Error: CAD file not found at {cad_file}")
        return
        
    temp_dir = tempfile.gettempdir()
    output_stl = os.path.join(temp_dir, "hex_dom_temp.stl")
    
    # Step 1: Convert STEP to STL
    print("\n[Step 1] Converting STEP to STL...")
    step1 = HighFidelityDiscretization()
    success = step1.convert_step_to_stl(cad_file, output_stl, min_size=0.5, max_size=10.0)
    
    if not success:
        print("Failed Step 1.")
        return
        
    # Step 2: CoACD Decomposition
    print("\n[Step 2] Running CoACD Decomposition...")
    step2 = ConvexDecomposition()
    parts, stats = step2.decompose_mesh(output_stl, threshold=0.05)
    
    if not parts:
        print("Failed Step 2.")
        return
        
    print(f"Decomposed into {len(parts)} parts.")
    
    # Volume Check Gate
    volume_error = stats.get('volume_error_pct', 100.0)
    if volume_error > 5.0:
        print(f"\n[Error] Volume mismatch {volume_error:.2f}% is too high (>5%). Aborting.")
        return
    
    # Step 3 & 4: Discrete Mesh Approach (no B-Rep conversion)
    print("\n[Step 3+4] Meshing directly from cleaned parts...")
    
    gmsh.initialize()
    gmsh.model.add("hex_dom_discrete")
    
    # Set tolerances
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-4)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    
    # Clean and merge each part as a discrete surface mesh
    for i, (verts, faces) in enumerate(parts):
        chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Clean
        chunk_mesh.merge_vertices()
        chunk_mesh.remove_degenerate_faces()
        chunk_mesh.remove_duplicate_faces()
        
        # Save and merge
        chunk_file = f"temp_chunk_{i}.stl"
        chunk_mesh.export(chunk_file)
        gmsh.merge(chunk_file)
        os.remove(chunk_file)
    
    # Generate 3D mesh directly from the discrete surfaces
    try:
        gmsh.model.mesh.generate(3)
        print(f"\n[SUCCESS] Generated 3D mesh!")
        gmsh.write("debug_step4_discrete.msh")
        gmsh.finalize()
        return True
    except Exception as e:
        print(f"\n[FAILURE] Meshing failed: {e}")
        gmsh.finalize()
        return False

if __name__ == "__main__":
    test_step4_discrete()
