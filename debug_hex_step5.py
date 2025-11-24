import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.hex_dominant_strategy import HighFidelityDiscretization, ConvexDecomposition
import gmsh
import trimesh

def test_step5_hex_subdivision(generate_hex_subdivision=True):
    """
    Test Step 5: Hex meshing via subdivision.
    Each tetrahedron is subdivided into 4 hexahedra.
    
    Args:
        generate_hex_subdivision: If True, use subdivision to create hexes.
                                  If False, just generate tets.
    """
    print("=== Testing Hex Dominant Meshing - Step 5 (Subdivision) ===")
    
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
    
    # Steps 3-5: Discrete Mesh + Optional Hex Subdivision
    print("\n[Step 3-4] Generating tetrahedral mesh...")
    
    gmsh.initialize()
    gmsh.model.add("hex_dom_step5_subdivision")
    
    # Set tolerances
    gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-4)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    
    # Clean and merge each part
    for i, (verts, faces) in enumerate(parts):
        chunk_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        chunk_mesh.merge_vertices()
        chunk_mesh.remove_degenerate_faces()
        chunk_mesh.remove_duplicate_faces()
        
        chunk_file = f"temp_chunk_{i}.stl"
        chunk_mesh.export(chunk_file)
        gmsh.merge(chunk_file)
        os.remove(chunk_file)
    
    # Classify discrete surfaces to create volumes
    try:
        angle = 40  # Angle in degrees for surface classification
        forceParametrizablePatches = False
        includeBoundary = True
        curveAngle = 180
        
        gmsh.model.mesh.classifySurfaces(angle * 3.14159 / 180, includeBoundary, forceParametrizablePatches, curveAngle * 3.14159 / 180)
        gmsh.model.mesh.createGeometry()
        
        # Create topology to generate volumes from classified surfaces
        s = gmsh.model.getEntities(2)  # Get all surfaces
        l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
        gmsh.model.geo.addVolume([l])
        gmsh.model.geo.synchronize()
        
        print(f"Created volume from {len(s)} classified surfaces")
    except Exception as e:
        print(f"Warning: Could not classify surfaces: {e}")
        print("Attempting direct volume meshing...")
    
    # Generate 3D tet mesh
    try:
        gmsh.model.mesh.generate(3)
        print("Generated tetrahedral mesh")
        
        # Count tets before subdivision
        tet_element_types = gmsh.model.mesh.getElementTypes()
        tet_counts = {}
        for etype in tet_element_types:
            elem_name = gmsh.model.mesh.getElementProperties(etype)[0]
            elem_tags, _ = gmsh.model.mesh.getElementsByType(etype)
            tet_counts[elem_name] = len(elem_tags)
        
        num_tets_initial = tet_counts.get("4-node tetrahedron", 0)
        print(f"Initial tet count: {num_tets_initial}")
        
    except Exception as e:
        print(f"[FAILURE] Tet meshing failed: {e}")
        gmsh.finalize()
        return False
    
    # Step 5: Apply Subdivision (Optional)
    if generate_hex_subdivision:
        print("\n[Step 5] Applying subdivision (tet -> hex conversion)...")
        
        # Subdivision algorithm: Each tet becomes 4 hexes
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # All hexes
        gmsh.model.mesh.refine()  # Trigger subdivision
        
        print("Subdivision applied")
    else:
        print("\n[Step 5] Skipping subdivision (keeping tets)")
    
    # Count final elements
    element_types = gmsh.model.mesh.getElementTypes()
    element_counts = {}
    
    for etype in element_types:
        elem_name = gmsh.model.mesh.getElementProperties(etype)[0]
        elem_tags, _ = gmsh.model.mesh.getElementsByType(etype)
        element_counts[elem_name] = len(elem_tags)
    
    print(f"\n=== Results ===")
    print("Element counts:")
    for name, count in element_counts.items():
        print(f"  {name}: {count}")
    
    # Calculate metrics (check for both naming conventions)
    num_hexes = element_counts.get("8-node hexahedron", 0) + element_counts.get("Hexahedron 8", 0)
    num_tets = element_counts.get("4-node tetrahedron", 0) + element_counts.get("Tetrahedron 4", 0)
    num_prisms = element_counts.get("6-node prism", 0) + element_counts.get("Prism 6", 0)
    num_pyramids = element_counts.get("5-node pyramid", 0) + element_counts.get("Pyramid 5", 0)
    
    total_3d = num_hexes + num_tets + num_prisms + num_pyramids
    hex_ratio = (num_hexes / total_3d * 100) if total_3d > 0 else 0
    
    print(f"\n3D Hex Ratio: {hex_ratio:.1f}% ({num_hexes}/{total_3d} volume elements)")
    
    if generate_hex_subdivision and num_tets_initial > 0:
        expected_hexes = num_tets_initial * 4
        print(f"Expected hexes from subdivision: {expected_hexes} (4 per tet)")
        if num_hexes == expected_hexes:
            print("✓ Subdivision ratio is correct (4 hexes per original tet)")
    
    # Save result
    output_filename = "debug_step5_hex_subdivision" if generate_hex_subdivision else "debug_step5_tet"
    gmsh.write(f"{output_filename}.msh")
    gmsh.write(f"{output_filename}.vtk")
    
    gmsh.finalize()
    
    # Report
    if hex_ratio == 100:
        print(f"\n[SUCCESS] 100% Hex mesh generated via subdivision!")
        return True
    elif hex_ratio > 0:
        print(f"\n[PARTIAL] Mixed mesh: {hex_ratio:.1f}% hexes")
        return True
    elif total_3d > 0:
        print(f"\n[INFO] Pure tet mesh: {num_tets} tetrahedra")
        return True
    else:
        print(f"\n[FAILURE] No 3D elements generated")
        return False

if __name__ == "__main__":
    # Test with subdivision enabled
    test_step5_hex_subdivision(generate_hex_subdivision=True)
