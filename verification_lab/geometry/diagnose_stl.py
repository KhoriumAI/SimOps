"""Diagnose STL file topology issues"""
import pyvista as pv
import sys

stl_file = "temp_stls/vol_24.stl"

try:
    mesh = pv.read(stl_file)
    
    print(f"STL File: {stl_file}")
    print(f"=========================================")
    print(f"Vertices: {mesh.n_points}")
    print(f"Cells (triangles): {mesh.n_cells}")
    print(f"Is manifold: {mesh.is_manifold}")
    
    # Check for holes and boundary edges
    edges = mesh.extract_feature_edges(
        boundary_edges=True, 
        non_manifold_edges=True, 
        feature_edges=False, 
        manifold_edges=False
    )
    
    print(f"Boundary edges: {edges.n_points} points, {edges.n_cells} edges")
    
    # Non-manifold check
    non_manifold = mesh.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False
    )
    
    print(f"Non-manifold edges: {non_manifold.n_cells}")
    
    # Try to fill holes
    filled = mesh.fill_holes(1000)
    print(f"Fill holes result: {filled.n_points} points")
    
    # Check normals
    print(f"Has consistent normals: checking...")
    mesh_with_normals = mesh.compute_normals(consistent_normals=True)
    
    print("")
    print("DIAGNOSIS:")
    if not mesh.is_manifold:
        print("  ❌ NOT MANIFOLD - Has topological defects")
    if edges.n_cells > 0:
        print(f"  ❌ HAS HOLES - {edges.n_cells} boundary edges detected")
    if non_manifold.n_cells > 0:
        print(f"  ❌ NON-MANIFOLD EDGES - {non_manifold.n_cells} edges shared by >2 faces")
    
    if mesh.is_manifold and edges.n_cells == 0 and non_manifold.n_cells == 0:
        print("  ✅ STL appears clean and watertight")
    else:
        print("\n  RECOMMENDATION: Use aggressive_healing=True or repair STL first")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
