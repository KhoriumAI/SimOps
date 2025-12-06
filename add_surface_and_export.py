import meshio
import numpy as np
import os

def extract_skin_and_export(input_file, output_file):
    print(f"Reading {input_file}...")
    mesh = meshio.read(input_file)
    
    # 1. LINEARIZE (tetra10 -> tetra)
    # Fluent prefers linear elements for basic CFD
    points = mesh.points
    cells = []
    
    tets = None
    
    for c in mesh.cells:
        if c.type == 'tetra10':
            print(f"Linearizing {len(c.data)} tetra10 elements...")
            tets = c.data[:, :4] # Keeping 4 corners
        elif c.type == 'tetra':
            tets = c.data
    
    if tets is None:
        print("Error: No tetrahedra found!")
        return

    # 2. EXTRACT SKIN (Boundary Faces)
    print("Extracting boundary faces (skin)...")
    
    # Face definitions for a tet: (0,1,2), (0,3,1), (0,2,3), (1,3,2)
    # Correct winding for outward normals matters, but getting the skin is step 1.
    tet_faces = np.array([
        [0, 2, 1],
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3]
    ])
    
    # Get all faces: (N_tets * 4, 3)
    all_faces = tets[:, tet_faces].reshape(-1, 3)
    
    # Sort vertex indices within each face to identify duplicates
    sorted_faces = np.sort(all_faces, axis=1)
    
    # Find unique faces and their counts
    # Using a structured array for performance if needed, or simple string conversion/void view
    # Void view trick for fast row comparison
    dtype = np.dtype((np.void, sorted_faces.dtype.itemsize * sorted_faces.shape[1]))
    packed_faces = np.ascontiguousarray(sorted_faces).view(dtype)
    
    unique_faces, return_inverse, counts = np.unique(packed_faces, return_inverse=True, return_counts=True)
    
    # Boundary faces are those that appear exactly once
    boundary_mask = (counts == 1)
    boundary_indices = np.where(boundary_mask[return_inverse])[0]
    
    # Get the original (unsorted, properly wound) faces
    skin_faces = all_faces[boundary_indices]
    
    print(f"Found {len(skin_faces)} boundary triangles.")
    
    if len(skin_faces) == 0:
        print("Error: No boundary faces found! Mesh might be degenerate or twisted.")
        return

    # 3. CREATE NEW MESH WITH BOTH VOL AND SURF
    
    # Cells: Volume (tetra) + Surface (triangle)
    new_cells = [
        ('tetra', tets),
        ('triangle', skin_faces)
    ]
    
    # Create meshio object
    # P.S. We can try to add cell_data for Physical Groups if we want, 
    # but meshio's TGrid writer might auto-detect element types.
    # Let's write to Gmsh 2.2 which handles this well.
    
    new_mesh = meshio.Mesh(
        points=points,
        cells=new_cells
    )
    
    # 4. EXPORT TO GMSH 2.2 (ASCII)
    # We use Gmsh format because we can then use Gmsh API to assign physical names strictly
    temp_gmsh = output_file + ".temp.msh"
    print(f"Writing intermediate mesh to {temp_gmsh}...")
    meshio.write(temp_gmsh, new_mesh, file_format="gmsh22", binary=False)
    
    # 5. ASSIGN PHYSICAL GROUPS VIA GMSH API
    # This is safer than relying on meshio's physical group handling which can be tricky
    import gmsh
    gmsh.initialize()
    gmsh.open(temp_gmsh)
    
    # Delete any existing groups
    gmsh.model.removePhysicalGroups(gmsh.model.getPhysicalGroups())
    
    # Create Volume Group
    vols = gmsh.model.getEntities(3)
    if not vols:
        # If no strict entities, usually loading a mesh creates discrete entities.
        # Let's check.
        pass
        
    # Standardize: Create Physical Volume "fluid" and Surface "wall"
    # We find all 3D elements and make them fluid
    # We find all 2D elements and make them wall
    
    # Get all 3D elements (tetrahedra)
    elem_types_3d = gmsh.model.mesh.getElementTypes(3)
    if elem_types_3d:
        p_vol = gmsh.model.addPhysicalGroup(3, [1]) # Assuming trivial entity 1
        # Actually better to classify by entity.
        # Since meshio output puts them in entities, lets just set names on all 3D tags
        tags_3d = [e[1] for e in gmsh.model.getEntities(3)]
        if tags_3d:
             p_vol = gmsh.model.addPhysicalGroup(3, tags_3d)
             gmsh.model.setPhysicalName(3, p_vol, "fluid")
    
    # Get all 2D elements (triangles)
    tags_2d = [e[1] for e in gmsh.model.getEntities(2)]
    if tags_2d:
         p_surf = gmsh.model.addPhysicalGroup(2, tags_2d)
         gmsh.model.setPhysicalName(2, p_surf, "wall")
         
    print(f"Assigned Physical Groups: fluid (3D), wall (2D)")
    
    # Final Export
    print(f"Exporting final clean mesh to {output_file}...")
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0) # Save only physical groups
    gmsh.write(output_file)
    gmsh.finalize()
    
    # Clean up
    try:
        os.remove(temp_gmsh)
    except:
        pass
    
    print("Done!")

if __name__ == "__main__":
    input_f = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_mesh.msh"
    output_f = "C:/Users/Owner/Downloads/MeshPackageLean/apps/cli/generated_meshes/Cube_with_skin.msh"
    extract_skin_and_export(input_f, output_f)
