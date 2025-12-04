"""
Fluent Mesh Repair Utility
===========================

Repairs mesh files that are missing surface triangles (causing "Null Domain Pointer" in Fluent).
Calculates the boundary skin from volume tetrahedra and exports in native Fluent format.

Usage:
    python repair_fluent_mesh.py <input_file.msh>
    
Or run without arguments to auto-detect files in generated_meshes/
"""

import meshio
import numpy as np
import os
import sys
from pathlib import Path

def repair_and_convert_to_fluent(input_file):
    """
    Repair a mesh file by calculating missing surface triangles and converting to Fluent format.
    
    Args:
        input_file: Path to the mesh file to repair
    """
    print(f"\n{'='*60}")
    print(f"FLUENT MESH REPAIR UTILITY")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        return False
        
    try:
        mesh = meshio.read(input_file)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return False

    print(f"Original mesh: {len(mesh.cells_dict.get('tetra', []))} tetrahedra")

    # 1. Extract Volume Elements
    if 'tetra' not in mesh.cells_dict:
        print("ERROR: No tetrahedra found in mesh!")
        return False
    
    tets = mesh.cells_dict['tetra']

    # 2. Calculate the "Skin" (Boundary Faces)
    # A face is on the boundary if it belongs to only ONE tetrahedron
    print("Calculating mesh skin (recovering lost surfaces)...")
    
    # Create all 4 faces for every tet (sort nodes to handle winding)
    faces = np.vstack([
        tets[:, [0, 1, 2]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]],
        tets[:, [1, 2, 3]]
    ])
    faces.sort(axis=1)
    
    # Find unique faces and their counts
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    
    # Faces that appear exactly ONCE are the boundary skin
    boundary_faces = unique_faces[counts == 1]
    
    print(f"Recovered {len(boundary_faces)} surface triangles")

    # 3. Create New Mesh Structure
    # Fluent needs both the Volume (Cells) and the Surface (Faces)
    new_cells = [
        ("triangle", boundary_faces),  # Skin
        ("tetra", tets)                # Volume
    ]

    # 4. Write as Native Fluent ASCII
    output_file = str(Path(input_file).with_suffix('')) + "_repaired.msh"
    print(f"Writing native Fluent file: {output_file}")
    
    new_mesh = meshio.Mesh(
        points=mesh.points,
        cells=new_cells
    )
    
    try:
        # Force ANSYS ASCII format (native Fluent)
        meshio.write(output_file, new_mesh, file_format="ansys")
        print(f"\n{'='*60}")
        print("SUCCESS: Mesh repaired!")
        print(f"{'='*60}")
        print(f"Import '{output_file}' into Fluent:")
        print("  File → Import → Mesh")
        return True
    except Exception as e:
        print(f"ERROR writing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User provided filename
        repair_and_convert_to_fluent(sys.argv[1])
    else:
        # Auto-detect in generated_meshes folder
        search_paths = [
            "apps/cli/generated_meshes",
            "generated_meshes",
            "."
        ]
        
        found = False
        for search_dir in search_paths:
            if os.path.exists(search_dir):
                msh_files = list(Path(search_dir).glob("*_fluent.msh"))
                if msh_files:
                    print(f"Found {len(msh_files)} Fluent mesh file(s) in {search_dir}:")
                    for i, f in enumerate(msh_files, 1):
                        print(f"  {i}. {f.name}")
                    
                    if len(msh_files) == 1:
                        repair_and_convert_to_fluent(str(msh_files[0]))
                        found = True
                        break
                    else:
                        choice = input(f"\nEnter number (1-{len(msh_files)}) or full path: ").strip()
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(msh_files):
                                repair_and_convert_to_fluent(str(msh_files[idx]))
                                found = True
                                break
                        except ValueError:
                            if os.path.exists(choice):
                                repair_and_convert_to_fluent(choice)
                                found = True
                                break
        
        if not found:
            print("No mesh files found. Usage:")
            print("  python repair_fluent_mesh.py <input_file.msh>")
