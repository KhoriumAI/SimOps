def write_golden_sample():
    filename = "SingleTet.msh"
    print(f"Writing {filename}...")
    with open(filename, 'w') as f:
        # 1. Header
        f.write('(0 "Golden Sample 1-Tet Mesh")\n')
        
        # 2. Dimensions (3D)
        f.write('(2 3)\n')
        
        # 3. Nodes (Zone 1)
        # (10 (zone-id start end type nd) ( ... ))
        f.write('(10 (0 1 4 0 3))\n')
        f.write('(10 (1 1 4 1 3)(\n')
        f.write('0.0 0.0 0.0\n')
        f.write('1.0 0.0 0.0\n')
        f.write('0.0 1.0 0.0\n')
        f.write('0.0 0.0 1.0\n')
        f.write('))\n')
        
        # 4. Cells (Zone 2 - Fluid)
        # (12 (zone-id start end type element-type))
        # Type 1=Fluid, Elem 2=Tet (FIXED: was 4=Hex)
        f.write('(12 (0 1 1 0 0))\n')
        f.write('(12 (2 1 1 1 2))\n')  # CRITICAL FIX: Element Type 2 = Tetrahedron
        
        # 5. Faces (Zone 3 - Boundary)
        # (13 (zone-id start end type element-type) ( ... ))
        # Type 3=Wall, Elem 2=Tri
        # Connectivity: n0 n1 n2 c0 c1
        # Since it's 1 tet, ALL 4 faces are boundaries (c1 = 0)
        f.write('(13 (0 1 4 0 0))\n')
        f.write('(13 (3 1 4 3 2)(\n')
        f.write('1 3 2 1 0\n') # Face 1
        f.write('1 2 4 1 0\n') # Face 2
        f.write('2 3 4 1 0\n') # Face 3
        f.write('3 1 4 1 0\n') # Face 4
        f.write('))\n')
        
    print("Done. Try loading 'SingleTet.msh' in Fluent.")

if __name__ == "__main__":
    write_golden_sample()
