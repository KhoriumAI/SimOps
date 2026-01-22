import gmsh

gmsh.initialize()
gmsh.open('.simops/uploads/053de30b_Cube_medium_fast_tet.msh')

types = gmsh.model.mesh.getElementTypes()
print('Element types:', types)
print('\nElement type reference:')
print('  1=Line, 2=Triangle, 3=Quad, 4=Tet, 5=Hex, 6=Prism, 7=Pyramid')
print('\nElements in mesh:')
for t in types:
    elems, node_tags = gmsh.model.mesh.getElementsByType(t)
    print(f'  Type {t}: {len(elems)} elements')

gmsh.finalize()
