"""
Debug script to inspect mesh cell types
"""
import meshio

mesh_file = r"apps\cli\generated_meshes\Cube_mesh.msh"
print(f"Reading: {mesh_file}")
mesh = meshio.read(mesh_file)

print(f"\nPoints: {len(mesh.points)}")
print(f"\nCells type: {type(mesh.cells)}")
print(f"Cells: {mesh.cells}")

if hasattr(mesh.cells, '__iter__'):
    for i, cell_block in enumerate(mesh.cells):
        print(f"\nCell block {i}:")
        print(f"  Type: {type(cell_block)}")
        if hasattr(cell_block, 'type'):
            print(f"  cell_block.type: {cell_block.type}")
        if hasattr(cell_block, 'data'):
            print(f"  cell_block.data shape: {cell_block.data.shape}")
        if isinstance(cell_block, (list, tuple)):
            print(f"  Tuple/List: {cell_block[0] if len(cell_block) > 0 else 'empty'}")
