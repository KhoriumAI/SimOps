"""
Conversion test script to verify the fixed Fluent mesh writer.
Converts Cube_mesh.msh to Fluent format using the corrected element type codes.
"""
import sys
sys.path.insert(0, r'c:\Users\Owner\Downloads\MeshPackageLean')

from core.write_fluent_mesh import convert_gmsh_to_fluent

if __name__ == "__main__":
    input_file = r"apps\cli\generated_meshes\Cube_mesh.msh"
    output_file = r"apps\cli\generated_meshes\Cube_test_fluent_FIXED.msh"
    
    print("=" * 60)
    print("FLUENT MESH CONVERSION TEST")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    convert_gmsh_to_fluent(input_file, output_file)
    
    print("=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Open ANSYS Fluent in 3D mode")
    print(f"2. Import: {output_file}")
    print("3. Verify you see 'tetrahedral cells' (not 'hexahedral')")
    print("=" * 60)
