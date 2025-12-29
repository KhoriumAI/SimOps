import gmsh
import sys
import os
import glob
import math

print("============================================================")
print("[DIAGNOSTIC] STL Geometry Validator")
print("============================================================")

# Find all STLs
stl_files = glob.glob(os.path.join("temp_stls", "**", "*.stl"), recursive=True)

if not stl_files:
    print("[X] No STL files found in temp_stls/")
    sys.exit(1)

print(f"Found {len(stl_files)} STL files. Testing first 5...\n")

for i, stl_path in enumerate(stl_files[:5]):
    print(f"[{i+1}] Testing: {os.path.basename(stl_path)}")
    print(f"    Size: {os.path.getsize(stl_path) / 1024:.1f} KB")
    
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Load STL
        gmsh.merge(stl_path)
        
        # Try to classify surfaces (this fails if not watertight)
        try:
            gmsh.model.mesh.classifySurfaces(angle=math.pi/3, boundary=True)
            gmsh.model.mesh.createGeometry()
            
            # Check if we got volumes
            surfaces = gmsh.model.getEntities(2)
            print(f"    [OK] Watertight - {len(surfaces)} surfaces detected")
            
        except Exception as e:
            print(f"    [X] LEAKY - Not watertight: {str(e)[:50]}")
        
        gmsh.finalize()
        
    except Exception as e:
        print(f"    [X] FAILED to load: {e}")
        try:
            gmsh.finalize()
        except:
            pass
    
    print()

print("=" * 60)
print("If most files show [X] LEAKY, your STEP export settings are")
print("creating non-manifold geometry. You may need to:")
print("  1. Use a different export method")
print("  2. Run a mesh repair tool (e.g., MeshLab, Blender)")
print("  3. Use voxel-based meshing instead")
