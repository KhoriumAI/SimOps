import gmsh
import glob
import os
import sys
import time

print("============================================================")
print("[SERIAL] Linear Volume Mesher (No Parallelism)")
print("============================================================")
print("Strategy: Process one part at a time with full CPU resources")
print("============================================================\n")

# Find all STLs
stl_files = glob.glob(os.path.join("temp_stls", "**", "*.stl"), recursive=True)

if not stl_files:
    print("[X] No STL files found in temp_stls/")
    sys.exit(1)

# Filter out tiny/corrupt files
MIN_SIZE = 1024  # 1KB
valid_files = [f for f in stl_files if os.path.getsize(f) > MIN_SIZE]

print(f"Found {len(stl_files)} STL files, {len(valid_files)} valid (>{MIN_SIZE} bytes)\n")

# Output directory
output_dir = "temp_stls/volume_meshes"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize Gmsh ONCE (reuse across all parts)
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # Silence gmsh
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D (robust)
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)

success_count = 0
failed_count = 0
start_time = time.time()

for i, stl_path in enumerate(valid_files):
    filename = os.path.basename(stl_path)
    vol_tag = filename.replace("vol_", "").replace(".stl", "")
    output_path = os.path.join(output_dir, f"vol_{vol_tag}.msh")
    
    # Skip if already done
    if os.path.exists(output_path):
        print(f"[{i+1}/{len(valid_files)}] [SKIP] {filename} (already exists)")
        success_count += 1
        continue
    
    part_start = time.time()
    
    try:
        # Load STL
        gmsh.merge(stl_path)
        
        # Classify surfaces (creates geometry from mesh)
        try:
            gmsh.model.mesh.classifySurfaces(angle=40 * 3.14159 / 180, boundary=True)
            gmsh.model.mesh.createGeometry()
        except Exception as e:
            print(f"[{i+1}/{len(valid_files)}] [FAIL] {filename} - Not watertight: {str(e)[:40]}")
            failed_count += 1
            gmsh.clear()
            continue
        
        # Get surfaces and create volume
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            print(f"[{i+1}/{len(valid_files)}] [FAIL] {filename} - No surfaces after classification")
            failed_count += 1
            gmsh.clear()
            continue
        
        # Create surface loop and volume
        try:
            surface_tags = [s[1] for s in surfaces]
            sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
            gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
        except Exception as e:
            print(f"[{i+1}/{len(valid_files)}] [FAIL] {filename} - Cannot create volume: {str(e)[:40]}")
            failed_count += 1
            gmsh.clear()
            continue
        
        # Generate 3D volume mesh
        gmsh.model.mesh.generate(3)
        
        # Save
        gmsh.write(output_path)
        
        elapsed = time.time() - part_start
        print(f"[{i+1}/{len(valid_files)}] [OK] {filename} ({elapsed:.1f}s)")
        success_count += 1
        
        # Clear for next iteration
        gmsh.clear()
        
    except Exception as e:
        elapsed = time.time() - part_start
        print(f"[{i+1}/{len(valid_files)}] [FAIL] {filename} ({elapsed:.1f}s): {str(e)[:50]}")
        failed_count += 1
        gmsh.clear()

gmsh.finalize()

total_time = time.time() - start_time
print(f"\n============================================================")
print(f"Job Complete ({total_time:.1f}s)")
print(f"  Success: {success_count}")
print(f"  Failed: {failed_count}")
print(f"  Output: {output_dir}")
print(f"============================================================")

if failed_count > 0:
    print(f"\nFailed parts likely have self-intersections or topology errors.")
    print(f"Consider using voxel-based meshing for those specific parts.")
