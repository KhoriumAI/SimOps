import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

def verify_integration():
    print("--- Verifying Modal SnappyHexMesh Integration ---")
    
    # 1. Setup paths
    step_file = Path("cad_files/model.step").absolute()
    stl_file = Path("cad_files/model.stl").absolute()
    output_mesh = Path("test_modal_result.msh").absolute()
    
    if not step_file.exists():
        print(f"ERROR: {step_file} not found!")
        return
        
    # 2. Convert STEP to STL (mimicking backend behavior)
    print(f"Converting {step_file.name} to STL...")
    try:
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.occ.importShapes(str(step_file))
        gmsh.model.occ.synchronize()
        gmsh.write(str(stl_file))
        gmsh.finalize()
        print(f"STL generated: {stl_file}")
    except ImportError:
        print("Gmsh not installed locally? Attempting to continue if STL exists...")
        if not stl_file.exists():
            print("ERROR: Cannot generate STL (gmsh missing) and STL not found.")
            return

    # 3. Configure Environment for Cloud Offloading
    os.environ['USE_MODAL_COMPUTE'] = 'true'
    os.environ['MODAL_OPENFOAM_APP'] = 'khorium-openfoam-snappy'
    # os.environ['S3_BUCKET_NAME'] is usually loaded from .env or defaults
    
    print(f"USE_MODAL_COMPUTE: {os.environ.get('USE_MODAL_COMPUTE')}")
    
    # 4. Import Strategy
    try:
        from strategies.openfoam_hex import generate_openfoam_hex_mesh
    except ImportError as e:
        print(f"ERROR: Could not import strategy: {e}")
        return

    # 5. Run Generation
    print("Triggering mesh generation...")
    start_time = time.time()
    
    result = generate_openfoam_hex_mesh(
        stl_path=str(stl_file),
        output_path=str(output_mesh),
        cell_size=2.0,
        verbose=True,
        mesh_scope='Internal'
    )
    
    duration = time.time() - start_time
    print(f"Generation took: {duration:.2f}s")
    
    # 6. Validate Result
    print(f"Result: {result}")
    
    if result.get('success'):
        if output_mesh.exists():
            size_kb = output_mesh.stat().st_size / 1024
            print(f"✅ SUCCESS: Mesh file created at {output_mesh} ({size_kb:.1f} KB)")
        else:
            print("❌ FAILURE: Success reported but output file missing!")
    else:
        print(f"❌ FAILURE: Generation failed: {result.get('error')}")

if __name__ == "__main__":
    verify_integration()
