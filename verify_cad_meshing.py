import gmsh
import sys
import os
import math
import json
import time

def verify_cad(filepath):
    print(f"\n{'='*60}")
    print(f"VERIFYING CAD: {os.path.basename(filepath)}")
    print(f"PLATFORM: {sys.platform}")
    print(f"{'='*60}\n")

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False

    try:
        gmsh.initialize()
        
        # --- STAGE 1: ENVIRONMENT & VERSION ---
        print("[1/4] Checking Environment...")
        try:
            v_str = gmsh.option.getString("General.Version")
            b_str = gmsh.option.getString("General.BuildOptions")
            print(f"  Gmsh Version: {v_str}")
            print(f"  Build Info: {b_str[:100]}...")
            print(f"  Default Tolerance: {gmsh.option.getNumber('Geometry.Tolerance')}")
        except Exception as e:
            print(f"  Warning: Could not get build info: {e}")

        # --- STAGE 2: CONFIGURATION ---
        print("\n[2/4] Applying Fast Preview Configuration...")
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.RandomFactor", 1e-9)
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 3)
        
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
        gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
        
        # --- STAGE 3: IMPORT & TOPOLOGY ---
        print("\n[3/4] Importing Geometry...")
        start_time = time.time()
        gmsh.model.add("Verification")
        gmsh.open(filepath)
        import_time = time.time() - start_time
        print(f"  Import took: {import_time:.2f}s")

        dims = gmsh.model.getEntities()
        n_vol_pre = len([d for d in dims if d[0] == 3])
        n_surf = len([d for d in dims if d[0] == 2])
        print(f"  Initial State -> Volumes: {n_vol_pre}, Surfaces: {n_surf}")

        # Sync model
        gmsh.model.occ.synchronize()

        v_final = gmsh.model.getEntities(3)
        n_vol_post = len(v_final)
        print(f"  Final State -> Volumes: {n_vol_post}, Surfaces: {n_surf}")

        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = math.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
        print(f"  Model Bounding Box Diagonal: {diag:.20f}")

        # --- STAGE 4: COARSE MESHING ---
        print("\n[4/4] Attempting Coarse 2D Meshing...")
        mesh_size = diag / 20.0
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size / 5.0)
        gmsh.option.setNumber("Mesh.Algorithm", 2)
        gmsh.option.setNumber("Mesh.MaxRetries", 3)
        
        m_start = time.time()
        try:
            gmsh.model.mesh.generate(2)
        except Exception as e:
            print(f"  [!] Meshing logic reported an error (non-fatal): {e}")
        
        m_time = time.time() - m_start
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        print(f"  Meshing took: {m_time:.2f}s")
        print(f"  Node Count: {n_nodes}")

        # SUCCESS CRITERIA
        success = True
        if n_vol_post == 0 and n_vol_pre > 0:
            print("\n[!] FAILURE: All volumes were lost.")
            success = False
        if n_nodes == 0:
            print("\n[!] FAILURE: Zero mesh nodes generated.")
            success = False
        
        # Smart volume threshold
        if n_vol_post < 151:
            print(f"\n[!] FAILURE: Detected only {n_vol_post} volumes. Assembly potentially truncated on this environment.")
            success = False

        if success:
            print(f"\n{'='*20} SUCCESS {'='*20}")
            print(f"CAD meshable with {n_nodes} nodes across {n_vol_post} volumes.")
            if n_vol_post > n_vol_pre:
                print(f"NOTE: Recovered {n_vol_post - n_vol_pre} volumes via topological sewing!")
        else:
            print(f"\n{'='*20} FAILURE {'='*20}")
            print("The system failed to produce a valid mesh or lost too many volumes.")

        gmsh.finalize()
        return success

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Automation failed: {e}")
        try: gmsh.finalize()
        except: pass
        return False

if __name__ == "__main__":
    cad_path = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/markm/Downloads/MeshPackageLean/cad_files/00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
    verify_cad(cad_path)
