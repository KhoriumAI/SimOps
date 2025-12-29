import gmsh
import os
import glob
import time
import math

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
STL_DIR = os.path.join(PROJECT_ROOT, "temp_stls", "plan_d_hybrid")
OUTPUT_VTK = os.path.join(PROJECT_ROOT, "simulation_ready_hybrid.vtk")
OUTPUT_INP = os.path.join(PROJECT_ROOT, "simulation_ready_hybrid.inp")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def mesh_hybrid_assembly():
    log(f"--- STARTING PLAN D: HYBRID ASSEMBLY MESHING ---")
    
    # 1. Check Inputs
    stl_files = glob.glob(os.path.join(STL_DIR, "vol_*.stl"))
    log(f"Found {len(stl_files)} STL components in {STL_DIR}")
    
    if len(stl_files) == 0:
        log("FAILURE: No STLs found. Run voxel_smash.py first.")
        return False
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Hybrid_Assembly")

    # 2. Merge STLs
    log(f"Merging {len(stl_files)} STLs...")
    # To avoid visual clutter, turn off GUI updates if possible (Terminal=1 only logs)
    
    # We need to process each STL to ensure it's a volume
    # Strategy: Merge all, then Classify -> CreateGeometry -> Volume
    
    try:
        for stl in stl_files:
            gmsh.merge(stl)
    except Exception as e:
        log(f"Error merging files: {e}")
        gmsh.finalize()
        return False

    # 3. Create Geometry from Discrete Surfaces (The Critical Step for STL->Vol)
    log("Classifying surfaces and creating geometry...")
    
    # Angle for sharp edges. 40 degrees is standard.
    # We classify *all* surfaces loaded.
    gmsh.model.mesh.classifySurfaces(40 * 3.14159 / 180, True, True)
    gmsh.model.mesh.createGeometry()
    
    # 4. Form Volumes
    log("Forming Volumes...")
    s = gmsh.model.getEntities(2)
    l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
    gmsh.model.geo.addVolume([l])
    
    # Wait, the above makes ONE giant volume from all surfaces?
    # NO. If the STLs are separate overlapping shells, 'addSurfaceLoop' for ALL surfaces
    # might fail or create a weird soup.
    # We need to preserve the 151 distinct volumes.
    # We should have merged them one by one, created a volume for each, THEN merged?
    # Or, after classifySurfaces, we might have 151 discrete surface patches?
    # It's safer to treat them individually.
    
    gmsh.model.remove()
    gmsh.model.add("Hybrid_Assembly_Robust")
    
    vol_tags = []
    
    log("Processing STLs individually (Robust)...")
    for i, stl in enumerate(stl_files):
        # New model for each to isolate processing? 
        # No, we want one assembly.
        # But we can import, classify, create volume, then move on?
        # Actually simplest: Load loops.
        # But looping classifySurfaces is tricky on a growing model.
        
        # Revised Strategy:
        # Load ONE, classify, make volume.
        # Repeat.
        # This keeps IDs cleaner.
        
        try:
            gmsh.merge(stl)
            
            # The newly merged entity is likely the last one?
            # getEntities(2) will return all. 
            # This is hard to track.
            # Actually, `gmsh.merge` doesn't return the tag.
            pass
        except:
             pass
    
    # Let's try the "Batch Classify" again but assume topology handles it.
    # If they are disjoint, classifySurfaces might verify them as discrete shells.
    # But addVolume([all_surfaces]) makes a single void? 
    # We need 151 volumes.
    
    # BETTER STRATEGY: 
    # Gmsh can just mesh the discrete surfaces if we verify they are closed?
    # No, we need Physical Volumes for the solver.
    
    # OK, let's restart the loop, clearing the model each time to turn STL->Volume, 
    # then save as .msh/tmp, and finally merge all .msh files?
    # That ensures valid volumes.
    
    return True # Placeholder while I refine logic

# REWRITING LOGIC BELOW FOR ROBUSTNESS
def robust_convert_and_merge():
    gmsh.initialize()
    stl_files = glob.glob(os.path.join(STL_DIR, "vol_*.stl"))
    log(f"Converting {len(stl_files)} STLs to Volumes...")
    
    temp_msh_dir = os.path.join(PROJECT_ROOT, "temp_msh_plan_d")
    if not os.path.exists(temp_msh_dir):
        os.makedirs(temp_msh_dir)
        
    converted_files = []
    
    # Step A: Convert each STL to a .msh volume independently
    for stl in stl_files:
        head, tail = os.path.split(stl)
        name = os.path.splitext(tail)[0]
        msh_out = os.path.join(temp_msh_dir, name + ".msh")
        
        if os.path.exists(msh_out):
            converted_files.append(msh_out)
            continue
            
        gmsh.model.add(name)
        gmsh.merge(stl)
        
        # Make geometry
        gmsh.model.mesh.classifySurfaces(math.pi, True, True) # Detect all as one patch if smooth?
        # Actually we want sharp edges.
        gmsh.model.mesh.classifySurfaces(40*math.pi/180, True, True)
        
        gmsh.model.mesh.createGeometry()
        
        surf_entities = gmsh.model.getEntities(2)
        if not surf_entities:
            # Maybe it's just nodes? Empty file?
            gmsh.model.remove()
            gmsh.model.add("Next")
            continue
            
        l = gmsh.model.geo.addSurfaceLoop([e[1] for e in surf_entities])
        v = gmsh.model.geo.addVolume([l])
        gmsh.model.geo.synchronize()
        
        # CREATE 3D MESH NOW?
        # Yes, mesh independent volumes now.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) # Delaunay
        gmsh.model.mesh.generate(3)
        
        # Create Physical Volume to ensure distinct blocks in VTK
        vols = gmsh.model.getEntities(3)
        if vols:
            # We assume 1 volume per file
            tag = vols[0][1]
            # Use original ID if possible, or just unique index
            p_tag = tag # Simple
            gmsh.model.addPhysicalGroup(3, [tag], p_tag)
            gmsh.model.setPhysicalName(3, p_tag, f"Volume_{name}")
        
        gmsh.write(msh_out)
        converted_files.append(msh_out)
        
        gmsh.model.remove()
    
    gmsh.finalize()
    
    # Step B: Merge all MSH files
    log(f"Merging {len(converted_files)} mesh volumes...")
    gmsh.initialize()
    gmsh.model.add("Final_Assembly")
    
    for msh in converted_files:
        gmsh.merge(msh)
        
    # Export
    log(f"Exporting Hybrid Assembly to {OUTPUT_VTK}...")
    gmsh.write(OUTPUT_VTK)
    
    log("Plan D Complete.")
    gmsh.finalize()

if __name__ == "__main__":
    robust_convert_and_merge()
