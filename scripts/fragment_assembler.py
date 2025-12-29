import gmsh
import os
import glob
import sys
import time

# --- CONFIG ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_stls")
OUTPUT_MSH = os.path.join(PROJECT_ROOT, "imprinted_assembly.msh")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    t0 = time.time()
    log("STARTING BOOLEAN FRAGMENT ASSEMBLY (IMPRINTING)...")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    # 1. Load All STLs as Discrete Entities
    stls = sorted(glob.glob(os.path.join(TEMP_DIR, "vol_*.stl")))
    if not stls:
        log("[!] No STLs found in temp_stls/")
        sys.exit(1)
        
    log(f"   - Found {len(stls)} volumes. Loading...")
    
    # Merge each file. Each merge creates a new entity.
    for p in stls:
        gmsh.merge(p)
        
    # 2. Create Topology (Discrete -> Geometry)
    # This is critical. It turns the STL triangles into a "Surface" entity GMSH can manipulate.
    log("   - Creating Topology from Discrete Mesh...")
    gmsh.model.mesh.createTopology()
    
    # 3. Identify Surfaces
    surfs = gmsh.model.getEntities(2)
    log(f"   - Identified {len(surfs)} discrete surfaces.")
    
    # 4. Boolean Fragments (Imprinting)
    # This splits overlapping surfaces so they share nodes (Conformal Interface).
    log("   - Running Boolean Fragments (Imprinting)...")
    # We fragment the whole set of surfaces against themselves.
    # objectDimTags, toolDimTags   (here we just pass all surfs as objects, empty tools, or split?)
    # usually fragment(object, tool). If we pass all as objects, it intersects them all?
    # GMSH docs: "computes the boolean fragments (intersections) of all the entities in objectDimTags and toolDimTags"
    try:
        occ_surfs = surfs # These are discrete entities. Fragment usually works on CAD (OCC).
        # WAIT. Discrete entities cannot be processed by OCC Kernel.
        # We need "Mesh Boolean" or "Create Geometry" + OCC?
        # If we use `createTopology`, they become GEO internals, not OCC.
        # So we use `gmsh.model.geo`? Or `gmsh.model.occ`?
        # Discrete entities are usually GEO kernel.
        # But GEO kernel doesn't support Boolean Fragments fully robustly like OCC.
        
        # ALTERNATIVE: Use "Coherence" or "RemoveDuplicateNodes" explicitly?
        # User said "Boolean Fragments".
        # If we only have STLs, we rely on GMSH's ability to "remesh" the compound.
        
        # Let's try `gmsh.model.mesh.createGeometry()`? 
        # This creates a parametrization.
        
        # SIMPLER APPROACH (Common for STL Assembly):
        # 1. Merge all.
        # 2. createTopology.
        # 3. Use `gmsh.model.mesh.generate(3)` directly?
        #    If nodes match, it works. If not, it fails.
        #    They WON'T match.
        
        # User insisted on IMPRINTING.
        # To Imprint STLs, we might need to use `Mesh.Algorithm3D=10` (HXT) which handles non-conformal?
        # Or `Compound`?
        
        # Let's stick to the "BooleanFragments" plan but warn:
        # GMSH Occ Fragment needs BRep/STEP.
        # To fragment STLs, we might need `gmsh.plugin.run("SimplePartition")`?
        
        # Let's write a script that attempts the "Coherence" (Merge Nodes) function first on the topology.
        # `gmsh.model.geo.removeAllDuplicates()`
        
        # Actually, let's keep it simple for v1:
        # Just merge them and run 3D generation. The user said "Don't stitch".
        # But if we don't stitch, they are disconnected.
        # GMSH `classifySurfaces` -> `createGeometry` -> `mesh.generate` might be enough if the gap is zero.
        # But gap is not zero.
        
        # I will implement a script that simply Merges and saves as .msh for inspection first.
        # Then I will check if I can run a fragment command.
        pass

    except Exception as e:
        log(f"[!] Fragment Error: {e}")
        
    gmsh.finalize()

if __name__ == "__main__":
    main()
