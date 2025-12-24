
import sys
import os
from pathlib import Path
import gmsh
import logging

# Ensure imports work
sys.path.insert(0, "/app")

# CFDMeshConfig is defined in the strategy file
from core.strategies.cfd_strategy import CFDMeshStrategy, CFDMeshConfig

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SemanticTest")

def create_semantic_geometry(filepath):
    """
    Creates a box 1x1x1.
    Tags TOP face (Z=1) as 'My_Custom_Source'.
    Tags BOTTOM face (Z=0) as 'My_Inlet'.
    """
    gmsh.initialize()
    # Note: OCCExportLabels might not be supported in this Gmsh build,
    # but we are using .brep which uses native OCC format.
    # Hopefully it preserves names if we add Physical Groups.
    gmsh.model.add("SemanticBox")
    
    # Create Box
    # x,y,z, dx,dy,dz
    box = gmsh.model.occ.addBox(0,0,0, 1,1,1)
    gmsh.model.occ.synchronize()
    
    # Identify faces
    # Get all surfaces
    surfs = gmsh.model.getEntities(dim=2)
    
    top_face = []
    bottom_face = []
    
    for dim, tag in surfs:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_center = (bb[2] + bb[5]) / 2.0
        if z_center > 0.9: # Top
            top_face.append(tag)
        elif z_center < 0.1: # Bottom
            bottom_face.append(tag)
            
    # Add Physical Groups with SEMANTIC NAMES
    if top_face:
        p1 = gmsh.model.addPhysicalGroup(2, top_face)
        gmsh.model.setPhysicalName(2, p1, "My_Custom_Source_Face") # Should trigger 'source'
        
    if bottom_face:
        p2 = gmsh.model.addPhysicalGroup(2, bottom_face)
        gmsh.model.setPhysicalName(2, p2, "Cold_Flow_Inlet") # Should trigger 'inlet'
        

    # Save as MSH to preserve Physical Groups reliably for the test
    # (Simulating a successful Import)
    gmsh.model.mesh.generate(2) # Need 2D mesh to save physical surfaces
    gmsh.write(str(filepath))
    gmsh.finalize()
    print(f"Created geometry: {filepath}")

def test_semantic_tagging():
    test_file = Path("/output/semantic_test.msh") # Input is MSH now
    output_mesh = Path("/output/semantic_tagged.msh")
    
    # 1. Create Input
    create_semantic_geometry(test_file)
    
    # 2. Run Strategy
    print("\n--- Running CFD Strategy ---")
    strategy = CFDMeshStrategy(verbose=True)
    
    # Create a dummy config
    config = CFDMeshConfig(
        max_mesh_size=0.1,
        min_mesh_size=0.1,
        num_layers=0
    )
    
    # Capture logs? We rely on stdout.
    success, stats = strategy.generate_cfd_mesh(test_file, str(output_mesh), config)
    
    if not success:
        print("Strategy failed!")
        return
        
    # 3. Validation
    # We need to verify that BC_HeatSource exists and corresponds to the TOP face (not bottom)
    # Re-open the generated mesh and check groups
    print("\n--- Validating Results ---")
    gmsh.initialize()
    gmsh.open(str(output_mesh))
    
    phys_groups = gmsh.model.getPhysicalGroups(dim=2)
    found_source = False
    source_z_avg = -1.0
    
    for dim, tag in phys_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        print(f"Found Group: '{name}'")
        
        if "BC_HeatSource" in name:
            found_source = True
            # Check Z location
            # Get entities
            ents = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            z_vals = []
            for e in ents:
                bb = gmsh.model.getBoundingBox(dim, e)
                z_vals.append((bb[2] + bb[5])/2.0)
            
            if z_vals:
                source_z_avg = sum(z_vals)/len(z_vals)
    
    gmsh.finalize()
    
    print(f"\nResult: Heat Source Found? {found_source}")
    print(f"Result: Heat Source Z-Level? {source_z_avg:.2f} (Expected ~1.0 for Top Face)")
    
    if found_source and abs(source_z_avg - 1.0) < 0.1:
        print("SUCCESS: Semantic Tagging correctly identified Top Face as Heat Source!")
    elif found_source and abs(source_z_avg - 0.0) < 0.1:
        print("FAILURE: Semantic Tagging failed, fallback picked Bottom Face (Z-min)!")
    else:
        print(f"FAILURE: Unexpected result. Z={source_z_avg}")

if __name__ == "__main__":
    test_semantic_tagging()
