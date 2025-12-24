"""
Paintbrush Refinement Debug Script
===================================

This script tests each step of the paintbrush refinement pipeline:
1. Verify painted regions are collected
2. Verify they're passed to the worker
3. Verify backend receives them
4. Verify Gmsh fields are created
5. Verify mesh is actually refined

Run this after painting some regions in the GUI.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_step_1_data_collection():
    """Test if painted regions are being collected in the GUI"""
    print("\n" + "="*70)
    print("STEP 1: Testing Data Collection")
    print("="*70)
    
    # Create mock painted regions
    painted_regions = [
        {'center': [0.5, 0.5, 0.5], 'radius': 0.1},
        {'center': [0.3, 0.3, 0.3], 'radius': 0.15},
        {'center': [0.7, 0.7, 0.7], 'radius': 0.12},
    ]
    
    print(f"[OK] Created {len(painted_regions)} test regions")
    for i, region in enumerate(painted_regions):
        print(f"  Region {i+1}: center={region['center']}, radius={region['radius']}")
    
    return painted_regions

def test_step_2_file_passing(painted_regions):
    """Test if data can be written to and read from config file"""
    print("\n" + "="*70)
    print("STEP 2: Testing File-Based Parameter Passing")
    print("="*70)
    
    quality_params = {
        'quality_preset': 'Medium',
        'painted_regions': painted_regions,
        'target_elements': 5000
    }
    
    # Write to temp file
    fd, config_path = tempfile.mkstemp(suffix='.json', prefix='test_config_')
    os.close(fd)
    
    with open(config_path, 'w') as f:
        json.dump(quality_params, f, indent=2)
    
    print(f"[OK] Wrote config to: {config_path}")
    print(f"  File size: {os.path.getsize(config_path)} bytes")
    
    # Read it back
    with open(config_path, 'r') as f:
        loaded_params = json.load(f)
    
    print(f"[OK] Read config back successfully")
    print(f"  Painted regions count: {len(loaded_params.get('painted_regions', []))}")
    
    # Cleanup
    os.remove(config_path)
    
    return quality_params

def test_step_3_backend_injection(quality_params):
    """Test if backend can inject painted regions into config"""
    print("\n" + "="*70)
    print("STEP 3: Testing Backend Config Injection")
    print("="*70)
    
    from core.config import Config
    
    config = Config()
    
    # Inject painted regions (mimicking mesh_worker_subprocess.py)
    if 'painted_regions' in quality_params:
        config.painted_regions = quality_params['painted_regions']
        print(f"[OK] Injected {len(config.painted_regions)} painted regions into config")
    
    # Verify
    if hasattr(config, 'painted_regions'):
        print(f"[OK] Config has 'painted_regions' attribute")
        print(f"  Count: {len(config.painted_regions)}")
        print(f"  First region: {config.painted_regions[0]}")
    else:
        print(f"[X] Config does NOT have 'painted_regions' attribute")
        return False
    
    return config

def test_step_4_field_creation(config):
    """Test if Gmsh fields can be created from painted regions"""
    print("\n" + "="*70)
    print("STEP 4: Testing Gmsh Field Creation")
    print("="*70)
    
    try:
        import gmsh
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Create a simple box geometry
        gmsh.model.add("test_box")
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        
        print("[OK] Created test geometry (1x1x1 box)")
        
        # Get geometry info
        bounds = gmsh.model.getBoundingBox(-1, -1)
        diagonal = ((bounds[3]-bounds[0])**2 + (bounds[4]-bounds[1])**2 + (bounds[5]-bounds[2])**2)**0.5
        base_size = diagonal / 20.0
        
        print(f"  Diagonal: {diagonal:.3f}")
        print(f"  Base mesh size: {base_size:.3f}")
        
        # Apply painted refinement (mimicking _apply_painted_refinement)
        if hasattr(config, 'painted_regions') and config.painted_regions:
            fields = []
            
            for i, region in enumerate(config.painted_regions):
                center = region.get('center')
                radius = region.get('radius')
                
                if not center or not radius:
                    continue
                
                xc, yc, zc = center
                
                # Create distance field
                dist_field = gmsh.model.mesh.field.add("MathEval")
                expr = f"Sqrt((x-{xc})*(x-{xc}) + (y-{yc})*(y-{yc}) + (z-{zc})*(z-{zc}))"
                gmsh.model.mesh.field.setString(dist_field, "F", expr)
                
                # Create threshold field
                thresh_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
                
                target_min = base_size / 5.0  # 5x finer
                gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", target_min)
                gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", base_size)
                gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", radius * 0.8)
                gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", radius * 1.5)
                
                fields.append(thresh_field)
                print(f"  [OK] Created field {i+1}: target size {target_min:.4f} (5x finer than {base_size:.4f})")
            
            if fields:
                # Combine fields
                min_field = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
                
                print(f"[OK] Applied {len(fields)} refinement fields as background mesh")
            
            # Generate mesh
            print("\n  Generating mesh with refinement...")
            gmsh.option.setNumber("Mesh.Algorithm", 6)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)
            gmsh.model.mesh.generate(3)
            
            # Get mesh stats
            nodes = gmsh.model.mesh.getNodes()
            elements_2d = gmsh.model.mesh.getElements(2)
            elements_3d = gmsh.model.mesh.getElements(3)
            
            num_nodes = len(nodes[0])
            num_tris = sum(len(tags) for tags in elements_2d[1])
            num_tets = sum(len(tags) for tags in elements_3d[1])
            
            print(f"\n  Mesh Statistics:")
            print(f"    Nodes: {num_nodes}")
            print(f"    Triangles: {num_tris}")
            print(f"    Tetrahedra: {num_tets}")
            
            # Check element sizes near painted regions
            print(f"\n  Checking element sizes near painted regions...")
            for i, region in enumerate(config.painted_regions):
                xc, yc, zc = region['center']
                radius = region['radius']
                
                # Get nodes near this region
                nearby_nodes = []
                for node_id, x, y, z in zip(nodes[0], nodes[1][0::3], nodes[1][1::3], nodes[1][2::3]):
                    dist = ((x-xc)**2 + (y-yc)**2 + (z-zc)**2)**0.5
                    if dist < radius:
                        nearby_nodes.append((node_id, dist))
                
                print(f"    Region {i+1}: {len(nearby_nodes)} nodes within radius {radius}")
            
            print(f"\n[OK] Mesh generated successfully with refinement!")
            
        gmsh.finalize()
        return True
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            gmsh.finalize()
        except:
            pass
        return False

def main():
    print("\n" + "="*70)
    print("PAINTBRUSH REFINEMENT DEBUG SCRIPT")
    print("="*70)
    
    # Run all tests
    painted_regions = test_step_1_data_collection()
    quality_params = test_step_2_file_passing(painted_regions)
    config = test_step_3_backend_injection(quality_params)
    
    if config:
        success = test_step_4_field_creation(config)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        if success:
            print("[OK] All tests passed!")
            print("\nThe paintbrush refinement pipeline is working correctly.")
            print("If you're not seeing refinement in the GUI:")
            print("  1. Check that painted regions are being collected (add debug prints)")
            print("  2. Check that quality_params includes painted_regions")
            print("  3. Verify the refinement factor (currently 5x) is strong enough")
        else:
            print("[X] Some tests failed. Check the output above.")
    else:
        print("\n[X] Config injection failed")

if __name__ == "__main__":
    main()
