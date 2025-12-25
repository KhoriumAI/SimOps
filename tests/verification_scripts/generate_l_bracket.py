"""
L-Bracket Test Geometry Generator
==================================

Creates a simple L-bracket STEP file for structural analysis verification.
The L-bracket is a common drone arm mounting configuration.

Geometry:
- Overall: 50mm x 50mm x 10mm L-shape
- Material: Al6061-T6 (implied)
- Expected behavior under 10G Z-load:
  - Stress concentration at inner corner
  - Low stress at free ends
"""

import sys
from pathlib import Path

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


def create_l_bracket_step(output_file: str = "L_bracket.step"):
    """
    Create a simple L-bracket STEP file.
    
    Dimensions (mm):
    - Horizontal arm: 50mm x 20mm x 10mm
    - Vertical arm: 20mm x 50mm x 10mm
    - Fillet radius at corner: 3mm
    """
    if not GMSH_AVAILABLE:
        print("Error: Gmsh is required to create STEP files")
        return False
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("L_bracket")
    
    # Use OCC kernel for CAD operations
    occ = gmsh.model.occ
    
    # Dimensions in meters (Gmsh default)
    # We'll use mm scale for easier visualization, then it's up to solver to interpret
    scale = 0.001  # mm to m
    
    # L-bracket dimensions (in mm, converted to m)
    arm_length = 50 * scale
    arm_width = 20 * scale
    thickness = 10 * scale
    fillet_radius = 3 * scale
    
    # Create horizontal arm (lying along X)
    box1 = occ.addBox(0, 0, 0, arm_length, arm_width, thickness)
    
    # Create vertical arm (along Y)
    box2 = occ.addBox(0, 0, 0, arm_width, arm_length, thickness)
    
    # Fuse the two boxes
    fused = occ.fuse([(3, box1)], [(3, box2)])
    
    # Get the fused volume
    occ.synchronize()
    
    volumes = gmsh.model.getEntities(dim=3)
    if volumes:
        vol_tag = volumes[0][1]
        
        # Try to add fillet to inner edges
        # Find edges at the inner corner
        try:
            edges = gmsh.model.getBoundary([(3, vol_tag)], combined=False, oriented=False, recursive=True)
            edge_tags = [e[1] for e in edges if e[0] == 1]
            
            # Fillet all short edges (the corner edges)
            if edge_tags and fillet_radius > 0:
                # Only fillet if it won't fail
                try:
                    occ.fillet([vol_tag], edge_tags[:4], [fillet_radius])
                except:
                    pass  # Skip fillet if it fails
        except:
            pass  # Continue without fillet
    
    occ.synchronize()
    
    # Export to STEP
    gmsh.write(output_file)
    
    # Get some stats
    volumes = gmsh.model.getEntities(dim=3)
    surfaces = gmsh.model.getEntities(dim=2)
    
    print(f"Created L-bracket: {output_file}")
    print(f"  Volumes: {len(volumes)}")
    print(f"  Surfaces: {len(surfaces)}")
    
    gmsh.finalize()
    return True


def create_simple_l_bracket_step(output_file: str = "L_bracket.step"):
    """
    Create a simpler L-bracket without fillets (more robust).
    """
    if not GMSH_AVAILABLE:
        print("Error: Gmsh is required to create STEP files")
        return False
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("L_bracket_simple")
    
    occ = gmsh.model.occ
    
    # Dimensions in mm (Gmsh works in any unit, we document as mm)
    # Using raw values - solver will interpret units
    
    # L-bracket as extrusion of L-shaped cross-section
    # Create 2D L-shape first
    
    # Points for L-shape cross-section (in XY plane)
    # Going counter-clockwise
    p1 = occ.addPoint(0, 0, 0)      # Origin
    p2 = occ.addPoint(50, 0, 0)     # Bottom right
    p3 = occ.addPoint(50, 20, 0)    # Right notch
    p4 = occ.addPoint(20, 20, 0)    # Inner corner
    p5 = occ.addPoint(20, 50, 0)    # Top left notch
    p6 = occ.addPoint(0, 50, 0)     # Top left
    
    # Lines
    l1 = occ.addLine(p1, p2)
    l2 = occ.addLine(p2, p3)
    l3 = occ.addLine(p3, p4)
    l4 = occ.addLine(p4, p5)
    l5 = occ.addLine(p5, p6)
    l6 = occ.addLine(p6, p1)
    
    # Wire and surface
    wire = occ.addCurveLoop([l1, l2, l3, l4, l5, l6])
    surface = occ.addPlaneSurface([wire])
    
    # Extrude to create 3D volume (10mm thickness in Z)
    extrusion = occ.extrude([(2, surface)], 0, 0, 10)
    
    occ.synchronize()
    
    # Export to STEP
    gmsh.write(output_file)
    
    # Get stats
    volumes = gmsh.model.getEntities(dim=3)
    surfaces = gmsh.model.getEntities(dim=2)
    
    print(f"Created simple L-bracket: {output_file}")
    print(f"  Dimensions: 50x50x10 mm (L-shape)")
    print(f"  Volumes: {len(volumes)}")
    print(f"  Surfaces: {len(surfaces)}")
    
    gmsh.finalize()
    return True


def main():
    """Generate L-bracket STEP file"""
    output_dir = Path(__file__).parent.parent.parent / "cad_files"
    output_file = output_dir / "L_bracket.step"
    
    print("Generating L-bracket test geometry...")
    success = create_simple_l_bracket_step(str(output_file))
    
    if success:
        print(f"\n✓ L-bracket STEP file created: {output_file}")
    else:
        print("\n✗ Failed to create L-bracket")
        sys.exit(1)


if __name__ == "__main__":
    main()
