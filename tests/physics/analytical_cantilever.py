"""
Analytical Cantilever Beam - Physics Validation Test
=====================================================

Classic strength of materials problem with known closed-form solution.
Used to validate FEA implementation, particularly element formulations.

Physics:
    Cantilever beam with end load:
    - Length L
    - Width b  
    - Height h
    - Young's modulus E
    - Load F at free end

Analytical Solutions:
    1. Tip deflection: δ = F·L³ / (3·E·I)  where I = b·h³/12
    2. Max stress: σ_max = M·c/I = (F·L)·(h/2) / I
    3. Stress distribution: Linear through thickness (tensile top, compressive bottom)

This test validates:
    - Node ordering (Tet4 vs Tet10)
    - Element stiffness formulation
    - Boundary condition application
"""

import gmsh
import numpy as np
from pathlib import Path

class CantileverBeamAnalytical:
    """
    Cantilever beam test case with analytical reference solution.
    """
    
    def __init__(self, length=1.0, width=0.1, height=0.1, 
                 E=70e9, nu=0.33, force=1000.0):
        """
        Parameters:
            length: Beam length (m)
            width: Beam width (m)
            height: Beam height (m)
            E: Young's modulus (Pa) - default is Aluminum
            nu: Poisson's ratio
            force: End load (N), negative for downward
        """
        self.L = length
        self.b = width
        self.h = height
        self.E = E
        self.nu = nu
        self.F = force
        
        # Second moment of area
        self.I = (self.b * self.h**3) / 12.0
        
    def tip_deflection(self) -> float:
        """
        Analytical tip deflection (m).
        δ = F·L³ / (3·E·I)
        """
        return (self.F * self.L**3) / (3 * self.E * self.I)
    
    def max_stress(self) -> float:
        """
        Maximum bending stress at fixed end (Pa).
        σ_max = M·c/I where M = F·L, c = h/2
        """
        M = abs(self.F) * self.L  # Bending moment at fixed end
        c = self.h / 2.0          # Distance to extreme fiber
        return (M * c) / self.I
    
    def stress_at(self, x: float, y: float, z: float) -> float:
        """
        Bending stress at any point (x, y, z) along beam.
        
        Coordinates:
            x: Along beam length (0 = fixed, L = free)
            y: Across width
            z: Through height (0 = bottom, h = top)
        
        Returns: Stress in Pa (positive = tension, negative = compression)
        """
        M_x = abs(self.F) * (self.L - x)  # Moment varies linearly
        c_z = z - (self.h / 2.0)          # Distance from neutral axis
        
        # σ = -M·z / I  (negative because positive z is tensile for negative F)
        return -(M_x * c_z) / self.I * np.sign(self.F)
    
    def generate_step_file(self, output_path: Path, mesh_size: float = 0.02):
        """
        Generate STEP geometry file for cantilever beam.
        
        Args:
            output_path: Where to save .step file
            mesh_size: Characteristic mesh size (m)
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        # Create box
        box = gmsh.model.occ.addBox(0, 0, 0, self.L, self.b, self.h)
        gmsh.model.occ.synchronize()
        
        # Get surfaces for BC application
        surfaces = gmsh.model.getEntities(2)
        
        # Fixed face (x=0)
        fixed_tags = []
        for dim, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(com[0]) < 1e-6:  # x ≈ 0
                fixed_tags.append(tag)
        
        # Create physical groups
        if fixed_tags:
            gmsh.model.addPhysicalGroup(2, fixed_tags, name="Fixed")
        
        gmsh.model.addPhysicalGroup(3, [box], name="Material_Aluminum")
        
        gmsh.model.occ.synchronize()
        
        # Export STEP
        gmsh.write(str(output_path))
        gmsh.finalize()
        
        return output_path
    
    def generate_config(self, output_path: Path, mesh_size: float = 0.02,
                       element_order: int = 1):
        """
        Generate JSON configuration for structural simulation.
        
        Args:
            output_path: Where to save .json config
            mesh_size: Mesh characteristic size (m)
            element_order: 1 for Tet4 (linear), 2 for Tet10 (quadratic)
        """
        import json
        
        config = {
            "version": "1.0",
            "job_name": f"Cantilever_Beam_{'Tet4' if element_order==1 else 'Tet10'}",
            "physics": {
                "simulation_type": "structural",
                "young_modulus": self.E,
                "poisson_ratio": self.nu,
                "density": 2700.0,  # kg/m³ (not critical for static)
                "gravity_load_g": 0.0,  # No gravity, pure bending load
                "material": "Test_Aluminum",
            },
            "meshing": {
                "second_order": element_order == 2,
                "mesh_size_multiplier": mesh_size / 0.05,  # Relative to default
            },
            # Point load will be applied via adapter (TODO: fix point load API)
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return output_path


def create_cantilever_test_case(output_dir: Path, element_order: int = 1):
    """
    Create complete test case: geometry + config + analytical reference.
    
    Args:
        output_dir: Directory to save files
        element_order: 1 for Tet4, 2 for Tet10
    
    Returns:
        Dict with 'step_file', 'config_file', 'analytical' (CantileverBeamAnalytical)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create analytical reference
    beam = CantileverBeamAnalytical(
        length=1.0,      # 1 meter
        width=0.1,       # 10 cm
        height=0.1,      # 10 cm  
        E=70e9,          # 70 GPa (Aluminum)
        nu=0.33,
        force=-1000.0    # 1 kN downward
    )
    
    # Generate files
    elem_type = "Tet4" if element_order == 1 else "Tet10"
    step_file = output_dir / f"cantilever_{elem_type}.step"
    config_file = output_dir / f"cantilever_{elem_type}.json"
    
    beam.generate_step_file(step_file, mesh_size=0.02)
    beam.generate_config(config_file, mesh_size=0.02, element_order=element_order)
    
    # Print analytical reference
    print(f"\n{'='*60}")
    print(f"Cantilever Beam Analytical Reference ({elem_type})")
    print(f"{'='*60}")
    print(f"Geometry: L={beam.L}m, b={beam.b}m, h={beam.h}m")
    print(f"Material: E={beam.E/1e9:.1f} GPa, nu={beam.nu}")
    print(f"Load:     F={beam.F:.1f} N (vertical)")
    print(f"\nAnalytical Solutions:")
    print(f"  Tip Deflection: delta = {beam.tip_deflection()*1000:.4f} mm")
    print(f"  Max Stress:     sigma = {beam.max_stress()/1e6:.2f} MPa") 
    print(f"  Location:       Fixed end, top fiber")
    print(f"{'='*60}\n")
    
    return {
        'step_file': step_file,
        'config_file': config_file,
        'analytical': beam,
    }


if __name__ == "__main__":
    # Create both Tet4 (baseline) and Tet10 (test) cases
    test_dir = Path(__file__).parent / "output" / "cantilever_validation"
    
    print("\n🔧 Generating Cantilever Beam Test Cases...")
    
    tet4_case = create_cantilever_test_case(test_dir, element_order=1)
    tet10_case = create_cantilever_test_case(test_dir, element_order=2)
    
    print(f"\n✅ Files created in: {test_dir}")
    print(f"   - Tet4 (baseline):  {tet4_case['step_file'].name}")
    print(f"   - Tet10 (suspect):  {tet10_case['step_file'].name}")
    print(f"\n📋 Next Steps:")
    print(f"   1. Run both simulations: python simops_worker.py <step_file> <config_file>")
    print(f"   2. Compare FEA results to analytical:")
    print(f"      - Tet4 tip deflection should be ~{tet4_case['analytical'].tip_deflection()*1000:.4f} mm")
    print(f"      - Tet10 tip deflection should be ~{tet10_case['analytical'].tip_deflection()*1000:.4f} mm")
    print(f"   3. If Tet10 is way off, node ordering is WRONG → fix permutation")
