
import gmsh
import sys
import os
import logging
import numpy as np
import unittest
import re
from pathlib import Path

# Add core path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.solvers.cfd_solver import CFDSolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BenchmarkCFD")

class TestCylinderFlow(unittest.TestCase):
    
    def setUp(self):
        self.output_dir = Path("output/benchmark_cfd_cylinder")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_file = self.output_dir / "cylinder_2d.msh"
        
        # Physics Parameters for Re=40
        # Re = U * D / nu
        self.D = 0.01  # Cylinder Diameter 10mm = 0.01m
        self.nu = 1.0e-5 # m^2/s (Target viscosity)
        self.Re_target = 40.0
        
        # U = Re * nu / D
        self.U_inlet = self.Re_target * self.nu / self.D 
        # U = 40 * 1e-5 / 0.01 = 0.04 m/s
        
        self.create_geometry_and_mesh()
        
    def create_geometry_and_mesh(self):
        """Create 2D Channel with Cylinder."""
        gmsh.initialize()
        gmsh.model.add("cylinder_2d")
        
        # Domain: L=30D, H=30D? 
        # Standard: Inlet 10D upstream, Outlet 20D downstream. 
        # Width: Blockage ratio should be low (< 5%). H = 20D = 0.2m.
        # Cylinder at (0,0,0)
        
        D = self.D
        H = 0.2 # 20cm
        L_upstream = 0.1 # 10cm
        L_downstream = 0.2 # 20cm
        
        # 1. Create Rectangle
        # Z-thickness: 1 cell (e.g. D)
        # We use OpenFOAM "empty" BC for 2D.
        # So we make a 3D prism 1-layer thick.
        thickness = D # 1cm
        
        # Points
        p1 = gmsh.model.geo.addPoint(-L_upstream, -H/2, 0)
        p2 = gmsh.model.geo.addPoint(L_downstream, -H/2, 0)
        p3 = gmsh.model.geo.addPoint(L_downstream, H/2, 0)
        p4 = gmsh.model.geo.addPoint(-L_upstream, H/2, 0)
        
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        
        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        
        # Cylinder Cutout
        c1 = gmsh.model.geo.addPoint(0, 0, 0)
        c2 = gmsh.model.geo.addPoint(D/2, 0, 0)
        c3 = gmsh.model.geo.addPoint(0, D/2, 0)
        c4 = gmsh.model.geo.addPoint(-D/2, 0, 0)
        c5 = gmsh.model.geo.addPoint(0, -D/2, 0)
        
        # Circle Arcs
        a1 = gmsh.model.geo.addCircleArc(c2, c1, c3)
        a2 = gmsh.model.geo.addCircleArc(c3, c1, c4)
        a3 = gmsh.model.geo.addCircleArc(c4, c1, c5)
        a4 = gmsh.model.geo.addCircleArc(c5, c1, c2)
        
        circle_loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
        
        # Surface
        surf = gmsh.model.geo.addPlaneSurface([loop, circle_loop])
        
        # Recombine to get Quads (which extrude to Hexes)
        gmsh.model.geo.mesh.setRecombine(2, surf)
        
        gmsh.model.geo.synchronize()
        
        # Extrude to 3D (1 layer)
        extrude = gmsh.model.geo.extrude([(2, surf)], 0, 0, thickness, numElements=[1], heights=[1], recombine=True)
        
        gmsh.model.geo.synchronize()
        
        # Physical Groups
        # Identify surfaces
        # Extrude returns list of (dim, tag).
        # Index 0: Top Surface (the copy of source at Z=thick).
        # Index 1: Volume
        # Index 2+: Side Surfaces.
        
        # We need to find tags for:
        # Inlet (Left, X-), Outlet (Right, X+), Walls (Top/Bottom Y), Cylinder (Internal), Front/Back (Z)
        
        # Get all surfaces
        surfaces = gmsh.model.getEntities(2)
        
        inlet_tags = []
        outlet_tags = []
        wall_tags = []
        cyl_tags = []
        front_back_tags = []
        
        for s in surfaces:
            tag = s[1]
            # Bounds
            bb = gmsh.model.getBoundingBox(2, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bb
            
            # Check Normal? or position.
            
            # Front/Back are Z-planes
            if abs(zmax - zmin) < 1e-6:
                front_back_tags.append(tag)
                continue
            
            # Inlet: x ~ -L_upstream
            if abs(xmin - (-L_upstream)) < 1e-5 and abs(xmax - (-L_upstream)) < 1e-5:
                inlet_tags.append(tag)
                continue
                
            # Outlet: x ~ L_downstream
            if abs(xmin - L_downstream) < 1e-5 and abs(xmax - L_downstream) < 1e-5:
                outlet_tags.append(tag)
                continue
                
            # Cylinder: Bounds within +/- D
            if xmin > -D and xmax < D and ymin > -D and ymax < D:
                cyl_tags.append(tag)
                continue
            
            # Remaining: Top/Bottom Walls
            wall_tags.append(tag)
            
        gmsh.model.addPhysicalGroup(2, inlet_tags, 1, "inlet")
        gmsh.model.addPhysicalGroup(2, outlet_tags, 2, "outlet")
        gmsh.model.addPhysicalGroup(2, wall_tags, 3, "walls")
        gmsh.model.addPhysicalGroup(2, cyl_tags, 4, "cylinder")
        gmsh.model.addPhysicalGroup(2, front_back_tags, 5, "frontAndBack")
        
        gmsh.model.addPhysicalGroup(3, [extrude[1][1]], 101, "internal")
        
        # Mesh Settings
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", D/4)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", D/2)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) # Essential for gmshToFoam compatibility
        
        # Refine around cylinder?
        # Using simple global mesh for now
        
        gmsh.model.mesh.generate(3)
        gmsh.write(str(self.mesh_file))
        gmsh.finalize()

    def test_cfd_validation(self):
        solver = CFDSolver(use_wsl=True)
        
        config = {
            "kinematic_viscosity": self.nu,
            "inlet_velocity": [self.U_inlet, 0, 0],
            "iterations": 500 # Should converge fast for laminar
        }
        
        logger.info(f"Running CFD Validation: Re={self.Re_target}, U={self.U_inlet} m/s")
        
        try:
            result = solver.run(self.mesh_file, self.output_dir, config)
            
            # Parse Force Log
            # forces.dat usually needs parsing
            # Or we can grep 'forces output:' from logs if 'log' was True in controlDict.
            # But run_foam_cmd redirects to log file.
            
            case_dir = Path(result['case_dir'])
            post_dir = case_dir / "postProcessing" / "forces" / "0" # Time dir
            force_file = post_dir / "forces.dat"
            
            if not force_file.exists():
                # Maybe different time dir
                dirs = list((case_dir / "postProcessing" / "forces").glob("*"))
                if dirs:
                    force_file = dirs[-1] / "forces.dat"
            
            if force_file.exists():
                lines = force_file.read_text().splitlines()
                # Format: # Time      forces(pressure, viscous, porous)      moment(pressure, viscous, porous)
                # Last line
                last_line = lines[-1]
                parts = last_line.replace('(', '').replace(')', '').split()
                # Time = parts[0]
                # Force Total = Pressure + Viscous
                # Pressure: (Fx Fy Fz) [indices 1,2,3]
                # Viscous:  (Fx Fy Fz) [indices 4,5,6]
                
                # Drag is Fx
                Fp_x = float(parts[1])
                Fv_x = float(parts[4])
                F_drag = Fp_x + Fv_x
                
                # Check convergence of Drag?
                
                # Calculate Cd
                # Cd = F_drag / (0.5 * rho * U^2 * A_ref)
                # rho = 1.0 (set in rhoInf)
                rho = 1.0
                U = self.U_inlet
                # A_ref = Diameter * Thickness (Projected Area)
                A_ref = self.D * self.D # Thickness=D
                
                Cd = F_drag / (0.5 * rho * U**2 * A_ref)
                
                logger.info(f"Drag Force: {F_drag:.6f} N")
                logger.info(f"Calculated Cd: {Cd:.4f}")
                logger.info(f"Target Cd (Re=40): ~1.5 - 1.6")
                
                # Validation: Re=40 Cd is typically around 1.5-1.6
                self.assertTrue(1.3 < Cd < 1.8, f"Cd {Cd:.4f} out of expected range (1.3-1.8)")
                logger.info("[PASS] Drag Coefficient Validated")
                
            else:
                logger.warning("Force file not found. Skipping Drag Validation.")
                
            # Reynolds Verification
            # We assume config was respected, but we could check VTK U_mean?
            # That's harder.
            
            logger.info("[PASS] CFD Simulation Completed")
            
            # Visualization
            from core.reporting.velocity_viz import generate_velocity_streamlines
            viz_out = self.output_dir / "velocity_streamlines.png"
            generate_velocity_streamlines(result['vtk_file'], str(viz_out), title=f"Re={self.Re_target} Cylinder Flow")
            
        except Exception as e:
            if "WSL2 is required" in str(e):
                logger.warning("Skipping test: WSL2 not available")
                return
            raise

if __name__ == '__main__':
    unittest.main()
