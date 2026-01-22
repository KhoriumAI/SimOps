"""
CAD to Mesh Converter (Skill)
=============================
Simple utility to convert STEP files to VTK/INP meshes using Gmsh.
Refactored from scripts/step_to_mesh.py.

Usage:
    python khorium_skills/toolbox/cad_to_mesh.py input.step output.vtk
"""

import gmsh
import sys
import os
import time
import argparse

def mesh_step_file(step_file: str, output_file: str, verbose: bool = True) -> bool:
    """
    Meshes a STEP file and saves to output_file.
    Returns True if successful.
    """
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] --- STARTING MESHING: {step_file} ---")
    
    try:
        if not gmsh.isInitialized():
            gmsh.initialize()
        
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("MeshSkill")

        # 1. LOAD STEP
        if verbose: print(f"[{time.strftime('%H:%M:%S')}] Importing STEP file...")
        try:
            gmsh.model.occ.importShapes(step_file)
            gmsh.model.occ.synchronize()
        except Exception as e:
            print(f"[X] FATAL: Could not import STEP. {e}")
            return False

        # 2. MESH SETUP
        if verbose: print(f"[{time.strftime('%H:%M:%S')}] Configuring Mesh options...")
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 5.0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) # 1=Delaunay (Robust)

        # 3. GENERATE MESH
        if verbose: print(f"[{time.strftime('%H:%M:%S')}] Meshing Volume...")
        gmsh.model.mesh.generate(3)

        # 4. EXPORT
        if verbose: print(f"[{time.strftime('%H:%M:%S')}] Exporting to {output_file}...")
        gmsh.write(output_file)
        
        # Also export INP if VTK
        if output_file.lower().endswith('.vtk'):
            inp_path = output_file.replace(".vtk", ".inp")
            gmsh.write(inp_path)
            if verbose: print(f"[{time.strftime('%H:%M:%S')}] -> Also exported to {inp_path}")

        gmsh.finalize()
        return True

    except Exception as e:
        print(f"[X] Mesh generation failed: {e}")
        try:
            gmsh.finalize()
        except:
            pass
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert STEP to Mesh")
    parser.add_argument("step_file", help="Input STEP file")
    parser.add_argument("output_file", help="Output Mesh file (vtk/msh/inp)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.step_file):
        print(f"Error: File not found: {args.step_file}")
        sys.exit(1)
        
    success = mesh_step_file(args.step_file, args.output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
