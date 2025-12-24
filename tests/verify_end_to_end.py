
import os
import sys
import shutil
import time
import json
import gmsh
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"

def generate_cylinder_step(output_path: Path, radius=0.05, height=0.2):
    try:
        gmsh.initialize()
        gmsh.model.add("Cylinder")
        # addCylinder(x, y, z, dx, dy, dz, r)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
        gmsh.model.occ.synchronize()
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()

def create_cfd_test_files():
    job_name = "Test_E2E_CFD"
    step_file = INPUT_DIR / f"{job_name}.step"
    json_file = INPUT_DIR / f"{job_name}.json"
    
    # Generate Geometry
    print(f"Generating {step_file}...")
    generate_cylinder_step(step_file)
    
    # Generate Config
    config = {
        "version": "1.0",
        "job_name": job_name,
        "physics": {
            "simulation_type": "cfd",
            "inlet_velocity": 5.0,
            "kinematic_viscosity": 1.5e-5,
            "material": "Air"
        },
        "meshing": { "mesh_size_multiplier": 2.0 }
    }
    json_file.write_text(json.dumps(config, indent=2))
    print(f"Created {json_file}")
    return job_name

def create_structural_test_files():
    job_name = "Test_E2E_Struct"
    step_file = INPUT_DIR / f"{job_name}.step"
    json_file = INPUT_DIR / f"{job_name}.json"
    
    # Generate Geometry
    print(f"Generating {step_file}...")
    generate_cylinder_step(step_file)
    
    # Generate Config
    config = {
        "version": "1.0",
        "job_name": job_name,
        "physics": {
            "simulation_type": "structural",
            "gravity_load_g": 9.81,
            "material": "Steel",
            "youngs_modulus": 200e9,
            "poissons_ratio": 0.3,
            "density": 7850
        },
        "meshing": { "mesh_size_multiplier": 2.0 }
    }
    json_file.write_text(json.dumps(config, indent=2))
    print(f"Created {json_file}")
    return job_name

def main():
    print("========================================")
    print(" END-TO-END VERIFICATION")
    print("========================================")
    print(f"Input Dir: {INPUT_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    
    # Ensure dirs exist
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Structural Test
    struct_job = create_structural_test_files()
    
    # 2. CFD Test
    cfd_job = create_cfd_test_files()
    
    print("\nWaiting for results (timeout 120s)...")
    
    # Poll for results
    timeout = 120
    start_time = time.time()
    
    struct_done = False
    cfd_done = False
    
    while time.time() - start_time < timeout:
        # Check output dir contents
        files = list(OUTPUT_DIR.glob("*"))
        
        # Check for Reports
        struct_report = OUTPUT_DIR / f"{struct_job}_structural_report.pdf"
        # Not sure exact name for CFD report, typically job_name_report.pdf or similar
        # Based on simops_worker.py: pdf_file = pdf_gen.generate(job_name=job_name...) => job_name + "_cfd_report.pdf"?
        # Let's search broadly
        cfd_report_candidates = list(OUTPUT_DIR.glob(f"{cfd_job}*report.pdf"))
        
        if not struct_done and struct_report.exists():
            print(f"[OK] Structural Report Found: {struct_report.name}")
            struct_done = True
            
        if not cfd_done and cfd_report_candidates:
            print(f"[OK] CFD Report Found: {cfd_report_candidates[0].name}")
            cfd_done = True
            
        if struct_done and cfd_done:
            print("\nSUCCESS: Both pipelines produced reports!")
            sys.exit(0)
            
        time.sleep(2)
        
    print("\nTIMEOUT waiting for results.")
    if not struct_done: print("[FAILED] Structural report missing")
    if not cfd_done: print("[FAILED] CFD report missing")
    sys.exit(1)

if __name__ == "__main__":
    main()
