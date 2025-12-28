import json
import os
from pathlib import Path

# Config
CAD_FILES = [
    r"cad_files\Cube.step",
    r"cad_files\Cylinder.step",
    r"cad_files\Airfoil.step",
    r"cad_files\RocketEngine.step",
    r"structural_test_env\L_bracket.step"
]

CONFIG_DIR = Path("configs_sweep")
JOBS_DIR = Path("jobs_queue")
OUTPUT_DIR = Path("output_sweep")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base Templates
# Calculate native CCX path
base_dir = Path(__file__).parent.parent.absolute()
ccx_native_path = base_dir / "calculix_native" / "CalculiX-2.23.0-win-x64" / "bin" / "ccx.exe"

# Fallback to wsl script if native not found (but we expect it to be there now)
ccx_path_str = str(ccx_native_path).replace("\\", "/")
if not ccx_native_path.exists():
    print(f"Warning: Native CCX not found at {ccx_native_path}. using batch script.")
    ccx_path_str = "ccx_wsl.bat"

TEMPLATES = {
    "structural": {
        "version": "1.0",
        "physics": {
            "simulation_type": "structural",
            "gravity_load_g": 1.0,
            "material": "Al6061-T6",
            "ccx_path": ccx_path_str
        },
        "meshing": {
            "mesh_size_multiplier": 1.0,
            "second_order": True
        }
    },
    "cfd": {
        "version": "1.0",
        "physics": {
            "simulation_type": "cfd",
            "inlet_velocity": 10.0,
            "ambient_temp_c": 25.0
        },
        "meshing": {
            "mesh_size_multiplier": 1.0,
            "num_layers": 3
        }
    },
    "thermal": {
        "version": "1.0",
        "physics": {
            "simulation_type": "thermal",
            "heat_load_watts": 10.0,
            "ambient_temp_c": 25.0,
            "ccx_path": ccx_path_str
        },
        "meshing": {
            "mesh_size_multiplier": 1.2
        }
    }
}

def setup():
    for cad_path_str in CAD_FILES:
        cad_path = Path(cad_path_str)
        job_base = cad_path.stem
        
        for sim_type, template in TEMPLATES.items():
            job_name = f"{job_base}_{sim_type}"
            
            # 1. Create Config
            config = template.copy()
            config["job_name"] = job_name
            config_file = CONFIG_DIR / f"{job_name}.json"
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # 2. Create Job Ticket
            job_ticket = {
                "intent": "run_sim",
                "args": {
                    "file_path": str(cad_path.absolute()),
                    "output_dir": str((OUTPUT_DIR / job_name).absolute()),
                    "config_path": str(config_file.absolute())
                }
            }
            
            job_file = JOBS_DIR / f"{job_name}.json"
            with open(job_file, 'w') as f:
                json.dump(job_ticket, f, indent=2)
                
            print(f"Generated job: {job_name}")

if __name__ == "__main__":
    setup()
