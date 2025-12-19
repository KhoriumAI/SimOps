#!/usr/bin/env python3
"""
SimOps Sidecar Wizard
=====================
Interactive CLI to generate simulation configuration sidecars (*.json).
Use this to quickly create "smart templates" or custom run configurations.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, List, Optional
import os

# Locate Core
CURRENT_DIR = Path(__file__).parent
REPO_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from core.mat_lib import MaterialLibrary
    from core.schemas.config_schema import SimulationConfig, PhysicsConfig, MeshingConfig
except ImportError as e:
    print(f"Error importing core modules: {e}")
    sys.exit(1)

# ANSI Colors
C_RESET = "\033[0m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BOLD = "\033[1m"

def print_header():
    print(f"\n{C_BOLD}{C_CYAN}" + "="*50)
    print("   SimOps Sidecar Generator (SimWizard v1.0)")
    print("="*50 + f"{C_RESET}\n")

def prompt(question: str, default: Any = None, type_cast: type = str) -> Any:
    """Robust prompt with default and type conversion"""
    
    # Format default hint
    if default is not None:
        if isinstance(default, bool):
            hint = "[Y/n]" if default else "[y/N]"
        else:
            hint = f"[{default}]"
    else:
        hint = ""
        
    prompt_text = f"{C_GREEN}? {question}{C_RESET} {C_CYAN}{hint}{C_RESET}: "
    
    while True:
        user_input = input(prompt_text).strip()
        
        # Handle Empty (Default)
        if not user_input:
            if default is not None:
                return default
            else:
                print(f"{C_RED}  Error: Value required.{C_RESET}")
                continue
                
        # Handle Bool
        if type_cast == bool:
            lower = user_input.lower()
            if lower in ('y', 'yes', 'true', '1'):
                return True
            elif lower in ('n', 'no', 'false', '0'):
                return False
            else:
                print(f"{C_RED}  Error: Please enter Y or N.{C_RESET}")
                continue
                
        # Handle Numeric
        try:
            return type_cast(user_input)
        except ValueError:
            print(f"{C_RED}  Error: Invalid {type_cast.__name__} value.{C_RESET}")
            continue

def get_material_choice() -> str:
    """Prompt for material, showing list"""
    lib = MaterialLibrary.get_instance()
    mats = sorted(lib.materials.keys())
    
    print(f"\n{C_BOLD}Available Materials:{C_RESET}")
    for i, m in enumerate(mats, 1):
        print(f"  {i}. {m}")
    
    while True:
        val = prompt("Select Material (Name or Number)", default="Aluminum")
        
        # Check by Number
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(mats):
                return mats[idx]
        
        # Check by Name
        if val in mats:
            return val
            
        print(f"{C_RED}  Material '{val}' not found in library.{C_RESET}")
        if prompt("Use custom string anyway?", default=False, type_cast=bool):
            return val

def run_wizard(target_cad: Optional[str] = None):
    print_header()
    
    defaults = SimulationConfig()
    
    # 1. File Handling
    if not target_cad:
        target_cad = prompt("Target CAD File (or just Job Name)", default="simulation")
    
    path = Path(target_cad)
    if path.suffix.lower() not in ['.step', '.stp', '.brep', '']: 
        # Just a name?
        job_name = path.stem
        json_path = path.with_suffix('.json') if path.suffix else Path(f"{job_name}.json")
    else:
        job_name = path.stem
        json_path = path.with_suffix('.json')
        
    print(f"Configuring for Job: {C_BOLD}{job_name}{C_RESET}")
    print(f"Output File:       {C_BOLD}{json_path}{C_RESET}")
    
    # 2. Physics
    print(f"\n{C_YELLOW}--- Physics Settings ---{C_RESET}")
    is_composite = prompt("Enable Anisotropic Readiness (Composite Template)?", default=False, type_cast=bool)
    
    material = "Aluminum"
    material_defs = {}
    
    if is_composite:
        material = "casing_material"
        material_defs = {
            "casing_material": {
                "type": "Orthotropic",
                "name": "Toray_T700_CFRP",
                "properties": {
                    "conductivity_matrix": [4.5, 0, 0, 0, 4.5, 0, 0, 0, 0.8],
                    "density": 1600,
                    "specific_heat": 1200
                }
            }
        }
        print(f"  {C_GREEN}[Ready]{C_RESET} Placeholder T700 CFRP matrix loaded into 'material_definitions'.")
    else:
        material = get_material_choice()
    
    is_transient = prompt("Enable Transient Analysis (Heating over time)?", default=True, type_cast=bool)
    
    duration = 60.0
    time_step = 2.0
    if is_transient:
        duration = prompt("Duration (seconds)", default=60.0, type_cast=float)
        time_step = prompt("Time Step (seconds)", default=duration/30.0, type_cast=float)
    
    h_coeff = prompt("Convection Coefficient (h) [0=None, 5-25=Air, 100+=Water]", default=25.0, type_cast=float)
    
    # Boundaries
    print(f"\n{C_YELLOW}--- Boundary Conditions ---{C_RESET}")
    if h_coeff > 0:
        print("  (Tip: If modeling a fin, you usually want the Tip to float (Convection Only))")
        fix_cold = prompt("Clamp Cold Boundary (Fixed T)?", default=False, type_cast=bool)
    else:
        fix_cold = True # Conduction needs sink usually
        
    heat_watts = prompt("Heat Source Power (Watts) [or Temp if Adapter maps it]", default=50.0, type_cast=float)
    
    # 3. Meshing
    print(f"\n{C_YELLOW}--- Quality / Meshing ---{C_RESET}")
    quality_mode = prompt("Quality Mode (1=Fast/Draft, 2=Standard, 3=HighFi)", default=2, type_cast=int)
    
    mesh_mult = 1.0
    second_order = False
    validate = False
    
    if quality_mode == 1:
        mesh_mult = 2.0 # Coarser
    elif quality_mode == 3:
        mesh_mult = 0.5 # Finer
        second_order = prompt("Use Second Order Elements (Tet10)? (Slower but accurate)", default=True, type_cast=bool)
        validate = prompt("Run Grid Convergence (3x runs)?", default=False, type_cast=bool)
        
    # 4. Construct Config
    config = SimulationConfig(
        job_name=job_name,
        physics=PhysicsConfig(
            material=material,
            heat_load_watts=heat_watts,
            convection_coeff=h_coeff,
            transient=is_transient,
            duration=duration,
            time_step=time_step,
            fix_cold_boundary=fix_cold
        ),
        meshing=MeshingConfig(
            mesh_size_multiplier=mesh_mult,
            second_order=second_order
        ),
        material_definitions=material_defs,
        validate_mesh=validate
    )
    
    # 5. Write
    print(f"\n{C_YELLOW}--- Review ---{C_RESET}")
    print(json.dumps(config.dict(), indent=2))
    
    if prompt("Save this configuration?", default=True, type_cast=bool):
        with open(json_path, 'w') as f:
            f.write(json.dumps(config.dict(), indent=2))
        print(f"\n{C_GREEN}Success! Configuration saved to {json_path}{C_RESET}")
        print("Drop this file alongside your CAD file to run with these settings.")
    else:
        print("\nCancelled.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SimOps Sidecar JSON")
    parser.add_argument("cad_file", nargs="?", help="Optional CAD file path to template from")
    args = parser.parse_args()
    
    try:
        run_wizard(args.cad_file)
    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(0)
