import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = r'C:\Users\markm\Downloads\MeshPackageLean'
sys.path.insert(0, PROJECT_ROOT)

from strategies.exhaustive_strategy import ExhaustiveMeshGenerator
from core.config import get_default_config

def verify():
    input_file = os.path.join(PROJECT_ROOT, "cad_files", "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP")
    output_file = os.path.join(PROJECT_ROOT, "integration_motherboard_result.msh")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run generate_core_sample.py first.")
        return

    print("=" * 60)
    print("INTEGRATION VERIFICATION RUN")
    print(f"Input: {input_file}")
    print("=" * 60)
    
    config = get_default_config()
    # Ensure parallel workers are enabled
    os.environ['MESH_MAX_WORKERS'] = '4'
    
    generator = ExhaustiveMeshGenerator(config)
    result = generator.generate_mesh(input_file, output_file)
    
    if result.success:
        print("\n[OK] VERIFICATION SUCCESSFUL!")
        print(f"Generated mesh: {output_file}")
    else:
        print("\n[X] VERIFICATION FAILED.")
        print(f"Error: {result.message}")

if __name__ == "__main__":
    verify()
