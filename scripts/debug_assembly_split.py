
import sys
import os
import multiprocessing
from pathlib import Path

# Add project root to path
# scripts/ -> root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.assembly_strategy import AssemblyMeshGenerator
from core.config import Config

def reproduce():
    # Setup
    print(f"Current working dir: {os.getcwd()}")
    print(f"Project root calculated: {project_root}")
    print(f"Project root exists: {project_root.exists()}")
    if project_root.exists():
        print(f"Root dir contents: {[x.name for x in project_root.glob('*')]}")
        
    input_file = str(project_root / "core_sample.step")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        # Try looking in cad_files
        cad_file = project_root / "cad_files" / "00-CM240411_BIANCA-C00_HEATER_BOAR_0704A1.STEP"
        if cad_file.exists():
            input_file = str(cad_file)
            print(f"Found alternative in cad_files: {input_file}")
        else:
             return

    print(f"Testing with {input_file}")

    # Initialize Generator
    config = Config()
    generator = AssemblyMeshGenerator(config)

    # Manually invoke the split logic
    print("Starting split...")
    try:
        stls = generator._split_assembly_to_stls(input_file)
        print(f"Split result: {len(stls)} files")
        if hasattr(generator, 'toxic_volumes'):
             print(f"Toxic volumes: {len(generator.toxic_volumes)}")
    except Exception as e:
        print(f"Split failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    reproduce()
