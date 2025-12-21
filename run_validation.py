
from pathlib import Path
from core.validation.runner import ValidationRunner
from core.validation.cases.rod_1d import Rod1DCase
from core.validation.cases.surface_flux_slab import SurfaceFluxSlabCase
from core.validation.cases.volumetric_source_block import VolumetricSourceBlockCase
import logging

def main():
    logging.basicConfig(level=logging.INFO)
    runner = ValidationRunner(Path("validation_results"))
    
    cases = [
        Rod1DCase,
        SurfaceFluxSlabCase,
        VolumetricSourceBlockCase
    ]
    
    success = runner.run_suite(cases)
    
    if success:
        print("\nAll Validation Tests PASSED")
        exit(0)
    else:
        print("\nSome Validation Tests FAILED")
        exit(1)

if __name__ == "__main__":
    main()
