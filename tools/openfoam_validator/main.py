import argparse
import sys
import logging
import json
from pathlib import Path
from .validator import OpenFOAMValidator

def main():
    parser = argparse.ArgumentParser(description="Validate an OpenFOAM case.")
    parser.add_argument("case_path", help="Path to the OpenFOAM case directory")
    parser.add_argument("--openfoam-path", help="Custom path to OpenFOAM installation")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    validator = OpenFOAMValidator(args.case_path, openfoam_path=args.openfoam_path)
    
    print(f"Validating case: {args.case_path}...")
    results = validator.validate()
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Human readable output
        print("-" * 40)
        print(f"Structure Check:   {results['structure']}")
        print(f"Environment Check: {results['environment']}")
        
        if "checkMesh" in results:
            print(f"checkMesh:         {results['checkMesh']}")
            
        if "solver" in results:
             print(f"Solver Detected:   {results['solver']}")
             
        if results["errors"]:
            print("\nERRORS:")
            for err in results["errors"]:
                print(f"  - {err}")
                
        if results["warnings"]:
            print("\nWARNINGS:")
            for warn in results["warnings"]:
                print(f"  - {warn}")
        print("-" * 40)
        
    # Exit code
    if results['structure'] == 'FAIL' or results['environment'] == 'FAIL' or results.get('checkMesh') == 'FAIL':
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
