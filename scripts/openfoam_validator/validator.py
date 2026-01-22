import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from . import utils

logger = logging.getLogger(__name__)

class OpenFOAMValidator:
    def __init__(self, case_path: str, openfoam_path: Optional[str] = None):
        self.case_path = Path(case_path).resolve()
        
        # Smart path resolution: if not found, try looking in parent (project root)
        if not self.case_path.exists():
            # assuming we are in scripts/ or tools/, try ../path
            alt_path = Path(case_path)
            if not alt_path.is_absolute():
                 # try resolving relative to parent of CWD
                 parent_resolve = (Path.cwd().parent / case_path).resolve()
                 if parent_resolve.exists():
                     self.case_path = parent_resolve
        self.openfoam_path = openfoam_path
        self.results = {
            "structure": "PENDING",
            "environment": "PENDING",
            "solver_check": "PENDING",
            "errors": [],
            "warnings": []
        }

    def validate(self) -> Dict[str, Any]:
        """Run all validation steps."""
        self._check_structure()
        
        # Only proceed to env check if structure is kinda okay (controlDict exists)
        if self.results["structure"] != "FAIL":
            self._check_environment()
            
        if self.results["structure"] != "FAIL" and self.results["environment"] != "FAIL":
            self._run_dry_run()
            
        return self.results

    def _check_structure(self):
        """Check for essential directories and files."""
        if not self.case_path.exists():
            self.results["structure"] = "FAIL"
            self.results["errors"].append(f"Case directory not found: {self.case_path}")
            return

        required_dirs = ["system", "constant", "0"]
        required_files = [("system", "controlDict"), ("system", "fvSchemes"), ("system", "fvSolution")]
        
        missing_dirs = []
        for d in required_dirs:
            if not (self.case_path / d).is_dir():
                missing_dirs.append(d)
                
        if missing_dirs:
            self.results["structure"] = "FAIL"
            self.results["errors"].append(f"Missing required directories: {missing_dirs}")
            return

        missing_files = []
        for d, f in required_files:
            if not (self.case_path / d / f).exists():
                missing_files.append(str(Path(d) / f))
                
        if missing_files:
            self.results["structure"] = "FAIL"
            self.results["errors"].append(f"Missing required files: {missing_files}")
        else:
            self.results["structure"] = "PASS"

    def _check_environment(self):
        """Check if OpenFOAM is runnable."""
        avail = utils.check_openfoam_availability(self.openfoam_path)
        if not avail['native'] and not avail['wsl']:
            self.results["environment"] = "FAIL"
            self.results["errors"].append("OpenFOAM executables (foamList/simpleFoam) not found in PATH or WSL.")
        else:
            self.results["environment"] = "PASS (WSL)" if avail['wsl'] and not avail['native'] else "PASS (Native)"
            self.env_avail = avail
            if not self.openfoam_path and avail.get('wsl_path'):
                self.openfoam_path = avail['wsl_path']

    def _run_dry_run(self):
        """Try to run the solver (dry run) or checkMesh."""
        prefix = utils.get_openfoam_cmd_prefix(self.env_avail)
        
        # 1. Run checkMesh
        self.results["checkMesh"] = "PENDING"
        
        if len(prefix) > 0 and prefix[0] == 'wsl':
             wsl_path = self._to_wsl_path(self.case_path)
             bash_cmd = "checkMesh"
             if self.openfoam_path:
                 # Prefer sourcing the bashrc for full environment (libs + path)
                 bashrc_path = f"{self.openfoam_path}/etc/bashrc"
                 # We check if it exists in WSL before trying to source it
                 bash_cmd = f"if [ -f {bashrc_path} ]; then source {bashrc_path}; else export PATH=\"{self.openfoam_path}/bin:$PATH\"; fi && checkMesh"
             cmd = ['wsl', 'bash', '-c', f"cd '{wsl_path}' && {bash_cmd}"]
        else:
             env = os.environ.copy()
             if self.openfoam_path:
                 env["PATH"] = str(Path(self.openfoam_path) / "bin") + os.pathsep + env["PATH"]
             cmd = ["checkMesh", "-case", str(self.case_path)]
             # Note: run_command doesn't currently support passing custom env. 
             # For native, we might need to adjust utils.run_command or just use absolute path.
             if self.openfoam_path:
                 checkmesh_path = Path(self.openfoam_path) / "bin" / "checkMesh"
                 if checkmesh_path.exists():
                     cmd = [str(checkmesh_path), "-case", str(self.case_path)]

        code, stdout, stderr = utils.run_command(cmd)

        if code != 0:
            self.results["checkMesh"] = "FAIL"
            self.results["errors"].append(f"checkMesh failed:\n{stderr}\n{stdout[-500:]}") # Last 500 chars
            # If checkMesh fails, solver likely will too, but we can try basic solver help
        else:
            self.results["checkMesh"] = "PASS"

        # 2. Run Solver (dry run if possible, often just running it for 1 sec is the test)
        solver = utils.find_solver(self.case_path)
        if not solver:
            self.results["warnings"].append("Could not determine solver from controlDict. Skipping solver run.")
            return

        self.results["solver"] = solver
        
        # We can try running the solver for 1 iteration or just see if it starts and fails on license/libs
        # There isn't a generic "dry-run" flag for all solvers, but running it usually fails fast if setup is wrong.
        # We will stop it after a short timeout or rely on error code.
        
        # Note: Running solver might rewrite files (time directories). 
        # Ideally we should run this in a temp copy or be careful.
        # For a "validator", maybe we shouldn't modify the dir.
        # safer to just check if `solver -help` works or `solver -dry-run` (if partially supported in newer versions)
        # Standard OpenFOAM doesn't have a universal dry-run. 
        # Let's try running it but capture immediately. 
        # Actually, let's just run `solver -validate`? No.
        
        # Let's checklibs? `foamToVTK` is also a good test that reads the whole mesh/fields.
        
        pass 
        
    def _to_wsl_path(self, path: Path) -> str:
        """Convert a windows path to WSL path (e.g. C:\\Users -> /mnt/c/Users)"""
        # Simplistic conversion
        parts = list(path.parts)
        if parts[0].endswith(':\\'):
            drive = parts[0][0].lower()
            return f"/mnt/{drive}/" + "/".join(parts[1:])
        return str(path).replace('\\', '/')
