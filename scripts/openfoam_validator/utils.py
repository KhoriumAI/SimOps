import os
import shutil
import subprocess
import platform
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# Common installation paths to check if not in PATH
COMMON_PATHS = [
    "/usr/lib/openfoam",
    "/opt/openfoam",
    "/usr/bin",
    "C:\\Program Files\\OpenFOAM",
]


def run_command(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> Tuple[int, str, str]:
    """
    Run a subprocess command with timeout and output capturing.
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            timeout=timeout,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
    except Exception as e:
        return -1, "", str(e)

def check_openfoam_availability(custom_path: Optional[str] = None) -> Dict[str, bool]:
    """
    Check if OpenFOAM is available via direct command or WSL.
    Returns a dict with 'native' and 'wsl' booleans.
    """
    status = {'native': False, 'wsl': False}

    # 1. Check native
    if custom_path and (Path(custom_path) / 'bin' / 'simpleFoam').exists():
        status['native'] = True
    elif shutil.which('foamList') or shutil.which('simpleFoam'):
        status['native'] = True
    else:
        # Check common linux paths
        for p in COMMON_PATHS:
            if Path(p).exists():
                # This is a bit naive but better than nothing
                status['native'] = True
                break

    # 2. Check WSL (Windows only)
    if platform.system() == "Windows":
        try:
            if shutil.which('wsl'):
                # Check custom path in WSL if provided
                if custom_path:
                    code, _, _ = run_command(['wsl', 'bash', '-c', f'ls "{custom_path}/bin/simpleFoam" || ls "{custom_path}/platforms/*/bin/simpleFoam"'])
                    if code == 0:
                        status['wsl'] = True
                        status['wsl_path'] = custom_path
                
                # If still not found, try command -v
                if not status['wsl']:
                    code, stdout, _ = run_command(['wsl', 'bash', '-c', 'command -v simpleFoam'])
                    if code == 0 and stdout.strip():
                        status['wsl'] = True
                        # Try to resolve the path to the installation root? 
                        # Might be hard from just the binary, but often not needed if it's already in PATH.
                
                # If still not found, look for installations in /opt or /usr/lib
                if not status['wsl']:
                    # Look for /opt/openfoam* or /usr/lib/openfoam*
                    find_cmd = "ls -d /opt/openfoam* /usr/lib/openfoam* 2>/dev/null | head -n 1"
                    code, stdout, _ = run_command(['wsl', 'bash', '-c', find_cmd])
                    if code == 0 and stdout.strip():
                        found_path = stdout.strip()
                        status['wsl'] = True
                        status['wsl_path'] = found_path

        except Exception:
            pass


    return status


def get_openfoam_cmd_prefix(availability: Dict[str, bool]) -> List[str]:
    """
    Get the command prefix to run OpenFOAM commands.
    Prioritizes Native over WSL.
    """
    if availability['native']:
        return []
    elif availability['wsl']:
        return ['wsl', 'bash', '-c']
    else:
        raise RuntimeError("OpenFOAM not found (checked native and WSL)")

def find_solver(case_path: Path) -> Optional[str]:
    """
    Attempt to find the solver application name from system/controlDict.
    """
    control_dict = case_path / "system" / "controlDict"
    if not control_dict.exists():
        return None
    
    # Rudimentary parsing of controlDict
    # Looking for 'application   solverName;'
    try:
        content = control_dict.read_text(encoding='utf-8')
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("application"):
                # defined as: application \s+ name;
                parts = line.split()
                if len(parts) >= 2:
                    app = parts[1].replace(';', '')
                    return app
    except Exception as e:
        logger.warning(f"Failed to parse controlDict: {e}")
        return None
    
    return None
