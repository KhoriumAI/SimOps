"""
MSH Format Utilities
=====================

Utilities for detecting, validating, and converting GMSH .msh files
to ensure compatibility with OpenFOAM's gmshToFoam converter.
"""

import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional
from core.logging.sim_logger import SimLogger

logger = SimLogger("MSHUtils")


def detect_msh_format(msh_file: Path) -> Dict[str, any]:
    """
    Detect MSH file format version and properties.

    Args:
        msh_file: Path to .msh file

    Returns:
        Dictionary containing:
        - version: Format version (e.g., "2.2", "4.1")
        - format: "ASCII" or "Binary"
        - size_type: Size of size_t (for binary format)
        - valid: Whether file appears valid
        - error: Error message if invalid
    """
    result = {
        "version": None,
        "format": None,
        "size_type": None,
        "valid": False,
        "error": None
    }

    if not msh_file.exists():
        result["error"] = f"File not found: {msh_file}"
        return result

    try:
        # Read first 500 bytes to identify format
        with open(msh_file, 'rb') as f:
            header = f.read(500)

        # Try to decode as ASCII
        try:
            header_text = header.decode('ascii')
        except UnicodeDecodeError:
            # Likely binary format or corrupted
            result["error"] = "Unable to decode file header - possibly binary or corrupted"
            return result

        # Check for MSH format markers
        # MSH 2.x format: $MeshFormat ... $EndMeshFormat
        # MSH 4.x format: $MeshFormat ... $EndMeshFormat

        if "$MeshFormat" not in header_text:
            result["error"] = "Not a valid GMSH .msh file - missing $MeshFormat marker"
            return result

        # Extract version from $MeshFormat section
        # Format: $MeshFormat\n<version> <file-type> <data-size>\n
        format_match = re.search(
            r'\$MeshFormat\s+([\d\.]+)\s+(\d)\s+(\d+)',
            header_text
        )

        if not format_match:
            result["error"] = "Could not parse $MeshFormat section"
            return result

        version_str = format_match.group(1)
        file_type = int(format_match.group(2))  # 0 = ASCII, 1 = Binary
        size_type = int(format_match.group(3))  # Size of size_t

        result["version"] = version_str
        result["format"] = "Binary" if file_type == 1 else "ASCII"
        result["size_type"] = size_type
        result["valid"] = True

        # Check compatibility with gmshToFoam
        major_version = int(version_str.split('.')[0])
        minor_version = int(version_str.split('.')[1]) if '.' in version_str else 0

        if major_version >= 4:
            result["gmshToFoam_compatible"] = False
            result["warning"] = f"MSH {version_str} often fails with gmshToFoam - recommend conversion to 2.2"
        elif major_version == 2 and minor_version == 2:
            result["gmshToFoam_compatible"] = True
        else:
            result["gmshToFoam_compatible"] = "uncertain"
            result["warning"] = f"MSH {version_str} compatibility with gmshToFoam is uncertain"

        logger.info(f"[MSH Detection] Version: {version_str}, Format: {result['format']}, Compatible: {result.get('gmshToFoam_compatible', 'unknown')}")

        return result

    except Exception as e:
        result["error"] = f"Failed to detect format: {str(e)}"
        logger.error(f"[MSH Detection] Error: {e}")
        return result


def convert_msh_format(input_file: Path, output_file: Path, target_version: float = 2.2, use_wsl: bool = True) -> Tuple[bool, str]:
    """
    Convert .msh file to a different format version using GMSH.

    Args:
        input_file: Input .msh file
        output_file: Output .msh file path
        target_version: Target MSH format version (default: 2.2 for OpenFOAM compatibility)
        use_wsl: Whether to run gmsh via WSL (Windows only)

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        import platform

        # Build gmsh command for format conversion
        # Using gmsh CLI: gmsh input.msh -format msh22 -o output.msh -v 0
        version_flag = f"msh{int(target_version * 10)}"  # 2.2 -> msh22, 4.1 -> msh41

        if target_version == 2.2:
            version_flag = "msh22"
        elif target_version == 4.1:
            version_flag = "msh41"
        else:
            return False, f"Unsupported target version: {target_version}"

        cmd = f"gmsh \"{input_file}\" -format {version_flag} -o \"{output_file}\" -v 0"

        logger.info(f"[MSH Conversion] Converting {input_file.name} to MSH {target_version}...")

        if use_wsl and platform.system() == "Windows":
            # Convert paths to WSL format
            input_wsl = str(input_file).replace('\\', '/').replace('C:', '/mnt/c')
            output_wsl = str(output_file).replace('\\', '/').replace('C:', '/mnt/c')
            cmd = f"gmsh '{input_wsl}' -format {version_flag} -o '{output_wsl}' -v 0"

            result = subprocess.run(
                ['wsl', 'bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=60
            )
        else:
            result = subprocess.run(
                ['bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=60
            )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"[MSH Conversion] Failed: {error_msg}")
            return False, f"gmsh conversion failed: {error_msg}"

        if not output_file.exists():
            return False, "Conversion appeared to succeed but output file not found"

        logger.info(f"[MSH Conversion] Successfully converted to {output_file.name}")
        return True, f"Converted to MSH {target_version}"

    except subprocess.TimeoutExpired:
        return False, "Conversion timed out after 60 seconds"
    except FileNotFoundError:
        return False, "gmsh not found in PATH - cannot convert format"
    except Exception as e:
        logger.error(f"[MSH Conversion] Exception: {e}")
        return False, f"Conversion failed: {str(e)}"


def parse_gmshToFoam_log(log_file: Path) -> Dict[str, any]:
    """
    Parse gmshToFoam log file to extract errors, warnings, and statistics.

    Args:
        log_file: Path to log.gmshToFoam

    Returns:
        Dictionary containing:
        - success: Whether conversion succeeded
        - num_cells: Number of cells converted
        - num_points: Number of points
        - num_faces: Number of faces
        - errors: List of error messages
        - warnings: List of warning messages
        - raw_log: Full log content (truncated)
    """
    result = {
        "success": False,
        "num_cells": 0,
        "num_points": 0,
        "num_faces": 0,
        "errors": [],
        "warnings": [],
        "raw_log": ""
    }

    if not log_file.exists():
        result["errors"].append("Log file not found")
        return result

    try:
        content = log_file.read_text(encoding='utf-8', errors='replace')
        result["raw_log"] = content[:2000]  # Keep first 2000 chars

        # Check for common error patterns
        error_patterns = [
            (r"FOAM FATAL ERROR", "Fatal error in gmshToFoam"),
            (r"Unknown element type", "Unknown element type - mesh may contain unsupported elements"),
            (r"Bad input stream", "Bad input stream - file may be corrupted or wrong format"),
            (r"Expected keyword", "Parse error - expected keyword not found"),
            (r"duplicate patch names", "Duplicate patch/boundary names detected"),
            (r"Cannot find file", "Input file not found or unreadable"),
        ]

        for pattern, message in error_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                result["errors"].append(message)

        # Check for warnings
        warning_patterns = [
            (r"Warning", "General warning detected"),
            (r"Unassigned", "Unassigned faces or cells detected"),
        ]

        for pattern, message in warning_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                result["warnings"].append(f"{message} (found {len(matches)} times)")

        # Extract statistics
        # Looking for patterns like "total: 89177" for cells
        cell_match = re.search(r"total:\s+(\d+)", content)
        if cell_match:
            result["num_cells"] = int(cell_match.group(1))

        # Extract points
        point_match = re.search(r"points:\s+(\d+)", content)
        if point_match:
            result["num_points"] = int(point_match.group(1))

        # Check for successful completion
        # gmshToFoam typically ends with "End" or writes polyMesh files
        if "End" in content or result["num_cells"] > 0:
            result["success"] = True

        if result["errors"]:
            result["success"] = False

        return result

    except Exception as e:
        result["errors"].append(f"Failed to parse log: {str(e)}")
        return result


def convert_via_fluent(msh_file: Path, case_dir: Path, use_wsl: bool = True) -> Tuple[bool, str]:
    """
    Fallback conversion route: GMSH → Fluent → OpenFOAM.

    This is more robust for problematic .msh files.

    Args:
        msh_file: Input .msh file
        case_dir: OpenFOAM case directory
        use_wsl: Whether to use WSL

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        import platform

        # Step 1: Convert .msh to Fluent format using gmsh
        fluent_file = case_dir / "fluent_mesh.msh"

        logger.info("[Fluent Fallback] Converting .msh to Fluent format...")

        if use_wsl and platform.system() == "Windows":
            msh_wsl = str(msh_file).replace('\\', '/').replace('C:', '/mnt/c')
            fluent_wsl = str(fluent_file).replace('\\', '/').replace('C:', '/mnt/c')
            cmd = f"gmsh '{msh_wsl}' -format msh -o '{fluent_wsl}' -v 0"

            result = subprocess.run(
                ['wsl', 'bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=60
            )
        else:
            cmd = f"gmsh \"{msh_file}\" -format msh -o \"{fluent_file}\" -v 0"
            result = subprocess.run(
                ['bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=60
            )

        if result.returncode != 0 or not fluent_file.exists():
            return False, f"Failed to create Fluent format: {result.stderr}"

        # Step 2: Use fluent3DMeshToFoam
        logger.info("[Fluent Fallback] Converting Fluent mesh to OpenFOAM...")

        case_wsl = str(case_dir).replace('\\', '/').replace('C:', '/mnt/c')
        foam_source = "source /usr/lib/openfoam/openfoam2312/etc/bashrc 2>/dev/null || source /opt/openfoam10/etc/bashrc 2>/dev/null || source /opt/openfoam13/etc/bashrc 2>/dev/null"

        if use_wsl and platform.system() == "Windows":
            cmd = f"{foam_source}; cd '{case_wsl}' && fluent3DMeshToFoam fluent_mesh.msh"
            result = subprocess.run(
                ['wsl', 'bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=120
            )
        else:
            cmd = f"{foam_source}; cd '{case_dir}' && fluent3DMeshToFoam fluent_mesh.msh"
            result = subprocess.run(
                ['bash', '-c', cmd],
                capture_output=True,
                text=True,
                timeout=120
            )

        if result.returncode != 0:
            return False, f"fluent3DMeshToFoam failed: {result.stderr or result.stdout}"

        # Check if polyMesh was created
        poly_mesh = case_dir / "constant" / "polyMesh"
        if not (poly_mesh / "points").exists():
            return False, "fluent3DMeshToFoam completed but polyMesh not found"

        logger.info("[Fluent Fallback] Successfully converted via Fluent route")
        return True, "Converted via Fluent fallback route"

    except subprocess.TimeoutExpired:
        return False, "Fluent conversion timed out"
    except Exception as e:
        logger.error(f"[Fluent Fallback] Exception: {e}")
        return False, f"Fluent conversion failed: {str(e)}"
