"""
Pre-Simulation Robustness Checks
=================================

Comprehensive validation before simulation starts to catch problems early.
Integrates geometry validation, mesh validation, and case setup checks.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

def run_pre_simulation_checks(
    job_name: str,
    output_dir: Path,
    mesh_file: Path,
    config: Dict,
    case_dir: Optional[Path] = None,
    expected_patches: Optional[List[str]] = None,
    cad_file: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Run comprehensive pre-simulation validation checks.
    
    Args:
        job_name: Simulation job name
        output_dir: Output directory
        mesh_file: Path to mesh file
        config: Simulation configuration dict
        case_dir: Optional OpenFOAM case directory
        expected_patches: Expected boundary patch names
        cad_file: Optional source CAD file for geometry validation
        
    Returns:
        (passed, message) tuple
    """
    checks_passed = []
    messages = []
    
    # Check 1: Geometry Validation (if CAD file provided)
    if cad_file and Path(cad_file).exists():
        try:
            from core.validation.geometry_validator import GeometryValidator
            
            logger.info("[Pre-Sim] Running geometry validation...")
            validator = GeometryValidator(verbose=False)
            
            # Estimate target mesh size from config if available
            target_mesh_size = config.get('meshing', {}).get('mesh_size_factor', 1.0)
            
            validation_result = validator.validate(cad_file, target_mesh_size)
            
            if validation_result.is_valid:
                checks_passed.append(True)
                messages.append("✓ Geometry validation passed")
                logger.info("[Pre-Sim] Geometry validation: PASSED")
            else:
                # Don't fail if auto-repair is possible
                if validation_result.can_auto_repair:
                    checks_passed.append(True)
                    messages.append(f"⚠ Geometry has issues but auto-repair possible ({len(validation_result.issues)} issues)")
                    logger.warning(f"[Pre-Sim] Geometry validation: ISSUES FOUND (auto-repairable)")
                else:
                    checks_passed.append(False)
                    messages.append(f"✗ Geometry validation failed:\n{validation_result.get_error_summary()}")
                    logger.error(f"[Pre-Sim] Geometry validation: FAILED")
                    
        except Exception as e:
            logger.warning(f"[Pre-Sim] Geometry validation skipped: {e}")
            checks_passed.append(True)  # Don't block on validator failure
            messages.append(f"⚠ Geometry validation skipped: {e}")
    
    # Check 2: Mesh File Exists
    if mesh_file and mesh_file.exists():
        checks_passed.append(True)
        messages.append(f"✓ Mesh file exists: {mesh_file.name}")
        logger.info(f"[Pre-Sim] Mesh file check: PASSED ({mesh_file.name})")
    else:
        checks_passed.append(False)
        messages.append(f"✗ Mesh file not found: {mesh_file}")
        logger.error(f"[Pre-Sim] Mesh file check: FAILED")
    
    # Check 3: Case Directory Setup (for OpenFOAM)
    if case_dir:
        if case_dir.exists():
            # Check for required OpenFOAM directories
            required_dirs = ['0', 'constant', 'system']
            missing_dirs = [d for d in required_dirs if not (case_dir / d).exists()]
            
            if missing_dirs:
                checks_passed.append(False)
                messages.append(f"✗ Case directory missing: {missing_dirs}")
                logger.error(f"[Pre-Sim] Case setup: FAILED (missing {missing_dirs})")
            else:
                checks_passed.append(True)
                messages.append("✓ OpenFOAM case directory structure valid")
                logger.info("[Pre-Sim] Case setup: PASSED")
                
                # Check for boundary patches in constant/polyMesh/boundary
                if expected_patches:
                    boundary_file = case_dir / 'constant' / 'polyMesh' / 'boundary'
                    if boundary_file.exists():
                        try:
                            with open(boundary_file, 'r') as f:
                                boundary_content = f.read()
                            
                            missing_patches = [p for p in expected_patches if p not in boundary_content]
                            if missing_patches:
                                checks_passed.append(False)
                                messages.append(f"⚠ Missing expected patches: {missing_patches}")
                                logger.warning(f"[Pre-Sim] Patch check: WARNINGS ({missing_patches} not found)")
                            else:
                                checks_passed.append(True)
                                messages.append(f"✓ All expected patches present: {expected_patches}")
                                logger.info("[Pre-Sim] Patch check: PASSED")
                        except Exception as e:
                            logger.warning(f"[Pre-Sim] Could not read boundary file: {e}")
        else:
            # Case directory doesn't exist yet - not necessarily an error
            logger.info("[Pre-Sim] Case directory will be created by solver")
    
    # Check 4: Configuration Sanity
    if config:
        # Check for obviously wrong configs
        physics = config.get('physics', {})
        
        # Example: inlet velocity shouldn't be negative or insanely high
        inlet_vel = physics.get('inlet_velocity', 1.0)
        if isinstance(inlet_vel, (int, float)):
            if inlet_vel < 0:
                checks_passed.append(False)
                messages.append(f"✗ Invalid inlet velocity: {inlet_vel} (negative)")
                logger.error(f"[Pre-Sim] Config check: FAILED (negative velocity)")
            elif inlet_vel > 500:
                checks_passed.append(False)
                messages.append(f"⚠ Unusually high inlet velocity: {inlet_vel} m/s (supersonic?)")
                logger.warning(f"[Pre-Sim] Config check: WARNING (very high velocity)")
            else:
                checks_passed.append(True)
                logger.info(f"[Pre-Sim] Config check: PASSED (u_inlet={inlet_vel})")
    
    # Aggregate results
    all_passed = all(checks_passed) if checks_passed else False
    summary = "\n".join(messages)
    
    if all_passed:
        final_message = f"[Pre-Sim Checks] ALL PASSED\n{summary}"
        logger.info("[Pre-Sim] Overall result: PASSED")
    else:
        final_message = f"[Pre-Sim Checks] FAILURES DETECTED\n{summary}"
        logger.error("[Pre-Sim] Overall result: FAILED")
    
    return all_passed, final_message
