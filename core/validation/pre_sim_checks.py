"""
Pre-Simulation Robustness Checks
=================================

Validates input data before simulation to prevent crashes and detect
problematic configurations early. Writes crash_log.txt on failure.

Checks implemented:
1. Patch/BC Verification - Ensure required patches exist in mesh
2. Physics Parameter Bounds - Validate velocity, temperature, material properties
3. Mesh Quality - Check for negative volumes, extreme aspect ratios
4. Geometry Scale Detection - Detect wrong unit scales
5. Disk Space - Ensure sufficient space for solver output
"""

import os
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    check_name: str
    passed: bool
    message: str
    severity: str = "error"  # "error", "warning", "info"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report"""
    job_name: str
    timestamp: str
    checks: List[ValidationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Report passes if no errors (warnings allowed)"""
        return all(c.passed or c.severity == "warning" for c in self.checks)
    
    @property
    def has_warnings(self) -> bool:
        return any(c.severity == "warning" and not c.passed for c in self.checks)
    
    def to_log_string(self) -> str:
        """Format report for crash log file"""
        lines = [
            "=" * 60,
            f"SIMOPS PRE-SIMULATION VALIDATION REPORT",
            "=" * 60,
            f"Job:       {self.job_name}",
            f"Timestamp: {self.timestamp}",
            f"Status:    {'PASSED' if self.passed else 'FAILED'}",
            "-" * 60,
        ]
        
        for check in self.checks:
            status = "[OK]  " if check.passed else ("[WARN]" if check.severity == "warning" else "[FAIL]")
            lines.append(f"{status} {check.check_name}")
            lines.append(f"         {check.message}")
            if check.details:
                for k, v in check.details.items():
                    lines.append(f"         - {k}: {v}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class PreSimulationValidator:
    """
    Validates simulation inputs before execution.
    Writes crash_log.txt to output folder on failure.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.results: List[ValidationResult] = []
    
    def validate_all(
        self,
        job_name: str,
        mesh_file: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        case_dir: Optional[Path] = None,
        expected_patches: Optional[List[str]] = None
    ) -> ValidationReport:
        """
        Run all validation checks and return report.
        
        Args:
            job_name: Name of the simulation job
            mesh_file: Path to mesh file (for mesh quality checks)
            config: Simulation configuration dict
            case_dir: OpenFOAM case directory (for patch verification)
            expected_patches: List of patch names expected by BCs
            
        Returns:
            ValidationReport with all check results
        """
        self.results = []
        
        # 1. Patch Name Verification
        if case_dir and expected_patches:
            self._check_patch_names(case_dir, expected_patches)
        
        # 2. Physics Parameter Bounds
        if config:
            self._check_physics_bounds(config)
        
        # 3. Mesh Quality (if mesh file provided)
        if mesh_file and mesh_file.exists():
            self._check_mesh_quality(mesh_file)
        
        # 4. Geometry Scale Detection
        if mesh_file and mesh_file.exists():
            self._check_geometry_scale(mesh_file)
        
        # 5. Disk Space
        self._check_disk_space()
        
        # Build report
        report = ValidationReport(
            job_name=job_name,
            timestamp=datetime.now().isoformat(),
            checks=self.results
        )
        
        # Write crash log if failed
        if not report.passed:
            self._write_crash_log(report)
        
        return report
    
    # =========================================================================
    # CHECK 1: Patch Name Verification
    # =========================================================================
    def _check_patch_names(self, case_dir: Path, expected_patches: List[str]):
        """Verify that required BC patches exist in the mesh"""
        check_name = "Patch Name Verification"
        
        try:
            boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
            
            if not boundary_file.exists():
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Boundary file not found: {boundary_file}",
                    severity="error"
                ))
                return
            
            # Parse boundary file for patch names
            content = boundary_file.read_text()
            
            # Simple regex to find patch names
            import re
            # Match pattern like "inlet\n{" or "outlet\n    {"
            patch_matches = re.findall(r'^\s*(\w+)\s*\n\s*\{', content, re.MULTILINE)
            found_patches = set(patch_matches)
            
            # Check each expected patch
            missing = []
            for expected in expected_patches:
                # Check exact match or partial match (inlet might be BC_Inlet)
                found = any(expected.lower() in p.lower() or p.lower() in expected.lower() 
                           for p in found_patches)
                if not found:
                    missing.append(expected)
            
            if missing:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Missing patches: {missing}. Found: {list(found_patches)}",
                    severity="error",
                    details={"missing": missing, "found": list(found_patches)}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=True,
                    message=f"All required patches found: {expected_patches}",
                    details={"found": list(found_patches)}
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=False,
                message=f"Error checking patches: {e}",
                severity="error"
            ))
    
    # =========================================================================
    # CHECK 2: Physics Parameter Bounds
    # =========================================================================
    def _check_physics_bounds(self, config: Dict[str, Any]):
        """Validate physics parameters are within reasonable bounds"""
        check_name = "Physics Parameter Bounds"
        issues = []
        warnings = []
        
        physics = config.get("physics", {})
        
        # Velocity check
        velocity = physics.get("inlet_velocity", 0)
        if velocity < 0:
            issues.append(f"Negative inlet velocity: {velocity}")
        elif velocity > 100:  # > 100 m/s might need compressible solver
            warnings.append(f"High velocity ({velocity} m/s) - consider compressible solver if > Mach 0.3")
        
        # Viscosity check
        nu = physics.get("kinematic_viscosity", 1.5e-5)
        if nu <= 0:
            issues.append(f"Non-positive viscosity: {nu}")
        elif nu < 1e-8 or nu > 1:
            warnings.append(f"Unusual viscosity value: {nu} m²/s")
        
        # Temperature bounds (if thermal)
        temp = physics.get("heat_source_temperature") or physics.get("ambient_temperature")
        if temp is not None:
            if temp < 0:
                issues.append(f"Negative temperature: {temp}K (below absolute zero)")
            elif temp > 10000:
                issues.append(f"Extreme temperature: {temp}K")
        
        # Reynolds number estimation
        if velocity > 0 and nu > 0:
            # Assume characteristic length of 1m if not provided
            L = physics.get("characteristic_length", 1.0)
            Re = (velocity * L) / nu
            if Re > 1e7:
                warnings.append(f"Very high Reynolds number: {Re:.2e} - turbulence modeling recommended")
            elif Re < 0.1:
                warnings.append(f"Very low Reynolds number: {Re:.2e} - creeping flow regime")
        
        # Report results
        if issues:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=False,
                message="; ".join(issues),
                severity="error",
                details={"physics_config": physics}
            ))
        elif warnings:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=False,
                message="; ".join(warnings),
                severity="warning",
                details={"physics_config": physics}
            ))
        else:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=True,
                message="All physics parameters within bounds"
            ))
    
    # =========================================================================
    # CHECK 3: Mesh Quality
    # =========================================================================
    def _check_mesh_quality(self, mesh_file: Path):
        """Check mesh for quality issues that could crash solver"""
        check_name = "Mesh Quality"
        
        try:
            # Try to read mesh with gmsh or meshio
            import meshio
            mesh = meshio.read(str(mesh_file))
            
            issues = []
            warnings = []
            
            # Check for empty mesh
            if len(mesh.points) == 0:
                issues.append("Mesh has no points")
            
            if len(mesh.cells) == 0:
                issues.append("Mesh has no cells")
            
            # Check for degenerate points (NaN, Inf)
            import numpy as np
            if np.any(np.isnan(mesh.points)):
                issues.append("Mesh contains NaN coordinates")
            if np.any(np.isinf(mesh.points)):
                issues.append("Mesh contains infinite coordinates")
            
            # Basic volume check for tets
            for cell_block in mesh.cells:
                if cell_block.type in ["tetra", "tetra10"]:
                    # Sample check first 100 elements
                    cells = cell_block.data[:min(100, len(cell_block.data))]
                    for cell in cells:
                        pts = mesh.points[cell[:4]]  # First 4 points for tet
                        # Compute signed volume
                        v0 = pts[1] - pts[0]
                        v1 = pts[2] - pts[0]
                        v2 = pts[3] - pts[0]
                        vol = np.dot(v0, np.cross(v1, v2)) / 6.0
                        if vol <= 0:
                            issues.append("Mesh contains negative/zero volume elements (inverted tets)")
                            break
            
            # Report
            if issues:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message="; ".join(issues),
                    severity="error"
                ))
            elif warnings:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message="; ".join(warnings),
                    severity="warning"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=True,
                    message=f"Mesh OK: {len(mesh.points)} points, {sum(len(c.data) for c in mesh.cells)} cells"
                ))
                
        except ImportError:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=True,
                message="Skipped (meshio not available)",
                severity="info"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=False,
                message=f"Error reading mesh: {e}",
                severity="warning"  # Downgrade to warning since file might be OpenFOAM format
            ))
    
    # =========================================================================
    # CHECK 4: Geometry Scale Detection
    # =========================================================================
    def _check_geometry_scale(self, mesh_file: Path):
        """Detect potential unit scale issues"""
        check_name = "Geometry Scale Detection"
        
        try:
            import meshio
            mesh = meshio.read(str(mesh_file))
            
            import numpy as np
            bounds = {
                'x': (mesh.points[:, 0].min(), mesh.points[:, 0].max()),
                'y': (mesh.points[:, 1].min(), mesh.points[:, 1].max()),
                'z': (mesh.points[:, 2].min(), mesh.points[:, 2].max()),
            }
            
            sizes = {axis: abs(b[1] - b[0]) for axis, b in bounds.items()}
            max_dim = max(sizes.values())
            min_dim = min(sizes.values()) if min(sizes.values()) > 0 else 1e-10
            
            warnings = []
            
            # Check for suspiciously small dimensions (might be in meters when mm expected)
            if max_dim < 0.01:
                warnings.append(f"Very small geometry (max dimension: {max_dim:.6f}m) - check if units are correct")
            
            # Check for suspiciously large dimensions
            if max_dim > 1000:
                warnings.append(f"Very large geometry (max dimension: {max_dim:.1f}m) - check if units are correct")
            
            # Check for extreme aspect ratios
            aspect_ratio = max_dim / min_dim
            if aspect_ratio > 1000:
                warnings.append(f"Extreme aspect ratio: {aspect_ratio:.1f}:1 - may cause meshing issues")
            
            if warnings:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message="; ".join(warnings),
                    severity="warning",
                    details={"dimensions": sizes, "aspect_ratio": aspect_ratio}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=True,
                    message=f"Geometry scale OK: {max_dim:.3f}m max dimension",
                    details={"dimensions": sizes}
                ))
                
        except ImportError:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=True,
                message="Skipped (meshio not available)",
                severity="info"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=True,  # Don't fail on this
                message=f"Could not check scale: {e}",
                severity="info"
            ))
    
    # =========================================================================
    # CHECK 5: Disk Space
    # =========================================================================
    def _check_disk_space(self, min_gb: float = 1.0):
        """Ensure sufficient disk space for solver output"""
        check_name = "Disk Space"
        
        try:
            total, used, free = shutil.disk_usage(self.output_dir)
            free_gb = free / (1024**3)
            
            if free_gb < min_gb:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Insufficient disk space: {free_gb:.2f}GB free (need {min_gb}GB)",
                    severity="error",
                    details={"free_gb": free_gb, "required_gb": min_gb}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name=check_name,
                    passed=True,
                    message=f"Disk space OK: {free_gb:.1f}GB available"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                check_name=check_name,
                passed=True,
                message=f"Could not check disk space: {e}",
                severity="info"
            ))
    
    # =========================================================================
    # Crash Log Writer
    # =========================================================================
    def _write_crash_log(self, report: ValidationReport):
        """Write detailed crash log to output directory"""
        crash_log_path = self.output_dir / f"{report.job_name}_CRASH_LOG.txt"
        
        try:
            crash_log_path.write_text(report.to_log_string())
            logger.error(f"Validation failed! Crash log written to: {crash_log_path}")
        except Exception as e:
            logger.error(f"Failed to write crash log: {e}")


# =============================================================================
# Convenience function for worker integration
# =============================================================================
def run_pre_simulation_checks(
    job_name: str,
    output_dir: Path,
    mesh_file: Path = None,
    config: Dict[str, Any] = None,
    case_dir: Path = None,
    expected_patches: List[str] = None
) -> Tuple[bool, str]:
    """
    Run all pre-simulation checks.
    
    Returns:
        Tuple of (passed: bool, message: str)
    """
    validator = PreSimulationValidator(output_dir)
    report = validator.validate_all(
        job_name=job_name,
        mesh_file=mesh_file,
        config=config,
        case_dir=case_dir,
        expected_patches=expected_patches
    )
    
    if report.passed:
        msg = "All pre-simulation checks passed"
        if report.has_warnings:
            msg += " (with warnings)"
        return True, msg
    else:
        return False, f"Pre-simulation validation failed. See crash log: {output_dir / f'{job_name}_CRASH_LOG.txt'}"
