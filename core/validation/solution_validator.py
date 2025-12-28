"""
Post-Simulation Solution Validation
====================================

Validates solution files after solver completes to catch silent failures
and ensure data quality before visualization.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
HAVE_PYVISTA = True  # We assume it's installed, let failure happen gracefully if not inside method

@dataclass
class SolutionIssue:
    """Represents a solution data quality problem"""
    severity: str  # 'error', 'warning'
    field: str  # Field name with the issue
    message: str
    suggested_fix: str

@dataclass
class SolutionValidationResult:
    """Result of solution validation"""
    is_valid: bool
    issues: List[SolutionIssue]
    fields_found: List[str]
    data_quality_score: float  # 0.0 to 1.0
    
    def get_error_summary(self) -> str:
        """Get human-readable error summary"""
        if self.is_valid:
            return f"Solution is valid (quality: {self.data_quality_score:.1%})"
            
        summary = f"Solution validation failed (quality: {self.data_quality_score:.1%}):\n"
        
        for issue in self.issues:
            icon = "❌" if issue.severity == 'error' else "⚠️"
            summary += f"  {icon} [{issue.field}] {issue.message}\n"
            if issue.suggested_fix:
                summary += f"     💡 {issue.suggested_fix}\n"
            
        return summary


class SolutionValidator:
    """Validates solution files for data quality"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def validate_cfd_solution(self, vtk_file: str, expected_fields: Optional[List[str]] = None) -> SolutionValidationResult:
        """
        Validate CFD solution file.
        
        Args:
            vtk_file: Path to VTK/VTU solution file
            expected_fields: List of expected field names (default: ['U', 'p'])
            
        Returns:
            SolutionValidationResult
        """
        if expected_fields is None:
            expected_fields = ['U', 'p']  # Velocity and pressure for CFD
            
        vtk_path = Path(vtk_file)
        
        if not vtk_path.exists():
            return SolutionValidationResult(
                is_valid=False,
                issues=[SolutionIssue(
                    severity='error',
                    field='file',
                    message=f"Solution file not found: {vtk_file}",
                    suggested_fix="Check if solver completed successfully"
                )],
                fields_found=[],
                data_quality_score=0.0
            )
        
        if not HAVE_PYVISTA:
            # Can't validate without PyVista
            return SolutionValidationResult(
                is_valid=True,  # Assume valid if we can't check
                issues=[SolutionIssue(
                    severity='warning',
                    field='system',
                    message="PyVista not available - skipping detailed validation",
                    suggested_fix=""
                )],
                fields_found=[],
                data_quality_score=0.5  # Unknown quality
            )
        
        issues = []
        
        try:
            # Load solution file
            if self.verbose:
                logger.info(f"Validating solution file: {vtk_path.name}")
                
            import pyvista as pv
            mesh = pv.read(str(vtk_path))
            
            # Get available fields
            point_fields = list(mesh.point_data.keys())
            cell_fields = list(mesh.cell_data.keys())
            all_fields = point_fields + cell_fields
            
            if self.verbose:
                logger.info(f"  Found {len(point_fields)} point fields: {point_fields[:5]}")
                logger.info(f"  Found {len(cell_fields)} cell fields: {cell_fields[:5]}")
            
            # Check 1: Required fields present
            missing_fields = []
            for field in expected_fields:
                if field not in all_fields:
                    missing_fields.append(field)
                    
            if missing_fields:
                issues.append(SolutionIssue(
                    severity='error',
                    field=', '.join(missing_fields),
                    message=f"Missing required fields: {missing_fields}",
                    suggested_fix="Check solver configuration and ensure it writes all required fields"
                ))
            
            # Check 2: Data validity (no NaN, Inf)
            quality_scores = []
            
            for field_name in all_fields:
                # Get data
                if field_name in mesh.point_data:
                    data = mesh.point_data[field_name]
                else:
                    data = mesh.cell_data[field_name]
                
                # Check for NaN/Inf
                if isinstance(data, np.ndarray):
                    field_issues = self._check_field_validity(field_name, data)
                    issues.extend(field_issues)
                    
                    # Calculate quality score for this field
                    total_values = data.size
                    if total_values > 0:
                        valid_values = np.sum(np.isfinite(data))
                        quality_scores.append(valid_values / total_values)
            
            # Check 3: CFD-specific checks
            if 'U' in all_fields:
                velocity_issues = self._check_velocity_field(mesh)
                issues.extend(velocity_issues)
            
            # Calculate overall quality score
            if quality_scores:
                data_quality_score = np.mean(quality_scores)
            else:
                data_quality_score = 0.0
            
            # Determine if valid
            error_count = len([i for i in issues if i.severity == 'error'])
            is_valid = error_count == 0 and data_quality_score >= 0.9
            
            if self.verbose:
                if is_valid:
                    logger.info(f"✓ Solution validation passed (quality: {data_quality_score:.1%})")
                else:
                    logger.warning(f"✗ Solution validation failed (quality: {data_quality_score:.1%})")
            
            return SolutionValidationResult(
                is_valid=is_valid,
                issues=issues,
                fields_found=all_fields,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            logger.error(f"Solution validation crashed: {e}")
            return SolutionValidationResult(
                is_valid=False,
                issues=[SolutionIssue(
                    severity='error',
                    field='system',
                    message=f"Validation crashed: {str(e)}",
                    suggested_fix="Check file format and integrity"
                )],
                fields_found=[],
                data_quality_score=0.0
            )
    
    def _check_field_validity(self, field_name: str, data: np.ndarray) -> List[SolutionIssue]:
        """Check if field data contains NaN or Inf values"""
        issues = []
        
        total_values = data.size
        nan_count = np.sum(np.isnan(data))
        inf_count = np.sum(np.isinf(data))
        
        if nan_count > 0:
            percentage = (nan_count / total_values) * 100
            severity = 'error' if percentage > 10 else 'warning'
            issues.append(SolutionIssue(
                severity=severity,
                field=field_name,
                message=f"Contains {nan_count} NaN values ({percentage:.1f}%)",
                suggested_fix="Check solver convergence and mesh quality"
            ))
        
        if inf_count > 0:
            percentage = (inf_count / total_values) * 100
            severity = 'error' if percentage > 10 else 'warning'
            issues.append(SolutionIssue(
                severity=severity,
                field=field_name,
                message=f"Contains {inf_count} Inf values ({percentage:.1f}%)",
                suggested_fix="Check for division by zero or numerical instabilities"
            ))
        
        return issues
    
    def _check_velocity_field(self, mesh: 'pv.DataSet') -> List[SolutionIssue]:
        """CFD-specific check for velocity field"""
        issues = []
        
        try:
            # Get velocity data
            if 'U' in mesh.point_data:
                U = mesh.point_data['U']
            elif 'U' in mesh.cell_data:
                U = mesh.cell_data['U']
            else:
                return issues
            
            # Calculate velocity magnitude
            if U.ndim == 2 and U.shape[1] >= 3:
                vel_mag = np.linalg.norm(U, axis=1)
            elif U.ndim == 1:
                vel_mag = np.abs(U)
            else:
                return issues
            
            # Check if velocity is non-zero somewhere
            max_vel = np.max(vel_mag[np.isfinite(vel_mag)])
            
            if max_vel < 1e-10:
                issues.append(SolutionIssue(
                    severity='error',
                    field='U',
                    message="Velocity field is zero everywhere - solver may not have converged",
                    suggested_fix="Check solver logs for convergence issues or increase iterations"
                ))
            
            # Check for unreasonably high velocities (potential divergence)
            if max_vel > 1000:  # 1000 m/s is supersonic - unlikely in typical CFD
                issues.append(SolutionIssue(
                    severity='warning',
                    field='U',
                    message=f"Extremely high velocity detected ({max_vel:.1f} m/s) - possible divergence",
                    suggested_fix="Check solver stability and time step settings"
                ))
            
        except Exception as e:
            logger.warning(f"Velocity field check failed: {e}")
        
        return issues
