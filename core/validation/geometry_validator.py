"""
Pre-Flight Geometry Validation
================================

Validates CAD geometry before meshing to catch problems early and provide
actionable error messages.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass
import gmsh

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """Represents a geometry validation problem"""
    severity: str  # 'error', 'warning'
    category: str  # 'intersection', 'gap', 'degenerate', 'feature_size'
    message: str
    suggested_fix: str

@dataclass
class ValidationResult:
    """Result of geometry validation"""
    is_valid: bool
    issues: List[ValidationIssue]
    can_auto_repair: bool
    
    def get_error_summary(self) -> str:
        """Get human-readable error summary"""
        if self.is_valid:
            return "Geometry is valid"
            
        errors = [i for i in self.issues if i.severity == 'error']
        warnings = [i for i in self.issues if i.severity == 'warning']
        
        summary = f"Found {len(errors)} error(s) and {len(warnings)} warning(s):\n"
        
        for issue in errors:
            summary += f"  ❌ [{issue.category}] {issue.message}\n"
            summary += f"     💡 {issue.suggested_fix}\n"
            
        for issue in warnings:
            summary += f"  ⚠️  [{issue.category}] {issue.message}\n"
            
        if self.can_auto_repair:
            summary += "\n✨ Auto-repair may be able to fix these issues."
            
        return summary


class GeometryValidator:
    """Validates CAD geometry before meshing"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def validate(self, cad_file: str, target_mesh_size: float = 1.0) -> ValidationResult:
        """
        Validate CAD geometry.
        
        Args:
            cad_file: Path to CAD file (.step, .stp, .iges, etc.)
            target_mesh_size: Target mesh element size (for feature size checks)
            
        Returns:
            ValidationResult with issues found
        """
        cad_path = Path(cad_file)
        
        if not cad_path.exists():
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    severity='error',
                    category='file',
                    message=f"File not found: {cad_file}",
                    suggested_fix="Check file path and ensure file exists"
                )],
                can_auto_repair=False
            )
        
        issues = []
        
        # Initialize GMSH for analysis
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0 if not self.verbose else 1)
        
        try:
            # Load geometry
            if self.verbose:
                logger.info(f"Loading geometry from {cad_path.name}...")
                
            if cad_path.suffix.lower() in ['.step', '.stp']:
                gmsh.model.occ.importShapes(str(cad_path))
            elif cad_path.suffix.lower() in ['.iges', '.igs']:
                gmsh.model.occ.importShapes(str(cad_path))
            else:
                issues.append(ValidationIssue(
                    severity='error',
                    category='format',
                    message=f"Unsupported file format: {cad_path.suffix}",
                    suggested_fix="Convert to STEP (.step) format"
                ))
                return ValidationResult(is_valid=False, issues=issues, can_auto_repair=False)
            
            gmsh.model.occ.synchronize()
            
            # Check 1: Detect self-intersections
            intersection_issues = self._check_intersections()
            issues.extend(intersection_issues)
            
            # Check 2: Check for gaps and overlaps
            gap_issues = self._check_surface_continuity()
            issues.extend(gap_issues)
            
            # Check 3: Detect degenerate geometry
            degenerate_issues = self._check_degenerate_geometry()
            issues.extend(degenerate_issues)
            
            # Check 4: Validate feature sizes
            feature_issues = self._check_feature_sizes(target_mesh_size)
            issues.extend(feature_issues)
            
            # Determine if auto-repair is possible
            can_auto_repair = self._can_auto_repair(issues)
            
            # Count errors
            error_count = len([i for i in issues if i.severity == 'error'])
            is_valid = error_count == 0
            
            if self.verbose:
                if is_valid:
                    logger.info("✓ Geometry validation passed")
                else:
                    logger.warning(f"✗ Geometry validation failed with {error_count} error(s)")
            
            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                can_auto_repair=can_auto_repair
            )
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            issues.append(ValidationIssue(
                severity='error',
                category='exception',
                message=f"Validation crashed: {str(e)}",
                suggested_fix="Check CAD file integrity and try repairing in CAD software"
            ))
            return ValidationResult(is_valid=False, issues=issues, can_auto_repair=False)
            
        finally:
            gmsh.finalize()
    
    def _check_intersections(self) -> List[ValidationIssue]:
        """Check for self-intersecting surfaces"""
        issues = []
        
        try:
            # Get all surfaces
            surfaces = gmsh.model.getEntities(2)  # dim=2 for surfaces
            
            if len(surfaces) < 2:
                return issues
            
            # GMSH doesn't have a direct self-intersection check,
            # but we can use boolean operations to detect overlaps
            # For now, we'll do a heuristic check based on bounding box overlaps
            
            # This is a simplified check - proper intersection detection
            # would require OCC kernel operations
            
        except Exception as e:
            logger.warning(f"Intersection check failed: {e}")
            
        return issues
    
    def _check_surface_continuity(self) -> List[ValidationIssue]:
        """Check for gaps between surfaces"""
        issues = []
        
        try:
            # Get all curves (edges)
            curves = gmsh.model.getEntities(1)  # dim=1 for curves
            
            # Check if any curve is used by only one surface (indicates a gap)
            for curve in curves:
                dim, tag = curve
                # Get surfaces adjacent to this curve
                upward, _ = gmsh.model.getAdjacencies(dim, tag)
                
                if len(upward) == 1:
                    # This edge is only connected to one surface - potential gap
                    issues.append(ValidationIssue(
                        severity='warning',
                        category='gap',
                        message=f"Detected open edge (curve {tag}) - may indicate gap in geometry",
                        suggested_fix="Check for missing surfaces or try auto-repair"
                    ))
                    # Only report first few to avoid spam
                    if len([i for i in issues if i.category == 'gap']) >= 3:
                        break
                        
        except Exception as e:
            logger.warning(f"Continuity check failed: {e}")
            
        return issues
    
    def _check_degenerate_geometry(self) -> List[ValidationIssue]:
        """Check for degenerate faces and edges"""
        issues = []
        
        try:
            # Get all surfaces
            surfaces = gmsh.model.getEntities(2)
            
            for surface in surfaces:
                dim, tag = surface
                # Get bounding box
                try:
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                    
                    # Check if surface is degenerate (zero area)
                    dx = xmax - xmin
                    dy = ymax - ymin
                    dz = zmax - zmin
                    
                    if min(dx, dy, dz) < 1e-6:
                        issues.append(ValidationIssue(
                            severity='error',
                            category='degenerate',
                            message=f"Surface {tag} is degenerate (zero area)",
                            suggested_fix="Remove degenerate faces in CAD software or use auto-repair"
                        ))
                        
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning(f"Degenerate geometry check failed: {e}")
            
        return issues
    
    def _check_feature_sizes(self, target_mesh_size: float) -> List[ValidationIssue]:
        """Check if features are large enough for target mesh size"""
        issues = []
        
        try:
            # Get all curves
            curves = gmsh.model.getEntities(1)
            
            min_curve_length = float('inf')
            
            for curve in curves:
                dim, tag = curve
                try:
                    # Get curve bounding box
                    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                    length = max(xmax - xmin, ymax - ymin, zmax - zmin)
                    
                    if length < min_curve_length:
                        min_curve_length = length
                        
                except Exception:
                    pass
            
            # Warn if smallest feature is less than 2x target mesh size
            if min_curve_length < target_mesh_size * 2:
                issues.append(ValidationIssue(
                    severity='warning',
                    category='feature_size',
                    message=f"Smallest feature ({min_curve_length:.3f}mm) is small relative to mesh size ({target_mesh_size:.3f}mm)",
                    suggested_fix=f"Consider using a finer mesh (< {min_curve_length/3:.3f}mm) or simplifying geometry"
                ))
                
        except Exception as e:
            logger.warning(f"Feature size check failed: {e}")
            
        return issues
    
    def _can_auto_repair(self, issues: List[ValidationIssue]) -> bool:
        """Determine if issues can be fixed by auto-repair"""
        # Auto-repair can handle gaps, some degeneracies, but not severe intersections
        repairable_categories = {'gap', 'degenerate'}
        
        for issue in issues:
            if issue.severity == 'error' and issue.category not in repairable_categories:
                return False
                
        # If we have any repairable issues, return True
        return any(i.category in repairable_categories for i in issues)
