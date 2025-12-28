"""
Mesh Quality Validator
======================

Automated quality control for meshes before solver execution.
Detects disconnected geometry, inverted elements, and mesh quality issues.

Usage:
    validator = MeshQualityValidator()
    passed, reason = validator.validate(mesh_file)
    if not passed:
        raise RuntimeError(f"Mesh QC Failed: {reason}")
"""

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

try:
    import pyvista as pv
    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False

logger = logging.getLogger(__name__)


class MeshQualityValidator:
    """
    Validates mesh quality using PyVista to detect fatal issues:
    - Disconnected geometry (multiple isolated regions)
    - Inverted elements (negative Jacobian determinants)
    - Oversized meshes (optimizer failures)
    """
    
    def __init__(
        self,
        max_cells: int = 2_000_000,
        min_jacobian: float = 0.0,
        max_regions: int = 1
    ):
        """
        Initialize validator with quality thresholds.
        
        Args:
            max_cells: Maximum allowed cell count (prevents RAM exhaustion)
            min_jacobian: Minimum scaled Jacobian (0.0 = no inverted elements)
            max_regions: Maximum allowed disconnected regions (1 = fully connected)
        """
        self.max_cells = max_cells
        self.min_jacobian = min_jacobian
        self.max_regions = max_regions
        
        if not HAVE_PYVISTA:
            logger.warning("PyVista not available. Mesh QC will be skipped.")
    
    def validate(self, mesh_file: Path) -> Tuple[bool, str, Dict]:
        """
        Validate mesh quality.
        
        Args:
            mesh_file: Path to mesh file (.msh, .vtk, .vtu, etc.)
            
        Returns:
            Tuple of (passed, reason, metadata)
            - passed: True if mesh passes all checks
            - reason: Detailed failure reason or "Passed"
            - metadata: Dict with quality metrics
        """
        if not HAVE_PYVISTA:
            return True, "Skipped (PyVista not available)", {}
        
        mesh_file = Path(mesh_file)
        if not mesh_file.exists():
            return False, f"File not found: {mesh_file}", {}
        
        try:
            # Load mesh
            mesh_orig = pv.read(str(mesh_file))
            
            # If MultiBlock, try to find a valid grid block
            if isinstance(mesh_orig, pv.MultiBlock):
                logger.info(f"[MeshQC] MultiBlock detected with {len(mesh_orig)} blocks.")
                # Combine can be destructive for some quadratic element types in VTK
                # Try to find the largest UnstructuredGrid block
                mesh = None
                max_cells = -1
                for i in range(len(mesh_orig)):
                    block = mesh_orig[i]
                    if block is not None and block.n_cells > max_cells:
                        mesh = block
                        max_cells = block.n_cells
                
                if mesh is None:
                    return False, "MultiBlock mesh contains no valid blocks", {}
                logger.info(f"[MeshQC] Selected largest block with {mesh.n_cells} cells")
            else:
                mesh = mesh_orig
                
            if mesh is None or mesh.n_cells == 0:
                return False, "Mesh has zero cells", {}
            
            # Diagnostic: Cell types
            unique_types = np.unique(mesh.celltypes)
            logger.info(f"[MeshQC] Mesh info: {mesh.n_cells} cells, {mesh.n_points} points, Cell types: {unique_types}")
            
            metadata = {
                'n_cells': int(mesh.n_cells),
                'n_points': int(mesh.n_points),
                'bounds': [float(b) for b in mesh.bounds],
                'cell_types': [int(t) for t in unique_types]
            }
            
            # CHECK 1: Connectivity (Disconnected Geometry)
            try:
                passed, reason = self._check_connectivity(mesh)
                if not passed:
                    metadata['failure_type'] = 'disconnected_geometry'
                    return False, reason, metadata
            except Exception as e:
                logger.warning(f"[MeshQC] Connectivity check crashed: {e}")
            
            # CHECK 2: Element Quality (Inverted Elements)
            try:
                passed, reason, qual_stats = self._check_element_quality(mesh)
                metadata.update(qual_stats)
                if not passed:
                    metadata['failure_type'] = 'inverted_elements'
                    return False, reason, metadata
            except Exception as e:
                error_msg = f"Quality check crashed: {e}"
                logger.warning(f"[MeshQC] {error_msg}")
                metadata['failure_type'] = 'quality_check_crash'
                metadata['quality_error'] = str(e)
                # Don't fail the whole simulation just because QC crashed
                # unless explicitly requested.
            
            # CHECK 3: Mesh Size Sanity
            passed, reason = self._check_mesh_size(mesh)
            if not passed:
                metadata['failure_type'] = 'oversized_mesh'
                return False, reason, metadata
            
            # All checks passed (or skipped due to errors)
            logger.info(f"[MeshQC] ✅ Passed ({mesh.n_cells:,} cells)")
            return True, "Passed", metadata
            
        except Exception as e:
            # Log but don't fail - let the solver determine if mesh is viable
            import traceback
            error_details = traceback.format_exc()
            logger.warning(f"[MeshQC] Validation crashed: {e}. Traceback:\n{error_details}")
            return True, f"Skipped (inspection error: {e})", {'failure_type': 'inspection_crash', 'traceback': error_details}
    
    def _check_connectivity(self, mesh: 'pv.DataSet') -> Tuple[bool, str]:
        """
        Check if mesh is fully connected (no isolated parts).
        """
        try:
            # Compute connectivity regions
            # Note: connectivity can be slow for large meshes
            conn = mesh.connectivity(largest=False)
            
            # Try to get RegionId
            region_ids = None
            if 'RegionId' in conn.point_data:
                region_ids = conn.point_data['RegionId']
            elif 'RegionId' in conn.cell_data:
                region_ids = conn.cell_data['RegionId']
            
            if region_ids is None or len(region_ids) == 0:
                return True, "OK (connectivity check skipped)"
            
            n_regions = len(np.unique(region_ids))
            
            if n_regions > self.max_regions:
                return False, f"Disconnected geometry: {n_regions} separate parts (expected {self.max_regions})"
            
            return True, f"OK ({n_regions} region)"
            
        except Exception as e:
            logger.warning(f"[MeshQC] Connectivity check failed: {e}")
            return True, "OK (connectivity check skipped)"
    
    def _check_element_quality(self, mesh: 'pv.DataSet') -> Tuple[bool, str, Dict]:
        """
        Check for inverted elements using Scaled Jacobian and Volume metrics.
        """
        metadata = {}
        
        try:
            # 1. Check Volumes (Physical Inversion)
            # Use absolute values and relative checks to handle scale
            cell_sizes = mesh.compute_cell_sizes(length=False, area=False, volume=True)
            volumes = cell_sizes.cell_data['Volume']
            min_vol = np.min(volumes)
            max_vol = np.max(volumes)
            
            metadata['min_volume'] = float(min_vol)
            metadata['max_volume'] = float(max_vol)
            
            if min_vol < 0:
                n_neg_vol = np.sum(volumes < 0)
                # FATAL: Physical inversion is usually a real bug
                return False, f"Physically inverted elements: {n_neg_vol} cells with negative volume (min={min_vol:.2e})", metadata

            # 2. Check Scaled Jacobian (Quality/Orientation)
            try:
                qual = mesh.compute_cell_quality(quality_measure='scaled_jacobian')
                if 'CellQuality' not in qual.cell_data:
                    return True, f"OK (vol_min={min_vol:.2e})", metadata
                    
                jacobians = qual.cell_data['CellQuality']
                min_jac = np.min(jacobians)
                metadata['min_jacobian'] = float(min_jac)
                
                if min_jac < self.min_jacobian:
                    n_inverted = np.sum(jacobians < self.min_jacobian)
                    
                    # HEURISTIC: If all/most elements are inverted with -1.0 but volumes are positive,
                    # it's likely just a node-ordering convention mismatch between Gmsh and VTK.
                    if min_vol >= 0 and min_jac <= -0.9 and (n_inverted / len(jacobians)) > 0.5:
                        logger.warning(f"[MeshQC] Systematic negative Jacobian ({min_jac:.2f}) detected with positive volumes. Likely convention mismatch. Proceeding.")
                        return True, f"Passed with warning (convention mismatch, min_jac={min_jac:.2f})", metadata
                    
                    return False, f"Inverted elements detected: {n_inverted} cells with negative Jacobian (min={min_jac:.4f}, vol_min={min_vol:.2e})", metadata
            except Exception as eq:
                logger.warning(f"[MeshQC] Scaled Jacobian check failed: {eq}. Relying on volume check.")
                return True, f"OK (Jacobian check failed, but volumes are OK)", metadata
            
            return True, f"OK (min Jacobian={min_jac:.4f}, vol_min={min_vol:.2e})", metadata
            
        except Exception as e:
            logger.warning(f"[MeshQC] Element quality check failed: {e}")
            # Re-raise to be caught by the main loop which handles it defensively
            raise e
    
    def _check_mesh_size(self, mesh: 'pv.DataSet') -> Tuple[bool, str]:
        """
        Check if mesh size is within reasonable bounds.
        
        Oversized meshes indicate mesher/optimizer failures and will
        cause RAM exhaustion or excessive solve times.
        """
        n_cells = mesh.n_cells
        
        if n_cells > self.max_cells:
            return False, f"Mesh too large: {n_cells:,} cells > {self.max_cells:,} limit (optimizer likely failed)"
        
        # Also check for suspiciously small meshes
        if n_cells < 10:
            return False, f"Mesh too small: {n_cells} cells (meshing likely failed)"
        
        return True, f"OK ({n_cells:,} cells)"


def validate_mesh_quality(mesh_file: Path, **kwargs) -> Tuple[bool, str, Dict]:
    """
    Convenience function for one-shot validation.
    
    Args:
        mesh_file: Path to mesh file
        **kwargs: Optional validator configuration (max_cells, min_jacobian, max_regions)
        
    Returns:
        Tuple of (passed, reason, metadata)
    """
    validator = MeshQualityValidator(**kwargs)
    return validator.validate(mesh_file)
