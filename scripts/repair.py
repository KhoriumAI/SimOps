"""
Geometry Repair & Health Audit Script
======================================

Standalone triager for incoming STEP/CAD files:
1. AUDIT: Generate JSON health report (watertight, manifold, volume > 0)
2. FIX: Attempt progressive repairs if audit fails
3. BATCH: Process folders of files with summary logging

Usage:
    # Single file
    python scripts/repair.py path/to/file.step
    
    # Batch folder
    python scripts/repair.py path/to/folder --batch
    
    # Output JSON report
    python scripts/repair.py path/to/folder --batch --output report.json

Author: MeshPackageLean Team
Date: 2025-12-30
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    print("[WARNING] gmsh not available - CAD analysis disabled")

try:
    import trimesh
    import numpy as np
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[WARNING] trimesh not available - mesh repair disabled")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class HealthReport:
    """Health report for a single geometry file - JSON serializable for dashboard."""
    
    file_path: str
    file_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Audit results
    is_watertight: bool = False
    is_manifold: bool = False
    volume: float = 0.0
    has_positive_volume: bool = False
    
    # Geometry stats
    num_volumes: int = 0
    num_faces: int = 0
    num_edges: int = 0
    bounding_box: List[float] = field(default_factory=list)
    
    # Issues detected
    issues: List[str] = field(default_factory=list)
    issue_count: int = 0
    
    # Repair status
    repair_attempted: bool = False
    repair_success: bool = False
    repair_method: str = ""
    
    # Overall status
    status: str = "UNKNOWN"  # HEALTHY, REPAIRABLE, UNREPAIRABLE, ERROR
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert numpy types to native Python types for JSON serialization
        for key, value in d.items():
            if hasattr(value, 'item'):  # numpy scalar
                d[key] = value.item()
            elif isinstance(value, (np.bool_, np.generic)):
                d[key] = value.item()
        return d
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


@dataclass
class BatchSummary:
    """Summary of batch processing - for dashboard logging."""
    
    total_files: int = 0
    files_processed: int = 0
    files_healthy: int = 0
    files_with_issues: int = 0
    files_repaired: int = 0
    files_unrepairable: int = 0
    files_errored: int = 0
    processing_time_seconds: float = 0.0
    reports: List[HealthReport] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_files": self.total_files,
                "files_processed": self.files_processed,
                "files_healthy": self.files_healthy,
                "files_with_issues": self.files_with_issues,
                "files_repaired": self.files_repaired,
                "files_unrepairable": self.files_unrepairable,
                "files_errored": self.files_errored,
                "processing_time_seconds": round(self.processing_time_seconds, 2)
            },
            "reports": [r.to_dict() for r in self.reports]
        }
    
    def print_summary(self):
        """Print summary in user-requested format."""
        print("\n" + "=" * 50)
        print("REPAIR SUMMARY")
        print("=" * 50)
        print(f"Files Processed:   {self.files_processed}")
        print(f"Files Healthy:     {self.files_healthy}")
        print(f"Files with Issues: {self.files_with_issues}")
        print(f"Files Repaired:    {self.files_repaired}")
        print(f"Files Unrepairable:{self.files_unrepairable}")
        print(f"Files Errored:     {self.files_errored}")
        print(f"Processing Time:   {self.processing_time_seconds:.2f}s")
        print("=" * 50)


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

def audit_step_file(step_path: str) -> HealthReport:
    """
    Audit a STEP file and generate health report.
    
    Uses:
    - gmsh/OCC for CAD-level analysis (volumes, faces, bounding box)
    - Temporary mesh export + Trimesh for watertight/manifold checks
    """
    report = HealthReport(
        file_path=str(step_path),
        file_name=os.path.basename(step_path)
    )
    
    if not os.path.exists(step_path):
        report.issues.append("File not found")
        report.status = "ERROR"
        return report
    
    if not GMSH_AVAILABLE:
        report.issues.append("gmsh not available")
        report.status = "ERROR"
        return report
    
    temp_stl = None
    
    try:
        # Initialize gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Quiet mode
        
        # Load STEP file
        logger.debug(f"Loading: {step_path}")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # === CAD-LEVEL ANALYSIS ===
        
        # Get entity counts
        volumes = gmsh.model.getEntities(dim=3)
        faces = gmsh.model.getEntities(dim=2)
        edges = gmsh.model.getEntities(dim=1)
        
        report.num_volumes = len(volumes)
        report.num_faces = len(faces)
        report.num_edges = len(edges)
        
        # Check for zero volumes
        if report.num_volumes == 0:
            report.issues.append("No solid volumes detected")
        
        # Get bounding box
        try:
            bbox = gmsh.model.getBoundingBox(-1, -1)  # All entities
            report.bounding_box = list(bbox)
        except:
            report.bounding_box = []
        
        # Calculate total volume using OCC mass properties
        total_volume = 0.0
        for dim, tag in volumes:
            try:
                mass = gmsh.model.occ.getMass(dim, tag)
                total_volume += mass
            except:
                pass
        
        report.volume = total_volume
        report.has_positive_volume = total_volume > 1e-10
        
        if not report.has_positive_volume:
            report.issues.append(f"Zero or negative volume: {total_volume:.2e}")
        
        # === MESH-LEVEL CHECKS (via Trimesh) ===
        
        if TRIMESH_AVAILABLE and report.num_volumes > 0:
            # Generate temporary mesh for Trimesh analysis
            try:
                # Coarse mesh for quick analysis
                gmsh.option.setNumber("Mesh.MeshSizeMin", 1.0)
                gmsh.option.setNumber("Mesh.MeshSizeMax", 10.0)
                gmsh.model.mesh.generate(2)  # Surface mesh only
                
                # Export to temp STL
                temp_stl = os.path.join(PROJECT_ROOT, "temp_audit.stl")
                gmsh.write(temp_stl)
                
                # Analyze with Trimesh
                mesh = trimesh.load(temp_stl, force='mesh', process=False)
                mesh.merge_vertices()
                
                report.is_watertight = mesh.is_watertight
                
                # Check manifold (edge manifold = each edge shared by exactly 2 faces)
                if hasattr(mesh, 'edges_unique') and hasattr(mesh, 'edges_unique_length'):
                    # Edges appearing more than 2 times are non-manifold
                    edge_counts = np.bincount(mesh.edges_unique_inverse)
                    non_manifold_edges = np.sum(edge_counts > 2)
                    report.is_manifold = non_manifold_edges == 0
                    
                    if not report.is_manifold:
                        report.issues.append(f"Non-manifold edges: {non_manifold_edges}")
                else:
                    # Fallback: use is_watertight as proxy
                    report.is_manifold = mesh.is_watertight
                
                if not report.is_watertight:
                    report.issues.append("Geometry is not watertight (has holes/gaps)")
                    
            except Exception as e:
                report.issues.append(f"Mesh analysis failed: {str(e)}")
            finally:
                # Cleanup temp file
                if temp_stl and os.path.exists(temp_stl):
                    try:
                        os.remove(temp_stl)
                    except:
                        pass
        
        # === DETERMINE STATUS ===
        report.issue_count = len(report.issues)
        
        if report.issue_count == 0:
            report.status = "HEALTHY"
        else:
            report.status = "NEEDS_REPAIR"
        
    except Exception as e:
        report.issues.append(f"Audit error: {str(e)}")
        report.status = "ERROR"
        logger.error(f"Audit failed for {step_path}: {e}")
    
    finally:
        try:
            gmsh.finalize()
        except:
            pass
        
        # Extra cleanup
        if temp_stl and os.path.exists(temp_stl):
            try:
                os.remove(temp_stl)
            except:
                pass
    
    return report


# =============================================================================
# REPAIR FUNCTIONS
# =============================================================================

def attempt_repair(step_path: str, report: HealthReport) -> HealthReport:
    """
    Attempt progressive repairs on a geometry file.
    
    Repair Strategy (progressive):
    1. OCC healShapes (CAD-level repair)
    2. Trimesh vertex merge + winding fix (mesh-level)
    3. Convex hull fallback (last resort)
    """
    if report.status == "HEALTHY":
        return report  # No repair needed
    
    report.repair_attempted = True
    repaired_path = step_path.replace(".step", "_repaired.step").replace(".STEP", "_repaired.STEP")
    
    # === LEVEL 1: OCC Healing ===
    try:
        logger.info(f"  Attempting OCC healShapes...")
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        
        # Apply OCC healing
        volumes = gmsh.model.getEntities(dim=3)
        if len(volumes) > 0:
            # healShapes attempts to fix small gaps, overlaps, etc.
            gmsh.model.occ.healShapes()
            gmsh.model.occ.synchronize()
            
            # Re-check volume
            new_volume = 0.0
            for dim, tag in gmsh.model.getEntities(dim=3):
                try:
                    new_volume += gmsh.model.occ.getMass(dim, tag)
                except:
                    pass
            
            if new_volume > 1e-10:
                # Export healed geometry
                gmsh.write(repaired_path)
                report.repair_success = True
                report.repair_method = "OCC_healShapes"
                report.status = "REPAIRED"
                logger.info(f"  ✓ OCC repair successful")
        
        gmsh.finalize()
        
    except Exception as e:
        logger.debug(f"  OCC repair failed: {e}")
        try:
            gmsh.finalize()
        except:
            pass
    
    # === LEVEL 2: Trimesh Repair (if OCC failed) ===
    if not report.repair_success and TRIMESH_AVAILABLE:
        try:
            logger.info(f"  Attempting Trimesh repair...")
            
            # Need to convert to mesh first
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            
            temp_stl = os.path.join(PROJECT_ROOT, "temp_repair.stl")
            gmsh.write(temp_stl)
            gmsh.finalize()
            
            # Load and repair with Trimesh
            mesh = trimesh.load(temp_stl, force='mesh', process=False)
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            
            # Apply repairs
            try:
                trimesh.repair.fix_winding(mesh)
                trimesh.repair.fix_inversion(mesh)
            except:
                pass
            
            if mesh.is_watertight:
                repaired_stl = step_path.replace(".step", "_repaired.stl").replace(".STEP", "_repaired.stl")
                mesh.export(repaired_stl)
                report.repair_success = True
                report.repair_method = "Trimesh_winding_fix"
                report.status = "REPAIRED"
                logger.info(f"  ✓ Trimesh repair successful")
            
            # Cleanup
            if os.path.exists(temp_stl):
                os.remove(temp_stl)
                
        except Exception as e:
            logger.debug(f"  Trimesh repair failed: {e}")
            try:
                gmsh.finalize()
            except:
                pass
    
    # === LEVEL 3: Convex Hull Fallback ===
    if not report.repair_success and TRIMESH_AVAILABLE:
        try:
            logger.info(f"  Attempting convex hull fallback...")
            
            # Generate mesh from original
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.occ.importShapes(step_path)
            gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(2)
            
            temp_stl = os.path.join(PROJECT_ROOT, "temp_hull.stl")
            gmsh.write(temp_stl)
            gmsh.finalize()
            
            mesh = trimesh.load(temp_stl, force='mesh', process=False)
            hull = mesh.convex_hull
            
            if hull.is_watertight and hull.volume > 1e-10:
                hull_stl = step_path.replace(".step", "_hull.stl").replace(".STEP", "_hull.stl")
                hull.export(hull_stl)
                report.repair_success = True
                report.repair_method = "Convex_hull"
                report.status = "REPAIRED"
                report.issues.append("WARNING: Replaced with convex hull (detail lost)")
                logger.info(f"  ✓ Convex hull fallback applied")
            
            # Cleanup
            if os.path.exists(temp_stl):
                os.remove(temp_stl)
                
        except Exception as e:
            logger.debug(f"  Convex hull failed: {e}")
            try:
                gmsh.finalize()
            except:
                pass
    
    # Final status
    if not report.repair_success:
        report.status = "UNREPAIRABLE"
        logger.warning(f"  ✗ All repair methods failed")
    
    return report


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_folder(folder_path: str, attempt_fixes: bool = True) -> BatchSummary:
    """
    Process all STEP files in a folder.
    
    Args:
        folder_path: Path to folder containing STEP files
        attempt_fixes: Whether to attempt repairs on bad files
        
    Returns:
        BatchSummary with all reports and statistics
    """
    summary = BatchSummary()
    start_time = time.time()
    
    # Find all STEP files
    step_extensions = ['.step', '.stp', '.STEP', '.STP']
    step_files = []
    
    folder = Path(folder_path)
    for ext in step_extensions:
        step_files.extend(folder.glob(f'*{ext}'))
    
    summary.total_files = len(step_files)
    logger.info(f"Found {summary.total_files} STEP files in {folder_path}")
    
    # Process each file
    for i, step_file in enumerate(step_files, 1):
        logger.info(f"[{i}/{summary.total_files}] Processing: {step_file.name}")
        
        try:
            # Audit
            report = audit_step_file(str(step_file))
            summary.files_processed += 1
            
            if report.status == "HEALTHY":
                summary.files_healthy += 1
                logger.info(f"  → HEALTHY")
            elif report.status == "ERROR":
                summary.files_errored += 1
                logger.error(f"  → ERROR: {report.issues}")
            else:
                summary.files_with_issues += 1
                logger.warning(f"  → Issues: {report.issues}")
                
                # Attempt repair
                if attempt_fixes:
                    report = attempt_repair(str(step_file), report)
                    
                    if report.repair_success:
                        summary.files_repaired += 1
                    else:
                        summary.files_unrepairable += 1
            
            summary.reports.append(report)
            
        except Exception as e:
            logger.error(f"  → FATAL ERROR: {e}")
            error_report = HealthReport(
                file_path=str(step_file),
                file_name=step_file.name,
                status="ERROR",
                issues=[f"Fatal error: {str(e)}"]
            )
            summary.reports.append(error_report)
            summary.files_errored += 1
    
    summary.processing_time_seconds = time.time() - start_time
    return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Geometry Health Audit & Repair Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python repair.py model.step                    # Audit single file
  python repair.py model.step --repair           # Audit and repair
  python repair.py ./cad_files --batch           # Batch process folder
  python repair.py ./cad_files --batch --output report.json
        """
    )
    
    parser.add_argument("path", help="STEP file or folder path")
    parser.add_argument("--batch", action="store_true", help="Process folder of files")
    parser.add_argument("--repair", action="store_true", help="Attempt repairs on bad files")
    parser.add_argument("--output", "-o", help="Output JSON report path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate path
    if not os.path.exists(args.path):
        logger.error(f"Path not found: {args.path}")
        sys.exit(1)
    
    # Process
    if args.batch or os.path.isdir(args.path):
        # Batch folder processing
        summary = process_folder(args.path, attempt_fixes=args.repair)
        summary.print_summary()
        
        # Output JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)
            logger.info(f"Report saved to: {args.output}")
        
    else:
        # Single file processing
        report = audit_step_file(args.path)
        
        if args.repair and report.status != "HEALTHY":
            report = attempt_repair(args.path, report)
        
        # Print report
        print("\n" + "=" * 50)
        print(f"HEALTH REPORT: {report.file_name}")
        print("=" * 50)
        print(f"Status:        {report.status}")
        print(f"Watertight:    {report.is_watertight}")
        print(f"Manifold:      {report.is_manifold}")
        print(f"Volume:        {report.volume:.6f}")
        print(f"Volumes:       {report.num_volumes}")
        print(f"Faces:         {report.num_faces}")
        
        if report.issues:
            print(f"\nIssues ({len(report.issues)}):")
            for issue in report.issues:
                print(f"  - {issue}")
        
        if report.repair_attempted:
            print(f"\nRepair Attempted: {report.repair_method or 'N/A'}")
            print(f"Repair Success:   {report.repair_success}")
        
        print("=" * 50)
        
        # Output JSON if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report.to_json())
            logger.info(f"Report saved to: {args.output}")
        
        # Exit code based on status
        if report.status == "HEALTHY" or report.status == "REPAIRED":
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
