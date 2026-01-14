import json
import logging
import os
import shutil
from datetime import datetime
from typing import Optional

from .models import GeometryStatus, GuardianResult
from .inspectors import TopologyInspector
from .repairers import ManifoldHealer
from .cleaners import GeometryCleanupTool

class GeometryGuardian:
    """
    The Gatekeeper.
    Orchestrates the Diagnose -> Decide -> Repair workflow.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("GeometryGuardian")
        
        # Instantiate the workers
        self.inspector = TopologyInspector()
        self.healer = ManifoldHealer()
        self.cleaner = GeometryCleanupTool(min_feature_size=self.config.get('min_feature_size'))

    def sanitize(self, input_path: str) -> GuardianResult:
        """
        Main entry point. Validates and optionally repairs the input geometry.
        
        Returns:
            GuardianResult: Contains the final file path and status.
        """
        start_time = datetime.now()
        max_repair_time = self.config.get('max_repair_time', 60) # Default 60s timeout
        
        lifecycle = []
        report_path = list( os.path.splitext(input_path) )
        report_path = f"{report_path[0]}.guardian_report.json"
        
        # --- 1. BASELINE INSPECTION (The "Doctor") ---
        self.logger.info(f"Inspecting geometry: {os.path.basename(input_path)}")
        
        try:
            # Check for fatal topology errors (Manifold/Watertight)
            scan_result = self.inspector.scan(input_path)
            lifecycle.append({
                "phase": "BASELINE_SCAN", 
                "timestamp": datetime.now().isoformat(),
                "result": scan_result
            })
            
        except Exception as e:
            self.logger.error(f"Inspection crashed: {e}")
            # FAIL OPEN: If diagnostics crash, warn but let the file through
            return GuardianResult(GeometryStatus.WARNING, input_path, report_path, input_path, lifecycle)

        # --- 2. DECISION MATRIX ---
        
        # CASE A: File is Healthy
        if scan_result.get('is_manifold', False):
            self.logger.info("Geometry Passed Inspection (Status: PRISTINE).")
            self._write_report(report_path, GeometryStatus.PRISTINE, lifecycle)
            return GuardianResult(GeometryStatus.PRISTINE, input_path, report_path, input_path, lifecycle)

        # CASE B: File is Critical (Needs Repair)
        self.logger.warning(f"Geometry Critical: {scan_result.get('details')}. Initiating Repair.")
        
        # --- 3. REPAIR PIPELINE (The "Surgeon") ---
        
        # Define the target path for the repaired file
        base, ext = os.path.splitext(input_path)
        restored_path = f"{base}_restored{ext}"
        
        # Strategy: Waterfall (OCC -> Trimesh -> Hull)
        strategies = [
            {"name": "OCC_Standard", "method": "occ", "fragment": False},
            {"name": "OCC_Aggressive", "method": "occ", "fragment": True},
            {"name": "Trimesh_Fix", "method": "trimesh"}, 
            {"name": "Convex_Hull_Fallback", "method": "convex_hull"}
        ]

        for strategy in strategies:
            # TIMEOUT CHECK
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > max_repair_time:
                self.logger.warning(f"Guardian repair timed out after {elapsed:.1f}s (Limit: {max_repair_time}s). Skipping remaining strategies.")
                lifecycle.append({"phase": "TIMEOUT", "elapsed": elapsed})
                break

            try:
                self.logger.info(f"Attempting Repair Strategy: {strategy['name']}")
                
                # [FIX] Capture the RETURN VALUE (it might be a new path like .stl)
                actual_repaired_path = self.healer.heal(input_path, restored_path, strategy)
                
                if actual_repaired_path:
                    # VERIFICATION: Scan the file that was ACTUALLY produced
                    verify_result = self.inspector.scan(actual_repaired_path)
                    
                    lifecycle.append({
                        "phase": "REPAIR_ATTEMPT",
                        "strategy": strategy['name'],
                        "success": True,
                        "verification": verify_result,
                        "output": actual_repaired_path
                    })

                    if verify_result.get('is_manifold'):
                        self.logger.info(f"Repair Successful via {strategy['name']}. Swapping files.")
                        self._write_report(report_path, GeometryStatus.RESTORED, lifecycle)
                        return GuardianResult(
                            GeometryStatus.RESTORED, 
                            actual_repaired_path,  # [FIX] Return the dynamic path
                            report_path, 
                            input_path, 
                            lifecycle
                        )
                    else:
                        self.logger.warning(f"Strategy {strategy['name']} produced {actual_repaired_path} but it failed verification.")
                else:
                    self.logger.warning(f"Repair Strategy {strategy['name']} failed to produce output.")
                    lifecycle.append({"phase": "REPAIR_ATTEMPT", "strategy": strategy['name'], "success": False})

            except Exception as e:
                self.logger.error(f"Repair Strategy {strategy['name']} crashed: {e}")
                lifecycle.append({"phase": "REPAIR_ERROR", "error": str(e)})

        # --- 4. TERMINAL STATE ---
        # If we reach here, all repairs failed.
        self.logger.error("All repair strategies failed. Geometry is TERMINAL.")
        self._write_report(report_path, GeometryStatus.TERMINAL, lifecycle)
        
        return GuardianResult(GeometryStatus.TERMINAL, input_path, report_path, input_path, lifecycle)

    def _write_report(self, path, status, lifecycle):
        """Writes the flight recorder data to JSON."""
        try:
            with open(path, 'w') as f:
                json.dump({
                    "status": status.value, 
                    "timestamp": datetime.now().isoformat(),
                    "lifecycle": lifecycle
                }, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not write Guardian report: {e}")