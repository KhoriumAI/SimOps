"""
Workflow Manager
================

Handles the mesh generation workflow, including starting/stopping workers,
updating progress, and handling completion events.
"""

import time
from pathlib import Path
from PyQt5.QtWidgets import QLabel, QProgressBar
from PyQt5.QtCore import Qt

class WorkflowManager:
    def __init__(self, window):
        self.window = window

    def start_mesh_generation(self):
        if not self.window.cad_file:
            return

        self.window.generate_btn.setEnabled(False)
        self.window.stop_btn.setEnabled(True)
        self.window.console.clear()

        # Reset progress
        for bar in self.window.phase_bars.values():
            bar.setValue(0)
            bar.setStyleSheet(bar.styleSheet().replace("background-color: #198754", "background-color: #0d6efd"))
        
        self.window.mesh_start_time = time.time()
        self.window.completed_phases = []
        self.window.phase_completion_times = {}
        self.window.master_progress = 0.0
        
        # Calculate weights
        element_target = self.window.target_elements.value()
        quality_preset = self.window.quality_preset.currentText()
        self.window.phase_weights = self.window.calculate_phase_weights(element_target, quality_preset)
        
        # Reset bars
        if hasattr(self.window, 'master_bar'):
            self.window.master_bar.setValue(0)
            self.window.master_bar.setFormat("0% - Starting...")
        if hasattr(self.window, 'current_process_bar'):
            self.window.current_process_bar.setValue(0)
        if hasattr(self.window, 'current_process_label'):
            self.window.current_process_label.setText("Initializing...")
            
        # Clear completed stages
        if hasattr(self.window, 'completed_stages_layout'):
            while self.window.completed_stages_layout.count() > 1: # Keep stretch
                item = self.window.completed_stages_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # Collect params
        quality_params = {
            "quality_preset": self.window.quality_preset.currentText(),
            "target_elements": self.window.target_elements.value(),
            "max_size_mm": self.window.max_size.value(),
            "curvature_adaptive": self.window.curvature_adaptive.isChecked(),
            "mesh_strategy": self.window.mesh_strategy.currentText(),
            "save_stl": self.window.save_stl.isChecked()
        }

        # Paintbrush
        if self.window.paintbrush_selector and self.window.paintbrush_selector.get_painted_regions():
            painted_regions = self.window.paintbrush_selector.get_painted_regions()
            quality_params["painted_regions"] = [region.to_dict() for region in painted_regions]
            self.window.add_log(f"[ART] Paintbrush Refinement: {len(painted_regions)} regions")

        self.window.last_quality_params = quality_params

        self.window.add_log("=" * 70)
        self.window.add_log("Starting PARALLEL mesh generation...")
        self.window.add_log(f"Strategy: {quality_params['mesh_strategy']}")
        self.window.add_log("=" * 70)

        self.window.worker.start(self.window.cad_file, quality_params)

    def stop_mesh_generation(self):
        if self.window.worker and self.window.worker.is_running:
            self.window.add_log("\n[!] Stopping mesh generation...")
            self.window.worker.stop()
            self.window.add_log("[OK] Mesh generation stopped by user")

            self.window.generate_btn.setEnabled(True)
            self.window.stop_btn.setEnabled(False)

            if self.window.animation_timer.isActive():
                self.window.animation_timer.stop()
            self.window.active_phase = None

    def on_mesh_finished(self, success, result):
        self.window.generate_btn.setEnabled(True)
        self.window.stop_btn.setEnabled(False)
        self.window.refine_btn.setEnabled(True)

        if self.window.animation_timer.isActive():
            self.window.animation_timer.stop()

        if success and result.get("success"):
            self.window.mesh_file = result["output_file"]
            self.window.add_log("\n" + "="*70)
            self.window.add_log("MESH GENERATION SUCCESSFUL")
            self.window.add_log(f"Output: {self.window.mesh_file}")
            self.window.add_log("="*70)

            # Update master bar
            if hasattr(self.window, 'master_bar'):
                self.window.master_bar.setValue(100)
                elapsed = time.time() - self.window.mesh_start_time
                self.window.master_bar.setFormat(f"100% - Complete ({elapsed:.1f}s)")
                self.window.master_bar.setStyleSheet(self.window.master_bar.styleSheet().replace("#0d6efd", "#198754"))

            # Handle Polyhedral Visualization
            if result.get('visualization_mode') == 'polyhedral':
                poly_file = result.get('polyhedral_data_file')
                if poly_file and Path(poly_file).exists():
                    self.window.add_log(f"[DEBUG] Loading polyhedral data from {poly_file}")
                    status = self.window.viewer.load_polyhedral_file(poly_file)
                    if status != "SUCCESS":
                        self.window.add_log("[WARN] Polyhedral load failed, falling back to surface mesh")
                        self.window.viewer.load_mesh_file(self.window.mesh_file, result)
                else:
                    self.window.add_log("[WARN] Polyhedral data missing, loading surface mesh")
                    self.window.viewer.load_mesh_file(self.window.mesh_file, result)
            
            # Handle Surface Visualization (Fallback for Polyhedral)
            elif result.get('visualization_mode') == 'surface':
                self.window.add_log(f"[DEBUG] Loading surface visualization...")
                self.window.viewer.load_mesh_file(self.window.mesh_file, result)
                
            # Handle Standard Mesh
            else:
                self.window.add_log(f"[WORKFLOW DEBUG] Standard mesh - calling load_mesh_file")
                self.window.add_log(f"[WORKFLOW DEBUG] mesh_file: {self.window.mesh_file}")
                self.window.add_log(f"[WORKFLOW DEBUG] viewer: {self.window.viewer}")
                self.window.add_log(f"[WORKFLOW DEBUG] has load_mesh_file: {hasattr(self.window.viewer, 'load_mesh_file')}")
                load_result = self.window.viewer.load_mesh_file(self.window.mesh_file, result)
                self.window.add_log(f"[WORKFLOW DEBUG] load_mesh_file returned: {load_result}")

            # Update quality ranges
            if 'per_element_quality' in result:
                self.window.update_quality_data_ranges(
                    result.get('per_element_quality'),
                    result.get('per_element_gamma'),
                    result.get('per_element_skewness'),
                    result.get('per_element_aspect_ratio')
                )
                
            # Show quality report
            if 'quality_metrics' in result:
                self.window.viewer.show_quality_report(result['quality_metrics'])
            
            # Log timing data
            self.log_mesh_timing(result)
                
        else:
            self.window.add_log("\n[!] MESH GENERATION FAILED")
            self.window.add_log(f"Error: {result.get('error')}")
            if hasattr(self.window, 'master_bar'):
                self.window.master_bar.setStyleSheet(self.window.master_bar.styleSheet().replace("#0d6efd", "#dc3545"))
                self.window.master_bar.setFormat("Failed")

    def log_mesh_timing(self, result):
        """Log mesh timing data to CSV for calibration"""
        try:
            import csv
            import datetime
            import os
            
            # Get timing data
            timings = result.get('timing_breakdown', {})
            total_time = time.time() - self.window.mesh_start_time
            
            # Get geometry info
            geom_info = getattr(self.window, 'current_geom_info', {}) or {}
            volume = geom_info.get('volume', 0)
            
            # Prepare row data
            row = {
                'timestamp': datetime.datetime.now().isoformat(),
                'filename': Path(self.window.cad_file).name if self.window.cad_file else "Unknown",
                'strategy': result.get('strategy', 'Unknown'),
                'total_elements': result.get('total_elements', 0),
                'total_nodes': result.get('total_nodes', 0),
                'volume_m3': volume,
                'total_time_s': round(total_time, 2),
                '2d_mesh_s': round(timings.get('2d_mesh', 0), 2),
                'surface_opt_s': round(timings.get('surface_opt', 0), 2),
                '3d_mesh_s': round(timings.get('3d_mesh', 0), 2),
                'volume_opt_s': round(timings.get('volume_opt', 0), 2),
                'quad_conv_s': round(timings.get('quadratic_conversion', 0), 2),
                'est_elements': self.window.target_elements.value()
            }
            
            # Define CSV file path
            log_dir = Path.home() / "Downloads" / "MeshPackageLean"
            log_file = log_dir / "mesh_timing_log.csv"
            
            # Check if file exists to write header
            file_exists = log_file.exists()
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                
            self.window.add_log(f"[LOG] Timing data saved to {log_file.name}")
            
        except Exception as e:
            print(f"Failed to log timing data: {e}")

    def refine_mesh_quality(self):
        if not self.window.mesh_file or not Path(self.window.mesh_file).exists():
            self.window.add_log("[!] No mesh loaded")
            return

        self.window.add_log("\n" + "=" * 70)
        self.window.add_log("MESH QUALITY REFINEMENT")
        self.window.add_log("=" * 70)

        self.window.refine_btn.setEnabled(False)
        self.window.generate_btn.setEnabled(False)

        try:
            import gmsh
            gmsh.initialize()
            gmsh.open(self.window.mesh_file)
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)

            num_passes = 5
            for i in range(num_passes):
                self.window.add_log(f"\nOptimization pass {i+1}/{num_passes}...")
                gmsh.model.mesh.optimize("Netgen")

            gmsh.write(self.window.mesh_file)
            gmsh.finalize()

            self.window.add_log("\n[OK] REFINEMENT COMPLETE")
            self.window.viewer.load_mesh_file(self.window.mesh_file, {})

        except Exception as e:
            self.window.add_log(f"\n[!] Refinement failed: {e}")
            import traceback
            self.window.add_log(traceback.format_exc())

        finally:
            self.window.refine_btn.setEnabled(True)
            self.window.generate_btn.setEnabled(True)

    def update_progress(self, phase, percentage, details=""):
        # Handle CoACD Preview
        if phase == 'coacd_preview':
            self.window.add_log(f"[DEBUG] Received CoACD preview update")
            if 'component_files' in details:
                self.window.viewer.load_component_visualization(details)
            return

        # Update current process bar
        if hasattr(self.window, 'current_process_bar'):
            self.window.current_process_bar.setValue(int(percentage))
            self.window.current_process_bar.setFormat(f"{int(percentage)}%")
            
        # Update label
        if hasattr(self.window, 'current_process_label'):
            phase_display = self.window.phase_base_names.get(phase, phase.replace('_', ' ').title())
            self.window.current_process_label.setText(f"{phase_display}: {details}")

        # Master progress calculation
        if phase in self.window.phase_weights:
            # Calculate completed portion
            completed_weight = sum(self.window.phase_weights.get(p, 0) for p in self.window.completed_phases)
            
            # Calculate current phase portion
            current_weight = self.window.phase_weights.get(phase, 0)
            current_contribution = (percentage / 100.0) * current_weight
            
            total_progress = min(99, completed_weight + current_contribution)
            
            if total_progress > self.window.master_progress:
                self.window.master_progress = total_progress
                if hasattr(self.window, 'master_bar'):
                    self.window.master_bar.setValue(int(total_progress))
                    self.window.update_eta(total_progress)

        # Animation
        if self.window.active_phase != phase:
            self.window.active_phase = phase
            if not self.window.animation_timer.isActive():
                self.window.animation_timer.start()

    def mark_phase_complete(self, phase):
        if phase not in self.window.completed_phases:
            self.window.completed_phases.append(phase)
            self.window.phase_completion_times[phase] = time.time()
            
            # Add to completed list UI
            if hasattr(self.window, 'completed_stages_layout'):
                phase_name = self.window.phase_base_names.get(phase, phase.replace('_', ' ').title())
                
                # Create completion widget
                lbl = QLabel(f"✓ {phase_name}")
                lbl.setStyleSheet("color: #198754; font-weight: bold; font-size: 10px;")
                self.window.completed_stages_layout.insertWidget(self.window.completed_stages_layout.count()-1, lbl)
