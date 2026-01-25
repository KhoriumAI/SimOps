"""
Mesh Worker - Background Threading
====================================

Subprocess mesh generation worker with progress tracking.
"""

import sys
import os
import json
import subprocess
import threading
import tempfile
import re
import psutil  # For process tree termination
from pathlib import Path
from multiprocessing import cpu_count
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# API Contract for type-safe communication
try:
    # Import from project root
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core.api_contract import MeshJobRequest, MeshJobResponse, JobStatus
    CONTRACT_AVAILABLE = True
except ImportError:
    CONTRACT_AVAILABLE = False


class WorkerSignals(QObject):
    """Signals for mesh worker thread"""
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int)  # (phase, percentage)
    phase_complete = pyqtSignal(str)  # Signal when phase is 100% done
    intermediate_update = pyqtSignal(str, str) # (phase, file_path)
    finished = pyqtSignal(bool, dict)


class MeshWorker:
    """Subprocess mesh generation worker"""

    def __init__(self):
        self.signals = WorkerSignals()
        self.thread = None
        self.process = None
        self.is_running = False
        self.phase_max = {}  # Track max value reached for each phase
        self.temp_config_file = None
        
        # Geometry counters for progress tracking
        self.total_curves = 1
        self.total_surfaces = 1
        self.current_curves = 0
        self.current_surfaces = 0

    def start(self, cad_file: str, quality_params: dict = None):
        if self.is_running:
            return

        self.is_running = True
        self.phase_max = {}
        self.current_curves = 0
        self.current_surfaces = 0
        self.thread = threading.Thread(target=self._run, args=(cad_file, quality_params), daemon=True)
        self.thread.start()

    def _emit_progress(self, phase: str, value: int):
        """Emit progress only if it's higher than before (prevent jitter)"""
        if phase not in self.phase_max:
            self.phase_max[phase] = 0

        if value > self.phase_max[phase]:
            self.phase_max[phase] = value
            self.signals.progress.emit(phase, value)

    def _complete_phase(self, phase: str):
        """Mark phase as 100% complete"""
        self.phase_max[phase] = 100
        self.signals.progress.emit(phase, 100)
        self.signals.phase_complete.emit(phase)

    def _parse_and_emit_line(self, line: str):
        """Parse a single log line and emit appropriate signals"""
        if not line:
            return

        # Parse gmsh Info lines
        if "Info" in line:
            # Geometry setup
            if "Geometry:" in line:
                # [17:31:52] Geometry: 1 volumes, 102 surfaces, 256 curves
                curves_match = re.search(r'(\d+) curves', line)
                surfaces_match = re.search(r'(\d+) surfaces', line)
                if curves_match:
                    self.total_curves = max(1, int(curves_match.group(1)))
                if surfaces_match:
                    self.total_surfaces = max(1, int(surfaces_match.group(1)))

            # 1D Meshing
            if "Meshing 1D" in line:
                self._emit_progress("1d", 1)
                self.current_curves = 0
            elif "Meshing curve" in line:
                self.current_curves += 1
                # Emit 5% increments as checkpoints
                step_size = max(1, self.total_curves // 20)
                if self.current_curves % step_size == 0 or self.current_curves == self.total_curves:
                    pct = int((self.current_curves / self.total_curves) * 100)
                    self._emit_progress("1d", pct)
            elif "Done meshing 1D" in line:
                self._complete_phase("1d")

            # 2D Meshing
            elif "Meshing 2D" in line:
                self._emit_progress("2d", 1)
                self.current_surfaces = 0
            elif "Meshing surface" in line:
                self.current_surfaces += 1
                # Emit 5% increments as checkpoints
                step_size = max(1, self.total_surfaces // 20)
                if self.current_surfaces % step_size == 0 or self.current_surfaces == self.total_surfaces:
                    pct = int((self.current_surfaces / self.total_surfaces) * 100)
                    self._emit_progress("2d", pct)
            elif "Done meshing 2D" in line:
                self._complete_phase("2d")

            # 3D Meshing
            elif "Meshing 3D" in line:
                self._emit_progress("3d", 5)
            elif "Point insertion" in line or "Tetrahedrizing" in line or "Swapping" in line:
                # HXT / Delaunay progress logs: [ 10%]  12345 elements
                # Or sometimes: Info    : Point insertion... [  0%]
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    # Scale to 70% of 3D phase
                    self._emit_progress("3d", 5 + int(pct * 0.7))
                else:
                    # Counter suggestion: if we see an element count, log it
                    # Info    : Point insertion... 123456 elements
                    elem_match = re.search(r'(\d+)\s+elements', line)
                    if elem_match:
                        count = int(elem_match.group(1))
                        if count > 1000: # Only log significant numbers
                            self.signals.log.emit(f"[3D] Refining: {count:,} elements...")
            elif "Reconstructing mesh" in line:
                self._emit_progress("3d", 80)
            elif "3D refinement" in line:
                self._emit_progress("3d", 90)
            elif "Done meshing 3D" in line:
                self._complete_phase("3d")

            # Quality Analysis
            elif "[Quality]" in line:
                # [Quality] SICN progress: 50% (523,456 / 1,046,912 tets)
                match = re.search(r'progress: (\d+)%', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("qual", pct)
                
                # Metric context to log
                if "SICN" in line:
                    self.signals.log.emit("[Quality] Calculating SICN (Signed Inverse Condition Number)...")
                elif "Gamma" in line:
                    self.signals.log.emit("[Quality] Calculating Gamma (Inscribed/Circumscribed ratio)...")
                elif "Angle" in line:
                    self.signals.log.emit("[Quality] Calculating Dihedral Angles...")

            # Polyhedral Conversion Progress
            elif "[Polyhedral]" in line:
                # [Polyhedral] Conversion progress: 45% (450/1000 nodes processed)
                match = re.search(r'progress: (\d+)%', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("3d", pct) # Polyhedral uses 3D bar

            # OpenFOAM Snapping Progress
            elif "[OpenFOAM]" in line:
                # [OpenFOAM] Snapping progress: 55% (5500/10000 points)
                match = re.search(r'Snapping progress: (\d+)%', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("3d", pct)

            # Optimization (Gmsh)
            elif "Optimizing mesh..." in line and "Netgen" not in line:
                self._emit_progress("opt", 10)
            elif "edge swaps" in line:
                self._emit_progress("opt", 60)
            elif "No ill-shaped tets" in line:
                self._emit_progress("opt", 90)
                self._emit_progress("qual", 95) # Also hint at qual starting
            elif "Done optimizing mesh" in line and "Netgen" not in line:
                self._complete_phase("opt")

            # Optimization (Netgen)
            elif "Optimizing mesh (Netgen)" in line:
                self._emit_progress("netgen", 10)
            elif "SplitImprove" in line:
                self._emit_progress("netgen", min(self.phase_max.get("netgen", 10) + 10, 80))
            elif "SwapImprove" in line:
                self._emit_progress("netgen", min(self.phase_max.get("netgen", 10) + 5, 85))
            elif "CombineImprove" in line:
                self._emit_progress("netgen", min(self.phase_max.get("netgen", 10) + 5, 90))
            elif "Done optimizing mesh (Wall" in line:
                self._complete_phase("netgen")

            # Higher order
            elif "Meshing order 2" in line:
                self._emit_progress("order2", 10)
            elif "order 2" in line:
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("order2", 10 + int(pct * 0.9))
            elif "Done meshing order 2" in line:
                self._complete_phase("order2")

        # Strategy attempts
        if "ATTEMPT" in line:
            match = re.search(r'ATTEMPT\s+(\d+)/(\d+)', line)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                pct = int((current / total) * 100)
                self._emit_progress("strategy", pct)

        # Quality analysis
        if "Analyzing mesh quality" in line:
            self._emit_progress("quality", 50)
        elif "Quality Score" in line:
            self._emit_progress("quality", 90)

        # Final result
        if line.startswith('{') and '"success"' in line:
            print(f"[GUI-WORKER] Found JSON result line: {line[:100]}...")
            try:
                result = json.loads(line)
                print(f"[GUI-WORKER] Parsed result, success={result.get('success')}")
                if result.get('success'):
                    # Load full result from file if available (prevents console spam)
                    if 'full_result_file' in result:
                        file_path = result['full_result_file']
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r') as f:
                                    full_result = json.load(f)
                                # Update result with full data
                                result.update(full_result)
                                print(f"[GUI-WORKER] Loaded full result from: {file_path}")
                            except Exception as e:
                                print(f"[GUI-WORKER] Failed to load full result file: {e}")

                    # Mark all active phases as complete
                    for phase in ["strategy", "1d", "2d", "3d", "opt", "netgen", "order2", "quality"]:
                        if self.phase_max.get(phase, 0) > 0:
                            self._complete_phase(phase)

                    self.signals.log.emit("Mesh generation completed!")
                    self.signals.log.emit(f"[DEBUG] Emitting finished signal with {len(result)} metrics")
                    self.signals.finished.emit(True, result)
                else:
                    self.signals.log.emit(f"Failed: {result.get('error')}")
                    self.signals.finished.emit(False, result)
            except Exception as e:
                print(f"[GUI-WORKER] Failed to parse JSON: {e}")
                pass
        else:
            # Debug: Show lines that look like they might be JSON
            if line.startswith('{') or ('"success"' in line and not line.startswith('Info')):
                print(f"[GUI-WORKER] Potential JSON line (didn't match): {line[:100]}")
            
            # Check for display updates
            if "[DISPLAY_UPDATE]" in line:
                try:
                    # Parse: [DISPLAY_UPDATE] phase=surface file=...
                    phase_match = re.search(r'phase=(\w+)', line)
                    file_match = re.search(r'file=(.+)', line)
                    if phase_match and file_match:
                        phase = phase_match.group(1)
                        file_path = file_match.group(1).strip()
                        self.signals.intermediate_update.emit(phase, file_path)
                        print(f"[GUI-WORKER] Emitting intermediate update: {phase} -> {file_path}")
                except Exception as e:
                    print(f"[GUI-WORKER] Failed to parse intermediate update: {e}")

            self.signals.log.emit(line)

    def _run(self, cad_file: str, quality_params: dict = None):
        try:
            self.signals.log.emit(f"Loading: {Path(cad_file).name}")
            self._emit_progress("strategy", 5)

            project_root = Path(__file__).parent.parent.parent.parent
            worker_script = project_root / "apps" / "cli" / "mesh_worker_subprocess.py"
            
            if not worker_script.exists():
                worker_script = project_root / "apps" / "cli" / "mesh_worker.py"
            
            # Checks for Modal Execution Logic
            use_modal = quality_params.get("use_modal", False) if quality_params else False
            
            cmd = []
            
            if use_modal:
                self.signals.log.emit("=" * 70)
                self.signals.log.emit("MODAL CLOUD EXECUTION ENABLED")
                self.signals.log.emit("Offloading job to Modal.com...")
                self.signals.log.emit("=" * 70)
                
                script_path = project_root / "scripts" / "run_local_modal.py"
                
                # Check if modal CLI is available (fallback to system path if python module fails)
                try:
                    # We use python -m modal to ensure we use the installed package
                    subprocess.run([sys.executable, "-m", "modal", "--version"], capture_output=True, check=True)
                except subprocess.CalledProcessError:
                     self.signals.log.emit("[ERROR] 'modal' module not found or failed. Please install modal.")
                     self.signals.finished.emit(False, {"error": "Modal module error"})
                     return

                # Construct modal run command
                cmd = [sys.executable, "-u", "-m", "modal", "run", str(script_path)]
                cmd.extend(["--input-file", cad_file])
                
                # Pass config file
                if quality_params:
                    try:
                        fd, config_path = tempfile.mkstemp(suffix='.json', prefix='modal_config_')
                        os.close(fd)
                        with open(config_path, 'w') as f:
                            json.dump(quality_params, f)
                        
                        cmd.extend(["--config-file", config_path])
                        self.temp_config_file = config_path
                        self.signals.log.emit(f"[DEBUG] Using config: {config_path}")
                    except Exception as e:
                        self.signals.log.emit(f"[ERROR] Failed to write config for Modal: {e}")
                        self.signals.finished.emit(False, {"error": f"Failed to write config: {e}"})
                        return

            else:
                # Parallel execution info
                cores = cpu_count()
                workers = max(1, cores - 2)
                self.signals.log.emit("=" * 70)
                self.signals.log.emit("PARALLEL EXECUTION MODE ENABLED")
                self.signals.log.emit(f"System: {cores} CPU cores detected")
                self.signals.log.emit(f"Using: {workers} parallel workers (strategies run concurrently)")
                self.signals.log.emit(f"Expected speedup: 3-5x faster than sequential mode")
                self.signals.log.emit("=" * 70)
                self.signals.log.emit("Starting parallel mesh generation...")
    
                cmd = [sys.executable, "-u", str(worker_script), cad_file]
                
                if quality_params:
                    try:
                        fd, config_path = tempfile.mkstemp(suffix='.json', prefix='mesh_config_')
                        os.close(fd)
                        with open(config_path, 'w') as f:
                            json.dump(quality_params, f)
                        cmd.extend(["--config-file", config_path])
                        self.temp_config_file = config_path
                    except Exception as e:
                        self.signals.log.emit(f"[ERROR] Failed to create config file: {e}")
                        cmd.extend(["--quality-params", json.dumps(quality_params)])

            self.signals.log.emit(f"[DEBUG] Executing command: {' '.join(cmd)}")
            print(f"[DEBUG] Executing command: {' '.join(cmd)}")

            # Pass PYTHONPATH
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH", "")
            if python_path:
                env["PYTHONPATH"] = f"{project_root}{os.pathsep}{python_path}"
            else:
                env["PYTHONPATH"] = str(project_root)
            
            # Ensure unbuffered output for Modal to stream logs
            env["PYTHONUNBUFFERED"] = "1"

            # Start subprocess
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Read output
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                self._parse_and_emit_line(line.strip())
            
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if return_code != 0:
                self.signals.log.emit(f"Process exited with code {return_code}")

        except Exception as e:
            self.signals.log.emit(f"Error: {str(e)}")
            self.signals.finished.emit(False, {"error": str(e)})
        finally:
             if self.temp_config_file and os.path.exists(self.temp_config_file):
                try:
                    os.unlink(self.temp_config_file)
                except:
                    pass
             self.is_running = False


    def stop(self, force=False):
        """Stop the mesh generation process and all child processes"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.process:
            try:
                # Use psutil to kill entire process tree
                parent = psutil.Process(self.process.pid)
                children = parent.children(recursive=True)
                
                # Kill children first (assembly workers, gmsh, etc.)
                for child in children:
                    try:
                        child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Kill parent (mesh_worker_subprocess.py)
                try:
                    parent.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # Wait briefly for cleanup
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                # Fallback to standard termination
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1)
                except:
                    pass
        
        # Clean up temp config file
        if self.temp_config_file and os.path.exists(self.temp_config_file):
            try:
                os.remove(self.temp_config_file)
            except:
                pass
    
    @staticmethod
    def cleanup_orphans():
        """Kill orphaned mesh worker processes from previous sessions"""
        killed_count = 0
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'mesh_worker_subprocess.py' in ' '.join(cmdline):
                        # This is an orphaned worker - kill it and all children
                        parent = psutil.Process(proc.info['pid'])
                        children = parent.children(recursive=True)
                        
                        for child in children:
                            try:
                                child.kill()
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        try:
                            parent.kill()
                            killed_count += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"[CLEANUP] Error during orphan cleanup: {e}")
        
        if killed_count > 0:
            print(f"[CLEANUP] Killed {killed_count} orphaned worker process(es)")
        return killed_count


class QualityAnalysisWorker(QThread):
    """Background worker for deferred quality analysis"""
    finished = pyqtSignal(bool, dict)
    log = pyqtSignal(str)

    def __init__(self, mesh_file: str):
        super().__init__()
        self.mesh_file = mesh_file
        self.is_running = True

    def run(self):
        try:
            self.log.emit(f"Starting background quality analysis for: {Path(self.mesh_file).name}")
            
            # Helper script to compute quality using Gmsh
            script = f"""
import gmsh
import json
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.merge(r"{self.mesh_file}")
    
    per_element_quality = {{}}
    per_element_gamma = {{}}
    per_element_skewness = {{}}
    per_element_aspect_ratio = {{}}
    
    # Get 3D elements (tetrahedra) - this is a volume mesh
    tet_types, tet_tags, tet_nodes = gmsh.model.mesh.getElements(3)
    all_qualities = []
    
    for elem_type, tags in zip(tet_types, tet_tags):
        if elem_type in [4, 11]:  # Linear & Quadratic Tets
            # Extract SICN quality (Inverse condition number)
            sicn_vals = gmsh.model.mesh.getElementQualities(tags.tolist(), "minSICN")
            gamma_vals = gmsh.model.mesh.getElementQualities(tags.tolist(), "gamma")
            
            for tag, sicn, gamma in zip(tags, sicn_vals, gamma_vals):
                tag_int = int(tag)
                per_element_quality[tag_int] = float(sicn)
                per_element_gamma[tag_int] = float(gamma)
                per_element_skewness[tag_int] = 1.0 - float(sicn)
                per_element_aspect_ratio[tag_int] = 1.0 / float(sicn) if sicn > 0 else 100.0
                all_qualities.append(sicn)
                
    gmsh.finalize()
    
    if not per_element_quality:
        print(json.dumps({"error": "No tetrahedral elements found"}), flush=True)
        sys.exit(0)
        
    # Calculate statistics
    sorted_q = sorted(all_qualities)
    idx_10 = max(0, int(len(sorted_q) * 0.10))
    
    avg_gamma = sum(per_element_gamma.values()) / len(per_element_gamma)
    avg_skewness = sum(per_element_skewness.values()) / len(per_element_skewness)
    avg_aspect_ratio = sum(per_element_aspect_ratio.values()) / len(per_element_aspect_ratio)

    result = {{
        "success": True,
        "per_element_quality": per_element_quality,
        "per_element_gamma": per_element_gamma,
        "per_element_skewness": per_element_skewness,
        "per_element_aspect_ratio": per_element_aspect_ratio,
        "quality_metrics": {{
            "sicn_min": min(all_qualities),
            "sicn_avg": sum(all_qualities) / len(all_qualities),
            "sicn_max": max(all_qualities),
            "sicn_10_percentile": sorted_q[idx_10],
            "gamma_avg": avg_gamma,
            "skewness_avg": avg_skewness,
            "aspect_ratio_avg": avg_aspect_ratio,
        }}
    }}
    
    print("JSON_RESULT:" + json.dumps(result), flush=True)

except Exception as e:
    print(json.dumps({"error": str(e)}), flush=True)
    sys.exit(1)
"""
            # Set PYTHONPATH to ensure imports work if run from source
            # Go up 4 levels from gui_app/workers.py to project root
            project_root = Path(__file__).parent.parent.parent.parent
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH", "")
            if python_path:
                env["PYTHONPATH"] = f"{project_root}{os.pathsep}{python_path}"
            else:
                env["PYTHONPATH"] = str(project_root)

            # Run using same python environment
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.log.emit(f"Quality analysis failed: {stderr}")
                self.finished.emit(False, {"error": stderr})
                return

            # Parse output
            result = None
            for line in stdout.splitlines():
                if line.startswith("JSON_RESULT:"):
                    result = json.loads(line[12:])
                    break
            
            if result and result.get("success"):
                self.log.emit("Quality analysis completed successfully")
                self.finished.emit(True, result)
            else:
                error = result.get("error") if result else "Unknown error"
                self.log.emit(f"Quality analysis failed: {error}")
                self.finished.emit(False, {"error": error})

        except Exception as e:
            self.log.emit(f"Worker exception: {e}")
            self.finished.emit(False, {"error": str(e)})

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()
