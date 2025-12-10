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
from pathlib import Path
from multiprocessing import cpu_count
from PyQt5.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    """Signals for mesh worker thread"""
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int)  # (phase, percentage)
    phase_complete = pyqtSignal(str)  # Signal when phase is 100% done
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

    def start(self, cad_file: str, quality_params: dict = None):
        if self.is_running:
            return

        self.is_running = True
        self.phase_max = {}
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
            # 1D Meshing
            if "Meshing 1D" in line:
                self._emit_progress("1d", 10)
            elif "Meshing curve" in line and "order 2" not in line:
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("1d", 10 + int(pct * 0.9))
            elif "Done meshing 1D" in line:
                self._complete_phase("1d")

            # 2D Meshing
            elif "Meshing 2D" in line:
                self._emit_progress("2d", 10)
            elif "Meshing surface" in line and "order 2" not in line:
                match = re.search(r'\[\s*(\d+)%\]', line)
                if match:
                    pct = int(match.group(1))
                    self._emit_progress("2d", 10 + int(pct * 0.9))
            elif "Done meshing 2D" in line:
                self._complete_phase("2d")

            # 3D Meshing
            elif "Meshing 3D" in line:
                self._emit_progress("3d", 10)
            elif "Tetrahedrizing" in line:
                self._emit_progress("3d", 30)
            elif "Reconstructing mesh" in line:
                self._emit_progress("3d", 50)
            elif "3D refinement" in line:
                self._emit_progress("3d", 70)
            elif "Done meshing 3D" in line:
                self._complete_phase("3d")

            # Optimization (Gmsh)
            elif "Optimizing mesh..." in line and "Netgen" not in line:
                self._emit_progress("opt", 10)
            elif "edge swaps" in line:
                self._emit_progress("opt", 60)
            elif "No ill-shaped tets" in line:
                self._emit_progress("opt", 90)
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
                    # Mark all active phases as complete
                    for phase in ["strategy", "1d", "2d", "3d", "opt", "netgen", "order2", "quality"]:
                        if self.phase_max.get(phase, 0) > 0:
                            self._complete_phase(phase)

                    self.signals.log.emit("Mesh generation completed!")
                    self.signals.log.emit(f"[DEBUG] About to emit finished signal with {len(result)} result keys")
                    self.signals.log.emit(f"[DEBUG] Result keys: {list(result.keys())}")
                    self.signals.log.emit(f"[DEBUG] per_element_quality present: {'per_element_quality' in result}")
                    print(f"[GUI-WORKER] Emitting finished signal with result keys: {list(result.keys())}")
                    self.signals.finished.emit(True, result)
                    self.signals.log.emit("[DEBUG] Finished signal emitted!")
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
            self.signals.log.emit(line)

    def _run(self, cad_file: str, quality_params: dict = None):
        try:
            self.signals.log.emit(f"Loading: {Path(cad_file).name}")
            self._emit_progress("strategy", 5)

            # Show parallel execution info
            cores = cpu_count()
            workers = max(1, cores - 2)
            self.signals.log.emit("=" * 70)
            self.signals.log.emit("PARALLEL EXECUTION MODE ENABLED")
            self.signals.log.emit(f"System: {cores} CPU cores detected")
            self.signals.log.emit(f"Using: {workers} parallel workers (strategies run concurrently)")
            self.signals.log.emit(f"Expected speedup: 3-5x faster than sequential mode")
            self.signals.log.emit("=" * 70)

            # CRITICAL: Path fix for gui_app package
            # We are now in gui_app/workers.py, need to go up 3 levels to project root
            project_root = Path(__file__).parent.parent.parent
            worker_script = project_root / "cli" / "mesh_worker_subprocess.py"
            
            if not worker_script.exists():
                # Fallback: check if mesh_worker.py exists instead
                worker_script = project_root / "cli" / "mesh_worker.py"
                
            self.signals.log.emit("Starting parallel mesh generation...")

            # Prepare command with quality parameters
            cmd = [sys.executable, str(worker_script), cad_file]
            
            # Use temporary file for configuration to avoid command line limits
            if quality_params:
                try:
                    # Create a named temp file that persists after close
                    fd, config_path = tempfile.mkstemp(suffix='.json', prefix='mesh_config_')
                    os.close(fd)
                    
                    with open(config_path, 'w') as f:
                        json.dump(quality_params, f)
                    
                    cmd.extend(["--config-file", config_path])
                    self.temp_config_file = config_path
                    self.signals.log.emit(f"[DEBUG] Wrote config to: {config_path}")
                except Exception as e:
                    self.signals.log.emit(f"[ERROR] Failed to create config file: {e}")
                    cmd.extend(["--quality-params", json.dumps(quality_params)])

            self.signals.log.emit(f"[DEBUG] Executing command: {' '.join(cmd)}")
            print(f"[DEBUG] Executing command: {' '.join(cmd)}")

            # Start subprocess with PIPE
            self.signals.log.emit("[DEBUG] Starting subprocess with PIPE...")
            
            # CRITICAL FIX: Pass PYTHONPATH to subprocess
            # macOS/Linux spawn method needs explicit path to find modules
            env = os.environ.copy()
            python_path = env.get("PYTHONPATH", "")
            if python_path:
                env["PYTHONPATH"] = f"{project_root}{os.pathsep}{python_path}"
            else:
                env["PYTHONPATH"] = str(project_root)
                
            # Set worker count from GUI slider
            if quality_params and "worker_count" in quality_params:
                env["MESH_MAX_WORKERS"] = str(quality_params["worker_count"])
                self.signals.log.emit(f"[DEBUG] Set MESH_MAX_WORKERS={quality_params['worker_count']}")
                
            self.signals.log.emit(f"[DEBUG] PYTHONPATH set to: {env['PYTHONPATH']}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env=env     # Pass environment with PYTHONPATH
            )
            self.signals.log.emit(f"[DEBUG] Subprocess started with PID: {self.process.pid}")

            # Read line by line from stdout
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                self._parse_and_emit_line(line)

            self.process.wait()
            self.signals.log.emit(f"[DEBUG] Subprocess finished with return code: {self.process.returncode}")

        except Exception as e:
            self.signals.log.emit(f"Exception: {str(e)}")
            self.signals.finished.emit(False, {"error": str(e)})
        finally:
            self.is_running = False
            # Clean up temp config file
            if self.temp_config_file and os.path.exists(self.temp_config_file):
                try:
                    os.remove(self.temp_config_file)
                except:
                    pass

    def stop(self):
        """Stop the running mesh generation subprocess"""
        if self.process and self.process.poll() is None:
            # Process is still running
            try:
                # Try graceful termination first
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Force kill if still running after 3 seconds
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                pass  # Best effort
        self.is_running = False
