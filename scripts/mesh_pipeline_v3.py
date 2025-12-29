import gmsh
import glob
import os
import math
import time
import multiprocessing
import struct
import sys

# =================CONFIGURATION=================
INPUT_DIR = "temp_stls"
OUTPUT_DIR = "temp_stls/volume_meshes"
LOG_FILE = "mesh_pipeline_log.txt"
TIMEOUT_SECONDS = 30
FORCE_REPROCESS = True  # Set to True to ignore existing meshes and reprocess everything
# ===============================================

def get_stl_bounds(stl_path):
    """
    Manually scans binary STL to find bounding box without using Gmsh
    (in case Gmsh crashes reading the file).
    """
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    try:
        with open(stl_path, 'rb') as f:
            header = f.read(80)
            count = struct.unpack('<I', f.read(4))[0]
            for _ in range(count):
                # Read Normal (3 floats) + 3 Vertices (9 floats) + Attribute (2 bytes)
                data = f.read(50) 
                if len(data) < 50: break
                # We only care about the vertices (offsets 12-48)
                floats = struct.unpack('<12fH', data)
                # Vertices are at indices 3-5, 6-8, 9-11
                for i in range(3, 12, 3):
                    x, y, z = floats[i], floats[i+1], floats[i+2]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    min_z, max_z = min(min_z, z), max(max_z, z)
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    except Exception:
        return None

def generate_bounding_box_mesh(stl_path, output_path):
    """
    The 'Nuclear Option': Replaces geometry with a simple block.
    """
    bounds = get_stl_bounds(stl_path)
    if not bounds:
        return False, "Could not parse STL for bounds"
        
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z
    
    # Initialize fresh environment
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("BoundingBox")
        # Create Box using OpenCascade kernel
        gmsh.model.occ.addBox(min_x, min_y, min_z, dx, dy, dz)
        gmsh.model.occ.synchronize()
        
        # Coarse mesh is fine for a box
        gmsh.model.mesh.generate(3)
        gmsh.write(output_path)
        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        gmsh.finalize()

def repair_worker(stl_path, output_path):
    """
    The 'Hospital' logic running in a separate process.
    """
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT
        gmsh.option.setNumber("Geometry.Tolerance", 1e-4) # Aggressive snap
        
        gmsh.merge(stl_path)
        # Force topology re-classification (The slow part)
        gmsh.model.mesh.classifySurfaces(40 * math.pi / 180, True, True)
        gmsh.model.mesh.createGeometry()
        gmsh.model.geo.synchronize()
        
        gmsh.model.mesh.generate(3)
        gmsh.write(output_path)
    except Exception:
        sys.exit(1) # Signal failure
    finally:
        gmsh.finalize()

class MeshPipeline:
    def __init__(self):
        self.failed_files = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._clear_log()

    def _clear_log(self):
        with open(LOG_FILE, "w") as f:
            f.write("=== Mesh Pipeline Log ===\n")

    def log(self, message):
        print(message)
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")

    def run_fast_pass(self):
        """Stage 1: Bulk Linear Processing"""
        self.log(f"\n[STAGE 1] Starting Fast Bulk Mesh...")
        stl_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.stl"), recursive=True)
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)

        success_count = 0
        
        for stl_path in stl_files:
            part_name = os.path.basename(stl_path).replace(".stl", ".msh")
            output_path = os.path.join(OUTPUT_DIR, part_name)
            
            if not FORCE_REPROCESS and os.path.exists(output_path):
                success_count += 1
                continue

            try:
                gmsh.clear()
                gmsh.merge(stl_path)
                
                # CRITICAL: Classify surfaces and create geometry (this makes it a volume)
                try:
                    gmsh.model.mesh.classifySurfaces(40 * math.pi / 180, boundary=True)
                    gmsh.model.mesh.createGeometry()
                    
                    # Get surfaces and create volume
                    surfaces = gmsh.model.getEntities(2)
                    if surfaces:
                        surface_tags = [s[1] for s in surfaces]
                        sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
                        gmsh.model.geo.addVolume([sl])
                        gmsh.model.geo.synchronize()
                    else:
                        # No surfaces means the STL was bad
                        self.log(f"  [FAIL] {os.path.basename(stl_path)} - No surfaces after classification")
                        self.failed_files.append(stl_path)
                        continue
                        
                except Exception as classify_error:
                    # Surface classification failed -> send to hospital
                    self.log(f"  [FAIL] {os.path.basename(stl_path)} - Classification failed, sending to hospital")
                    self.failed_files.append(stl_path)
                    continue
                
                # Generate 3D volume mesh
                gmsh.model.mesh.generate(3)
                gmsh.write(output_path)
                self.log(f"  [OK] {os.path.basename(stl_path)}")
                success_count += 1
            except Exception as e:
                self.log(f"  [FAIL] {os.path.basename(stl_path)} - Sent to Hospital")
                self.failed_files.append(stl_path)

        gmsh.finalize()
        self.log(f"[STAGE 1 COMPLETE] Success: {success_count} | To Repair: {len(self.failed_files)}")

    def run_hospital_pass(self):
        """Stage 2: Repair with Timeout -> Fallback to Bounding Box"""
        if not self.failed_files:
            return

        self.log(f"\n[STAGE 2] Hospital Admission for {len(self.failed_files)} parts...")
        
        for stl_path in self.failed_files:
            part_name = os.path.basename(stl_path).replace(".stl", ".msh")
            output_path = os.path.join(OUTPUT_DIR, part_name)
            
            self.log(f"  Treating: {os.path.basename(stl_path)}...")

            # 1. Attempt Repair in Subprocess (to allow hard kill)
            p = multiprocessing.Process(target=repair_worker, args=(stl_path, output_path))
            p.start()
            p.join(TIMEOUT_SECONDS)

            if p.is_alive():
                self.log(f"    [TIMEOUT] Repair exceeded {TIMEOUT_SECONDS}s. Terminating.")
                p.terminate()
                p.join()
                repair_success = False
            else:
                # Check exit code (0 = success)
                repair_success = (p.exitcode == 0)

            # 2. Verify Output Exists
            if repair_success and os.path.exists(output_path):
                self.log(f"    [FIXED] Repair successful.")
            else:
                # 3. FALLBACK: Bounding Box
                self.log(f"    [FALLBACK] Replacing with Bounding Box...")
                bb_success, msg = generate_bounding_box_mesh(stl_path, output_path)
                if bb_success:
                    self.log(f"    [BOXED] Geometry replaced with bounds.")
                else:
                    self.log(f"    [TERMINAL] Even Bounding Box failed: {msg}")

# =================EXECUTION=================
if __name__ == "__main__":
    multiprocessing.freeze_support() # Required for Windows
    start_time = time.time()
    
    pipeline = MeshPipeline()
    pipeline.run_fast_pass()
    pipeline.run_hospital_pass()
    
    elapsed = time.time() - start_time
    print(f"\nTotal Pipeline Time: {elapsed:.2f}s")
