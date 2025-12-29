import gmsh
import glob
import os
import math
import struct
import time

# =================CONFIGURATION=================
INPUT_DIR = "temp_stls"
OUTPUT_DIR = "temp_stls/volume_meshes"
LOG_FILE = "mesh_pipeline_log.txt"
FORCE_REPROCESS = False  # Set to True to ignore existing meshes
# ===============================================

def get_stl_bounds(stl_path):
    """
    Manually scans binary STL to find bounding box (Zero-Dependency).
    """
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')
    
    try:
        with open(stl_path, 'rb') as f:
            header = f.read(80)
            count_bytes = f.read(4)
            if len(count_bytes) < 4: return None
            count = struct.unpack('<I', count_bytes)[0]
            
            # Robust loop: stop if file ends prematurely
            for _ in range(count):
                data = f.read(50)
                if len(data) < 50: break
                floats = struct.unpack('<12fH', data)
                for i in range(3, 12, 3): # Vertices only
                    x, y, z = floats[i], floats[i+1], floats[i+2]
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    min_z, max_z = min(min_z, z), max(max_z, z)
        
        if min_x == float('inf'): return None
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    except Exception:
        return None

def generate_bounding_box_mesh(stl_path, output_path):
    """
    Replaces geometry with a simple thermal block.
    """
    bounds = get_stl_bounds(stl_path)
    if not bounds:
        return False, "Could not parse STL headers"
        
    min_x, min_y, min_z, max_x, max_y, max_z = bounds
    dx, dy, dz = max_x - min_x, max_y - min_y, max_z - min_z
    
    # Initialize fresh environment
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("BoundingBox")
        gmsh.model.occ.addBox(min_x, min_y, min_z, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(output_path)
        return True, "Success"
    except Exception as e:
        return False, str(e)
    finally:
        gmsh.finalize()

class MeshPipeline:
    def __init__(self):
        self.failed_files = []
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._clear_log()

    def _clear_log(self):
        with open(LOG_FILE, "w") as f:
            f.write("=== Mesh Pipeline Log (v4 - Fail-Fast) ===\n")

    def log(self, message):
        print(message)
        with open(LOG_FILE, "a") as f:
            f.write(message + "\n")

    def run_fast_pass(self):
        """Stage 1: Bulk Linear Processing with Volume Creation"""
        self.log(f"\n[STAGE 1] Starting Fast Bulk Mesh...")
        stl_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.stl"), recursive=True)
        
        # Initialize Once for Speed
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10) # HXT

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
                
                # CRITICAL: Create volume geometry from surface mesh
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
                        # No surfaces -> box it
                        self.log(f"  [FAIL] {os.path.basename(stl_path)} - No surfaces, will box")
                        self.failed_files.append(stl_path)
                        continue
                        
                except Exception:
                    # Classification failed -> box it
                    self.log(f"  [FAIL] {os.path.basename(stl_path)} - Classification failed, will box")
                    self.failed_files.append(stl_path)
                    continue
                
                # Generate 3D volume mesh
                gmsh.model.mesh.generate(3)
                gmsh.write(output_path)
                self.log(f"  [OK] {os.path.basename(stl_path)}")
                success_count += 1
                
            except Exception as e:
                self.log(f"  [FAIL] {os.path.basename(stl_path)} - Will box")
                self.failed_files.append(stl_path)

        gmsh.finalize()
        self.log(f"[STAGE 1 COMPLETE] Success: {success_count} | To Box: {len(self.failed_files)}")

    def run_fallback_pass(self):
        """Stage 2: Immediate Bounding Box (No Repair Attempt)"""
        if not self.failed_files:
            return

        self.log(f"\n[STAGE 2] Boxing {len(self.failed_files)} parts (no repair attempt)...")
        
        boxed = 0
        terminal = 0
        
        for stl_path in self.failed_files:
            part_name = os.path.basename(stl_path).replace(".stl", ".msh")
            output_path = os.path.join(OUTPUT_DIR, part_name)
            
            success, msg = generate_bounding_box_mesh(stl_path, output_path)
            
            if success:
                self.log(f"  [BOXED] {os.path.basename(stl_path)}")
                boxed += 1
            else:
                self.log(f"  [TERMINAL] {os.path.basename(stl_path)} - {msg}")
                terminal += 1
        
        self.log(f"[STAGE 2 COMPLETE] Boxed: {boxed} | Terminal Failures: {terminal}")

# =================EXECUTION=================
if __name__ == "__main__":
    start_time = time.time()
    
    pipeline = MeshPipeline()
    pipeline.run_fast_pass()
    pipeline.run_fallback_pass()
    
    elapsed = time.time() - start_time
    print(f"\nTotal Pipeline Time: {elapsed:.2f}s")
