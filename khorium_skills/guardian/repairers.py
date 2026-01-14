import os
import logging
import shutil
import tempfile

# Lazy imports to prevent worker startup lag
try:
    import gmsh
except ImportError:
    pass

try:
    import trimesh
    import numpy as np
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

class ManifoldHealer:
    """
    The Surgeon.
    Implements the exact progressive repair strategy from the original script.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def heal(self, input_path: str, output_path: str, strategy: dict) -> str | None:
        """
        Attempts to heal the geometry. 
        Returns the path to the valid file (extension may change), or None.
        """
        method = strategy.get('method', 'occ')
        
        try:
            # DISPATCHER
            if method == 'occ':
                # OCC keeps the same extension (.step)
                if self._heal_occ(input_path, output_path, strategy):
                    return output_path
            
            elif method == 'trimesh':
                # Trimesh outputs STL
                final_path = self._ensure_extension(output_path, ".stl")
                if self._heal_trimesh(input_path, final_path, strategy):
                    return final_path
                    
            elif method == 'convex_hull':
                # Hull outputs STL
                final_path = self._ensure_extension(output_path, ".stl")
                if self._heal_convex_hull(input_path, final_path):
                    return final_path
                    
        except Exception as e:
            self.logger.error(f"Repair strategy '{method}' crashed: {e}")
        
        return None

    def _heal_occ(self, input_path: str, output_path: str, strategy: dict) -> bool:
        """
        Level 1: CAD-Native Healing
        Logic derived from 'attempt_repair' -> Level 1 in repair.py
        """
        try:
            if not gmsh.isInitialized():
                gmsh.initialize()
            
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("Healer_OCC")
            
            # Load
            gmsh.model.occ.importShapes(input_path)
            gmsh.model.occ.synchronize()
            
            # Action 1: Standard Healing
            # "healShapes attempts to fix small gaps, overlaps, etc."
            self.logger.info("Running OCC healShapes...")
            gmsh.model.occ.healShapes()
            
            # Action 2: Fragmentation (Boolean Union for Assemblies)
            # New robust addition for assemblies (optional via strategy)
            if strategy.get('fragment', False):
                vols = gmsh.model.getEntities(3)
                if len(vols) > 1:
                    self.logger.info(f"Fragmenting {len(vols)} volumes...")
                    gmsh.model.occ.fragment(vols, vols)
            
            gmsh.model.occ.synchronize()
            
            # Verification: Check Mass > 0
            # "if new_volume > 1e-10"
            total_volume = 0.0
            for dim, tag in gmsh.model.getEntities(3):
                try:
                    total_volume += gmsh.model.occ.getMass(dim, tag)
                except:
                    pass

            if total_volume <= 1e-10:
                self.logger.warning(f"OCC Repair failed: Result has zero volume ({total_volume})")
                return False

            # Export
            gmsh.write(output_path)
            gmsh.model.remove()
            
            return os.path.exists(output_path)

        except Exception as e:
            self.logger.warning(f"OCC Repair failed: {e}")
            return False

    def _heal_trimesh(self, input_path: str, output_path: str, strategy: dict) -> bool:
        """
        Level 2: Discrete Mesh Repair
        Logic derived from 'attempt_repair' -> Level 2 in repair.py
        """
        if not TRIMESH_AVAILABLE:
            self.logger.warning("Trimesh not available. Skipping Level 2 repair.")
            return False

        temp_stl = None
        try:
            # Step A: Convert CAD to intermediate STL
            # Replicates: gmsh.model.mesh.generate(2) -> gmsh.write(temp_stl)
            temp_stl = self._convert_step_to_temp_stl(input_path)
            if not temp_stl:
                return False

            # Step B: Load into Trimesh
            # Replicates: trimesh.load(..., process=False)
            mesh = trimesh.load(temp_stl, force='mesh', process=False)
            
            # Action: Merge Vertices
            # Replicates: mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            
            # Action: Fix Normals & Winding
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_inversion(mesh)
            
            # Verification: Is it watertight?
            # "if mesh.is_watertight:"
            if mesh.is_watertight:
                mesh.export(output_path)
                self.logger.info(f"Trimesh repair successful. Output: {output_path}")
                return True
            else:
                self.logger.warning("Trimesh repair finished but mesh is still not watertight.")
                return False

        except Exception as e:
            self.logger.warning(f"Trimesh repair failed: {e}")
            return False
        finally:
            if temp_stl and os.path.exists(temp_stl):
                try: os.remove(temp_stl)
                except: pass

    def _heal_convex_hull(self, input_path: str, output_path: str) -> bool:
        """
        Level 3: The Nuclear Option
        Logic derived from 'attempt_repair' -> Level 3 in repair.py
        """
        if not TRIMESH_AVAILABLE:
            return False

        temp_stl = None
        try:
            self.logger.info("Attempting Convex Hull fallback...")
            
            # Step A: Convert
            temp_stl = self._convert_step_to_temp_stl(input_path)
            if not temp_stl: return False

            # Step B: Compute Hull
            mesh = trimesh.load(temp_stl, force='mesh', process=False)
            hull = mesh.convex_hull

            # Verification: Watertight + Volume Check
            # "if hull.is_watertight and hull.volume > 1e-10:"
            if hull.is_watertight and hull.volume > 1e-10:
                hull.export(output_path)
                self.logger.info("Convex Hull generated.")
                return True
            
            return False

        except Exception as e:
            self.logger.warning(f"Convex Hull failed: {e}")
            return False
        finally:
            if temp_stl and os.path.exists(temp_stl):
                try: os.remove(temp_stl)
                except: pass

    def _convert_step_to_temp_stl(self, input_path: str) -> str | None:
        """
        Helper: Uses Gmsh to convert STEP -> STL for Trimesh processing.
        Replicates the setup/teardown in Level 2/3 of repair.py.
        """
        try:
            if not gmsh.isInitialized(): gmsh.initialize()
            gmsh.model.add("Converter")
            gmsh.model.occ.importShapes(input_path)
            gmsh.model.occ.synchronize()
            
            # Generate coarse surface mesh for repair
            # Replicates: gmsh.model.mesh.generate(2)
            # SPEED OPTIMIZATION: Use smart sizing to prevent over-meshing
            bbox = gmsh.model.getBoundingBox(-1, -1)
            dims = [bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]]
            diag = (dims[0]**2 + dims[1]**2 + dims[2]**2)**0.5 or 1.0
            
            gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 50.0) # Faster than 100.0
            gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 10.0)
            gmsh.option.setNumber("Mesh.Algorithm", 1) 
            gmsh.model.mesh.generate(2)
            
            fd, temp_path = tempfile.mkstemp(suffix=".stl")
            os.close(fd)
            
            gmsh.write(temp_path)
            gmsh.model.remove()
            return temp_path
        except Exception:
            return None

    def _ensure_extension(self, path: str, new_ext: str) -> str:
        """Helper: Swaps file extension (e.g., .step -> .stl)"""
        base, _ = os.path.splitext(path)
        return f"{base}{new_ext}"