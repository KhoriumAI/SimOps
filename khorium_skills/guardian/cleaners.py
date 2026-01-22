import gmsh
import math
from typing import Dict, List, Optional
import logging

class GeometryCleanupTool:
    """
    Wraps geometry defeaturing and analysis logic.
    Refactored from geometry_cleanup.py.
    """

    def __init__(self, min_feature_size: Optional[float] = None):
        self.logger = logging.getLogger(__name__)
        self.min_feature_size = min_feature_size
        self._stats = {}

    def analyze(self) -> Dict:
        """
        Analyze loaded geometry for problematic features (small curves, sharp edges).
        Assumes a GMSH model is currently open.
        """
        stats = {
            'small_curves': [],
            'sharp_edges': [],
            'thin_surfaces': [],
            'flags': []
        }

        try:
            # 1. Basic Dimensions
            # We use a try-catch block for bounding box in case model is empty
            try:
                bb = gmsh.model.getBoundingBox(-1, -1)
                diagonal = math.sqrt((bb[3]-bb[0])**2 + (bb[4]-bb[1])**2 + (bb[5]-bb[2])**2)
            except:
                diagonal = 1.0
            
            # Auto-set threshold if missing (default to 0.1% of diagonal)
            threshold = self.min_feature_size if self.min_feature_size else (diagonal / 1000.0)

            # 2. Detect Small Curves
            curves = gmsh.model.getEntities(dim=1)
            for dim, tag in curves:
                try:
                    c_bb = gmsh.model.getBoundingBox(dim, tag)
                    length = math.sqrt((c_bb[3]-c_bb[0])**2 + (c_bb[4]-c_bb[1])**2 + (c_bb[5]-c_bb[2])**2)
                    if length < threshold:
                        stats['small_curves'].append(tag)
                except:
                    pass

            # 3. Detect Sharp Edges (Circles with tiny radii)
            for dim, tag in curves:
                try:
                    if gmsh.model.getType(dim, tag) == "Circle":
                        c_bb = gmsh.model.getBoundingBox(dim, tag)
                        radius = math.sqrt((c_bb[3]-c_bb[0])**2 + (c_bb[4]-c_bb[1])**2 + (c_bb[5]-c_bb[2])**2) / 2.0
                        if radius < threshold:
                            stats['sharp_edges'].append(tag)
                except:
                    pass

            # 4. Generate Flags
            if len(stats['small_curves']) > 0:
                stats['flags'].append(f"Detected {len(stats['small_curves'])} micro-curves (<{threshold:.4f})")
            
            if len(stats['sharp_edges']) > 0:
                stats['flags'].append(f"Detected {len(stats['sharp_edges'])} sharp/tiny edges")

            self._stats = stats
            return stats

        except Exception as e:
            self.logger.warning(f"Cleanup analysis failed: {e}")
            return stats