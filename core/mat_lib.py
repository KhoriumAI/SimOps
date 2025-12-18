
import json
import os
from pathlib import Path
from typing import Dict, Optional, Union, List

# Helper to locate the materials.json relative to this file
MATERIALS_FILE = Path(__file__).parent / "materials.json"

class MaterialLibrary:
    _instance = None
    
    def __init__(self):
        self.materials = {}
        self._load_materials()
        
    def _load_materials(self):
        if not MATERIALS_FILE.exists():
            print(f"[WARN] Material library not found at {MATERIALS_FILE}")
            return
            
        try:
            with open(MATERIALS_FILE, 'r') as f:
                self.materials = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load material library: {e}")
            
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def get_material(self, name: str) -> Optional[Dict]:
        return self.materials.get(name)
    
    def get_conductivity(self, name: str) -> Union[float, List[List[float]], None]:
        mat = self.get_material(name)
        if mat:
            return mat.get("conductivity")
        return None

def get_material_conductivity(name: str) -> Union[float, List[List[float]], None]:
    """Convenience function to get conductivity by name"""
    return MaterialLibrary.get_instance().get_conductivity(name)
