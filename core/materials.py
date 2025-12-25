from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class MaterialProperties:
    name: str
    density: float # kg/m^3
    conductivity: float # W/mK
    specific_heat: float # J/kgK
    elastic_modulus: Optional[float] = None # Pa
    poisson_ratio: Optional[float] = None

# NIST / MPDB Standard Values
MATERIAL_DB: Dict[str, MaterialProperties] = {
    "Aluminum_6061_T6": MaterialProperties(
        name="Aluminum 6061-T6",
        density=2700.0,
        conductivity=167.0,
        specific_heat=896.0,
        elastic_modulus=68.9e9,
        poisson_ratio=0.33
    ),
    "Copper_Pure": MaterialProperties(
        name="Copper (Pure)",
        density=8960.0,
        conductivity=401.0,
        specific_heat=385.0,
        elastic_modulus=110e9,
        poisson_ratio=0.34
    ),
    "ABS_Plastic": MaterialProperties(
        name="ABS Generic",
        density=1040.0,
        conductivity=0.25,
        specific_heat=1200.0,
        elastic_modulus=2.2e9,
        poisson_ratio=0.35
    ),
    "Steel_304": MaterialProperties(
        name="Stainless Steel 304",
        density=7900.0,
        conductivity=16.2, # Average @ room temp
        specific_heat=500.0,
        elastic_modulus=193e9,
        poisson_ratio=0.29
    ),
    "Steel_Generic": MaterialProperties(
        name="Steel (Generic)",
        density=7850.0,
        conductivity=50.0,
        specific_heat=450.0,
        elastic_modulus=200e9,
        poisson_ratio=0.30
    )
}

def get_material(key: str) -> MaterialProperties:
    """
    Retrieve material properties from the database by key.
    Raises KeyError if not found.
    """
    if key not in MATERIAL_DB:
        raise KeyError(f"Material '{key}' not found in database. Available: {list(MATERIAL_DB.keys())}")
    return MATERIAL_DB[key]
