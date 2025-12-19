"""
SimOps Configuration Schema
===========================

Defines the Pydantic models for the Sidecar Protocol.
This ensures valid "Day 5" configuration inputs.
"""

from typing import List, Optional, Dict, Literal, Union
from pydantic import BaseModel, Field

class GeometrySelector(BaseModel):
    """How to find a face/volume"""
    type: Literal["z_min", "z_max", "box", "all_remaining"]
    tolerance: float = 0.001
    bounds: Optional[List[float]] = None  # [xmin, ymin, zmin, xmax, ymax, zmax]

class TaggingRule(BaseModel):
    """
    Rule to apply a semantic tag to a geometric entity.
    """
    tag_name: str  # e.g. "BC_HeatSource"
    entity_type: Literal["surface", "volume"]
    selector: GeometrySelector

class PhysicsConfig(BaseModel):
    """Physics parameters for the solver"""
    material: str = Field(default="Aluminum", description="Solid material name")
    heat_load_watts: float = Field(default=50.0, description="Heat source power")
    inlet_velocity: float = Field(default=5.0, description="Inlet flow velocity")
    ambient_temp_c: float = Field(default=25.0, description="Ambient air temp (Celsius)")
    
    # Temperature Boundary Conditions
    heat_source_temperature: float = Field(default=373.0, description="Hot boundary temperature (K)")
    ambient_temperature: float = Field(default=293.0, description="Ambient/convection temperature (K)")
    
    # Material Properties (explicit overrides)
    # If None, defaults are loaded from 'material' name via library
    thermal_conductivity: Optional[Union[float, List[List[float]]]] = Field(
        default=None, 
        description="W/mK. Scalar or Table [[Temp, Value], ...]"
    )
    density: Optional[float] = Field(default=None, description="kg/m^3")
    specific_heat: Optional[Union[float, List[List[float]]]] = Field(
        default=None, 
        description="J/kgK. Scalar or Table"
    )
    
    # Advanced Physics
    convection_coeff: float = Field(default=25.0, description="Convection h (W/m2K). If >0, applies *FILM to surfaces.")
    
    # Transient Analysis
    transient: bool = Field(default=True, description="Enable time-dependent analysis")
    time_step: float = Field(default=2.0, description="Time step size (seconds)")
    duration: float = Field(default=60.0, description="Total simulation time (seconds)")
    initial_temperature: Optional[float] = Field(default=None, description="Initial condition (K)")
    
    # Boundary Controls
    fix_hot_boundary: bool = Field(default=True, description="Enforce fixed T at Source")
    fix_cold_boundary: bool = Field(default=False, description="Enforce fixed T at Sink (Disable for Convection Tip)")
    heat_source_at_z_min: bool = Field(default=True, description="Heat source at Z_min (base). Default True for typical 'base heater' scenario.")
    unit_scaling: float = Field(default=1.0, description="Scale factor for node coordinates (e.g. 0.001 for mm -> m)")


class MeshingConfig(BaseModel):
    """Advanced meshing controls"""
    second_order: bool = Field(default=False, description="Use quadratic elements (Tet10)")
    mesh_size_multiplier: float = Field(default=1.0, description="Global scaling factor")

class MaterialDefinition(BaseModel):
    """Custom material properties override"""
    k: Union[float, List[List[float]]]
    rho: float
    cp: Union[float, List[List[float]]]

class SimulationConfig(BaseModel):

    """
    Root configuration object (The Sidecar)
    """
    version: str = "1.0"
    job_name: Optional[str] = None
    
    # Physics settings
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    
    # Meshing settings
    meshing: MeshingConfig = Field(default_factory=MeshingConfig)
    
    # Advanced: Custom Material Defs (Composites, CFRP, etc)
    material_definitions: Dict[str, MaterialDefinition] = {}
    
    # Advanced: Override standard tagging logic
    # If empty, uses "Golden Template" defaults
    tagging_rules: List[TaggingRule] = []

    # Validation
    validate_mesh: bool = Field(default=False, description="Run GCI study (3x runtime)")
