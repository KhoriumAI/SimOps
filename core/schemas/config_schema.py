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
    inlet_velocity: Union[float, List[float]] = Field(default=5.0, description="Inlet flow velocity (m/s). Scalar or Vector [vx, vy, vz]")
    ambient_temp_c: float = Field(default=25.0, description="Ambient air temp (Celsius)")
    
    # Temperature Boundary Conditions (LEGACY - prefer _c versions above)
    # These default to None; the Celsius fields above are primary
    heat_source_temperature: Optional[float] = Field(default=None, description="Hot boundary temperature (K)")
    ambient_temperature: Optional[float] = Field(default=None, description="Ambient/convection temperature (K)")
    
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

    # Flux Boundary Conditions
    surface_flux_wm2: Optional[float] = Field(default=None, description="Surface Heat Flux in W/m^2 (applied to Heat Source boundary).")
    volumetric_heat_wm3: Optional[float] = Field(default=None, description="Volumetric Heat Generation in W/m^3.")

    # Simulation Selector
    simulation_type: Literal["thermal", "structural", "cfd"] = Field(default="thermal", description="Type of simulation")
    
    # CFD Specifics
    kinematic_viscosity: float = Field(default=1e-5, description="Fluid kinematic viscosity (m2/s)")
    virtual_wind_tunnel: Optional[bool] = Field(default=None, description="Enable Virtual Wind Tunnel (External Flow). If None, auto-detects based on velocity.")
    mesh_scale_factor: Optional[float] = Field(default=1.0, description="Scale geometry (e.g. 0.001 for mm->m)")
    
    # Structural - Gravity
    gravity_load_g: float = Field(default=0.0, description="Gravity load in Gs")
    
    # Structural - Tip Load (Vector X, Y, Z)
    # Only applies to nodes in 'tip_load_selection'? Or auto-detect?
    # For MVP, auto-detect Z-max or similar?
    # Let's simple: tip_load: Optional[List[float]] = None
    tip_load: Optional[List[float]] = Field(default=None, description="Tip Load Vector [Fx, Fy, Fz] in Newtons")
    
    # Material
    youngs_modulus: Optional[float] = Field(default=None, description="Young's Modulus in MPa")
    poissons_ratio: Optional[float] = Field(default=None, description="Poisson's Ratio. Overrides material default.")
    
    # Solver Configuration
    ccx_path: Optional[str] = Field(default=None, description="Path to CalculiX binary (ccx.exe or wrapper script)")


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
