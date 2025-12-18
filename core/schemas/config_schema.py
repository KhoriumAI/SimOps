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
    ambient_temp_c: float = Field(default=25.0, description="Ambient air temp")

class MeshingConfig(BaseModel):
    """Advanced meshing controls"""
    second_order: bool = Field(default=False, description="Use quadratic elements (Tet10)")
    mesh_size_multiplier: float = Field(default=1.0, description="Global scaling factor")

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
    
    # Advanced: Override standard tagging logic
    # If empty, uses "Golden Template" defaults
    tagging_rules: List[TaggingRule] = []

    # Validation
    validate_mesh: bool = Field(default=False, description="Run GCI study (3x runtime)")
