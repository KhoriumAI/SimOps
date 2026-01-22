"""
SimOps Simulation Configuration Schemas
=======================================
Production-ready Pydantic models for simulation setup.
Merged from Forge TASK_00_SHARED.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

class BCType(str, Enum):
    """Boundary condition types supported by the simulation engine."""
    HEAT_SOURCE = "heat_source"
    CONVECTION = "convection"
    FIXED_TEMPERATURE = "fixed_temperature"

class Material(BaseModel):
    """Material properties for thermal simulation."""
    name: str = Field(..., description="Material designation (e.g., Al6061)")
    volume_name: Optional[str] = Field(None, description="Volume identifier. If None, applies globally.")
    thermal_conductivity: Optional[float] = Field(None, description="W/mK", gt=0)
    density: Optional[float] = Field(None, description="kg/m3", gt=0)
    specific_heat: Optional[float] = Field(None, description="J/kgK", gt=0)

class BoundaryCondition(BaseModel):
    """Physics-based boundary condition."""
    type: BCType = Field(..., description="Type of physics boundary")
    target: str = Field(..., description="Geometric identifier or physical group")
    value: float = Field(..., description="Magnitude of the BC")
    ambient_temperature: Optional[float] = Field(None, description="Celsius (required for convection)")

class SolverSettings(BaseModel):
    """Solver configuration for transient/steady-state analysis."""
    transient: bool = Field(True, description="True for transient, False for steady-state")
    duration: float = Field(60.0, description="Simulation time in seconds")
    time_step: float = Field(1.0, description="Time step in seconds", gt=0)
    iterations: int = Field(50, description="Max iterations", gt=0)

class MeshSettings(BaseModel):
    """Meshing parameters."""
    max_size_mm: float = Field(5.0, description="Maximum element size in mm", gt=0)
    min_size_mm: float = Field(1.0, description="Minimum element size in mm", gt=0)
    element_order: int = Field(2, ge=1, le=2, description="1=linear, 2=quadratic")

class SimConfig(BaseModel):
    """Root simulation configuration object."""
    job_name: str = Field(..., description="Unique job identifier")
    cad_file: str = Field(..., description="Path to input STEP/IGES file")
    materials: List[Material] = Field(default_factory=list)
    boundary_conditions: List[BoundaryCondition] = Field(default_factory=list)
    solver_settings: SolverSettings = Field(default_factory=SolverSettings)
    mesh_settings: MeshSettings = Field(default_factory=MeshSettings)
    ambient_temperature: float = Field(293.15, description="Ambient temperature in Kelvin", gt=0)
    heat_source_temperature: float = Field(373.15, description="Heat source temperature in Kelvin", gt=0)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "job_name": "Heatsink_Test",
                "cad_file": "model.step",
                "materials": [{"name": "Copper", "thermal_conductivity": 401.0}],
                "boundary_conditions": [
                    {"type": "heat_source", "target": "base", "value": 50.0}
                ]
            }
        }
