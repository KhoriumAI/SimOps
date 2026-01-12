"""
Simulation Template System
==========================

Provides standardized templates for thermal simulations with:
- Material library with physical properties
- Boundary condition presets
- Solver configuration templates
- Pass/fail criteria definitions

This ensures rigorous, repeatable simulation setups.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Any
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# MATERIAL LIBRARY
# =============================================================================

@dataclass
class MaterialProperties:
    """Physical properties for thermal simulation"""
    name: str
    thermal_conductivity: float  # W/m·K
    density: float  # kg/m³
    specific_heat: float  # J/kg·K
    
    # Optional properties
    melting_point_c: Optional[float] = None
    max_service_temp_c: Optional[float] = None
    thermal_expansion: Optional[float] = None  # 1/K
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'thermal_conductivity': self.thermal_conductivity,
            'density': self.density,
            'specific_heat': self.specific_heat,
            'melting_point_c': self.melting_point_c,
            'max_service_temp_c': self.max_service_temp_c,
            'thermal_expansion': self.thermal_expansion,
        }


# Comprehensive material library
MATERIAL_LIBRARY = {
    # Aluminum alloys
    "Aluminum_6061": MaterialProperties(
        name="Aluminum 6061-T6",
        thermal_conductivity=167.0,
        density=2700.0,
        specific_heat=896.0,
        melting_point_c=582.0,
        max_service_temp_c=150.0,
        thermal_expansion=23.6e-6,
    ),
    "Aluminum_7075": MaterialProperties(
        name="Aluminum 7075-T6",
        thermal_conductivity=130.0,
        density=2810.0,
        specific_heat=960.0,
        melting_point_c=477.0,
        max_service_temp_c=120.0,
        thermal_expansion=23.4e-6,
    ),
    "Aluminum_Pure": MaterialProperties(
        name="Pure Aluminum",
        thermal_conductivity=237.0,
        density=2700.0,
        specific_heat=900.0,
        melting_point_c=660.0,
        max_service_temp_c=200.0,
    ),
    
    # Steels
    "Steel_AISI_1020": MaterialProperties(
        name="AISI 1020 Carbon Steel",
        thermal_conductivity=51.9,
        density=7870.0,
        specific_heat=486.0,
        melting_point_c=1515.0,
        max_service_temp_c=400.0,
        thermal_expansion=11.7e-6,
    ),
    "Steel_304SS": MaterialProperties(
        name="304 Stainless Steel",
        thermal_conductivity=16.2,
        density=8000.0,
        specific_heat=500.0,
        melting_point_c=1400.0,
        max_service_temp_c=870.0,
        thermal_expansion=17.2e-6,
    ),
    "Steel_316SS": MaterialProperties(
        name="316 Stainless Steel",
        thermal_conductivity=16.3,
        density=8000.0,
        specific_heat=500.0,
        melting_point_c=1375.0,
        max_service_temp_c=870.0,
        thermal_expansion=15.9e-6,
    ),
    
    # Copper alloys
    "Copper_Pure": MaterialProperties(
        name="Pure Copper",
        thermal_conductivity=401.0,
        density=8960.0,
        specific_heat=385.0,
        melting_point_c=1085.0,
        max_service_temp_c=200.0,
        thermal_expansion=16.5e-6,
    ),
    "Copper_C11000": MaterialProperties(
        name="Copper C11000 (ETP)",
        thermal_conductivity=388.0,
        density=8940.0,
        specific_heat=385.0,
        melting_point_c=1083.0,
        max_service_temp_c=200.0,
    ),
    "Brass_C26000": MaterialProperties(
        name="Brass C26000",
        thermal_conductivity=120.0,
        density=8530.0,
        specific_heat=380.0,
        melting_point_c=950.0,
        max_service_temp_c=150.0,
    ),
    
    # High-temp alloys
    "Inconel_718": MaterialProperties(
        name="Inconel 718",
        thermal_conductivity=11.4,
        density=8190.0,
        specific_heat=435.0,
        melting_point_c=1336.0,
        max_service_temp_c=700.0,
        thermal_expansion=13.0e-6,
    ),
    "Inconel_625": MaterialProperties(
        name="Inconel 625",
        thermal_conductivity=9.8,
        density=8440.0,
        specific_heat=410.0,
        melting_point_c=1350.0,
        max_service_temp_c=982.0,
        thermal_expansion=12.8e-6,
    ),
    "Titanium_Ti6Al4V": MaterialProperties(
        name="Titanium Ti-6Al-4V",
        thermal_conductivity=6.7,
        density=4430.0,
        specific_heat=526.0,
        melting_point_c=1660.0,
        max_service_temp_c=315.0,
        thermal_expansion=8.6e-6,
    ),
    
    # Plastics/Polymers
    "ABS": MaterialProperties(
        name="ABS Plastic",
        thermal_conductivity=0.17,
        density=1050.0,
        specific_heat=1386.0,
        melting_point_c=105.0,
        max_service_temp_c=80.0,
    ),
    "PEEK": MaterialProperties(
        name="PEEK",
        thermal_conductivity=0.25,
        density=1320.0,
        specific_heat=320.0,
        melting_point_c=343.0,
        max_service_temp_c=250.0,
    ),
    "Nylon_66": MaterialProperties(
        name="Nylon 6/6",
        thermal_conductivity=0.25,
        density=1140.0,
        specific_heat=1670.0,
        melting_point_c=255.0,
        max_service_temp_c=80.0,
    ),
    
    # Ceramics
    "Alumina_Al2O3": MaterialProperties(
        name="Alumina (Al2O3)",
        thermal_conductivity=35.0,
        density=3950.0,
        specific_heat=880.0,
        melting_point_c=2072.0,
        max_service_temp_c=1700.0,
    ),
    
    # Other
    "Glass_Borosilicate": MaterialProperties(
        name="Borosilicate Glass",
        thermal_conductivity=1.14,
        density=2230.0,
        specific_heat=830.0,
        melting_point_c=820.0,
        max_service_temp_c=500.0,
    ),
}


def get_material(name: str) -> MaterialProperties:
    """Get material properties by name"""
    if name in MATERIAL_LIBRARY:
        return MATERIAL_LIBRARY[name]
    
    # Try case-insensitive match
    for key, mat in MATERIAL_LIBRARY.items():
        if key.lower() == name.lower() or mat.name.lower() == name.lower():
            return mat
    
    raise KeyError(f"Material '{name}' not found in library. "
                   f"Available: {list(MATERIAL_LIBRARY.keys())}")


# =============================================================================
# BOUNDARY CONDITION TYPES
# =============================================================================

class BCType(Enum):
    """Boundary condition types for thermal simulation"""
    FIXED_TEMPERATURE = "fixed_temp"
    CONVECTION = "convection"
    HEAT_FLUX = "heat_flux"
    RADIATION = "radiation"
    ADIABATIC = "adiabatic"


@dataclass
class BoundaryCondition:
    """A single boundary condition specification"""
    bc_type: BCType
    surface_tag: str  # Physical group name or tag
    
    # For fixed temperature
    temperature_c: Optional[float] = None
    
    # For convection
    ambient_temp_c: Optional[float] = None
    htc: Optional[float] = None  # Heat transfer coefficient W/m²·K
    
    # For heat flux
    flux_w_m2: Optional[float] = None
    
    # For radiation
    emissivity: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'type': self.bc_type.value,
            'surface_tag': self.surface_tag,
            'temperature_c': self.temperature_c,
            'ambient_temp_c': self.ambient_temp_c,
            'htc': self.htc,
            'flux_w_m2': self.flux_w_m2,
            'emissivity': self.emissivity,
        }


# =============================================================================
# SIMULATION TEMPLATE
# =============================================================================

@dataclass
class ThermalSimulationTemplate:
    """Complete template for thermal simulation setup"""
    
    name: str
    description: str = ""
    
    # Solver
    solver: Literal["calculix", "openfoam_laplacian", "openfoam_buoyant", "auto"] = "auto"
    
    # Material
    material: str = "Aluminum_6061"
    
    # Boundary conditions (list of BCs)
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    
    # Simplified BCs for quick setup (alternative to full BC list)
    hot_surface_temp_c: Optional[float] = None
    cold_surface_temp_c: Optional[float] = None
    ambient_temp_c: float = 25.0
    convection_coeff: float = 25.0
    
    # Simulation type
    steady_state: bool = True
    duration_s: float = 60.0
    time_step_s: float = 1.0
    
    # Solver settings
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    
    # Pass/fail criteria
    max_temp_limit_c: Optional[float] = None  # Auto from material if None
    min_temp_limit_c: float = -40.0
    max_gradient_c_mm: Optional[float] = None
    
    # Mesh requirements
    min_elements: int = 1000
    max_aspect_ratio: float = 20.0
    
    def __post_init__(self):
        # Auto-set max temp from material if not specified
        if self.max_temp_limit_c is None:
            try:
                mat = get_material(self.material)
                if mat.max_service_temp_c:
                    self.max_temp_limit_c = mat.max_service_temp_c
                else:
                    self.max_temp_limit_c = 150.0  # Default
            except KeyError:
                self.max_temp_limit_c = 150.0
    
    def get_material_properties(self) -> MaterialProperties:
        """Get the material properties object"""
        return get_material(self.material)
    
    def to_config_dict(self) -> Dict:
        """Convert to configuration dictionary for orchestrator"""
        mat = self.get_material_properties()
        
        return {
            'solver': self.solver,
            'physics': {
                'simulation_type': 'thermal',
                'material': self.material,
                'thermal_conductivity': mat.thermal_conductivity,
                'density': mat.density,
                'specific_heat': mat.specific_heat,
                'ambient_temp_c': self.ambient_temp_c,
                'source_temp_c': self.hot_surface_temp_c or 100.0,
                'convection_coeff': self.convection_coeff,
                'transient': not self.steady_state,
                'duration': self.duration_s,
                'time_step': self.time_step_s,
                'max_iterations': self.max_iterations,
                'convergence_tolerance': self.convergence_tolerance,
            },
            'pass_fail': {
                'max_temp_limit_c': self.max_temp_limit_c,
                'min_temp_limit_c': self.min_temp_limit_c,
                'max_gradient_c_mm': self.max_gradient_c_mm,
            },
            'mesh_requirements': {
                'min_elements': self.min_elements,
                'max_aspect_ratio': self.max_aspect_ratio,
            },
            'boundary_conditions': [bc.to_dict() for bc in self.boundary_conditions],
        }
    
    def save(self, filepath: Path):
        """Save template to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_config_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ThermalSimulationTemplate':
        """Load template from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        physics = data.get('physics', {})
        pass_fail = data.get('pass_fail', {})
        mesh_req = data.get('mesh_requirements', {})
        
        return cls(
            name=filepath.stem,
            solver=data.get('solver', 'auto'),
            material=physics.get('material', 'Aluminum_6061'),
            hot_surface_temp_c=physics.get('source_temp_c'),
            cold_surface_temp_c=None,
            ambient_temp_c=physics.get('ambient_temp_c', 25.0),
            convection_coeff=physics.get('convection_coeff', 25.0),
            steady_state=not physics.get('transient', False),
            duration_s=physics.get('duration', 60.0),
            time_step_s=physics.get('time_step', 1.0),
            max_iterations=physics.get('max_iterations', 1000),
            convergence_tolerance=physics.get('convergence_tolerance', 1e-6),
            max_temp_limit_c=pass_fail.get('max_temp_limit_c', 150.0),
            min_temp_limit_c=pass_fail.get('min_temp_limit_c', -40.0),
            max_gradient_c_mm=pass_fail.get('max_gradient_c_mm'),
            min_elements=mesh_req.get('min_elements', 1000),
            max_aspect_ratio=mesh_req.get('max_aspect_ratio', 20.0),
        )


# =============================================================================
# PRESET TEMPLATE LIBRARY
# =============================================================================

TEMPLATE_LIBRARY: Dict[str, ThermalSimulationTemplate] = {
    "electronics_cooling": ThermalSimulationTemplate(
        name="electronics_cooling",
        description="Electronics enclosure cooling analysis",
        solver="calculix",
        material="Aluminum_6061",
        hot_surface_temp_c=85.0,
        ambient_temp_c=25.0,
        convection_coeff=50.0,
        steady_state=True,
        max_temp_limit_c=100.0,
    ),
    
    "heat_sink": ThermalSimulationTemplate(
        name="heat_sink",
        description="Heat sink thermal performance evaluation",
        solver="calculix",
        material="Aluminum_6061",
        hot_surface_temp_c=100.0,
        ambient_temp_c=25.0,
        convection_coeff=100.0,  # High for forced convection
        steady_state=True,
        max_temp_limit_c=120.0,
    ),
    
    "led_housing": ThermalSimulationTemplate(
        name="led_housing",
        description="LED housing thermal management",
        solver="calculix",
        material="Aluminum_6061",
        hot_surface_temp_c=80.0,
        ambient_temp_c=35.0,  # Higher ambient for enclosed fixtures
        convection_coeff=10.0,  # Natural convection
        steady_state=True,
        max_temp_limit_c=85.0,  # LED junction temp limit
    ),
    
    "motor_housing": ThermalSimulationTemplate(
        name="motor_housing",
        description="Electric motor housing heat dissipation",
        solver="calculix",
        material="Steel_AISI_1020",
        hot_surface_temp_c=120.0,
        ambient_temp_c=40.0,
        convection_coeff=20.0,
        steady_state=True,
        max_temp_limit_c=180.0,
    ),
    
    "conduction_bench": ThermalSimulationTemplate(
        name="conduction_bench",
        description="Pure conduction benchmark test",
        solver="calculix",
        material="Copper_Pure",
        hot_surface_temp_c=100.0,
        cold_surface_temp_c=20.0,
        ambient_temp_c=20.0,
        convection_coeff=0.0,  # No convection
        steady_state=True,
        max_temp_limit_c=200.0,
    ),
    
    "rocket_nozzle": ThermalSimulationTemplate(
        name="rocket_nozzle",
        description="Rocket nozzle extreme thermal analysis",
        solver="calculix",
        material="Inconel_718",
        hot_surface_temp_c=800.0,
        ambient_temp_c=25.0,
        convection_coeff=500.0,  # Very high due to exhaust flow
        steady_state=False,  # Transient required
        duration_s=10.0,
        time_step_s=0.1,
        max_temp_limit_c=1000.0,
    ),
    
    "battery_pack": ThermalSimulationTemplate(
        name="battery_pack",
        description="Li-ion battery pack thermal analysis",
        solver="calculix",
        material="Aluminum_6061",
        hot_surface_temp_c=60.0,
        ambient_temp_c=25.0,
        convection_coeff=30.0,
        steady_state=False,  # Transient for charge/discharge cycles
        duration_s=3600.0,  # 1 hour
        time_step_s=10.0,
        max_temp_limit_c=45.0,  # Battery safety limit
    ),
    
    "pcb_analysis": ThermalSimulationTemplate(
        name="pcb_analysis",
        description="PCB thermal hotspot analysis",
        solver="calculix",
        material="Glass_Borosilicate",  # FR4 approximation
        hot_surface_temp_c=90.0,
        ambient_temp_c=25.0,
        convection_coeff=15.0,
        steady_state=True,
        max_temp_limit_c=105.0,  # Component junction limit
    ),
    
    "exhaust_manifold": ThermalSimulationTemplate(
        name="exhaust_manifold",
        description="Automotive exhaust manifold",
        solver="calculix",
        material="Steel_304SS",
        hot_surface_temp_c=700.0,
        ambient_temp_c=50.0,  # Engine bay
        convection_coeff=40.0,
        steady_state=True,
        max_temp_limit_c=900.0,
    ),
    
    "cryogenic_vessel": ThermalSimulationTemplate(
        name="cryogenic_vessel",
        description="Cryogenic vessel heat leak analysis",
        solver="calculix",
        material="Steel_304SS",
        hot_surface_temp_c=-150.0,  # LN2 side
        cold_surface_temp_c=-196.0,
        ambient_temp_c=25.0,
        convection_coeff=5.0,  # With insulation
        steady_state=True,
        min_temp_limit_c=-200.0,
        max_temp_limit_c=50.0,
    ),
}


def get_template(name: str) -> ThermalSimulationTemplate:
    """Get a template by name"""
    if name in TEMPLATE_LIBRARY:
        return TEMPLATE_LIBRARY[name]
    
    raise KeyError(f"Template '{name}' not found. "
                   f"Available: {list(TEMPLATE_LIBRARY.keys())}")


def list_templates() -> List[str]:
    """List all available template names"""
    return list(TEMPLATE_LIBRARY.keys())


def list_materials() -> List[str]:
    """List all available material names"""
    return list(MATERIAL_LIBRARY.keys())


# =============================================================================
# OPENFOAM CASE GENERATION
# =============================================================================

class OpenFOAMCaseGenerator:
    """Generate OpenFOAM case files from template"""
    
    def __init__(self, template: ThermalSimulationTemplate):
        self.template = template
        self.material = template.get_material_properties()
    
    def generate_laplacian_foam_case(self, case_dir: Path):
        """Generate complete laplacianFoam case for pure conduction"""
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        (case_dir / "0").mkdir(exist_ok=True)
        (case_dir / "constant").mkdir(exist_ok=True)
        (case_dir / "system").mkdir(exist_ok=True)
        
        # Generate files
        self._write_control_dict(case_dir / "system" / "controlDict")
        self._write_fv_schemes(case_dir / "system" / "fvSchemes")
        self._write_fv_solution(case_dir / "system" / "fvSolution")
        self._write_transport_properties(case_dir / "constant" / "transportProperties")
        self._write_temperature_field(case_dir / "0" / "T")
    
    def _write_control_dict(self, filepath: Path):
        content = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     laplacianFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {self.template.duration_s if not self.template.steady_state else 1};
deltaT          {self.template.time_step_s if not self.template.steady_state else 1};
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
'''
        filepath.write_text(content)
    
    def _write_fv_schemes(self, filepath: Path):
        content = '''FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
}

laplacianSchemes
{
    default         Gauss linear orthogonal;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         orthogonal;
}
'''
        filepath.write_text(content)
    
    def _write_fv_solution(self, filepath: Path):
        content = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    T
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       {self.template.convergence_tolerance};
        relTol          0.01;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
}}
'''
        filepath.write_text(content)
    
    def _write_transport_properties(self, filepath: Path):
        # Calculate thermal diffusivity: DT = k / (rho * cp)
        DT = self.material.thermal_conductivity / (
            self.material.density * self.material.specific_heat
        )
        
        content = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

// Material: {self.material.name}
// k = {self.material.thermal_conductivity} W/m·K
// rho = {self.material.density} kg/m³
// cp = {self.material.specific_heat} J/kg·K

DT              DT [ 0 2 -1 0 0 0 0 ] {DT:.6e};
'''
        filepath.write_text(content)
    
    def _write_temperature_field(self, filepath: Path):
        # Convert to Kelvin for OpenFOAM
        T_hot = (self.template.hot_surface_temp_c or 100.0) + 273.15
        T_cold = (self.template.cold_surface_temp_c or self.template.ambient_temp_c) + 273.15
        T_init = self.template.ambient_temp_c + 273.15
        
        content = f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}}

dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_init};

boundaryField
{{
    hotWall
    {{
        type            fixedValue;
        value           uniform {T_hot};
    }}
    
    coldWall
    {{
        type            fixedValue;
        value           uniform {T_cold};
    }}
    
    defaultFaces
    {{
        type            zeroGradient;
    }}
}}
'''
        filepath.write_text(content)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Materials
    'MaterialProperties',
    'MATERIAL_LIBRARY',
    'get_material',
    'list_materials',
    
    # Boundary conditions
    'BCType',
    'BoundaryCondition',
    
    # Templates
    'ThermalSimulationTemplate',
    'TEMPLATE_LIBRARY',
    'get_template',
    'list_templates',
    
    # OpenFOAM
    'OpenFOAMCaseGenerator',
]
