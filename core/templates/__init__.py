"""
Simulation Templates module
"""

from .simulation_templates import (
    # Materials
    MaterialProperties,
    MATERIAL_LIBRARY,
    get_material,
    list_materials,
    
    # Boundary conditions
    BCType,
    BoundaryCondition,
    
    # Templates
    ThermalSimulationTemplate,
    TEMPLATE_LIBRARY,
    get_template,
    list_templates,
    
    # OpenFOAM
    OpenFOAMCaseGenerator,
)

__all__ = [
    'MaterialProperties',
    'MATERIAL_LIBRARY',
    'get_material',
    'list_materials',
    'BCType',
    'BoundaryCondition',
    'ThermalSimulationTemplate',
    'TEMPLATE_LIBRARY',
    'get_template',
    'list_templates',
    'OpenFOAMCaseGenerator',
]
