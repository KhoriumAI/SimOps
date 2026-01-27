"""
OpenFOAM Boundary Parser
========================

Parses OpenFOAM boundary files to extract patch names and types.
"""

import re
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def parse_boundary_file(case_dir: Path) -> Dict[str, str]:
    """
    Parses the constant/polyMesh/boundary file to extract patch names and types.
    
    Args:
        case_dir: Path to OpenFOAM case directory
        
    Returns:
        Dictionary mapping patch_name -> patch_type (e.g., 'inlet' -> 'patch')
    """
    boundary_file = case_dir / "constant" / "polyMesh" / "boundary"
    
    if not boundary_file.exists():
        raise FileNotFoundError(f"Mesh boundary file not found: {boundary_file}")

    content = boundary_file.read_text()

    patches = {}
    
    # Regex to find patch definitions: name { type patch; ... }
    pattern = re.compile(r'([a-zA-Z0-9_]+)\s*\{\s*type\s+([a-zA-Z0-9_]+);', re.DOTALL)
    
    for match in pattern.finditer(content):
        name = match.group(1)
        p_type = match.group(2)
        
        # Filter out false positives
        if name not in ['boundary', 'FoamFile', 'Foo']: 
            patches[name] = p_type
            
    logger.info(f"[OpenFOAM Parser] Found {len(patches)} patches: {list(patches.keys())}")
    return patches


def classify_thermal_bc(patch_name: str, patch_type: str, config: Dict) -> Dict:
    """
    Classify a boundary patch for thermal boundary conditions.
    
    Args:
        patch_name: Name of the patch
        patch_type: Type from boundary file ('patch', 'wall', 'empty', etc.)
        config: Configuration dict with optional:
            - hot_patches: List of patch names that should be hot (fixedValue)
            - cold_patches: List of patch names that should be cold (fixedValue)
            - hot_temperature: Temperature for hot patches (default: 350K)
            - cold_temperature: Temperature for cold patches (default: 300K)
            - ambient_temperature: Initial/ambient temperature (default: 300K)
    
    Returns:
        Dict with 'type' and 'value' for the boundaryField entry
    """
    patch_lower = patch_name.lower()
    
    # Check explicit configuration
    hot_patches = config.get('hot_patches', [])
    cold_patches = config.get('cold_patches', [])
    
    if patch_name in hot_patches or any(hp.lower() in patch_lower for hp in hot_patches):
        return {
            'type': 'fixedValue',
            'value': config.get('hot_temperature', 350.0)
        }
    
    if patch_name in cold_patches or any(cp.lower() in patch_lower for cp in cold_patches):
        return {
            'type': 'fixedValue',
            'value': config.get('cold_temperature', 300.0)
        }
    
    # Heuristic-based classification
    # Hot patches: contains 'hot', 'source', 'chip', 'heater'
    if any(keyword in patch_lower for keyword in ['hot', 'source', 'chip', 'heater', 'power']):
        return {
            'type': 'fixedValue',
            'value': config.get('hot_temperature', 350.0)
        }
    
    # Cold patches: contains 'cold', 'ambient', 'cool', 'sink'
    if any(keyword in patch_lower for keyword in ['cold', 'ambient', 'cool', 'sink', 'bottom']):
        return {
            'type': 'fixedValue',
            'value': config.get('cold_temperature', 300.0)
        }
    
    # Default: zeroGradient (insulated wall)
    # This is appropriate for most walls in thermal-only simulations
    return {
        'type': 'zeroGradient',
        'value': None  # Not needed for zeroGradient
    }
