# core/solvers/openfoam_parser.py
import re
import os
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

def parse_boundary_file(case_dir: str) -> Dict[str, str]:
    """
    Parses the constant/polyMesh/boundary file to extract patch names and types.
    Returns a dictionary: { 'patch_name': 'patch_type' }
    """
    boundary_file = os.path.join(str(case_dir), "constant", "polyMesh", "boundary")
    
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(f"Mesh boundary file not found: {boundary_file}")

    with open(boundary_file, 'r') as f:
        content = f.read()

    # Regex to find patch definitions: name { type patch; ... }
    # This captures the block structure of OpenFOAM files roughly.
    # We look for a name, then a block starting with { containing type.
    # Note: Regex parsing of nested braces is fragile, but OpenFOAM boundary files 
    # usually have a flat structure inside the list.
    
    # Structure:
    # patchIdentifier
    # {
    #     type patch;
    #     ...
    # }
    
    patches = {}
    
    # Improved Regex:
    # 1. Match identifier (alphanumeric+underscores)
    # 2. Match opening brace
    # 3. Match 'type' property
    # 4. Capture the type value
    
    # We loop through matches.
    # NOTE: The file starts with a header and a count. We skip those generally by regex nature.
    
    pattern = re.compile(r'([a-zA-Z0-9_]+)\s*\{\s*type\s+([a-zA-Z0-9_]+);', re.DOTALL)
    
    for match in pattern.finditer(content):
        name = match.group(1)
        p_type = match.group(2)
        
        # Filter out common false positives if regex is too greedy (e.g. 'boundaryField' in 0/U)
        # But this parses polyMesh/boundary, which has cleaner structure.
        if name not in ['boundary', 'Foo', 'FoamFile']: 
            patches[name] = p_type
            
    logger.info(f"[OpenFOAM Parser] Found patches: {patches}")
    return patches

def verify_patches(required_patches: List[str], available_patches: Dict[str, str]) -> bool:
    """
    Ensures all physics-required patches exist in the mesh.
    Required patches are logical names (e.g. "inlet", "outlet").
    """
    available_names = set(available_patches.keys())
    required_set = set(required_patches)
    
    # Exact match check
    missing = required_set - available_names
    
    if missing:
        # Check if we have only case sensitivity issues?
        raise ValueError(f"FATAL: Mesh missing required patches for simulation: {missing}. Available: {list(available_names)}")
        
    return True
