"""
CAD Cleaning Module
===================

Provides CAD geometry preprocessing before meshing.
This module is architecturally airlocked from meshing strategies
to allow independent modification and toggling.

Main entry point: apply_cad_cleaning(config)
"""

from .sharp_edge_classifier import classify_sharp_edges, apply_sharp_edge_fields
from .geometry_cleanup import GeometryCleanup

__all__ = [
    'apply_cad_cleaning',
    'classify_sharp_edges',
    'apply_sharp_edge_fields',
    'GeometryCleanup',
]


def apply_cad_cleaning(config, max_size_mm: float = None, log_fn=None) -> dict:
    """
    Main entry point for CAD cleaning pathway.
    
    Runs all enabled CAD cleaning operations before meshing.
    
    Args:
        config: Configuration object (must have cad_cleaning attribute)
        max_size_mm: Maximum element size (used for sharp edge field sizing)
        log_fn: Optional logging function (accepts string)
    
    Returns:
        Dictionary with cleaning results and statistics
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)
    
    results = {
        'sharp_edges_detected': 0,
        'fields_applied': False,
        'enabled': False,
    }
    
    # Check if CAD cleaning is enabled
    cad_cleaning_config = getattr(config, 'cad_cleaning', None)
    if cad_cleaning_config is None:
        # No config → use defaults (enabled)
        enabled = True
        threshold_deg = 30.0
    else:
        enabled = getattr(cad_cleaning_config, 'enabled', True)
        threshold_deg = getattr(cad_cleaning_config, 'sharp_angle_threshold_deg', 30.0)
    
    if not enabled:
        log("[CAD Cleaning] Disabled by configuration")
        return results
    
    results['enabled'] = True
    log(f"[CAD Cleaning] Running sharp edge classification (threshold: {threshold_deg}°)...")
    
    # Classify sharp edges (verbose=True for diagnostic output)
    sharp_curves = classify_sharp_edges(threshold_deg=threshold_deg, log_fn=log, verbose=True)
    results['sharp_edges_detected'] = len(sharp_curves)
    
    if sharp_curves and max_size_mm:
        log(f"[CAD Cleaning] Detected {len(sharp_curves)} sharp edges, applying size fields...")
        apply_sharp_edge_fields(sharp_curves, max_size=max_size_mm)
        results['fields_applied'] = True
        log(f"[CAD Cleaning] Sharp edge fields applied (size: {max_size_mm}mm)")
    elif sharp_curves:
        log(f"[CAD Cleaning] Detected {len(sharp_curves)} sharp edges (no max_size_mm provided, skipping fields)")
    else:
        log("[CAD Cleaning] No sharp edges detected")
    
    return results
