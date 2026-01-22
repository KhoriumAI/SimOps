"""
GUI Stability Utilities - Part of the Khorium Toolbox.

This module provides generalized solutions for common GUI rendering and 
stability issues across different OS configurations.
"""

import gmsh

def apply_high_dpi_fix(font_size=12, graphics_font_size=14):
    """
    Forces High-DPI awareness in Gmsh to prevent 'Black Bar' artifacts 
    when dragging windows between monitors with different scaling factors.
    
    Args:
        font_size (int): Base font size for GUI widgets.
        graphics_font_size (int): Font size for 3D viewport text.
    """
    # Force High-DPI awareness - queries physical pixels instead of virtualized OS coordinates
    gmsh.option.setNumber("General.HighResolutionGraphics", 1) 

    # Scale GUI fonts/widgets manually for high-resolution displays
    gmsh.option.setNumber("General.FontSize", font_size) 
    gmsh.option.setNumber("General.GraphicsFontSize", graphics_font_size)

    print(f"[KHORIUM] High-DPI awareness enabled (Fonts: {font_size}/{graphics_font_size})")

def get_python_dpi_instructions():
    """
    Returns instructions for manual Windows OS-level DPI override.
    """
    return (
        "Windows OS Fix for High-DPI Artifacts:\n"
        "1. Right-click your python.exe -> Properties.\n"
        "2. Compatibility Tab -> Change high DPI settings.\n"
        "3. Check 'Override high DPI scaling behavior'.\n"
        "4. Select 'Application' from the dropdown."
    )
