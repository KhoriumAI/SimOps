"""
Structural Visualization Module
================================

Post-processing pipeline for structural FEM results:
- Von Mises stress extraction from CalculiX .frd files
- Mesh warping by displacement
- Pass/Fail evaluation against material yield strength
- PDF report generation

Usage:
    from core.reporting.structural_viz import generate_structural_report
    report = generate_structural_report(result_data, output_dir, job_name)
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

# Material constants
AL6061_YIELD_STRENGTH_MPA = 276.0  # MPa


def extract_von_mises_from_frd(frd_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Von Mises stress and displacement from CalculiX .frd results file.
    
    Args:
        frd_file: Path to CalculiX .frd output file
        
    Returns:
        Tuple of (node_coords, displacement, von_mises_stress)
    """
    node_coords = []
    displacement = []
    stress_components = []  # S11, S22, S33, S12, S13, S23
    
    current_block = None
    node_ids = []
    
    with open(frd_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Node coordinates block
        if line.startswith('2C') and 'NODAL' in line:
            current_block = 'NODES'
            i += 1
            continue
        
        # Displacement block
        if line.startswith('1PSTEP') and 'DISP' in lines[i-1] if i > 0 else False:
            current_block = 'DISP'
            i += 1
            continue
            
        if ' 100CL' in line and 'DISP' in line:
            current_block = 'DISP'
            i += 1
            continue
        
        # Stress block
        if ' 100CL' in line and 'STRESS' in line:
            current_block = 'STRESS'
            i += 1
            continue
        
        # End of block
        if line.startswith('-3'):
            current_block = None
            i += 1
            continue
        
        # Parse node data
        if current_block == 'NODES' and line.startswith('-1'):
            try:
                parts = line.split()
                if len(parts) >= 4:
                    node_id = int(parts[1])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    node_ids.append(node_id)
                    node_coords.append([x, y, z])
            except:
                pass
        
        # Parse displacement data
        if current_block == 'DISP' and line.startswith('-1'):
            try:
                parts = line.split()
                if len(parts) >= 4:
                    dx, dy, dz = float(parts[2]), float(parts[3]), float(parts[4])
                    displacement.append([dx, dy, dz])
            except:
                pass
        
        # Parse stress data (6 components)
        if current_block == 'STRESS' and line.startswith('-1'):
            try:
                parts = line.split()
                if len(parts) >= 7:
                    # S11, S22, S33, S12, S13, S23
                    stress = [float(parts[j]) for j in range(2, 8)]
                    stress_components.append(stress)
            except:
                pass
        
        i += 1
    
    node_coords = np.array(node_coords) if node_coords else np.zeros((0, 3))
    displacement = np.array(displacement) if displacement else np.zeros_like(node_coords)
    stress_components = np.array(stress_components) if stress_components else np.zeros((len(node_coords), 6))
    
    # Calculate Von Mises stress from components
    # σ_vm = sqrt(0.5 * ((σ11-σ22)² + (σ22-σ33)² + (σ33-σ11)² + 6*(σ12² + σ23² + σ13²)))
    if len(stress_components) > 0:
        s = stress_components
        von_mises = np.sqrt(0.5 * (
            (s[:, 0] - s[:, 1])**2 +
            (s[:, 1] - s[:, 2])**2 +
            (s[:, 2] - s[:, 0])**2 +
            6 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
        ))
    else:
        von_mises = np.zeros(len(node_coords))
    
    return node_coords, displacement, von_mises


def generate_structural_visualization(
    node_coords: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    von_mises: np.ndarray,
    output_path: Path,
    title: str = "Structural Analysis",
    warp_scale: float = 100.0,
    **kwargs
) -> bool:
    """
    Generate visualization with warped mesh colored by Von Mises stress.
    
    Args:
        node_coords: Node coordinates (N, 3)
        elements: Element connectivity (M, 4 or 10)
        displacement: Displacement vectors (N, 3)
        von_mises: Von Mises stress values (N,)
        output_path: Output image path
        title: Title for the plot
        warp_scale: Scale factor for displacement visualization
        
    Returns:
        True if successful
    """
    if not PYVISTA_AVAILABLE:
        logger.warning("PyVista not available for visualization")
        return False
    
    try:
        # Create PyVista mesh
        nodes_per_elem = elements.shape[1] if elements.ndim == 2 else 4
        
        # PyVista format: [num_points, p0, p1, p2, p3, ...]
        # elements[:, 0] is the element tag, elements[:, 1:] are the node tags
        # Subtract 1 to convert from 1-indexed GMSH/CalculiX to 0-indexed PyVista
        
        # Determine if first column is tags
        # Adapter output might be 0-based indices without tags now.
        has_tags = False
        if nodes_per_elem in [5, 11]:
             has_tags = True
        elif nodes_per_elem == 10 and elements.shape[1] == 10:
             # If shape is 10, assume NO tags (standard Tet10 node count)
             # Unless we have heuristics check? 
             # Safe assumption: My remapping produces 10 cols.
             has_tags = False
        elif nodes_per_elem == 4 and elements.shape[1] == 4:
             has_tags = False
             
        start_col = 1 if has_tags else 0
        
        # Check 0-based
        # If min value is 0, it's 0-based. If min >= 1, likely 1-based.
        sample_slice = elements[:, start_col:start_col+4]
        is_one_based = np.min(sample_slice) >= 1
        offset = 1 if is_one_based else 0
        
        if nodes_per_elem in [10, 11]: # TET10
            # TET10 - only use corner nodes for visualization
            indices = (elements[:, start_col:start_col+4] - offset).astype(int)
            cells = np.column_stack([
                np.full(len(elements), 4),
                indices
            ]).flatten()
            cell_type = np.full(len(elements), 10)  # VTK_TETRA
        elif nodes_per_elem in [4, 5]: # TET4
            indices = (elements[:, start_col:start_col+4] - offset).astype(int)
            cells = np.column_stack([
                np.full(len(elements), 4),
                indices
            ]).flatten()
            cell_type = np.full(len(elements), 10)  # VTK_TETRA
        else:
            # Fallback/Direct
            indices = (elements - offset).astype(int)
            cells = np.column_stack([
                np.full(len(elements), indices.shape[1]),
                indices
            ]).flatten()
            cell_type = np.full(len(elements), 10) # Assume Tet
        
        # Create unstructured grid
        mesh = pv.UnstructuredGrid(cells, cell_type, node_coords)
        
        # Add data
        # von_mises is already in MPa from the solver/adapter
        mesh.point_data['Von Mises Stress (MPa)'] = von_mises
        mesh.point_data['Displacement'] = displacement
        
        # Apply warp
        if np.max(np.abs(displacement)) > 1e-12:
            warped = mesh.warp_by_vector('Displacement', factor=warp_scale)
        else:
            warped = mesh
        
        # Extract surface for better visualization
        surface = warped.extract_surface()
        
        # Setup plotter
        # Width/height ratio should be 4:3 for report
        pl = pv.Plotter(off_screen=True, window_size=[1024, 768])
        pl.set_background('white')
        
        # Add mesh with stress coloring
        pl.add_mesh(
            surface,
            scalars='Von Mises Stress (MPa)',
            cmap='jet',
            show_edges=False,
            lighting=True,
            scalar_bar_args={
                'title': 'Von Mises (MPa)',
                'n_labels': 5,
                'fmt': '%.1f',
                'title_font_size': 14,
                'label_font_size': 10,
                'color': 'black',
                'position_x': 0.78, # Move left to give label space
                'position_y': 0.1,
                'width': 0.18, # Increase width significantly
                'height': 0.8
            }
        )
        
        # Find and annotate max stress location
        max_idx = np.argmax(von_mises)
        max_stress_mpa = von_mises[max_idx]
        max_pos = node_coords[max_idx]
        
        if np.max(np.abs(displacement)) > 1e-12:
            # Apply warp to annotation position
            max_pos = max_pos + displacement[max_idx] * warp_scale
        
        pl.add_point_labels(
            [max_pos],
            [f"Max: {max_stress_mpa:.2f} MPa"],
            font_size=14,
            point_size=15,
            point_color='red',
            text_color='black',
            shape='rounded_rect',
            fill_shape=True,
            shape_color='white',
            shape_opacity=0.8
        )
        
        # Camera setup
        pl.enable_parallel_projection()
        
        view = kwargs.get('view', 'iso')
        if view == 'iso':
            pl.view_isometric()
        elif view == 'top':
            pl.view_xy()  # Z up
        elif view == 'front':
            pl.view_xz()  # Y up (or -Y?) - Standard engineering front view often XZ
            pl.camera.up = (0, 0, 1) # Ensure Z is up
        elif view == 'side':
            pl.view_yz()  # X up?
            pl.camera.up = (0, 0, 1)
            
        pl.camera.zoom(1.2)
        
        # Add title
        pl.add_text(
            f"{title} ({view.capitalize()} View)",
            position='upper_left',
            font_size=18,
            color='black'
        )
        
        # Save
        pl.screenshot(str(output_path))
        pl.close()
        
        logger.info(f"Visualization saved: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return False


def generate_multi_view_structural_viz(
    node_coords: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    von_mises: np.ndarray,
    output_dir: Path,
    job_name: str,
    title: str = "Structural Analysis",
    warp_scale: float = 100.0
) -> List[str]:
    """
    Generate multiple visualization views (Iso, Top, Front, Side).
    Returns list of generated image paths.
    """
    views = ['iso', 'top', 'front', 'side']
    generated_images = []
    
    for view in views:
        img_path = output_dir / f"{job_name}_stress_{view}.png"
        success = generate_structural_visualization(
            node_coords, elements, displacement, von_mises,
            img_path, title, warp_scale, view=view
        )
        if success:
            generated_images.append(str(img_path))
            
    return generated_images


def evaluate_pass_fail(
    max_stress_pa: float,
    yield_strength_pa: float = AL6061_YIELD_STRENGTH_MPA * 1e6,
    safety_factor: float = 1.0
) -> Tuple[bool, str]:
    """
    Evaluate pass/fail based on stress vs yield strength.
    
    Args:
        max_stress_pa: Maximum Von Mises stress in Pa
        yield_strength_pa: Material yield strength in Pa
        safety_factor: Required safety factor (default: 1.0)
        
    Returns:
        Tuple of (passed: bool, message: str)
    """
    allowable = yield_strength_pa / safety_factor
    margin = (allowable - max_stress_pa) / allowable * 100
    
    if max_stress_pa < allowable:
        return True, f"PASS - Max stress {max_stress_pa/1e6:.1f} MPa < Yield {allowable/1e6:.1f} MPa (Margin: {margin:.1f}%)"
    else:
        return False, f"FAIL - Max stress {max_stress_pa/1e6:.1f} MPa > Yield {allowable/1e6:.1f} MPa (Over by: {-margin:.1f}%)"


def generate_structural_report(
    result: Dict,
    output_dir: Path,
    job_name: str,
    g_factor: float = 10.0
) -> Dict:
    """
    Generate complete structural analysis report.
    
    Args:
        result: Result dictionary from CalculiX solver
        output_dir: Output directory
        job_name: Job name for file naming
        g_factor: G-factor used for the analysis
        
    Returns:
        Dict with report file paths and summary
    """
    output_dir = Path(output_dir)
    
    report_output = {
        'png_file': None,
        'pdf_file': None,
        'max_stress_mpa': 0,
        'max_displacement_mm': 0,
        'passed': False,
        'message': ''
    }
    
    # Extract data from result
    node_coords = np.array(result.get('node_coords', []))
    elements = np.array(result.get('elements', []))
    displacement = result.get('displacement')
    von_mises = result.get('von_mises')
    
    if von_mises is None or len(von_mises) == 0:
        logger.warning("No Von Mises stress data available")
        return report_output
    
    von_mises = np.array(von_mises)
    displacement = np.array(displacement) if displacement is not None else np.zeros_like(node_coords)
    
    # Calculate metrics
    # CCX outputs MPa for stress, mm for displacement (if scale=1.0)
    max_stress_mpa = np.max(von_mises)
    
    # Displacement
    max_disp_val = np.max(np.linalg.norm(displacement, axis=1)) if displacement.size > 0 else 0
    # max_disp_val is in mm
    max_disp_mm = max_disp_val
    
    # Display string for displacement
    if max_disp_mm < 0.1:
        disp_display = f"{max_disp_mm * 1000:.2f} um"
    else:
        disp_display = f"{max_disp_mm:.4f} mm"
        
    report_output['max_stress_mpa'] = max_stress_mpa
    report_output['max_displacement_mm'] = max_disp_mm
    report_output['displacement_display'] = disp_display
    
    # Pass/Fail evaluation
    max_stress_pa = max_stress_mpa * 1e6
    passed, message = evaluate_pass_fail(max_stress_pa)
    report_output['passed'] = passed
    report_output['message'] = message
    
    logger.info(f"[StructuralReport] Max Stress: {max_stress_mpa:.2f} MPa")
    logger.info(f"[StructuralReport] Max Displacement: {max_disp_mm:.4f} mm")
    logger.info(f"[StructuralReport] {message}")
    
    # Generate visualization
    # png_path = output_dir / f"{job_name}_stress.png" 
    # Use multi-view generator
    title = f"Static Structural: {g_factor:.0f}G Z-Load"
    
    generated_images = generate_multi_view_structural_viz(
        node_coords=node_coords,
        elements=elements,
        displacement=displacement,
        von_mises=von_mises,
        output_dir=output_dir,
        job_name=job_name,
        title=title,
        warp_scale=100.0
    )
    
    # viz_success = len(generated_images) > 0
    
    if generated_images:
        # Assume first image (iso) is primary for legacy support
        report_output['png_file'] = generated_images[0]
    
    # Generate PDF report
    try:
        from core.reporting.structural_report import StructuralPDFReportGenerator
        
        pdf_data = {
            'job_name': job_name,
            'g_factor': g_factor,
            'max_stress_mpa': max_stress_mpa,
            'max_displacement_mm': max_disp_mm,
            'displacement_display': disp_display,
            'yield_strength_mpa': AL6061_YIELD_STRENGTH_MPA,
            'passed': passed,
            'message': message,
            'num_nodes': len(node_coords),
            'num_elements': len(elements),
            'material': 'Al6061-T6',
            'success': passed # Ensure success status is passed correctly
        }
        
        generator = StructuralPDFReportGenerator()
        pdf_file = generator.generate(
            job_name=job_name,
            output_dir=output_dir,
            data=pdf_data,
            image_paths=generated_images # Pass all images
        )
        report_output['pdf_file'] = str(pdf_file)
        
    except ImportError:
        logger.warning("PDF generator not available")
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
    
    return report_output


def generate_report_from_vtk(
    vtk_file: str,
    output_dir: Path,
    job_name: str,
    g_factor: float = 10.0
) -> Dict:
    """
    Generate report from VTK file with stress data.
    
    Args:
        vtk_file: Path to VTK file with Von Mises stress
        output_dir: Output directory
        job_name: Job name
        g_factor: G-factor used
        
    Returns:
        Report output dictionary
    """
    if not PYVISTA_AVAILABLE:
        logger.error("PyVista required for VTK processing")
        return {}
    
    try:
        mesh = pv.read(vtk_file)
        
        result = {
            'node_coords': mesh.points,
            'von_mises': mesh.point_data.get('VonMisesStress', mesh.point_data.get('Von Mises (MPa)', np.zeros(len(mesh.points)))),
            'displacement': mesh.point_data.get('Displacement', np.zeros((len(mesh.points), 3)))
        }
        
        # Handle cell-to-point conversion if needed
        if result['von_mises'] is None or len(result['von_mises']) == 0:
            if 'VonMisesStress' in mesh.cell_data:
                mesh = mesh.cell_data_to_point_data()
                result['von_mises'] = mesh.point_data.get('VonMisesStress', np.zeros(len(mesh.points)))
        
        # Extract elements from cells
        cells = mesh.cells
        cell_types = mesh.celltypes
        
        elements = []
        offset = 0
        for ctype in cell_types:
            if ctype == 10:  # VTK_TETRA
                nodes = cells[offset+1:offset+5]
                elements.append(nodes)
                offset += 5
            elif ctype == 24:  # VTK_QUADRATIC_TETRA
                nodes = cells[offset+1:offset+11]
                elements.append(nodes)
                offset += 11
            else:
                # Skip other types, estimate offset
                offset += 5
        
        result['elements'] = np.array(elements) if elements else np.zeros((0, 4), dtype=int)
        
        return generate_structural_report(result, output_dir, job_name, g_factor)
        
    except Exception as e:
        logger.error(f"Failed to process VTK file: {e}")
        return {}
