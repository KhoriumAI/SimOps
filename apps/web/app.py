"""
Khorium MeshGen Web Interface
==============================

Exact replica of desktop GUI (gui_final.py) in the browser:
- Light theme matching desktop colors
- Same layout and progress bars
- VTK 3D visualization
- Quality overlay matching desktop style
"""

import streamlit as st
import streamlit.components.v1 as components
import subprocess
import tempfile
import json
import os
import time
import re
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple
from queue import Queue

# Page configuration
st.set_page_config(
    page_title="Khorium MeshGen",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- VTK HEADLESS SETUP FOR VERCEL ---
# Try to configure PyVista for headless server environments
try:
    import pyvista as pv
    # Force off-screen rendering for serverless environments
    pv.OFF_SCREEN = True
    # Disable Jupyter backend hooks that might look for a display
    pv.set_jupyter_backend(None)
except ImportError:
    pass
# -------------------------------------

# Custom CSS - COMPREHENSIVE Light Theme Enforcement
st.markdown("""
<style>
    /* --- GLOBAL THEME OVERRIDES --- */
    /* Force entire app to light background */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }

    /* Force ALL text to be dark gray, overriding Streamlit's auto-theming */
    p, h1, h2, h3, h4, h5, h6, li, span, label, .stMarkdown, .stText, div {
        color: #212529 !important;
    }

    /* --- SIDEBAR FIXES --- */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6;
    }
    /* Ensure all text in sidebar is dark */
    [data-testid="stSidebar"] * {
        color: #212529 !important;
    }

    /* --- INPUT WIDGETS (The tricky part for contrast) --- */
    /* Text Inputs, Number Inputs, Select Boxes */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #212529 !important;
        border-color: #dee2e6 !important;
    }
    /* Dropdown options text */
    div[role="listbox"] li {
        background-color: #ffffff !important;
        color: #212529 !important;
    }
    /* Sliders */
    .stSlider div[data-baseweb="slider"] {
        color: #212529 !important;
    }
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #dee2e6;
    }
    [data-testid="stFileUploader"] section {
        background-color: #f8f9fa !important;
    }
    [data-testid="stFileUploader"] small {
        color: #6c757d !important;
    }

    /* --- LAYOUT ELEMENTS --- */
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #0d6efd !important;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d !important;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }

    /* Progress bars container */
    .progress-container {
        background-color: white;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    /* Actual progress bar color (Green) */
    .stProgress > div > div > div > div {
        background-color: #28a745 !important;
    }

    /* Settings groups */
    .settings-group {
        background-color: #f8f9fa; /* Slightly darker than sidebar for contrast */
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0d6efd !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #0b5ed7 !important;
    }

    /* Console Output / Text Areas */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
        font-family: 'Courier New', monospace;
        border: 1px solid #dee2e6 !important;
    }

    /* Alerts (Success, Info, Warning) - force text color */
    .stAlert div[data-baseweb="alert"] {
        color: #212529 !important;
    }

    /* --- CUSTOM COMPONENTS --- */
    /* Quality overlay */
    .quality-overlay {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .quality-grade { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: #212529; }
    .quality-excellent { color: #198754 !important; }
    .quality-good { color: #0d6efd !important; }
    .quality-fair { color: #ffc107 !important; }
    .quality-poor { color: #fd7e14 !important; }
    .quality-critical { color: #dc3545 !important; }
</style>
""", unsafe_allow_html=True)


class MeshWorker:
    """Mesh generation worker matching desktop version"""

    def __init__(self):
        self.process = None
        self.is_running = False
        self.output_queue = Queue()
        self.phase_progress = {
            '1D Mesh': 0,
            '2D Mesh': 0,
            '3D Mesh': 0,
            'Optimization': 0,
            'Quality': 0,
            'Export': 0
        }
        self.phase_max = {}  # Track max to prevent jitter

    def start(self, cad_file: str, quality_params: dict):
        """Start mesh generation subprocess"""
        self.is_running = True
        self.phase_max = {}

        # Ensure the worker script exists
        worker_script = "mesh_worker_subprocess.py"
        if not os.path.exists(worker_script):
             # Fallback if running in a different directory structure
             self.output_queue.put(f"ERROR: {worker_script} not found in current directory")
             self.is_running = False
             return

        cmd = [
            "python3",
            worker_script,
            cad_file,
            "--quality-params",
            json.dumps(quality_params)
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Start output reader thread
        thread = threading.Thread(target=self._read_output, daemon=True)
        thread.start()

    def _read_output(self):
        """Read subprocess output and parse progress (matching desktop parsing)"""
        if not self.process:
            return

        for line in iter(self.process.stdout.readline, ''):
            if not line:
                break
            self.output_queue.put(line.strip())
            self._parse_progress(line)

        self.is_running = False
        self.process.wait()

    def _emit_progress(self, phase: str, value: int):
        """Emit progress only if it's higher than before (prevent jitter)"""
        if phase not in self.phase_max:
            self.phase_max[phase] = 0

        if value > self.phase_max[phase]:
            self.phase_max[phase] = value
            self.phase_progress[phase] = value

    def _complete_phase(self, phase: str):
        """Mark phase as 100% complete"""
        self.phase_max[phase] = 100
        self.phase_progress[phase] = 100

    def _parse_progress(self, line: str):
        """Parse progress from output (EXACT match to desktop gui_final.py)"""
        # 1D Meshing
        if "Meshing 1D" in line:
            self._emit_progress('1D Mesh', 10)
        elif "Meshing curve" in line and "order 2" not in line:
            match = re.search(r'\[\s*(\d+)%\]', line)
            if match:
                pct = int(match.group(1))
                self._emit_progress('1D Mesh', 10 + int(pct * 0.9))
        elif "Done meshing 1D" in line:
            self._complete_phase('1D Mesh')

        # 2D Meshing
        elif "Meshing 2D" in line:
            self._emit_progress('2D Mesh', 10)
        elif "Meshing surface" in line and "order 2" not in line:
            match = re.search(r'\[\s*(\d+)%\]', line)
            if match:
                pct = int(match.group(1))
                self._emit_progress('2D Mesh', 10 + int(pct * 0.9))
        elif "Done meshing 2D" in line:
            self._complete_phase('2D Mesh')

        # 3D Meshing
        elif "Meshing 3D" in line:
            self._emit_progress('3D Mesh', 10)
        elif "Tetrahedrizing" in line:
            self._emit_progress('3D Mesh', 30)
        elif "Reconstructing mesh" in line:
            self._emit_progress('3D Mesh', 50)
        elif "3D refinement" in line:
            self._emit_progress('3D Mesh', 70)
        elif "Done meshing 3D" in line:
            self._complete_phase('3D Mesh')

        # Optimization
        elif "Optimizing mesh" in line:
            self._emit_progress('Optimization', 10)
        elif "Smoothing" in line:
            self._emit_progress('Optimization', 50)
        elif "Optimization complete" in line:
            self._complete_phase('Optimization')

        # Quality Analysis
        elif "Computing quality" in line or "Analyzing quality" in line:
            self._emit_progress('Quality', 40)
        elif "Quality analysis complete" in line:
            self._complete_phase('Quality')

        # Export
        elif "Writing" in line or "Exporting" in line:
            self._emit_progress('Export', 50)
        elif "Mesh saved" in line or "Export complete" in line:
            self._complete_phase('Export')

    def get_latest_output(self):
        """Get all new output lines"""
        lines = []
        while not self.output_queue.empty():
            lines.append(self.output_queue.get())
        return lines


def parse_quality_from_output(output: str) -> Dict:
    """Parse mesh quality metrics from subprocess output"""
    quality = {
        'total_elements': 0,
        'total_nodes': 0,
        'sicn_min': None,
        'sicn_mean': None,
        'gamma_min': None,
        'aspect_ratio_max': None,
        'aspect_ratio_mean': None,
        'max_skewness': None,
        'geometric_accuracy': None,
        'execution_time': None,
        'success': False
    }

    # Parse counts
    elem_match = re.search(r'Total elements:\s*(\d+)', output)
    if elem_match:
        quality['total_elements'] = int(elem_match.group(1))
        quality['success'] = True

    node_match = re.search(r'Total nodes:\s*(\d+)', output)
    if node_match:
        quality['total_nodes'] = int(node_match.group(1))

    # Parse SICN
    sicn_min_match = re.search(r'SICN min:\s*([-\d.]+)', output)
    if sicn_min_match:
        quality['sicn_min'] = float(sicn_min_match.group(1))

    sicn_mean_match = re.search(r'SICN mean:\s*([-\d.]+)', output)
    if sicn_mean_match:
        quality['sicn_mean'] = float(sicn_mean_match.group(1))

    # Parse gamma
    gamma_match = re.search(r'Gamma min:\s*([-\d.]+)', output)
    if gamma_match:
        quality['gamma_min'] = float(gamma_match.group(1))

    # Parse aspect ratio
    ar_max_match = re.search(r'Aspect ratio max:\s*([-\d.]+)', output)
    if ar_max_match:
        quality['aspect_ratio_max'] = float(ar_max_match.group(1))

    ar_mean_match = re.search(r'Aspect ratio mean:\s*([-\d.]+)', output)
    if ar_mean_match:
        quality['aspect_ratio_mean'] = float(ar_mean_match.group(1))

    # Parse skewness
    skew_match = re.search(r'Max skewness:\s*([-\d.]+)', output)
    if skew_match:
        quality['max_skewness'] = float(skew_match.group(1))

    # Parse geometric accuracy
    geom_match = re.search(r'Geometric accuracy:\s*([-\d.]+)', output)
    if geom_match:
        quality['geometric_accuracy'] = float(geom_match.group(1))

    # Parse execution time
    time_match = re.search(r'Total execution time:\s*([\d.]+)s', output)
    if time_match:
        quality['execution_time'] = float(time_match.group(1))

    return quality


def get_quality_grade(metrics: Dict) -> Tuple[str, str]:
    """Get quality grade matching desktop logic"""
    geom_accuracy = metrics.get('geometric_accuracy')
    sicn_min = metrics.get('sicn_min')

    # Primary: geometric accuracy (like desktop)
    if geom_accuracy is not None:
        if geom_accuracy >= 0.95:
            return "Excellent", "quality-excellent"
        elif geom_accuracy >= 0.85:
            return "Good", "quality-good"
        elif geom_accuracy >= 0.75:
            return "Fair", "quality-fair"
        elif geom_accuracy >= 0.60:
            return "Poor", "quality-poor"
        else:
            return "Critical", "quality-critical"

    # Fallback: SICN
    if sicn_min is not None:
        if sicn_min < 0.0001:
            return "Critical", "quality-critical"
        elif sicn_min < 0.1:
            return "Very Poor", "quality-poor"
        elif sicn_min < 0.2:
            return "Poor", "quality-poor"
        elif sicn_min < 0.3:
            return "Fair", "quality-fair"
        elif sicn_min < 0.5:
            return "Good", "quality-good"
        else:
            return "Excellent", "quality-excellent"

    return "Unknown", "quality-fair"


def load_cad_for_visualization(cad_file: str) -> Optional[str]:
    """
    Load CAD file (STEP/STP) and create interactive VTK visualization HTML
    """
    if not os.path.exists(cad_file):
        return None

    try:
        import pyvista as pv
        import gmsh

        # Initialize gmsh (headless mode)
        if not gmsh.is_initialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        try:
            # Load CAD file
            gmsh.model.add("cad_preview")
            gmsh.model.occ.importShapes(cad_file)
            gmsh.model.occ.synchronize()

            # Generate a coarse surface mesh for quick preview
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)
            gmsh.model.mesh.generate(2)

            # Export to temporary file
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
                temp_stl = tmp.name
            gmsh.write(temp_stl)
            gmsh.finalize()

            # Load with PyVista
            mesh = pv.read(temp_stl)
            os.unlink(temp_stl)

        except Exception as e:
            if gmsh.is_initialized():
                gmsh.finalize()
            st.error(f"CAD engine error: {e}")
            return None

        # Create plotter (HEADLESS for Vercel)
        plotter = pv.Plotter(
            window_size=[900, 700],
            notebook=False,
            off_screen=True  # CRITICAL for Vercel
        )

        plotter.add_mesh(
            mesh,
            show_edges=True,
            color='lightblue',
            opacity=0.9,
            edge_color='navy',
            line_width=1
        )

        plotter.add_axes()
        plotter.view_isometric()
        plotter.background_color = '#343a40'

        # Export to HTML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plotter.export_html(tmp_path)
            with open(tmp_path, 'r') as f:
                html = f.read()
        finally:
            plotter.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return html

    except ImportError as e:
        st.error(f"Missing dependency for 3D: {e}")
        return None
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None


def load_mesh_for_visualization(mesh_file: str) -> Optional[str]:
    """
    Load mesh and create interactive VTK visualization HTML
    """
    if not os.path.exists(mesh_file):
        return None

    try:
        import pyvista as pv

        mesh = pv.read(mesh_file)

        # Create plotter (HEADLESS for Vercel)
        plotter = pv.Plotter(
            window_size=[900, 700],
            notebook=False,
            off_screen=True  # CRITICAL for Vercel
        )

        scalars = None
        if hasattr(mesh, 'cell_data') and mesh.cell_data:
            for name in ['quality', 'sicn', 'Quality', 'SICN']:
                if name in mesh.cell_data:
                    scalars = name
                    break

        if scalars:
            plotter.add_mesh(
                mesh,
                scalars=scalars,
                show_edges=True,
                edge_color='gray',
                opacity=0.95,
                cmap='RdYlGn',
                clim=[0, 1],
                scalar_bar_args={'title': 'Quality', 'vertical': True}
            )
        else:
            plotter.add_mesh(
                mesh,
                show_edges=True,
                color='lightblue',
                opacity=0.9,
                edge_color='gray'
            )

        plotter.add_axes()
        plotter.view_isometric()
        plotter.background_color = '#343a40'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plotter.export_html(tmp_path)
            with open(tmp_path, 'r') as f:
                html = f.read()
        finally:
            plotter.close()
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return html

    except Exception as e:
        st.error(f"Mesh visualization error: {e}")
        return None


def main():
    # Header
    st.markdown('<div class="main-header">Khorium MeshGen</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced Mesh Generation * Quality-Driven * GPU Optimized</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'worker' not in st.session_state:
        st.session_state.worker = None
    if 'mesh_generated' not in st.session_state:
        st.session_state.mesh_generated = False
    if 'quality_metrics' not in st.session_state:
        st.session_state.quality_metrics = {}
    if 'output_log' not in st.session_state:
        st.session_state.output_log = []
    if 'mesh_file_path' not in st.session_state:
        st.session_state.mesh_file_path = None
    if 'generation_complete' not in st.session_state:
        st.session_state.generation_complete = False
    if 'cad_file_path' not in st.session_state:
        st.session_state.cad_file_path = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None

    # Sidebar
    with st.sidebar:
        st.markdown("### CAD File")

        uploaded_file = st.file_uploader(
            "Upload STEP/STP file",
            type=["step", "stp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            if st.session_state.uploaded_filename != uploaded_file.name:
                # Save with original extension to help gmsh identify it
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    st.session_state.cad_file_path = tmp_file.name
                    st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.mesh_file_path = None
                st.session_state.generation_complete = False

            st.success(f"{uploaded_file.name}")
            st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")

        st.markdown("---")
        st.markdown("### Quality Settings")

        quality_preset = st.selectbox(
            "Preset",
            ["Fast", "Medium", "High", "Production"],
            index=1
        )

        st.markdown('<div class="settings-group">', unsafe_allow_html=True)
        st.markdown('<div class="settings-label" style="color: #212529; font-weight: bold;">Mesh Size</div>', unsafe_allow_html=True)
        max_size_mm = st.slider(
            "Max Element Size",
            min_value=0.1,
            max_value=20.0,
            value=3.0,
            step=0.1,
            format="%.1f mm"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="settings-group">', unsafe_allow_html=True)
        st.markdown('<div class="settings-label" style="color: #212529; font-weight: bold;">Optimization</div>', unsafe_allow_html=True)
        enable_adaptive = st.checkbox("Adaptive Sizing", value=True)
        enable_refinement = st.checkbox("Iterative Refinement", value=True)
        parallel_enabled = st.checkbox("Parallel Processing", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="settings-group">', unsafe_allow_html=True)
        st.markdown('<div class="settings-label" style="color: #212529; font-weight: bold;">Material</div>', unsafe_allow_html=True)
        material = st.selectbox(
            "Material",
            ["steel", "aluminum", "titanium", "copper", "inconel"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        quality_params = {
            "quality_preset": quality_preset,
            "max_size_mm": max_size_mm,
            "adaptive_sizing_enabled": enable_adaptive,
            "iterative_refinement_enabled": enable_refinement,
            "parallel_enabled": parallel_enabled,
            "material": material
        }

        st.markdown("---")

        can_generate = uploaded_file is not None and (st.session_state.worker is None or not st.session_state.worker.is_running)

        if st.button("Generate Mesh", type="primary", disabled=not can_generate):
            if st.session_state.cad_file_path:
                st.session_state.worker = MeshWorker()
                st.session_state.worker.start(st.session_state.cad_file_path, quality_params)
                st.session_state.mesh_generated = False
                st.session_state.generation_complete = False
                st.session_state.output_log = []
                st.rerun()

    # Main content
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown("### [ART] 3D Visualization")

        visualization_html = None
        viz_type = None

        # Try to load visualization
        if st.session_state.mesh_file_path and os.path.exists(st.session_state.mesh_file_path):
            with st.spinner("Loading mesh visualization..."):
                visualization_html = load_mesh_for_visualization(st.session_state.mesh_file_path)
                viz_type = "mesh"
        elif st.session_state.cad_file_path and os.path.exists(st.session_state.cad_file_path):
            with st.spinner("Loading CAD visualization..."):
                visualization_html = load_cad_for_visualization(st.session_state.cad_file_path)
                viz_type = "cad"

        if visualization_html:
            if viz_type == "mesh":
                st.info(f"[OK] Displaying mesh: {st.session_state.uploaded_filename}")
            else:
                st.info(f"Displaying CAD preview: {st.session_state.uploaded_filename}")
            
            components.html(visualization_html, height=700, scrolling=False)
        else:
            # Fallback message if visualization fails (common on Vercel)
            if st.session_state.cad_file_path:
                st.warning("3D Visualization unavailable in this environment. (Likely missing OpenGL libraries on server)")
            else:
                st.info("🎬 Upload a STEP/STP file to begin")

        # Progress bars
        if st.session_state.worker and st.session_state.worker.is_running:
            st.markdown("### Generation Progress")
            for phase, progress in st.session_state.worker.phase_progress.items():
                st.markdown(f'<div class="progress-container"><div class="progress-label" style="color: #212529;">{phase}</div>', unsafe_allow_html=True)
                st.progress(progress / 100.0, text=f"{progress}%")
                st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(0.5)
            st.rerun()

        elif st.session_state.generation_complete:
            st.markdown("### [OK] Generation Complete")
            for phase in ['1D Mesh', '2D Mesh', '3D Mesh', 'Optimization', 'Quality', 'Export']:
                st.markdown(f'<div class="progress-container"><div class="progress-label" style="color: #212529;">{phase}</div>', unsafe_allow_html=True)
                st.progress(1.0, text="100%")
                st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 📈 Quality Report")

        if st.session_state.quality_metrics and st.session_state.quality_metrics.get('success'):
            quality = st.session_state.quality_metrics
            grade, grade_class = get_quality_grade(quality)

            st.markdown(f"""
            <div class="quality-overlay">
                <div class="quality-title" style="color: #212529; font-weight: bold; font-size: 1.3rem;">Mesh Quality Report</div>
                <div class="quality-grade">
                    Grade: <span class="{grade_class}">{grade}</span>
                </div>
                <hr style="border-color: #dee2e6;">
                <div style="color: #495057; margin: 0.4rem 0;"><span style="font-weight: 600; display: inline-block; width: 140px;">Elements:</span> {quality.get('total_elements', 0):,}</div>
                <div style="color: #495057; margin: 0.4rem 0;"><span style="font-weight: 600; display: inline-block; width: 140px;">Nodes:</span> {quality.get('total_nodes', 0):,}</div>
                <hr style="border-color: #dee2e6;">
                <div style="color: #495057; margin: 0.4rem 0;"><span style="font-weight: 600; display: inline-block; width: 140px;">SICN Min:</span> {quality.get('sicn_min', 0):.4f}</div>
                <div style="color: #495057; margin: 0.4rem 0;"><span style="font-weight: 600; display: inline-block; width: 140px;">Aspect Ratio:</span> {quality.get('aspect_ratio_mean', 0):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.mesh_file_path and os.path.exists(st.session_state.mesh_file_path):
                with open(st.session_state.mesh_file_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Mesh (.msh)",
                        data=f,
                        file_name=os.path.basename(st.session_state.mesh_file_path),
                        mime="application/octet-stream",
                        use_container_width=True
                    )
        else:
            st.info("Generate a mesh to see quality report")

        st.markdown("### 📋 Console Output")
        if st.session_state.worker:
            new_lines = st.session_state.worker.get_latest_output()
            st.session_state.output_log.extend(new_lines)
            if not st.session_state.worker.is_running and not st.session_state.generation_complete:
                st.session_state.generation_complete = True
                full_output = '\n'.join(st.session_state.output_log)
                st.session_state.quality_metrics = parse_quality_from_output(full_output)
                if st.session_state.quality_metrics.get('success'):
                    # Assume mesh is saved in current working dir by subprocess
                    mesh_files = list(Path(".").glob("*.msh"))
                    if mesh_files:
                        st.session_state.mesh_file_path = str(max(mesh_files, key=lambda p: p.stat().st_mtime))
                st.rerun()

        log_text = '\n'.join(st.session_state.output_log[-100:])
        st.text_area("Log", log_text, height=400, label_visibility="collapsed")

if __name__ == "__main__":
    main()