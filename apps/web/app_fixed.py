"""
Khorium MeshGen Web Interface - FIXED VERSION (Threading Issues Resolved)
===========================================================================

FIXES:
1. [OK] Gmsh threading issues resolved (uses subprocess)
2. [OK] Better dependency checking
3. [OK] Clearer error messages
4. [OK] Fallback to text-based info when 3D unavailable
"""

import streamlit as st
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

# Check dependencies
PLOTLY_AVAILABLE = False
MESHIO_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    pass

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    pass

# Simplified CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d6efd;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .progress-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .progress-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    .quality-overlay {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    .quality-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #212529;
        margin-bottom: 1rem;
    }
    .quality-grade {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .quality-excellent { color: #198754; }
    .quality-good { color: #0d6efd; }
    .quality-fair { color: #ffc107; }
    .quality-poor { color: #fd7e14; }
    .quality-critical { color: #dc3545; }
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    .metric-label {
        font-weight: 600;
        color: #495057;
    }
    .metric-value {
        color: #212529;
        font-family: 'Monaco', monospace;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #212529;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0d6efd;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #0d6efd;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1e7dd;
        border-left: 4px solid #198754;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .settings-group {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .settings-label {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


class MeshWorker:
    """Mesh generation worker"""

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
        self.phase_max = {}

    def start(self, cad_file: str, quality_params: dict):
        """Start mesh generation subprocess"""
        self.is_running = True
        self.phase_max = {}

        # Path to mesh worker (now in ../cli/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        worker_script = os.path.join(script_dir, "..", "cli", "mesh_worker_subprocess.py")
        worker_script = os.path.abspath(worker_script)

        if not os.path.exists(worker_script):
            # Fallback: try mesh_worker.py
            worker_script = os.path.join(script_dir, "..", "cli", "mesh_worker.py")
            worker_script = os.path.abspath(worker_script)

        if not os.path.exists(worker_script):
            self.output_queue.put(f"ERROR: mesh_worker_subprocess.py not found in apps/cli/")
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

        thread = threading.Thread(target=self._read_output, daemon=True)
        thread.start()

    def _read_output(self):
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
        if phase not in self.phase_max:
            self.phase_max[phase] = 0
        if value > self.phase_max[phase]:
            self.phase_max[phase] = value
            self.phase_progress[phase] = value

    def _complete_phase(self, phase: str):
        self.phase_max[phase] = 100
        self.phase_progress[phase] = 100

    def _parse_progress(self, line: str):
        """Parse progress from Gmsh output"""
        if "Meshing 1D" in line:
            self._emit_progress('1D Mesh', 10)
        elif "Meshing curve" in line and "order 2" not in line:
            match = re.search(r'\[\s*(\d+)%\]', line)
            if match:
                self._emit_progress('1D Mesh', 10 + int(int(match.group(1)) * 0.9))
        elif "Done meshing 1D" in line:
            self._complete_phase('1D Mesh')
        elif "Meshing 2D" in line:
            self._emit_progress('2D Mesh', 10)
        elif "Meshing surface" in line and "order 2" not in line:
            match = re.search(r'\[\s*(\d+)%\]', line)
            if match:
                self._emit_progress('2D Mesh', 10 + int(int(match.group(1)) * 0.9))
        elif "Done meshing 2D" in line:
            self._complete_phase('2D Mesh')
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
        elif "Optimizing mesh" in line:
            self._emit_progress('Optimization', 10)
        elif "Smoothing" in line:
            self._emit_progress('Optimization', 50)
        elif "Optimization complete" in line or "Done optimizing mesh" in line:
            self._complete_phase('Optimization')
        elif "Computing quality" in line or "Analyzing quality" in line:
            self._emit_progress('Quality', 40)
        elif "Quality analysis complete" in line:
            self._complete_phase('Quality')
        elif "Writing" in line or "Exporting" in line:
            self._emit_progress('Export', 50)
        elif "Mesh saved" in line or "Export complete" in line:
            self._complete_phase('Export')

    def get_latest_output(self):
        lines = []
        while not self.output_queue.empty():
            lines.append(self.output_queue.get())
        return lines


def get_mesh_info_without_viz(mesh_file: str) -> Optional[Dict]:
    """
    Get mesh statistics without full 3D visualization
    Fallback when visualization libraries not available
    """
    if not MESHIO_AVAILABLE:
        return None

    try:
        import meshio
        mesh = meshio.read(mesh_file)

        info = {
            'num_points': len(mesh.points),
            'num_cells': sum(len(cells.data) for cells in mesh.cells),
            'cell_types': [cell.type for cell in mesh.cells],
            'bounds': {
                'x': (float(mesh.points[:, 0].min()), float(mesh.points[:, 0].max())),
                'y': (float(mesh.points[:, 1].min()), float(mesh.points[:, 1].max())),
                'z': (float(mesh.points[:, 2].min()), float(mesh.points[:, 2].max()))
            }
        }
        return info
    except Exception as e:
        st.error(f"Error reading mesh file: {e}")
        return None


def load_mesh_with_plotly(mesh_file: str) -> Optional[dict]:
    """Load mesh and create Plotly 3D visualization"""
    if not (PLOTLY_AVAILABLE and MESHIO_AVAILABLE and NUMPY_AVAILABLE):
        return None

    try:
        import plotly.graph_objects as go
        import meshio
        import numpy as np

        mesh = meshio.read(mesh_file)
        vertices = mesh.points

        # Get triangular faces
        faces = None
        if "triangle" in mesh.cells_dict:
            faces = mesh.cells_dict["triangle"]
        elif "tetra" in mesh.cells_dict:
            # Extract surface from tets
            tets = mesh.cells_dict["tetra"]
            from collections import Counter
            all_faces = []
            for tet in tets:
                all_faces.extend([
                    tuple(sorted([tet[0], tet[1], tet[2]])),
                    tuple(sorted([tet[0], tet[1], tet[3]])),
                    tuple(sorted([tet[0], tet[2], tet[3]])),
                    tuple(sorted([tet[1], tet[2], tet[3]]))
                ])
            face_counts = Counter(all_faces)
            faces = np.array([list(f) for f, count in face_counts.items() if count == 1])

        if faces is None or len(faces) == 0:
            return None

        # Create Plotly mesh
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.8,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                hoverinfo='skip'
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='#ddd', backgroundcolor='#f8f9fa'),
                yaxis=dict(showgrid=True, gridcolor='#ddd', backgroundcolor='#f8f9fa'),
                zaxis=dict(showgrid=True, gridcolor='#ddd', backgroundcolor='#f8f9fa'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#ffffff',
            height=600
        )

        return fig

    except Exception as e:
        st.warning(f"3D visualization failed: {e}")
        return None


def get_cad_info_without_viz(cad_file: str) -> Optional[Dict]:
    """
    Get CAD file info without visualization
    Uses subprocess to avoid Gmsh threading issues
    """
    try:
        # Create a simple Python script to extract CAD info
        script = f"""
import gmsh
import sys

try:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("info")
    gmsh.model.occ.importShapes("{cad_file}")
    gmsh.model.occ.synchronize()

    surfaces = gmsh.model.getEntities(dim=2)
    curves = gmsh.model.getEntities(dim=1)
    volumes = gmsh.model.getEntities(dim=3)

    print(f"SURFACES: {{len(surfaces)}}")
    print(f"CURVES: {{len(curves)}}")
    print(f"VOLUMES: {{len(volumes)}}")

    bbox = gmsh.model.getBoundingBox(-1, -1)
    print(f"BBOX: {{bbox[0]:.3f}},{{bbox[1]:.3f}},{{bbox[2]:.3f}},{{bbox[3]:.3f}},{{bbox[4]:.3f}},{{bbox[5]:.3f}}")

    gmsh.finalize()
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

        # Run in subprocess to avoid threading issues
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            info = {}
            for line in result.stdout.split('\n'):
                if line.startswith('SURFACES:'):
                    info['surfaces'] = int(line.split(':')[1].strip())
                elif line.startswith('CURVES:'):
                    info['curves'] = int(line.split(':')[1].strip())
                elif line.startswith('VOLUMES:'):
                    info['volumes'] = int(line.split(':')[1].strip())
                elif line.startswith('BBOX:'):
                    bbox = [float(x) for x in line.split(':')[1].strip().split(',')]
                    info['bounds'] = {
                        'x': (bbox[0], bbox[3]),
                        'y': (bbox[1], bbox[4]),
                        'z': (bbox[2], bbox[5])
                    }
            return info
        else:
            return None

    except Exception as e:
        return None


def parse_quality_from_output(output: str) -> Dict:
    """Parse mesh quality metrics"""
    quality = {
        'total_elements': 0,
        'total_nodes': 0,
        'sicn_min': None,
        'sicn_mean': None,
        'aspect_ratio_max': None,
        'aspect_ratio_mean': None,
        'success': False
    }

    elem_match = re.search(r'Total elements:\s*(\d+)', output)
    if elem_match:
        quality['total_elements'] = int(elem_match.group(1))
        quality['success'] = True

    node_match = re.search(r'Total nodes:\s*(\d+)', output)
    if node_match:
        quality['total_nodes'] = int(node_match.group(1))

    sicn_min_match = re.search(r'SICN min:\s*([-\d.]+)', output)
    if sicn_min_match:
        quality['sicn_min'] = float(sicn_min_match.group(1))

    sicn_mean_match = re.search(r'SICN mean:\s*([-\d.]+)', output)
    if sicn_mean_match:
        quality['sicn_mean'] = float(sicn_mean_match.group(1))

    ar_max_match = re.search(r'Aspect ratio max:\s*([-\d.]+)', output)
    if ar_max_match:
        quality['aspect_ratio_max'] = float(ar_max_match.group(1))

    ar_mean_match = re.search(r'Aspect ratio mean:\s*([-\d.]+)', output)
    if ar_mean_match:
        quality['aspect_ratio_mean'] = float(ar_mean_match.group(1))

    return quality


def get_quality_grade(metrics: Dict) -> Tuple[str, str]:
    """Get quality grade"""
    sicn_min = metrics.get('sicn_min')

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


def main():
    # Header
    st.markdown('<div class="main-header">🔷 Khorium MeshGen</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced 3D Mesh Generation * Quality-Driven * Web-Optimized</div>', unsafe_allow_html=True)

    # Show dependency status
    with st.expander("Dependency Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if PLOTLY_AVAILABLE:
                st.success("[OK] Plotly")
            else:
                st.error("Plotly (pip install plotly)")
        with col2:
            if MESHIO_AVAILABLE:
                st.success("[OK] Meshio")
            else:
                st.error("Meshio (pip install meshio)")
        with col3:
            if NUMPY_AVAILABLE:
                st.success("[OK] NumPy")
            else:
                st.error("NumPy (pip install numpy)")

        if not (PLOTLY_AVAILABLE and MESHIO_AVAILABLE and NUMPY_AVAILABLE):
            st.warning("Install missing dependencies for 3D visualization:\n```bash\npip install plotly meshio numpy\n```")

    # Initialize session state
    if 'worker' not in st.session_state:
        st.session_state.worker = None
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
        st.markdown('<div class="settings-label">Mesh Size</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="settings-label">Optimization</div>', unsafe_allow_html=True)
        enable_adaptive = st.checkbox("Adaptive Sizing", value=True)
        enable_refinement = st.checkbox("Iterative Refinement", value=True)
        parallel_enabled = st.checkbox("Parallel Processing", value=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="settings-group">', unsafe_allow_html=True)
        st.markdown('<div class="settings-label">Material</div>', unsafe_allow_html=True)
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
                st.session_state.generation_complete = False
                st.session_state.output_log = []
                st.rerun()

    # Main content
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown('<div class="section-header">[ART] Visualization</div>', unsafe_allow_html=True)

        # Try mesh visualization first
        if st.session_state.mesh_file_path and os.path.exists(st.session_state.mesh_file_path):
            st.success(f"[OK] Mesh generated: {st.session_state.uploaded_filename}")

            if PLOTLY_AVAILABLE and MESHIO_AVAILABLE and NUMPY_AVAILABLE:
                with st.spinner("Loading mesh visualization..."):
                    fig = load_mesh_with_plotly(st.session_state.mesh_file_path)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: show mesh info
                    info = get_mesh_info_without_viz(st.session_state.mesh_file_path)
                    if info:
                        st.info(f"Mesh Info:\n- Points: {info['num_points']:,}\n- Cells: {info['num_cells']:,}\n- Types: {', '.join(info['cell_types'])}")
            else:
                # Show mesh info without viz
                info = get_mesh_info_without_viz(st.session_state.mesh_file_path)
                if info:
                    st.info(f"Mesh Info:\n- Points: {info['num_points']:,}\n- Cells: {info['num_cells']:,}\n- Types: {', '.join(info['cell_types'])}")
                st.warning("Install dependencies for 3D visualization:\n```bash\npip install plotly meshio numpy\n```")

        # Or show CAD info
        elif st.session_state.cad_file_path and os.path.exists(st.session_state.cad_file_path):
            st.info(f"CAD File: {st.session_state.uploaded_filename}")

            # Get CAD info (no visualization to avoid threading issues)
            info = get_cad_info_without_viz(st.session_state.cad_file_path)
            if info:
                st.markdown(f"""
                <div class="success-box">
                    <strong>📐 CAD Geometry Info:</strong><br>
                    * Volumes: {info.get('volumes', 0)}<br>
                    * Surfaces: {info.get('surfaces', 0)}<br>
                    * Curves: {info.get('curves', 0)}<br>
                    <small>Click "Generate Mesh" to create tetrahedral mesh</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>CAD File Uploaded</strong><br>
                    Click "Generate Mesh" to begin meshing<br>
                    <small>(3D preview unavailable - will show after meshing)</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">🎬 Upload a STEP/STP file to begin</div>', unsafe_allow_html=True)

        # Progress bars
        if st.session_state.worker and st.session_state.worker.is_running:
            st.markdown('<div class="section-header">Generation Progress</div>', unsafe_allow_html=True)
            for phase, progress in st.session_state.worker.phase_progress.items():
                st.markdown(f'<div class="progress-container"><div class="progress-label">{phase}</div>', unsafe_allow_html=True)
                st.progress(progress / 100.0, text=f"{progress}%")
                st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(0.5)
            st.rerun()

        elif st.session_state.generation_complete:
            st.markdown('<div class="section-header">[OK] Generation Complete</div>', unsafe_allow_html=True)
            for phase in ['1D Mesh', '2D Mesh', '3D Mesh', 'Optimization', 'Quality', 'Export']:
                st.markdown(f'<div class="progress-container"><div class="progress-label">{phase}</div>', unsafe_allow_html=True)
                st.progress(1.0, text="100%")
                st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">📈 Quality Report</div>', unsafe_allow_html=True)

        if st.session_state.quality_metrics and st.session_state.quality_metrics.get('success'):
            quality = st.session_state.quality_metrics
            grade, grade_class = get_quality_grade(quality)

            st.markdown(f"""
            <div class="quality-overlay">
                <div class="quality-title">Mesh Quality Report</div>
                <div class="quality-grade">
                    Grade: <span class="{grade_class}">{grade}</span>
                </div>
                <hr style="border-color: #dee2e6; margin: 1rem 0;">
                <div class="metric-row">
                    <span class="metric-label">Elements:</span>
                    <span class="metric-value">{quality.get('total_elements', 0):,}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Nodes:</span>
                    <span class="metric-value">{quality.get('total_nodes', 0):,}</span>
                </div>
                <hr style="border-color: #dee2e6; margin: 1rem 0;">
                <div class="metric-row">
                    <span class="metric-label">SICN Min:</span>
                    <span class="metric-value">{quality.get('sicn_min', 0):.4f}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Aspect Ratio:</span>
                    <span class="metric-value">{quality.get('aspect_ratio_mean', 0):.2f}</span>
                </div>
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
            st.markdown('<div class="info-box">Generate a mesh to see quality report</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">📋 Console Output</div>', unsafe_allow_html=True)
        if st.session_state.worker:
            new_lines = st.session_state.worker.get_latest_output()
            st.session_state.output_log.extend(new_lines)
            if not st.session_state.worker.is_running and not st.session_state.generation_complete:
                st.session_state.generation_complete = True
                full_output = '\n'.join(st.session_state.output_log)
                st.session_state.quality_metrics = parse_quality_from_output(full_output)
                if st.session_state.quality_metrics.get('success'):
                    mesh_files = list(Path(".").glob("*.msh"))
                    if mesh_files:
                        st.session_state.mesh_file_path = str(max(mesh_files, key=lambda p: p.stat().st_mtime))
                st.rerun()

        log_text = '\n'.join(st.session_state.output_log[-100:])
        st.text_area("Log", log_text, height=400, label_visibility="collapsed")

if __name__ == "__main__":
    main()
