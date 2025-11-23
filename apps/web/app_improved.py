"""
Khorium MeshGen Web Interface - IMPROVED VERSION
=================================================

FIXED ISSUES:
1. [OK] Web-compatible 3D visualization using Plotly (works on Vercel!)
2. [OK] Clean theme with readable text (no light-on-light conflicts)
3. [OK] Proper fallbacks for missing dependencies
4. [OK] Interactive mesh/CAD viewer in browser
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

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Khorium MeshGen",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS - works with config.toml theme
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d6efd;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .subtitle {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
        font-size: 1rem;
    }

    /* Progress container */
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

    /* Quality overlay card */
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

    /* Quality grade colors */
    .quality-excellent { color: #198754; }
    .quality-good { color: #0d6efd; }
    .quality-fair { color: #ffc107; }
    .quality-poor { color: #fd7e14; }
    .quality-critical { color: #dc3545; }

    /* Metric row */
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
        font-family: 'Monaco', 'Courier New', monospace;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #212529;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0d6efd;
    }

    /* Info boxes */
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

    /* Console output */
    .stTextArea textarea {
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 0.85rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.75rem;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #0d6efd;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f8f9fa;
    }

    /* Settings groups */
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

        worker_script = "mesh_worker_subprocess.py"
        if not os.path.exists(worker_script):
            self.output_queue.put(f"ERROR: {worker_script} not found")
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
        """Read subprocess output"""
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
        """Emit progress only if higher"""
        if phase not in self.phase_max:
            self.phase_max[phase] = 0
        if value > self.phase_max[phase]:
            self.phase_max[phase] = value
            self.phase_progress[phase] = value

    def _complete_phase(self, phase: str):
        """Mark phase complete"""
        self.phase_max[phase] = 100
        self.phase_progress[phase] = 100

    def _parse_progress(self, line: str):
        """Parse progress from Gmsh output"""
        # 1D
        if "Meshing 1D" in line:
            self._emit_progress('1D Mesh', 10)
        elif "Meshing curve" in line and "order 2" not in line:
            match = re.search(r'\[\s*(\d+)%\]', line)
            if match:
                self._emit_progress('1D Mesh', 10 + int(int(match.group(1)) * 0.9))
        elif "Done meshing 1D" in line:
            self._complete_phase('1D Mesh')

        # 2D
        elif "Meshing 2D" in line:
            self._emit_progress('2D Mesh', 10)
        elif "Meshing surface" in line and "order 2" not in line:
            match = re.search(r'\[\s*(\d+)%\]', line)
            if match:
                self._emit_progress('2D Mesh', 10 + int(int(match.group(1)) * 0.9))
        elif "Done meshing 2D" in line:
            self._complete_phase('2D Mesh')

        # 3D
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
        elif "Optimization complete" in line or "Done optimizing mesh" in line:
            self._complete_phase('Optimization')

        # Quality
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
        """Get new output lines"""
        lines = []
        while not self.output_queue.empty():
            lines.append(self.output_queue.get())
        return lines


def load_mesh_with_plotly(mesh_file: str) -> Optional[dict]:
    """
    Load mesh and create Plotly 3D visualization
    WORKS ON VERCEL - No OpenGL required!
    """
    try:
        import plotly.graph_objects as go
        import numpy as np

        # Try multiple mesh libraries
        mesh_data = None

        # Try meshio first (lightweight, pure Python)
        try:
            import meshio
            mesh = meshio.read(mesh_file)

            # Extract vertices and faces
            vertices = mesh.points

            # Get triangular faces (convert if needed)
            if "triangle" in mesh.cells_dict:
                faces = mesh.cells_dict["triangle"]
            elif "tetra" in mesh.cells_dict:
                # Extract surface from tets
                tets = mesh.cells_dict["tetra"]
                # Extract unique surface triangles
                from collections import Counter
                all_faces = []
                for tet in tets:
                    # 4 faces per tet
                    all_faces.extend([
                        tuple(sorted([tet[0], tet[1], tet[2]])),
                        tuple(sorted([tet[0], tet[1], tet[3]])),
                        tuple(sorted([tet[0], tet[2], tet[3]])),
                        tuple(sorted([tet[1], tet[2], tet[3]]))
                    ])
                # Surface faces appear once, interior faces twice
                face_counts = Counter(all_faces)
                faces = np.array([list(f) for f, count in face_counts.items() if count == 1])
            else:
                raise ValueError("No triangular or tetrahedral cells found")

            mesh_data = (vertices, faces)

        except ImportError:
            # Fallback: try pyvista
            try:
                import pyvista as pv
                mesh = pv.read(mesh_file)

                # Extract surface
                if hasattr(mesh, 'extract_surface'):
                    surface = mesh.extract_surface()
                else:
                    surface = mesh

                vertices = surface.points
                faces = surface.faces.reshape(-1, 4)[:, 1:]  # Skip first column (vertex count)
                mesh_data = (vertices, faces)
            except:
                pass

        if mesh_data is None:
            return None

        vertices, faces = mesh_data

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
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.2,
                    roughness=0.5
                ),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo='skip'
            )
        ])

        # Layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='#ddd', showbackground=True, backgroundcolor='#f8f9fa'),
                yaxis=dict(showgrid=True, gridcolor='#ddd', showbackground=True, backgroundcolor='#f8f9fa'),
                zaxis=dict(showgrid=True, gridcolor='#ddd', showbackground=True, backgroundcolor='#f8f9fa'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#ffffff',
            height=600
        )

        return fig

    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None


def load_cad_with_plotly(cad_file: str) -> Optional[dict]:
    """
    Load CAD and create Plotly 3D visualization
    WORKS ON VERCEL - No OpenGL required!
    """
    try:
        import plotly.graph_objects as go
        import gmsh

        # Initialize Gmsh
        if not gmsh.is_initialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        try:
            # Load CAD
            gmsh.model.add("cad_preview")
            gmsh.model.occ.importShapes(cad_file)
            gmsh.model.occ.synchronize()

            # Generate coarse surface mesh for preview
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 10.0)
            gmsh.model.mesh.generate(2)

            # Get mesh data
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            vertices = node_coords.reshape(-1, 3)

            # Get triangular faces
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
            faces = []
            for elem_type, tags, nodes in zip(elem_types, elem_tags, elem_node_tags):
                if elem_type == 2:  # Triangle
                    nodes_array = nodes.reshape(-1, 3)
                    # Convert from 1-based to 0-based indexing
                    faces.append(nodes_array - 1)

            if faces:
                faces = np.vstack(faces)
            else:
                gmsh.finalize()
                return None

            gmsh.finalize()

        except Exception as e:
            if gmsh.is_initialized():
                gmsh.finalize()
            st.error(f"CAD loading error: {e}")
            return None

        # Create Plotly figure
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightblue',
                opacity=0.9,
                flatshading=True,
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.8,
                    specular=0.2,
                    roughness=0.5
                ),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo='skip'
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='#ddd', showbackground=True, backgroundcolor='#f8f9fa'),
                yaxis=dict(showgrid=True, gridcolor='#ddd', showbackground=True, backgroundcolor='#f8f9fa'),
                zaxis=dict(showgrid=True, gridcolor='#ddd', showbackground=True, backgroundcolor='#f8f9fa'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#ffffff',
            height=600
        )

        return fig

    except Exception as e:
        st.error(f"CAD visualization error: {e}")
        return None


def parse_quality_from_output(output: str) -> Dict:
    """Parse mesh quality metrics from output"""
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
                st.session_state.mesh_generated = False
                st.session_state.generation_complete = False
                st.session_state.output_log = []
                st.rerun()

    # Main content
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.markdown('<div class="section-header">[ART] 3D Visualization</div>', unsafe_allow_html=True)

        fig = None
        viz_type = None

        # Try to load visualization
        if st.session_state.mesh_file_path and os.path.exists(st.session_state.mesh_file_path):
            with st.spinner("Loading mesh visualization..."):
                fig = load_mesh_with_plotly(st.session_state.mesh_file_path)
                viz_type = "mesh"
        elif st.session_state.cad_file_path and os.path.exists(st.session_state.cad_file_path):
            with st.spinner("Loading CAD preview..."):
                fig = load_cad_with_plotly(st.session_state.cad_file_path)
                viz_type = "cad"

        if fig:
            if viz_type == "mesh":
                st.success(f"[OK] Displaying mesh: {st.session_state.uploaded_filename}")
            else:
                st.info(f"CAD Preview: {st.session_state.uploaded_filename}")

            st.plotly_chart(fig, use_container_width=True)
        else:
            if st.session_state.cad_file_path:
                st.markdown('<div class="warning-box">3D visualization unavailable. Install dependencies: <code>pip install plotly meshio numpy</code></div>', unsafe_allow_html=True)
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
