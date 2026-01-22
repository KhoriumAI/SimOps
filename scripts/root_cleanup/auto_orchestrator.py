"""
SimOps Auto-Orchestrator
========================
Real-time simulation queue with live processing and visualization.
Perfect for demonstrations and batch processing.
"""

import streamlit as st
import os
import time
import subprocess
import glob
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

# Configuration
st.set_page_config(layout="wide", page_title="SimOps Auto-Orchestrator", page_icon="🚀")

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# State management
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'completed_jobs' not in st.session_state:
    st.session_state.completed_jobs = []
if 'current_job' not in st.session_state:
    st.session_state.current_job = None

# --- Styling ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .success { color: #00ff88; font-weight: bold; }
    .processing { color: #00d4ff; font-weight: bold; }
    .queued { color: #ffaa00; font-weight: bold; }
    h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
with st.sidebar:
    st.title("🚀 SimOps Commander")
    
    physics_type = st.selectbox(
        "Physics Template",
        ["Structural", "Thermal", "CFD"],
        help="Select the simulation type for uploaded files"
    )
    
    # Material selection
    if physics_type == "Structural":
        material = st.selectbox("Material", ["Aluminum", "Steel", "ABS_Plastic", "Titanium"])
        gravity = st.slider("Gravity Load (g)", 0.0, 10.0, 1.0, 0.1)
    elif physics_type == "Thermal":
        material = st.selectbox("Material", ["Aluminum", "Copper", "Steel", "ABS_Plastic"])
        heat_load = st.number_input("Heat Load (W)", 0.0, 500.0, 50.0)
    else:  # CFD
        material = "Air"
        velocity = st.slider("Inlet Velocity (m/s)", 0.5, 25.0, 5.0, 0.5)
    
    st.divider()
    
    # File upload
    uploaded_files = st.file_uploader(
        "📁 Batch Upload CAD Files",
        accept_multiple_files=True,
        type=['step', 'stp', 'iges', 'igs'],
        help="Upload multiple CAD files for batch processing"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save CAD file
            cad_path = INPUT_DIR / uploaded_file.name
            with open(cad_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Create sidecar config
            config = {
                "version": "1.0",
                "job_name": Path(uploaded_file.name).stem,
                "physics": {
                    "simulation_type": physics_type.lower(),
                    "material": material,
                    "ccx_path": str(Path.cwd() / "ccx_wsl.bat")
                },
                "meshing": {
                    "second_order": False,
                    "mesh_size_multiplier": 1.0
                }
            }
            
            # Add physics-specific parameters
            if physics_type == "Structural":
                config["physics"]["gravity_load_g"] = gravity
            elif physics_type == "Thermal":
                config["physics"]["heat_load_watts"] = heat_load
            else:  # CFD
                config["physics"]["inlet_velocity"] = velocity
            
            # Save config
            config_path = INPUT_DIR / f"{Path(uploaded_file.name).stem}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        st.success(f"✅ Queued {len(uploaded_files)} simulations!")
        time.sleep(1)
        st.rerun()
    
    st.divider()
    
    # Auto-process toggle
    auto_process = st.toggle("⚡ Auto-Process Queue", value=False)
    
    if st.button("🔄 Process Queue Now", type="primary", use_container_width=True):
        st.session_state.processing = True
        st.rerun()

# --- Main UI ---
st.title("🎯 Real-Time Simulation Orchestrator")

# Statistics
col1, col2, col3, col4 = st.columns(4)

pending_files = list(INPUT_DIR.glob("*.step")) + list(INPUT_DIR.glob("*.stp"))
pending_count = len([f for f in pending_files if not f.name.startswith("USED_")])
completed_count = len(st.session_state.completed_jobs)
processing_count = 1 if st.session_state.current_job else 0

with col1:
    st.metric("⏳ Queued", pending_count, delta=None)
with col2:
    st.metric("⚙️ Processing", processing_count, delta=None)
with col3:
    st.metric("✅ Completed", completed_count, delta=None)
with col4:
    success_rate = (completed_count / max(completed_count + pending_count, 1)) * 100
    st.metric("📊 Success Rate", f"{success_rate:.0f}%", delta=None)

st.divider()

# Processing loop
if st.session_state.processing or auto_process:
    # Get next file to process
    pending = [f for f in pending_files if not f.name.startswith("USED_")]
    
    if pending and not st.session_state.current_job:
        next_file = pending[0]
        st.session_state.current_job = {
            'name': next_file.stem,
            'path': str(next_file),
            'start_time': datetime.now()
        }
    
    # Process current job
    if st.session_state.current_job:
        job = st.session_state.current_job
        
        st.subheader(f"⚙️ Processing: {job['name']}")
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run simulation
            try:
                status_text.text("🔧 Initializing simulation...")
                progress_bar.progress(10)
                time.sleep(0.5)
                
                status_text.text("🔨 Meshing geometry...")
                progress_bar.progress(30)
                
                # Actually run the simulation
                cmd = ["python", "simops_worker.py", job['path'], "-o", str(OUTPUT_DIR)]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 min timeout
                )
                
                progress_bar.progress(70)
                status_text.text("🧮 Solving physics...")
                time.sleep(0.5)
                
                progress_bar.progress(90)
                status_text.text("📊 Generating results...")
                time.sleep(0.5)
                
                progress_bar.progress(100)
                
                # Check for results
                job_results = {
                    'name': job['name'],
                    'completed_at': datetime.now().isoformat(),
                    'duration': (datetime.now() - job['start_time']).total_seconds(),
                    'success': result.returncode == 0,
                    'output': result.stdout if result.returncode == 0 else result.stderr
                }
                
                # Find generated files
                result_json = OUTPUT_DIR / f"{job['name']}_result.json"
                if result_json.exists():
                    with open(result_json) as f:
                        job_results['metrics'] = json.load(f)
                
                vtk_file = OUTPUT_DIR / f"{job['name']}_structural.vtk"
                if not vtk_file.exists():
                    vtk_file = OUTPUT_DIR / f"{job['name']}_thermal.vtk"
                if vtk_file.exists():
                    job_results['vtk'] = str(vtk_file)
                
                png_file = list(OUTPUT_DIR.glob(f"{job['name']}*.png"))
                if png_file:
                    job_results['visualization'] = str(png_file[0])
                
                if job_results['success']:
                    status_text.markdown(f"<span class='success'>✅ Simulation Complete!</span>", unsafe_allow_html=True)
                    st.balloons()
                else:
                    status_text.markdown(f"<span class='error'>❌ Simulation Failed</span>", unsafe_allow_html=True)
                
                st.session_state.completed_jobs.append(job_results)
                
            except subprocess.TimeoutExpired:
                st.error("⏱️ Simulation timeout (5 min)")
                job_results = {
                    'name': job['name'],
                    'success': False,
                    'error': 'Timeout'
                }
                st.session_state.completed_jobs.append(job_results)
            
            except Exception as e:
                st.error(f"💥 Error: {e}")
                job_results = {
                    'name': job['name'],
                    'success': False,
                    'error': str(e)
                }
                st.session_state.completed_jobs.append(job_results)
            
            finally:
                st.session_state.current_job = None
                time.sleep(1)
                st.rerun()
    
    elif not pending:
        st.session_state.processing = False
        st.info("✅ Queue empty - all simulations complete!")

# --- Results Gallery ---
if st.session_state.completed_jobs:
    st.divider()
    st.subheader("📊 Completed Simulations")
    
    # Create results grid
    cols_per_row = 3
    jobs = st.session_state.completed_jobs[-12:]  # Show last 12
    
    for idx in range(0, len(jobs), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, job in enumerate(jobs[idx:idx + cols_per_row]):
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**{job['name']}**")
                    
                    # Show visualization if available
                    if 'visualization' in job and Path(job['visualization']).exists():
                        st.image(job['visualization'], use_container_width=True)
                    else:
                        st.info("📊 Results available in output/")
                    
                    # Show metrics
                    if 'metrics' in job:
                        metrics = job['metrics']
                        if 'solve_time_s' in metrics:
                            st.caption(f"⏱️ {metrics['solve_time_s']:.1f}s")
                        if 'max_temp_K' in metrics:
                            st.caption(f"🌡️ Max: {metrics['max_temp_K']:.1f}K")
                        if 'max_stress' in metrics:
                            st.caption(f"💪 Stress: {metrics['max_stress']/1e6:.1f} MPa")
                    
                    # Status badge
                    if job['success']:
                        st.success("✅ Success", icon="✅")
                    else:
                        st.error("❌ Failed", icon="❌")

# Auto-refresh when processing
if st.session_state.processing or auto_process:
    time.sleep(2)
    st.rerun()
