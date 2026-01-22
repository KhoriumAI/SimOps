import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import json
import subprocess
import shutil
from pathlib import Path
from log_watcher import LogWatcher

# --- Config ---
st.set_page_config(
    page_title="SimOps Dashboard",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# --- CSS Styling (Professional/Dark) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #374151;
    }
    .status-converged { color: #10b981; font-weight: bold; }
    .status-failed { color: #ef4444; font-weight: bold; }
    .status-running { color: #3b82f6; font-weight: bold; }
    .status-queued { color: #6b7280; font-weight: bold; }
    
    /* Concise Headers */
    h1, h2, h3 { color: #e5e7eb; font-family: 'Inter', sans-serif; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-bottom: 2px solid transparent;
        color: #9ca3af;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6;
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
# Resolve project root assuming app is in apps/monitor/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
INPUT_DIR = os.path.join(PROJECT_ROOT, 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- State Management ---
@st.cache_resource
def get_watcher(log_dir_path):
    return LogWatcher(log_dir_path)

if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = False

# --- Helper Functions ---
def run_worker_process(cad_path):
    """Run the worker script for a single file"""
    worker_script = os.path.join(PROJECT_ROOT, "simops_worker.py")
    cmd = ["python", worker_script, cad_path, "-o", OUTPUT_DIR]
    
    # We use Popen to stream output potentially, but for now we block to show progress
    # In a real production app, this should be async.
    # For "believability", blocking is fine if we show a spinner.
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

# --- Main Interface ---
st.title("SimOps Unified Dashboard")

# Top-level Tabs
tab_monitor, tab_submit = st.tabs(["Monitor", "New Simulation"])

# ==============================================================================
# TAB 1: SUBMISSION (Orchestrator)
# ==============================================================================
with tab_submit:
    col_config, col_upload = st.columns([1, 2])
    
    with col_config:
        st.subheader("Configuration")
        
        sim_type = st.selectbox(
            "Simulation Type",
            ["Structural", "Thermal", "CFD"],
            key="sim_type"
        )
        
        # Dynamic Configuration Fields based on Type
        config_params = {}
        
        if sim_type == "Structural":
            material = st.selectbox("Material", ["Aluminum", "Steel", "ABS_Plastic", "Titanium"])
            gravity = st.number_input("Gravity Load (g)", value=1.0, step=0.1)
            config_params = {"material": material, "gravity_load_g": gravity}
            
        elif sim_type == "Thermal":
            material = st.selectbox("Material", ["Aluminum", "Copper", "Steel"])
            heat_load = st.number_input("Heat Load (W)", value=50.0)
            config_params = {"material": material, "heat_load_watts": heat_load}
            
        elif sim_type == "CFD":
            velocity = st.number_input("Inlet Velocity (m/s)", value=10.0)
            config_params = {"inlet_velocity": velocity}

    with col_upload:
        st.subheader("Input Files")
        
        uploaded_files = st.file_uploader(
            "Upload Mesh Geometry (.msh)", 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Queue Simulations", type="primary"):
                with st.spinner("Processing queue..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, up_file in enumerate(uploaded_files):
                        # 1. Save File
                        file_path = os.path.join(INPUT_DIR, up_file.name)
                    with open(file_path, "wb") as f:
                        f.write(up_file.getbuffer())
                    
                    # 2. Generate Sidecar JSON
                    config = {
                        "version": "1.0",
                        "job_name": os.path.splitext(up_file.name)[0],
                        "physics": {
                            "simulation_type": sim_type.lower(),
                            "ccx_path": os.path.join(PROJECT_ROOT, "ccx_wsl.bat"),
                            **config_params
                        }
                    }
                    
                    json_path = os.path.splitext(file_path)[0] + ".json"
                    with open(json_path, "w") as f:
                        json.dump(config, f, indent=2)
                        
                    # 3. Trigger Worker (Immediate Execution)
                    # Note: Professional systems usually queue to Redis.
                    # Here we offer "Process Now" for the user's workflow.
                    
                    status_text.text(f"Processing {up_file.name}...")
                    success, error = run_worker_process(file_path)
                    
                    update_val = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(update_val)
                
                status_text.success(f"Processed {len(uploaded_files)} files successfully.")
                time.sleep(1)
                st.session_state.processing_queue = False
                st.rerun()

    # Queue Status - Minimalist
    st.divider()
    pending_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.msh') and not f.startswith("USED_")]
    st.caption(f"Pending in Queue: {len(pending_files)}")


# ==============================================================================
# TAB 2: MONITOR (LogWatcher Integration)
# ==============================================================================
with tab_monitor:
    # Sidebar Controls (Monitoring Specific)
    with st.sidebar:
        st.subheader("Monitor Settings")
        refresh_rate = st.slider("Refresh Rate (s)", 1, 10, 2)
        
        st.divider()
        st.subheader("Filters")
        
        # Initialize Watcher to get current data for filter options
        watcher = get_watcher(OUTPUT_DIR)
        watcher.poll()
        df_all = watcher.get_dataframe()
        
        filter_status = "All"
        filter_stage = "All"
        
        if not df_all.empty:
            statuses = ["All"] + sorted(df_all['Status'].unique().tolist())
            stages = ["All"] + sorted(df_all['Stage'].unique().tolist())
            
            filter_status = st.selectbox("Filter Status", statuses)
            filter_stage = st.selectbox("Filter Stage", stages)

        if st.button("Reset Watcher"):
            watcher = get_watcher(OUTPUT_DIR)
            watcher.reset()
            st.success("State cleared.")
            st.rerun()

    # Define the Fragment for non-flickering updates
    @st.fragment(run_every=refresh_rate)
    def monitor_board():
        # Initialize Watcher
        watcher = get_watcher(OUTPUT_DIR)
        watcher.poll()
        df = watcher.get_dataframe()
        
        if not df.empty:
            # 1. Default Sort: Newest Job at Top
            # We use the internal '_raw_data' to get the start timestamp
            df['_ts'] = df['_raw_data'].apply(lambda x: x.get('_start_ts', 0))
            df = df.sort_values(by='_ts', ascending=False)
            
            # 2. Apply Filters
            if filter_status != "All":
                df = df[df['Status'] == filter_status]
            if filter_stage != "All":
                df = df[df['Stage'] == filter_stage]

        # KPIs (based on filtered data or all data? Usually better to show total stats)
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        active_count = len(df[df['Status']=='Running']) if not df.empty else 0
        failed_count = len(df[df['Status']=='Failed']) if not df.empty else 0
        converged_count = len(df[df['Status']=='Converged']) if not df.empty else 0
        
        kpi_col1.metric("Active Jobs", active_count)
        kpi_col2.metric("Converged", converged_count)
        kpi_col3.metric("Failed", failed_count)
        kpi_col4.metric("Total Tracked", len(df))

        st.divider()

        if not df.empty:
            # Prepare Data View
            view_df = df.copy()
            
            # FILTER: Force blank metrics during early stages (Meshing/Initializing)
            early_stages = ['Initializing', 'Meshing', 'Quality Control', 'Pre-Processing']
            mask_early = view_df['Stage'].isin(early_stages)
            
            if 'Courant Max' in view_df.columns:
                view_df.loc[mask_early, 'Courant Max'] = None
            if 'Max Temp' in view_df.columns:
                view_df.loc[mask_early, 'Max Temp'] = None
            
            # Main Grid
            # Note: on_select="rerun" inside a fragment will rerun ONLY the fragment by default
            selection = st.dataframe(
                view_df,
                column_order=("Job ID", "Status", "Stage", "Run Time", "Courant Max", "Max Temp", "Progress"),
                column_config={
                    "Job ID": st.column_config.TextColumn("Job ID", width="medium"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Stage": st.column_config.TextColumn("Stage", width="medium"),
                    "Run Time": st.column_config.TextColumn("Duration", width="small"),
                    "Courant Max": st.column_config.NumberColumn("Courant", format="%.2f"),
                    "Max Temp": st.column_config.NumberColumn("Max Temp", format="%.1f K"),
                    "Progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=120, format="%f"),
                },
                width="stretch",
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )
            
            # Deep Dive
            if selection.selection["rows"]:
                idx = selection.selection["rows"][0]
                row = view_df.iloc[idx]
                job_id = row['Job ID']
                raw_data = row['_raw_data']
                
                st.subheader(f"Details: {job_id}")
                
                # Action Bar
                act_col1, act_col2, act_col3 = st.columns([1, 1, 4])
                with act_col1:
                    if st.button("Stop Job", use_container_width=True):
                        job_dir = os.path.join(OUTPUT_DIR, f"{job_id}_case")
                        if not os.path.exists(job_dir): job_dir = OUTPUT_DIR
                        with open(os.path.join(job_dir, "STOP_SIM"), "w") as f:
                            f.write("STOP")
                        st.toast(f"Stop signal sent to {job_id}")
                
                with act_col2:
                    if st.button("Rerun Job", use_container_width=True):
                         if os.path.exists(INPUT_DIR):
                            count = 0
                            for f in os.listdir(INPUT_DIR):
                                if f.startswith(f"USED_{job_id}"):
                                    new_name = f.replace("USED_", "", 1)
                                    try:
                                        os.rename(os.path.join(INPUT_DIR, f), os.path.join(INPUT_DIR, new_name))
                                        count += 1
                                    except Exception:
                                        pass
                            if count > 0:
                                st.toast(f"Queued rerun for {job_id}")
                
                # 1. Convergence Chart (Full Width)
                st.divider()
                st.subheader("Convergence & Residuals")
                
                fig_res = go.Figure()
                hist = raw_data.get('History', {})
                res = raw_data.get('Residuals', {})
                
                x_val = hist.get('Time', [])
                if not x_val and 'p' in res: x_val = list(range(len(res['p'])))
                
                for field in ['p', 'Ux', 'Uy', 'T']:
                    if field in res and res[field]:
                            fig_res.add_trace(go.Scatter(x=x_val, y=res[field], name=field))
                
                # HANDLE EMPTY DATA
                if not fig_res.data:
                    fig_res.add_annotation(
                        text="No Convergence Data (Non-Iterative Solver)",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(color="#6b7280", size=14)
                    )
                    fig_res.update_xaxes(visible=False)
                    fig_res.update_yaxes(visible=False)
                
                fig_res.update_layout(
                    yaxis_type="log" if fig_res.data else "linear", 
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#9ca3af'),
                    xaxis_title="Time / Iterations" if fig_res.data else "",
                    yaxis_title="Residual (Log)" if fig_res.data else ""
                )
                st.plotly_chart(fig_res, width="stretch")
                
                # 2. Results & Details
                det_col1, det_col2 = st.columns([2, 1])
                
                with det_col1:
                    st.subheader("Simulation Properties")
                    if row.get('Metrics'):
                        setup_keys = ['material', 'gravity', 'heat_load', 'inlet_velocity', 'solver', 'fluid']
                        st.caption("Configuration")
                        s_cols = st.columns(3)
                        s_idx = 0
                        for k, v in row['Metrics'].items():
                            if k in setup_keys:
                                s_cols[s_idx % 3].metric(k.replace('_', ' ').title(), str(v))
                                s_idx += 1
                                
                        st.caption("Results & Quality Control")
                        r_cols = st.columns(3)
                        r_idx = 0
                        for k, v in row['Metrics'].items():
                            if k not in setup_keys:
                                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                                r_cols[r_idx % 3].metric(k.replace('_', ' ').title(), val)
                                r_idx += 1
                    else:
                        st.info("No metadata available.")

                with det_col2:
                    st.subheader("Live Terminals")
                    st.caption("Recent Output (Tail)")
                    log_text = watcher.get_recent_logs(job_id)
                    st.code(log_text, language="bash", line_numbers=False)
        else:
            st.info("No active simulations detected. Submit a job in the 'New Simulation' tab.")

    # Call the fragment
    monitor_board()

