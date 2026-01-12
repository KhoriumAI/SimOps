"""
Khorium Modal Compute Service - Production Worker
Deployed to Modal.com for serverless GPU meshing.

Usage:
    modal deploy modal_service.py
"""
import modal
import os
import time
import json
import math
from pathlib import Path

# --- CONFIGURATION ---
APP_NAME = "khorium-production"

# --- 1. DEFINE THE CONTAINER IMAGE ---
# Get the project root (parent of backend directory)
project_root = Path(__file__).parent.parent

# Build image with embedded code using add_local_dir (Modal 1.3+ API)
image = (
    modal.Image.debian_slim()
    # Install system dependencies for Gmsh and OpenGL
    .apt_install(
        "libgl1-mesa-glx", 
        "libglu1-mesa", 
        "libxrender1", 
        "libxcursor1",
        "libxcomposite1",
        "libxdamage1",
        "libxrandr2",
        "libxtst6",
        "libxi6",
        "libasound2",
        "libnss3",
        "libatk1.0-0",
        "libatk-bridge2.0-0",
        "libcups2",
        "libdrm2",
        "libgbm1",
        # Additional X11 font dependencies required by Gmsh
        "libxft2",
        "libfontconfig1",
        "libfreetype6",
        "fontconfig",
        "libxinerama1",
    )
    # Install Python dependencies
    .pip_install(
        "numpy", 
        "trimesh", 
        "boto3", 
        "gmsh", 
        "cascadio", 
        "scipy",
        "meshio",
        "requests",  # Required by mesh_worker_subprocess
        "scikit-image",  # Required for marching cubes in strategies
    )
    # Embed local code directories into the image
    .add_local_dir(project_root / "core", remote_path="/root/MeshPackageLean/core")
    .add_local_dir(project_root / "strategies", remote_path="/root/MeshPackageLean/strategies")
    .add_local_dir(project_root / "apps" / "cli", remote_path="/root/MeshPackageLean/apps/cli")
    .add_local_dir(project_root / "converters", remote_path="/root/MeshPackageLean/converters")
)

app = modal.App(APP_NAME)

# --- 2. DEFINE SECRETS ---
aws_secret = modal.Secret.from_name("my-aws-secret")


# --- 3. HELPER FUNCTIONS ---

def _run_tet_strategy(cad_file: str, output_dir: str, quality_params: dict, 
                      algorithm_2d: int = 6, algorithm_3d: int = 10, 
                      name: str = "tet", optimize: bool = False) -> dict:
    """
    Run a tetrahedral meshing strategy with configurable Gmsh algorithms.
    This is a lightweight wrapper that uses Gmsh directly with OpenMP threading.
    
    Algorithm3D options:
    - 1: Delaunay (standard, stable)
    - 4: Frontal
    - 10: HXT (parallel, fast)
    """
    import gmsh
    
    start_time = time.time()
    mesh_name = Path(cad_file).stem
    quality_preset = quality_params.get('quality_preset', 'medium') if quality_params else 'medium'
    output_file = f"{output_dir}/{mesh_name}_{quality_preset}_{name}.msh"
    
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        
        # Load CAD
        print(f"[{name.upper()}] Loading CAD: {cad_file}")
        gmsh.model.occ.importShapes(cad_file)
        gmsh.model.occ.synchronize()
        
        # Get bounding box for sizing
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diagonal = ((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)**0.5
        
        # Set mesh sizes
        if quality_params:
            max_size = quality_params.get('max_size_mm', diagonal / 15.0)
            min_size = quality_params.get('min_size_mm', diagonal / 100.0)
            element_order = quality_params.get('element_order', 1)
            curvature_adaptive = quality_params.get('curvature_adaptive', True)
        else:
            max_size = diagonal / 15.0
            min_size = diagonal / 100.0
            element_order = 1
            curvature_adaptive = True
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1 if curvature_adaptive else 0)
        gmsh.option.setNumber("Mesh.MinimumCircleNodes", 12)
        gmsh.option.setNumber("Mesh.ElementOrder", element_order)
        
        # Set algorithms
        gmsh.option.setNumber("Mesh.Algorithm", algorithm_2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)
        
        # Generate mesh
        print(f"[{name.upper()}] Generating 3D mesh (Algorithm3D={algorithm_3d})...")
        mesh_start = time.time()
        gmsh.model.mesh.generate(3)
        mesh_time = time.time() - mesh_start
        print(f"[{name.upper()}] Mesh generation: {mesh_time:.2f}s")
        
        # Optional optimization passes
        if optimize:
            print(f"[{name.upper()}] Running optimization passes...")
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
            gmsh.option.setNumber("Mesh.Smoothing", 5)
            gmsh.model.mesh.optimize("", force=True)
        
        # Get element counts
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        num_nodes = len(node_tags)
        num_elements = sum(len(tags) for tags in elem_tags)
        
        # Compute quality metrics
        quality_metrics = {}
        tet_tags = []
        for et, tags in zip(elem_types, elem_tags):
            if et in [4, 11]:  # Tet4 or Tet10
                tet_tags.extend(tags)
        
        if tet_tags:
            sicn_values = gmsh.model.mesh.getElementQualities(tet_tags, "minSICN")
            quality_metrics['sicn_min'] = float(min(sicn_values))
            quality_metrics['sicn_avg'] = float(sum(sicn_values) / len(sicn_values))
            quality_metrics['sicn_max'] = float(max(sicn_values))
        
        # Write output
        gmsh.write(output_file)
        gmsh.finalize()
        
        total_time = time.time() - start_time
        print(f"[{name.upper()}] SUCCESS! {num_elements} elements in {total_time:.2f}s")
        
        return {
            'success': True,
            'output_file': output_file,
            'strategy': name,
            'message': f'{name}: {num_elements} elements in {total_time:.1f}s',
            'total_elements': num_elements,
            'total_nodes': num_nodes,
            'metrics': {
                'total_elements': num_elements,
                'total_nodes': num_nodes,
                'mesh_time_seconds': mesh_time,
                'total_time_seconds': total_time
            },
            'quality_metrics': quality_metrics
        }
        
    except Exception as e:
        import traceback
        try:
            gmsh.finalize()
        except:
            pass
        return {
            'success': False,
            'message': f'{name} failed: {str(e)}',
            'traceback': traceback.format_exc()
        }


# --- 4. THE REMOTE WORKER FUNCTIONS ---

@app.function(
    image=image,
    timeout=900,
    secrets=[aws_secret],
)
def mesh_single_volume_task(bucket: str, key: str, volume_tag: int, quality_params: dict = None):
    """
    Mesh a single volume of an assembly in isolation.
    """
    import sys
    import os
    import boto3
    import subprocess
    from pathlib import Path

    # Set up paths
    root_dir = Path("/root/MeshPackageLean")
    sys.path.insert(0, str(root_dir))
    
    local_input_dir = Path("/tmp/meshgen_input")
    local_input_dir.mkdir(parents=True, exist_ok=True)
    local_input_file = local_input_dir / Path(key).name
    
    local_output_dir = Path("/tmp/meshgen_output")
    local_output_dir.mkdir(parents=True, exist_ok=True)
    output_msh = local_output_dir / f"vol_{volume_tag}.msh"

    # Download CAD
    s3 = boto3.client("s3")
    try:
        if not local_input_file.exists():
            print(f"[Vol {volume_tag}] Downloading {key}...")
            s3.download_file(bucket, key, str(local_input_file))
    except Exception as e:
        return {"success": False, "tag": volume_tag, "error": f"S3 Download failed: {e}"}

    # Run Isolation Worker
    worker_script = root_dir / "core" / "isolation_worker_script.py"
    
    strategy = quality_params.get('mesh_strategy', 'tet_hxt_optimized') if quality_params else 'tet_hxt_optimized'
    if not strategy: strategy = 'tet_hxt_optimized'
        
    order = str(quality_params.get('element_order', 1)) if quality_params else "1"

    cmd = [
        sys.executable,
        str(worker_script),
        "--input", str(local_input_file),
        "--output", str(output_msh),
        "--tag", str(volume_tag),
        "--strategy", strategy,
        "--order", order
    ]
    
    print(f"[Vol {volume_tag}] Starting meshing (Strategy: {strategy})...")
    try:
        # Run with timeout and real-time output capturing
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )
        
        logs = []
        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"[Vol {volume_tag}] {line}")
                logs.append(line)
        
        process.wait()
        
        if process.returncode == 0 and output_msh.exists():
            # Upload fragment to temporary S3 location
            fragment_key = key.replace("uploads/", "temp_fragments/").replace(".step", "").replace(".stp", "") + f"/vol_{volume_tag}.msh"
            s3.upload_file(str(output_msh), bucket, fragment_key)
            # Try to find quality metrics in the logs
            quality_metrics = {}
            for line in logs:
                if line.startswith('{') and '"per_element_quality"' in line:
                    try:
                        data = json.loads(line)
                        if 'quality_metrics' in data:
                            quality_metrics = data['quality_metrics']
                    except: 
                        pass

            return {
                "success": True, 
                "tag": volume_tag, 
                "fragment_key": fragment_key,
                "log": logs[-200:],
                "quality_metrics": quality_metrics,
                "total_elements": quality_metrics.get('total_elements', 0)
            }
        else:
            return {
                "success": False, 
                "tag": volume_tag, 
                "error": f"Worker failed (Exit {process.returncode})",
                "stderr": "",
                "stdout": "\n".join(logs[-100:]) 
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "tag": volume_tag, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "tag": volume_tag, "error": str(e)}

@app.function(
    image=image,
    gpu="T4",
    timeout=1200,
    secrets=[aws_secret],
)
def generate_mesh(bucket: str, key: str, quality_params: dict = None, webhook_url: str = None, job_id: str = None):
    """
    Generate mesh in Modal serverless GPU container.
    Handles both single-part and parallel assembly meshing.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        quality_params: Mesh quality parameters
        webhook_url: Optional webhook URL to call on completion
        job_id: Optional job ID for CloudWatch logging
    """
    import boto3
    import sys
    import os
    import time
    import gmsh
    import json
    import requests
    from pathlib import Path
    from datetime import datetime
    
    # Add project root to path for imports
    sys.path.insert(0, "/root/MeshPackageLean")
    
    # Import logic
    from apps.cli.mesh_worker_subprocess import (
        generate_fast_tet_delaunay_mesh,
        generate_hex_dominant_mesh
    )
    
    # Get job_id from Modal context if not provided
    if not job_id:
        try:
            # Modal provides function_call_id in the context
            import modal
            if hasattr(modal, 'function_call_id'):
                job_id = modal.function_call_id()
            elif hasattr(modal, 'current_function_call'):
                call = modal.current_function_call()
                if call:
                    job_id = call.object_id
        except Exception as e:
            print(f"[Modal] Could not get job_id from context: {e}")
    
    # Setup CloudWatch Logs client
    logs_client = None
    log_group_name = None
    log_stream_name = None
    
    if job_id:
        try:
            logs_client = boto3.client('logs', region_name=os.environ.get('AWS_REGION', 'us-west-1'))
            log_group_name = f"/modal/jobs/{job_id}"
            log_stream_name = f"job-{job_id}"
            
            # Create log group if it doesn't exist
            try:
                logs_client.create_log_group(logGroupName=log_group_name)
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass
            
            # Create log stream
            try:
                logs_client.create_log_stream(logGroupName=log_group_name, logStreamName=log_stream_name)
            except logs_client.exceptions.ResourceAlreadyExistsException:
                pass
        except Exception as e:
            print(f"[Modal] Warning: Could not setup CloudWatch logging: {e}")
            logs_client = None
    
    def log_to_cloudwatch(message: str):
        """Helper to log to CloudWatch"""
        if logs_client and log_group_name and log_stream_name:
            try:
                timestamp = int(time.time() * 1000)
                logs_client.put_log_events(
                    logGroupName=log_group_name,
                    logStreamName=log_stream_name,
                    logEvents=[{
                        'timestamp': timestamp,
                        'message': message
                    }]
                )
            except Exception as e:
                # Don't fail the job if CloudWatch logging fails
                print(f"[Modal] CloudWatch log error: {e}")
    
    def log(message: str):
        """Unified logging that goes to both stdout and CloudWatch"""
        print(message)
        log_to_cloudwatch(message)
    
    def send_webhook(result_data: dict):
        """Send webhook notification to backend when job completes"""
        if not webhook_url:
            log("[Modal] No webhook URL provided, skipping notification")
            return
        
        try:
            # Prepare payload matching webhook endpoint schema
            status = 'completed' if result_data.get('success') else 'failed'
            payload = {
                'job_id': job_id or 'unknown',
                'status': status,
            }
            
            if result_data.get('success'):
                payload['result'] = {
                    'success': True,
                    's3_output_path': result_data.get('s3_output_path'),
                    'strategy': result_data.get('strategy'),
                    'total_nodes': result_data.get('total_nodes', 0),
                    'total_elements': result_data.get('total_elements', 0),
                    'quality_metrics': result_data.get('quality_metrics', {}),
                    'metrics': {
                        'total_time_seconds': result_data.get('modal_duration_seconds', 0)
                    }
                }
            else:
                payload['error'] = result_data.get('message', 'Unknown error')
            
            # Send POST request to webhook endpoint
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                log(f"[Modal] Webhook notification sent successfully: {response.json()}")
            else:
                log(f"[Modal] Webhook notification failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            # Don't fail the job if webhook fails - just log it
            log(f"[Modal] Webhook error (non-fatal): {e}")
        except Exception as e:
            log(f"[Modal] Webhook error (non-fatal): {e}")
    
    log(f"[Modal] Starting job for s3://{bucket}/{key}")
    
    # Setup Paths
    local_input_dir = Path("/tmp/meshgen_input")
    local_input_dir.mkdir(parents=True, exist_ok=True)
    local_input_file = local_input_dir / Path(key).name
    
    # Download S3
    s3 = boto3.client("s3")
    try:
        print(f"[Modal] Downloading {key}...")
        s3.download_file(bucket, key, str(local_input_file))
    except Exception as e:
        return {"success": False, "message": f"FAILED to download from S3: {e}"}

    # 1. Analyze Geometry (Assembly Detection)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(local_input_file))
        volumes = gmsh.model.getEntities(3)
        vol_tags = [v[1] for v in volumes]
        num_volumes = len(vol_tags)
        log(f"[Modal] Geometry analysis: {num_volumes} volumes detected.")
    except Exception as e:
        gmsh.finalize()
        error_result = {"success": False, "message": f"Geometry load failed: {e}"}
        send_webhook(error_result)
        return error_result
    gmsh.finalize()

    start_time = time.time()
    local_output_dir = Path("/tmp/meshgen_output")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # 2. DECISION: Parallel Assembly vs Single Part
    # Use parallel if > 3 volumes
    if num_volumes > 3:
        log(f"[Modal] *** ENGAGING PARALLEL ASSEMBLY MESHING ({num_volumes} parts) ***")
        
        # Fan-out tasks
        try:
            results = list(mesh_single_volume_task.map(
                [bucket] * num_volumes,
                [key] * num_volumes,
                vol_tags,
                [quality_params] * num_volumes
            ))
        except Exception as e:
             error_result = {"success": False, "message": f"Parallel map failed: {e}"}
             send_webhook(error_result)
             return error_result
        
        # Process Results
        successful_fragments = []
        failures = []
        for res in results:
            if res.get('success'):
                successful_fragments.append(res)
            else:
                failures.append(res)
        
        log(f"[Modal] Parallel complete: {len(successful_fragments)}/{num_volumes} success, {len(failures)} failed.")
        
        if len(successful_fragments) == 0:
             error_result = {"success": False, "message": "All volume meshing tasks failed."}
             send_webhook(error_result)
             return error_result
             
        # Merge Fragments
        log("[Modal] Merging fragments...")
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.model.add("Assembly")
        
        # Download and merge each fragment
        for frag in successful_fragments:
            frag_key = frag['fragment_key']
            local_frag = local_output_dir / f"frag_{frag['tag']}.msh"
            try:
                s3.download_file(bucket, frag_key, str(local_frag))
                gmsh.merge(str(local_frag))
                # Renumber to ensure unique IDs across the assembly
                gmsh.model.mesh.renumberNodes()
                gmsh.model.mesh.renumberElements()
            except Exception as e:
                log(f"[Modal] Failed to merge fragment {frag['tag']}: {e}")
        
        # Finalize Assembly Mesh (SaveAll=1 to keep everything)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        
        output_file = str(local_output_dir / "final_assembly.msh")
        gmsh.write(output_file)
        
        # Get final counts
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        num_nodes = len(node_tags)
        num_elements = sum(len(tags) for tags in elem_tags)
        
        gmsh.finalize()
        
        # Calculate timings
        total_time = time.time() - start_time
        
        # Aggregate quality metrics from fragments
        # We need to compute global stats from the fragment stats
        # For simplicity, we can just use the worst/best/avg of the fragments
        # Or, ideally, we should re-compute them on the final merged mesh
        # But re-computing on the whole mesh might be slow in Python.
        # Let's aggregate the fragment metrics if available.
        
        merged_quality = {
            'min_quality': 1.0, 
            'max_quality': 0.0, 
            'avg_quality': 0.0,
            'min_aspect_ratio': 1.0,
            'max_aspect_ratio': 1.0,
             # We can't easily merge the full per-element arrays here without potentially 
             # exceeding payload limits (though S3 handles that). 
             # For the UI summary bars, we usually rely on 'quality_metrics' dict.
        }
        
        # Helper to safely get metric
        def safe_get(d, key, flow, default):
            val = d.get(key)
            if val is None: return default
            return val

        # Aggregation logic
        global_min_q = 1.0
        global_max_q = 0.0
        total_avg_q_weighted = 0.0
        total_elems = 0
        
        for frag in successful_fragments:
            qm = frag.get('quality_metrics', {})
            n_el = frag.get('total_elements', 0)
            if n_el > 0:
                global_min_q = min(global_min_q, safe_get(qm, 'min_quality', min, 1.0))
                global_max_q = max(global_max_q, safe_get(qm, 'max_quality', max, 0.0))
                
                avg = safe_get(qm, 'avg_quality', float, 0.0)
                total_avg_q_weighted += avg * n_el
                total_elems += n_el
        
        if total_elems > 0:
            merged_quality['min_quality'] = global_min_q
            merged_quality['max_quality'] = global_max_q
            merged_quality['avg_quality'] = total_avg_q_weighted / total_elems

        result_data = {
            'success': True,
            'output_file': output_file,
            'strategy': 'parallel_surgical',
            'total_elements': num_elements,
            'total_nodes': num_nodes,
            'metrics': {
                'total_elements': num_elements,
                'total_time_seconds': total_time
            },
            'quality_metrics': merged_quality
        }
        
    else:
        # 3. STANDARD SINGLE PART MESHING
        log("[Modal] standard single-part meshing...")
        os.environ['OMP_NUM_THREADS'] = '4'
        mesh_strategy = quality_params.get('mesh_strategy', '') if quality_params else ''
        
        try:
            if 'Hex Dominant' in mesh_strategy or 'hex_dominant' in mesh_strategy:
                result_data = generate_hex_dominant_mesh(str(local_input_file), str(local_output_dir), quality_params)
            elif 'Tet Delaunay' in mesh_strategy or 'tet_delaunay' in mesh_strategy:
                result_data = _run_tet_strategy(str(local_input_file), str(local_output_dir), quality_params, algorithm_2d=6, algorithm_3d=1, name="tet_delaunay")
            elif 'Tet Frontal' in mesh_strategy or 'tet_frontal' in mesh_strategy:
                result_data = _run_tet_strategy(str(local_input_file), str(local_output_dir), quality_params, algorithm_2d=6, algorithm_3d=4, name="tet_frontal")
            elif 'Tet HXT' in mesh_strategy or 'tet_hxt' in mesh_strategy or 'HXT' in mesh_strategy:
                result_data = _run_tet_strategy(str(local_input_file), str(local_output_dir), quality_params, algorithm_2d=6, algorithm_3d=10, name="tet_hxt", optimize=True)
            else:
                 result_data = generate_fast_tet_delaunay_mesh(str(local_input_file), str(local_output_dir), quality_params)
        except Exception as e:
            import traceback
            error_result = {"success": False, "message": f"Meshing engine crashed: {e}", "traceback": traceback.format_exc()}
            send_webhook(error_result)
            return error_result

    # 4. Upload Result
    duration = time.time() - start_time
    result_data['modal_duration_seconds'] = duration
    
    if result_data.get('success'):
        output_file = result_data.get('output_file')
        if output_file and os.path.exists(output_file):
            output_key = key.replace("uploads/", "mesh/").replace(".step", ".msh").replace(".stp", ".msh")
            log(f"[Modal] Uploading mesh to s3://{bucket}/{output_key}...")
            s3.upload_file(output_file, bucket, output_key)
            result_data['s3_output_path'] = f"s3://{bucket}/{output_key}"
            
            # Result JSON
            result_json_path = str(Path(output_file).with_suffix('')) + '_result.json'
            # (Create it if it doesn't exist for the assembly case)
            if not os.path.exists(result_json_path):
                with open(result_json_path, 'w') as f:
                    json.dump(result_data, f)

            if os.path.exists(result_json_path):
                 json_key = output_key.replace(".msh", "_result.json")
                 s3.upload_file(result_json_path, bucket, json_key)
                 result_data['s3_result_json_path'] = f"s3://{bucket}/{json_key}"
                 
            # Quality JSON
            qual_json = str(Path(output_file).with_suffix('')) + '.quality.json'
            if os.path.exists(qual_json):
                q_key = output_key.replace(".msh", ".quality.json")
                s3.upload_file(qual_json, bucket, q_key)
    
    # Send webhook notification before returning
    send_webhook(result_data)
    log(f"[Modal] Job completed: success={result_data.get('success')}")
    
    return result_data


@app.function(
    image=image,
    timeout=300,
    secrets=[aws_secret],
)
def generate_preview_mesh(bucket: str, key: str):
    """
    Generate CAD preview in Modal (fast triangulation for UI display).
    """
    import boto3
    import sys
    import gmsh
    
    print(f"[Modal] Generating preview for s3://{bucket}/{key}")
    
    # Download from S3
    s3 = boto3.client("s3")
    local_input = f"/tmp/{Path(key).name}"
    s3.download_file(bucket, key, local_input)
    
    try:
        # Use Gmsh for fast surface triangulation
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 1)  # MeshAdapt (fast)
        
        gmsh.open(local_input)
        gmsh.model.occ.synchronize()
        
        # Calculate mesh sizing 
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
        diag = ((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)**0.5
        gmsh.option.setNumber("Mesh.MeshSizeMin", diag / 100.0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 20.0)
        
        # Generate surface mesh only (dim=2)
        gmsh.model.mesh.generate(2)
        
        # Extract mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = {int(tag): [node_coords[3*i], node_coords[3*i+1], node_coords[3*i+2]] 
                 for i, tag in enumerate(node_tags)}
        
        vertices = []
        elem_types, _, node_tags_list = gmsh.model.mesh.getElements(2)
        
        for etype, enodes in zip(elem_types, node_tags_list):
            if etype == 2:  # 3-node triangle
                try:
                    enodes_list = enodes.astype(int).tolist()
                except AttributeError:
                    enodes_list = [int(n) for n in enodes]
                
                for i in range(0, len(enodes_list), 3):
                    n1, n2, n3 = enodes_list[i], enodes_list[i+1], enodes_list[i+2]
                    if n1 in nodes and n2 in nodes and n3 in nodes:
                        vertices.extend(nodes[n1] + nodes[n2] + nodes[n3])
        
        gmsh.finalize()
        
        preview_data = {
            "vertices": vertices,
            "numVertices": len(vertices) // 3,
            "numTriangles": len(vertices) // 9,
            "isPreview": True,
            "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
            "status": "success"
        }
        
        # Save and upload preview JSON
        preview_json_path = "/tmp/preview.json"
        with open(preview_json_path, 'w') as f:
            json.dump(preview_data, f)
            
        preview_key = key.replace("uploads/", "previews/").replace(".step", ".json").replace(".stp", ".json")
        s3.upload_file(preview_json_path, bucket, preview_key)
        
        return {
            "success": True,
            "preview_path": f"s3://{bucket}/{preview_key}",
            "numVertices": preview_data["numVertices"],
            "numTriangles": preview_data["numTriangles"],
            "bbox": preview_data["bbox"]
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False, 
            "message": f"Preview generation failed: {e}",
            "traceback": traceback.format_exc()
        }
