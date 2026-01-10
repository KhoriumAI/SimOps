
import sys
import os
import boto3
import time
import json
import argparse
from pathlib import Path

# Add backend to path so we can import modal_service
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from backend.modal_service import app, generate_mesh
    from backend.config import Config
except ImportError as e:
    print(f"Error importing backend modules: {e}")
    print("Make sure you are running this from the project root or scripts directory.")
    sys.exit(1)

def upload_to_s3(file_path: Path, bucket: str) -> str:
    """Uploads local file to S3 and returns the key."""
    s3 = boto3.client('s3')
    key = f"uploads/local_dev/{file_path.name}"
    print(f"Uploading {file_path} to s3://{bucket}/{key}...")
    s3.upload_file(str(file_path), bucket, key)
    return key

def download_from_s3(bucket: str, key: str, local_path: Path):
    """Downloads a file from S3 to a local path."""
    s3 = boto3.client('s3')
    print(f"Downloading s3://{bucket}/{key} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))

@app.local_entrypoint()
def main(input_file: str, bucket: str = None, strategy: str = "tet_hxt", order: int = 1, config_file: str = None):
    """
    Run the meshing pipeline on Modal using local code.
    
    Args:
        input_file: Path to the local .step/.stp file.
        bucket: S3 bucket name (optional, defaults to config).
        strategy: Meshing strategy (default: tet_hxt).
        order: Element order (default: 1).
        config_file: Path to a JSON config file containing quality_params.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return

    # Determine bucket
    if not bucket:
        bucket = os.environ.get('S3_BUCKET_NAME')
        if not bucket:
            # Fallback to hardcoded dev bucket if not set (risky but helpful for dev)
            bucket = Config.S3_BUCKET_NAME
            print(f"Using configured bucket: {bucket}")

    if not bucket:
        print("Error: No S3 bucket specified. Set S3_BUCKET_NAME env var or pass --bucket.")
        return

    # Load config file if provided
    params = {}
    if config_file:
        config_path = Path(config_file)
        if config_path.exists():
            print(f"Loading config from {config_path}...")
            with open(config_path, 'r') as f:
                params = json.load(f)
        else:
            print(f"Warning: Config file {config_file} not found.")

    # Override config with CLI args if not present or specific args passed
    # (Prioritize config if it exists, but ensure defaults if not)
    if "mesh_strategy" not in params:
        params["mesh_strategy"] = strategy
    if "element_order" not in params:
        params["element_order"] = order
    
    # Ensure other defaults
    if "quality_preset" not in params: params["quality_preset"] = "medium"
    if "curvature_adaptive" not in params: params["curvature_adaptive"] = True

    # 1. Upload input to S3
    try:
        s3_key = upload_to_s3(input_path, bucket)
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
        return

    # 2. Trigger Modal Job
    print(f"Triggering Modal job for {s3_key}...")
    print(f"Params: {json.dumps(params, indent=2)}")
    
    start_time = time.time()
    try:
        # .remote() calls the function and waits for the result
        result = generate_mesh.remote(bucket, s3_key, params)
        
        duration = time.time() - start_time
        print("\n" + "="*40)
        print(f"Job Initial Complete in {duration:.2f}s")
        print("="*40)
        
        # print(json.dumps(result, indent=2))
        
        if result.get('success'):
            print(f"\nSUCCESS! Mesh output: {result.get('s3_output_path')}")
            
            # 3. Download Result
            output_dir = project_root / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            # Construct local filename
            mesh_filename = Path(result.get('output_file', 'output.msh')).name
            local_mesh_path = output_dir / mesh_filename
            
            # Parse S3 key from s3_output_path (s3://bucket/key)
            s3_path = result.get('s3_output_path', '')
            if s3_path.startswith('s3://'):
                # s3://bucket/key -> key
                parts = s3_path.replace('s3://', '').split('/', 1)
                if len(parts) == 2:
                    result_key = parts[1]
                    try:
                        download_from_s3(bucket, result_key, local_mesh_path)
                        print(f"Downloaded mesh to: {local_mesh_path}")
                        result['local_mesh_path'] = str(local_mesh_path)
                        result['output_file'] = str(local_mesh_path) # CRITICAL: GUI expects 'output_file' to be the full local path
                    except Exception as e:
                        print(f"Failed to download result mesh: {e}")
            
            # Print JSON result at the end for the GUI worker to parse
            print(json.dumps(result))
            
        else:
            print(f"\nFAILURE: {result.get('message')}")
            print(json.dumps(result))
            
    except Exception as e:
        print(f"\nRPC Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Please run with: modal run scripts/run_local_modal.py --input-file <path>")
