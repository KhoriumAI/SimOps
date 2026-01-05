import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, shell=True):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=shell, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0

def deploy_modal():
    print("=" * 60)
    print("Modal Compute Migration - Deployment Script")
    print("=" * 60)
    
    # 1. Check Modal CLI
    print("\n[1/4] Checking Modal CLI...")
    if not run_command("modal --version"):
        print("Error: Modal CLI not found. Please run 'pip install modal' and 'modal token new'")
        return

    # 2. Check AWS Secrets
    print("\n[2/4] Checking Modal secrets...")
    # This is a bit hard to check programmatically without parsing table output, 
    # but we can try to find 'my-aws-secret' in the list.
    result = subprocess.run("modal secret list", shell=True, text=True, capture_output=True)
    if "my-aws-secret" not in result.stdout:
        print("Warning: 'my-aws-secret' not found in Modal secrets.")
        print("Please create it with: modal secret create my-aws-secret AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...")
        # Continue anyway, maybe user named it differently or it's a false negative
    else:
        print("Success: 'my-aws-secret' found.")

    # 3. Deploy the service
    print("\n[3/4] Deploying mesh_service.py to Modal...")
    service_path = Path(__file__).parent.parent / "backend" / "modal_service.py"
    if not service_path.exists():
        print(f"Error: {service_path} not found.")
        return

    if run_command(f"modal deploy {service_path}"):
        print("\nSuccess: Modal service deployed!")
    else:
        print("\nError: Deployment failed.")
        return

    # 4. Verify
    print("\n[4/4] Verifying deployment...")
    run_command("modal app list")
    
    print("\n" + "=" * 60)
    print("Deployment completed!")
    print("Don't forget to set USE_MODAL_COMPUTE=true in your .env file")
    print("=" * 60)

if __name__ == "__main__":
    deploy_modal()
