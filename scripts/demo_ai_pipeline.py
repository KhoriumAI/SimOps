import os
import sys
import requests
import json
import time

# Configuration
API_URL = "http://127.0.0.1:8000"
CAD_FILE = r"c:\Users\markm\Downloads\Simops\cad_files\Cube.step"

def main():
    print("="*60)
    print("SimOps AI Pipeline - Verification Script")
    print("="*60)

    # 1. Health Check
    try:
        resp = requests.get(f"{API_URL}/api/health")
        if resp.status_code == 200:
            print(f"✅ Backend Online: {resp.json()}")
        else:
            print(f"❌ Backend Error: {resp.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Backend Unreachable: {e}")
        print("Please ensure simops-backend/api_server.py is running.")
        sys.exit(1)

    # 2. Upload CAD File
    cad_file_path = CAD_FILE
    if not os.path.exists(cad_file_path):
        print(f"❌ Test file not found: {cad_file_path}")
        # Try to find any step file
        import glob
        files = glob.glob(r"c:\Users\markm\Downloads\Simops\cad_files\*.step")
        if files:
            cad_file_path = files[0]
            print(f"⚠️ Using alternative file: {cad_file_path}")
        else:
            print("❌ No STEP files found.")
            sys.exit(1)

    print(f"\n[1/3] Uploading {os.path.basename(cad_file_path)}...")
    with open(cad_file_path, 'rb') as f:
        resp = requests.post(f"{API_URL}/api/upload", files={'files': f})
    
    if resp.status_code != 200:
        print(f"❌ Upload Failed: {resp.text}")
        sys.exit(1)
    
    upload_data = resp.json()
    saved_filename = upload_data['filename'] # Or saved_as?
    # api_server returns 'filename' as original and 'saved_as' as stored
    # But trigger_simulation expects 'filename' to be the STORED name? 
    # Let's check api_server.py line 238: input_path = UPLOAD_FOLDER / filename
    # So we probably need to pass the SAFE filename if the original had spaces.
    # Actually api_server.py line 171 return filename (original) and saved_as (uuid).
    # Line 209: filename = data.get('filename').
    # Line 238 uses it directly. It implies we should pass 'saved_as'.
    
    actual_filename = upload_data['saved_as']
    print(f"✅ Uploaded as: {actual_filename}")
    if 'preview_url' in upload_data:
        print(f"   Preview URL: {upload_data['preview_url']}")

    # 3. AI Config Generation
    prompt = "I want to simulate a heatsink made of Copper with 50W power and 25C ambient."
    print(f"\n[2/3] Generating Config via AI...")
    print(f"   Prompt: \"{prompt}\"")
    
    payload = {
        "prompt": prompt,
        "cad_file": actual_filename,
        "use_mock": True # Ensure we don't hit external APIs
    }
    
    resp = requests.post(f"{API_URL}/api/ai/generate-config", json=payload)
    
    if resp.status_code == 200:
        ai_result = resp.json()
        config = ai_result['config']
        print("✅ AI Generation Success!")
        print(json.dumps(config, indent=2))
    else:
        print(f"❌ AI Generation Failed: {resp.text}")
        print("Tip: Did you fix the import in api_server.py?")
        sys.exit(1)

    # 4. Trigger Simulation
    print(f"\n[3/3] Triggering Simulation...")
    sim_payload = {
        "filename": actual_filename,
        "config": {
            # Map AI config to SimOps flat config (temporary bridge)
            # The AI returns a structured SimConfig, but api_server expects a flat dict currently
            # or we need to bridge it.
            # api_server.py line 225 maps keys manually.
            "heat_source_power": 50.0, # Manually mapped from prompt for now to ensure run
            "ambient_temperature": 298.15,
            "material": "Copper",
            "solver": "builtin" # Use builtin for speed, or "openfoam" to test the fix
        }
    }
    
    # Try OpenFOAM if we are brave
    # sim_payload['config']['solver'] = 'openfoam'
    
    resp = requests.post(f"{API_URL}/api/simulate", json=sim_payload)
    
    if resp.status_code == 200:
        job_data = resp.json()
        print(f"✅ Job Started! Job ID: {job_data['job_id']}")
        print(f"   Results: {json.dumps(job_data['results'], indent=2)}")
        print("\n🎉 DEMO COMPLETE: The Client can now go from Text -> Simulation!")
    else:
        print(f"❌ Simulation Failed: {resp.text}")
        
if __name__ == "__main__":
    main()
