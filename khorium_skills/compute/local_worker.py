import os
import subprocess
import sys
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)
# Using the path specified by the user
TEMP_DIR = r"C:\Users\Owner\Downloads\mesh_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/mesh', methods=['POST', 'GET'])
def mesh_endpoint():
    if request.method == 'GET':
        return "Threadripper High-Speed Node: READY", 200
        
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file stream"}), 400

    local_step = os.path.join(TEMP_DIR, file.filename)
    local_json = local_step.replace(".step", ".json").replace(".STEP", ".json")
    
    # Clean up old files to ensure fresh results
    if os.path.exists(local_json): os.remove(local_json)
    
    file.save(local_step)
    print(f"[THREADRIPPER] Processing {file.filename}...")
    
    try:
        # Determine the path to the CLI script (should be in the same directory)
        cli_script = os.path.join(os.path.dirname(__file__), "mesh_fast_cli.py")
        
        # We call the standalone script to avoid the "Signal" threading error
        # Use sys.executable to ensure we use the same Python environment
        result = subprocess.run(
            [sys.executable, cli_script, local_step, local_json],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode == 0 and os.path.exists(local_json):
            print(f"[SUCCESS] Mesh complete for {file.filename}")
            return send_file(local_json)
        else:
            print(f"[ERROR] Mesh script failed: {result.stderr}")
            return jsonify({"error": "Tessellation failed", "details": result.stderr}), 500
            
    except Exception as e:
        print(f"[CRITICAL] Worker Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Disable the reloader to prevent double-initialization issues on Windows
    app.run(host='0.0.0.0', port=8000, use_reloader=False)
