
import sys
import os
from flask import Flask

# Add backend to path
backend_path = os.path.join(os.getcwd(), 'simops-backend')
sys.path.insert(0, backend_path)

print(f"Checking backend at: {backend_path}")

try:
    from api_server import app
    print("Successfully imported api_server")
    
    # Check for new route
    routes = [str(p) for p in app.url_map.iter_rules()]
    
    if '/api/uploads/<path:filename>' in routes:
        print("[PASS] Upload serving route found")
    else:
        print("[FAIL] Upload serving route MISSING")
        sys.exit(1)
        
    print("Backend verification passed.")
    sys.exit(0)
    
except ImportError as e:
    print(f"[FAIL] Could not import api_server: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Unexpected error: {e}")
    sys.exit(1)
