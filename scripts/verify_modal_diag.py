
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.modal_client import modal_client
    
    print("Testing ModalClient.diagnose()...")
    diag = modal_client.diagnose()
    print(f"Result: {diag}")
    
    if hasattr(modal_client, 'diagnose'):
        print("PASS: diagnose method exists")
    else:
        print("FAIL: diagnose method missing")
        
except Exception as e:
    print(f"FAIL: Exception during import or execution: {e}")
    import traceback
    traceback.print_exc()
