import os
import json
import subprocess
from pathlib import Path

def check_file(path, description):
    if path.exists():
        print(f"✅ {description} found: {path.relative_to(Path.cwd())}")
        return True
    else:
        print(f"❌ {description} MISSING: {path}")
        return False

def verify_structure():
    base_path = Path(__file__).parent
    print(f"\nVerifying structure in {base_path}...")
    
    required = [
        (base_path / "src-tauri/src/system_check.rs", "System health module"),
        (base_path / "src-tauri/src/backend_manager.rs", "Backend manager module"),
        (base_path / "src-tauri/src/main.rs", "Main entry point"),
        (base_path / "src-tauri/tauri.conf.json", "Tauri config"),
        (base_path / "src/App.tsx", "Frontend UI"),
        (base_path / "package.json", "Node package config")
    ]
    
    all_ok = True
    for path, desc in required:
        if not check_file(path, desc):
            all_ok = False
            
    return all_ok

def verify_tauri_config():
    print("\nValidating tauri.conf.json configuration...")
    config_path = Path(__file__).parent / "src-tauri/tauri.conf.json"
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Check bundle ID
        identifier = config.get("tauri", {}).get("bundle", {}).get("identifier")
        if identifier == "com.khorium.simops":
            print("✅ Bundle identifier correct: com.khorium.simops")
        else:
            print(f"❌ Bundle identifier INCORRECT: {identifier}")
            return False
            
        # Check frontend path
        dist_dir = config.get("build", {}).get("distDir")
        if "simops-frontend/dist" in dist_dir:
            print(f"✅ Frontend dist path appears correct: {dist_dir}")
        else:
            print(f"❌ Frontend dist path might be wrong: {dist_dir}")
            return False
            
        # Check HTTP permissions
        http_scope = config.get("tauri", {}).get("allowlist", {}).get("http", {}).get("scope", [])
        if "http://localhost:5000/*" in http_scope:
            print("✅ HTTP permissions for backend configured")
        else:
            print("❌ HTTP permissions for backend MISSING")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error reading tauri.conf.json: {e}")
        return False

def check_rust_compilation():
    print("\nChecking Rust code compilation (cargo check)...")
    tauri_path = Path(__file__).parent / "src-tauri"
    
    try:
        # Run cargo check to verify code integrity without full build
        result = subprocess.run(
            ["cargo", "check"], 
            cwd=tauri_path, 
            capture_output=True, 
            text=True,
            shell=True
        )
        
        if result.returncode == 0:
            print("✅ Rust code compiles successfully")
            return True
        else:
            print("❌ Rust compilation FAILED:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running cargo check: {e}")
        return False

if __name__ == "__main__":
    print("=== SimOps Task 06 Verification ===")
    
    success = True
    if not verify_structure(): success = False
    if not verify_tauri_config(): success = False
    
    # Optional: only check compilation if user has Rust installed
    # check_rust_compilation() 
    
    if success:
        print("\n🎉 TASK_06_DESKTOP_SHELL VERIFICATION PRE-CHECK PASSED!")
        print("Final step: Run 'npm run build' to generate the installer.")
    else:
        print("\n❌ TASK_06_DESKTOP_SHELL VERIFICATION FAILED!")
        exit(1)
