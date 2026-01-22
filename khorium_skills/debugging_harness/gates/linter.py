import subprocess
import os

def run_linter(root_dir=None):
    """
    Gate 1: Syntax & Linter Check (Repurposed from DEVOPS_CHECKLIST.ps1)
    Runs project-specific integrity checks.
    """
    if root_dir is None:
        root_dir = os.getcwd()

    print(f"[*] Running Gate 1 (Integrity & Lints) in {root_dir}...")
    
    errors = []
    
    # 1. Database Connectivity & Initialization
    print("    [1/6] Checking Database Connectivity...")
    try:
        process = subprocess.run(
            ["python", os.path.join("backend", "ensure_db.py")],
            cwd=root_dir,
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"DB Check Failed:\n{process.stdout}\n{process.stderr}")
    except Exception as e:
        errors.append(f"Failed to run ensure_db.py: {str(e)}")

    # 2. Type Safety Check (mypy)
    print("    [2/6] Running Type Safety Check (mypy)...")
    try:
        process = subprocess.run(
            ["python", "-m", "mypy", ".", "--config-file", "mypy.ini"],
            cwd=os.path.join(root_dir, "backend"),
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"Mypy Errors:\n{process.stdout}")
    except Exception as e:
        errors.append(f"Failed to run mypy: {str(e)}")

    # 3. Database Schema Sync (alembic)
    print("    [3/6] Checking Database Schema Sync (alembic)...")
    try:
        process = subprocess.run(
            ["python", "-m", "alembic", "check"],
            cwd=os.path.join(root_dir, "backend"),
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"Alembic Check Failed:\n{process.stdout}")
    except Exception as e:
        errors.append(f"Failed to run alembic check: {str(e)}")

    # 4. Environment Variable Audit
    print("    [4/6] Running Environment Variable Audit...")
    try:
        process = subprocess.run(
            ["python", os.path.join("scripts", "check_env_vars.py")],
            cwd=root_dir,
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"Env Check Failed:\n{process.stdout}")
    except Exception as e:
        errors.append(f"Failed to run check_env_vars.py: {str(e)}")

    # 5. Email Configuration Audit
    print("    [5/6] Running Email Configuration Audit...")
    try:
        process = subprocess.run(
            ["python", os.path.join("scripts", "verify_email.py")],
            cwd=root_dir,
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"Email Audit Failed:\n{process.stdout}")
    except Exception as e:
        errors.append(f"Failed to run verify_email.py: {str(e)}")

    # 6. Modal Compute Audit
    print("    [6/6] Running Modal Compute Audit...")
    try:
        process = subprocess.run(
            ["python", os.path.join("scripts", "verify_compute_config.py")],
            cwd=root_dir,
            capture_output=True,
            text=True
        )
        if process.returncode != 0:
            errors.append(f"Modal Compute Audit Failed:\n{process.stdout}")
    except Exception as e:
        errors.append(f"Failed to run verify_compute_config.py: {str(e)}")

    success = len(errors) == 0
    return success, "\n---\n".join(errors)

if __name__ == "__main__":
    success, logs = run_linter()
    if success:
        print("✅ Gate 1 Passed.")
    else:
        print(f"❌ Gate 1 Failed:\n{logs}")
