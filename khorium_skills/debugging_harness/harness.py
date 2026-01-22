"""
TDA Harness: Test-Driven Agent Harness
Orchestrates multiple verification gates to ensure code quality.
Philosophy: All checks should be integrated into the main pipeline (DEVOPS_CHECKLIST.ps1)
to avoid script sprawl and ensure a single-point-of-failure verification.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
from gates.linter import run_linter
from gates.browser import run_browser_check
from gates.container import run_container_check

def main():
    parser = argparse.ArgumentParser(description="TDA Harness: Test-Driven Agent Harness")
    parser.add_argument("--check", help="Path to the custom success condition script (e.g. scripts/check_fix.py)")
    parser.add_argument("--url", default="http://localhost:3000", help="Frontend URL for Gate 2")
    parser.add_argument("--api", default="http://localhost:5000/api/health", help="API URL for Gate 3")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for the agent loop (handled by the AI agent, not this script)")
    parser.add_argument("--skip-gates", nargs="*", default=[], help="Gates to skip (e.g. browser container)")
    
    args = parser.parse_args()
    root_dir = os.getcwd()

    print("\n" + "="*50)
    print("   TDA HARNESS: STARTING VERIFICATION")
    print("="*50 + "\n")

    failure_logs = []

    # 1. Custom Check Script (The "Success Condition")
    if args.check:
        print(f"[*] Running Custom Success Condition: {args.check}")
        try:
            process = subprocess.run(
                ["python", args.check],
                capture_output=True,
                text=True
            )
            if process.returncode != 0:
                print(f"❌ Success Condition Failed:\n{process.stdout}\n{process.stderr}")
                failure_logs.append(f"Success Condition Failure ({args.check}):\n{process.stdout}\n{process.stderr}")
            else:
                print("✅ Success Condition Passed.")
        except Exception as e:
            print(f"❌ Failed to run check script: {str(e)}")
            failure_logs.append(f"Harness Error: Failed to run check script {args.check}: {str(e)}")

    # 2. Gate 1: Integrity & Linter
    if "linter" not in args.skip_gates:
        success, logs = run_linter(root_dir)
        if not success:
            failure_logs.append(f"Gate 1 (Linter) Failure:\n{logs}")
            print("❌ Gate 1 (Linter) Failed.")
        else:
            print("✅ Gate 1 (Linter) Passed.")

    # 3. Gate 2: Browser Check
    if "browser" not in args.skip_gates:
        success, logs = run_browser_check(args.url)
        if not success:
            failure_logs.append(f"Gate 2 (Browser) Failure:\n{logs}")
            print("❌ Gate 2 (Browser) Failed.")
        else:
            print("✅ Gate 2 (Browser) Passed.")

    # 4. Gate 3: Container Integration
    if "container" not in args.skip_gates:
        success, logs = run_container_check(args.api)
        if not success:
            failure_logs.append(f"Gate 3 (Container) Failure:\n{logs}")
            print("❌ Gate 3 (Container) Failed.")
        else:
            print("✅ Gate 3 (Container) Passed.")

    # 5. Integrated Pipeline: DevOps Checklist
    if "pipeline" not in args.skip_gates:
        print("[*] Running Integrated Pipeline (DEVOPS_CHECKLIST.ps1)...")
        try:
            # Run PowerShell script
            process = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", "DEVOPS_CHECKLIST.ps1", "-PreCommit", "-NoWait"],
                capture_output=True,
                text=True
            )
            if process.returncode != 0:
                print("❌ Integrated Pipeline Failed.")
                failure_logs.append(f"Integrated Pipeline Failure (DEVOPS_CHECKLIST.ps1):\n{process.stdout}\n{process.stderr}")
            else:
                print("✅ Integrated Pipeline Passed.")
        except Exception as e:
            print(f"❌ Failed to run Integrated Pipeline: {str(e)}")
            failure_logs.append(f"Harness Error: Failed to run DEVOPS_CHECKLIST.ps1: {str(e)}")

    print("\n" + "="*50)
    if not failure_logs:
        print("🎉 ALL GATES PASSED! CODE IS DEPLOYMENT READY.")
        print("💡 REMINDER: Ensure your check logic is permanently integrated into the pipeline (e.g. scripts/validate_happy_path.py or DEVOPS_CHECKLIST.ps1).")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print("🚨 VERIFICATION FAILED. SEE LOGS BELOW.")
        print("="*50 + "\n")
        for log in failure_logs:
            print(log)
            print("-" * 20)
        sys.exit(1)

if __name__ == "__main__":
    main()
