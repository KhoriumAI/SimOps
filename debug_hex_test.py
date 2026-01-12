import subprocess
import sys
import json
from pathlib import Path

# Find a STEP file to test with
test_files = list(Path("C:/Users/markm/Downloads/MeshPackageLean/cad_files").glob("*.step"))
if not test_files:
    print("No test STEP files found")
    sys.exit(1)

test_file = str(test_files[0])
print(f"Testing with: {test_file}")

# Run the worker subprocess
worker_script = "C:/Users/markm/Downloads/MeshPackageLean/apps/cli/mesh_worker_subprocess.py"
output_dir = "C:/Users/markm/Downloads/MeshPackageLean/output"

quality_params = json.dumps({"mesh_strategy": "Hex Subdivision"})

cmd = [sys.executable, worker_script, test_file, output_dir, "--quality-params", quality_params]

print(f"Running command: {' '.join(cmd)}")
print("=" * 80)

result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# Try to parse the JSON output
lines = result.stdout.strip().split('\n')
for line in reversed(lines):
    if line.startswith('{'):
        try:
            data = json.loads(line)
            print(f"\nParsed result: {json.dumps(data, indent=2)}")
            break
        except:
            print(f"Failedto parse line as JSON: {line}")
