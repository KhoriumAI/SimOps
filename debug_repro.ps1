$env:PYTHONPATH="c:\Users\markm\Downloads\MeshPackageLean"
$env:MeshPackageLean_ROOT="c:\Users\markm\Downloads\MeshPackageLean"
$env:MESH_VERBOSE="1"

# Create a dummy quality params json - Use single quotes around JSON to prevent parsing issues
$qualityParams = '{\"mesh_strategy\": \"HighSpeed GPU\", \"target_elements\": 5000}'

write-host "Running mesh worker with params: $qualityParams"
python apps/cli/mesh_worker_subprocess.py "apps\desktop\test_cube.step" "repro_output" --quality-params $qualityParams
