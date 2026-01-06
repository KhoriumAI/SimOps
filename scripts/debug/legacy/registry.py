import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

COMMANDS = {
    # INFRASTRUCTURE
    "docker_build": "docker build -t sim_worker .",
    "docker_prune": "docker system prune -f",
    
    # SIMULATION
    "run_parallel_mesher": "python scripts/parallel_vol_mesher.py",
    "run_voxel_repair_v7": f"python {os.path.join(SCRIPTS_DIR, 'voxel_repair_tool_v7.py')} {{input}} {{output}}",
    "run_voxel_repair_v5": f"python {os.path.join(SCRIPTS_DIR, 'voxel_repair_tool_v5.py')} {{input}} {{output}}",
    "run_voxel_repair_v4": f"python {os.path.join(SCRIPTS_DIR, 'voxel_repair_tool_v4.py')} {{input}} {{output}}",
    "run_voxel_repair_v3": f"python {os.path.join(SCRIPTS_DIR, 'voxel_repair_tool_v3.py')} {{input}} {{output}}",
    "run_robust_mesher": "python scripts/parallel_vol_mesher.py",
    "run_multi_extractor": "python multi_stage_extractor.py",
    "run_preserved_smash": "python scripts/voxel_smash_preserved.py",
    "run_shell_preserver": "python scripts/shell_preserver.py",
    "run_step_export": f"python {os.path.join(SCRIPTS_DIR, 'robust_step_exporter.py')}",
    "run_robust_export": "python scripts/export_robust_stl.py",
    "run_iso_export": "python scripts/export_isolated.py",
    "run_tetwild": "docker run --rm -v \"{project_root}:/data\" yixinhu/tetwild:latest --input /data/fused_temp.stl --output /data/robust_mesh.msh --level 2",
    "check_temp_count": "powershell -Command \"(dir temp_stls).Count\"",
    "list_temp_files": "powershell -Command \"dir temp_stls | Select-Object Name\"",
    "run_merge": "python scripts/merge_and_mesh.py",
    "diagnose_volumes": f"python {os.path.join(SCRIPTS_DIR, 'diagnose_volumes.py')} {{mesh_path}}",
    "test_step_load": "python robust_diagnostic.py",
    
    # UTILITIES
    "kill_dispatcher": "powershell -Command \"Get-Process python | ForEach-Object { $cmdLine = (Get-CimInstance Win32_Process -Filter \\\"ProcessId=$($_.Id)\\\").CommandLine; if ($cmdLine -like \\\"*dispatcher.py*\\\") { Stop-Process -Id $_.Id -Force } }\"",
    "cleanup_logs": "del /q jobs_log\\*.log.processing",
    "list_files": "dir \"{directory}\"",
    "ping": "ping 127.0.0.1 -n 4"
}
