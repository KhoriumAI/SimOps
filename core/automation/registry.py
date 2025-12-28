# core/automation/registry.py
"""
Whitelister for SimOps Command Dispatcher.
Only commands defined here can be executed by the dispatcher.
"""

# The whitelist of approved commands.
# Use {args} placeholders for dynamic inputs.
COMMANDS = {
    # --- INFRASTRUCTURE ---
    "docker_rebuild_fast": "rebuild_workers.bat 1", # Select option 1 in batch
    "docker_rebuild_full": "rebuild_workers.bat 2",
    "docker_kill": "docker-compose kill worker",
    "docker_up": "docker-compose up -d worker",
    "prune_logs": "powershell \"Get-ChildItem -Path jobs_log -Filter *.json | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } | Remove-Item -Force\"",
    
    # --- FULL PIPELINE (SimOps Worker) ---
    "run_sim": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" -o \"{output_dir}\" -c \"{config_path}\"",
    
    # --- STRUCTURAL (CalculiX) ---
    "structural_mesh": "python -m core.strategies.structural_strategy --input \"{input_step}\" --output \"{output_msh}\" --size {size_mult}",
    "structural_solve": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage solve --physics structural -o \"{output_dir}\"",
    "structural_report": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage report --physics structural -o \"{output_dir}\"",
    
    # --- THERMAL (CalculiX) ---
    "thermal_mesh": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage mesh --physics thermal -o \"{output_dir}\"",
    "thermal_solve": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage solve --physics thermal -o \"{output_dir}\"",
    "thermal_report": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage report --physics thermal -o \"{output_dir}\"",
    
    # --- CFD (OpenFOAM) ---
    "cfd_mesh": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage mesh --physics cfd -o \"{output_dir}\"",
    "cfd_setup": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage setup --physics cfd -o \"{output_dir}\"",
    "cfd_solve": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage solve --physics cfd -o \"{output_dir}\"",
    "cfd_report": "python \"{base_dir}/simops_worker.py\" \"{file_path}\" --stage report --physics cfd -o \"{output_dir}\"",
    
    # --- UTILITIES ---
    "validate_batch": "python \"{base_dir}/scripts/validate_batch.py\"",
    "clean_output": "powershell \"Remove-Item -Path output/* -Recurse -Force\"",
    "clean_sweep": "powershell \"Remove-Item -Path output_sweep/* -Recurse -Force; Remove-Item -Path jobs_log/* -Recurse -Force\"",
    "purge_queue": "powershell \"Remove-Item -Path jobs_queue/* -Recurse -Force\"",
    "check_system": "systeminfo | findstr /B /C:\"OS Name\" /C:\"OS Version\" /C:\"System Type\"",
    "check_ccx": "\"{base_dir}/ccx_wsl.bat\" --version",
    "locate_ccx": "powershell \"Get-ChildItem -Path ../CalculiX-Windows-master -Filter *.exe -Recurse -ErrorAction SilentlyContinue | Select-Object FullName\"",
    "ls": "powershell \"Get-ChildItem -Path {path} | Select-Object Name, Mode\"",
    "read_log": "powershell \"Get-Content -Path {log_path} -Raw\"",
    "read_file": "powershell \"Get-Content -Path {path} -Raw\"",
    "stop_simops": "powershell \"Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force\"",
    "list_logs": "powershell \"Get-ChildItem -Path jobs_log -Filter *.json | Select-Object Name\"",
    "inspect_job": "powershell \"(Get-Content -Path jobs_log/{job_id}.result.json -Raw | ConvertFrom-Json).stderr + (Get-Content -Path jobs_log/{job_id}.result.json -Raw | ConvertFrom-Json).stdout\"",
    "get_process": "powershell \"Get-Process -Name {name} -ErrorAction SilentlyContinue\"",
    "stop_process": "powershell \"Stop-Process -Name {name} -Force -ErrorAction SilentlyContinue\""
}
