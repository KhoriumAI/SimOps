# SimOps S3 Interface Contract ("The Geneva Convention")

This document defines the binding agreement between the Trigger Service (Muaz) and the Simulation Worker (Antigravity).

## 1. Folder Taxonomy

All S3 operations occur within the `uploads` prefix, organized by a unique `batch_id` (a UUID).

```text
s3://{bucket_name}/uploads/{batch_id}/
├── input/                  # WRITABLE by Trigger/User
│   ├── metadata.json       # Job configuration (Material, BCs)
│   └── model.step          # CAD geometry (STEP, IGES, or BREP)
│
├── output/                 # WRITABLE by Worker
│   ├── result.vtk          # Visualization output (Paraview)
│   ├── report.json         # Simulation results (Temps, Metadata)
│   └── simulation.log      # Orchestrator log copy
│
└── logs/                   # WRITABLE by Worker
    ├── stdout.log          # Raw solver stdout
    └── stderr.log          # Raw solver stderr
```

## 2. Trigger Logic

1.  **Watcher:** The Trigger Service monitors `s3://.../input/`.
2.  **Condition:** A job is considered "Ready" when `metadata.json` AND a geometry file (e.g., `*.step`) are present in the `input/` folder.
3.  **Action:** The Trigger Service pushes a job payload to the Redis Queue.

### Job Payload (Redis)
```json
{
  "batch_id": "550e8400-e29b-41d4-a716-446655440000",
  "bucket": "simops-data",
  "input_key": "uploads/550e8400.../input/model.step",
  "config_key": "uploads/550e8400.../input/metadata.json"
}
```

## 3. Metadata Structure (`metadata.json`)

If this file is missing, the worker will use defaults (Generic Steel, Auto-BCs).

```json
{
  "version": "1.0",
  "simulation_type": "thermal_steady",
  "material": "Aluminum_6061",      // Overrides thermal_conductivity if present
  "parameters": {
    "thermal_conductivity": 150.0,  // Fallback if material not found [W/mK]
    "heat_source_temp_k": 800.0,    // Hot end temperature [K]
    "ambient_temp_k": 300.0,        // Cold end temperature [K]
    "grading": "0.15"               // Mesh grading (0.1 - 1.0)
  },
  "solver": "calculix"              // "calculix" (default) or "openfoam"
}
```

## 4. Output Contract

After completion, the worker guarantees:
1.  `output/result.vtk`: 3D scalar field of temperatures.
2.  `output/report.json`:
    ```json
    {
      "status": "success",
      "batch_id": "...",
      "max_temperature": 798.5,
      "min_temperature": 301.2,
      "solve_time_sec": 4.2
    }
    ```
