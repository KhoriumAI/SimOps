# Redundancy & Cleanup Report

During the documentation process, the following folders were identified as potentially redundant, outdated, or misplaced.

## Recommended Actions

### 1. `distutils/`
- **Status:** Anomalous System Folder
- **Description:** This folder likely appeared due to an accidental copy or virtual environment issue. It is not part of the source code.
- **Action:** **DELETE**

### 2. `MESH-26/`
- **Status:** Artifact
- **Description:** Seems to be a temporary folder from a specific Jira/Task ticket.
- **Action:** **DELETE** (if no longer needed) or Move to `docs/archive/`

### 3. `temp_*` Folders
- `temp_defeatured`, `temp_failures`, `temp_geometry`, `temp_meshes`, `temp_stls`, `temp_volumes`
- **Status:** Temporary Runtime Data
- **Description:** Directories created during execution to store intermediate files.
- **Action:** **KEEP** (but ensure they are in `.gitignore`) or **CLEAN** (delete contents)

### 4. `gpu_experiments/`
- **Status:** Prototyping
- **Description:** Contains C++ source code (`main.cpp`, `CMakeLists.txt`). If this feature is active, it should be moved to `core/native` or similar; if it's dead code, archive it.
- **Action:** **REVIEW**

### 5. `jobs_log/`, `jobs_queue/`
- **Status:** Runtime Logs/State
- **Description:** Created by the application at runtime.
- **Action:** **KEEP** (ensure in `.gitignore`)

### 6. `dist/`, `build/` (if present)
- **Status:** Build Artifacts
- **Action:** **DELETE** (can be regenerated)
