# Antigravity Agent Rules

Critical: The following files and directories MUST be ignored by all Antigravity agents during indexing, project-wide searches, and context gathering. These contain large binary data or temporary artifacts that consume excessive memory and context space.

## Data Exclusions (Extensions)
- `*.vtk`
- `*.vtu`
- `*.msh`
- `*.obj`
- `*.stl`
- `*.cgns`
- `*.bdf`
- `*.brep`

## Directory Exclusions
- `output/`
- `generated_meshes/`
- `temp_*/`
- `jobs_log/`
- `sim_build/`
- `meshing_output/`
- `debug_geometry/`

## Rationale
Simulation data in this project can exceed hundreds of megabytes per file. indexing these files causes extreme RAM usage and consumes the agent's context window with un-indexable binary or geometric data.
