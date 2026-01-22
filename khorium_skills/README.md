# Khorium Skills

AI coding assistant skill modules for Gemini/Claude integration. These modules provide specialized capabilities that can be invoked by AI agents working on this codebase.

## Structure

```
khorium_skills/
├── compute/          # Local worker and CLI tools
├── forge_workflow/     # Hub-and-spoke prompting architecture
├── guardian/         # Mesh quality guardian system
├── researcher/       # Deep logic and research protocols for hard problems
├── rulebook/         # Developer standards and test-first protocols
├── tda_harness/      # Test-driven agent harness for closed-loop debugging
└── toolbox/          # Utility scripts for diagnostics
```

## Modules

### `compute/`
Local compute utilities for running mesh generation jobs.
- `local_worker.py` - Local job execution worker
- `mesh_fast_cli.py` - CLI for fast mesh generation

### `guardian/`
The Guardian system validates mesh quality and repairs problematic geometry.
- `guardian.py` - Main validation orchestrator
- `inspectors.py` - Geometry inspection utilities
- `repairers.py` - Automatic geometry repair tools
- `cleaners.py` - Mesh cleanup utilities

### `toolbox/`
Utility scripts for mesh diagnostics and analysis.
- `cad_to_mesh.py` - CAD to mesh conversion
- `cfd_quality.py` - CFD quality metric calculations
- `mesh_diagnostic.py` - Mesh diagnostic tools
- `msh_analyzer.py` - MSH file analyzer
- `run_guardian.py` - Guardian execution script

### `rulebook/`
**Khorium Developer Rulebook** - Senior Engineer protocols for building robust, test-first features.
- Test-first development (write verification scripts before implementation)
- Pragmatic testing hierarchy: CLI > API > Browser > Docker
- Coding standards for Python (typer, pydantic) and React
- Strict debugging loop protocols
- Definition of done criteria

### `researcher/`
**Deep Logic & Research Rulebook** - Principal Research Engineer protocols for solving unknown hard problems.
- O.H.E.C. protocol (Observation, Hypothesis, Experiment, Conclusion)
- First Principles analysis
- Binary search debugging for complex failures
- Input sanitization defense

### `forge_workflow/`
**Forge Workflow** - A hub-and-spoke architecture for parallel execution and clean integration of large problems.
- Stage 1: The Architect (Decomposition & Interface Definition)
- Stage 2: The Worker (Parallel Isolated Execution in `_forge/`)
- Stage 3: The Weaver (Integration, Merge & Cleanup)
- Enforces strict root protection and isolation protocols.

### `tda_harness/`
**Test-Driven Agent Harness** - Closed-loop debugging system with three verification gates.
- Gate 1: Syntax & Linter (mypy, flake8)
- Gate 2: Headless Browser (Playwright)
- Gate 3: Container Integration (Docker Compose)
- Automated quality gates to prevent expensive debugging cycles

## Usage

These skills are typically invoked by AI assistants during code development and debugging. They can also be run standalone:

```bash
# Run guardian on a mesh
python khorium_skills/toolbox/run_guardian.py input.msh

# Analyze mesh quality
python khorium_skills/toolbox/msh_analyzer.py input.msh
```
