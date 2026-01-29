# Khorium Skills

AI coding assistant skill modules for Gemini/Claude integration. These modules provide specialized capabilities that can be invoked by AI agents working on this codebase.

## Structure

```
khorium_skills/
├── orchestrator/     # Master Orchestrator (The Architect)
├── compute/          # Local worker and CLI tools
├── forge_workflow/   # Parallel execution and task generation
├── frontend_engineering_precision/ # High-fidelity UI and engineering UX patterns
├── guardian/         # Mesh quality guardian system
├── researcher/       # Deep logic and research protocols for hard problems
├── rulebook/         # Developer standards and test-first protocols
├── tda_harness/      # Test-driven agent harness for closed-loop debugging
└── toolbox/          # Utility scripts for diagnostics
```

## Modules

### `orchestrator/`
**Master Orchestrator (The Architect)** - Strategic planning, approach negotiation, and sub-skill delegation.
- Phase-based execution (Discovery, Proposal, Negotiation, Handoff)
- Rule of Three: Forces presentation of 3 distinct strategies
- 2-Turn Lock: Prevents implementation until agreement is reached

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

### `tda_harness/`
**Test-Driven Agent Harness** - Closed-loop debugging system with three verification gates.
- Gate 1: Syntax & Linter (mypy, flake8)
- Gate 2: Headless Browser (Playwright)
- Gate 3: Container Integration (Docker Compose)
- Automated quality gates to prevent expensive debugging cycles

### `forge_workflow/`
**Forge Workflow** - Parallel execution system for complex features across multiple files.
- Task JSON generation for worker agents
- Multistage implementation orchestration

### `frontend_engineering_precision/`
**Frontend Engineering Precision** - Standards for dense, engineering-grade UIs.
- High-fidelity component patterns (SmartInput, MeshViewer)
- Performance-first React patterns

## Usage

These skills are typically invoked by AI assistants during code development and debugging. They can also be run standalone:

```bash
# Run guardian on a mesh
python khorium_skills/toolbox/run_guardian.py input.msh

# Analyze mesh quality
python khorium_skills/toolbox/msh_analyzer.py input.msh
```
