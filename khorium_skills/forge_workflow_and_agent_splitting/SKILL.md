---
name: Forge Workflow
description: A hub-and-spoke prompting architecture for parallel execution and clean integration of large problems.
---

# Forge Workflow (Orchestration & Parallel Execution)

The Forge Workflow is a sophisticated **"Orchestration"** architecture designed for **Smart Implementation Pathways** across multiple agents. It enforces strict **File Protocols** and **Interface Contracts** to ensure that a principal agent (The Architect) can design a complex solution and hand off execution to independent sub-agents (The Workers).

## THE ORCHESTRATOR'S MISSION

Your goal is to be the **Architect**. You are not here to write the code for every sub-task. You are here to:
1.  **Decompose** a large problem into isolated, parallelizable tasks.
2.  **Define Boundaries** so sub-agents cannot step on each other's toes.
3.  **Generate Prompt Documents** that provide sufficient context, reasoning, and technical specifications for each sub-agent to work in total isolation.

---

## THE "CLEAN PROJECT" PROTOCOL (NON-NEGOTIABLE)

1.  **Root Protection**: Sub-agents are **FORBIDDEN** from modifying any existing files in the root directory.
2.  **Project Workspace**: All work happens inside a dedicated project folder: `AI_Agent_Projects/[Project_Name]/`.
3.  **Containment**:
    - **Prompts**: `AI_Agent_Projects/[Project_Name]/prompts/`
    - **Tasks**: `AI_Agent_Projects/[Project_Name]/[Task_ID]/`
    - **Artifacts**: All scripts, documentation, and logic MUST reside inside the assigned `[Task_ID]` folder.
4.  **Exceptions**: If a script *must* be placed outside the task folder, a documentation file inside the task folder must explicitly explain why.
5.  **Shadow Copies**: If a task requires modifying existing code, copy it to the task folder, modify it there, and mark it as a "Shadow Copy".

---

## STAGE 1: THE ARCHITECT (PLANNING & ORCHESTRATION)

**Goal**: Analyze the problem, define interfaces, and generate standalone prompt documents for sub-agents.

### Responsibilities
- Break down the large problem into independent, parallelizable sub-tasks.
- Create the project directory: `AI_Agent_Projects/[Project_Name]/`.
- **Generate Prompt Files**: Create `AI_Agent_Projects/[Project_Name]/prompts/` and generate standalone markdown documents (e.g., `#1_TASK_ID.md`).
- These documents must contain:
    - **Mission**: Clear, specific objective for the sub-agent.
    - **Context**: Relevant code snippets, architecture diagrams, and reasoning.
    - **Interfaces**: API signatures, data models, and mocking requirements.
    - **Verification**: Exact requirements for the `verify_task.py` script.

### Output Requirements
The Architect must output:
1.  A **JSON Task List** (`AI_Agent_Projects/[Project_Name]/task_list.json`) summarizing the plan.
2.  A set of **Prompt Documents** in the `prompts/` subdirectory.

---

## STAGE 2: THE WORKER (ISOLATED EXECUTION)

**Goal**: Execute ONE specific task from the Architect's plan using the provided prompt document.

### Responsibilities
- **Sandboxing**: Only write files to the assigned `AI_Agent_Projects/[Project_Name]/[Task_ID]/` directory.
- **Documentation**: If you create a script, put it IN your folder. If you create a doc, put it IN your folder.
- **Independence**: Work strictly from the provided prompt document.
- **Self-Verification**: Every worker folder must contain a `verify_task.py` that proves correctness in isolation.

---

## STAGE 3: THE WEAVER (INTEGRATION & VALIDATION)

**Goal**: Read all isolated worker folders, merge them into the main codebase, and finalize.

### Execution Protocol
1.  **Review**: Read every module in `AI_Agent_Projects/[Project_Name]/`. Resolve conflicts.
2.  **Merge**: Move code from task sub-folders to target directories (e.g., `src/`, `core/`).
3.  **Refactor**: Update imports to reflect the transition from mock paths to real paths.
4.  **Validate**: Run final integration tests.
5.  **Purge**: Delete the `AI_Agent_Projects/[Project_Name]/` directory once (and only once) integration tests pass.

---

## ANTI-PATTERNS TO AVOID

- ❌ Modifying root files before the Weaver stage.
- ❌ Circular dependencies between Architect-defined tasks.
- ❌ Missing `verify_task.py` in Worker folders.
- ❌ Direct imports between `_forge/` sub-folders (use mocks instead).
- ❌ Forgetting to delete `_forge/` after a successful merge.
