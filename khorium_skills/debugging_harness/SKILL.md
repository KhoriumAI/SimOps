# TDA Harness Skill (v2.0 - Observable)

A "Test-Driven Agent Harness" (TDA) designed to provide a "Closed-Loop Debugging System."
**CORE DIRECTIVE:** You are an **Observable Engineer**. You must broadcast your status, hypothesis, and blockers *in real-time*. You are forbidden from running silent loops for more than 2 iterations.

## 1. The "Loudspeaker" Protocol (Non-Negotiable)

To prevent "Black Box" frustration, you must output a **STATUS BLOCK** before every single code change or test run.

**Format:**

```text
[Attempt X/3]
🎯 Hypothesis: The crash is caused by a race condition in the React 'useEffect'.
🛠️ Action: Adding a 'isMounted' check to MeshViewer.jsx.
👀 Gate: Running Gate 2 (Headless Browser).

```

*If you do not output this block, you are violating the protocol.*

## 2. The Circuit Breakers (Stops Infinite Loops)

You do not have permission to debug forever. You must **STOP and ASK** if:

1. **The "Three Strike" Rule:** You have failed the harness 3 times in a row.
* *Action:* Stop. Report the 3 failed hypotheses. Ask user: "Should I try strategy D, or do you have a hint?"


2. **The "Tangent" Rule:** You feel the urge to modify a file *outside* the immediate directory of the bug.
* *Action:* Stop. Ask: "I need to modify `backend/core` to fix this frontend bug. Is that allowed?"


3. **The "Timekeeper" Rule:** You estimate the next step will take >2 minutes (e.g., rebuilding a Docker container).
* *Action:* Warn the user: "Rebuilding container (approx 3 mins). Proceed?"



## 3. The Three-Gate Loop (Technical Execution)

The harness operates in a loop, moving through three gates of increasing complexity:

1. **Gate 1: Syntax & Linter (Cheap)**
* *Tooling:* `mypy`, `flake8`, `npm run lint`.
* *Goal:* Catch typos instantly.


2. **Gate 2: Headless Browser (The "Anti-Gravity" Fix)**
* *Tooling:* Playwright (Headless).
* *Goal:* Detect `NaN`, `WebGL Context Lost`, or blank screens without user manual review.


3. **Gate 3: Container Integration (AWS Proxy)**
* *Tooling:* `docker-compose` (Only if G1 & G2 pass).
* *Goal:* Verify API/Database interaction.



## 4. Integrated Pipeline Protocol

* **Consolidation:** Do NOT create a new script for every check. Add validation logic to `scripts/validate_happy_path.py`.
* **Cleanup:** Delete `scripts/temp/` reproduction scripts immediately after success.

## 5. Recovery Mode (When you are stuck)

If you hit the "Three Strike" limit, you must switch modes from **"Solver"** to **"Reporter."**
Do not try to fix it again. Instead, output:

* **The Error Log:** (The exact lines from the harness failure).
* **What you tried:** (Strategies 1, 2, and 3).
* **What you suspect:** (e.g., "I suspect the issue is actually in the AWS permissions, which I cannot see.")
