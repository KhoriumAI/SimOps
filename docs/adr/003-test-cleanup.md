# ADR 003: Automated Test Cleanup Pipeline
*Status:* Accepted
*Date:* 2025-12-28
*Tags:* #testing #automation #devops #cleanup

## 1. Context & Problem Statement
The SimOps root directory was significantly cluttered with temporary input (`.inp`), result (`.frd`), and status (`.sta`) files from manual and automated physics verification tests.
* *The Constraint:* Persistent temporary files can lead to confusion during debugging and accidental commits of non-source artifacts.
* *The Goal:* Ensure a clean development environment by automating the removal of transient test data.

## 2. Technical Decision
Implemented a multi-tier cleanup strategy involving standalone scripts, git configuration, and script-level automation.
* *Mechanism:* 
    - `scripts/cleanup_test_files.py`: A centralized script for purging known patterns (`si_*`, `benchmark.*`, etc.).
    - `.gitignore`: Updated patterns to prevent tracking of transient solver files.
    - `run_batch_verification.py`: Integrated pre-flight cleanup.
    - `verify_si_physics.py`: Integrated post-flight cleanup using the `atexit` module.
* *Dependencies:* `os`, `shutil`, `atexit`.

## 3. Mathematical & Physical Implications
* *Conservation:* Not applicable.
* *Stability:* Prevents stale result files from being inadvertently parsed in subsequent test runs.
* *Geometric Constraints:* None.

## 4. Performance Trade-offs
* *Compute Cost:* Negligible. Cleanup takes milliseconds.
* *Memory Cost:* No change.

## 5. Verification Plan
* *Sanity Check:* Executed `python scripts/cleanup_test_files.py` and confirmed no transient files remained in the root directory.
* *Regression:* Batch verification script successfully executes with the new cleanup step.
