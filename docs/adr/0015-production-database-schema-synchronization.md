# 0015: Production Database Schema Synchronization
*Status:* Accepted
*Date:* 2026-01-03
*Tags:* #backend #devops #database #rds

## 1. Context & Problem Statement
With the transition to AWS RDS PostgreSQL (ADR-0014), the database schema is no longer automatically managed by file-based SQLite migrations. Recent updates to `backend/models.py` (adding `job_id`, `boundary_zones`, `preview_path`, etc.) resulted in a schema mismatch, causing `psycopg2.errors.UndefinedColumn` errors during API calls.

* *The Constraint:* The production server lacks an automated migration tool (like Alembic) at this stage.
* *The Goal:* Ensure the production RDS schema is in parity with the application models without data loss.

## 2. Technical Decision
We implemented a manual schema patching procedure using AWS SSM (Systems Manager) to execute SQL remediation scripts directly on the production instance.

* *Mechanism:* 
    * Used `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` to ensure idempotency.
    * Executed via a Python bridge script on the EC2 instance to leverage existing environment variables and SQLAlchemy connectivity.
    * Patched `mesh_results`, `projects`, and `users` tables.
* *Dependencies:* 
    * AWS SSM access for the EC2 instance.
    * `sqlalchemy` and `psycopg2-binary` in the production venv.

## 3. Mathematical & Physical Implications
* **Data Integrity:** By using `IF NOT EXISTS` and specific defaults (e.g., `BIGINT DEFAULT 0`), we ensure that existing records remain valid and data is not corrupted during the "live" patch.
* **Service Availability:** The patch was applied without requiring service downtime, as PostgreSQL supports concurrent DDL for simple column additions.

## 4. Performance Trade-offs
* **Execution Latency:** Manual patching via SSM introduces a slight operational delay compared to automated migrations.
* **Risk:** High risk of drift if schema changes are not documented and applied consistently across Dev and Staging environments.

## 5. Verification Plan
* **Sanity Check:** Verified health endpoint `/api/health` returns 200/healthy.
* **Direct Verification:** Simulated a `Project.to_dict(include_results=True)` call within the production environment to ensure all missing columns are correctly polled from the database.
* **Regression:** Confirmed that existing project retrieval workflows (Project ID `4726df9...`) are fully functional.
