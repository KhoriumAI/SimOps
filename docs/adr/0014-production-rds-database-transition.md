# 0014: Production RDS Database Transition & Environment Management
*Status:* Accepted
*Date:* 2026-01-03
*Tags:* #backend #devops #database #security

## 1. Context & Problem Statement
The Dev (Sandbox) environment was previously relying on a local SQLite database file (`mesh_app.db`). This led to several critical issues:
* **Data Persistence:** User accounts and projects were lost during deployments as the SQLite file was frequently overwritten or reset.
* **Scalability:** SQLite is not suitable for concurrent production workloads.
* **Reliability:** A 504 Gateway Timeout error was identified during registration, partially caused by database locking and partially by a configuration mismatch between the frontend calling a production-ready domain and the backend missing necessary connectivity to the persistent database.

## 2. Technical Decision
We transitioned the production-grade Dev environment to a managed **AWS RDS PostgreSQL** instance (`khorium-webdev-db`).

* **Mechanism:** 
    * Switched `SQLALCHEMY_DATABASE_URI` from SQLite to PostgreSQL.
    * Implemented mandatory `sslmode=require` for RDS connectivity.
    * Adopted a secure environment variable injection pattern using **GitHub Secrets**.
* **Dependencies:** 
    * `psycopg2-binary` for PostgreSQL connectivity.
    * AWS RDS Instance (`khorium-webdev-db`) configured for VPC access.

## 3. Mathematical & Physical Implications
* **Concurrency:** The transition to PostgreSQL allows for reliable concurrent access to the `User` and `MeshResult` tables, preventing the application-level "hangs" seen with SQLite.
* **Integrity:** Database constraints (Unique emails, Foreign Keys) are now strictly enforced by the PostgreSQL engine.

## 4. Performance Trade-offs
* **Compute Cost:** Slight overhead for SSL/TLS handshakes on every database connection.
* **Latency:** Database operations now involve local network latency (EC2 to RDS) rather than direct disk I/O, though this is negligible compared to the stability gains.

## 5. Verification Plan
* **Sanity Check:** Use the updated `backend/diagnose_auth.py` script on the EC2 instance to verify:
    1. `DATABASE_URL` is correctly picked up from `.env`.
    2. Connection to the `postgres` database is successful.
    3. User records persist after a gunicorn restart.
* **Regression:** Verified that existing users (IDs 1-4) are accessible and can authenticate successfully via the `/api/auth/login` endpoint.
