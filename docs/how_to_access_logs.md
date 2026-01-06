# How-To: Access Backend Logs

**Last Updated:** 2026-01-05
**Audience:** Developers, Support, DevOps

## Overview
This guide explains how to access and interpret the backend logs for the Mesh Generation Service. These logs are essential for tracking job status, debugging failures, and verifying Job IDs.

---

## 1. Quick Access (Frontend)

For immediate feedback on a running job, use the web interface.

1.  **Start a Mesh Generation** for a project.
2.  Watch the **Logs Panel** on the right side of the screen.
3.  Look for the **Job ID** at the very beginning of the log stream:
    ```
    [14:30:01] [INFO] Job Started: MSH-0105-A7F2
    [14:30:02] [INFO] Starting mesh generation...
    ```
    *Note: If the job fails immediately, the error and Job ID will still be visible here.*

---

## 2. Deep Dive (Server Files)

For historical data, debugging crashed jobs, or auditing, access the persistent log files on the backend server.

### Log Location
All job logs are stored in the `backend/logs/` directory.

- **Local Development:** `.../MeshPackageLean/backend/logs/`
- **Production Server:** `/app/backend/logs/` (typically)

### File Format
Logs are rotated monthly and stored in **JSON Lines (.jsonl)** format.
- Filename pattern: `jobs_YYYY-MM.jsonl`
- Example: `jobs_2026-01.jsonl`

### How to Read
Each line is a valid JSON object. You can use any text editor, `cat`, `grep`, or `jq` to parse them.

**Example Entry:**
```json
{
  "timestamp": "2026-01-05T14:30:01.123Z",
  "job_id": "MSH-0105-A7F2",
  "job_type": "mesh",
  "status": "started",
  "user_email": "admin@khorium.com",
  "project_id": "550e8400-e29b-...",
  "filename": "bracket_v2.step",
  "quality_params": { ... }
}
```

### Searching by Job ID
If you have a Job ID (e.g., `MSH-0105-A7F2`), you can find all related events (Start, Complete, Error) by searching the file:

**Linux/Mac/PowerShell:**
```bash
grep "MSH-0105-A7F2" backend/logs/jobs_2026-01.jsonl
```

---

## 3. Job ID Format Key

Job IDs are designed to be human-readable and time-ordered.

`TYPE - DATE - UNIQUE`

- **TYPE**: `MSH` (Meshing) or `IMP` (Import)
- **DATE**: `MMDD` (Month Day)
- **UNIQUE**: 4-character Hex string

**Example:** `MSH-0105-B9E1`
- **Job Type:** Mesh Generation
- **Date:** January 5th
- **ID:** B9E1

---

## 4. Troubleshooting

**Q: I don't see a Job ID in the frontend logs.**
A: Ensure you are running the latest version of `api_server.py`. The Job ID generation was added in the Jan 5, 2026 update.

**Q: The log file is missing.**
A: The backend automatically creates the `logs/` directory and monthly files on the first write. If the directory is empty, no jobs have been run yet in the current month.

**Q: Logs show "Status: started" but no completion.**
A: This usually indicates a hard crash of the worker process or the server itself. Check the server's main console output or system logs (e.g., `journalctl`) for critical python errors that might have bypassed the logger.
