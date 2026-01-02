# Job Extraction Guide: Logs and CAD Files

This guide helps you extract logs and uploaded CAD files from AWS instances using a mesh Job ID (e.g., `MSH-0102-E5B6`).

## Quick Reference

**Job ID Format:**
- `IMP-MMDD-XXXX` → Import (upload) job
- `MSH-MMDD-XXXX` → Mesh generation job

**Example:** `MSH-0102-E5B6` = Mesh job on January 2nd with unique ID `E5B6`

---

## Architecture Overview

### Job Logging System
- **Storage Location:** `backend/logs/jobs_YYYY-MM.jsonl`
- **Format:** JSON Lines (one JSON object per line)
- **Rotation:** Monthly files (e.g., `jobs_2026-01.jsonl` for January 2026)

### File Storage System

The application uses two storage backends depending on environment:

#### Local Development
- **CAD Files:** `backend/uploads/`
- **Mesh Files:** `backend/outputs/`
- **Structure:** Flat structure with filenames like `{project_id}_{filename}`

#### Production (AWS)
- **Storage:** AWS S3 bucket `muaz-webdev-assets`
- **CAD Files:** `s3://muaz-webdev-assets/{user_email}/uploads/{filename}`
- **Mesh Files:** `s3://muaz-webdev-assets/{user_email}/mesh/{filename}`
- **Structure:** User email-based folders

---

## Step 1: Extract Job Logs

### On the AWS EC2 Instance

SSH into the EC2 instance and navigate to the backend logs directory:

```bash
cd /home/ubuntu/MeshPackageLean/backend/logs
```

### Find the Log File

Job ID `MSH-0102-E5B6` means the job was created on January 2nd (01-02). Look for the log file for that month:

```bash
# For January jobs
cat jobs_2026-01.jsonl | grep "MSH-0102-E5B6"
```

### Extract All Entries for a Job ID

```bash
cat jobs_2026-01.jsonl | jq '. | select(.job_id == "MSH-0102-E5B6")'
```

Or without `jq`:

```bash
grep "MSH-0102-E5B6" jobs_2026-01.jsonl
```

### Understanding Log Entries

Each log entry contains:
- `timestamp`: When the event occurred
- `job_id`: The unique job identifier
- `job_type`: Either "import" or "mesh"
- `user_email`: User who initiated the job
- `project_id`: Associated project ID
- `filename`: Original CAD filename
- `status`: Job status (started, processing, completed, error)
- `strategy`: Mesh strategy used (for mesh jobs)
- `quality_params`: Quality settings
- `details`: Additional metadata

**Example log entries:**
```json
{"timestamp": "2026-01-02T15:30:45.123456Z", "job_id": "MSH-0102-E5B6", "job_type": "mesh", "user_email": "user@example.com", "project_id": "abc-123", "filename": "model.step", "status": "started", "strategy": "Tetrahedral (HXT)", "quality_params": null, "details": {}}
{"timestamp": "2026-01-02T15:35:22.654321Z", "job_id": "MSH-0102-E5B6", "status": "completed", "details": {"node_count": 15000, "element_count": 87000}}
```

### Download Logs to Your Local Machine

```bash
# From your local machine
scp ubuntu@<ec2-ip>:/home/ubuntu/MeshPackageLean/backend/logs/jobs_2026-01.jsonl ./
```

Then filter locally:
```bash
cat jobs_2026-01.jsonl | grep "MSH-0102-E5B6"
```

---

## Step 2: Extract CAD Files

### Locate the File Information

First, get the user email and filename from the log:

```bash
cat jobs_2026-01.jsonl | jq '. | select(.job_id == "MSH-0102-E5B6") | {user_email, filename, project_id}'
```

Example output:
```json
{
  "user_email": "user@example.com",
  "filename": "combustion_chamber.step",
  "project_id": "abc-123"
}
```

### Download from S3

The CAD file is stored in S3 with this structure:
```
s3://muaz-webdev-assets/{user_email}/uploads/{filename}
```

Using AWS CLI on the EC2 instance (or your local machine with AWS credentials):

```bash
# List files for a user
aws s3 ls s3://muaz-webdev-assets/user@example.com/uploads/

# Download specific file
aws s3 cp s3://muaz-webdev-assets/user@example.com/uploads/combustion_chamber.step ./
```

### Alternative: Query Database

If logs don't have complete information, query the database directly:

```bash
# SSH into EC2
cd /home/ubuntu/MeshPackageLean/backend

# Open Python shell
python3

# Inside Python
from models import db, Project, MeshResult
from api_server import app

with app.app_context():
    # Find by project ID (from logs)
    project = Project.query.get('abc-123')
    if project:
        print(f"Filename: {project.filename}")
        print(f"User: {project.user_id}")
        print(f"CAD Path: {project.cad_file_path}")
        print(f"Upload Date: {project.created_at}")
```

---

## Step 3: Extract Mesh Files (if generated)

Mesh files follow the same S3 structure:
```
s3://muaz-webdev-assets/{user_email}/mesh/{mesh_filename}
```

### Find Mesh File Name

From the database:
```python
with app.app_context():
    project = Project.query.get('abc-123')
    if project and project.results:
        for result in project.results:
            print(f"Strategy: {result.strategy}")
            print(f"Mesh Path: {result.mesh_file_path}")
            print(f"Status: {result.status}")
            print(f"Nodes: {result.node_count}")
```

### Download Mesh File

```bash
aws s3 cp s3://muaz-webdev-assets/user@example.com/mesh/abc-123_mesh_hxt.msh ./
```

---

## Automation Script

Save this as `extract_job.py` in the `backend` folder:

```python
#!/usr/bin/env python3
"""
Extract logs and files for a given Job ID
Usage: python extract_job.py MSH-0102-E5B6
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from job_logger import get_logs_by_job_id

def extract_job(job_id):
    """Extract all information for a given job ID"""
    
    print(f"=== Extracting Job: {job_id} ===\n")
    
    # Get logs
    logs = get_logs_by_job_id(job_id)
    
    if not logs:
        print(f"❌ No logs found for job ID: {job_id}")
        print(f"   Make sure the job ID is correct and logs exist.")
        return
    
    print(f"✅ Found {len(logs)} log entries\n")
    
    # Display logs
    print("--- Log Entries ---")
    for log in logs:
        print(json.dumps(log, indent=2, default=str))
        print()
    
    # Extract key information
    first_log = logs[0]
    user_email = first_log.get('user_email')
    filename = first_log.get('filename')
    project_id = first_log.get('project_id')
    job_type = first_log.get('job_type')
    
    print("--- Key Information ---")
    print(f"Job Type: {job_type}")
    print(f"User Email: {user_email}")
    print(f"Filename: {filename}")
    print(f"Project ID: {project_id}")
    print()
    
    # Provide S3 paths
    if user_email and filename:
        print("--- S3 File Locations ---")
        cad_path = f"s3://muaz-webdev-assets/{user_email}/uploads/{filename}"
        print(f"CAD File: {cad_path}")
        print(f"\nTo download:")
        print(f"  aws s3 cp {cad_path} ./")
        
        if job_type == 'mesh' and project_id:
            print(f"\nMesh files (check actual filenames in database):")
            mesh_path = f"s3://muaz-webdev-assets/{user_email}/mesh/"
            print(f"  aws s3 ls {mesh_path}")
            print(f"  aws s3 cp {mesh_path}{project_id}_*.msh ./")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python extract_job.py <JOB_ID>")
        print("Example: python extract_job.py MSH-0102-E5B6")
        sys.exit(1)
    
    job_id = sys.argv[1]
    extract_job(job_id)
```

### Usage

```bash
cd /home/ubuntu/MeshPackageLean/backend
python3 extract_job.py MSH-0102-E5B6
```

---

## Troubleshooting

### Problem: No logs found for Job ID

**Solutions:**
1. Check the date in the Job ID (MMDD) and look in the correct monthly log file
2. The job might be in the previous month if it was created near midnight
3. Make sure you're on the correct EC2 instance (production vs staging)

### Problem: AWS S3 access denied

**Solutions:**
1. Ensure the EC2 instance has the correct IAM role attached
2. Check AWS credentials with: `aws s3 ls s3://muaz-webdev-assets/`
3. Contact the AWS administrator for S3 bucket permissions

### Problem: Database connection failed

**Solutions:**
1. Check environment variables in `/home/ubuntu/MeshPackageLean/backend/.env`
2. Verify RDS instance is running
3. Check security group rules allow EC2 to connect to RDS

### Problem: File not found in S3

**Solutions:**
1. The job may have failed during upload - check log status
2. Files might be in local storage if run in development mode
3. Check the database `projects` table for the actual `cad_file_path`

---

## Additional Tools

### List All Jobs for a User

```bash
cd /home/ubuntu/MeshPackageLean/backend
python3 << EOF
from job_logger import get_recent_logs

logs = get_recent_logs(limit=100, user_email='user@example.com')
for log in logs:
    print(f"{log['job_id']} | {log['job_type']} | {log['status']} | {log.get('filename', 'N/A')}")
EOF
```

### Export Logs for Date Range

```bash
cd /home/ubuntu/MeshPackageLean/backend
python3 << EOF
from job_logger import get_logs_by_date_range
from datetime import datetime

start = datetime(2026, 1, 1)
end = datetime(2026, 1, 31)

logs = get_logs_by_date_range(start, end, job_type='mesh')
print(f"Found {len(logs)} mesh jobs in January 2026")
EOF
```

---

## Security Notes

- Job logs may contain sensitive information (user emails, file names)
- Always use secure connections (SSH, AWS IAM roles)
- Do not share AWS credentials or log files publicly
- Delete downloaded files after analysis to comply with data retention policies
