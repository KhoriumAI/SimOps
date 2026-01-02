#!/usr/bin/env python3
"""
Extract logs and files for a given Job ID

Usage: python extract_job.py MSH-0102-E5B6

This script queries the job logging system and database to retrieve:
- All log entries for the specified job ID
- User email, filename, and project ID
- S3 file paths for CAD and mesh files
- Download commands for AWS CLI
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from job_logger import get_logs_by_job_id


def extract_job(job_id):
    """
    Extract all information for a given job ID
    
    Args:
        job_id: Job ID in format MSH-MMDD-XXXX or IMP-MMDD-XXXX
    """
    
    print(f"{'='*60}")
    print(f"  Extracting Job: {job_id}")
    print(f"{'='*60}\n")
    
    # Get logs from job_logger
    logs = get_logs_by_job_id(job_id)
    
    if not logs:
        print(f"❌ No logs found for job ID: {job_id}")
        print(f"\nPossible reasons:")
        print(f"  1. Job ID is incorrect or doesn't exist")
        print(f"  2. Logs are in a different monthly file")
        print(f"  3. Job was created on a different instance")
        print(f"\nTip: Check the date in job ID (MMDD) to find the right log file")
        return
    
    print(f"✅ Found {len(logs)} log entries\n")
    
    # Display all log entries
    print(f"{'─'*60}")
    print("LOG ENTRIES")
    print(f"{'─'*60}\n")
    
    for i, log in enumerate(logs, 1):
        print(f"Entry {i}/{len(logs)}:")
        print(json.dumps(log, indent=2, default=str))
        print()
    
    # Extract key information from the first log entry
    first_log = logs[0]
    user_email = first_log.get('user_email')
    filename = first_log.get('filename')
    project_id = first_log.get('project_id')
    job_type = first_log.get('job_type')
    strategy = first_log.get('strategy')
    timestamp = first_log.get('timestamp')
    
    # Get final status from last log entry
    last_log = logs[-1]
    final_status = last_log.get('status')
    
    # Display key information
    print(f"{'─'*60}")
    print("KEY INFORMATION")
    print(f"{'─'*60}\n")
    print(f"  Job ID:       {job_id}")
    print(f"  Job Type:     {job_type or 'N/A'}")
    print(f"  Status:       {final_status or 'N/A'}")
    print(f"  User Email:   {user_email or 'N/A'}")
    print(f"  Filename:     {filename or 'N/A'}")
    print(f"  Project ID:   {project_id or 'N/A'}")
    print(f"  Timestamp:    {timestamp or 'N/A'}")
    
    if job_type == 'mesh' and strategy:
        print(f"  Strategy:     {strategy}")
    
    # Extract details from logs
    details_list = []
    for log in logs:
        if log.get('details'):
            details_list.append(log['details'])
    
    if details_list:
        print(f"\n  Details:")
        for details in details_list:
            for key, value in details.items():
                print(f"    - {key}: {value}")
    
    print()
    
    # Provide S3 file locations and download commands
    if user_email and filename:
        print(f"{'─'*60}")
        print("S3 FILE LOCATIONS")
        print(f"{'─'*60}\n")
        
        # CAD file
        cad_path = f"s3://muaz-webdev-assets/{user_email}/uploads/{filename}"
        print(f"CAD File:")
        print(f"  Location: {cad_path}")
        print(f"  Download: aws s3 cp {cad_path} ./\n")
        
        # Mesh files (if mesh job)
        if job_type == 'mesh':
            if project_id:
                mesh_dir = f"s3://muaz-webdev-assets/{user_email}/mesh/"
                print(f"Mesh Files:")
                print(f"  Directory: {mesh_dir}")
                print(f"  List:      aws s3 ls {mesh_dir}")
                print(f"  Download:  aws s3 cp {mesh_dir}{project_id}_*.msh ./")
            else:
                print(f"⚠️  Project ID not found in logs. Cannot determine mesh file location.")
        
        print()
    else:
        print(f"⚠️  Missing user_email or filename in logs. Cannot determine S3 paths.\n")
    
    # Additional database query suggestion
    print(f"{'─'*60}")
    print("DATABASE QUERY (Optional)")
    print(f"{'─'*60}\n")
    
    if project_id:
        print(f"To get more details from the database, run:\n")
        print(f"  cd /home/ubuntu/MeshPackageLean/backend")
        print(f"  python3 -c \"")
        print(f"from models import db, Project, MeshResult")
        print(f"from api_server import app")
        print(f"")
        print(f"with app.app_context():")
        print(f"    project = Project.query.get('{project_id}')")
        print(f"    if project:")
        print(f"        print(f'CAD Path: {{project.cad_file_path}}')")
        print(f"        print(f'User ID: {{project.user_id}}')")
        print(f"        print(f'Created: {{project.created_at}}')")
        print(f"        if project.results:")
        print(f"            for r in project.results:")
        print(f"                print(f'Mesh: {{r.mesh_file_path}} ({{r.status}})')")
        print(f"  \"")
    else:
        print(f"  No project ID available. Cannot generate database query.")
    
    print()
    print(f"{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}\n")


def list_recent_jobs(limit=20, job_type=None):
    """List recent jobs"""
    from job_logger import get_recent_logs
    
    print(f"{'='*60}")
    print(f"  Recent Jobs (limit: {limit})")
    print(f"{'='*60}\n")
    
    logs = get_recent_logs(limit=limit, job_type=job_type)
    
    if not logs:
        print("No jobs found.")
        return
    
    print(f"{'Job ID':<18} {'Type':<8} {'Status':<12} {'Filename':<30}")
    print(f"{'-'*18} {'-'*8} {'-'*12} {'-'*30}")
    
    seen_jobs = set()
    for log in logs:
        job_id = log.get('job_id')
        if job_id and job_id not in seen_jobs:
            seen_jobs.add(job_id)
            jtype = log.get('job_type', 'N/A')[:7]
            status = log.get('status', 'N/A')[:11]
            filename = log.get('filename', 'N/A')[:29]
            print(f"{job_id:<18} {jtype:<8} {status:<12} {filename:<30}")
    
    print()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Job Extraction Tool")
        print("=" * 60)
        print("\nUsage:")
        print("  python extract_job.py <JOB_ID>        - Extract specific job")
        print("  python extract_job.py --list [N]       - List recent N jobs (default: 20)")
        print("  python extract_job.py --list-mesh [N]  - List recent N mesh jobs")
        print("\nExamples:")
        print("  python extract_job.py MSH-0102-E5B6")
        print("  python extract_job.py IMP-0101-A7F3")
        print("  python extract_job.py --list 50")
        print("  python extract_job.py --list-mesh 30")
        print()
        sys.exit(1)
    
    arg = sys.argv[1]
    
    if arg == '--list':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        list_recent_jobs(limit=limit)
    elif arg == '--list-mesh':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        list_recent_jobs(limit=limit, job_type='mesh')
    elif arg == '--list-import':
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        list_recent_jobs(limit=limit, job_type='import')
    else:
        # Assume it's a job ID
        job_id = arg
        extract_job(job_id)
