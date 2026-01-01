#!/usr/bin/env python3
"""
Job Logger Module for Mesh Generation Platform

Provides unique human-readable Job IDs and structured logging for every
import and mesh operation. Logs are stored as JSON-lines files with
automatic monthly rotation.

Job ID Format:
- IMP-MMDD-XXXX  (Import job, month/day, 4 hex chars)
- MSH-MMDD-XXXX  (Mesh job, month/day, 4 hex chars)

Example: MSH-0101-A7F3 = Mesh job on Jan 1st with unique suffix A7F3
"""

import os
import json
import uuid
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Thread-safe lock for file writes
_log_lock = threading.Lock()

# Log directory - configurable via environment variable
LOG_DIR = Path(os.environ.get('JOB_LOG_DIR', Path(__file__).parent / 'logs'))


def _ensure_log_dir():
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def generate_job_id(job_type: str = 'MSH') -> str:
    """
    Generate a unique, human-readable job ID.
    
    Format: {TYPE}-{MMDD}-{4 HEX CHARS}
    Example: MSH-0101-A7F3
    
    Args:
        job_type: 'IMP' for import, 'MSH' for mesh
        
    Returns:
        Unique job ID string
    """
    now = datetime.utcnow()
    date_part = now.strftime('%m%d')
    unique_part = uuid.uuid4().hex[:4].upper()
    return f"{job_type}-{date_part}-{unique_part}"


def _get_log_file() -> Path:
    """Get current log file path (rotates monthly)."""
    _ensure_log_dir()
    month_str = datetime.utcnow().strftime('%Y-%m')
    return LOG_DIR / f'jobs_{month_str}.jsonl'


def _write_log_entry(entry: Dict[str, Any]):
    """Thread-safe write of a log entry."""
    log_file = _get_log_file()
    with _log_lock:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=str) + '\n')


def log_import_job(
    job_id: str,
    user_email: str,
    project_id: str,
    filename: str,
    file_size: int,
    status: str = 'started',
    details: Optional[Dict] = None
) -> None:
    """
    Log an import (file upload) job.
    
    Args:
        job_id: Unique job identifier
        user_email: Email of the user
        project_id: Associated project ID
        filename: Original filename
        file_size: File size in bytes
        status: Job status (started, completed, error)
        details: Additional details dict
    """
    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'job_id': job_id,
        'job_type': 'import',
        'user_email': user_email,
        'project_id': project_id,
        'filename': filename,
        'file_size': file_size,
        'status': status,
        'details': details or {}
    }
    _write_log_entry(entry)
    print(f"[JOB LOG] {job_id} | IMPORT | {status} | {filename} | User: {user_email}")


def log_mesh_job(
    job_id: str,
    user_email: str,
    project_id: str,
    filename: str,
    status: str = 'started',
    strategy: Optional[str] = None,
    quality_params: Optional[Dict] = None,
    details: Optional[Dict] = None
) -> None:
    """
    Log a mesh generation job.
    
    Args:
        job_id: Unique job identifier
        user_email: Email of the user
        project_id: Associated project ID
        filename: Source CAD filename
        status: Job status (started, processing, completed, error)
        strategy: Mesh strategy used
        quality_params: Quality parameters
        details: Additional details dict
    """
    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'job_id': job_id,
        'job_type': 'mesh',
        'user_email': user_email,
        'project_id': project_id,
        'filename': filename,
        'status': status,
        'strategy': strategy,
        'quality_params': quality_params,
        'details': details or {}
    }
    _write_log_entry(entry)
    print(f"[JOB LOG] {job_id} | MESH | {status} | {filename} | Strategy: {strategy}")


def log_job_update(
    job_id: str,
    status: str,
    details: Optional[Dict] = None
) -> None:
    """
    Log a status update for an existing job.
    
    Args:
        job_id: Job identifier to update
        status: New status
        details: Additional details
    """
    entry = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'job_id': job_id,
        'status': status,
        'details': details or {}
    }
    _write_log_entry(entry)
    print(f"[JOB LOG] {job_id} | UPDATE | {status}")


def get_logs_by_job_id(job_id: str) -> List[Dict]:
    """
    Retrieve all log entries for a specific job ID.
    
    Args:
        job_id: Job ID to search for
        
    Returns:
        List of matching log entries
    """
    results = []
    _ensure_log_dir()
    
    for log_file in LOG_DIR.glob('jobs_*.jsonl'):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if job_id in line:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get('job_id') == job_id:
                                results.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            continue
    
    return sorted(results, key=lambda x: x.get('timestamp', ''))


def get_recent_logs(
    limit: int = 100,
    job_type: Optional[str] = None,
    user_email: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict]:
    """
    Get recent log entries with optional filtering.
    
    Args:
        limit: Maximum number of entries to return
        job_type: Filter by job type ('import' or 'mesh')
        user_email: Filter by user email
        status: Filter by status
        
    Returns:
        List of matching log entries (most recent first)
    """
    results = []
    _ensure_log_dir()
    
    # Get log files sorted by date (newest first)
    log_files = sorted(LOG_DIR.glob('jobs_*.jsonl'), reverse=True)
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process lines in reverse (newest first)
            for line in reversed(lines):
                if len(results) >= limit:
                    break
                try:
                    entry = json.loads(line.strip())
                    
                    # Apply filters
                    if job_type and entry.get('job_type') != job_type:
                        continue
                    if user_email and entry.get('user_email') != user_email:
                        continue
                    if status and entry.get('status') != status:
                        continue
                    
                    results.append(entry)
                except json.JSONDecodeError:
                    continue
        except Exception:
            continue
        
        if len(results) >= limit:
            break
    
    return results


def get_logs_by_date_range(
    start_date: datetime,
    end_date: datetime,
    job_type: Optional[str] = None
) -> List[Dict]:
    """
    Get log entries within a date range.
    
    Args:
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        job_type: Optional filter by job type
        
    Returns:
        List of matching log entries
    """
    results = []
    _ensure_log_dir()
    
    for log_file in LOG_DIR.glob('jobs_*.jsonl'):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        timestamp = datetime.fromisoformat(
                            entry.get('timestamp', '').replace('Z', '+00:00')
                        )
                        
                        if start_date <= timestamp.replace(tzinfo=None) <= end_date:
                            if job_type is None or entry.get('job_type') == job_type:
                                results.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception:
            continue
    
    return sorted(results, key=lambda x: x.get('timestamp', ''))
