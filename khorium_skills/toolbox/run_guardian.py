"""
Guardian CLI
============
Command-line interface for the Geometry Guardian system.
Replaces the legacy repair.py script.

Usage:
    python khorium_skills/toolbox/run_guardian.py path/to/file.step
    python khorium_skills/toolbox/run_guardian.py path/to/folder --batch
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime

# Ensure we can import from khorium_skills
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    from khorium_skills.guardian.guardian import GeometryGuardian
    from khorium_skills.guardian.models import GeometryStatus
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import Guardian package. {e}")
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GuardianCLI")

def process_single_file(guardian, path):
    """Run guardian on a single file."""
    logger.info(f"Processing: {path}")
    result = guardian.sanitize(path)
    
    print("\n" + "=" * 50)
    print(f"GUARDIAN RESULT: {os.path.basename(path)}")
    print("=" * 50)
    print(f"Final Status: {result.status.name}")
    print(f"Output Path:  {result.output_path}")
    print(f"Report Path:  {result.report_path}")
    
    # Print lifecycle summary
    print("\nLifecycle Events:")
    for event in result.lifecycle:
        phase = event.get('phase', 'UNKNOWN')
        success = event.get('success', True)
        print(f"  - {phase}: {'SUCCESS' if success else 'FAILED'}")
        if 'strategy' in event:
            print(f"    Strategy: {event['strategy']}")
        if 'error' in event:
            print(f"    Error: {event['error']}")
            
    return result

def process_batch(guardian, folder_path):
    """Run guardian on all STEP files in a folder."""
    files = list(Path(folder_path).glob("*.step")) + list(Path(folder_path).glob("*.STEP"))
    
    summary = {
        "total": len(files),
        "passed": 0,
        "repaired": 0,
        "failed": 0,
        "details": []
    }
    
    logger.info(f"Starting batch processing of {len(files)} files...")
    
    for f in files:
        res = guardian.sanitize(str(f))
        
        stat = {
            "file": f.name,
            "status": res.status.name,
            "output": res.output_path
        }
        summary['details'].append(stat)
        
        if res.status == GeometryStatus.PRISTINE:
            summary['passed'] += 1
        elif res.status == GeometryStatus.RESTORED:
            summary['repaired'] += 1
        else:
            summary['failed'] += 1
            
    print("\n" + "=" * 50)
    print("BATCH SUMMARY")
    print("=" * 50)
    print(f"Total Files: {summary['total']}")
    print(f"Pristine:    {summary['passed']}")
    print(f"Repaired:    {summary['repaired']}")
    print(f"Failed:      {summary['failed']}")
    print("=" * 50)

    # Save summary json
    summary_path = os.path.join(folder_path, "guardian_batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Batch summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Guardian Geometry Repair Tool")
    parser.add_argument("path", help="Path to STEP file or directory")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode")
    parser.add_argument("--min-feature", type=float, help="Minimum feature size for cleanup (optional)", default=None)
    
    args = parser.parse_args()
    
    config = {}
    if args.min_feature:
        config['min_feature_size'] = args.min_feature
        
    guardian = GeometryGuardian(config)
    
    if args.batch:
        if not os.path.isdir(args.path):
            logger.error("Path must be a directory for --batch mode")
            sys.exit(1)
        process_batch(guardian, args.path)
    else:
        if not os.path.exists(args.path):
            logger.error("File not found")
            sys.exit(1)
        process_single_file(guardian, args.path)

if __name__ == "__main__":
    main()
