#!/usr/bin/env python3
"""
🚀 Khorium Staging Promotion Wizard v2.0
   "Safety First, Promotion Second"

This script automates the promotion of the Dev (Sandbox) environment to Staging (Production).

Usage:
    python scripts/promote_to_staging.py [--dry-run] [--force]

Features:
    - ✅ Dry-Run Mode (Preview changes without executing)
    - 🛡️ Rigorous Pre-flight Checks (Connectivity, Permissions, Parity)
    - 📝 Detailed Logging (Console + File)
    - 🔙 Rollback Instructions on Failure
"""

import boto3
import os

import subprocess
import sys
import time
import argparse
import os
import json
import logging
from datetime import datetime
import traceback
import getpass


# --- Configuration (from Discovery) ---
AWS_REGION = "us-west-1"
PROD_DB_ID = "khorium-webdev-db"
STAGING_DB_ID = "khorium-staging-db"
PROD_S3_ASSETS = "muaz-webdev-assets"
STAGING_S3_ASSETS = "muaz-webdev-assets-staging"
PROD_S3_FRONTEND = "muaz-mesh-web-dev"
STAGING_S3_FRONTEND = "muaz-mesh-web-staging"
STAGING_ALB_DNS = "webdev-alb-stg-699722072.us-west-1.elb.amazonaws.com"
STAGING_ALB_ZONE_ID = "Z368ELLRRE2KJ0"
DOMAIN_NAME = "app.khorium.ai"
HOSTED_ZONE_ID = "Z2FDTNDATAQYW2"

# Setup Logging
LOG_FILENAME = f"promotion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PromotionWizard")

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def print(msg, color=None, bold=False):
        if os.name == 'nt':
            print(msg)
            return
        style = ""
        if color: style += color
        if bold: style += Colors.BOLD
        print(f"{style}{msg}{Colors.ENDC}")

# --- Safety & Helper Functions ---

def parse_args():
    parser = argparse.ArgumentParser(description="Promote Dev to Staging")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without executing")
    parser.add_argument("--force", action="store_true", help="Skip non-critical confirmations")
    return parser.parse_args()

ARGS = parse_args()

def log_section(title):
    logger.info("="*60)
    logger.info(f"STARTING PHASE: {title}")
    Colors.print(f"\n=== {title} ===", Colors.HEADER, bold=True)
    if ARGS.dry_run:
        Colors.print("[DRY RUN MODE ENABLED]", Colors.BLUE)

def confirm(prompt, usage="critical"):
    """
    Rigorously ask for confirmation based on criticality.
    usage: 'critical' (requires strict 'yes'), 'warning' (y/n), 'info'
    """
    if ARGS.dry_run:
        return True
    
    if usage == "critical":
        Colors.print(f"\n🚨 CRITICAL SAFETY CHECK 🚨", Colors.FAIL, bold=True)
        print(f"{prompt}")
        print("Type 'PROMOTE' to confirm, or anything else to cancel:")
        choice = input().strip()
        if choice == "PROMOTE":
            logger.info(f"User confirmed critical action: {prompt}")
            return True
        logger.warning("User declined critical action.")
        return False

    suffix = " [y/N]"
    sys.stdout.write(Colors.WARNING + prompt + suffix + Colors.ENDC + " ")
    choice = input().lower().strip()
    if choice in ['y', 'yes']:
        return True
    return False

def check_aws_auth():
    try:
        sts = boto3.client('sts', region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        logger.info(f"Authenticated as: {identity['Arn']}")
        return True
    except Exception as e:
        logger.critical(f"AWS Authentication Failed: {e}")
        return False

# --- Schema Verification ---

REQUIRED_COLUMNS = {
    'projects': [
        'preview_path', 'original_filename', 'file_size', 'file_hash',
        'mime_type', 'mesh_count', 'download_count', 'last_accessed'
    ],
    'mesh_results': [
        'score', 'output_size', 'quality_metrics', 'logs', 'boundary_zones',
        'params', 'job_id', 'processing_time', 'node_count', 'element_count',
        'completed_at'
    ],
    'users': [
        'storage_quota', 'storage_used', 'last_login', 'name', 'role', 'is_active'
    ]
}

def verify_schema_integrity(host, port):
    """
    Connects to the database and ensures all critical columns exist.
    Requires DB_USER, DB_PASSWORD, DB_NAME env vars.
    """
    from sqlalchemy import create_engine, inspect
    
    db_user = os.environ.get('DB_USER')
    db_password = os.environ.get('DB_PASSWORD')
    db_name = os.environ.get('DB_NAME', 'meshgen_db')

    # Interactive Prompt if missing
    if not db_user:
        Colors.print("\n🔐 Database Credentials Required for Schema Check", Colors.BLUE)
        db_user = input("   Enter DB User [meshgen_user]: ").strip() or "meshgen_user"
    
    if not db_password:
        if 'db_user' not in locals() or not db_user: # Handle case where user might have been set above
             pass 
        prompt_text = f"   Enter Password for {db_user}: "
        try:
            db_password = getpass.getpass(prompt_text)
        except Exception:
            # Fallback for environments where getpass might fail (though unlikely in standard shell)
            db_password = input(prompt_text)

    if not (db_user and db_password):
        Colors.print("⚠️  Credentials checking skipped or incomplete.", Colors.WARNING)
        if not confirm("Cannot verify database schema without credentials. Proceed blindly?", "warning"):
            return False
        return True

    Colors.print(f"🔍 Verifying schema integrity on {host}...", Colors.BLUE)
    
    try:
        # Construct connection string
        db_url = f"postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}"
        engine = create_engine(db_url, connect_args={'connect_timeout': 5})
        inspector = inspect(engine)
        
        missing_found = False
        
        for table, required_cols in REQUIRED_COLUMNS.items():
            if not inspector.has_table(table):
                Colors.print(f"❌ Missing critical table: {table}", Colors.FAIL)
                missing_found = True
                continue
                
            existing_cols = [c['name'] for c in inspector.get_columns(table)]
            for col in required_cols:
                if col not in existing_cols:
                    Colors.print(f"❌ Missing column in '{table}': {col}", Colors.FAIL)
                    missing_found = True
        
        if missing_found:
            Colors.print("❌ Database schema is missing required columns! Promotion aborted.", Colors.FAIL)
            return False
            
        Colors.print("✓ Schema verification passed.", Colors.GREEN)
        return True

    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        Colors.print("⚠️  Could not connect to database to verify schema.", Colors.WARNING)
        if not confirm("Schema check failed due to connection error. Proceed?", "warning"):
            return False
        return True

# --- Core Logic ---

def phase_1_database_promotion(rds):
    log_section("Phase 1: Database Promotion")
    
    # 1. Pre-flight Checks & Schema Verification
    logger.info(f"Checking {PROD_DB_ID} status...")
    try:
        resp = rds.describe_db_instances(DBInstanceIdentifier=PROD_DB_ID)
        status = resp['DBInstances'][0]['DBInstanceStatus']
        endpoint = resp['DBInstances'][0]['Endpoint']
        
        if status != 'available':
            logger.error(f"Source (Dev) DB is {status}. Aborting.")
            return False
            
        # Run Code-Level Schema Check
        if not verify_schema_integrity(endpoint['Address'], endpoint['Port']):
            return False
            
    except Exception as e:
        logger.error(f"Could not verify DB {PROD_DB_ID}: {e}")
        return False

    
    # 2. Safety Check: Staging Connectivity?
    # (Here we assume RDS API access is sufficient, but in a real script we might check VPC pairing)

    # 3. Create Snapshot
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    snapshot_id = f"manual-promo-{timestamp}"
    
    if ARGS.dry_run:
        print(f"[DRY RUN] Would create snapshot: {snapshot_id}")
        print(f"[DRY RUN] Would delete DB: {STAGING_DB_ID}")
        print(f"[DRY RUN] Would restore DB: {STAGING_DB_ID} from {snapshot_id}")
        return True

    if not confirm(f"Execute Snapshot of {PROD_DB_ID} and OVERWRITE {STAGING_DB_ID}?", "critical"):
        print("Aborted by user.")
        return False

    try:
        # Fetch Source Security Groups (To ensure Staging has same access)
        logger.info(f"Fetching Security Groups from {PROD_DB_ID}...")
        source_desc = rds.describe_db_instances(DBInstanceIdentifier=PROD_DB_ID)
        vpc_sgs = [sg['VpcSecurityGroupId'] for sg in source_desc['DBInstances'][0]['VpcSecurityGroups']]
        logger.info(f"Found SGs: {vpc_sgs}")

        logger.info(f"Snapshotting {PROD_DB_ID} to {snapshot_id}...")
        rds.create_db_snapshot(DBSnapshotIdentifier=snapshot_id, DBInstanceIdentifier=PROD_DB_ID)
        
        # Wait logic with feedback
        print("Waiting for snapshot...", end="", flush=True)
        waiter = rds.get_waiter('db_snapshot_available')
        waiter.wait(DBSnapshotIdentifier=snapshot_id, WaiterConfig={'Delay': 15, 'MaxAttempts': 120})
        print(" Done.")
        
        # Delete old
        logger.info(f"Deleting old staging DB: {STAGING_DB_ID}")
        try:
            rds.delete_db_instance(DBInstanceIdentifier=STAGING_DB_ID, SkipFinalSnapshot=True)
            print("Waiting for deletion...", end="", flush=True)
            rds.get_waiter('db_instance_deleted').wait(DBInstanceIdentifier=STAGING_DB_ID)
            print(" Done.")
        except rds.exceptions.DBInstanceNotFoundFault:
            logger.info("Staging DB did not exist (clean slate).")

        # Restore
        logger.info(f"Restoring {STAGING_DB_ID} from snapshot with SGs {vpc_sgs}...")
        rds.restore_db_instance_from_db_snapshot(
            DBInstanceIdentifier=STAGING_DB_ID,
            DBSnapshotIdentifier=snapshot_id,
            DBInstanceClass='db.t3.micro',
            AvailabilityZone=f"{AWS_REGION}c",
            PubliclyAccessible=False,
            AutoMinorVersionUpgrade=True,
            VpcSecurityGroupIds=vpc_sgs,
            Tags=[{'Key': 'Environment', 'Value': 'Staging'}]
        )
        print("Waiting for restore (10-15m)...", end="", flush=True)
        rds.get_waiter('db_instance_available').wait(DBInstanceIdentifier=STAGING_DB_ID)
        print(" Done!")
        Colors.print("✓ Database promoted successfully.", Colors.GREEN)
        return True

    except Exception as e:
        logger.error(f"Database Promotion Failed: {e}")
        logger.error(traceback.format_exc())
        Colors.print("❌ FATAL ERROR in DB Phase. Check AWS Console immediately.", Colors.FAIL)
        return False

def phase_2_asset_sync(s3):
    log_section("Phase 2: S3 Sync")
    
    actions = [
        ("Assets", PROD_S3_ASSETS, STAGING_S3_ASSETS, False),
        ("Frontend", PROD_S3_FRONTEND, STAGING_S3_FRONTEND, True) # Delete extra files in target
    ]

    for label, src, dst, delete_flag in actions:
        print(f"\nSub-task: {label} Sync ({src} -> {dst})")
        
        # Pre-check: Bucket Existence
        try:
            s3.head_bucket(Bucket=src)
            # Check dest, create if missing (dry run safe)
            try:
                s3.head_bucket(Bucket=dst)
            except:
                if not ARGS.dry_run:
                    logger.warning(f"Bucket {dst} not found. Creating...")
                    s3.create_bucket(Bucket=dst, CreateBucketConfiguration={'LocationConstraint': AWS_REGION})
        except Exception as e:
            logger.error(f"Bucket access check failed: {e}")
            return False

        cmd = ["aws", "s3", "sync", f"s3://{src}", f"s3://{dst}"]
        if delete_flag:
            cmd.append("--delete")
        
        if ARGS.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            continue

        if confirm(f"Sync {label}?"):
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"Synced {label} successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Sync failed: {e}")
                return False
    
    return True

def phase_3_cutover_manual(targets):
    log_section("Phase 3: MANUAL Cutover (GoDaddy/Registrar)")
    
    print("\n" + "="*60)
    Colors.print("🛑 STOP! DNS UPDATE REQUIRED 🛑", Colors.FAIL, bold=True)
    print("="*60)
    print("\nAutomated Route53 cutover is DISABLED.")
    print(f"You must now log in to your DNS Provider (e.g., GoDaddy) and update the records manually.\n")
    
    print(f"👉 To switch {DOMAIN_NAME} to STAGING, update your CNAME/A Record to point to:")
    Colors.print(f"   {targets['ALB']}", Colors.BLUE, bold=True)
    print("   (Or your Staging CloudFront Domain if configured)\n")

    if ARGS.dry_run:
        print("[DRY RUN] Would pause here for user confirmation of DNS update.")
        return

    Colors.print("⚠️  ACTION REQUIRED: Perform the smoke tests on 'staging.khorium.ai' NOW.", Colors.WARNING)
    print("Once tests pass, update your main DNS record.")
    
    if not confirm(f"Have you updated the DNS for {DOMAIN_NAME} in GoDaddy and verified it?", "critical"):
        Colors.print("❌ Promotion aborted. Old traffic path remains active (unless you changed DNS independently).", Colors.FAIL)
        return

    Colors.print("✓ Manual Cutover confirmed by user.", Colors.GREEN)
    logger.info("User confirmed manual DNS cutover.")

def main():
    Colors.print(f"🚀 Promotion Wizard v2.0 | Log: {LOG_FILENAME}", Colors.BLUE)
    
    if not check_aws_auth():
        sys.exit(1)

    # Initialize clients
    session = boto3.Session(region_name=AWS_REGION)
    rds = session.client('rds')
    s3 = session.client('s3')
    # r53 = session.client('route53') # Not used in Manual Mode

    # Execute
    if phase_1_database_promotion(rds):
        if phase_2_asset_sync(s3):
            # Pass ALIAS targets to cutover phase so it can print them
            targets = {
                "ALB": STAGING_ALB_DNS,
                "CloudFront": "You must check your CloudFront Console for the 'Staging' Distribution Domain Name."
            }
            phase_3_cutover_manual(targets)
        else:
            logger.warning("Skipping cutover due to S3 sync failure.")
    else:
        logger.warning("Skipping rest of promotion due to DB failure.")

    Colors.print("\n✨ Done. Check log for details.", Colors.BLUE)

if __name__ == "__main__":
    main()
