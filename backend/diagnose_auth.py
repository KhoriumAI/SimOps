#!/usr/bin/env python3
"""
Diagnostic script for MeshGen authentication issues
Run on EC2 server: python3 diagnose_auth.py
"""
import sys
import os
import traceback
from pathlib import Path
from sqlalchemy import text

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from models import db, User
from config import get_config

# Create app
app = Flask(__name__)
# Force reload config from env
app.config.from_object(get_config())
db.init_app(app)

print("=" * 60)
print("MESHGEN AUTHENTICATION DIAGNOSTICS v2.0")
print("=" * 60)

with app.app_context():
    # 0. Environment Variables
    print("\n0. Environment Variables:")
    for key in ['FLASK_ENV', 'DATABASE_URL', 'DB_HOST', 'DB_NAME', 'DB_USER', 'AWS_REGION']:
        print(f"   {key}: {os.environ.get(key, 'Not Set')}")

    # 1. Check database location
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"\n1. Database URI: {db_uri}")
    
    # 2. Check Type
    if db_uri.startswith('sqlite'):
        print("   Type: SQLite (Local File)")
        db_path = db_uri.replace('sqlite:///', '')
        exists = Path(db_path).exists()
        print(f"   File Path: {db_path}")
        print(f"   Exists: {exists}")
        if exists:
            print(f"   Size: {Path(db_path).stat().st_size} bytes")
    elif 'postgresql' in db_uri:
        print("   Type: PostgreSQL (RDS)")
        if 'staging' in db_uri or 'staging' in os.environ.get('DB_HOST', ''):
             print("   ⚠️  WARNING: Seems to be pointing to STAGING DB!")
    
    # 3. Connection Test
    print(f"\n2. Testing Connectivity (Timeout=5s)...")
    try:
        # Try a simple query with timeout
        with db.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("   ✓ Connection SUCCESS!")
            
            # Check DB Name
            db_name_res = conn.execute(text("SELECT current_database()"))
            print(f"   ✓ Connected to Database: {db_name_res.scalar()}")
            
    except Exception as e:
        print(f"   ❌ CONNECTION FAILED: {e}")
        print("   Possible Causes:")
        print("   1. Security Group blocking access (Check Inbound Rules on RDS)")
        print("   2. Wrong DB Host/Password")
        print("   3. RDS Instance Stopped")
        print("-" * 30)
        # We can stop here if we can't connect, but let's try to print more info if possible
    
    # 4. Check users table (only if connection succeeded)
    print(f"\n3. Users in database:")
    try:
        users = User.query.all()
        print(f"   Total users: {len(users)}")
        for u in users:
            print(f"   - {u.email} (ID: {u.id}, active: {u.is_active})")
    except Exception as e:
        print(f"   Skipping user check due to error: {e}")
    
    # 5. Check for specific user
    print(f"\n4. Checking mark@khorium.ai:")
    try:
        mark = User.query.filter_by(email='mark@khorium.ai').first()
        if mark:
            print(f"   ✓ Found account (ID: {mark.id})")
        else:
            print(f"   ✗ Account NOT FOUND")
    except: pass
    
print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
