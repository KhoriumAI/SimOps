#!/usr/bin/env python3
"""
Diagnostic script for MeshGen authentication issues
Run on EC2 server: python3 diagnose_auth.py
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask
from models import db, User
from config import get_config

# Create app
app = Flask(__name__)
app.config.from_object(get_config())
db.init_app(app)

print("=" * 60)
print("MESHGEN AUTHENTICATION DIAGNOSTICS")
print("=" * 60)

with app.app_context():
    # 1. Check database location
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"\n1. Database URI: {db_uri}")
    
    # 2. Check if database file exists
    if db_uri.startswith('sqlite:///'):
        db_path = db_uri.replace('sqlite:///', '')
        print(f"   Database path: {db_path}")
        print(f"   File exists: {Path(db_path).exists()}")
        if Path(db_path).exists():
            print(f"   File size: {Path(db_path).stat().st_size} bytes")
    
    # 3. Check users table
    print(f"\n2. Users in database:")
    try:
        users = User.query.all()
        print(f"   Total users: {len(users)}")
        for u in users:
            print(f"   - {u.email} (ID: {u.id}, active: {u.is_active})")
    except Exception as e:
        print(f"   ERROR querying users: {e}")
    
    # 4. Check for mark@khorium.ai specifically
    print(f"\n3. Checking mark@khorium.ai:")
    try:
        mark = User.query.filter_by(email='mark@khorium.ai').first()
        if mark:
            print(f"   ✓ Found account")
            print(f"   - ID: {mark.id}")
            print(f"   - Active: {mark.is_active}")
            print(f"   - Password hash starts: {mark.password_hash[:30]}...")
            print(f"   - Created: {mark.created_at}")
        else:
            print(f"   ✗ Account NOT FOUND")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # 5. Check blueprint registration
    print(f"\n4. Registered routes:")
    from api_server import create_app
    test_app = create_app()
    
    auth_routes = [r for r in test_app.url_map.iter_rules() if 'auth' in r.rule]
    print(f"   Auth routes found: {len(auth_routes)}")
    for route in auth_routes:
        print(f"   - {route.methods} {route.rule}")
    
    # Check for /api/auth/me specifically
    me_route = [r for r in test_app.url_map.iter_rules() if r.rule == '/api/auth/me']
    if me_route:
        print(f"\n   ✓ /api/auth/me route is registered")
    else:
        print(f"\n   ✗ /api/auth/me route NOT FOUND - auth blueprint issue!")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)
