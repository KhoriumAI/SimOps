import os
import sys
import argparse
from pathlib import Path
import psycopg2  # type: ignore
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT  # type: ignore

# Add current directory and parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_database_url

def ensure_database_exists():
    """
    Check if the target database exists on the PostgreSQL server.
    If not, connect to 'postgres' and create it.
    """
    db_url = get_database_url()
    
    # We only care about PostgreSQL
    if not db_url.startswith('postgresql'):
        print(f"[DB] Skipping database creation check for non-Postgres URL: {db_url.split(':')[0]}...")
        return True

    # Parse URL manually for psycopg2
    # Format: postgresql://user:pass@host:port/dbname
    import urllib.parse as urlparse
    url = urlparse.urlparse(db_url)
    dbname = url.path[1:]
    user = url.username
    password = url.password
    host = url.hostname
    port = url.port or 5432

    print(f"[DB] Checking if database '{dbname}' exists on {host}...")

    try:
        # 1. Try connecting to the target database directly
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            connect_timeout=5
        )
        conn.close()
        print(f"[DB] Database '{dbname}' already exists.")
        return True
    except psycopg2.OperationalError as e:
        if 'does not exist' in str(e):
            print(f"[DB] Database '{dbname}' does not exist. Attempting to create it...")
            return create_database(host, port, user, password, dbname)
        else:
            print(f"[DB] Connection error: {e}")
            return False

def create_database(host, port, user, password, dbname):
    """Connect to 'postgres' and create the target database"""
    try:
        # Connect to 'postgres' database
        conn = psycopg2.connect(
            dbname='postgres',
            user=user,
            password=password,
            host=host,
            port=port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f'CREATE DATABASE "{dbname}"')
        print(f"[DB] Database '{dbname}' created successfully.")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Failed to create database: {e}")
        return False

def init_tables():
    """Run db.create_all() to ensure tables exist"""
    print("[DB] Initializing tables...")
    from api_server import create_app
    from models import db
    
    app = create_app()
    with app.app_context():
        try:
            db.create_all()
            print("[DB] Tables initialized (db.create_all).")
            return True
        except Exception as e:
            print(f"[DB] Error initializing tables: {e}")
            return False

def run_migrations():
    """Run alembic migrations"""
    print("[DB] Running migrations (alembic upgrade head)...")
    try:
        import subprocess
        # Get backend directory
        backend_dir = Path(__file__).parent
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=str(backend_dir),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[DB] Migrations applied successfully.")
            return True
        else:
            print(f"[DB] Migrations failed:\n{result.stderr}")
            # If migrations fail, we might still be okay if tables were created by create_all
            return False
    except Exception as e:
        print(f"[DB] Failed to run migrations: {e}")
        return False

if __name__ == "__main__":
    if ensure_database_exists():
        if init_tables():
            # Apply migrations if possible
            run_migrations()
        else:
            sys.exit(1)
    else:
        sys.exit(1)
