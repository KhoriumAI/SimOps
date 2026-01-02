
import sqlite3
import os
from pathlib import Path

def migrate():
    db_path = Path('backend/instance/mesh_app.db')
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    print(f"Migrating database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Projects table
    cursor.execute("PRAGMA table_info(projects)")
    columns = [c[1] for c in cursor.fetchall()]
    
    project_migrations = [
        ('preview_path', 'VARCHAR(500)'),
        ('original_filename', 'VARCHAR(255)'),
        ('file_size', 'BIGINT DEFAULT 0'),
        ('file_hash', 'VARCHAR(64)'),
        ('mime_type', 'VARCHAR(100)'),
        ('mesh_count', 'INTEGER DEFAULT 0'),
        ('download_count', 'INTEGER DEFAULT 0'),
        ('last_accessed', 'DATETIME')
    ]

    for col_name, col_type in project_migrations:
        if col_name not in columns:
            print(f"Adding {col_name} to projects...")
            try:
                cursor.execute(f"ALTER TABLE projects ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")

    # 2. Mesh results table
    cursor.execute("PRAGMA table_info(mesh_results)")
    columns = [c[1] for c in cursor.fetchall()]
    
    mesh_results_migrations = [
        ('boundary_zones', 'JSON'),
        ('params', 'JSON'),
        ('job_id', 'VARCHAR(50)')
    ]

    for col_name, col_type in mesh_results_migrations:
        if col_name not in columns:
            print(f"Adding {col_name} to mesh_results...")
            try:
                cursor.execute(f"ALTER TABLE mesh_results ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")

    # 3. Users table
    cursor.execute("PRAGMA table_info(users)")
    columns = [c[1] for c in cursor.fetchall()]
    
    user_migrations = [
        ('storage_quota', 'BIGINT DEFAULT 1073741824'),
        ('storage_used', 'BIGINT DEFAULT 0'),
        ('last_login', 'DATETIME')
    ]

    for col_name, col_type in user_migrations:
        if col_name not in columns:
            print(f"Adding {col_name} to users...")
            try:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
