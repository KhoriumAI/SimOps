
import os
import sys
from pathlib import Path
from sqlalchemy import text, inspect

# Add current directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from api_server import create_app
from models import db

def migrate():
    print("Initializing migration...")
    
    # Create app context to get access to the database configuration
    app = create_app()
    
    with app.app_context():
        engine = db.engine
        print(f"Connected to database: {engine.url}")
        
        inspector = inspect(engine)
        
        # 1. Projects table
        if inspector.has_table('projects'):
            print("Checking 'projects' table...")
            columns = [c['name'] for c in inspector.get_columns('projects')]
            
            project_migrations = [
                ('preview_path', 'VARCHAR(500)'),
                ('original_filename', 'VARCHAR(255)'),
                ('file_size', 'BIGINT DEFAULT 0'),
                ('file_hash', 'VARCHAR(64)'),
                ('mime_type', 'VARCHAR(100)'),
                ('mesh_count', 'INTEGER DEFAULT 0'),
                ('download_count', 'INTEGER DEFAULT 0'),
                ('last_accessed', 'TIMESTAMP')
            ]
            
            for col_name, col_type in project_migrations:
                if col_name not in columns:
                    print(f"Adding {col_name} to projects...")
                    try:
                        with engine.connect() as conn:
                            # Handle database specific syntax
                            if 'sqlite' in str(engine.url):
                                # SQLite doesn't support generic ALTER syntax fully or behaves differently, 
                                # but ADD COLUMN is standard. TIMESTAMP is not standard in SQLite but works as affinity.
                                # Replacing TIMESTAMP with DATETIME for SQLite compatibility if needed, 
                                # generally SQLAlchemy handles this but raw SQL does not.
                                if col_type == 'TIMESTAMP': col_type = 'DATETIME'
                            
                            conn.execute(text(f"ALTER TABLE projects ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
                        print(f"Successfully added {col_name}")
                    except Exception as e:
                        print(f"Error adding {col_name}: {e}")

        # 2. Mesh results table
        if inspector.has_table('mesh_results'):
            print("Checking 'mesh_results' table...")
            columns = [c['name'] for c in inspector.get_columns('mesh_results')]
            
            mesh_results_migrations = [
                ('score', 'FLOAT'),
                ('output_size', 'BIGINT DEFAULT 0'),
                ('quality_metrics', 'JSON'),
                ('logs', 'JSON'),
                ('boundary_zones', 'JSON'),
                ('params', 'JSON'),
                ('job_id', 'VARCHAR(50)'),
                ('processing_time', 'FLOAT'),
                ('node_count', 'INTEGER'),
                ('element_count', 'INTEGER'),
                ('completed_at', 'TIMESTAMP')
            ]
            
            for col_name, col_type in mesh_results_migrations:
                if col_name not in columns:
                    print(f"Adding {col_name} to mesh_results...")
                    try:
                        with engine.connect() as conn:
                            if 'sqlite' in str(engine.url):
                                if col_type == 'TIMESTAMP': col_type = 'DATETIME'
                                if col_type == 'JSON': col_type = 'TEXT' # SQLite doesn't have native JSON
                            
                            conn.execute(text(f"ALTER TABLE mesh_results ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
                        print(f"Successfully added {col_name}")
                    except Exception as e:
                        print(f"Error adding {col_name}: {e}")

        # 3. Users table
        if inspector.has_table('users'):
             print("Checking 'users' table...")
             columns = [c['name'] for c in inspector.get_columns('users')]
             
             user_migrations = [
                ('storage_quota', 'BIGINT DEFAULT 1073741824'),
                ('storage_used', 'BIGINT DEFAULT 0'),
                ('last_login', 'TIMESTAMP'),
                ('name', 'VARCHAR(100)'),
                ('role', 'VARCHAR(20) DEFAULT \'user\''),
                ('is_active', 'BOOLEAN DEFAULT TRUE')
            ]
             
             for col_name, col_type in user_migrations:
                if col_name not in columns:
                    print(f"Adding {col_name} to users...")
                    try:
                        with engine.connect() as conn:
                            if 'sqlite' in str(engine.url):
                                if col_type == 'TIMESTAMP': col_type = 'DATETIME'
                                if 'TRUE' in col_type: col_type = col_type.replace('TRUE', '1')
                            
                            conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))
                            conn.commit()
                        print(f"Successfully added {col_name}")
                    except Exception as e:
                        print(f"Error adding {col_name}: {e}")


        # 4. Feedback table
        if not inspector.has_table('feedbacks'):
             print("Creating 'feedbacks' table...")
             try:
                 # We can use the model to create the table
                 from models import Feedback
                 Feedback.__table__.create(engine)
                 print("Successfully created 'feedbacks' table")
             except Exception as e:
                 print(f"Error creating feedbacks table: {e}")

    print("Migration check complete.")

if __name__ == "__main__":
    migrate()
