
import sys
import os
from pathlib import Path
from sqlalchemy import text, create_engine

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api_server import create_app

def migrate():
    app = create_app()
    db_uri = app.config['SQLALCHEMY_DATABASE_URI']
    print(f"Adding reset columns to database: {db_uri}")
    
    engine = create_engine(db_uri)
    
    with engine.connect() as conn:
        # Add reset_token
        try:
            conn.execute(text("ALTER TABLE users ADD COLUMN reset_token VARCHAR(100)"))
            print("Added reset_token column.")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                print("reset_token column already exists.")
            else:
                print(f"reset_token column skip: {e}")
        
        # Add reset_token_expires
        try:
            # Use TIMESTAMP for Postgres, will work for SQLite too
            conn.execute(text("ALTER TABLE users ADD COLUMN reset_token_expires TIMESTAMP"))
            print("Added reset_token_expires column.")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate column" in str(e).lower():
                print("reset_token_expires column already exists.")
            else:
                print(f"reset_token_expires column skip: {e}")
        
        # Add index
        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_reset_token ON users (reset_token)"))
            print("Created index.")
        except Exception as e:
            print(f"Index skip: {e}")
            
        conn.commit()
    print("Migration successful!")

if __name__ == "__main__":
    try:
        migrate()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
