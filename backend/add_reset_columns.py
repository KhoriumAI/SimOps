
import sys
import os
from pathlib import Path
from sqlalchemy import text

# Add root directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api_server import create_app
from backend.models import db

def migrate():
    app = create_app()
    with app.app_context():
        print(f"Adding reset columns to database: {app.config['SQLALCHEMY_DATABASE_URI']}")
        
        try:
            # Add reset_token
            try:
                db.session.execute(text("ALTER TABLE users ADD COLUMN reset_token VARCHAR(100)"))
                db.session.commit()
                print("Added reset_token column.")
            except Exception as e:
                db.session.rollback()
                print(f"reset_token column skip: {e}")
            
            # Add reset_token_expires
            try:
                db.session.execute(text("ALTER TABLE users ADD COLUMN reset_token_expires DATETIME"))
                db.session.commit()
                print("Added reset_token_expires column.")
            except Exception as e:
                db.session.rollback()
                print(f"reset_token_expires column skip: {e}")
            
            # Add index
            try:
                # CREATE INDEX IF NOT EXISTS works in SQLite 3.3.0+ and Postgres
                db.session.execute(text("CREATE INDEX IF NOT EXISTS ix_users_reset_token ON users (reset_token)"))
                db.session.commit()
                print("Created index.")
            except Exception as e:
                db.session.rollback()
                print(f"Index skip: {e}")
                
            print("Migration successful!")
                
        except Exception as e:
            print(f"Migration error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    migrate()
