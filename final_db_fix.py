import sys
import os
# Ensure the backend directory is in the path
sys.path.append('/home/ec2-user/backend')

try:
    from backend.config import get_database_url
    db_uri = get_database_url()
    print(f"Found DB URI: {db_uri.split('@')[-1]}") # host only for safety
except Exception as e:
    print(f"Failed to import config: {e}")
    sys.exit(1)

from sqlalchemy import create_engine, text
try:
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE projects ADD COLUMN preview_path VARCHAR;"))
        conn.commit()
        print("SUCCESS: 'preview_path' column added to 'projects' table.")
except Exception as e:
    print(f"Database error: {e}")
