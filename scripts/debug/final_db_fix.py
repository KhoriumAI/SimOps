import sys
import os
from pathlib import Path
# Ensure the backend directory is in the path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'backend'))

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
