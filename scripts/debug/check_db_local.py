
import sqlite3
import os
import sys

# Add root and backend to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'backend'))

# Validating path... found at backend/instance/mesh_app.db
DB_PATH = str(ROOT_DIR / 'backend' / 'instance' / 'mesh_app.db')


def fix_db():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(mesh_results)")
    columns = [info[1] for info in cursor.fetchall()]
    print(f"Current columns in mesh_results: {columns}")

    # Columns to add
    new_columns = [
        ('modal_job_id', 'VARCHAR(100)'),
        ('modal_status', 'VARCHAR(20)'),
        ('modal_started_at', 'DATETIME'),
        ('modal_completed_at', 'DATETIME')
    ]

    for col_name, col_type in new_columns:
        if col_name not in columns:
            print(f"Adding missing column: {col_name}")
            try:
                cursor.execute(f"ALTER TABLE mesh_results ADD COLUMN {col_name} {col_type}")
                print("  Success.")
            except Exception as e:
                print(f"  Failed: {e}")
        else:
            print(f"Column {col_name} already exists.")

    conn.commit()
    conn.close()
    print("Database schema update complete.")

if __name__ == '__main__':
    fix_db()

