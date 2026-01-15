
import sqlite3
import os
from pathlib import Path

db_path = Path("backend/instance/mesh_app.db")
if not db_path.exists():
    print(f"Error: Database not found at {db_path}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash, is_active FROM users")
    users = cursor.fetchall()
    print("Found users:")
    for u in users:
        print(f"ID: {u[0]}, Email: {u[1]}, Active: {u[3]}, Hash starts with: {u[2][:20]}...")
    conn.close()
