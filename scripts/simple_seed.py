
import sqlite3
import os
from pathlib import Path
import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

dbs = ["backend/instance/mesh_app.db", "backend/instance/meshgen.db"]
email = 'mark@khorium.ai'
password = 'password'
hashed = hash_password(password)

for db_rel_path in dbs:
    db_path = Path(db_rel_path)
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET password_hash=?, is_active=1 WHERE email=?", (hashed, email))
        if cursor.rowcount == 0:
            cursor.execute("INSERT INTO users (email, password_hash, name, role, is_active) VALUES (?, ?, ?, ?, ?)", 
                           (email, hashed, 'Mark', 'admin', 1))
        conn.commit()
        conn.close()
        print(f"Updated {db_path}")

print("Done! Use password: 'password'")
