
import sqlite3
import os
from pathlib import Path
import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

dbs = ["backend/instance/mesh_app.db", "backend/instance/meshgen.db"]
email = 'mark@khorium.ai'
password = 'Password123!'
hashed = hash_password(password)

for db_rel_path in dbs:
    db_path = Path(db_rel_path)
    if not db_path.exists():
        print(f"Skipping {db_path} (not found)")
        continue
        
    print(f"Processing {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if not cursor.fetchone():
        print(f"Table 'users' not found in {db_path}")
        conn.close()
        continue

    cursor.execute("SELECT id FROM users WHERE email=?", (email,))
    user = cursor.fetchone()

    if user:
        print(f"User exists in {db_path}. Updating password...")
        cursor.execute("UPDATE users SET password_hash=?, is_active=1 WHERE email=?", (hashed, email))
    else:
        print(f"Creating user in {db_path}...")
        cursor.execute("INSERT INTO users (email, password_hash, name, role, is_active) VALUES (?, ?, ?, ?, ?)", 
                       (email, hashed, 'Mark', 'admin', 1))

    conn.commit()
    conn.close()

print("Done!")
