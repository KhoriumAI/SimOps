
import os
import sys
from pathlib import Path

# Add the current directory to sys.path so we can import 'backend'
sys.path.append(os.getcwd())

from backend.api_server import create_app
from backend.models import db, User

def list_users():
    app = create_app()
    with app.app_context():
        print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        users = User.query.all()
        print(f"Total Users: {len(users)}")
        for i, user in enumerate(users):
            print(f"{i+1}. {user.email} (ID: {user.id}, Hash length: {len(user.password_hash) if user.password_hash else 0})")

if __name__ == "__main__":
    list_users()
