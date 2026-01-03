
import sys
import os
from pathlib import Path

# Setup paths - assume we are running from backend/
# Add parent dir to path so we can import config if needed (though api_server does it too)
sys.path.append(str(Path(__file__).parent.parent))

from api_server import create_app
from models import db, Project

app = create_app()

with app.app_context():
    # Get the latest project
    project = Project.query.order_by(Project.created_at.desc()).first()
    if project:
        print(f"Project ID: {project.id}")
        print(f"Filename: {project.filename}")
        print(f"Filepath: {project.filepath}")
        print(f"User ID: {project.user_id}")
        print(f"Status: {project.status}")
    else:
        print("No projects found.")
