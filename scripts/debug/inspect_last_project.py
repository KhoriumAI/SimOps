
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))

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
