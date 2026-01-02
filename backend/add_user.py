
from backend.api_server import app
from backend.models import db, User
from werkzeug.security import generate_password_hash

def add_user():
    with app.app_context():
        email = "mark@khorium.ai"
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            print(f"User {email} already exists.")
            return

        user = User(
            email=email,
            password_hash=generate_password_hash("password123"), # Default password, user can change later
            name="Mark",
            role="admin"
        )
        db.session.add(user)
        db.session.commit()
        print(f"User {email} created successfully.")

if __name__ == "__main__":
    add_user()
