from backend.api_server import app, db
from backend.models import User
import os

print(f"Seeding database...")
with app.app_context():
    try:
        db.create_all()
        
        # Check for admin user
        admin = User.query.get(1)
        if not admin:
            print("Creating default admin user...")
            admin = User(
                id=1,
                username="admin",
                email="admin@khorium.com",
                password_hash="fake_hash" 
            )
            db.session.add(admin)
            db.session.commit()
            print("[SUCCESS] Created admin user (ID 1).")
        else:
            print("[INFO] Admin user (ID 1) already exists.")
            
    except Exception as e:
        print(f"[ERROR] Seeding failed: {e}")
        import traceback
        traceback.print_exc()
