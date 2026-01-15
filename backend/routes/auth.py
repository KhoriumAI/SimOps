"""
Authentication routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, 
    create_refresh_token, 
    jwt_required, 
    get_jwt_identity,
    get_jwt
)
import bcrypt
from datetime import datetime, timezone, timedelta
import secrets
from email_validator import validate_email, EmailNotValidError

from models import db, User, TokenBlocklist

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


@auth_bp.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def check_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    name = data.get('name', '').strip()
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    print(f"[AUTH] Validating email: {email}")
    # Validate email format using email-validator library
    try:
        # Normalize email (lowercase, strip whitespace)
        validated = validate_email(email, check_deliverability=False)
        email = validated.normalized  # Use normalized version
    except EmailNotValidError as e:
        return jsonify({'error': f'Invalid email: {str(e)}'}), 400
    
    print(f"[AUTH] Checking if user exists: {email}")
    if User.query.filter_by(email=email).first():
        print(f"[AUTH] Registration failed: Email {email} already exists")
        return jsonify({'error': 'Email already registered'}), 409
    
    user = User(
        email=email,
        password_hash=hash_password(password),
        name=name if name else None,
        role='user'
    )
    
    try:
        print(f"[AUTH] Committing new user to DB: {email}")
        db.session.add(user)
        db.session.commit()
        print(f"[AUTH] New user registered: {email} (ID: {user.id})")
    except Exception as e:
        db.session.rollback()
        print(f"[AUTH] Registration error for {email}: {e}")
        return jsonify({'error': 'Failed to create user'}), 500
    
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    
    return jsonify({
        'message': 'User registered successfully',
        'user': user.to_dict(),
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    print(f"[AUTH] Login attempt for: {email}")
    user = User.query.filter_by(email=email).first()
    
    if not user:
        print(f"[AUTH] Login failed: User {email} not found")
        return jsonify({'error': 'Invalid email or password'}), 401
        
    if not check_password(password, user.password_hash):
        print(f"[AUTH] Login failed: Incorrect password for {email}")
        return jsonify({'error': 'Invalid email or password'}), 401
    
    if not user.is_active:
        print(f"[AUTH] Login failed: Account deactivated for {email}")
        return jsonify({'error': 'Account is deactivated'}), 401
    
    print(f"[AUTH] Login successful for: {email}")
    access_token = create_access_token(identity=str(user.id))
    refresh_token = create_refresh_token(identity=str(user.id))
    
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'access_token': access_token,
        'refresh_token': refresh_token
    })


@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Request a password reset"""
    data = request.get_json()
    if not data or not data.get('email'):
        return jsonify({'error': 'Email is required'}), 400
    
    email = data.get('email').strip().lower()
    user = User.query.filter_by(email=email).first()
    
    # Security: Don't reveal if user exists or not
    # Always return 200 to prevent user enumeration
    message = 'If an account exists with this email, a reset link has been sent.'
    
    if not user:
        print(f"[AUTH] Password reset requested for non-existent email: {email}")
        return jsonify({'message': message}), 200
        
    # Generate reset token (valid for 1 hour)
    token = secrets.token_urlsafe(32)
    user.reset_token = token
    user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
    
    try:
        db.session.commit()
        print(f"[AUTH] Generated reset token for {email}: {token[:10]}...")
        
        # Send email
        send_reset_email(user.email, token)
        
    except Exception as e:
        db.session.rollback()
        print(f"[AUTH] Error during password reset request: {e}")
        return jsonify({'error': 'Internal server error'}), 500
        
    return jsonify({'message': message}), 200


@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password using token"""
    data = request.get_json()
    if not data or not data.get('token') or not data.get('password'):
        return jsonify({'error': 'Token and password are required'}), 400
        
    token = data.get('token')
    password = data.get('password')
    
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
        
    user = User.query.filter_by(reset_token=token).first()
    
    if not user:
        return jsonify({'error': 'Invalid or expired token'}), 400
        
    if user.reset_token_expires < datetime.utcnow():
        return jsonify({'error': 'Token has expired'}), 400
        
    # Update password
    # Updated password (using hash_password defined in this file)
    user.password_hash = hash_password(password)
    user.reset_token = None
    user.reset_token_expires = None
    
    try:
        db.session.commit()
        print(f"[AUTH] Password reset successful for: {user.email}")
        return jsonify({'message': 'Password has been reset successfully'}), 200
    except Exception as e:
        db.session.rollback()
        print(f"[AUTH] Error resetting password: {e}")
        return jsonify({'error': 'Failed to reset password'}), 500


def send_reset_email(email, token):
    """
    Send password reset email to user.
    Uses SMTP settings from environment or fallback to print.
    """
    import os
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    frontend_url = os.environ.get('FRONTEND_URL', 'https://app.khorium.ai')
    reset_link = f"{frontend_url}/reset-password?token={token}"
    
    print(f"[EMAIL] Password reset link for {email}: {reset_link}")
    
    # Retrieve SMTP settings from environment
    smtp_server = os.environ.get('MAIL_SERVER')
    smtp_port = int(os.environ.get('MAIL_PORT', 587))
    smtp_user = os.environ.get('MAIL_USERNAME')
    smtp_pass = os.environ.get('MAIL_PASSWORD')
    
    if not smtp_server or not smtp_user or not smtp_pass:
        print("[EMAIL] WARNING: SMTP settings not configured. Email not sent.")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = os.environ.get('MAIL_DEFAULT_SENDER', smtp_user)
        msg['To'] = email
        msg['Subject'] = "Khorium MeshGen - Password Reset Request"
        
        body = f"""
Hello,

You requested a password reset for your Khorium MeshGen account.
Click the link below to set a new password:

{reset_link}

This link will expire in 1 hour.
If you did not request this, please ignore this email.

Best regards,
The Khorium Team
"""
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            
        print(f"[EMAIL] Successfully sent reset email to {email}")
        return True
    except Exception as e:
        print(f"[EMAIL] ERROR sending email to {email}: {e}")
        return False


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    
    if not user or not user.is_active:
        return jsonify({'error': 'User not found'}), 401
    
    access_token = create_access_token(identity=str(current_user_id))
    return jsonify({'access_token': access_token})


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    current_user_id = int(get_jwt_identity())
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'user': user.to_dict()})


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """
    Logout user - blocks both access and refresh tokens.
    
    Request body (optional):
        refresh_token: The refresh token to also revoke
    
    This ensures both tokens are invalidated, preventing
    the refresh token from being used to get a new access token.
    """
    from flask_jwt_extended import decode_token
    
    # Block the access token (from Authorization header)
    jwt_data = get_jwt()
    access_jti = jwt_data['jti']
    user_id = int(get_jwt_identity())
    access_expires = datetime.fromtimestamp(jwt_data['exp'], tz=timezone.utc)
    
    blocked_access = TokenBlocklist(
        jti=access_jti,
        token_type='access',
        user_id=user_id,
        expires_at=access_expires
    )
    
    try:
        db.session.add(blocked_access)
        
        # Also block refresh token if provided in request body
        data = request.get_json(silent=True) or {}
        refresh_token = data.get('refresh_token')
        
        if refresh_token:
            try:
                # Decode the refresh token to get its JTI
                refresh_data = decode_token(refresh_token)
                refresh_jti = refresh_data['jti']
                refresh_expires = datetime.fromtimestamp(refresh_data['exp'], tz=timezone.utc)
                
                # Only block if it belongs to the same user
                if int(refresh_data['sub']) == user_id:
                    blocked_refresh = TokenBlocklist(
                        jti=refresh_jti,
                        token_type='refresh',
                        user_id=user_id,
                        expires_at=refresh_expires
                    )
                    db.session.add(blocked_refresh)
            except Exception as e:
                # Invalid refresh token - ignore, still logout successfully
                pass
        
        db.session.commit()
    except Exception:
        pass
    
    return jsonify({'message': 'Logout successful'})


def check_if_token_revoked(jwt_header, jwt_payload) -> bool:
    """Check if token is revoked"""
    jti = jwt_payload.get('jti')
    # Check token blocklist
    
    try:
        token = TokenBlocklist.query.filter_by(jti=jti).first()
        is_revoked = token is not None
        # Token revocation check complete
        return is_revoked
    except Exception as e:
        print(f"[BLOCKLIST CHECK] Error: {e}")
        return False
