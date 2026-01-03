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
from datetime import datetime, timezone
from email_validator import validate_email, EmailNotValidError

from models import db, User, TokenBlocklist

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')


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
