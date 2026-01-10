
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

REQUIRED_KEYS = [
    'FLASK_ENV',
    'SECRET_KEY',
    'JWT_SECRET_KEY',
    'DATABASE_URL',
    'USE_S3',
    'AWS_REGION',
    'MAIL_SERVER',
    'MAIL_PORT',
    'MAIL_USERNAME',
    'MAIL_PASSWORD',
    'MAIL_DEFAULT_SENDER',
    'FRONTEND_URL'
]

OPTIONAL_KEYS = [
    'S3_BUCKET_NAME',
    'CORS_ORIGINS',
    'SLACK_WEBHOOK_URL'
]

def validate_env():
    """
    Validate that all required environment variables are set.
    """
    missing_keys = []
    placeholder_keys = []
    
    print("Checking environment variables...")
    
    for key in REQUIRED_KEYS:
        value = os.environ.get(key)
        if not value:
            missing_keys.append(key)
        elif 'replace_with' in value or 'placeholder' in value:
            placeholder_keys.append(key)
            
    if missing_keys:
        print(f"\n[ERROR] Missing required environment variables:")
        for key in missing_keys:
            print(f"  - {key}")
            
    if placeholder_keys:
        print(f"\n[WARNING] The following variables still have placeholder values:")
        for key in placeholder_keys:
            print(f"  - {key} = {os.environ.get(key)}")
            
    if not missing_keys and not placeholder_keys:
        print("\n[SUCCESS] All required environment variables are set and appear valid.")
        return True
    
    if missing_keys:
        return False
        
    # Valid layout, even if some are placeholders (dev mode)
    return True

if __name__ == "__main__":
    if not validate_env():
        sys.exit(1)
    sys.exit(0)
