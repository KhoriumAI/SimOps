#!/usr/bin/env python3
"""
Verify Email Configuration Script

Checks if the SMTP settings in the environment variables are valid
and attempts to connect to the mail server.
Optionally sends a test email.

Usage:
    python verify_email.py [--send-to <email>]
"""
import os
import sys
import smtplib
import argparse
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Verify Email/SMTP Configuration")
    parser.add_argument("--send-to", help="Send a test email to this address", default=None)
    args = parser.parse_args()

    # Load environment variables from backend/.env
    # Assuming this script is in scripts/
    root_dir = Path(__file__).parent.parent
    backend_env = root_dir / "backend" / ".env"
    
    if backend_env.exists():
        print(f"[*] Loading environment from {backend_env}")
        load_dotenv(backend_env)
    else:
        print("[!] Warning: backend/.env not found, using current environment variables")

    # Check required variables
    required_vars = ['MAIL_SERVER', 'MAIL_PORT', 'MAIL_USERNAME', 'MAIL_PASSWORD']
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        print(f"[FAIL] Missing environment variables: {', '.join(missing)}")
        return 1

    mail_server = os.environ.get('MAIL_SERVER')
    mail_port = int(os.environ.get('MAIL_PORT', 587))
    mail_username = os.environ.get('MAIL_USERNAME')
    mail_password = os.environ.get('MAIL_PASSWORD')
    frontend_url = os.environ.get('FRONTEND_URL')

    print(f"[*] Checking SMTP Configuration:")
    print(f"    Server: {mail_server}")
    print(f"    Port: {mail_port}")
    print(f"    User: {mail_username}")
    print(f"    Frontend URL: {frontend_url or 'NOT SET (Password resets will allow broken links)'}")

    if not frontend_url:
        print("[WARN] FRONTEND_URL is not set. Password reset links will be broken.")

    # Test Connection
    try:
        print(f"[*] Connecting to {mail_server}:{mail_port}...")
        server = smtplib.SMTP(mail_server, mail_port, timeout=10)
        server.starttls()
        print("[*] TLS started. Logging in...")
        server.login(mail_username, mail_password)
        print("[PASS] SMTP Connection and Login successful!")
        
        if args.send_to:
            print(f"[*] Sending test email to {args.send_to}...")
            msg = MIMEMultipart()
            msg['From'] = os.environ.get('MAIL_DEFAULT_SENDER', mail_username)
            msg['To'] = args.send_to
            msg['Subject'] = "Khorium DevOps: Email Verification Test"
            
            body = "This is a test email from the verify_email.py script.\nIf you received this, your SMTP configuration is working correctly."
            msg.attach(MIMEText(body, 'plain'))
            
            server.send_message(msg)
            print(f"[PASS] Test email sent to {args.send_to}")
            
        server.quit()
        return 0
        
    except Exception as e:
        print(f"[FAIL] SMTP Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
