"""
Webhook signature verification utilities for Modal webhooks
"""
import hmac
import hashlib
import os
from typing import Optional


def verify_webhook_signature(payload: bytes, signature: Optional[str], secret: Optional[str] = None) -> bool:
    """
    Verify webhook signature using HMAC-SHA256.
    
    Args:
        payload: Raw request body bytes
        signature: Signature from X-Modal-Signature header
        secret: Webhook secret (defaults to MODAL_WEBHOOK_SECRET env var)
    
    Returns:
        True if signature is valid, False otherwise
    """
    if not signature:
        return False
    
    if not secret:
        secret = os.environ.get('MODAL_WEBHOOK_SECRET')
    
    if not secret:
        # In development, allow unsigned webhooks if secret not set
        # In production, this should be required
        print("[WEBHOOK] Warning: MODAL_WEBHOOK_SECRET not set, skipping signature verification")
        return True
    
    # Modal uses HMAC-SHA256 with format: sha256=<hash>
    if signature.startswith('sha256='):
        signature = signature[7:]
    
    # Compute expected signature
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    # Constant-time comparison to prevent timing attacks
    return hmac.compare_digest(expected_signature, signature)


def extract_signature_from_header(header_value: Optional[str]) -> Optional[str]:
    """
    Extract signature from header value.
    Handles formats like: sha256=abc123 or just abc123
    """
    if not header_value:
        return None
    
    if '=' in header_value:
        return header_value.split('=', 1)[1]
    
    return header_value


